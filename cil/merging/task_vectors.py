from typing import List, Tuple

import numpy as np
import torch


class TaskVector:
    def __init__(
        self,
        pretrained_checkpoint=None,
        finetuned_checkpoint=None,
        vector=None,
        finetuned_state_dict=None,
        lazy=True,
    ):
        """Initializes the task vector from a pretrained and a finetuned checkpoints.

        This can either be done by passing two state dicts (one corresponding to the
        pretrained model, and another to the finetuned model), or by directly passying in
        the task vector state dict.
        """
        self._lazy = lazy
        self._finetuned_checkpoint = finetuned_checkpoint
        self._finetuned_state_dict = finetuned_state_dict

        if vector is not None:
            self.vector = vector
        else:
            assert pretrained_checkpoint
            self._pretrained_checkpoint = pretrained_checkpoint

        if not self._lazy:
            self.load(save=True)

    def load(self, save=False):
        with torch.no_grad():
            pretrained_state_dict = torch.load(self._pretrained_checkpoint).state_dict()

            finetuned_state_dict = self._finetuned_state_dict

            if finetuned_state_dict:
                print(
                    f"Creating task vector from finetuned_state_dict based on {self._pretrained_checkpoint=}"
                )
            elif self._finetuned_checkpoint:
                print(
                    f"Creating task vector from {self._finetuned_checkpoint=} based on {self._pretrained_checkpoint=}"
                )
                finetuned_state_dict = torch.load(
                    self._finetuned_checkpoint
                ).state_dict()

            vector = {}
            # print(pretrained_state_dict.keys())
            # print(finetuned_state_dict.keys())
            for key in pretrained_state_dict:
                # print(pretrained_state_dict[key].dtype)
                if pretrained_state_dict[key].dtype in [torch.int64, torch.uint8]:
                    print(
                        f"Key {key} has dtype {pretrained_state_dict[key].dtype} -- skipping!"
                    )
                    continue
                vector[key] = finetuned_state_dict[key] - pretrained_state_dict[key]
            if save:
                self.vector = vector

            return vector

    def __add__(self, other):
        """Add two task vectors together."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                if key not in other.vector:
                    print(f"Warning, key {key} is not present in both task vectors.")
                    continue
                new_vector[key] = self.vector[key] + other.vector[key]
        return TaskVector(vector=new_vector)

    def __radd__(self, other):
        if other is None or isinstance(other, int):
            return self
        return self.__add__(other)

    def __neg__(self):
        """Negate a task vector."""
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = -self.vector[key]
        return TaskVector(vector=new_vector)

    def __truediv__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] / other
        return TaskVector(vector=new_vector)

    def __mul__(self, other):
        with torch.no_grad():
            new_vector = {}
            for key in self.vector:
                new_vector[key] = self.vector[key] * other
        return TaskVector(vector=new_vector)

    def apply_to(self, pretrained_checkpoint, scaling_coef=1.0):
        """Apply a task vector to a pretrained model."""
        with torch.no_grad():
            pretrained_model = torch.load(pretrained_checkpoint)
            new_state_dict = {}
            pretrained_state_dict = pretrained_model.state_dict()
            for key in pretrained_state_dict:
                if key not in self.vector:
                    print(
                        f"Warning: key {key} is present in the pretrained state dict but not in the task vector"
                    )
                    continue
                new_state_dict[key] = (
                    pretrained_state_dict[key] + scaling_coef * self.vector[key]
                )
        pretrained_model.load_state_dict(new_state_dict, strict=False)
        return pretrained_model


def merge_rnd_mix(task_vectors):
    """Randomly mix multiple task vectors together."""
    if len(task_vectors) == 0:
        return task_vectors[0]

    with torch.no_grad():
        new_vector = {}
        for key in task_vectors[0].vector:
            _rand_indices = torch.randint(
                0, len(task_vectors), task_vectors[0].vector[key].shape
            )
            new_vector[key] = task_vectors[0].vector[key] * (_rand_indices == 0)
            for i in range(1, len(task_vectors)):
                new_vector[key] += task_vectors[i].vector[key] * (_rand_indices == i)

    return TaskVector(vector=new_vector)


def merge_max_abs(task_vectors: List[TaskVector], percentage=100):
    """Mix multiple task vectors together by highest parameter value."""
    if len(task_vectors) == 0:
        return task_vectors[0]

    higher_rates = []
    with torch.no_grad():
        new_vector = task_vectors[0].load()

        # Iterate over the remaining task vectors
        for task_vector in task_vectors[1:]:
            current_task_vector = task_vector.load()

            higher_rates.append([])
            # Iterate over keys in the first task vector
            for key in new_vector:
                current_tensor = current_task_vector[key]
                # Get the initial tensor for the current key
                max_abs_tensor = new_vector[key]

                # Update max_abs_tensor to keep the element-wise maximum absolute values
                max_abs_tensor = torch.where(
                    current_tensor.abs() >= max_abs_tensor.abs(),
                    current_tensor,
                    max_abs_tensor,
                )

                top_max_abs_tensor, rate = update_tensor_by_higher_values(
                    max_abs_tensor, current_tensor, percentage=percentage
                )
                if "adaptmlp" in key or "router" in key or "noise" in key:
                    higher_rates[-1].append(rate)
                # Assign the final tensor to the new_vector dictionary
                new_vector[key] = top_max_abs_tensor

    return TaskVector(vector=new_vector), higher_rates


def update_tensor_by_higher_values(
    tensor_a, tensor_b, percentage=50
) -> Tuple[torch.Tensor, float]:
    """
    Updates tensor_a with the top percentage of tensor_b values, but only considers
    positions where tensor_b is higher than tensor_a.

    Args:
        tensor_a (torch.Tensor): Tensor to be updated
        tensor_b (torch.Tensor): Tensor used for finding top values
        percentage (float): Percentage of higher values to consider (0 to 100)

    Returns:
        torch.Tensor: Updated version of tensor_a
        float: Percentage of higher values in tensor_b that were used for updating tensor_a
    """
    if not 0 <= percentage <= 100:
        raise ValueError("Percentage must be between 0 and 100")

    higher_mask = tensor_b.abs() > tensor_a.abs()
    higher_values = tensor_b[higher_mask]
    if higher_values.numel() == 0:
        return tensor_a.clone(), 0

    k = max(1, int(higher_values.numel() * (percentage / 100)))
    top_k_values, top_k_indices = torch.topk(higher_values, k)
    threshold = top_k_values[-1]
    final_mask = (tensor_b > tensor_a) & (tensor_b >= threshold)
    updated_tensor = torch.where(final_mask, tensor_b, tensor_a)

    return updated_tensor, higher_values.numel() / tensor_b.numel()


def update_tensor_by_top_k(tensor_a, tensor_b, k=20):
    """
    Updates tensor_a based on the top k maximum absolute values in tensor_b.

    Args:
        tensor_a (torch.Tensor): Tensor to be updated
        tensor_b (torch.Tensor): Tensor used for finding top k values
        k (int): Number of top values to consider (default: 20)

    Returns:
        torch.Tensor: Updated version of tensor_a
    """
    abs_b = torch.abs(tensor_b)
    top_k_values, top_k_indices = torch.topk(abs_b.flatten(), k)
    mask = torch.zeros_like(tensor_b)
    indices = np.unravel_index(top_k_indices.cpu().numpy(), tensor_b.shape)
    mask[indices] = 1

    # Update tensor_a only at the positions where mask is 1
    # Keep original values where mask is 0
    updated_tensor = torch.where(mask == 1, tensor_b, tensor_a)

    return updated_tensor
