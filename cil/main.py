import json
import logging
import os
import statistics
from pathlib import Path

import hydra
import torch
from continual_clip import utils
from continual_clip.datasets import build_cl_scenarios
from continual_clip.models import load_model
from continuum.metrics import Logger
from merging.task_vectors import TaskVector, merge_max_abs, merge_rnd_mix
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from tqdm import tqdm

adapter_paths = []
eval_dataset, classes_names = None, None


@hydra.main(config_path=None, config_name=None, version_base="1.1")
def main(cfg: DictConfig) -> None:
    continual_clip(cfg)
    search_evaluate_merging(cfg)


def continual_clip(cfg: DictConfig) -> None:
    global adapter_paths, eval_dataset, classes_names
    adapter_paths = []

    cfg.workdir = utils.get_workdir(path=os.getcwd())
    cfg.dataset_root = os.path.join(cfg.workdir, cfg.dataset_root)

    utils.save_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg.class_order = utils.get_class_order(os.path.join(cfg.workdir, cfg.class_order))
    model = load_model(cfg, device)

    eval_dataset, classes_names = build_cl_scenarios(
        cfg, is_train=False, transforms=model.transforms
    )
    print(eval_dataset, eval_dataset)
    # print('eval_classname', classes_names)
    train_dataset, train_classes_names = build_cl_scenarios(
        cfg, is_train=True, transforms=model.transforms
    )
    # print('train_classes_names', train_classes_names)
    model.classes_names = classes_names

    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    with open(cfg.log_path, "w+") as f:
        pass

    acc_list = []
    metric_logger = Logger(list_subsets=["test"])

    # test
    for task_id, _ in enumerate(eval_dataset):
        # breakpoint()
        # if task_id == 2: break
        logging.info(f"Evaluation for task {task_id} has started.")
        # breakpoint()
        model.adaptation(
            task_id, cfg, train_dataset, train_classes_names
        )  # task id 已经传入model

        eval_loader = DataLoader(eval_dataset[: task_id + 1], batch_size=64)
        # breakpoint()
        for inputs, targets, task_ids in tqdm(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task_ids)
            metric_logger.add(
                [outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test"
            )

        acc_list.append(100 * metric_logger.accuracy)
        with open(cfg.log_path, "a+") as f:
            f.write(
                json.dumps(
                    {
                        "task": task_id,
                        "acc": round(100 * metric_logger.accuracy, 2),
                        "avg_acc": round(
                            100 * metric_logger.average_incremental_accuracy, 2
                        ),
                        "forgetting": round(100 * metric_logger.forgetting, 6),
                        "acc_per_task": [
                            round(100 * acc_t, 2)
                            for acc_t in metric_logger.accuracy_per_task
                        ],
                        "bwt": round(100 * metric_logger.backward_transfer, 2),
                        "fwt": round(100 * metric_logger.forward_transfer, 2),
                    }
                )
                + "\n"
            )
            metric_logger.end_task()

        path = log_dir / f"adapter-{task_id:03d}.pth"
        adapter_paths.append(str(path))
        print(f"Save model to {str(path)}")
        utils.torch_save_whole(model, str(path))

        # assert 1 == 2
    with open(cfg.log_path, "a+") as f:
        f.write(
            json.dumps(
                {
                    "last": round(acc_list[-1], 2),
                    "avg": round(statistics.mean(acc_list), 2),
                }
            )
            + "\n"
        )


def eval_single_dataset(model, log_file, ignore_task_ids):

    global eval_dataset, classes_names
    print("-------Eval on MagMax merged model--------")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # eval_dataset, classes_names = build_cl_scenarios(
    #     cfg, is_train=False, transforms=model.transforms
    # )
    # print(eval_dataset, eval_dataset)
    # model.classes_names = classes_names
    acc_list = []
    # log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    metric_logger = Logger(list_subsets=["test"])

    for task_id, _ in enumerate(eval_dataset):
        if task_id in ignore_task_ids:
            continue
        # breakpoint()
        # if task_id == 2: break
        logging.info(f"MagMax Evaluation for task {task_id} has started.")
        # breakpoint()
        # model.adaptation(task_id, cfg, train_dataset, train_classes_names)  # task id 已经传入model

        eval_loader = DataLoader(eval_dataset[: task_id + 1], batch_size=64)
        model.update_class_history(task_id)
        # breakpoint()
        for inputs, targets, task_ids in tqdm(eval_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, task_ids)
            metric_logger.add(
                [outputs.cpu().argmax(dim=1), targets.cpu(), task_ids], subset="test"
            )

        acc_list.append(100 * metric_logger.accuracy)
        with open(log_file, "a+") as f:
            f.write(
                json.dumps(
                    {
                        "task": task_id,
                        "acc": round(100 * metric_logger.accuracy, 2),
                        "avg_acc": round(
                            100 * metric_logger.average_incremental_accuracy, 2
                        ),
                        "forgetting": round(100 * metric_logger.forgetting, 6),
                        "acc_per_task": [
                            round(100 * acc_t, 2)
                            for acc_t in metric_logger.accuracy_per_task
                        ],
                        "bwt": round(100 * metric_logger.backward_transfer, 2),
                        "fwt": round(100 * metric_logger.forward_transfer, 2),
                    }
                )
                + "\n"
            )
            metric_logger.end_task()

    with open(str(log_file), "a+") as f:
        f.write(
            json.dumps(
                {
                    "last": round(acc_list[-1], 2),
                    "avg": round(statistics.mean(acc_list), 2),
                }
            )
            + "\n"
        )


def search_evaluate_merging(cfg: DictConfig) -> None:
    global adapter_paths
    n_splits = len(adapter_paths)

    # TODO: choose better pretrained
    ignore_task_id = len(adapter_paths) - 1
    pretrained_task = cfg.get("pretrained_task", 'last')
    if pretrained_task == 'last':
        pass
    if pretrained_task == 'middle':
        ignore_task_id = len(adapter_paths) // 2
    if pretrained_task == 'first':
        ignore_task_id = 0

    pretrained_checkpoint = adapter_paths[ignore_task_id]
    task_vectors = [
        TaskVector(pretrained_checkpoint, ckpt)
        for i, ckpt in enumerate(adapter_paths)
        if i != ignore_task_id
    ]

    funcs_and_coeffs = [
        # (merge_rnd_mix, np.linspace(0.5, 1.5, num=n_coeffs+1)[1:]),
        # (merge_max_abs, np.linspace(0.0, 1.0, num=n_coeffs+1)[1:]),
        # (sum, np.linspace(0.0, 2.0/n_splits, num=n_coeffs+1)[1:]),
        # (merge_rnd_mix, [1.0]),
        # (merge_max_abs, [0.25, 0.5, 0.75]),
        (merge_max_abs, [0.5]),
        # (sum, [1.0 / (n_splits - 1)]),
    ]

    log_file = "metric-magmax.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f, coeffs in funcs_and_coeffs:
        func_name = f.__name__
        print(f"\nMerging with function: {func_name}")
        with open(log_file, "a+") as f_log:
            f_log.write(f"\nMerging with function: {func_name}" + "\n")
        merged_tv = f(task_vectors)

        # Apply the resulting task vector
        results = {}
        for scaling_coef in coeffs:
            print(f"Scaling coeff: {scaling_coef}")
            with open(log_file, "a+") as f_log:
                f_log.write(f"Scaling coeff: {scaling_coef}" + "\n")
            model = merged_tv.apply_to(pretrained_checkpoint, scaling_coef=scaling_coef)
            model.to(device)
            # Evaluate
            # eval_single_dataset(model, log_file, [ ignore_task_id ])
            eval_single_dataset(model, log_file, [])


if __name__ == "__main__":
    main()
