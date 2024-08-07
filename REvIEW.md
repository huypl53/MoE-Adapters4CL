# About MoE Adapters

## [def main()](mtil/src/main.py:39)

- Main program
- To finetune adapters, run [finetune](#finetune)

## [finetune()](mtil/src/models/finetune.py:14)

- Params that contains none of these `["adaptmlp","router", "noise" ]` are frozen -> **Most of CLIP are frozen**
- Load fronzen parameters then set them frozen. They are params from previous train tasks
- Only params of to-train task contain string `["adaptmlp","router", "noise" ]` are `trained` and tracked by `optimizer`

## [build_model()](mtil/clip/model.py:776)

- Init sizes of [CLIP](#clip) layers
- Load `state_dict` from one of these [\_MODELS](mtil/clip/clip.py:19). Default is `ViT-B/16`
- Restore `state_dict` excepts keys in `["input_resolution", "context_length", "vocab_size"]` to CLIP

## [CLIP](mtil/clip/model.py#L597)

**Attributes**:

- `self.visual`: [VisualTransformer](#visualtransformer)
- `self.transformer`: part of `Text encoder` which is run at [self.encode_text()](mtil/clip/model.py:715)

## [VisualTransformer](mtil/clip/model.py:554)

- Contains a [Transformer](#transformer)

## [Transformer](mtil/clip/model.py:542)

- Contain N x [ResidualAttentionBlock](#residualattentionblock). `N = vision_layers`

## [ResidualAttentionBlock](mtil/clip/model.py:287)

**Attributes**:

- `self.adaptmlp_list`: list of 22 `Adapter`
- `self.d_model`: width = vision_width = 768
- `self.n_head`: heads = vision_width // 64
- `self.attn_mask`: attn_mask
- `self.adapter_flag`: adapter_flag
- `self.args`:
- `self.text_or_image`:
- `self.i`:

- at `mtil/clip/model.py:496`, x.shape = [?, N, d_model]
