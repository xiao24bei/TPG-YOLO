# TPG-YOLO

[![Core and YOLO checks](https://github.com/xiao24bei/TPG-YOLO/actions/workflows/tests.yml/badge.svg)](https://github.com/xiao24bei/TPG-YOLO/actions/workflows/tests.yml)

Core implementation of **TPG-YOLO: Low-light object detection based on texture latent priors and deep residual fusion**, reconstructed against `TPG-YOLO_JEI.docx` on 2026-11-17.

This compact release implements the final detection-side method in Sections 3.3–3.5. It is a reconstruction of the described modules, not a recovered snapshot of the original experiment code. The supplied examples use synthetic inputs to check execution; no paper accuracy or latency results are claimed.

**For reviewers:** start with the [paper-to-code correspondence](docs/PAPER_CODE_MAPPING.md), [core implementation](tpg_yolo/fusion.py), and [verification report](TEST_REPORT.md). [中文使用说明](使用说明.md) is also available.

The final update is `P5' = P5 + gamma * G(concat(P5, A(resize(reshape(LN(z_t))))))`, with `gamma=0.1` initially. P3 and P4 are unchanged. Texture gating is disabled.

## Included modules

| File | Function | Manuscript |
| --- | --- | --- |
| `tpg_yolo/fusion.py` — `GlobalPriorAdapter` | Affine-free LayerNorm, 16×16 latent layout, bilinear resize, 1×1 channel projection | Eqs. (5)–(7) |
| `tpg_yolo/fusion.py` — `JointResidualEncoder` | Concatenation, pointwise convolution, GroupNorm/SiLU, depthwise convolution, GroupNorm/SiLU | Eqs. (8)–(9) |
| `tpg_yolo/fusion.py` — `TextureP5Fusion` | Learnable residual coefficient, texture-to-P5 fusion; P3/P4 identity | Eq. (10) |
| `tpg_yolo/yolo.py` | Low-level YOLO11s head-input integration and standard detection loss | Sections 3.1 and 3.7 |
| `tpg_yolo/priors.py` | Offline texture-cache validation and optional fixed shuffled-prior control | Sections 3.2 and 4.4 |
| `demo.py`, `tests/` | Forward/backward, one-batch optimization and state-dict checks | Execution verification |

The final branch has **531,969 parameters** for 512-channel P5. It instantiates one branch, with no mixture selector and no texture gate. The full Tex-LDM network, its weights, dataset preparation, multi-prior/gating ablations and full benchmark training/evaluation runners are outside this release.

## Install and run

Use Python 3.11 in an isolated environment. Run commands from this folder.

```bash
git clone https://github.com/xiao24bei/TPG-YOLO.git
cd TPG-YOLO
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python demo.py
python -m unittest discover -s tests -p 'test_core.py' -v
```

Optional YOLO11s integration:

```bash
python -m pip install -r requirements-yolo.txt
python demo.py --yolo
python -m unittest discover -s tests -v
```

The demo builds YOLO11s from its YAML with random initialization and does not download weights. Its one-batch step is a wiring check. Formal training would require the paper's COCO initialization, real enhanced images, aligned priors and labels.

## Use the core fusion independently

```python
import torch
from tpg_yolo import TextureP5Fusion

fusion = TextureP5Fusion(channels=512)
# P3/P4/P5 must be features immediately before the detection head.
p3 = torch.randn(2, 128, 80, 80)
p4 = torch.randn(2, 256, 40, 40)
p5 = torch.randn(2, 512, 20, 20)
z_t = torch.randn(2, 256)  # Replace with the corresponding Tex-LDM outputs.
p3_new, p4_new, p5_new = fusion([p3, p4, p5], z_t)
```

The module follows the device and dtype of its P5 input for prior adaptation. Move the module and features to the same device before use. Rectangular feature maps are supported. The 16×16 latent layout has no pixel correspondence to the enhanced image.

## Upstream image and prior contract

Tex-LDM must generate the enhanced RGB image and `z_t` **in the same inference pass**. Store one `.npz` per image, with a scalar string `image_id` and a finite `float32[256]` vector `z_t`:

```python
import numpy as np
from tpg_yolo.priors import load_texture_prior

# texture_vector is the actual upstream output, not Sobel or random data.
np.savez("sample_001.npz", image_id="sample_001", z_t=texture_vector.astype(np.float32))
z_t = load_texture_prior("sample_001.npz", "sample_001")
```

For a batch, stack vectors in exactly the same order as the enhanced images and labels. Freeze the upstream model and detach its outputs before detection training. A Sobel map, gradient condition vector or randomly initialized encoder is not a substitute for the generated texture latent prior.

The minimal reader checks the stored ID, shape, type and finiteness. It cannot prove that an image and a prior were generated together; preserve upstream generation provenance separately.

For zero-prior controls, pass `torch.zeros_like(z_t)` through the same fusion branch. This is different from disabling fusion because the joint encoder still sees P5. For shuffled-prior controls, compute `fixed_shuffle_indices(N, seed)` once for each complete split and use the mapping to select prior files; do not shuffle across splits or regenerate the mapping every batch. This derangement rule is a reconstruction choice, not a recovered experimental seed or mapping.

## Low-level YOLO11s calls

```python
import torch
from tpg_yolo.yolo import TPGYOLO

model = TPGYOLO(nc=12)  # ExDark: 12; DARK FACE: 1
model.train()
loss, loss_items = model(batch)
loss.sum().backward()

model.eval()
with torch.no_grad():
    decoded, raw = model(images, z_t=texture_priors)
```

`batch` contains `img`, `z_t`, `batch_idx`, `cls` and `bboxes`; see the executable `demo.py` for exact shapes. Images must already be RGB floating-point `[B,3,H,W]` in [0,1], with dimensions compatible with YOLO's stride of 32. Training boxes are normalized `xywh` coordinates after image preprocessing. Inference output is decoded predictions **before NMS**; letterboxing, NMS and coordinate restoration remain the caller's responsibility. This class is not a replacement for the high-level `YOLO(...).train()` workflow.

The adapter targets Ultralytics 8.3.58 and inserts the fusion immediately before `Detect`, using its actual P5 channel count. Priors are required for each call. Per-call prior state is cleared even on failure. Calls on one model instance must be sequential; concurrent serving, compilation, export, distributed training and test-time augmentation are not supported or validated by this minimal adapter.

For portable saving, use `torch.save(model.state_dict(), path)` and restore to a newly constructed `TPGYOLO(nc=the_same_class_count)` with `load_state_dict(torch.load(path, weights_only=True))`. Historical checkpoints with different module names are not directly compatible with this package.

## Manuscript settings for a future full training runner

These are transcribed settings, not experiments executed for this release:

| Setting | ExDark | DARK FACE |
| --- | --- | --- |
| Classes | 12 | 1 |
| Detector input size | 640 | 1280 |
| Batch size | 16 | 8 |
| Epochs | 150 | 150 |
| Optimizer / initial LR / weight decay | SGD / 0.01 / 0.0005 | Same |
| Box / class / DFL weights | 7.5 / 0.5 / 1.5 | Same |
| Mosaic / MixUp | Disabled | Disabled |
| Residual coefficient initialization | 0.1, unconstrained | Same |
| Final checkpoint | Last epoch | Last epoch |
| Evaluation confidence / NMS IoU / max detections | 0.001 / 0.7 / 300 | Same |

Single-image augmentation must retain image/prior identity. The manuscript describes 20-step Tex-LDM sampling; this release neither implements nor verifies that sampler. Exact upstream checkpoint, splits, augmentation magnitudes, seeds and original final detector weights must be resolved separately for numerical reproduction.

Implementation choices retained from the local recovered fusion code include projection bias, `align_corners=False`, LayerNorm/GroupNorm epsilon 1e-5 and affine GroupNorm. At C=512 the count is `2*C*C + 15*C + 1 = 531969`. See `TEST_REPORT.md` for checks actually run.

## Dependencies and provenance

PyTorch and NumPy are external dependencies. The optional YOLO11s adapter depends on Ultralytics, which retains its own licensing terms and notices. No third-party source tree or pretrained weights are bundled. A repository-wide license is not newly assigned here.

The implementation was checked against the manuscript and an existing local fusion reference. The manuscript is the authority for the final one-branch structure; the source document is not redistributed in this repository.

## Code availability statement

The core implementation of texture-prior adaptation and P5 residual fusion, together with a YOLO11s integration example and verification tests, is available at https://github.com/xiao24bei/TPG-YOLO.
