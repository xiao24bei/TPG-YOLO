# Manuscript to code correspondence

This reference implementation covers the final detection-side configuration of **TPG-YOLO: Low-light object detection based on texture latent priors and deep residual fusion**. Equation and section numbers below refer to the supplied JEI manuscript. Upstream restoration and full benchmark reproduction are separate from the included core modules.

## Final model

| Manuscript specification | Implementation | Check |
| --- | --- | --- |
| Section 3.2: one 256-dimensional texture latent vector per enhanced image | `priors.py: load_texture_prior`; input `z_t` is `[B,256]` | ID, float32 storage, shape and finiteness |
| Eq. (5): affine-free layer normalization | `GlobalPriorAdapter.norm` with `elementwise_affine=False` | Independent normalization formula |
| Eq. (6): reshape to `[B,1,16,16]` | `GlobalPriorAdapter.forward` | Forward calculation |
| Eq. (7): bilinear resizing and 1×1 channel adaptation | `F.interpolate` followed by `projection` | Rectangular and square maps |
| Eq. (8): concatenate feature and adapted texture channels | `JointResidualEncoder.forward` | Residual formula and texture dependence |
| Eq. (9): 1×1 convolution, one-group GN, SiLU, 3×3 depthwise convolution, one-group GN, SiLU | `JointResidualEncoder.layers`; both encoder convolutions have `bias=False` | Layer configuration and gradients |
| Eq. (10): `P5 + gamma * residual` | `TextureP5Fusion.forward` | Positive, zero and negative gamma |
| Section 3.5: gamma initially 0.1, sign unconstrained | Scalar `nn.Parameter`, no sigmoid or clamp | Formula check |
| Sections 3.1 and 3.5: P3/P4 identity, fusion immediately before Detect | `TPGYOLO._before_detect` | Actual Detect input hooks; P3/P4 object identity |
| Section 3.6: final texture-only P5 configuration, gate disabled | Exactly one `TextureP5Fusion`; no selector or gate parameters | 531,969 fusion parameters |
| Section 3.7: frozen upstream outputs; optimize detector and fusion | `TPGYOLO.loss` detaches image/prior inputs and uses YOLO detection criterion | No input gradients, finite detector/fusion gradients and parameter update |
| Eq. (11), Section 4.2: box / class / DFL losses, weights 7.5 / 0.5 / 1.5 | Pinned Ultralytics criterion and explicitly assigned weights | Real synthetic detection-loss step |

The core fusion remains mathematically differentiable with respect to its inputs, which allows isolated gradient checks. The detection training adapter enforces the paper's frozen-upstream boundary by detaching the enhanced image and prior before the detector forward pass.

## Parameter accounting

For P5 channel count `C=512`:

| Component | Parameters |
| --- | ---: |
| Affine-free LayerNorm | 0 |
| 1×1 prior projection, including bias | 1,024 |
| 1×1 joint convolution, no bias | 524,288 |
| First GroupNorm affine parameters | 1,024 |
| 3×3 depthwise convolution, no bias | 4,608 |
| Second GroupNorm affine parameters | 1,024 |
| Residual coefficient | 1 |
| **Total** | **531,969** |

## Explicit implementation choices

The manuscript specifies the operator sequence but does not state every library option. This implementation uses LayerNorm and GroupNorm epsilon `1e-5`, affine GroupNorm, a biased prior projection, and bilinear interpolation with `align_corners=False`. These choices agree with the local fusion reference used during reconstruction. They are documented implementation choices rather than additional claims about the text of the manuscript.

## Boundaries of this release

- Tex-LDM's gradient conditioning and diffusion generation (Section 3.2, Eqs. (3)–(4)) remain upstream. The input is its generated texture latent prior, not its Sobel map or condition vector. No substitute restoration network is supplied.
- Section 3.6's generalized multi-prior, multi-scale and gated ablation machinery is outside this compact release. Its final selected one-branch configuration is implemented.
- Input image preprocessing, annotation transforms, full data loaders, NMS and final benchmark evaluation remain the integrating runner's responsibility. The adapter returns decoded predictions before NMS.
- The 150-epoch protocol, dataset splits, seed statistics and 20-step restoration protocol are described in the manuscript but were not reconstructed as a complete experiment here. The repository does not establish numerical reproduction of the paper's accuracy or latency tables.

See [the test report](../TEST_REPORT.md) for execution evidence and [the README](../README.md) for runnable examples.
