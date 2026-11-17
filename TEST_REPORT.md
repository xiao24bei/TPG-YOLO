# Verification report

Date: 2026-11-17. Tests ran on macOS ARM64, CPU, Python 3.11.15, PyTorch 2.8.0, NumPy 1.26.4 and Ultralytics 8.3.58.

## Executed checks

`python -m unittest discover -s tests -v`: **7 tests passed**.

1. Final-branch parameter count is 531,969. Outputs agree with an independently assembled normalization/residual formula; P3/P4 retain object identity; gamma=0 gives exact P5 identity.
2. Outputs depend on the texture input. Gradients reach the texture input and P5; every fusion parameter has finite gradients. Non-square feature maps execute successfully.
3. Wrong prior dimension, mismatched batch, NaN and integer priors are rejected.
4. The NPZ reader checks image identity and float32 storage; seeded shuffled mappings are bijective and have no fixed points for split sizes 2, 3 and 19.
5. A YOLO11s one-batch detection loss/backward/SGD step has finite losses and gradients, updates fusion weights and propagates gradients into the detector. Evaluation, model deepcopy and a strict state-dict round trip preserve predictions. Missing or invalid priors are rejected; the adapter clears per-call state after a failure and remains usable.
6. Layer ordering, affine-free LayerNorm, convolution channel counts/kernels/biases, one-group normalization, depthwise groups and gamma initialization match the manuscript. Both zero and negative gamma follow the unconstrained residual formula.
7. Hooks on the actual YOLO11s Detect module verify that fusion occurs immediately before the head, P3/P4 are passed through, and P5 equals the core module output. The training test also verifies that image/prior inputs receive no gradients and that loss weights are explicitly 7.5 / 0.5 / 1.5.

`python demo.py --yolo`: **passed**.

- Core tensors use P3/P4/P5 sizes corresponding to a 640×640 input: `[2,128,80,80]`, `[2,256,40,40]`, `[2,512,20,20]`.
- The optional full-detector check uses two synthetic 64×64 RGB inputs and 12 classes to keep execution small. It completes one optimization step and produces finite decoded predictions before NMS with shape `[2,16,84]`.

An additional local comparison loaded identical parameters from the previously recovered single texture-to-P5 branch. All output tensors matched exactly (`rtol=0`, `atol=0`) on rectangular inputs. That older recovery package is a local reference, not required to run this release.

## Verification scope

These are computational and integration checks. No 150-epoch training, Tex-LDM sampling, dataset benchmark, mAP reproduction or latency measurement was performed. Random example priors and model initialization are not paper assets. MPS, CUDA, distributed execution, concurrent calls on one model instance and export were not tested.

Tests used an existing local Python environment with the pinned Ultralytics distribution loaded from an isolated package directory. A completely fresh environment installation was not separately tested. The tests in this folder can be rerun after installing the documented requirements.
