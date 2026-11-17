"""Synthetic forward/backward examples; no benchmark scores are produced."""
import argparse

import torch

from tpg_yolo import TextureP5Fusion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yolo", action="store_true", help="also run a YOLO11s training step")
    args = parser.parse_args()
    torch.manual_seed(17)
    torch.set_num_threads(2)
    # Feature sizes correspond to 640x640 detector inputs.
    features = [torch.randn(2, c, h, h) for c, h in [(128, 80), (256, 40), (512, 20)]]
    z_t = torch.randn(2, 256)  # Synthetic only; replace with frozen Tex-LDM outputs.
    fusion = TextureP5Fusion()
    fused = fusion(features, z_t)
    fused[-1].square().mean().backward()
    print("Fusion parameters:", sum(p.numel() for p in fusion.parameters()))
    print("P3/P4 unchanged:", fused[0] is features[0] and fused[1] is features[1])
    print("Output shapes:", [tuple(f.shape) for f in fused])
    print("Fusion backward: passed")
    if args.yolo:
        from tpg_yolo.yolo import TPGYOLO

        model = TPGYOLO(nc=12).train()  # Random initialization; no downloads.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0005)
        batch = {
            "img": torch.rand(2, 3, 64, 64),
            "z_t": z_t,
            "batch_idx": torch.tensor([0, 1]),
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25], [0.4, 0.4, 0.2, 0.2]]),
        }
        optimizer.zero_grad()
        loss, components = model(batch)
        loss.sum().backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            predictions = model(batch["img"], z_t=z_t)[0]
        print("YOLO one-batch optimization: passed")
        print("Finite loss:", bool(torch.isfinite(loss).all()))
        print("Decoded predictions before NMS:", tuple(predictions.shape))
        print("Synthetic wiring check only; these are not manuscript results.")


if __name__ == "__main__":
    main()
