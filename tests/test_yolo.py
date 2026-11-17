import copy
import io
import unittest

import torch

from tpg_yolo.yolo import TPGYOLO


class YoloTests(unittest.TestCase):
    def test_training_inference_and_reload(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        model = TPGYOLO(nc=2).train()
        batch = dict(
            img=torch.rand(2, 3, 64, 64, requires_grad=True),
            z_t=torch.randn(2, 256, requires_grad=True),
            batch_idx=torch.tensor([0, 1]), cls=torch.tensor([[0.], [1.]]),
            bboxes=torch.tensor([[.5, .5, .3, .3], [.5, .5, .2, .2]]),
        )
        before = model.texture_fusion.encoder.layers[0].weight.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01)
        loss, _ = model(batch)
        self.assertTrue(torch.isfinite(loss).all())
        loss.sum().backward()
        self.assertIsNone(batch["img"].grad)
        self.assertIsNone(batch["z_t"].grad)
        self.assertEqual((model.args.box, model.args.cls, model.args.dfl), (7.5, 0.5, 1.5))
        for p in model.texture_fusion.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model.parameters()))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.texture_fusion.encoder.layers[0].weight))
        model.eval()
        with torch.no_grad():
            y = model(batch["img"], z_t=batch["z_t"])[0]
            self.assertTrue(torch.isfinite(y).all())
            clone = copy.deepcopy(model)
            torch.testing.assert_close(clone(batch["img"], z_t=batch["z_t"])[0], y)
            # Portable state_dict, rather than pickling model objects.
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = TPGYOLO(nc=2).eval()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            torch.testing.assert_close(restored(batch["img"], z_t=batch["z_t"])[0], y)
            with self.assertRaises(ValueError):
                model(batch["img"])
            with self.assertRaises(ValueError):
                model(batch["img"], z_t=torch.zeros(1, 256))
            self.assertIsNone(model._active_texture)
            torch.testing.assert_close(model(batch["img"], z_t=batch["z_t"])[0], y)

    def test_fusion_position_and_head_inputs(self):
        torch.set_num_threads(2)
        model = TPGYOLO(nc=2).eval()
        seen = {}

        def record_before(module, args):
            seen["before"] = list(args[0])

        def record_after(module, args):
            seen["after"] = list(args[0])

        head = model.model[-1]
        before_hook = head.register_forward_pre_hook(record_before, prepend=True)
        after_hook = head.register_forward_pre_hook(record_after)
        z = torch.randn(1, 256)
        try:
            with torch.no_grad():
                model(torch.rand(1, 3, 64, 64), z_t=z)
                expected = model.texture_fusion(seen["before"], z)
            self.assertIs(seen["before"][0], seen["after"][0])
            self.assertIs(seen["before"][1], seen["after"][1])
            torch.testing.assert_close(seen["after"][2], expected[2])
            self.assertEqual(model.texture_fusion.channels, 512)
        finally:
            before_hook.remove()
            after_hook.remove()


if __name__ == "__main__":
    unittest.main()
