import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tpg_yolo import TextureP5Fusion
from tpg_yolo.priors import fixed_shuffle_indices, load_texture_prior


class CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_formula_parameters_and_identity(self):
        model = TextureP5Fusion()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 531969)
        features = [torch.randn(2, c, h, h) for c, h in [(128, 8), (256, 4), (512, 2)]]
        z = torch.randn(2, 256)
        result = model(features, z)
        self.assertIs(result[0], features[0])
        self.assertIs(result[1], features[1])
        # Independently assemble the normalization, resizing and residual formula.
        standardized = (z - z.mean(-1, keepdim=True)) / (z.var(-1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        latent = F.interpolate(standardized.reshape(2, 1, 16, 16), (2, 2), mode="bilinear", align_corners=False)
        residual = model.encoder(features[2], model.adapter.projection(latent))
        torch.testing.assert_close(result[2], features[2] + model.gamma * residual)
        with torch.no_grad():
            model.gamma.zero_()
        torch.testing.assert_close(model(features, z)[2], features[2], rtol=0, atol=0)
        with torch.no_grad():
            model.gamma.fill_(-0.2)
        torch.testing.assert_close(model(features, z)[2], features[2] - 0.2 * residual)

    def test_texture_dependence_and_backward(self):
        model = TextureP5Fusion(16)
        features = [torch.randn(2, c, 5, 7, requires_grad=True) for c in (4, 8, 16)]
        z = torch.randn(2, 256, requires_grad=True)
        result = model(features, z)
        self.assertFalse(torch.allclose(result[2], model(features, torch.zeros_like(z))[2]))
        result[2].square().mean().backward()
        self.assertGreater(z.grad.abs().sum().item(), 0)
        self.assertGreater(features[2].grad.abs().sum().item(), 0)
        for param in model.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())

    def test_paper_layer_configuration(self):
        model = TextureP5Fusion()
        self.assertFalse(model.adapter.norm.elementwise_affine)
        self.assertEqual(tuple(model.adapter.norm.normalized_shape), (256,))
        self.assertAlmostEqual(model.gamma.item(), 0.1, places=7)
        layers = model.encoder.layers
        self.assertEqual([type(x).__name__ for x in layers],
                         ["Conv2d", "GroupNorm", "SiLU", "Conv2d", "GroupNorm", "SiLU"])
        self.assertEqual((layers[0].in_channels, layers[0].out_channels), (1024, 512))
        self.assertEqual(layers[0].kernel_size, (1, 1))
        self.assertIsNone(layers[0].bias)
        self.assertIsNone(layers[3].bias)
        self.assertEqual(layers[3].kernel_size, (3, 3))
        self.assertEqual(layers[3].groups, 512)
        self.assertEqual((layers[1].num_groups, layers[4].num_groups), (1, 1))

    def test_reject_invalid_input(self):
        model = TextureP5Fusion(16)
        features = [torch.randn(2, c, 4, 4) for c in (4, 8, 16)]
        for z in (torch.zeros(2, 255), torch.zeros(1, 256), torch.full((2, 256), float("nan")), torch.ones(2, 256, dtype=torch.long)):
            with self.assertRaises(ValueError):
                model(features, z)

    def test_prior_identity_and_shuffle(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "sample.npz"
            z = np.linspace(-1, 1, 256, dtype=np.float32)
            np.savez(path, image_id="sample", z_t=z)
            torch.testing.assert_close(load_texture_prior(path, "sample"), torch.from_numpy(z))
            with self.assertRaises(ValueError):
                load_texture_prior(path, "other")
            np.savez(path, image_id="sample", z_t=z.astype(np.float64))
            with self.assertRaises(ValueError):
                load_texture_prior(path, "sample")
        for n in (2, 3, 19):
            mapping = fixed_shuffle_indices(n, seed=17)
            self.assertTrue(torch.equal(mapping.sort().values, torch.arange(n)))
            self.assertTrue((mapping != torch.arange(n)).all())
            self.assertTrue(torch.equal(mapping, fixed_shuffle_indices(n, 17)))


if __name__ == "__main__":
    unittest.main()
