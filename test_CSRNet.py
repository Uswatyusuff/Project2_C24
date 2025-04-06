import unittest
import torch
import torch.nn as nn
from model import CSRNet, make_layers


class TestCSRNetModel(unittest.TestCase):

    def setUp(self):
        """Create a model instance for testing"""
        self.model = CSRNet(load_weights=True)  # Change to False if no internet
        self.model.eval()

    def test_model_structure(self):
        """Test if model has frontend, backend, and output_layer"""
        self.assertIsInstance(self.model.frontend, nn.Sequential)
        self.assertIsInstance(self.model.backend, nn.Sequential)
        self.assertIsInstance(self.model.output_layer, nn.Conv2d)

    def test_make_layers_output(self):
        """Test make_layers builds a valid nn.Sequential"""
        cfg = [64, 'M', 128]
        layers = make_layers(cfg)
        self.assertIsInstance(layers, nn.Sequential)
        # Expecting 5 layers: Conv+ReLU, MaxPool, Conv+ReLU
        self.assertEqual(len(layers), 5)

    def test_output_shape(self):
        """Test model output shape with dummy input"""
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = self.model(dummy_input)
        self.assertEqual(output.shape[1], 1)  # Output channels should be 1

    def test_forward_pass_no_error(self):
        """Test if forward pass completes without error"""
        dummy_input = torch.randn(1, 3, 224, 224)
        try:
            _ = self.model(dummy_input)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

    def test_weights_initialization(self):
        """Check that convolution weights are initialized (not all zeros)"""
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                self.assertFalse(torch.all(m.weight == 0), "Conv2d weights should not be all zeros")

    def test_output_layer_config(self):
        """Test if output layer reduces channels correctly"""
        self.assertEqual(self.model.output_layer.out_channels, 1)
        self.assertEqual(self.model.output_layer.kernel_size, (1, 1))


if __name__ == "__main__":
    unittest.main()
