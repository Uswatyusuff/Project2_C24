import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn import functional as F

# Define model URL for pretrained weights
model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features  # ✅ Fix: Ensure self.features is an nn.Sequential

        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x):
        if not isinstance(self.features, nn.Sequential):  # Debugging print
            print(f"❌ ERROR: self.features is of type {type(self.features)}")

        x = self.features(x)  # ✅ Ensure this is a PyTorch module
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reg_layer(x)
        mu = self.density_layer(x)

        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)

        return mu, mu_normed


# Function to create VGG layers
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)  # ✅ Fix: Ensure it returns nn.Sequential


# VGG19 configuration
cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


# Function to load the VGG19 model with optional pre-trained weights
def vgg19():
    """VGG 19-layer model (configuration "E") with pre-trained weights."""
    features = make_layers(cfg['E'])  # ✅ Fix: Ensure this is nn.Sequential
    model = VGG(features)  # ✅ Fix: Pass `features` correctly

    # Load pre-trained weights
    pretrained_dict = model_zoo.load_url(model_urls['vgg19'])
    model_dict = model.state_dict()

    # Only load weights that exist in both state dicts
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    return model  # ✅ Fix: Removed extra parentheses