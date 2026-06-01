import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic ResNet block with two 3x3 convolutions.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        Initialize BasicBlock.

        Args:
            in_planes: Number of input channels
            planes: Number of output channels
            stride: Stride for the first convolution
        """
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the block."""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet architecture adapted for CIFAR-10.

    Uses smaller initial kernel and no max pooling for 32x32 images.
    """

    def __init__(self, block: nn.Module, num_blocks: list, num_classes: int = 10):
        """
        Initialize ResNet.

        Args:
            block: Building block (BasicBlock or Bottleneck)
            num_blocks: List of number of blocks in each layer
            num_classes: Number of output classes
        """
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR-10 adapted: smaller kernel, stride 1, no max pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block: nn.Module, planes: int, num_blocks: int, stride: int):
        """Create a layer with multiple blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


def ResNet18(num_classes: int = 10) -> ResNet:
    """
    Construct ResNet-18 model for CIFAR-10.

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)

    Returns:
        ResNet-18 model
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


def ResNet34(num_classes: int = 10) -> ResNet:
    """
    Construct ResNet-34 model for CIFAR-10.

    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)

    Returns:
        ResNet-34 model
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)
