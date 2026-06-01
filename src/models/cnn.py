import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """
    Convolutional Neural Network for MNIST digit classification.

    Architecture:
        - Conv2d (1 -> 32, 3x3) + ReLU + MaxPool
        - Conv2d (32 -> 64, 3x3) + ReLU + MaxPool
        - Dropout (0.25)
        - Fully Connected (9216 -> 128) + ReLU
        - Dropout (0.5)
        - Fully Connected (128 -> 10)
    """

    def __init__(self, dropout_p: float = 0.5):
        """
        Initialize the MNIST CNN.

        Args:
            dropout_p: Dropout probability for regularization
        """
        super(MNISTNet, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout layers
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(dropout_p)

        # Fully connected layers
        # After 2 pooling layers: 28x28 -> 14x14 -> 7x7
        # 64 channels * 7 * 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)

        Returns:
            Output logits of shape (batch_size, 10)
        """
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Dropout after conv layers
        x = self.dropout1(x)

        # Flatten for FC layers
        x = torch.flatten(x, 1)

        # First FC layer
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        # Output layer
        x = self.fc2(x)

        return x
