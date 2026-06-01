# Vision Training Pipeline with Docker and Hydra

A complete computer vision training pipeline supporting MNIST and CIFAR-10 datasets with Docker containerization, Hydra configuration management, and GPU-accelerated data augmentation using torchvision v2.

## Features

- **Multiple Datasets**: Support for MNIST and CIFAR-10 datasets
- **Efficient Data Loading**: Custom datasets with `__getitems__` method for batch reading
- **GPU Augmentation**: Dataset-specific data augmentation on GPU using torchvision v2 transforms
- **Hydra Configuration**: Flexible CLI and configuration management with Hydra
- **Docker Support**: Fully containerized with NVIDIA GPU support, single image for both datasets
- **Multiple Architectures**:
  - CNN for MNIST (grayscale 28x28)
  - ResNet-18 for CIFAR-10 (RGB 32x32)

## Project Structure

```
mnist/
├── config/
│   ├── config.yaml          # Hydra config for MNIST
│   └── config_cifar.yaml    # Hydra config for CIFAR-10
├── src/
│   ├── data/
│   │   ├── mnist_dataset.py # MNIST dataset with __getitems__
│   │   └── cifar_dataset.py # CIFAR-10 dataset with __getitems__
│   ├── models/
│   │   ├── cnn.py           # CNN for MNIST
│   │   └── resnet.py        # ResNet for CIFAR-10
│   └── utils/
├── train.py                 # MNIST training script
├── train_cifar.py           # CIFAR-10 training script
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration (shared)
├── docker-compose.yml       # Multi-service orchestration
└── .dockerignore            # Docker ignore rules
```

## Installation

### Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run training:
```bash
# MNIST
python train.py

# CIFAR-10
python train_cifar.py
```

### Docker Setup with docker-compose (Recommended)

```bash
# Build the shared image
docker-compose build

# Train MNIST
docker-compose up mnist-trainer

# Train CIFAR-10
docker-compose up cifar-trainer

# Train both simultaneously (not recommended if sharing same GPU)
docker-compose up
```

### Manual Docker Setup

1. Build the Docker image:
```bash
docker build -t vision-trainer .
```

2. Run training with GPU support:
```bash
# MNIST
docker run --gpus all -v ${PWD}/data:/app/data -v ${PWD}/checkpoints:/app/checkpoints vision-trainer python3 train.py

# CIFAR-10
docker run --gpus all -v ${PWD}/data:/app/data -v ${PWD}/checkpoints:/app/checkpoints vision-trainer python3 train_cifar.py
```

## Configuration with Hydra

The training is configured using Hydra. You can override any configuration parameter from the command line:

### MNIST Configuration

```bash
# Train with default settings
python train.py

# Change batch size and learning rate
python train.py training.batch_size=256 training.learning_rate=0.0005

# Train for more epochs
python train.py training.epochs=50

# Adjust model dropout
python train.py model.dropout_p=0.3

# Disable random erasing augmentation
python train.py augmentation.random_erasing.enabled=false

# Increase augmentation strength
python train.py augmentation.random_affine.degrees=15 augmentation.random_affine.scale=[0.8,1.2]
```

### CIFAR-10 Configuration

```bash
# Train with default settings
python train_cifar.py

# Change batch size and learning rate
python train_cifar.py training.batch_size=512 training.learning_rate=0.05

# Train for more epochs
python train_cifar.py training.epochs=300

# Disable color jitter
python train_cifar.py augmentation.color_jitter.enabled=false

# Adjust augmentation parameters
python train_cifar.py augmentation.random_horizontal_flip.p=0.7 augmentation.random_crop.padding=8

# With docker-compose run
docker-compose run --rm cifar-trainer python3 train_cifar.py augmentation.random_erasing.enabled=false
```

### Configuration Parameters

**Data Configuration (both datasets):**
- `data.root`: Path to store dataset (default: "./data")
- `data.download`: Download dataset if not present (default: true)
- `data.num_workers`: Number of data loading workers (default: 4)

**MNIST Model Configuration:**
- `model.dropout_p`: Dropout probability (default: 0.5)

**MNIST Training Configuration:**
- `training.batch_size`: Batch size (default: 128)
- `training.epochs`: Number of epochs (default: 20)
- `training.learning_rate`: Learning rate (default: 0.001, Adam optimizer)
- `training.weight_decay`: Weight decay for regularization (default: 0.0001)
- `training.scheduler_step_size`: LR scheduler step size (default: 7)
- `training.scheduler_gamma`: LR scheduler decay factor (default: 0.1)
- `training.checkpoint_dir`: Directory to save checkpoints (default: "./checkpoints")

**CIFAR-10 Model Configuration:**
- `model.architecture`: Model architecture (default: "ResNet18")
- `model.num_classes`: Number of output classes (default: 10)

**CIFAR-10 Training Configuration:**
- `training.batch_size`: Batch size (default: 128)
- `training.epochs`: Number of epochs (default: 200)
- `training.learning_rate`: Learning rate (default: 0.1, SGD optimizer)
- `training.momentum`: SGD momentum (default: 0.9)
- `training.weight_decay`: Weight decay for regularization (default: 0.0005)
- Uses CosineAnnealingLR scheduler
- `training.checkpoint_dir`: Directory to save checkpoints (default: "./checkpoints")

**MNIST Augmentation Configuration:**
- `augmentation.random_affine.enabled`: Enable random affine transforms (default: true)
- `augmentation.random_affine.degrees`: Rotation range in degrees (default: 10)
- `augmentation.random_affine.translate`: Translation range (default: [0.1, 0.1])
- `augmentation.random_affine.scale`: Scale range (default: [0.9, 1.1])
- `augmentation.random_erasing.enabled`: Enable random erasing (default: true)
- `augmentation.random_erasing.p`: Probability of applying (default: 0.1)
- `augmentation.random_erasing.scale`: Size range (default: [0.02, 0.1])
- `augmentation.normalize.mean`: Normalization mean (default: [0.1307])
- `augmentation.normalize.std`: Normalization std (default: [0.3081])

**CIFAR-10 Augmentation Configuration:**
- `augmentation.random_crop.enabled`: Enable random crop (default: true)
- `augmentation.random_crop.size`: Crop size (default: 32)
- `augmentation.random_crop.padding`: Padding before crop (default: 4)
- `augmentation.random_horizontal_flip.enabled`: Enable horizontal flip (default: true)
- `augmentation.random_horizontal_flip.p`: Flip probability (default: 0.5)
- `augmentation.color_jitter.enabled`: Enable color jitter (default: true)
- `augmentation.color_jitter.brightness`: Brightness factor (default: 0.2)
- `augmentation.color_jitter.contrast`: Contrast factor (default: 0.2)
- `augmentation.color_jitter.saturation`: Saturation factor (default: 0.2)
- `augmentation.color_jitter.hue`: Hue factor (default: 0.0)
- `augmentation.random_erasing.enabled`: Enable random erasing (default: true)
- `augmentation.random_erasing.p`: Probability of applying (default: 0.1)
- `augmentation.random_erasing.scale`: Size range (default: [0.02, 0.1])
- `augmentation.normalize.mean`: Normalization mean (default: [0.4914, 0.4822, 0.4465])
- `augmentation.normalize.std`: Normalization std (default: [0.2470, 0.2435, 0.2616])

## GPU Augmentation

The project uses torchvision v2 transforms applied on the GPU for better performance. **All augmentations are fully configurable via Hydra** - you can enable/disable individual augmentations or adjust their parameters from the command line or config files.

**MNIST Training Augmentations:**
- Random affine transformations (rotation, translation, scaling)
- Random erasing for regularization
- Normalization with MNIST statistics (mean: 0.1307, std: 0.3081)

**CIFAR-10 Training Augmentations:**
- Random crop with padding (32x32 with 4px padding)
- Random horizontal flip (50% probability)
- Color jitter (brightness, contrast, saturation)
- Random erasing for regularization
- Normalization with CIFAR-10 statistics (mean: [0.4914, 0.4822, 0.4465], std: [0.2470, 0.2435, 0.2616])

**Validation (both datasets):**
- Only normalization (no augmentation)

## Dataset Implementation

Both `MNISTDataset` and `CIFAR10Dataset` classes implement `__getitem__` and `__getitems__` methods:

- `__getitem__(idx)`: Returns a single sample
- `__getitems__(indices)`: Returns multiple samples in batch for efficient reading

This allows the DataLoader to retrieve batches more efficiently, as requested.

## Model Architectures

**MNIST CNN Model:**
- 2 convolutional layers (32 and 64 filters, 3x3 kernels)
- Max pooling layers (2x2)
- Dropout for regularization (25% after conv, 50% after FC)
- 2 fully connected layers (128 hidden units, 10 output classes)
- ~1.2M parameters

**CIFAR-10 ResNet-18 Model:**
- 18 layers total with residual connections
- Adapted for 32x32 images (smaller initial kernel, no max pooling)
- 4 residual layer groups (64, 128, 256, 512 channels)
- Batch normalization throughout
- Global average pooling + FC layer (10 output classes)
- ~11M parameters

## Docker Architecture

**Single Image for Both Datasets:**
- Uses one Docker image (`vision-trainer`) for both MNIST and CIFAR-10
- Different entry points via docker-compose services
- Shared dependencies (no additional packages needed)
- NVIDIA CUDA 12.1 with cuDNN 8 for GPU support
- Ubuntu 22.04 base with Python 3.10

**Benefits:**
- Easier maintenance (one image to update)
- Smaller disk usage
- Identical environment for both tasks

To run with specific GPU:
```bash
docker run --gpus '"device=0"' -v ${PWD}/data:/app/data vision-trainer python3 train.py
```

## Outputs

- **Logs**: Hydra automatically creates timestamped output directories:
  - MNIST: `./outputs/YYYY-MM-DD/HH-MM-SS/`
  - CIFAR-10: `./outputs/cifar/YYYY-MM-DD/HH-MM-SS/`
- **Checkpoints**: Best models saved to:
  - MNIST: `./checkpoints/best_model.pt`
  - CIFAR-10: `./checkpoints/best_model_cifar.pt`
- **Metrics**: Training and validation loss/accuracy logged during training

## Example Training Output

### MNIST Training

```
Configuration:
data:
  root: ./data
  download: true
  num_workers: 4
model:
  dropout_p: 0.5
training:
  batch_size: 128
  epochs: 20
  learning_rate: 0.001
  ...

Using device: cuda
Loading datasets...
Creating model...
Model parameters: 1,199,882
Starting training...

Epoch 1/20: 100%|███| 469/469 [00:15<00:00, 30.12it/s, loss=0.234, acc=92.8%]
Validating: 100%|████████████| 79/79 [00:02<00:00, 35.21it/s]
Epoch 1/20 - Train Loss: 0.2340, Val Loss: 0.0876, Val Acc: 97.23%
Saved best model with accuracy: 97.23%
...
```

### CIFAR-10 Training

```
CIFAR-10 Training Configuration:
data:
  root: ./data
  download: true
  num_workers: 4
model:
  architecture: ResNet18
  num_classes: 10
training:
  batch_size: 128
  epochs: 200
  learning_rate: 0.1
  momentum: 0.9
  ...

Using device: cuda
Loading CIFAR-10 datasets...
Creating ResNet18 model...
Model parameters: 11,173,962
Starting training...

Epoch 1/200: 100%|███| 391/391 [00:45<00:00, 8.64it/s, loss=1.845, acc=32.4%]
Validating: 100%|████████████| 79/79 [00:05<00:00, 14.23it/s]
Epoch 1/200 - LR: 0.100000, Train Loss: 1.8451, Val Loss: 1.623, Val Acc: 41.23%
Saved best model with accuracy: 41.23%
...
```

## License

MIT
