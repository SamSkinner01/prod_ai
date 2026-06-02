import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
import logging
from pathlib import Path
from tqdm import tqdm
import mlflow
import mlflow.pytorch

from src.data.cifar_dataset import CIFAR10Dataset
from src.models.resnet import ResNet18
from src.utils.mlflow_utils import (
    log_misclassified_images_mlflow,
    log_sample_predictions_mlflow,
    log_confusion_matrix_samples_mlflow
)

log = logging.getLogger(__name__)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


class CIFAR10GPUAugmentation(nn.Module):
    """
    GPU-based data augmentation for CIFAR-10 using torchvision v2 transforms.

    This module applies data augmentation on the GPU for better performance.
    """

    def __init__(self, cfg: DictConfig, training: bool = True):
        """
        Initialize GPU augmentation pipeline for CIFAR-10.

        Args:
            cfg: Augmentation configuration from Hydra
            training: If True, applies training augmentations, otherwise minimal transforms
        """
        super().__init__()
        self.training = training

        if training:
            # Build training augmentations based on config
            transforms = []

            # Random crop
            if cfg.random_crop.enabled:
                transforms.append(v2.RandomCrop(
                    cfg.random_crop.size,
                    padding=cfg.random_crop.padding
                ))

            # Random horizontal flip
            if cfg.random_horizontal_flip.enabled:
                transforms.append(v2.RandomHorizontalFlip(
                    p=cfg.random_horizontal_flip.p
                ))

            # Color jitter
            if cfg.color_jitter.enabled:
                transforms.append(v2.ColorJitter(
                    brightness=cfg.color_jitter.brightness,
                    contrast=cfg.color_jitter.contrast,
                    saturation=cfg.color_jitter.saturation,
                    hue=cfg.color_jitter.hue
                ))

            # Random erasing
            if cfg.random_erasing.enabled:
                transforms.append(v2.RandomErasing(
                    p=cfg.random_erasing.p,
                    scale=tuple(cfg.random_erasing.scale)
                ))

            # Normalization (always applied)
            transforms.append(v2.Normalize(
                mean=cfg.normalize.mean,
                std=cfg.normalize.std
            ))

            self.transform = v2.Compose(transforms)
        else:
            # Validation/test only needs normalization
            self.transform = v2.Normalize(
                mean=cfg.normalize.mean,
                std=cfg.normalize.std
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transforms on GPU."""
        return self.transform(x)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    augmentation: nn.Module,
    epoch: int
) -> float:
    """
    Train for one epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        augmentation: GPU augmentation module
        epoch: Current epoch number

    Returns:
        Average training loss for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (data, target) in enumerate(pbar):
        # Move data to GPU
        data, target = data.to(device), target.to(device)

        # Apply GPU augmentation
        data = augmentation(data)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })

    return running_loss / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    augmentation: nn.Module,
    collect_samples: bool = False
) -> tuple[float, float, dict]:
    """
    Validate the model.

    Args:
        model: Neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        augmentation: GPU augmentation module (without training transforms)
        collect_samples: If True, collects sample images for visualization

    Returns:
        Tuple of (average_loss, accuracy, samples_dict)
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    # For collecting visualization samples
    all_images = []
    all_predictions = []
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(val_loader, desc="Validating")):
            # Store original images (before normalization) for visualization
            original_data = data.clone() if collect_samples and batch_idx == 0 else None

            data, target = data.to(device), target.to(device)

            # Apply GPU normalization (no training augmentations)
            data = augmentation(data)

            output = model(data)
            loss = criterion(output, target)

            val_loss += loss.item()
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = probabilities.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            # Collect samples from first batch for visualization
            if collect_samples and batch_idx == 0:
                all_images = original_data
                all_predictions = predicted.cpu()
                all_targets = target.cpu()
                all_probabilities = confidence.cpu()

    avg_loss = val_loss / len(val_loader)
    accuracy = 100. * correct / total

    samples = {
        'images': all_images,
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities
    } if collect_samples else {}

    return avg_loss, accuracy, samples


@hydra.main(version_base=None, config_path="config", config_name="config_cifar")
def main(cfg: DictConfig) -> None:
    """
    Main training function for CIFAR-10 with Hydra configuration.

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    log.info("CIFAR-10 Training Configuration:\n" + OmegaConf.to_yaml(cfg))

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Create datasets
    log.info("Loading CIFAR-10 datasets...")
    train_dataset = CIFAR10Dataset(
        root=cfg.data.root,
        train=True,
        download=cfg.data.download
    )

    val_dataset = CIFAR10Dataset(
        root=cfg.data.root,
        train=False,
        download=cfg.data.download
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True
    )

    # Create model
    log.info(f"Creating {cfg.model.architecture} model...")
    model = ResNet18(num_classes=cfg.model.num_classes).to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create GPU augmentation modules
    train_augmentation = CIFAR10GPUAugmentation(cfg.augmentation, training=True).to(device)
    val_augmentation = CIFAR10GPUAugmentation(cfg.augmentation, training=False).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg.training.learning_rate,
        momentum=cfg.training.momentum,
        weight_decay=cfg.training.weight_decay
    )

    # Learning rate scheduler with cosine annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.training.epochs
    )

    # MLflow setup - use SQLite backend
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("cifar10_training")
    mlflow.start_run(run_name=f"bs{cfg.training.batch_size}_lr{cfg.training.learning_rate}")

    # Log hyperparameters
    mlflow.log_params({
        "batch_size": cfg.training.batch_size,
        "learning_rate": cfg.training.learning_rate,
        "epochs": cfg.training.epochs,
        "momentum": cfg.training.momentum,
        "weight_decay": cfg.training.weight_decay,
        "architecture": cfg.model.architecture,
        "num_classes": cfg.model.num_classes,
        "optimizer": "SGD",
        "scheduler": "CosineAnnealingLR",
        "augmentation_crop": cfg.augmentation.random_crop.enabled,
        "augmentation_flip": cfg.augmentation.random_horizontal_flip.enabled,
        "augmentation_color_jitter": cfg.augmentation.color_jitter.enabled,
        "augmentation_erasing": cfg.augmentation.random_erasing.enabled,
    })

    log.info("MLflow tracking enabled")
    log.info("View results: mlflow ui --port 5000")

    # Training loop
    log.info("Starting training...")
    best_acc = 0.0

    for epoch in range(1, cfg.training.epochs + 1):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            device, train_augmentation, epoch
        )

        # Validate and collect samples for visualization
        val_loss, val_acc, samples = validate(
            model, val_loader, criterion, device, val_augmentation,
            collect_samples=(epoch % 10 == 0 or epoch == 1)  # Visualize every 10 epochs
        )

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # MLflow logging
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'learning_rate': current_lr
        }, step=epoch)

        # Log sample predictions and misclassified images
        if samples:
            # Log sample predictions with confidence scores
            log_sample_predictions_mlflow(
                images=samples['images'],
                predictions=samples['predictions'],
                targets=samples['targets'],
                probabilities=samples['probabilities'],
                class_names=CIFAR10_CLASSES,
                step=epoch,
                num_images=16,
                mean=cfg.augmentation.normalize.mean,
                std=cfg.augmentation.normalize.std,
                artifact_path='validation_samples'
            )

            # Log misclassified images
            num_misclassified = log_misclassified_images_mlflow(
                images=samples['images'],
                predictions=samples['predictions'],
                targets=samples['targets'],
                class_names=CIFAR10_CLASSES,
                step=epoch,
                max_images=16,
                mean=cfg.augmentation.normalize.mean,
                std=cfg.augmentation.normalize.std,
                artifact_path='misclassified'
            )

            if num_misclassified > 0:
                log.info(f"Logged {num_misclassified} misclassified images to MLflow")

            # Log confusion matrix samples
            log_confusion_matrix_samples_mlflow(
                images=samples['images'],
                predictions=samples['predictions'],
                targets=samples['targets'],
                class_names=CIFAR10_CLASSES,
                step=epoch,
                mean=cfg.augmentation.normalize.mean,
                std=cfg.augmentation.normalize.std,
                artifact_path='confusion'
            )

        # Log results
        log.info(
            f"Epoch {epoch}/{cfg.training.epochs} - "
            f"LR: {current_lr:.6f}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            checkpoint_path = Path(cfg.training.checkpoint_dir) / "best_model_cifar.pt"
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_path)

            log.info(f"Saved best model with accuracy: {best_acc:.2f}%")

    # Log final metrics and model
    mlflow.log_metric("best_val_accuracy", best_acc)

    # Log the best model as an artifact
    mlflow.pytorch.log_model(model, "model")

    log.info(f"Training completed! Best validation accuracy: {best_acc:.2f}%")
    mlflow.end_run()


if __name__ == "__main__":
    main()
