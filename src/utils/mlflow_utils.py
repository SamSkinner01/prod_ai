import mlflow
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple
from pathlib import Path
import tempfile
import torchvision


def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """
    Denormalize a tensor for visualization.

    Args:
        tensor: Normalized tensor
        mean: Mean used for normalization
        std: Std used for normalization

    Returns:
        Denormalized tensor
    """
    mean = torch.tensor(mean).view(-1, 1, 1)
    std = torch.tensor(std).view(-1, 1, 1)
    return tensor * std + mean


def log_misclassified_images_mlflow(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    step: int,
    max_images: int = 16,
    mean: List[float] = [0.0],
    std: List[float] = [1.0],
    artifact_path: str = "misclassified"
) -> int:
    """
    Log misclassified images to MLflow.

    Args:
        images: Batch of images
        predictions: Predicted class indices
        targets: Ground truth class indices
        class_names: List of class names
        step: Current step/epoch number
        max_images: Maximum number of images to log
        mean: Mean used for normalization
        std: Std used for normalization
        artifact_path: Path prefix for artifacts

    Returns:
        Number of misclassified images logged
    """
    # Find misclassified indices
    misclassified_mask = predictions != targets
    misclassified_indices = torch.where(misclassified_mask)[0]

    if len(misclassified_indices) == 0:
        return 0

    # Limit to max_images
    num_to_log = min(len(misclassified_indices), max_images)
    indices = misclassified_indices[:num_to_log]

    # Get misclassified images, predictions, and targets
    misc_images = images[indices].cpu()
    misc_preds = predictions[indices].cpu()
    misc_targets = targets[indices].cpu()

    # Denormalize images
    misc_images = denormalize(misc_images, mean, std)
    misc_images = torch.clamp(misc_images, 0, 1)

    # Create a grid of images
    grid = torchvision.utils.make_grid(misc_images, nrow=4, normalize=False, pad_value=1)

    # Add labels
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    ax.axis('off')

    # Add title with predictions vs ground truth
    title_lines = []
    for i, (pred, target) in enumerate(zip(misc_preds, misc_targets)):
        if i % 4 == 0 and i > 0:
            title_lines.append('\n')
        title_lines.append(f"Pred: {class_names[pred]} | GT: {class_names[target]}  ")

    fig.suptitle(''.join(title_lines), fontsize=10, y=0.98)
    plt.tight_layout()

    # Save to temporary file and log to MLflow
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
        mlflow.log_artifact(tmp.name, artifact_path=f"{artifact_path}/step_{step}")
        Path(tmp.name).unlink()

    plt.close(fig)

    return num_to_log


def log_sample_predictions_mlflow(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    probabilities: torch.Tensor,
    class_names: List[str],
    step: int,
    num_images: int = 16,
    mean: List[float] = [0.0],
    std: List[float] = [1.0],
    artifact_path: str = "predictions"
) -> None:
    """
    Log sample predictions to MLflow with confidence scores.

    Args:
        images: Batch of images
        predictions: Predicted class indices
        targets: Ground truth class indices
        probabilities: Prediction probabilities
        class_names: List of class names
        step: Current step/epoch number
        num_images: Number of images to log
        mean: Mean used for normalization
        std: Std used for normalization
        artifact_path: Path prefix for artifacts
    """
    # Limit to num_images
    num_to_log = min(len(images), num_images)
    sample_images = images[:num_to_log].cpu()
    sample_preds = predictions[:num_to_log].cpu()
    sample_targets = targets[:num_to_log].cpu()
    sample_probs = probabilities[:num_to_log].cpu()

    # Denormalize images
    sample_images = denormalize(sample_images, mean, std)
    sample_images = torch.clamp(sample_images, 0, 1)

    # Create figure with subplots
    ncols = 4
    nrows = (num_to_log + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3 * nrows))
    axes = axes.flatten() if num_to_log > 1 else [axes]

    for idx in range(num_to_log):
        img = sample_images[idx]
        pred = sample_preds[idx].item()
        target = sample_targets[idx].item()
        prob = sample_probs[idx].item()

        # Handle grayscale vs RGB
        if img.shape[0] == 1:
            axes[idx].imshow(img.squeeze(), cmap='gray')
        else:
            axes[idx].imshow(img.permute(1, 2, 0))

        # Color code: green if correct, red if wrong
        color = 'green' if pred == target else 'red'
        axes[idx].set_title(
            f'Pred: {class_names[pred]} ({prob:.2%})\nGT: {class_names[target]}',
            color=color,
            fontsize=8
        )
        axes[idx].axis('off')

    # Hide extra subplots
    for idx in range(num_to_log, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()

    # Save to temporary file and log to MLflow
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
        mlflow.log_artifact(tmp.name, artifact_path=f"{artifact_path}/step_{step}")
        Path(tmp.name).unlink()

    plt.close(fig)


def log_confusion_matrix_samples_mlflow(
    images: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
    class_names: List[str],
    step: int,
    mean: List[float] = [0.0],
    std: List[float] = [1.0],
    artifact_path: str = "confusion"
) -> None:
    """
    Log examples from the confusion matrix (common misclassifications) to MLflow.

    Args:
        images: Batch of images
        predictions: Predicted class indices
        targets: Ground truth class indices
        class_names: List of class names
        step: Current step/epoch number
        mean: Mean used for normalization
        std: Std used for normalization
        artifact_path: Path prefix for artifacts
    """
    # Find all unique (target, prediction) pairs where they differ
    misclassified_mask = predictions != targets
    if misclassified_mask.sum() == 0:
        return

    misc_images = images[misclassified_mask].cpu()
    misc_preds = predictions[misclassified_mask].cpu()
    misc_targets = targets[misclassified_mask].cpu()

    # Group by (target, prediction) pairs
    pairs = {}
    for img, pred, target in zip(misc_images, misc_preds, misc_targets):
        key = (target.item(), pred.item())
        if key not in pairs:
            pairs[key] = []
        if len(pairs[key]) < 4:  # Keep max 4 examples per pair
            pairs[key].append(img)

    # Log top confusion pairs
    sorted_pairs = sorted(pairs.items(), key=lambda x: len(x[1]), reverse=True)
    for (target, pred), imgs in sorted_pairs[:5]:  # Top 5 confusion pairs
        if len(imgs) > 0:
            imgs_tensor = torch.stack(imgs)
            imgs_tensor = denormalize(imgs_tensor, mean, std)
            imgs_tensor = torch.clamp(imgs_tensor, 0, 1)

            grid = torchvision.utils.make_grid(imgs_tensor, nrow=4, normalize=False, pad_value=1)

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.imshow(grid.permute(1, 2, 0).numpy())
            ax.set_title(f'GT: {class_names[target]} → Pred: {class_names[pred]}')
            ax.axis('off')
            plt.tight_layout()

            # Save to temporary file and log to MLflow
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                plt.savefig(tmp.name, bbox_inches='tight', dpi=100)
                mlflow.log_artifact(
                    tmp.name,
                    artifact_path=f"{artifact_path}/step_{step}/GT_{class_names[target]}_Pred_{class_names[pred]}"
                )
                Path(tmp.name).unlink()

            plt.close(fig)
