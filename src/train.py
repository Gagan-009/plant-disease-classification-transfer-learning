import os
import sys
import time
import datetime
import argparse

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim

from src.utils import set_seed, save_checkpoint, load_checkpoint, log_experiment
from src.dataset import get_dataloader
from src.models import build_resnet18_feature_extractor, count_parameters
from src.metrics import compute_classification_metrics
from src.predict import predict_single_image

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = float(epoch_loss)
    return metrics


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_targets, all_preds)
    metrics["loss"] = float(epoch_loss)
    return metrics


def run_pipeline(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/val.csv",
    test_csv: str = "data/splits/test.csv",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 0.001,
    samples_per_class: int = 5,
    seed: int = 42,
    checkpoint_path: str = "models/resnet18_feature_extractor_smoke.pth",
    ledger_path: str = "results/metrics/experiment_ledger.csv"
):
    print("=" * 70)
    print("PLANT DISEASE CLASSIFICATION - RESNET18 FEATURE EXTRACTION PIPELINE")
    print("=" * 70)

    # 1. Set seed for exact reproducibility
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device in use: {device}")

    # 2. Build DataLoaders
    print(f"\n[1/7] Loading stratified datasets (samples_per_class={samples_per_class})...")
    train_loader, train_dataset = get_dataloader(
        csv_file=train_csv,
        batch_size=batch_size,
        is_train=True,
        samples_per_class=samples_per_class,
        seed=seed
    )
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_to_idx)

    val_loader, val_dataset = get_dataloader(
        csv_file=val_csv,
        batch_size=batch_size,
        is_train=False,
        samples_per_class=min(2, samples_per_class) if samples_per_class else None,
        class_to_idx=class_to_idx,
        seed=seed
    )

    test_loader, test_dataset = get_dataloader(
        csv_file=test_csv,
        batch_size=batch_size,
        is_train=False,
        samples_per_class=min(2, samples_per_class) if samples_per_class else None,
        class_to_idx=class_to_idx,
        seed=seed
    )

    print(f"Verified classes: {num_classes}")
    print(f"Train subset size: {len(train_dataset)} images")
    print(f"Validation subset size: {len(val_dataset)} images")
    print(f"Test subset size: {len(test_dataset)} images")

    # 3. Model initialization & Backbone Freezing
    print("\n[2/7] Initializing official ImageNet pretrained ResNet18...")
    model = build_resnet18_feature_extractor(num_classes=num_classes, freeze_backbone=True)
    model.to(device)
    param_counts = count_parameters(model)
    print(f"Total parameters: {param_counts['total_params']:,}")
    print(f"Frozen backbone parameters: {param_counts['frozen_params']:,}")
    print(f"Trainable classifier head parameters: {param_counts['trainable_params']:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    # 4. Training loop
    print(f"\n[3/7] Training feature extractor for {epochs} epochs...")
    start_time = time.time()

    best_val_f1 = -1.0
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m = evaluate(model, val_loader, criterion, device)
        epoch_sec = time.time() - epoch_start

        print(f"Epoch [{epoch}/{epochs}] ({epoch_sec:.1f}s) | "
              f"Train Loss: {train_m['loss']:.4f}, Acc: {train_m['accuracy']:.4f}, Macro-F1: {train_m['macro_f1']:.4f} | "
              f"Val Loss: {val_m['loss']:.4f}, Acc: {val_m['accuracy']:.4f}, Macro-F1: {val_m['macro_f1']:.4f}")
        best_val_f1 = max(best_val_f1, val_m["macro_f1"])

    total_train_time = time.time() - start_time
    print(f"Training completed in {total_train_time:.1f} seconds.")

    # 5. Checkpoint Save
    print(f"\n[4/7] Saving checkpoint to {checkpoint_path}...")
    save_checkpoint(
        filepath=checkpoint_path,
        model=model,
        class_to_idx=class_to_idx,
        optimizer=optimizer,
        epoch=epochs,
        extra_info={"num_classes": num_classes, "architecture": "ResNet18-FeatureExtraction"}
    )

    # 6. Checkpoint Reload & Test Set Evaluation
    print(f"\n[5/7] Reloading checkpoint to verify restore integrity...")
    reloaded_model = build_resnet18_feature_extractor(num_classes=num_classes, freeze_backbone=True)
    load_checkpoint(checkpoint_path, reloaded_model, device=device)

    print("\n[6/7] Evaluating reloaded model on Test set...")
    test_m = evaluate(reloaded_model, test_loader, criterion, device)
    print("Test Evaluation Results:")
    print(f"  Test Loss:            {test_m['loss']:.4f}")
    print(f"  Test Accuracy:        {test_m['accuracy']:.4f} ({test_m['accuracy']*100:.2f}%)")
    print(f"  Test Macro-Precision: {test_m['macro_precision']:.4f}")
    print(f"  Test Macro-Recall:    {test_m['macro_recall']:.4f}")
    print(f"  Test Macro-F1:        {test_m['macro_f1']:.4f}")

    # 7. Single-Image Inference Test
    sample_test_img = test_dataset.filepaths[0]
    true_class_name = test_dataset.classes[test_dataset.labels[0]]
    print(f"\n[7/7] Executing single-image inference on: {sample_test_img}...")
    inf_res = predict_single_image(sample_test_img, checkpoint_path, device=device.type)
    print(f"  True Label:      {true_class_name}")
    print(f"  Predicted Label: {inf_res['predicted_class']}")
    print(f"  Confidence:      {inf_res['confidence']:.4f} ({inf_res['confidence']*100:.2f}%)")

    # 8. Log into Experiment Ledger
    record = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "experiment_name": "ResNet18_Feature_Extraction_Smoke_Test",
        "architecture": "ResNet18",
        "strategy": "Feature_Extraction_Frozen_Backbone",
        "pretrained_weights": "ImageNet_Default",
        "num_classes": num_classes,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "total_params": param_counts["total_params"],
        "trainable_params": param_counts["trainable_params"],
        "training_time_sec": round(total_train_time, 2),
        "test_loss": round(test_m["loss"], 4),
        "test_accuracy": round(test_m["accuracy"], 4),
        "test_macro_precision": round(test_m["macro_precision"], 4),
        "test_macro_recall": round(test_m["macro_recall"], 4),
        "test_macro_f1": round(test_m["macro_f1"], 4),
        "checkpoint_file": checkpoint_path
    }
    log_experiment(ledger_path, record)
    print("\n" + "=" * 70)
    print("ALL MILESTONE 2 DELIVERABLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    return record

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 feature extractor on Plant Disease dataset")
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--samples_per_class", type=int, default=5, help="Samples per class for smoke test (None for full)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    run_pipeline(
        epochs=args.epochs,
        batch_size=args.batch_size,
        samples_per_class=args.samples_per_class,
        seed=args.seed
    )
