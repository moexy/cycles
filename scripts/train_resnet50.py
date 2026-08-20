"""Fine-tune ResNet-50 on the EstrousBank multi-lab cytology dataset."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from cycles.core.models import get_device
from cycles.stages.cnn import CNNTrainerService, CNNTrainingConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Fine-tune ResNet-50 on EstrousBank dataset")
    parser.add_argument(
        "--train-dir",
        type=Path,
        default=Path("/Volumes/SSD/Bioinformatics/EstrousBank_Work/splits/train"),
        help="Path to training stage folders",
    )
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=Path("/Volumes/SSD/Bioinformatics/EstrousBank_Work/splits/val"),
        help="Path to validation stage folders",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/resnet50_estrousbank_finetuned.pt"),
        help="Path to save best checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="auto", choices=("auto", "mps", "cuda", "cpu"))
    parser.add_argument("--no-freeze", action="store_true", help="Train all layers instead of freezing backbone")
    args = parser.parse_args()

    device = get_device(args.device if args.device != "auto" else None)
    print("==================================================")
    print("ResNet-50 Fine-Tuning on EstrousBank Dataset")
    print(f"Device: {device}")
    print(f"Train Dir: {args.train_dir}")
    print(f"Val Dir: {args.val_dir}")
    print(f"Output Checkpoint: {args.output}")
    print(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}")
    print(f"Freeze Backbone (layers 1-3): {not args.no_freeze}")
    print("==================================================")

    config = CNNTrainingConfig(
        architecture="resnet50",
        img_size=224,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        freeze_backbone=not args.no_freeze,
        trainable_layers=("layer4", "fc"),
        aggressive_stain_augmentation=True,
        num_workers=4,
        output_path=args.output,
    )

    trainer = CNNTrainerService(device=device)
    t_start = time.perf_counter()

    def on_batch(batch_idx: int, total_batches: int, running_loss: float, running_acc: float) -> None:
        if batch_idx % 25 == 0 or batch_idx == total_batches:
            elapsed = time.perf_counter() - t_start
            print(f"  [Batch {batch_idx:03d}/{total_batches:03d}] Running Loss: {running_loss:.4f}, Acc: {running_acc*100:5.1f}% ({elapsed:.0f}s)")

    def on_epoch(m):
        elapsed = time.perf_counter() - t_start
        print(
            f"-> EPOCH {m.epoch:02d}/{config.epochs:02d} [{elapsed:4.0f}s]: "
            f"Train Loss={m.train_loss:.4f} (Acc={m.train_accuracy*100:5.1f}%) | "
            f"Val Loss={m.val_loss:.4f} (Acc={m.val_accuracy*100:5.1f}%) | "
            f"LR={m.learning_rate:.6f}"
        )

    result = trainer.train(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        config=config,
        progress_callback=on_epoch,
        batch_callback=on_batch,
    )
    print("==================================================")
    print(f"Fine-Tuning Finished in {(time.perf_counter() - t_start):.1f}s!")
    print(f"Best Val Accuracy: {result.best_val_accuracy*100:.1f}% (Epoch {result.best_epoch})")
    print(f"Saved Checkpoint: {result.checkpoint_path}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
