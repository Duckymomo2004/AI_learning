from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ConstantLR, LinearLR, SequentialLR

from dataloader import build_cifar10_dataloaders
from model import resnet50_cbam
from trainer import ClassificationTrainer

DATA_ROOT = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 20


def main():
    dataloaders, _, class_names = build_cifar10_dataloaders(batch_size=BATCH_SIZE, root=DATA_ROOT, download=True)
    model = resnet50_cbam(num_classes=len(class_names))
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    total_steps = EPOCHS * len(dataloaders["train"])
    warmup_steps = max(10, int(0.05 * total_steps))
    decay_steps = int(0.05 * total_steps)
    steady_steps = total_steps - warmup_steps - decay_steps
    scheduler = SequentialLR(
        optimizer,
        [
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps),
            ConstantLR(optimizer, factor=1.0, total_iters=steady_steps),
            LinearLR(optimizer, start_factor=1.0, end_factor=0.05, total_iters=decay_steps),
        ],
        milestones=[warmup_steps, warmup_steps + steady_steps],
    )
    trainer = ClassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=DEVICE,
        save_path=Path(__file__).with_name("best.pt"),
        scheduler_step_per_batch=True,
    )
    trainer.fit(dataloaders, num_epochs=EPOCHS)


if __name__ == "__main__":
    main()
