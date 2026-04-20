from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn, optim

from dataloader import build_cifar10_dataloaders
from model import MLPMixer
from trainer import ClassificationTrainer

DATA_ROOT = ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 20


def main():
    dataloaders, _, class_names = build_cifar10_dataloaders(batch_size=BATCH_SIZE, root=DATA_ROOT, download=True)
    model = MLPMixer(
        image_size=32,
        channels=3,
        patch_size=4,
        dim=256,
        depth=6,
        num_classes=len(class_names),
        token_expansion=0.5,
        channel_expansion=4,
        dropout=0.1,
    )
    trainer = ClassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW(model.parameters(), lr=3e-4),
        device=DEVICE,
        save_path=Path(__file__).with_name("best.pt"),
    )
    trainer.fit(dataloaders, num_epochs=EPOCHS)


if __name__ == "__main__":
    main()
