from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import nn, optim

from dataloader import build_imagefolder_dataloaders
from model import MobileNetV2
from trainer import ClassificationTrainer

TRAIN_DIR = ROOT / "data" / "images" / "train"
VAL_DIR = ROOT / "data" / "images" / "validation"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 20


def main():
    dataloaders, _, class_names = build_imagefolder_dataloaders(TRAIN_DIR, VAL_DIR, batch_size=BATCH_SIZE, image_size=224)
    model = MobileNetV2(num_classes=len(class_names))
    trainer = ClassificationTrainer(
        model=model,
        criterion=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        device=DEVICE,
        save_path=Path(__file__).with_name("best.pt"),
    )
    trainer.fit(dataloaders, num_epochs=EPOCHS)


if __name__ == "__main__":
    main()
