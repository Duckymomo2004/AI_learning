from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import optim
from torch.utils.data import DataLoader

from dataloader import MiniImagenetEpisodeDataset
from model import RelationNet
from trainer import RelationNetTrainer

DATA_ROOT = ROOT / "data" / "miniimagenet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WAY = 5
K_SHOT = 1
K_QUERY = 15
IMAGE_SIZE = 84
TRAIN_EPISODES = 3000
VAL_EPISODES = 600
EPOCHS = 20


def main():
    train_set = MiniImagenetEpisodeDataset(DATA_ROOT, "base", N_WAY, K_SHOT, K_QUERY, TRAIN_EPISODES, IMAGE_SIZE)
    val_set = MiniImagenetEpisodeDataset(DATA_ROOT, "val", N_WAY, K_SHOT, K_QUERY, VAL_EPISODES, IMAGE_SIZE)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    model = RelationNet(n_way=N_WAY, k_shot=K_SHOT, image_size=IMAGE_SIZE)
    trainer = RelationNetTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        device=DEVICE,
        save_path=Path(__file__).with_name("best.pt"),
    )
    trainer.fit(train_loader, val_loader=val_loader, epochs=EPOCHS)


if __name__ == "__main__":
    main()
