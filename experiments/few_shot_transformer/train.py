from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
from torch import optim

from dataloader import SetDataManager
from model import Conv4, FewShotTransformer
from trainer import MetaLearningTrainer

DATA_ROOT = ROOT / "data" / "miniimagenet"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_WAY = 5
K_SHOT = 1
N_QUERY = 15
N_EPISODE = 100
EPOCHS = 20


def main():
    manager = SetDataManager(image_size=84, n_way=N_WAY, k_shot=K_SHOT, n_query=N_QUERY, n_episode=N_EPISODE)
    train_loader = manager.get_data_loader(DATA_ROOT / "base.json", aug=True)
    val_loader = manager.get_data_loader(DATA_ROOT / "val.json", aug=False)
    model = FewShotTransformer(
        model_func=lambda: Conv4(dataset="miniImagenet", flatten=True),
        n_way=N_WAY,
        k_shot=K_SHOT,
        n_query=N_QUERY,
        variant="cosine",
    )
    trainer = MetaLearningTrainer(
        model=model,
        optimizer=optim.Adam(model.parameters(), lr=1e-3),
        device=DEVICE,
        save_path=Path(__file__).with_name("best.pt"),
    )
    trainer.fit(train_loader, val_loader=val_loader, epochs=EPOCHS)


if __name__ == "__main__":
    main()
