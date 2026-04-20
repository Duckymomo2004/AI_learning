from __future__ import annotations

import copy
from pathlib import Path

import torch


class MetaLearningTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device | str | None = None,
        save_path: str | Path | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.save_path = Path(save_path) if save_path is not None else None
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_accuracy = 0.0
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_acc": [],
        }

    def _run_epoch(self, dataloader, training: bool, max_batches: int | None = None) -> tuple[float, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_acc = 0.0
        total_batches = 0

        for batch_index, (episodes, _) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            episodes = episodes.to(self.device)
            if getattr(self.model, "change_way", False):
                self.model.n_way = episodes.size(0)

            if training:
                acc, loss = self.model.set_forward_loss(episodes)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                with torch.no_grad():
                    correct, count = self.model.correct(episodes)
                    acc = correct / count
                    scores = self.model.set_forward(episodes)
                    targets = self.model.episode_targets(self.device)
                    loss = torch.nn.functional.cross_entropy(scores, targets)

            total_loss += float(loss.item())
            total_acc += float(acc)
            total_batches += 1

        return total_loss / total_batches, total_acc / total_batches

    def fit(
        self,
        train_loader,
        val_loader=None,
        epochs: int = 1,
        max_batches_per_epoch: int | None = None,
    ) -> dict[str, list[float]]:
        for _ in range(epochs):
            train_loss, train_acc = self._run_epoch(
                train_loader,
                training=True,
                max_batches=max_batches_per_epoch,
            )
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if val_loader is not None:
                val_loss, val_acc = self._run_epoch(
                    val_loader,
                    training=False,
                    max_batches=max_batches_per_epoch,
                )
                del val_loss
                self.history["val_acc"].append(val_acc)
                if val_acc >= self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.best_state = copy.deepcopy(self.model.state_dict())
                    if self.save_path is not None:
                        self.save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), self.save_path)

        self.model.load_state_dict(self.best_state)
        return self.history
