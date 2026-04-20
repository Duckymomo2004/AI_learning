from __future__ import annotations

import copy
from pathlib import Path

import torch


class RelationNetTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: torch.device | str | None = None,
        save_path: str | Path | None = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = Path(save_path) if save_path is not None else None
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_accuracy = 0.0
        self.best_loss = float("inf")
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

    def _run_epoch(self, dataloader, training: bool, max_batches: int | None = None) -> tuple[float, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        total_batches = 0

        for batch_index, (support_x, support_y, query_x, query_y) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)

            with torch.set_grad_enabled(training):
                loss, _ = self.model.episode_loss(support_x, support_y, query_x, query_y)
                predictions, correct = self.model.predict(support_x, support_y, query_x, query_y)
                del predictions

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            total_loss += loss.item()
            total_correct += int(correct.item())
            total_items += int(query_y.numel())
            total_batches += 1

        epoch_loss = total_loss / total_batches
        epoch_acc = total_correct / total_items
        return epoch_loss, epoch_acc

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

            if self.scheduler is not None:
                self.scheduler.step()

            if val_loader is not None:
                with torch.no_grad():
                    val_loss, val_acc = self._run_epoch(
                        val_loader,
                        training=False,
                        max_batches=max_batches_per_epoch,
                    )
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_acc >= self.best_accuracy:
                    self.best_accuracy = val_acc
                    self.best_state = copy.deepcopy(self.model.state_dict())
                if val_loss <= self.best_loss:
                    self.best_loss = val_loss
                    if self.save_path is not None:
                        self.save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), self.save_path)

        self.model.load_state_dict(self.best_state)
        return self.history
