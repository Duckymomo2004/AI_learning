from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


class ClassificationTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | torch.optim.lr_scheduler.OneCycleLR | None = None,
        device: torch.device | str | None = None,
        save_path: str | Path | None = None,
        scheduler_step_per_batch: bool = False,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.save_path = Path(save_path) if save_path is not None else None
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.best_accuracy = 0.0
        self.best_loss = float("inf")
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": [],
        }

    def _run_epoch(self, dataloader, training: bool, max_batches: int | None = None) -> tuple[float, float]:
        self.model.train(mode=training)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_index, (inputs, labels) in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.set_grad_enabled(training):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if training:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    if self.scheduler is not None and self.scheduler_step_per_batch:
                        self.scheduler.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(dim=1) == labels).sum().item()
            total_samples += inputs.size(0)

        epoch_loss = total_loss / total_samples
        epoch_acc = total_correct / total_samples
        return epoch_loss, epoch_acc

    def fit(
        self,
        dataloaders: dict[str, torch.utils.data.DataLoader],
        num_epochs: int,
        eval_phase: str = "test",
        max_batches_per_epoch: int | None = None,
    ) -> dict[str, list[float]]:
        for _ in range(num_epochs):
            train_loss, train_acc = self._run_epoch(
                dataloaders["train"],
                training=True,
                max_batches=max_batches_per_epoch,
            )
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)

            if self.scheduler is not None and not self.scheduler_step_per_batch:
                self.scheduler.step()

            if eval_phase in dataloaders:
                eval_loss, eval_acc = self._run_epoch(
                    dataloaders[eval_phase],
                    training=False,
                    max_batches=max_batches_per_epoch,
                )
                self.history["test_loss"].append(eval_loss)
                self.history["test_acc"].append(eval_acc)

                if eval_acc >= self.best_accuracy:
                    self.best_accuracy = eval_acc
                    self.best_state = copy.deepcopy(self.model.state_dict())
                if eval_loss <= self.best_loss:
                    self.best_loss = eval_loss
                    if self.save_path is not None:
                        self.save_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(self.model.state_dict(), self.save_path)

        self.model.load_state_dict(self.best_state)
        return self.history

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        max_batches: int | None = None,
    ) -> dict[str, float]:
        loss, accuracy = self._run_epoch(dataloader, training=False, max_batches=max_batches)
        return {"loss": loss, "accuracy": accuracy}
