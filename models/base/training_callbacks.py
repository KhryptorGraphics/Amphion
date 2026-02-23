# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TrainerState:
    """Snapshot of trainer state passed to every callback hook.

    Attributes:
        epoch: Current epoch index (0-based).
        step: Current global step count.
        train_loss: Training loss for the most recent epoch or batch, if available.
        valid_loss: Validation loss for the most recent epoch, if available.
        checkpoint_path: Path to the checkpoint that was just saved, if applicable.
        extra: Arbitrary key/value pairs that a trainer or callback may populate.
    """

    epoch: int
    step: int
    train_loss: Optional[float] = None
    valid_loss: Optional[float] = None
    checkpoint_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class TrainingCallback:
    """Base class for training lifecycle callbacks.

    Subclass this and override the hook methods you care about.  All hooks are
    no-ops by default so subclasses only need to implement the events they need.

    Hook firing order during a training run::

        on_train_begin
        for each epoch:
            on_epoch_start
            for each batch:
                on_batch_start
                <forward / backward pass>
                on_batch_end          # fires once per gradient update step
            on_epoch_end
            [on_checkpoint_saved]     # fires when a checkpoint is written
        on_train_end
    """

    def on_train_begin(self, state: TrainerState) -> None:
        """Called once before the training loop starts.

        Args:
            state: Trainer state at the beginning of training.
        """

    def on_train_end(self, state: TrainerState) -> None:
        """Called once after the training loop finishes (or is interrupted).

        Args:
            state: Trainer state at the end of training.
        """

    def on_epoch_start(self, state: TrainerState) -> None:
        """Called at the beginning of each epoch, before any batches are processed.

        Args:
            state: Trainer state at the start of the epoch.
        """

    def on_epoch_end(self, state: TrainerState) -> None:
        """Called at the end of each epoch, after all batches and validation.

        Args:
            state: Trainer state at the end of the epoch.  ``train_loss`` and
                ``valid_loss`` are populated when available.
        """

    def on_batch_start(self, state: TrainerState) -> None:
        """Called before each gradient-accumulation step begins.

        Args:
            state: Trainer state before the batch.
        """

    def on_batch_end(self, state: TrainerState) -> None:
        """Called after each gradient update step completes.

        Fires once per *optimizer step*, not once per micro-batch, so the
        frequency matches ``self.step`` increments in the trainer.

        Args:
            state: Trainer state after the batch.  ``train_loss`` reflects the
                loss for the completed step when available.
        """

    def on_checkpoint_saved(self, state: TrainerState) -> None:
        """Called immediately after a checkpoint has been written to disk.

        Only fires on the main process.

        Args:
            state: Trainer state at checkpoint time.  ``checkpoint_path``
                contains the directory where the checkpoint was saved.
        """
