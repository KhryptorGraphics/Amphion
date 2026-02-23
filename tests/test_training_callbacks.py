# Copyright (c) 2023 Amphion.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for TrainingCallback, TrainerState, and BaseTrainer callback integration."""

import logging
import unittest
from unittest.mock import MagicMock, call, patch

from models.base.training_callbacks import TrainerState, TrainingCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _RecordingCallback(TrainingCallback):
    """Callback that records every event it receives."""

    def __init__(self):
        self.events = []

    def on_train_begin(self, state: TrainerState) -> None:
        self.events.append(("on_train_begin", state))

    def on_train_end(self, state: TrainerState) -> None:
        self.events.append(("on_train_end", state))

    def on_epoch_start(self, state: TrainerState) -> None:
        self.events.append(("on_epoch_start", state))

    def on_epoch_end(self, state: TrainerState) -> None:
        self.events.append(("on_epoch_end", state))

    def on_batch_start(self, state: TrainerState) -> None:
        self.events.append(("on_batch_start", state))

    def on_batch_end(self, state: TrainerState) -> None:
        self.events.append(("on_batch_end", state))

    def on_checkpoint_saved(self, state: TrainerState) -> None:
        self.events.append(("on_checkpoint_saved", state))


class _RaisingCallback(TrainingCallback):
    """Callback that always raises an exception on every hook."""

    def on_train_begin(self, state: TrainerState) -> None:
        raise RuntimeError("boom")

    def on_train_end(self, state: TrainerState) -> None:
        raise RuntimeError("boom")

    def on_epoch_start(self, state: TrainerState) -> None:
        raise RuntimeError("boom")

    def on_epoch_end(self, state: TrainerState) -> None:
        raise RuntimeError("boom")

    def on_batch_start(self, state: TrainerState) -> None:
        raise RuntimeError("boom")

    def on_batch_end(self, state: TrainerState) -> None:
        raise RuntimeError("boom")

    def on_checkpoint_saved(self, state: TrainerState) -> None:
        raise RuntimeError("boom")


def _make_fire_callbacks_host():
    """Return a minimal object exposing ``_fire_callbacks`` and ``register_callback``
    without requiring a full ``BaseTrainer`` initialisation (which needs Accelerate,
    GPUs, and a config object).
    """
    logger = logging.getLogger("test_callbacks")

    class _MinimalHost:
        def __init__(self):
            self.callbacks: list = []
            self.logger = logger

        def register_callback(self, callback: TrainingCallback) -> None:
            if callback not in self.callbacks:
                self.callbacks.append(callback)

        def _fire_callbacks(self, event: str, state: TrainerState) -> None:
            for callback in self.callbacks:
                try:
                    getattr(callback, event)(state)
                except Exception as exc:
                    self.logger.warning(
                        f"Callback {callback.__class__.__name__}.{event} raised an "
                        f"exception and was skipped: {exc}"
                    )

    return _MinimalHost()


# ---------------------------------------------------------------------------
# TrainerState tests
# ---------------------------------------------------------------------------


class TestTrainerState(unittest.TestCase):
    """Tests for the TrainerState dataclass."""

    def test_required_fields(self):
        state = TrainerState(epoch=0, step=0)
        self.assertEqual(state.epoch, 0)
        self.assertEqual(state.step, 0)

    def test_optional_fields_default_to_none(self):
        state = TrainerState(epoch=1, step=10)
        self.assertIsNone(state.train_loss)
        self.assertIsNone(state.valid_loss)
        self.assertIsNone(state.checkpoint_path)

    def test_extra_defaults_to_empty_dict(self):
        state = TrainerState(epoch=0, step=0)
        self.assertEqual(state.extra, {})

    def test_extra_is_independent_per_instance(self):
        """Each instance should get its own ``extra`` dict (not a shared default)."""
        s1 = TrainerState(epoch=0, step=0)
        s2 = TrainerState(epoch=1, step=1)
        s1.extra["key"] = "value"
        self.assertNotIn("key", s2.extra)

    def test_set_all_fields(self):
        state = TrainerState(
            epoch=3,
            step=42,
            train_loss=0.5,
            valid_loss=0.6,
            checkpoint_path="/tmp/ckpt",
            extra={"custom": True},
        )
        self.assertEqual(state.epoch, 3)
        self.assertEqual(state.step, 42)
        self.assertAlmostEqual(state.train_loss, 0.5)
        self.assertAlmostEqual(state.valid_loss, 0.6)
        self.assertEqual(state.checkpoint_path, "/tmp/ckpt")
        self.assertEqual(state.extra, {"custom": True})

    def test_mutable_extra(self):
        state = TrainerState(epoch=0, step=0)
        state.extra["foo"] = "bar"
        self.assertEqual(state.extra["foo"], "bar")


# ---------------------------------------------------------------------------
# TrainingCallback tests
# ---------------------------------------------------------------------------


class TestTrainingCallbackBase(unittest.TestCase):
    """Tests for the base TrainingCallback class."""

    def setUp(self):
        self.callback = TrainingCallback()
        self.state = TrainerState(epoch=0, step=0)

    def test_on_train_begin_is_noop(self):
        result = self.callback.on_train_begin(self.state)
        self.assertIsNone(result)

    def test_on_train_end_is_noop(self):
        result = self.callback.on_train_end(self.state)
        self.assertIsNone(result)

    def test_on_epoch_start_is_noop(self):
        result = self.callback.on_epoch_start(self.state)
        self.assertIsNone(result)

    def test_on_epoch_end_is_noop(self):
        result = self.callback.on_epoch_end(self.state)
        self.assertIsNone(result)

    def test_on_batch_start_is_noop(self):
        result = self.callback.on_batch_start(self.state)
        self.assertIsNone(result)

    def test_on_batch_end_is_noop(self):
        result = self.callback.on_batch_end(self.state)
        self.assertIsNone(result)

    def test_on_checkpoint_saved_is_noop(self):
        result = self.callback.on_checkpoint_saved(self.state)
        self.assertIsNone(result)

    def test_subclass_can_override_hooks(self):
        cb = _RecordingCallback()
        state = TrainerState(epoch=2, step=5)
        cb.on_train_begin(state)
        self.assertEqual(cb.events, [("on_train_begin", state)])

    def test_all_hooks_callable(self):
        """All expected hook names exist and are callable."""
        expected_hooks = [
            "on_train_begin",
            "on_train_end",
            "on_epoch_start",
            "on_epoch_end",
            "on_batch_start",
            "on_batch_end",
            "on_checkpoint_saved",
        ]
        for hook in expected_hooks:
            with self.subTest(hook=hook):
                self.assertTrue(callable(getattr(self.callback, hook)))


# ---------------------------------------------------------------------------
# _fire_callbacks / register_callback tests
# ---------------------------------------------------------------------------


class TestFireCallbacks(unittest.TestCase):
    """Tests for ``_fire_callbacks`` and ``register_callback`` behaviour."""

    def setUp(self):
        self.host = _make_fire_callbacks_host()
        self.state = TrainerState(epoch=0, step=0)

    # --- register_callback ---------------------------------------------------

    def test_register_callback_adds_to_list(self):
        cb = _RecordingCallback()
        self.host.register_callback(cb)
        self.assertIn(cb, self.host.callbacks)

    def test_register_same_callback_twice_is_idempotent(self):
        cb = _RecordingCallback()
        self.host.register_callback(cb)
        self.host.register_callback(cb)
        self.assertEqual(self.host.callbacks.count(cb), 1)

    def test_register_multiple_distinct_callbacks(self):
        cb1 = _RecordingCallback()
        cb2 = _RecordingCallback()
        self.host.register_callback(cb1)
        self.host.register_callback(cb2)
        self.assertEqual(len(self.host.callbacks), 2)

    # --- _fire_callbacks – basic dispatch ------------------------------------

    def test_fire_calls_correct_hook(self):
        cb = _RecordingCallback()
        self.host.register_callback(cb)
        self.host._fire_callbacks("on_train_begin", self.state)
        self.assertEqual(len(cb.events), 1)
        self.assertEqual(cb.events[0][0], "on_train_begin")

    def test_fire_passes_state_to_hook(self):
        cb = _RecordingCallback()
        self.host.register_callback(cb)
        state = TrainerState(epoch=3, step=7, train_loss=0.12)
        self.host._fire_callbacks("on_epoch_end", state)
        received_state = cb.events[0][1]
        self.assertIs(received_state, state)

    def test_fire_calls_all_registered_callbacks(self):
        cb1 = _RecordingCallback()
        cb2 = _RecordingCallback()
        self.host.register_callback(cb1)
        self.host.register_callback(cb2)
        self.host._fire_callbacks("on_batch_end", self.state)
        self.assertEqual(len(cb1.events), 1)
        self.assertEqual(len(cb2.events), 1)

    def test_fire_preserves_registration_order(self):
        """Callbacks should be invoked in the order they were registered."""
        order = []

        class _OrderedCallback(TrainingCallback):
            def __init__(self, tag):
                self._tag = tag

            def on_train_begin(self, state):
                order.append(self._tag)

        cb_a = _OrderedCallback("A")
        cb_b = _OrderedCallback("B")
        cb_c = _OrderedCallback("C")
        self.host.register_callback(cb_a)
        self.host.register_callback(cb_b)
        self.host.register_callback(cb_c)
        self.host._fire_callbacks("on_train_begin", self.state)
        self.assertEqual(order, ["A", "B", "C"])

    def test_fire_no_callbacks_does_not_raise(self):
        """Firing when no callbacks are registered should succeed silently."""
        self.host._fire_callbacks("on_epoch_start", self.state)

    # --- exception isolation -------------------------------------------------

    def test_raising_callback_does_not_abort_other_callbacks(self):
        """A callback that raises must not prevent subsequent callbacks from firing."""
        raising_cb = _RaisingCallback()
        recording_cb = _RecordingCallback()
        self.host.register_callback(raising_cb)
        self.host.register_callback(recording_cb)
        self.host._fire_callbacks("on_train_begin", self.state)
        # recording_cb should still have received the event
        self.assertEqual(len(recording_cb.events), 1)
        self.assertEqual(recording_cb.events[0][0], "on_train_begin")

    def test_raising_callback_logs_warning(self):
        raising_cb = _RaisingCallback()
        self.host.register_callback(raising_cb)
        with self.assertLogs("test_callbacks", level="WARNING") as log_ctx:
            self.host._fire_callbacks("on_epoch_end", self.state)
        self.assertTrue(
            any("_RaisingCallback" in msg for msg in log_ctx.output),
            "Expected a warning mentioning the callback class name",
        )

    def test_multiple_raising_callbacks_all_logged(self):
        """Every raising callback should produce a warning."""
        for _ in range(3):
            self.host.register_callback(_RaisingCallback())
        with self.assertLogs("test_callbacks", level="WARNING") as log_ctx:
            self.host._fire_callbacks("on_batch_start", self.state)
        self.assertEqual(len(log_ctx.output), 3)

    # --- all events dispatched -----------------------------------------------

    def test_all_events_dispatched_correctly(self):
        """Every supported event name is forwarded to the correct hook method."""
        events = [
            "on_train_begin",
            "on_train_end",
            "on_epoch_start",
            "on_epoch_end",
            "on_batch_start",
            "on_batch_end",
            "on_checkpoint_saved",
        ]
        cb = _RecordingCallback()
        self.host.register_callback(cb)
        for event in events:
            self.host._fire_callbacks(event, self.state)

        fired = [e[0] for e in cb.events]
        self.assertEqual(fired, events)

    def test_checkpoint_saved_state_carries_path(self):
        cb = _RecordingCallback()
        self.host.register_callback(cb)
        state = TrainerState(
            epoch=5,
            step=100,
            train_loss=0.3,
            checkpoint_path="/ckpts/epoch-0005",
        )
        self.host._fire_callbacks("on_checkpoint_saved", state)
        received = cb.events[0][1]
        self.assertEqual(received.checkpoint_path, "/ckpts/epoch-0005")


# ---------------------------------------------------------------------------
# Integration: mock-based TrainingCallback with MagicMock
# ---------------------------------------------------------------------------


class TestCallbackWithMagicMock(unittest.TestCase):
    """Verify _fire_callbacks using MagicMock-based callbacks."""

    def setUp(self):
        self.host = _make_fire_callbacks_host()
        self.state = TrainerState(epoch=1, step=50, train_loss=0.25)

    def test_mock_callback_on_train_begin_called_once(self):
        mock_cb = MagicMock(spec=TrainingCallback)
        self.host.register_callback(mock_cb)
        self.host._fire_callbacks("on_train_begin", self.state)
        mock_cb.on_train_begin.assert_called_once_with(self.state)

    def test_mock_callback_on_epoch_end_receives_state(self):
        mock_cb = MagicMock(spec=TrainingCallback)
        self.host.register_callback(mock_cb)
        self.host._fire_callbacks("on_epoch_end", self.state)
        mock_cb.on_epoch_end.assert_called_once_with(self.state)

    def test_mock_callback_batch_end_called_per_fire(self):
        mock_cb = MagicMock(spec=TrainingCallback)
        self.host.register_callback(mock_cb)
        for _ in range(5):
            self.host._fire_callbacks("on_batch_end", self.state)
        self.assertEqual(mock_cb.on_batch_end.call_count, 5)

    def test_two_mock_callbacks_both_invoked(self):
        cb1 = MagicMock(spec=TrainingCallback)
        cb2 = MagicMock(spec=TrainingCallback)
        self.host.register_callback(cb1)
        self.host.register_callback(cb2)
        self.host._fire_callbacks("on_checkpoint_saved", self.state)
        cb1.on_checkpoint_saved.assert_called_once_with(self.state)
        cb2.on_checkpoint_saved.assert_called_once_with(self.state)


if __name__ == "__main__":
    unittest.main()
