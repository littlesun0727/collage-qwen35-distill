"""CPU callback lifecycle tests; SWIFT registration needs no installed model."""
import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


def load_callback():
    callbacks = ModuleType('swift.callbacks')

    class SwiftCallback:
        def __init__(self, args, trainer):
            self.args, self.trainer = args, trainer

    callbacks.TrainerCallback = SwiftCallback
    callbacks.callbacks_map = {}
    spec = importlib.util.spec_from_file_location('epoch_checkpoint_test', Path(__file__).with_name('epoch_checkpoint.py'))
    module = importlib.util.module_from_spec(spec)
    with patch.dict('sys.modules', {'swift.callbacks': callbacks}):
        spec.loader.exec_module(module)
    return module.SaveEachEpochCallback, callbacks.callbacks_map


class EpochCheckpointTests(unittest.TestCase):
    def setUp(self):
        cls, registry = load_callback()
        self.assertIs(registry['collage_save_each_epoch'], cls)
        self.callback = cls(SimpleNamespace(), None)
        self.state = SimpleNamespace(global_step=0, epoch=0., is_world_process_zero=False)
        self.control = SimpleNamespace(should_save=False, should_evaluate=False, should_training_stop=False)
        self.callback.on_train_begin(None, self.state, self.control)

    def simulate(self, first, last, steps_per_epoch, save_steps):
        saved = []
        for step in range(first, last + 1):
            self.state.global_step = step
            self.state.epoch = step / steps_per_epoch
            self.control.should_save = step % save_steps == 0
            if self.control.should_save:
                saved.append(step)
                self.callback.on_save(None, self.state, self.control)
                self.control.should_save = False
            if step % steps_per_epoch == 0 or step == last:
                self.callback.on_epoch_end(None, self.state, self.control)
                if self.control.should_save:
                    saved.append(step)
                    self.callback.on_save(None, self.state, self.control)
                    self.control.should_save = False
        return saved

    def test_steps_plus_epochs_and_coincident_save_once(self):
        self.assertEqual(self.simulate(1, 15, steps_per_epoch=5, save_steps=3),
                         [3, 5, 6, 9, 10, 12, 15])

    def test_resume_mid_epoch_and_at_boundary(self):
        for resume_step, expected in ((7, [9, 10, 12, 15]), (10, [12, 15])):
            with self.subTest(resume=resume_step):
                self.state.global_step = resume_step
                self.callback.on_train_begin(None, self.state, self.control)
                self.callback.on_epoch_end(None, self.state, self.control)
                self.assertFalse(self.control.should_save)
                self.assertEqual(self.simulate(resume_step + 1, 15, 5, 3), expected)

    def test_worker_rank_also_requests_save_without_evaluation(self):
        self.state.global_step, self.state.epoch = 5, 1.
        self.callback.on_epoch_end(None, self.state, self.control)
        self.assertTrue(self.control.should_save)
        self.assertFalse(self.control.should_evaluate)

    def test_shortened_final_epoch_keeps_checkpoint(self):
        self.assertEqual(self.simulate(1, 7, steps_per_epoch=5, save_steps=100), [5, 7])

    def test_no_empty_epoch_checkpoint_and_other_flags_preserved(self):
        self.callback.on_epoch_end(None, self.state, self.control)
        self.assertFalse(self.control.should_save)
        self.control.should_evaluate = True
        self.control.should_training_stop = True
        self.state.global_step, self.state.epoch = 5, 1.
        self.callback.on_epoch_end(None, self.state, self.control)
        self.assertTrue(self.control.should_evaluate)
        self.assertTrue(self.control.should_training_stop)


if __name__ == '__main__':
    unittest.main()
