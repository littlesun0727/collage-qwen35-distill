"""SWIFT 4.1 external callback: epoch checkpoints in addition to step saves.

Loaded with --external_plugins and --callbacks collage_save_each_epoch.
Only request a save through TrainerControl; Trainer owns distributed I/O and
the full optimizer/scheduler/RNG/trainer state needed for resuming training.
"""
from swift.callbacks import TrainerCallback, callbacks_map


class SaveEachEpochCallback(TrainerCallback):
    def __init__(self, args, trainer):
        super().__init__(args, trainer)
        self.last_saved_step = -1

    def on_train_begin(self, args, state, control, **kwargs):
        # A resumed global_step already belongs to the checkpoint we loaded.
        self.last_saved_step = state.global_step
        return control

    def on_save(self, args, state, control, **kwargs):
        self.last_saved_step = state.global_step
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        if state.global_step > 0 and state.global_step != self.last_saved_step:
            # Set the flag on every rank. Restricting it to rank 0 can deadlock
            # distributed checkpointing when Trainer needs collective operations.
            control.should_save = True
            if state.is_world_process_zero:
                print(f'[checkpoint] epoch={state.epoch} ended; saving checkpoint-{state.global_step}', flush=True)
        return control


callbacks_map['collage_save_each_epoch'] = SaveEachEpochCallback
