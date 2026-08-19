import math


class StepCosineAnnealingLR:
    """Decay the learning rate using a cosine annealing schedule at each STEP (not epoch).

    Supports optimizers with multiple parameter groups. `initial_lr` and `final_lr` can be:
      - float: applied uniformly to all param groups
      - list / tuple of floats: one value per param group
      - None for `initial_lr`: use each param group's current lr as its own starting lr

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        steps_init (int): Global step already consumed (resume).
        num_steps (int): Total number of steps in the training process.
        initial_lr (float | list | tuple | None): Starting lr(s). If None use current group lrs.
        final_lr (float | list | tuple): Target lr(s) at the end of schedule.
    """

    def __init__(self, optimizer, steps_init, num_steps, initial_lr, final_lr):
        self.optimizer = optimizer
        self.num_steps = max(1, num_steps)
        self.i_step = steps_init

        n_groups = len(self.optimizer.param_groups)

        # Normalize initial lrs
        if initial_lr is None:
            self.initial_lrs = [g["lr"] for g in self.optimizer.param_groups]
        elif isinstance(initial_lr, (list, tuple)):
            assert len(initial_lr) == n_groups, (
                "Length of initial_lr list must match number of param groups"
            )
            self.initial_lrs = list(map(float, initial_lr))
        else:  # scalar
            self.initial_lrs = [float(initial_lr)] * n_groups

        # Normalize final lrs
        if isinstance(final_lr, (list, tuple)):
            assert len(final_lr) == n_groups, (
                "Length of final_lr list must match number of param groups"
            )
            self.final_lrs = list(map(float, final_lr))
        else:
            self.final_lrs = [float(final_lr)] * n_groups

        # If resuming mid-way, advance param group lrs accordingly
        if self.i_step > 0:
            self._apply_step_lrs(self.i_step)

    def _apply_step_lrs(self, step_idx):
        """Compute and apply cosine-annealed lrs for a given step."""
        s = min(step_idx, self.num_steps)
        cosine_factor = (1 + math.cos(math.pi * s / self.num_steps)) / 2
        for pg, i_lr, f_lr in zip(
            self.optimizer.param_groups, self.initial_lrs, self.final_lrs, strict=False
        ):
            pg["lr"] = f_lr + (i_lr - f_lr) * cosine_factor

    def step(self):
        """Advance one step and update learning rates."""
        self.i_step += 1
        if self.i_step > self.num_steps:
            self._apply_step_lrs(self.num_steps)
            return
        self._apply_step_lrs(self.i_step)

    def get_lr_dict(self):
        """Get a dictionary of current learning rates."""
        return {
            f"lr_param_group_{i}": pg["lr"]
            for i, pg in enumerate(self.optimizer.param_groups)
        }

    def get_step(self):
        return self.i_step
