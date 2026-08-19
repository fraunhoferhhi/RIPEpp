class Warmup2Linear:
    """Linear warmup and then linear decay.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        warmup_steps (int): Number of steps for linear warmup.
        num_steps (int): Total number of steps in the training process.
        initial_lr (float): Initial learning rate (reached after warm up).
        final_lr (float): Final learning rate.
    """

    def __init__(
        self, optimizer, warmup_steps, num_steps, initial_lr, final_lr, steps_init
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.num_steps = num_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.i_step = steps_init

        assert warmup_steps < num_steps, (
            "Warmup steps must be smaller than total steps."
        )

        self.warmup_factor = initial_lr / warmup_steps
        self.decay_factor = (final_lr - initial_lr) / (num_steps - warmup_steps)

    def step(self):
        """Decay the learning rate by decay_factor."""
        self.i_step += 1

        if self.i_step > self.num_steps:
            return

        if self.i_step < self.warmup_steps:
            lr = self.i_step * self.warmup_factor
        else:
            lr = self.initial_lr + (self.i_step - self.warmup_steps) * self.decay_factor

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_last_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_step(self):
        return self.i_step
