import math


class StepCosineAnnealingWarmRestartsLR:
    """Decay the learning rate using a cosine annealing schedule with warm restarts at each STEP
    (not epoch).

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        num_steps (int): Total number of steps in the training process.
        initial_lr (float): Initial learning rate.
        final_lr (float): Final learning rate.
        T_0 (int): Number of steps for the first restart.
        T_mult (int): A factor increases T_i after a restart. Default: 1.
    """

    def __init__(
        self, optimizer, steps_init, num_steps, initial_lr, final_lr, T_0, T_mult=1
    ):
        self.optimizer = optimizer
        self.num_steps = num_steps
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.i_step = steps_init
        self.T_0 = T_0
        self.T_mult = T_mult
        self.T_i = T_0
        self.cycle = 0

    def step(self):
        """Decay the learning rate using a cosine annealing schedule with warm restarts."""
        self.i_step += 1

        if self.i_step > self.num_steps:
            return

        if self.i_step >= self.T_i:
            self.cycle += 1
            self.T_i = self.T_0 * (self.T_mult**self.cycle)

        lr = (
            self.final_lr
            + (self.initial_lr - self.final_lr)
            * (1 + math.cos(math.pi * (self.i_step % self.T_i) / self.T_i))
            / 2
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_last_lr(self):
        return self.optimizer.param_groups[0]["lr"]

    def get_step(self):
        return self.i_step
