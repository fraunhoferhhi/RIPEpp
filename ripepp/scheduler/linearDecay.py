class LinearDecay:
    """Linearly decaying function.

    Args:
        start_val (float): Initial value.
        end_val (float): Final value.
        steps_total (int): Total number of steps in the training process.
    """

    def __init__(self, start_val, end_val, steps_total):
        self.start_val = start_val
        self.end_val = end_val
        self.steps_total = steps_total

    def __call__(self, step):
        if step >= self.steps_total:
            return self.end_val
        return (
            self.start_val + (self.end_val - self.start_val) * step / self.steps_total
        )
