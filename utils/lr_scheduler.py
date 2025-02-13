import math

import tensorflow as tf
from torch.optim.lr_scheduler import LambdaLR


# TensorFlow Learning Rate Schedule
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr, peak_lr, min_lr, warmup_steps, total_steps):
        super(WarmupCosineDecay, self).__init__()
        self.initial_lr = initial_lr
        self.decay_steps = total_steps - warmup_steps
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps

        if self.decay_steps <= 0:
            raise ValueError(
                "Argument `total_steps` must be > warmup_steps. "
                f"Received: total_steps={total_steps}; warmup_steps:{warmup_steps}"
            )

    def warmup_function(self, step, warmup_steps, warmup_target, initial_lr):
        completed_fraction = step / warmup_steps
        total_warmup = warmup_target - initial_lr
        lr = total_warmup * completed_fraction + initial_lr
        return lr

    def decay_function(self, step, decay_steps):
        progress = step / decay_steps
        pi = math.pi
        cosine_decayed = 0.5 * (1 + tf.math.cos(pi * progress))
        lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decayed
        return lr

    def __call__(self, step):
        initial_lr = tf.convert_to_tensor(self.initial_lr)
        dtype = initial_lr.dtype
        decay_steps = tf.cast(self.decay_steps, dtype)
        global_step = tf.cast(step, dtype)
        warmup_target = tf.cast(self.peak_lr, dtype)
        warmup_steps = tf.cast(self.warmup_steps, dtype)
        global_step = tf.minimum(global_step, decay_steps + warmup_steps)
        # Warmup phase
        lr = tf.cond(
            global_step < warmup_steps,
            lambda: self.warmup_function(
                global_step,
                warmup_steps,
                warmup_target,
                initial_lr,
            ),
            lambda: self.decay_function(global_step - warmup_steps, decay_steps),
        )
        return lr


# PyTorch Learning Rate Scheduler using LambdaLR
def get_torch_lr_scheduler(
    optimizer, initial_lr, peak_lr, min_lr, warmup_steps, total_steps
):
    """
    Returns a torch.optim.lr_scheduler.LambdaLR that implements a warmup cosine decay schedule.
    """

    def lr_lambda(current_step):
        lr_increment = (peak_lr - initial_lr) / warmup_steps

        if current_step < warmup_steps:
            return (initial_lr + current_step * lr_increment) / peak_lr
        else:
            progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = min_lr + (peak_lr - min_lr) * cosine_decay
            return lr / peak_lr

    return LambdaLR(optimizer, lr_lambda)
