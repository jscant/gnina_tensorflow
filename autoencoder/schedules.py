"""
Created on Mon Sep 07 17:15:09 2020

@author: scantleb
@brief: Learning schedule class definitions

Optimisers can be given a learning rate which changes depending on the stage of
training. The classes here are all derived from the keras LearningRateSchedule
class; see Keras documentation for more information.
"""

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers.schedules import LearningRateSchedule


class OneCycle(LearningRateSchedule):
    """1cycle learning rate schedule.

    The 1cycle method is outlined in https://arxiv.org/pdf/1803.09820.pdf.
    In the first phase, the learning rate is increaced linearly from the
    minimum to the maximum. In the second phase, the learning rate is
    decreased in a cosine fashion from the maximum back down the the minimum.

    The minimum and maximum learning rates are hyperparameters that should be
    determined, for example using the LR-Range test (found here in
    utilities/lr_range_test.py).
    """

    def __init__(self, min_lr, max_lr, iterations):
        """Instantiate 1cycle scheduler.

        Arguments:
            min_lr: the minimum learning rate
            max_lr: the maximum learning rate
            iterations: total total batches to learn from in training
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.iterations = iterations
        self.peak_iter = np.floor(0.45 * iterations)

    def __call__(self, step):
        """Overloaded method; see base class (LearningRateSchedule).

        Returns learning rate given the training step.

        Arguments:
            step: current training iteration
        """
        step = K.get_value(step)
        phase = float(step <= self.peak_iter)
        rng = self.max_lr - self.min_lr
        progress = (step - self.peak_iter) / (self.iterations - self.peak_iter)
        phase_1_lr = self.min_lr + (step / self.peak_iter) * rng
        phase_2_lr = self.min_lr + (
                rng * 0.5 * (np.cos(np.pi * progress) + 1))
        return (phase * phase_1_lr) + ((1 - phase) * phase_2_lr)

    def get_config(self):
        """Overloaded method; see base class (LearningRateSchedule)."""
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'iterations': self.iterations,
            'peak_iter': self.peak_iter,
        }


class WarmRestartCosine(LearningRateSchedule):
    """Cosine decay scheduler with warm restarts.

    The learning rate starts at a maximum value, and decays following a cosine
    decay (a graph which looks like cos(x), 0 <= x < pi) to the minimum value.
    It then jumps back up the the maximum, and decays again, in a process that
    is repeated every T iterations. The hyperparameters for this are therefore
    the period, the maximum and minimum learning rates, and the rate at which
    the maximum learning rate decays every cycle.
    """

    def __init__(self, min_lr, max_lr, period, beta=1.0):
        """Instantiate warm resetart scheduler.

        Arguments:
            min_lr: the minimum learning rate
            max_lr: the maximum learning rate
            period: period over which to repeat the learning rate pattern
            beta: sets rate of decay of max_lr every cycle (restart). If set
                to 1, max_lr remains the same between cycles; values less than
                1 cause exponential decay in the maximum learning rate
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.period = period
        self.beta = beta

    def __call__(self, step):
        """Overloaded method; see base class (LearningRateSchedule).

        Returns learning rate given the training step.

        Arguments:
            step: current training iteration
        """
        step = K.get_value(step)
        cycle = step // self.period
        decayed_max_lr = max(
            self.min_lr, self.max_lr * (self.beta ** cycle))
        rng = decayed_max_lr - self.min_lr
        progress = (step % self.period) / self.period
        return self.min_lr + 0.5 * rng * (np.cos(np.pi * progress) + 1)

    def get_config(self):
        """Overloaded method; see base class (LearningRateSchedule)."""
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'period': self.period,
            'beta': self.beta,
        }


class StepWiseDecay(LearningRateSchedule):
    """Step-wise decay rate scheduler.

    Every t iterations, the learning rate is reduced by multiplying it by
    0 < beta < 1. This gives a learning rate which follows a step-wise
    approximation to exponential decay. The learning rate will never fall
    below the minimum learning rate specified.
    """

    def __init__(self, min_lr, max_lr, t=2500, beta=0.5):
        """Instantiate step-wise decay scheduler.

        Arguments:
            min_lr: the minimum learning rate
            max_lr: the maximum learning rate
            t: number of iterations at each learning rate
            beta: multiplier for learning rate at the end of each
                decay_time iterations
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.t = t
        self.beta = beta

    def __call__(self, step):
        """Overloaded method; see base class (LearningRateSchedule).

        Returns learning rate given the training step.

        Arguments:
            step: current training iteration
        """
        step = K.get_value(step)
        stage = step // self.t
        prospective_lr = self.max_lr * (self.beta ** stage)
        return max(self.min_lr, prospective_lr)

    def get_config(self):
        """Overloaded method; see base class (LearningRateSchedule)."""
        return {
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            't': self.t,
            'beta': self.beta
        }


class ConstantLearningRateSchedule(LearningRateSchedule):
    """Learning rate scheduler which does not change learning rate."""

    def __init__(self, learning_rate):
        """Instantiate constant learning rate schedule

        Arguments:
            learning_rate/lr: constant learning rate
        """
        self.lr = learning_rate

    def __call__(self, step):
        """Overloaded method; see base class (LearningRateSchedule).

        Returns learning rate given the training step.

        Arguments:
            step: current training iteration
        """
        return self.lr

    def get_config(self):
        """Overloaded method; see base class (LearningRateSchedule)."""
        return {
            'lr': self.lr
        }
