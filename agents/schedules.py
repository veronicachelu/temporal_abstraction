import tensorflow as tf

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p
        # self.steps = 0

    def value(self, t):
        """See Schedule.value"""
        # t = self.steps
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        # self.steps += 1

        return self.initial_p + fraction * (self.final_p - self.initial_p)


class TFLinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.
        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = tf.constant(schedule_timesteps, dtype=tf.float32)
        self.final_p = tf.constant(final_p, dtype=tf.float32)
        self.initial_p = tf.constant(initial_p, dtype=tf.float32)
        # self.fraction = tf.Variable(0.0, dtype=tf.float32)
        # self.steps = 0

    def value(self, t):
        """See Schedule.value"""
        # t = self.steps
        fraction = tf.minimum(tf.cast(t, dtype=tf.float32) / self.schedule_timesteps,
                              tf.constant(1.0, dtype=tf.float32))
        # self.steps += 1

        return self.initial_p + fraction * (self.final_p - self.initial_p)