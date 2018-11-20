import numpy as np
import tensorflow as tf
import tensorflow.contrib.rnn as rnn

class SingleStepLSTM(object):

    def __init__(self,x,size,step_size):
        lstm = rnn.BasicLSTMCell(size, state_is_tuple=True)

        c_init = np.zeros((1, lstm.state_size.c), np.float32)
        h_init = np.zeros((1, lstm.state_size.h), np.float32)
        self.state_init = [c_init, h_init]

        c_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.c],
                name='c_in')
        h_in = tf.placeholder(tf.float32,
                shape=[1, lstm.state_size.h],
                name='h_in')
        self.state_in = [c_in, h_in]

        state_in = rnn.LSTMStateTuple(c_in, h_in)

        lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
            lstm, x, initial_state=state_in, sequence_length=step_size,
            time_major=False)
        lstm_outputs = tf.reshape(lstm_outputs, [-1, size])

        lstm_c, lstm_h = lstm_state
        self.state_out = [lstm_c[:1, :], lstm_h[:1, :]]
        self.output = lstm_outputs
