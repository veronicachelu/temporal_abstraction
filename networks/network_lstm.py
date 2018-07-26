import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_embedding import EmbeddingNetwork
import os

"""Function approximation network for the option critic policies and value functions when options are given as embeddings corresponding to the spectral decomposition of the SR matrixand keeping track of the past with lsmts"""
class LSTMNetwork(EmbeddingNetwork):
  def __init__(self, scope, config, action_size):
    super(LSTMNetwork, self).__init__(scope, config, action_size)

  """Build the encoder for the latent representation space"""
  def build_feature_net(self, out):
    """keep a copy of the input"""
    input = out
    with tf.variable_scope("fi"):
      for i, nb_filt in enumerate(self.config.fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fi_{}".format(i))
        if i < len(self.config.fc_layers) - 1:
          out = tf.nn.relu(out)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      fi_relu = tf.nn.relu(out)

      """Placeholder for the previous rewards"""
      self.prev_rewards = tf.placeholder(shape=[None], dtype=tf.float32, name="Prev_Rewards")
      self.prev_rewards_expanded = tf.expand_dims(self.prev_rewards, 1)

      """Placeholder for the previous actions"""
      self.prev_actions = tf.placeholder(shape=[None], dtype=tf.int32, name="Prev_Actions")
      self.prev_actions_onehot = tf.one_hot(self.prev_actions, self.action_size, dtype=tf.float32,
                                            name="Prev_Actions_OneHot")
      """The merged representation of the input"""
      hidden = tf.concat([fi_relu, self.prev_rewards_expanded, self.prev_actions_onehot], 1,
                         name="Concatenated_input")
      """Preparing the input to the RNN, need to add a batch size of one since the current batch size will be used as the sequence size"""
      rnn_in = tf.expand_dims(hidden, [0], name="RNN_input")
      """This is the sequence size"""
      step_size = tf.shape(input)[:1]

      """Use a normal LSTM cell for the recurrent network"""
      lstm_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(self.config.sf_layers[-1])

      """Initialize the cell's state with zeros"""
      c_init = np.zeros((1, lstm_cell.state_size.c), np.float32)
      h_init = np.zeros((1, lstm_cell.state_size.h), np.float32)
      self.state_init = [c_init, h_init]

      """Placeholders for the previous state of the cell to plug in"""
      c_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.c], name="c_in")
      h_in = tf.placeholder(tf.float32, [1, lstm_cell.state_size.h], name="h_in")
      self.state_in = (c_in, h_in)

      """Create a state tuple for the previous state of the cell"""
      state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)

      """Rollout the LSTM for the sequence of inputs"""
      lstm_outputs, lstm_state = tf.nn.dynamic_rnn(
        lstm_cell, rnn_in, initial_state=state_in, sequence_length=step_size,
        time_major=False)

      """Get the next state of the LSTM"""
      lstm_c, lstm_h = lstm_state
      self.state_out = (lstm_c[:1, :], lstm_h[:1, :])

      """Get the output of the LSTM.
      This is the latent state representation"""
      self.fi = tf.reshape(lstm_outputs, [-1, self.config.sf_layers[-1]], name="fi_rnn")

      self.fi_relu = tf.nn.relu(self.fi)

      self.summaries_aux.append(tf.contrib.layers.summarize_activation(self.fi))

      """Plug in the option's direction"""
      self.option_direction_placeholder = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32,
                                                         name="option_direction")
      self.fi_option = tf.add(tf.stop_gradient(self.fi), self.option_direction_placeholder)

