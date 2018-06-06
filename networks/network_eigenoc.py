import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_base import BaseNetwork
import os


class EignOCNetwork(BaseNetwork):
  def __init__(self, scope, config, action_size, lr, network_optimizer, total_steps_tensor=None):
    super(EignOCNetwork, self).__init__(scope, config, action_size, lr, network_optimizer, total_steps_tensor)
    self.random_option_prob = tf.Variable(self.config.initial_random_option_prob, trainable=False,
                                         name="prob_of_random_option", dtype=tf.float32)
    self.build_network()

  def build_feature_net(self, out):
    with tf.variable_scope("fi"):
      for i, nb_filt in enumerate(self.fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fi_{}".format(i))
        if i < len(self.fc_layers) - 1:
          out = layers.layer_norm(out, scale=True, center=True)
          out = tf.nn.elu(out)

      out = layers.layer_norm(out, scale=True, center=True)
      out = tf.nn.elu(out)
      self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
      self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      self.summaries_option.append(tf.contrib.layers.summarize_activation(out))
      self.fi_relu = out
      # self.fi_relu = tf.identity(self.layer_norm_fn(self.fi, relu=True), "fi_relu")
      # self.summaries_sf.append(tf.contrib.layers.summarize_activation(self.fi_relu))
      # self.summaries_aux.append(tf.contrib.layers.summarize_activation(self.fi_relu))
      # self.summaries_option.append(tf.contrib.layers.summarize_activation(self.fi_relu))


  def layer_norm_fn(self, x, relu=True):
    x = layers.layer_norm(x, scale=True, center=True)
    if relu:
      x = tf.nn.relu(x)
    return x

  def build_next_frame_prediction_net(self):
    with tf.variable_scope("aux_action_fc"):
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
      actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.fc_layers[-1],
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc")

    with tf.variable_scope("aux_next_frame"):
      out = tf.add(self.fi, actions)
      # out = tf.nn.relu(out)
      for i, nb_filt in enumerate(self.aux_fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fc_{}".format(i))
        if i < len(self.aux_fc_layers) - 1:
          out = tf.nn.relu(out)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      self.next_obs = tf.reshape(out,
                                 (-1, self.config.input_size[0], self.config.input_size[1], self.config.history_size))

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.observation = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = layers.flatten(out, scope="flatten")
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")

      self.decrease_prob_of_random_option = tf.assign_sub(self.random_option_prob, tf.constant(
        (self.config.initial_random_option_prob - self.config.final_random_option_prob) / self.config.explore_options_episodes))

      self.build_feature_net(out)
      self.build_option_term_net()
      self.build_option_q_val_net()

      if self.config.eigen:
        self.build_eigen_option_q_val_net()

      self.build_intraoption_policies_nets()
      self.build_SF_net(layer_norm=False)
      # self.build_next_frame_prediction_net()
      self.build_placeholders(self.config.history_size)

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()
