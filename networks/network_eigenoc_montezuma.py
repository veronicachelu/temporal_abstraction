import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_base import BaseNetwork
import os

class EignOCMontezumaNetwork(BaseNetwork):
  def __init__(self, scope, config, action_size, total_steps_tensor=None):
    super(EignOCMontezumaNetwork, self).__init__(scope, config, action_size, total_steps_tensor)
    self.build_network()

  def build_feature_net(self, out):
    with tf.variable_scope("fi"):
      for i, (kernel_size, stride, pad, nb_kernels) in enumerate(self.config.conv_layers):
        out = layers.conv2d(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                            stride=stride, activation_fn=None,
                            padding="SAME" if pad > 0 else "VALID",
                            variables_collections=tf.get_collection("variables"),
                            outputs_collections="activations", scope="conv_{}".format(i))
        out = self.layer_norm_fn(out, relu=True)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      out = layers.flatten(out, scope="flatten")

      for i, nb_filt in enumerate(self.fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fc_{}".format(i))
        if i < len(self.fc_layers) - 1:
          out = self.layer_norm_fn(out, relu=True)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      self.fi = out
      self.fi_relu = tf.nn.relu(self.fi)

      return out

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
          out = self.layer_norm_fn(out, relu=True)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))

      out = tf.reshape(out, (
      -1, self.config.aux_upconv_reshape[0], self.config.aux_upconv_reshape[1], self.config.aux_upconv_reshape[2]))
      for i, (kernel_size, stride, padding, nb_kernels) in enumerate(self.config.upconv_layers):
        out = layers.conv2d_transpose(out, num_outputs=nb_kernels, kernel_size=kernel_size,
                                      stride=stride, activation_fn=None,
                                      padding="SAME" if padding > 0 else "VALID",
                                      variables_collections=tf.get_collection("variables"),
                                      outputs_collections="activations", scope="upconv_{}".format(i))
        if i < len(self.config.upconv_layers) - 1:
          out = self.layer_norm_fn(out, relu=True)
        self.next_obs = out

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.observation = tf.placeholder(shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
                                        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = self.build_feature_net(out)
      out = self.build_option_term_net(out)
      _ = self.build_option_q_val_net(out)

      if self.config.eigen:
        self.build_eigen_option_q_val_net()

      self.build_intraoption_policies_nets()
      self.build_SF_net(layer_norm=True)
      self.build_next_frame_prediction_net()

      if self.scope != 'global':
        self.build_placeholders(self.config.channel_size)
        self.build_losses()
        self.gradients_and_summaries()