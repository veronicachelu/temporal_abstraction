import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_base import BaseNetwork
import os

class EmbeddingNetwork(BaseNetwork):
  def __init__(self, scope, config, action_size, total_steps_tensor=None):
    super(EmbeddingNetwork, self).__init__(scope, config, action_size, total_steps_tensor)
    self.build_network()

  def build_feature_net(self, out):
    with tf.variable_scope("fi"):
      for i, nb_filt in enumerate(self.fc_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt,
                                     activation_fn=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="fc_{}".format(i))

        if i < len(self.fc_layers) - 1:
          out = tf.nn.relu(out)
        self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.summaries_option.append(tf.contrib.layers.summarize_activation(out))
      self.fi = out
      self.fi_relu = tf.nn.relu(self.fi)
      self.option_direction_placeholder = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32,
                                                         name="option_direction")
      self.fi_option = tf.add(tf.stop_gradient(self.fi_relu), self.option_direction_placeholder)

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
          out = tf.nn.relu(out)
        self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
      self.next_obs = tf.reshape(out,
                                 (-1, self.config.input_size[0], self.config.input_size[1], self.config.history_size))

  def build_eigen_option_q_val_net(self):
    with tf.variable_scope("eigen_option_q_val"):
      self.eigen_q_val = layers.fully_connected(self.fi_option, num_outputs=1,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="eigen_q_val")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.eigen_q_val))

  def build_intraoption_policies_nets(self):
    with tf.variable_scope("eigen_option_i_o_policies"):
      self.option = layers.fully_connected(self.fi_option, num_outputs=self.action_size,
                                      activation_fn=tf.nn.softmax,
                                      biases_initializer=None,
                                      variables_collections=tf.get_collection("variables"),
                                      outputs_collections="activations", scope="policy")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.option))

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.observation = tf.placeholder(shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
                                        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = layers.flatten(out, scope="flatten")

      _ = self.build_feature_net(out)
      _ = self.build_option_term_net()
      _ = self.build_option_q_val_net()

      if self.config.eigen:
        self.build_eigen_option_q_val_net()

      self.build_intraoption_policies_nets()
      self.build_SF_net(layer_norm=False)
      self.build_next_frame_prediction_net()
      self.build_placeholders(self.config.history_size)

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

    def build_losses(self):
      self.responsible_actions = self.get_responsible_actions(self.option, self.actions_placeholder)

      q_val = self.get_q(self.options_placeholder)
      o_term = self.get_o_term(self.options_placeholder)

      self.image_summaries.append(
        tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

      if self.config.sr_matrix == "dynamic":
        self.sf_matrix_size = self.config.sf_matrix_size
      else:
        self.sf_matrix_size = 104
      self.matrix_sf = tf.placeholder(shape=[1, self.sf_matrix_size, self.sf_layers[-1]],
                                      dtype=tf.float32, name="matrix_sf")
      self.eigenvalues, _, ev = tf.svd(self.matrix_sf, full_matrices=False, compute_uv=True)
      self.eigenvectors = tf.transpose(tf.conj(ev), perm=[0, 2, 1])

      with tf.name_scope('sf_loss'):
        sf_td_error = self.target_sf - self.sf
      self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

      with tf.name_scope('aux_loss'):
        aux_error = self.next_obs - self.target_next_obs
      self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

      if self.config.eigen:
        with tf.name_scope('eigen_critic_loss'):
          eigen_td_error = self.target_eigen_return - self.eigen_q_val
          self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(eigen_td_error))

      with tf.name_scope('critic_loss'):
        td_error = self.target_return - q_val
      self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * tf.square(td_error))

      with tf.name_scope('termination_loss'):
        self.term_loss = tf.reduce_mean(
          o_term * (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + 0.01))

      with tf.name_scope('entropy_loss'):
        self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.policies *
                                                                              tf.log(self.policies + 1e-7),
                                                                              axis=1))
      with tf.name_scope('policy_loss'):
        self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
          eigen_td_error))

      self.option_loss = self.policy_loss - self.entropy_loss + self.critic_loss + self.term_loss
      if self.config.eigen:
        self.option_loss += self.eigen_critic_loss

  def build_option_term_net(self):
    with tf.variable_scope("eigen_option_term"):
      out = tf.stop_gradient(self.fi_option)
      self.termination = layers.fully_connected(out, num_outputs=1,
                                                activation_fn=tf.nn.sigmoid,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="fc_option_term")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.termination))

      return out

  def build_losses(self):
    self.responsible_actions = self.get_responsible_actions(self.option, self.actions_placeholder)

    q_val = self.get_q(self.options_placeholder)
    # o_term = self.get_o_term(self.options_placeholder)

    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

    if self.config.sr_matrix == "dynamic":
      self.sf_matrix_size = self.config.sf_matrix_size
    else:
      self.sf_matrix_size = 104
    self.matrix_sf = tf.placeholder(shape=[1, self.sf_matrix_size, self.sf_layers[-1]],
                                    dtype=tf.float32, name="matrix_sf")
    self.eigenvalues, _, ev = tf.svd(self.matrix_sf, full_matrices=False, compute_uv=True)
    self.eigenvectors = tf.transpose(tf.conj(ev), perm=[0, 2, 1])

    with tf.name_scope('sf_loss'):
      sf_td_error = self.target_sf - self.sf
    self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    with tf.name_scope('aux_loss'):
      aux_error = self.next_obs - self.target_next_obs
    self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

    if self.config.eigen:
      with tf.name_scope('eigen_critic_loss'):
        eigen_td_error = self.target_eigen_return - self.eigen_q_val
        self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(eigen_td_error))

    with tf.name_scope('critic_loss'):
      td_error = self.target_return - q_val
    self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * tf.square(td_error))

    with tf.name_scope('termination_loss'):
      self.term_loss = tf.reduce_mean(
        self.termination * (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + 0.01))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.option *
                                                                            tf.log(self.option + 1e-7)))
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(eigen_td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.critic_loss + self.term_loss
    if self.config.eigen:
      self.option_loss += self.eigen_critic_loss