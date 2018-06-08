import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_base import BaseNetwork
import os


class EmbeddingNetwork(BaseNetwork):
  def __init__(self, scope, config, action_size, lr, network_optimizer, total_steps_tensor=None):
    super(EmbeddingNetwork, self).__init__(scope, config, action_size, lr, network_optimizer, total_steps_tensor)
    self.random_option_prob = tf.Variable(self.config.initial_random_option_prob, trainable=False,
                                          name="prob_of_random_option", dtype=tf.float32)
    self.summaries_term = []
    self.summaries_critic = []
    self.summaries_eigen_critic = []
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
      self.fi_relu = tf.nn.elu(self.fi)
      self.option_direction_placeholder = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32,
                                                         name="option_direction")
      self.fi_option = tf.add(tf.stop_gradient(self.fi), self.option_direction_placeholder)

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
    with tf.variable_scope("eigen_option_q"):
      # out = tf.stop_gradient(self.fi_option)
      out = self.fi_option
      self.eigen_q_val = layers.fully_connected(out, num_outputs=1,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="eigen_q_val")
      self.eigen_q_val = tf.squeeze(self.eigen_q_val, 1)
      self.summaries_eigen_critic.append(tf.contrib.layers.summarize_activation(self.eigen_q_val))

  def build_intraoption_policies_nets(self):
    with tf.variable_scope("option_pi"):
      # out = tf.stop_gradient(self.fi_option)
      out = self.fi_option
      self.option = layers.fully_connected(out, num_outputs=self.action_size,
                                           activation_fn=tf.nn.softmax,
                                           biases_initializer=None,
                                           variables_collections=tf.get_collection("variables"),
                                           outputs_collections="activations", scope="policy")
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.option))

  def build_option_term_net(self):
    with tf.variable_scope("option_term"):
      out = tf.stop_gradient(self.fi_option)
      # out = self.fi_option
      self.termination = layers.fully_connected(out, num_outputs=1,
                                                activation_fn=tf.nn.sigmoid,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="termination")
      self.termination = tf.squeeze(self.termination, 1)
      self.summaries_term.append(tf.contrib.layers.summarize_activation(self.termination))

  def build_option_q_val_net(self):
    with tf.variable_scope("option_q"):
      # out = tf.stop_gradient(self.fi_relu)
      out = self.fi_relu
      self.q_val = layers.fully_connected(out, num_outputs=(
        self.nb_options + self.action_size) if self.config.include_primitive_options else self.nb_options,
                                          activation_fn=None,
                                          variables_collections=tf.get_collection("variables"),
                                          outputs_collections="activations", scope="q_val")
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.q_val))
      self.max_q_val = tf.reduce_max(self.q_val, 1)
      self.max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
      self.exp_options = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0, maxval=(
        self.nb_options + self.action_size) if self.config.include_primitive_options else self.nb_options,
                                           dtype=tf.int32)
      self.local_random = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0., maxval=1., dtype=tf.float32,
                                            name="rand_options")
      self.condition = self.local_random > self.random_option_prob

      self.current_option = tf.where(self.condition, self.max_options, self.exp_options, name="current_option")
      self.primitive_action = tf.where(self.current_option >= self.nb_options,
                                       tf.ones_like(self.current_option),
                                       tf.zeros_like(self.current_option))
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.current_option))
      self.v = tf.identity(self.max_q_val * (1 - self.random_option_prob) + \
               self.random_option_prob * tf.reduce_mean(self.q_val, axis=1), name="V")
      self.summaries_critic.append(tf.contrib.layers.summarize_activation(self.v))

      return out

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.observation = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = layers.flatten(out, scope="flatten")

      _ = self.build_feature_net(out)
      _ = self.build_option_term_net()
      _ = self.build_option_q_val_net()

      self.build_eigen_option_q_val_net()

      self.build_intraoption_policies_nets()
      self.build_SF_net(layer_norm=False)
      self.build_next_frame_prediction_net()
      self.build_placeholders(self.config.history_size)

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  def build_placeholders(self, next_frame_channel_size):
    self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
    self.target_next_obs = tf.placeholder(
      shape=[None, self.config.input_size[0], self.config.input_size[1], next_frame_channel_size], dtype=tf.float32,
      name="target_next_obs")
    self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="options")
    self.target_eigen_return = tf.placeholder(shape=[None], dtype=tf.float32)
    self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
    self.primitive_actions_placeholder = tf.placeholder(shape=[None], dtype=tf.bool,
                                                        name="primitive_actions_placeholder")

  def build_losses(self):
    self.responsible_actions = self.get_responsible_actions(self.option, self.actions_placeholder)

    self.q_val_o = self.get_q(self.options_placeholder)
    # o_term = self.get_o_term(self.options_placeholder)

    self.only_non_primitve_options = tf.map_fn(lambda x: tf.cond(tf.less(x, self.nb_options), lambda: x, lambda: 0),
                                               self.options_placeholder)

    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs * 255 * 128, self.target_next_obs * 255 * 128], 2), max_outputs=30))

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
        eigen_td_error = tf.where(self.primitive_actions_placeholder, tf.zeros_like(self.target_eigen_return),
                                  self.target_eigen_return - self.eigen_q_val)
        self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(eigen_td_error))

    with tf.name_scope('critic_loss'):
      td_error = self.target_return - self.q_val_o
    self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * huber_loss(td_error))

    with tf.name_scope('termination_loss'):
      self.term_err = (tf.stop_gradient(self.q_val_o) - tf.stop_gradient(self.v) + self.config.delib_margin)

      self.term_loss = tf.reduce_mean(tf.where(self.primitive_actions_placeholder, tf.zeros_like(self.q_val_o),
                                               self.termination * self.term_err))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.where(self.primitive_actions_placeholder,
                                                                       tf.zeros_like(self.option),
                                                                       self.option * tf.log(self.option + 1e-7)))
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(
        tf.where(self.primitive_actions_placeholder, tf.zeros_like(self.responsible_actions),
                 tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
                   eigen_td_error)))

    self.option_loss = self.policy_loss - self.entropy_loss + self.eigen_critic_loss

  def take_gradient(self, loss):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
    global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
    gradients = tf.gradients(loss, local_vars)
    var_norms = tf.global_norm(local_vars)
    grads, grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm_value)
    apply_grads = self.network_optimizer.apply_gradients(zip(grads, global_vars))
    return grads, apply_grads

  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    self.grads_sf, self.apply_grads_sf = self.take_gradient(self.sf_loss)
    self.grads_aux, self.apply_grads_aux = self.take_gradient(self.aux_loss)
    self.grads_critic, self.apply_grads_critic = self.take_gradient(self.critic_loss)
    self.grads_eigen_critic, self.apply_grad_eigen_critic = self.take_gradient(self.eigen_critic_loss)
    self.grads_term, self.apply_grads_term = self.take_gradient(self.term_loss)

    with tf.control_dependencies([self.apply_grad_eigen_critic]):
      self.grads_option, self.apply_grads_option = self.take_gradient(self.option_loss)


    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss),
        gradient_summaries(zip(self.grads_sf, local_vars))])
    self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss),
                                                 gradient_summaries(zip(self.grads_aux, local_vars))])
    options_to_merge = self.summaries_option + [
                                                tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                                                tf.summary.scalar('avg_policy_loss', self.policy_loss),
                                                tf.summary.scalar('random_option_prob', self.random_option_prob),
                                                tf.summary.scalar('LR', self.lr),
                                                tf.summary.scalar('avg_eigen_critic_loss', self.eigen_critic_loss),
                                                gradient_summaries(zip(self.grads_eigen_critic, local_vars)),
                                                gradient_summaries(zip(self.grads_option, local_vars),)]

    self.merged_summary_option = tf.summary.merge(options_to_merge)

    self.merged_summary_term = tf.summary.merge(
      self.summaries_term + [tf.summary.scalar('avg_termination_loss', self.term_loss)] + [
        tf.summary.scalar('avg_termination_error', tf.reduce_mean(self.term_err)),
        gradient_summaries(zip(self.grads_term, local_vars))])

    self.merged_summary_critic = tf.summary.merge(
      self.summaries_term + [tf.summary.scalar('avg_critic_loss', self.critic_loss),] + [
        gradient_summaries(zip(self.grads_critic, local_vars))])

    # self.merged_summary_eigen_critic = tf.summary.merge(
    #   self.summaries_eigen_critic + [tf.summary.scalar('avg_eigen_critic_loss', self.eigen_critic_loss),
    #                          gradient_summaries(zip(self.grads_eigen_critic, local_vars))])

