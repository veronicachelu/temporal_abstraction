import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_base import BaseNetwork
import os


class SomNetwork(BaseNetwork):
  def __init__(self, scope, config, action_size, total_steps_tensor=None):
    super(SomNetwork, self).__init__(scope, config, action_size, total_steps_tensor)
    self.summaries_reward = []
    self.random_option_prob = tf.Variable(self.config.initial_random_option_prob, trainable=False,
                                                   name="prob_of_random_option", dtype=tf.float32)
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

      return out

  def build_reward_pred_net(self):
    out = tf.stop_gradient(self.fi_relu)
    out = layers.fully_connected(out, num_outputs=1,
                                 activation_fn=None, biases_initializer=None,
                                 variables_collections=tf.get_collection("variables"),
                                 outputs_collections="activations", scope="reward")
    self.summaries_reward.append(tf.contrib.layers.summarize_activation(out))
    self.r = out

    # with tf.variable_scope("reward_pred_i"):
    #   self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="options")
    #   self.options_fc = layers.fully_connected(tf.cast(self.options_placeholder, tf.float32)[..., None], num_outputs=self.sf_layers[-1],
    #                                    activation_fn=None,
    #
    #                                    variables_collections=tf.get_collection("variables"),
    #                                    outputs_collections="activations", scope="fc")
    #
    # self.fi_o = tf.add(tf.stop_gradient(self.fi), self.options_fc)
    out = tf.stop_gradient(self.fi_relu)
    # out = layers.fully_connected(out, num_outputs=self.nb_options,
    #                              activation_fn=None, biases_initializer=None,
    #                              variables_collections=tf.get_collection("variables"),
    #                              outputs_collections="activations", scope="reward_i")
    out = layers.fully_connected(out, num_outputs=self.nb_options * self.action_size,
                                 activation_fn=None, biases_initializer=None,
                                 variables_collections=tf.get_collection("variables"),
                                 outputs_collections="activations", scope="reward_i")
    self.summaries_reward.append(tf.contrib.layers.summarize_activation(out))
    self.r_i = tf.reshape(out, (-1, self.nb_options, self.action_size))
    # self.r_i = out

  def get_w(self):
    with tf.variable_scope("reward", reuse=True):
      v = tf.get_variable("weights")
    return v

  # def get_wg(self):
  #   with tf.variable_scope("reward_i", reuse=True):
  #     v = tf.get_variable("weights")
  #   return v

  def get_wg(self):
    with tf.variable_scope("reward_i", reuse=True):
      v = tf.get_variable("weights")
      v = tf.reshape(v, (self.fc_layers[-1], self.nb_options, self.action_size))
    return v


  def build_next_frame_prediction_net(self):
    with tf.variable_scope("aux_action_fc"):
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
      actions = layers.fully_connected(tf.cast(self.actions_placeholder[..., None], tf.float32),  num_outputs=self.fc_layers[-1],
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

  def build_SF_net(self, layer_norm=False):
    # with tf.variable_scope("aux_option_fc"):
    #   self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="options")
    #   self.options_fc = layers.fully_connected(self.options_placeholder[..., None], num_outputs=self.sf_layers[-1],
    #                                    activation_fn=None,
    #                                    variables_collections=tf.get_collection("variables"),
    #                                    outputs_collections="activations", scope="fc")

    with tf.variable_scope("sf"):
      # self.fi_o = tf.add(tf.stop_gradient(self.fi), self.options_fc)
      out = tf.stop_gradient(self.fi_relu)
      for i, nb_filt in enumerate(self.sf_layers):
        out = layers.fully_connected(out, num_outputs=nb_filt * (self.nb_options + self.action_size),
                                     activation_fn=None,
                                     biases_initializer=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="sf_{}".format(i))
        if i < len(self.sf_layers) - 1:
          if layer_norm:
            out = self.layer_norm_fn(out, relu=True)
          else:
            out = tf.nn.relu(out)
        self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
      self.sf = tf.reshape(out, (-1, (self.nb_options + self.action_size), self.sf_layers[-1]))

  def build_option_q_val_net(self):
    self.w = self.get_w()
    with tf.variable_scope("option_q_val"):
      self.q_val = tf.map_fn(lambda x: tf.matmul(x, self.w), self.sf)
      self.q_val = tf.squeeze(self.q_val, 2)
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.q_val))
      self.max_q_val = tf.reduce_max(self.q_val, 1)
      self.max_options = tf.cast(tf.argmax(self.q_val, 1), dtype=tf.int32)
      self.exp_options = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0, maxval=(
        self.nb_options + self.action_size) if self.config.include_primitive_options else self.nb_options,
                                           dtype=tf.int32)
      self.local_random = tf.random_uniform(shape=[tf.shape(self.q_val)[0]], minval=0., maxval=1., dtype=tf.float32,
                                            name="rand_options")
      self.condition = self.local_random > self.random_option_prob

      self.current_option = tf.where(self.condition, self.max_options, self.exp_options)
      self.primitive_action = tf.where(self.current_option >= self.nb_options,
                                       tf.ones_like(self.current_option),
                                       tf.zeros_like(self.current_option))
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.current_option))
      self.v = self.max_q_val * (1 - self.random_option_prob) + \
               self.random_option_prob * tf.reduce_mean(self.q_val, axis=1)
      self.summaries_option.append(tf.contrib.layers.summarize_activation(self.v))

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.observation = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], self.config.history_size],
        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = layers.flatten(out, scope="flatten")

      _ = self.build_feature_net(out)
      _ = self.build_option_term_net()

      self.build_intraoption_policies_nets()
      self.build_SF_net(layer_norm=False)
      self.build_next_frame_prediction_net()
      self.build_reward_pred_net()

      _ = self.build_option_q_val_net()

      self.decrease_prob_of_random_option = tf.assign_sub(self.random_option_prob, tf.constant(
        (self.config.initial_random_option_prob - self.config.final_random_option_prob) / self.config.explore_options_episodes))

      if self.scope != 'global':
        self.build_placeholders(self.config.history_size)
        self.build_losses()
        self.gradients_and_summaries()

  def build_placeholders(self, next_frame_channel_size):
    self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
    self.target_next_obs = tf.placeholder(
      shape=[None, self.config.input_size[0], self.config.input_size[1], next_frame_channel_size], dtype=tf.float32,
      name="target_next_obs")
    self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="options")
    self.target_r = tf.placeholder(shape=[None], dtype=tf.float32)
    self.target_r_i = tf.placeholder(shape=[None], dtype=tf.float32)
    self.sf_td_error_target = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32,
                                             name="sf_td_error_target")
    self.wg = self.get_wg()
    self.sf_o = self.get_sf_o(self.options_placeholder)

  def build_losses(self):
    self.policies = self.get_intra_option_policies(self.options_placeholder)
    self.responsible_actions = self.get_responsible_actions(self.policies, self.actions_placeholder)

    # if self.config.eigen:
    #   eigen_q_val = self.get_eigen_q(self.options_placeholder)
    q_val = self.get_q(self.options_placeholder)
    o_term = self.get_o_term(self.options_placeholder)
    # r_i_o = self.get_r_i_o(self.options_placeholder)
    # wg_o = self.get_wg_o(self.options_placeholder)

    r_i_o_a = self.get_r_i_o_a(self.options_placeholder, self.actions_placeholder)
    wg_o_a = self.get_wg_o_a(self.options_placeholder, self.actions_placeholder)

    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

    # self.matrix_sf = tf.placeholder(shape=[self.config.sf_matrix_size, self.sf_layers[-1]],
    #                                 dtype=tf.float32, name="matrix_sf")
    # if self.config.sf_matrix_size is None:
    #   self.config.sf_matrix_size = self.nb_states
    # self.matrix_sf = tf.placeholder(shape=[1, self.config.sf_matrix_size, self.sf_layers[-1]],
    #                                 dtype=tf.float32, name="matrix_sf")
    # self.eigenvalues, _, ev = tf.svd(self.matrix_sf, full_matrices=True, compute_uv=True)
    # self.eigenvectors = tf.transpose(tf.conj(ev), perm=[0, 2, 1])

    with tf.name_scope('sf_loss'):
      self.sf_td_error = self.target_sf - self.sf_o
    self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(self.sf_td_error))

    with tf.name_scope('reward_loss'):
      reward_error = self.r - self.target_r
    self.reward_loss = tf.reduce_mean(self.config.reward_coef * huber_loss(reward_error))

    with tf.name_scope('reward_loss_i'):
      reward_i_error = r_i_o_a - self.target_r_i
    self.reward_i_loss = tf.reduce_mean(self.config.reward_i_coef * huber_loss(reward_i_error))

    with tf.name_scope('aux_loss'):
      aux_error = self.next_obs - self.target_next_obs
    self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

    # if self.config.eigen:
    #   with tf.name_scope('eigen_critic_loss'):
    #     eigen_td_error = self.target_eigen_return - eigen_q_val
    #     self.eigen_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(eigen_td_error))

    # with tf.name_scope('critic_loss'):
    #   td_error = self.target_return - q_val
    # self.critic_loss = tf.reduce_mean(0.5 * self.config.critic_coef * tf.square(td_error))

    with tf.name_scope('termination_loss'):
      self.term_loss = tf.reduce_mean(
        o_term * (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + 0.01))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.policies *
                                                                            tf.log(self.policies + 1e-7),
                                                                            axis=1))
    with tf.name_scope('policy_loss'):
      advantage = tf.map_fn(lambda i: tf.matmul(self.sf_td_error_target[i][None, ...], wg_o_a[i]),
                            tf.range(tf.shape(self.sf_td_error_target)[0]), dtype=tf.float32)
      self.advantage = tf.squeeze(advantage, axis=[1, 2])
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(advantage))

    self.option_loss = self.policy_loss - self.entropy_loss + self.term_loss

  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    grads_list, grad_norm_list, apply_grads_list = self.compute_gradients(
      [self.sf_loss, self.reward_loss, self.reward_i_loss, self.aux_loss, self.option_loss])
    grads_sf, grads_reward, grads_reward_i, grads_aux, grads_option = grads_list
    grads_sf_norm, grads_reward_norm, grads_reward_i_norm, grads_aux_norm, grads_option_norm = grad_norm_list
    self.apply_grads_sf, self.apply_grads_reward, self.apply_grads_reward_i, self.apply_grads_aux, self.apply_grads_option = apply_grads_list

    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss),
                           tf.summary.scalar('avg_sf_td_error', tf.reduce_mean(self.sf_td_error)),
                           tf.summary.scalar('gradient_norm_sf', grads_sf_norm),
                           gradient_summaries(zip(grads_sf, local_vars))])
    self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                               [tf.summary.scalar('aux_loss', self.aux_loss),
                                                tf.summary.scalar('gradient_norm_aux',
                                                                  grads_aux_norm),
                                                gradient_summaries(zip(grads_aux, local_vars))])
    self.merged_summary_option = tf.summary.merge(self.summaries_option + [
      tf.summary.scalar('avg_termination_loss', self.term_loss),
      tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
      tf.summary.scalar('avg_policy_loss', self.policy_loss),
      tf.summary.scalar('advantage', tf.reduce_mean(self.advantage)),
      tf.summary.scalar('avg_option_loss', self.option_loss),
      tf.summary.scalar('gradient_norm_option', grads_option_norm),
      gradient_summaries(zip(grads_option, local_vars))])
    self.merged_summary_reward = tf.summary.merge(self.summaries_reward + [
      tf.summary.scalar('avg_reward_loss', self.reward_loss),
      tf.summary.scalar('gradient_norm_reward', grads_reward_norm),
      gradient_summaries(zip(grads_reward, local_vars))])

    self.merged_summary_reward_i = tf.summary.merge(self.summaries_reward + [
      tf.summary.scalar('avg_reward_i_loss', self.reward_i_loss),
      tf.summary.scalar('gradient_norm_reward_i', grads_reward_i_norm),
      gradient_summaries(zip(grads_reward_i, local_vars))])

  # def get_r_i_o(self, o):
  #   options_taken_one_hot = tf.one_hot(o, self.config.nb_options,
  #                                      name="options_one_hot")
  #   r_i_o = tf.reduce_sum(tf.multiply(self.r_i, options_taken_one_hot),
  #                         reduction_indices=1, name="option_r_i")
  #   return r_i_o

  def get_r_i_o_a(self, o, a):
    options_taken_one_hot = tf.one_hot(o, self.nb_options,
                                       name="options_one_hot")[..., None]
    options_taken_one_hot_tile = tf.tile(options_taken_one_hot, (1, 1, self.action_size))
    actions_taken_one_hot = tf.one_hot(a, self.action_size,
                                       name="actions_one_hot")
    r_i_o = tf.reduce_sum(tf.multiply(self.r_i, options_taken_one_hot_tile),
                          reduction_indices=1, name="option_r_i_o")
    r_i_o_a = tf.reduce_sum(tf.multiply(r_i_o, actions_taken_one_hot),
                          reduction_indices=1, name="option_r_i_oa")
    return r_i_o_a

  def get_sf_o(self, o):
    options_taken_one_hot = tf.one_hot(o, (
      self.config.nb_options + self.action_size) if self.config.include_primitive_options else self.config.nb_options,
                                       name="options_one_hot")
    options_taken_one_hot_tile = tf.tile(options_taken_one_hot[..., None], (1, 1, self.sf_layers[-1]))
    sf_o = tf.reduce_sum(tf.multiply(self.sf, options_taken_one_hot_tile),
                         reduction_indices=1, name="SF_o")
    return sf_o

  # def get_wg_o(self, o):
  #   options_taken_one_hot = tf.one_hot(o, self.config.nb_options,
  #                                      name="options_one_hot")
  #   options_taken_one_hot_tile = tf.tile(options_taken_one_hot[..., None], (1, 1, self.fc_layers[-1]))
  #   wg_tile = tf.tile(tf.transpose(self.wg, (1, 0))[None, ...], (tf.shape(options_taken_one_hot)[0], 1, 1))
  #   wg_o = tf.reduce_sum(tf.multiply(wg_tile, options_taken_one_hot_tile),
  #                        reduction_indices=1, name="wg_o")
  #   wg_o = wg_o[..., None]
  #
  #   return wg_o

  def get_wg_o_a(self, o, a):
    options_taken_one_hot = tf.one_hot(o, self.nb_options,
                                       name="options_one_hot")[..., None]
    actions_taken_one_hot = tf.one_hot(a, self.action_size,
                                       name="actions_one_hot")
    options_taken_one_hot_tile = tf.tile(options_taken_one_hot[..., None], (1, 1, self.action_size, self.fc_layers[-1]))
    wg_tile = tf.tile(tf.transpose(self.wg, (1, 2, 0))[None, ...], (tf.shape(options_taken_one_hot)[0], 1, 1, 1))
    wg_o = tf.reduce_sum(tf.multiply(wg_tile, options_taken_one_hot_tile),
                         reduction_indices=1, name="wg_o")
    actions_taken_one_hot_tile = tf.tile(actions_taken_one_hot[..., None], (1, 1, self.fc_layers[-1]))
    wg_o_a = tf.reduce_sum(tf.multiply(wg_o, actions_taken_one_hot_tile),
                         reduction_indices=1, name="wg_o_a")
    wg_o_a = wg_o_a[..., None]

    return wg_o_a
