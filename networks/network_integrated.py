import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_base import BaseNetwork
import os


class IntegratedNetwork(BaseNetwork):
  def __init__(self, scope, config, action_size, total_steps_tensor=None):
    super(IntegratedNetwork, self).__init__(scope, config, action_size, total_steps_tensor)
    self.summaries_reward = []
    self.summaries_reward_i = []
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
    self.r = layers.fully_connected(out, num_outputs=1,
                                    activation_fn=None, biases_initializer=None,
                                    variables_collections=tf.get_collection("variables"),
                                    outputs_collections="activations", scope="reward")
    self.summaries_reward.append(tf.contrib.layers.summarize_activation(self.r))

    self.options_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="options")
    self.option_fc = layers.fully_connected(tf.cast(self.options_placeholder[..., None], tf.float32),
                                            num_outputs=self.fc_layers[-1],
                                            activation_fn=None,
                                            variables_collections=tf.get_collection("variables"),
                                            outputs_collections="activations", scope="reward_fc")
    out = tf.add(tf.stop_gradient(self.fi_actions), self.option_fc)
    self.r_i = layers.fully_connected(out, num_outputs=1,
                                      activation_fn=None, biases_initializer=None,
                                      variables_collections=tf.get_collection("variables"),
                                      outputs_collections="activations", scope="reward_i")
    self.summaries_reward_i.append(tf.contrib.layers.summarize_activation(self.r_i))

    self.w = self.get_w()
    self.w_i = self.get_wi()

  def get_w(self):
    with tf.variable_scope("reward", reuse=True):
      v = tf.get_variable("weights")
    return v

  def get_wi(self):
    with tf.variable_scope("reward_i", reuse=True):
      v = tf.get_variable("weights")
      # v = tf.reshape(v, (self.nb_options, self.action_size, self.fc_layers[-1]))
    return v

  def build_next_frame_prediction_net(self):
    with tf.variable_scope("aux_action_fc"):
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
      self.fc_actions = layers.fully_connected(tf.cast(self.actions_placeholder[..., None], tf.float32),
                                               num_outputs=self.fc_layers[-1],
                                               activation_fn=None,
                                               variables_collections=tf.get_collection("variables"),
                                               outputs_collections="activations", scope="fc")

    with tf.variable_scope("aux_next_frame"):
      self.fi_actions = tf.add(self.fi, self.fc_actions)
      out = self.fi_actions
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
    with tf.variable_scope("sf"):
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

  # def build_SF_net(self, layer_norm=False):
  #   with tf.variable_scope("sf"):
  #     out = tf.stop_gradient(self.fi_relu)
  #     for i, nb_filt in enumerate(self.sf_layers):
  #       out = layers.fully_connected(out, num_outputs=nb_filt,
  #                                    activation_fn=None,
  #                                    biases_initializer=None,
  #                                    variables_collections=tf.get_collection("variables"),
  #                                    outputs_collections="activations", scope="sf_{}".format(i))
  #       if i < len(self.sf_layers) - 1:
  #         if layer_norm:
  #           out = self.layer_norm_fn(out, relu=True)
  #         else:
  #           out = tf.nn.relu(out)
  #       self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
  #     self.sf = out

  def build_option_q_val_net(self):
    with tf.variable_scope("q_val"):
      self.q_val = tf.reduce_sum(tf.stop_gradient(self.sf) *
                                 tf.tile((tf.squeeze(self.w, 1)[None, ...])[None, ...],
                                         [tf.shape(self.sf)[0], self.nb_options + self.action_size, 1]), axis=2)
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
      self.exp_sf = self.get_sf_o(self.max_options) * (1 - self.random_option_prob) + \
               self.random_option_prob * tf.reduce_mean(self.sf, axis=1)
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
      self.build_placeholders(self.config.history_size)

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  def build_placeholders(self, next_frame_channel_size):
    self.target_sf = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32, name="target_SF")
    self.target_next_obs = tf.placeholder(
      shape=[None, self.config.input_size[0], self.config.input_size[1], next_frame_channel_size], dtype=tf.float32,
      name="target_next_obs")
    self.target_r = tf.placeholder(shape=[None], dtype=tf.float32)
    self.target_r_i = tf.placeholder(shape=[None], dtype=tf.float32)
    self.sf_td_error_target = tf.placeholder(shape=[None, self.sf_layers[-1]], dtype=tf.float32,
                                             name="sf_td_error_target")
    self.sf_o = self.get_sf_o(self.options_placeholder)

  def build_losses(self):
    self.policies = self.get_intra_option_policies(self.options_placeholder)
    self.responsible_actions = self.get_responsible_actions(self.policies, self.actions_placeholder)

    q_val = self.get_q(self.options_placeholder)
    o_term = self.get_o_term(self.options_placeholder)
    # wi_oa = self.get_wi_oa(self.options_placeholder, self.actions_placeholder)

    self.image_summaries.append(
      tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))

    if self.config.sr_matrix == "dynamic":
      self.sf_matrix_size = self.config.sf_matrix_size
    else:
      self.sf_matrix_size = 104
    self.matrix_sf = tf.placeholder(shape=[None, self.sf_matrix_size, self.sf_layers[-1]],
                                    dtype=tf.float32, name="matrix_sf")
    self.eigenvalues, _, ev = tf.svd(self.matrix_sf, full_matrices=False, compute_uv=True)
    self.eigenvectors = tf.transpose(tf.conj(ev), perm=[0, 2, 1])

    with tf.name_scope('sf_loss'):
      self.sf_td_error = self.target_sf - self.sf_o
    self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(self.sf_td_error))

    with tf.name_scope('reward_loss'):
      reward_error = self.target_r - self.r
      self.reward_loss = tf.reduce_mean(self.config.reward_coef * huber_loss(reward_error))

    with tf.name_scope('reward_loss_i'):
      reward_i_error = self.target_r_i - self.r_i
      self.reward_i_loss = tf.reduce_mean(self.config.reward_i_coef * huber_loss(reward_i_error))

    with tf.name_scope('aux_loss'):
      aux_error = self.next_obs - self.target_next_obs
    self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

    with tf.name_scope('termination_loss'):
      self.term_loss = tf.reduce_mean(
        o_term * (tf.stop_gradient(q_val) - tf.stop_gradient(self.v) + 0.01))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(tf.reduce_sum(self.policies *
                                                                            tf.log(self.policies + 1e-7),
                                                                            axis=1))
    with tf.name_scope('policy_loss'):
      self.advantage = tf.reduce_sum(
        self.sf_td_error_target * tf.tile(tf.squeeze(self.w_i, 1)[None, ...], [tf.shape(self.sf_td_error_target)[0], 1]),
        axis=1)
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(self.advantage))

    self.option_loss = self.policy_loss - self.entropy_loss + self.term_loss

  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    grads_list, grad_norm_list, apply_grads_list = self.compute_gradients(
      [self.sf_loss, self.reward_loss, self.reward_i_loss, self.aux_loss, self.option_loss])
    grads_sf, grads_reward, grads_reward_i, grads_aux, grads_option = grads_list
    # grads_sf, grads_reward, grads_reward_i, grads_aux = grads_list
    grads_sf_norm, grads_reward_norm, grads_reward_i_norm, grads_aux_norm, grads_option_norm = grad_norm_list
    # grads_sf_norm, grads_reward_norm, grads_reward_i_norm, grads_aux_norm = grad_norm_list
    self.apply_grads_sf, self.apply_grads_reward, self.apply_grads_reward_i, self.apply_grads_aux, self.apply_grads_option = apply_grads_list
    # self.apply_grads_sf, self.apply_grads_reward, self.apply_grads_reward_i, self.apply_grads_aux = apply_grads_list

    self.grads_sf = grads_sf
    self.grads_sf_norm = grads_sf_norm

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

    self.merged_summary_reward_i = tf.summary.merge(self.summaries_reward_i + [
      tf.summary.scalar('avg_reward_i_loss', self.reward_i_loss),
      tf.summary.scalar('gradient_norm_reward_i', grads_reward_i_norm),
      gradient_summaries(zip(grads_reward_i, local_vars))])

  def get_sf_o(self, o):
    indices = tf.stack([tf.range(tf.shape(o)[0]), o], axis=1)
    sf_o = tf.gather_nd(self.sf, indices)

    return sf_o

  def get_wi_oa(self, o, a):
    indices = tf.stack([tf.range(tf.shape(o)[0]), o, a], axis=1)
    wi_o_a = tf.gather_nd(tf.tile(self.w_i[None, ...], [tf.shape(o)[0], 1, 1, 1]), indices, name="wi_o_a")
    return wi_o_a
