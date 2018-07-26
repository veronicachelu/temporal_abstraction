import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
import os

"""Non-Linear function approximation network for the successor representation when observations are high-dimensional pixel-space inputs"""
class DynSRNetwork():
  def __init__(self, scope, config, action_size):
    self._scope = scope
    """The size of the input space flatten out"""
    self.nb_states = config.input_size[0] * config.input_size[1]
    self.config = config

    """Creating buffers for holding summaries"""
    self.image_summaries = []
    self.summaries_sf = []
    self.summaries_aux = []

    """Instantiating optimizer"""
    self.network_optimizer = config.network_optimizer(
      self.config.lr, name='network_optimizer')

    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, config.input_size[0], config.input_size[1], config.history_size],
                                        dtype=tf.float32, name="Inputs")
      out = self.observation
      out = layers.flatten(out, scope="flatten")

      """State space encoder into latent fi(s)"""
      with tf.variable_scope("fi"):
        for i, nb_filt in enumerate(self.config.fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="fc_fi_{}".format(i))

          if i < len(self.config.fc_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.fi = out

      """Successor representation mapping to latent psi(s)"""
      with tf.variable_scope("succ_feat"):
        out = tf.stop_gradient(tf.nn.relu(self.fi))
        for i, nb_filt in enumerate(self.config.sf_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf_{}".format(i))
          if i < len(self.config.sf_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_sf.append(tf.contrib.layers.summarize_activation(out))
        self.sf = out

      """Plugging in the current action taken into the environment for next frame prediction"""
      with tf.variable_scope("action_fc"):
        self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.float32, name="Actions")
        actions = layers.fully_connected(self.actions_placeholder[..., None], num_outputs=self.config.fc_layers[-1],
                                         activation_fn=None,
                                         variables_collections=tf.get_collection("variables"),
                                         outputs_collections="activations", scope="action_fc")

      """Decoder from latent space fi(s) to the next state"""
      with tf.variable_scope("aux_fc"):
        out = tf.add(self.fi, actions)
        out = tf.nn.relu(out)
        for i, nb_filt in enumerate(self.config.aux_fc_layers):
          out = layers.fully_connected(out, num_outputs=nb_filt,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="aux_fc_{}".format(i))
          if i < len(self.config.aux_fc_layers) - 1:
            out = tf.nn.relu(out)
          self.summaries_aux.append(tf.contrib.layers.summarize_activation(out))
        self.next_obs = tf.reshape(out, (-1, config.input_size[0], config.input_size[1], config.history_size))

      if scope != 'global':
        """Placeholder for the target successor representation at the next time step"""
        self.target_sf = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32, name="target_SF")

        """Placeholder for the target observation at the next time step - for self-supervised prediction of the next frame"""
        self.target_next_obs = tf.placeholder(
          shape=[None, config.input_size[0], config.input_size[1], config.history_size], dtype=tf.float32,
          name="target_next_obs")

        """Adding comparison of predicted frame and actual next frame to tensorboard"""
        self.image_summaries.append(
          tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2),
                           max_outputs=30))

        """Building losses"""
        with tf.name_scope('sf_loss'):
          """TD error of successor representations"""
          sf_td_error = self.target_sf - self.sf
          self.sf_loss = tf.reduce_mean(huber_loss(sf_td_error))

        with tf.name_scope('aux_loss'):
          """L2 loss for the next frame prediction"""
          aux_error = self.next_obs - self.target_next_obs
          self.aux_loss = tf.reduce_mean(self.config.aux_coef * huber_loss(aux_error))

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        gradients_sf = tf.gradients(self.sf_loss, local_vars)
        grads_sf,  grad_norms_sf = tf.clip_by_global_norm(gradients_sf, self.config.gradient_clip_norm_value)

        gradients_aux = tf.gradients(self.aux_loss, local_vars)
        grads_aux, grad_norms_aux = tf.clip_by_global_norm(gradients_aux, self.config.gradient_clip_norm_value)

        self.merged_summary_sf = tf.summary.merge(
          self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss), gradient_summaries(zip(grads_sf, local_vars))])
        self.merged_summary_aux = tf.summary.merge(self.image_summaries + self.summaries_aux +
                                                   [tf.summary.scalar('aux_loss', self.aux_loss),
                                                     gradient_summaries(zip(grads_aux, local_vars))])

        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads_sf = self.network_optimizer.apply_gradients(zip(grads_sf, global_vars))
        self.apply_grads_aux = self.network_optimizer.apply_gradients(zip(grads_aux, global_vars))