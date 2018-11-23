import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries, huber_loss
import numpy as np
from networks.network_eigenoc import EignOCNetwork
import os
from online_clustering import OnlineCluster

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)

    return _initializer

class AttentionNetwork(EignOCNetwork):
  def __init__(self, scope, config, action_size):
    self.goal_embedding_size = config.sf_layers[-1]
    super(AttentionNetwork, self).__init__(scope, config, action_size)

    if self.config.use_clustering:
      self.init_clustering()

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.target_sf = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32, name="target_SF")
      self.target_return = tf.placeholder(shape=[None], dtype=tf.float32)
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
      self.observation = tf.placeholder(
        shape=[None, self.nb_states],
        dtype=tf.float32, name="observation_placeholder")
      self.observation_image = tf.placeholder(
        shape=[None, self.config.input_size[0], self.config.input_size[1], 1],
        dtype=tf.float32, name="observation_image_placeholder")

      goal_clusters = tf.placeholder(shape=[self.config.nb_options,
                                            self.goal_embedding_size],
                                     dtype=tf.float32,
                                     name="goal_clusters")
      self.goal_clusters = tf.nn.l2_normalize(goal_clusters, 1)

      self.image_summaries.append(
        tf.summary.image('observation', self.observation_image, max_outputs=30))

      ## ------ SR -------#
      with tf.variable_scope("succ_feat"):
        self.sf = layers.fully_connected(self.observation,
                                     num_outputs=self.goal_embedding_size,
                                     activation_fn=None,
                                     biases_initializer=None,
                                     scope="sf")

      with tf.variable_scope("option_manager_policy"):
        goal_hat = layers.fully_connected(self.observation,
																					num_outputs=self.goal_embedding_size,
																					activation_fn=None,
																					scope="goal_hat")
        self.query_goal = self.l2_normalize(goal_hat, 1)
        self.query_content_match = tf.einsum('bj, ij -> bi', self.query_goal, self.goal_clusters,
																						 name="query_content_match")
        # self.query_content_match_sharp = self.config.sharpening_factor * self.query_content_match
        self.attention_weights = tf.contrib.distributions.RelaxedOneHotCategorical(self.config.temperature, logits=self.query_content_match).sample()
        # self.attention_weights = tf.nn.softmax(self.query_content_match_sharp, name="attention_weights")
        self.current_unnormalized_goal = tf.einsum('bi, ij -> bj', self.attention_weights, self.goal_clusters,
                                                   name="unnormalized_g")
        self.g = tf.identity(self.l2_normalize(self.current_unnormalized_goal, 1), name="g")

      with tf.variable_scope("option_features"):
        intrinsic_features = layers.fully_connected(self.observation,
                                                    num_outputs=self.action_size * self.goal_embedding_size,
                                                    activation_fn=None,
                                                    scope="intrinsic_features")
        policy_features = tf.reshape(intrinsic_features, [-1, self.action_size,
                                                          self.goal_embedding_size],
                                     name="policy_features")
        value_features = tf.identity(intrinsic_features, name="value_features")

      with tf.variable_scope("option_value"):
        v_embedding = tf.get_variable("v_embedding",
                                          shape=[
                                            self.action_size * self.goal_embedding_size + self.goal_embedding_size,
                                            1],
                                          initializer=normalized_columns_initializer(1.0))
        v = tf.matmul(tf.concat([value_features,
                                     self.g], 1), v_embedding,
                          name="fc_option_value")
        self.v = tf.squeeze(v, 1)

      with tf.variable_scope("option_pi"):
        policy = tf.einsum('bj,bij->bi', self.g, policy_features)
        self.g_policy = tf.nn.softmax(policy, name="policy")

        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.g_policy))

      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()

  def build_losses(self):
    """Get the probabilities for each action taken under the intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.g_policy, self.actions_placeholder)

    """Building losses"""
    with tf.name_scope('sf_loss'):
      sf_td_error = self.target_sf - self.sf
      self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    with tf.name_scope('critic_loss'):
      td_error = self.target_return - self.v
      self.critic_loss = tf.reduce_mean(0.5 * tf.square(td_error))

    with tf.name_scope('entropy_loss'):
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(self.g_policy * tf.log(self.g_policy + 1e-7))
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
                          td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.critic_loss

  def l2_normalize(self, x, axis):
      norm = tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis, keepdims=True))
      return tf.maximum(x, 1e-8) / tf.maximum(norm, 1e-8)

  def cosine_similarity(self, v1, v2, axis):
    v1_norm = self.l2_normalize(v1, axis)
    v2_norm = self.l2_normalize(v2, axis)
    sim = tf.matmul(
      v1_norm, v2_norm, transpose_b=True)

    return sim

  """Build gradients for the losses with respect to the network params.
      Build summaries and update ops"""
  def gradients_and_summaries(self):
    local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    """Gradients and update ops"""
    self.grads_sf, self.apply_grads_sf = self.take_gradient(self.sf_loss)
    self.grads_option, self.apply_grads_option = self.take_gradient(self.option_loss)

    """Summaries"""
    self.merged_summary_sf = tf.summary.merge(self.image_summaries +
      self.summaries_sf + [tf.summary.scalar('SF_loss', self.sf_loss),
        gradient_summaries(zip(self.grads_sf, local_vars))])
    self.merged_summary_option = tf.summary.merge(self.summaries_option + \
                                                  [tf.summary.scalar('Entropy_loss', self.entropy_loss),
                                                   tf.summary.scalar('Policy_loss', self.policy_loss),
                                                   tf.summary.scalar('Critic_loss', self.critic_loss), ])

  def init_clustering(self):
    if self.scope == 'global':
      l = "0"
      if self.config.resume:
        checkpoint = self.config.load_from
        ckpt = tf.train.get_checkpoint_state(os.path.join(checkpoint, "models"))
        model_checkpoint_path = ckpt.model_checkpoint_path
        episode_checkpoint = os.path.basename(model_checkpoint_path).split(".")[0].split("-")[1]
        l = episode_checkpoint

      cluster_model_path = os.path.join(self.config.logdir, "cluster_models")
      self.goal_clusters_path = os.path.join(cluster_model_path, "goal_clusters_{}.pkl".format(l))

      """If the path exists, load them. Otherwise initialize all goals with zeros"""
      if os.path.exists(self.goal_clusters_path):
        self.goal_clusters = np.load(self.goal_clusters_path)
        self.goals_init = True
      else:
        self.goal_clusters = OnlineCluster(self.config.max_clusters, self.config.nb_options, self.goal_embedding_size)#np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
        self.goals_init = False
