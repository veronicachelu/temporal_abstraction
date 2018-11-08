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

"""Function approximation network for the option critic policies and value functions when options are given as embeddings corresponding to the spectral decomposition of the SR matrix"""
class AttentionNetwork(EignOCNetwork):
  def __init__(self, scope, config, action_size):
    self.goal_embedding_size = config.sf_layers[-1]
    super(AttentionNetwork, self).__init__(scope, config, action_size)

    if self.config.use_clustering:
      self.init_clustering()


  def build_losses(self):
    """Get the probabilities for each action taken under the intra-option policy"""
    self.responsible_actions = self.get_responsible_actions(self.option_policy, self.actions_placeholder)

    """Adding comparison of predicted frame and actual next frame to tensorboard"""
    # self.image_summaries.append(
    #   tf.summary.image('next', tf.concat([self.next_obs, self.target_next_obs], 2), max_outputs=30))
		#
    """Building losses"""
    with tf.name_scope('sf_loss'):
      """TD error of successor representations"""
      sf_td_error = self.target_sf - self.sf
      self.sf_loss = tf.reduce_mean(self.config.sf_coef * huber_loss(sf_td_error))

    """If we use eigendirections for the options, than do TD on the eigen intra-option critics"""
    with tf.name_scope('mix_critic_loss'):
      """Zero out where the option was a primitve one"""
      mix_td_error = self.target_mix_return - self.value_mix
      self.mix_critic_loss = tf.reduce_mean(0.5 * self.config.eigen_critic_coef * tf.square(mix_td_error))

    """Add an entropy regularization for each intra-option policy, driving exploration in the action space of intra-option policies"""
    with tf.name_scope('entropy_loss'):
      """Zero out primitive options"""
      self.entropy_loss = -self.entropy_coef * tf.reduce_mean(self.option_policy * tf.log(self.option_policy + 1e-7))
    """Learn intra-option policies with policy gradients"""
    with tf.name_scope('policy_loss'):
      self.policy_loss = -tf.reduce_mean(tf.log(self.responsible_actions + 1e-7) * tf.stop_gradient(
                          mix_td_error))

    self.option_loss = self.policy_loss - self.entropy_loss + self.mix_critic_loss

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
    self.merged_summary_sf = tf.summary.merge(
      self.summaries_sf + [tf.summary.scalar('avg_sf_loss', self.sf_loss),
        gradient_summaries(zip(self.grads_sf, local_vars))])
    options_to_merge = self.summaries_option +\
                       [tf.summary.scalar('avg_entropy_loss', self.entropy_loss),
                        tf.summary.scalar('avg_policy_loss', self.policy_loss),
                        tf.summary.scalar('avg_eigen_critic_loss', self.mix_critic_loss), ]

    self.merged_summary_option = tf.summary.merge(options_to_merge)


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
      self.direction_clusters_path = os.path.join(cluster_model_path, "direction_clusters_{}.pkl".format(l))

      """If the path exists, load them. Otherwise initialize all directions with zeros"""
      if os.path.exists(self.direction_clusters_path):
        self.direction_clusters = np.load(self.direction_clusters_path)
        self.directions_init = True
      else:
        self.direction_clusters = OnlineCluster(self.config.nb_options, self.goal_embedding_size)#np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
        self.directions_init = False

  def build_network(self):
    with tf.variable_scope(self.scope):
      self.target_sf = tf.placeholder(shape=[None, self.config.sf_layers[-1]], dtype=tf.float32, name="target_SF")
      self.target_mix_return = tf.placeholder(shape=[None], dtype=tf.float32)
      self.actions_placeholder = tf.placeholder(shape=[None], dtype=tf.int32, name="Actions")
      self.observation = tf.placeholder(
        shape=[None, self.nb_states],
        dtype=tf.float32, name="Inputs")

      with tf.variable_scope("succ_feat"):
        self.sf = layers.fully_connected(self.observation,
                                     num_outputs=self.goal_embedding_size,
                                     activation_fn=None,
                                     biases_initializer=None,
                                     variables_collections=tf.get_collection("variables"),
                                     outputs_collections="activations", scope="sf")
      direction_clusters = tf.placeholder(shape=[self.config.nb_options,
                                                      self.goal_embedding_size],
                                               dtype=tf.float32,
                                               name="direction_clusters")
      self.direction_clusters = tf.nn.l2_normalize(direction_clusters, 1)

      with tf.variable_scope("option_features"):
        intra_features = layers.fully_connected(self.observation,
                                                num_outputs=self.action_size * self.goal_embedding_size,
                                                activation_fn=None,
                                                variables_collections=tf.get_collection("variables"),
                                                outputs_collections="activations", scope="U")
        self.policy_features = tf.reshape(intra_features, [-1, self.action_size,
                                                           self.goal_embedding_size],
                                          name="policy_features")
        self.value_features = tf.identity(intra_features, name="value_features")

      with tf.variable_scope("option_policy"):
        direction_features = layers.fully_connected(self.observation,
                                                    num_outputs=self.goal_embedding_size,
                                                    activation_fn=None,
                                                    variables_collections=tf.get_collection("variables"),
                                                    outputs_collections="activations", scope="direction_features")
        self.query_direction = self.l2_normalize(direction_features, 1)

        self.query_content_match = tf.tensordot(self.query_direction, self.direction_clusters, axes=[[1], [1]], name="query_content_match")
        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.query_content_match))

        self.attention_weights = tf.nn.softmax(self.query_content_match, name="attention_weights")
        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.attention_weights))

        self.current_unnormalized_direction = tf.tensordot(self.attention_weights, self.direction_clusters, axes=[[1], [0]])

        self.current_option_direction = tf.identity(self.l2_normalize(self.current_unnormalized_direction, 1), name="current_option_direction")
        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.current_option_direction))

      with tf.variable_scope("eigen_option_q_val"):
        value_embedding = tf.get_variable("value_embedding",
                                          shape=[
                                            self.action_size * self.goal_embedding_size + self.goal_embedding_size,
                                            1],
                                          initializer=normalized_columns_initializer(1.0))
        self.value_mix = tf.matmul(tf.concat([self.value_features,
                                              self.current_option_direction], 1),
                                   value_embedding,
                                   name="fc_option_value")
        self.value_mix = tf.squeeze(self.value_mix, 1)

      with tf.variable_scope("option_pi"):
        policy = tf.einsum('bj,bij->bi', self.current_option_direction, self.policy_features)
        self.option_policy = tf.nn.softmax(policy, name="policy")

        self.summaries_option.append(tf.contrib.layers.summarize_activation(self.option_policy))


      if self.scope != 'global':
        self.build_losses()
        self.gradients_and_summaries()