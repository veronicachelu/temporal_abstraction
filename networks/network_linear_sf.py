import tensorflow as tf
import tensorflow.contrib.layers as layers
from config_utility import gradient_summaries
from online_clustering import OnlineCluster
"""Linear function approximation network for the successor representation when states are one-hot"""
class LinearSFNetwork():
  def __init__(self, scope, config, action_size):
    self._scope = scope
    """The size of the input space flatten out"""
    self.nb_states = config.input_size[0] * config.input_size[1]
    self.config = config
    self.network_optimizer = config.network_optimizer(
      self.config.lr, name='network_optimizer')

    self.init_clustering()
    with tf.variable_scope(scope):
      self.observation = tf.placeholder(shape=[None, self.nb_states],
                                        dtype=tf.float32, name="Inputs")
      self.sf = layers.fully_connected(self.observation, num_outputs=self.nb_states,
                                       activation_fn=None,
                                       variables_collections=tf.get_collection("variables"),
                                       outputs_collections="activations", scope="sf")
      if scope != 'global':
        self.target_sf = tf.placeholder(shape=[None, self.nb_states], dtype=tf.float32, name="target_sf")

        with tf.name_scope('sf_loss'):
          sf_td_error = self.target_sf - self.sf
          self.loss = tf.reduce_mean(tf.square(sf_td_error))

        loss_summaries = [tf.summary.scalar('avg_sf_loss', self.loss)]

        local_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        gradients = tf.gradients(self.loss, local_vars)
        grads, grad_norms = tf.clip_by_global_norm(gradients, self.config.gradient_clip_norm_value)

        self.merged_summary = tf.summary.merge(loss_summaries + [
          gradient_summaries(zip(grads, local_vars))])
        global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
        self.apply_grads = self.network_optimizer.apply_gradients(zip(grads, global_vars))

  def init_clustering(self):
    self.direction_clusters = OnlineCluster(4, self.nb_states)#np.zeros((self.config.nb_options, self.config.sf_layers[-1]))
