import threading

import datetime
import os
import gym
import tensorflow as tf
import tools
import utility
from tools import wrappers
import configs


def _create_environment(config):
  if isinstance(config.env, str):
    env = gym.make(config.env)
  else:
    env = config.env()
  if config.max_length:
    env = wrappers.LimitDuration(env, config.max_length)
  env = wrappers.FrameHistoryGrayscaleResize(env)
  # env = tools.wrappers.ClipAction(env)
  env = wrappers.ConvertTo32Bit(env)
  return env


def train(config, env_processes, logdir):
  tf.reset_default_graph()
  sess = tf.Session()
  # sess_config = tf.ConfigProto(allow_soft_placement=True)
  # sess_config.gpu_options.allow_growth = True
  with sess:
    with tf.device("/cpu:0"):
      with config.unlocked:
        config.logdir = logdir
        config.network_optimizer = getattr(tf.train, config.network_optimizer)
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        envs = [_create_environment(config) for _ in range(config.num_agents)]
        action_size = envs[0].action_space.n
        global_network = config.network("global", config, action_size)
        agents = [config.agent(envs[i], i, global_step, config) for i in range(config.num_agents)]

      saver = utility.define_saver(exclude=(r'.*_temporary/.*',))
      # if FLAGS.resume:
      #   ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
      #   print("Loading Model from {}".format(ckpt.model_checkpoint_path))
      #   saver.restore(sess, ckpt.model_checkpoint_path)
      # else:
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

      coord = tf.train.Coordinator()

      agent_threads = []
      for agent in agents:
        thread = threading.Thread(target=(lambda: agent.play(sess, coord, saver)))
        thread.start()
        agent_threads.append(thread)

      while True:
        if FLAGS.show_training:
          for env in envs:
            env.render()

      coord.join(agent_threads)


def recreate_directory_structure(logdir):
  if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)
  if not FLAGS.resume and FLAGS.train:
    tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)


def main(_):
  utility.set_up_logging()
  if not FLAGS.config:
    raise KeyError('You must specify a configuration.')
  if FLAGS.logdir and os.path.exists(FLAGS.logdir):
    run_number = [int(f.split("-")[0]) for f in os.listdir(FLAGS.logdir) if os.path.isdir(os.path.join(FLAGS.logdir, f)) and FLAGS.config in f]
    run_number = max(run_number) + 1 if len(run_number) > 0 else 0
  else:
    run_number = 0
  logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
    FLAGS.logdir, '{}-{}'.format(run_number, FLAGS.config)))
  # recreate_directory_structure(logdir)
  try:
    config = utility.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)
  train(config, FLAGS.env_processes, logdir)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string(
    'logdir', None,
    'Base directory to store logs.')
  tf.app.flags.DEFINE_string(
    'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
    'Sub directory to store logs.')
  tf.app.flags.DEFINE_string(
    'config', None,
    'Configuration to execute.')
  tf.app.flags.DEFINE_boolean(
    'env_processes', True,
    'Step environments in separate processes to circumvent the GIL.')
  tf.app.flags.DEFINE_boolean(
    'train', True,
    'Training.')
  tf.app.flags.DEFINE_boolean(
    'resume', False,
    'Resume.')
  tf.app.flags.DEFINE_boolean(
    'show_training', False,
    'Show gym envs.')
  tf.app.run()
