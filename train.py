import threading

import datetime
import functools
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


def _define_loop(graph, logdir, train_steps, eval_steps):
  """Create and configure a training loop with training and evaluation phases.

  Args:
    graph: Object providing graph elements via attributes.
    logdir: Log directory for storing checkpoints and summaries.
    train_steps: Number of training steps per epoch.
    eval_steps: Number of evaluation steps per epoch.

  Returns:
    Loop object.
  """
  loop = tools.Loop(
    logdir, graph.step, graph.should_log, graph.do_report,
    graph.force_reset)
  loop.add_phase(
    'train', graph.done, graph.score, graph.summary, train_steps,
    report_every=None,
    log_every=train_steps // 2,
    checkpoint_every=None,
    feed={graph.is_training: True})
  loop.add_phase(
    'eval', graph.done, graph.score, graph.summary, eval_steps,
    report_every=eval_steps,
    log_every=eval_steps // 2,
    checkpoint_every=10 * eval_steps,
    feed={graph.is_training: False})
  return loop


def train(config, env_processes):
  """Training and evaluation entry point yielding scores.

  Resolves some configuration attributes, creates environments, graph, and
  training loop. By default, assigns all operations to the CPU.

  Args:
    config: Object providing configurations via attributes.
    env_processes: Whether to step environments in separate processes.

  Yields:
    Evaluation scores.
  """
  tf.reset_default_graph()

  sess_config = tf.ConfigProto(allow_soft_placement=True)
  sess_config.gpu_options.allow_growth = True
  with tf.Session(config=sess_config) as sess:
    with config.unlocked:
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
  sess.run(tf.global_variables_initializer())

  coord = tf.train.Coordinator()

  agent_threads = []
  for agent in agents:
    thread = threading.Thread(target=(lambda: agent.play(sess, coord, saver)))
    thread.start()
    agent_threads.append(thread)

  # while True:
  #   if FLAGS.show_training:
  #     for env in envs:
  #       # time.sleep(1)
  #       # with main_lock:
  #       env.render()

  # coord.join(agent_threads)
    # total_steps = int(
    #     config.steps / config.update_every *
    #     (config.update_every + config.eval_episodes))
    #
    # utility.initialize_variables(sess, saver, config.logdir)
    # for score in loop.run(sess, saver, total_steps):
    #   yield score


def main(_):
  """Create or load configuration and launch the trainer."""
  utility.set_up_logging()
  if not FLAGS.config:
    raise KeyError('You must specify a configuration.')
  logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
    FLAGS.logdir, '{}-{}'.format(FLAGS.timestamp, FLAGS.config)))
  try:
    config = utility.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)
  train(config, FLAGS.env_processes)
  # for score in train(config, FLAGS.env_processes):
  #   tf.logging.info('Score {}.'.format(score))


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
  tf.app.run()
