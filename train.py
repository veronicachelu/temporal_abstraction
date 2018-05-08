import threading
import matplotlib
matplotlib.use('Agg')
import datetime
import os
import gym
import tensorflow as tf
import tools
import config_utility as utility
from env_tools import env_wrappers
import configs
from env_tools import _create_environment
from threading import Barrier, Thread

def train(config, logdir):
  tf.reset_default_graph()
  sess = tf.Session(config=tf.ConfigProto(
      allow_soft_placement=True, log_device_placement=True))
  tf.gfile.MakeDirs(logdir)
  with sess:
    with config.unlocked:
      with tf.device("/cpu:0"):
        config.logdir = logdir
        config.stage_logdir = logdir
        config.network_optimizer = getattr(tf.train, config.network_optimizer)
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        envs = [_create_environment(config) for _ in range(config.num_agents)]
        action_size = envs[0].action_space.n
        global_network = config.network("global", config, action_size)
        b = Barrier(config.num_agents)
      if FLAGS.task == "matrix":
        with tf.device("/cpu:0"):
          agent = config.target_agent(envs[0], 0, global_step, config, None)
      elif FLAGS.task == "option":
        with tf.device("/cpu:0"):
          agent = config.target_agent(envs[0], 0, global_step, config, None)
      elif FLAGS.task == "eigenoption":
        with tf.device("/cpu:0"):
          agent = config.target_agent(envs[0], 0, global_step, config, None)
      elif FLAGS.task == "eval":
        with tf.device("/cpu:0"):
          agent = config.target_agent(envs[0], 0, global_step, config, global_network)
      else:
        if config.behaviour_agent:
          with tf.device("/cpu:0"):
            agents = [config.target_agent(envs[i], i, global_step, config, global_network, b) for i in range(config.num_agents-1)]
          with tf.device("/device:GPU:0"):
            agents.append(config.behaviour_agent(envs[config.num_agents-1], "behaviour", global_step, config, global_network, b))
        else:
          with tf.device("/cpu:0"):
            agents = [config.target_agent(envs[i], i, global_step, config, global_network, b) for i in
                    range(config.num_agents)]

    saver = loader = utility.define_saver(exclude=(r'.*_temporary/.*',))
    if FLAGS.resume:
      sess.run(tf.global_variables_initializer())
      print(os.path.join(FLAGS.load_from, "models"))
      ckpt = tf.train.get_checkpoint_state(os.path.join(FLAGS.load_from, "models"))
      print("Loading Model from {}".format(ckpt.model_checkpoint_path))
      loader.restore(sess, ckpt.model_checkpoint_path)
      sess.run(tf.local_variables_initializer())
    else:
      sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

    coord = tf.train.Coordinator()


    agent_threads = []
    if FLAGS.task == "matrix":
      thread = threading.Thread(target=(lambda: agent.build_matrix(sess, coord, saver)))
      thread.start()
      agent_threads.append(thread)
    elif FLAGS.task == "option":
      thread = threading.Thread(target=(lambda: agent.plot_options(sess, coord, saver)))
      thread.start()
      agent_threads.append(thread)
    elif FLAGS.task == "eigenoption":
      thread = threading.Thread(target=(lambda: agent.viz_options(sess, coord, saver)))
      thread.start()
      agent_threads.append(thread)
    elif FLAGS.task == "eval":
      thread = threading.Thread(target=(lambda: agent.eval(sess, coord, saver)))
      thread.start()
      agent_threads.append(thread)
    else:
      for agent in agents:
        thread = threading.Thread(target=(lambda: agent.play(sess, coord, saver)))
        thread.start()
        agent_threads.append(thread)

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
  if FLAGS.load_from:
    logdir = FLAGS.logdir = FLAGS.load_from
  else:
    if FLAGS.logdir and os.path.exists(FLAGS.logdir):
      run_number = [int(f.split("-")[0]) for f in os.listdir(FLAGS.logdir) if os.path.isdir(os.path.join(FLAGS.logdir, f)) and FLAGS.config in f]
      run_number = max(run_number) + 1 if len(run_number) > 0 else 0
    else:
      run_number = 0
    logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
      FLAGS.logdir, '{}-{}'.format(run_number, FLAGS.config)))
  try:
    config = utility.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)
  train(config, logdir)


if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string(
    'logdir', './logdir',
    'Base directory to store logs.')
  tf.app.flags.DEFINE_string(
    'timestamp', datetime.datetime.now().strftime('%Y%m%dT%H%M%S'),
    'Sub directory to store logs.')
  tf.app.flags.DEFINE_string(
    'config', "eigenoc_exploration",
    'Configuration to execute.')
  tf.app.flags.DEFINE_boolean(
    'env_processes', True,
    'Step environments in separate processes to circumvent the GIL.')
  tf.app.flags.DEFINE_boolean(
    'train', True,
    'Training.')
  tf.app.flags.DEFINE_boolean(
    'resume', False,
    #'resume', True,
    'Resume.')
  tf.app.flags.DEFINE_boolean(
    'show_training', False,
    'Show gym envs.')
  tf.app.flags.DEFINE_string(
    'task', "sf",
    'Task nature')
  tf.app.flags.DEFINE_string(
    'load_from', None,
    #'load_from', "./logdir/16-eigenoc_dyn",
    'Load directory to load models from.')
  tf.app.run()
