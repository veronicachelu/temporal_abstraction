import threading
import matplotlib

matplotlib.use('Agg')
import os
import tensorflow as tf
import tools
import config_utility as utility
import configs
from env_tools import _create_environment
from threading import Barrier, Thread
from tools.rmsprop_applier import RMSPropApplier


def run(config, logdir):
  """Reset the graph."""
  tf.reset_default_graph()

  """Create a global session."""
  from tensorflow.python import debug as tf_debug
  sess = tf.Session(config=tf.ConfigProto(
    allow_soft_placement=True, log_device_placement=False))
  # sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type='curses')
  # sess = tf_debug.TensorBoardDebugWrapperSession(
  #   sess, "localhost:2333")

  """Make log directory if it does not exist."""
  tf.gfile.MakeDirs(logdir)

  with sess:
    with config.unlocked:
      """Run everything on cpu."""
      with tf.device("/cpu:0"):

        """Add some tensorflow flag information to the config."""
        config.logdir = logdir
        config.resume = FLAGS.resume
        config.load_from = FLAGS.load_from

        """Instantiated the network optimizer."""
        config.network_optimizer = getattr(tf.train, config.network_optimizer)

        """Create global_step and global_episode vars."""
        global_step = tf.Variable(0, dtype=tf.int32, name='global_step', trainable=False)
        global_episode = tf.Variable(0, dtype=tf.int32, name='global_episode', trainable=False)

        """Create environments for each agent."""
        envs = [_create_environment(config) for _ in range(config.num_agents)]
        action_size = envs[0].action_space.n

        """Create the shared global network which agents will use to aggregate gradients and propagate weights back to 
          their local networks."""
        global_network = config.network("global", config, action_size)

        """Creating global barrier used to stop all agents at a specific episode before switching to the next task.
        Only used when the goal_locations from the config file contains more than one location, corresponding to the 
        continual task scenarios in which the reward signal is constantly changing over multiple tasks."""
        b = Barrier(config.num_agents)

      """Switching between multiple tasks."""
      if FLAGS.task == "build_sr_matrix" or FLAGS.task == "cluster":
        with tf.device("/cpu:0"):
          agents = [config.target_agent(sess, envs[i], i, global_step, global_episode, config, global_network, b) for i
                    in
                    range(config.num_agents)]
          # agent = config.target_agent(sess, envs[0], 0, global_step, global_episode, config, global_network, b)
      elif FLAGS.task == "plot_options":
        with tf.device("/cpu:0"):
          agent = config.target_agent(sess, envs[0], 0, global_step, global_episode, config, global_network, b)
      elif FLAGS.task == "eigenoption":
        with tf.device("/cpu:0"):
          agent = config.target_agent(sess, envs[0], 0, global_step, global_episode, config, global_network, b)
      elif FLAGS.task == "evaluate":
        with tf.device("/cpu:0"):
          agent = config.target_agent(sess, envs[0], 0, global_step, global_episode, config, global_network, b)
      elif FLAGS.task == "train":
        """The classical task of training agents in the environment."""
        with tf.device("/cpu:0"):
          agents = [config.target_agent(sess, envs[i], i, global_step, global_episode, config, global_network, b) for i in
                    range(config.num_agents)]

    """Construct saver and loader of weights."""
    saver = loader = utility.define_saver(exclude=(r'.*_temporary/.*', r'beta*/.*'))

    """Initialize vars."""
    utility.initialize_variables(sess, loader, checkpoint=FLAGS.load_from, resume=FLAGS.resume)

    """Create thread coordinator."""
    coord = tf.train.Coordinator()

    agent_threads = []

    """Construct one or more threads according to how many agents we need, dependent on the task."""
    if FLAGS.task == "build_sr_matrix":
      thread = threading.Thread(target=(lambda: agent.build_SR_matrix()))
      thread.start()
      agent_threads.append(thread)
    elif FLAGS.task == "cluster":
      for agent in agents:
        thread = threading.Thread(target=(lambda: agent.cluster(coord)))
        thread.start()
        agent_threads.append(thread)
    elif FLAGS.task == "plot_options":
      thread = threading.Thread(target=(lambda: agent.plot_high_level_directions(coord, saver)))
      thread.start()
      agent_threads.append(thread)
    elif FLAGS.task == "eigenoption":
      thread = threading.Thread(target=(lambda: agent.viz_options2(sess, coord, saver)))
      thread.start()
      agent_threads.append(thread)
    elif FLAGS.task == "evaluate":
      thread = threading.Thread(target=(lambda: agent.evaluate(coord, saver)))
      thread.start()
      agent_threads.append(thread)
    elif FLAGS.task == "train":
      """This is the classical task of training. Starting threads in parallel for all agents"""
      for agent in agents:
        thread = threading.Thread(target=(lambda: agent.play(coord, saver)))
        thread.start()
        agent_threads.append(thread)

    """Join all threads and exit."""
    coord.join(agent_threads)


def recreate_directory_structure(logdir):
  if not tf.gfile.Exists(logdir):
    tf.gfile.MakeDirs(logdir)
  if not FLAGS.resume and FLAGS.train:
    tf.gfile.DeleteRecursively(logdir)
    tf.gfile.MakeDirs(logdir)


"""Main entry point."""
def main(_):

  """Configure logging."""
  utility.set_up_logging()

  """Assert configuration and set-up directory log structure of the configuration."""
  if not FLAGS.config:
    raise KeyError('You must specify a configuration.')
  if FLAGS.load_from:
    logdir = FLAGS.logdir = FLAGS.load_from
  else:
    """If config log directory already exists, increase the counter number and setup log dir."""
    if FLAGS.logdir and os.path.exists(FLAGS.logdir):
      run_number = [int(f.split("-")[0]) for f in os.listdir(FLAGS.logdir) if
                    os.path.isdir(os.path.join(FLAGS.logdir, f)) and FLAGS.config in f]
      run_number = max(run_number) + 1 if len(run_number) > 0 else 0
    else:
      run_number = 0
    logdir = FLAGS.logdir and os.path.expanduser(os.path.join(
      FLAGS.logdir, '{}-{}'.format(run_number, FLAGS.config)))
  """If config log directory already exists, try to load config file from it. Otherwise create a new config file 
  coresponding to the user specified config from the config.py"""
  try:
    config = utility.load_config(logdir)
  except IOError:
    config = tools.AttrDict(getattr(configs, FLAGS.config)())
    config = utility.save_config(config, logdir)

  """Run the task specified."""
  run(config, logdir)


"""Command prompt argument configuration."""
if __name__ == '__main__':
  FLAGS = tf.app.flags.FLAGS
  tf.app.flags.DEFINE_string(
    'logdir', './logdir',
    'Base directory to store logs.')
  tf.app.flags.DEFINE_string(
    'config', "lstm",
    'Configuration to execute. Configuration details can be found in the config.py file.')
  tf.app.flags.DEFINE_boolean(
    'train', True,
    'Train = True (training), train = False (eval when appropriate)"')
  tf.app.flags.DEFINE_boolean(
    'resume', False,
    'resume = True (training), resume = False (resuming from checkpoint model, please specify '
    'checkpoint configuration with "load_from=./logdir/0-default")"')
  tf.app.flags.DEFINE_string(
    'task', "train",
    'Task nature: choose from: "train", ')
  tf.app.flags.DEFINE_string(
    'load_from', None,
    # 'load_from', "./logdir/3-lstm",
    'Directory of the configuration to load models from and resume training.')
  tf.app.run()
