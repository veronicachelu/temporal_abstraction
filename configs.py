from agents import EmbeddingAgent
from agents import AttentionAgent
from agents import AttentionWTermAgent
from agents import AttentionFeudalAgent
from agents import EigenOCAgent
from agents import LinearSFAgent
from agents import EigenOCAgentDyn
from agents import DynSRAgent
from agents import LSTMAgent
from env_tools import GridWorld
import functools
from networks import EmbeddingNetwork
from networks import AttentionNetwork
from networks import AttentionWTermNetwork
from networks import AttentionFeudalNetwork
from networks import EignOCNetwork
from networks import LinearSFNetwork
from networks import DynSRNetwork
from networks import LSTMNetwork


def default():
	"""The number of agents to run asyncronously"""
	num_agents = 2

	"""The size of the observation space"""
	input_size = (13, 13)
	"""If the history size is 3 than it uses rgb channels. Otherwise, if  > 3 it stacks the last number of grayscale frames"""
	history_size = 3

	"""Which optimizer to use"""
	network_optimizer = 'AdamOptimizer'
	"""Learning rate"""
	lr = 0.0001
	"""Discount factor"""
	discount = 0.99

	"""Weighting coeficients for individual losses when summing out the final loss of the agent"""
	sf_coef = 1 # corresponds to the TD error of the successor representation component of the loss
	aux_coef = 1 # corresponds to the auxilary loss of next frame prediction
	entropy_coef = 0.01 # corresponds to the policy entropy maximization for exploration purposes
	critic_coef = 1 # corresponds to the TD error of the action value function of the policy over options
	eigen_critic_coef = 1 # corresponds to the TD error of the critic of the intra-option policies

	"""Number of iteration between the updates of the local thread-agent networks from the global network"""
	target_update_iter_aux = 1 # update the local network parameters corresponding to the auxilary next frame prediction loss
	target_update_iter_sf = 30 # update the successor representation parameters of the local networks
	target_update_iter_option = 30 # update the options parameters of the local network

	"""The goal positions corresponding the the specific reward position on the grid. In the case of multiple elements in the list this corresponds to the continual learning case in which the reward signal is in constant change every "move_goal_nb_of_ep" episodes"""
	goal_locations = [(11, 7), (5, 2)]  # , (1, 10), (2, 2), (6, 2)]
	# goal_locations = [(1, 11), (3, 2)]

	"""Move to the next task specified in the goal_locations after the specfied number of episodes"""
	move_goal_nb_of_ep = 500

	"""The environment in which to run experiments. Experiments were performed in the 4 Rooms domain"""
	env = functools.partial(
		GridWorld, goal_locations, "./mdps/4rooms.mdp")

	"""How many steps to take in the environment before bootstraping the with the target estimation and doing the updates"""
	max_update_freq = 30 # this corresponds to taking a maximum of 30-step returns
	min_update_freq = 5 # in cases where the current option terminates faster, do a minimum of 5-step returns
	aux_update_freq = 1 # the update of next frame prediction is done every step

	"""The maximum number of steps to execute in the environment"""
	steps = -1  # 10M

	"""The probability of taking the random option in the environment, instead of the gready option. """
	final_random_option_prob = 0.1
	initial_random_option_prob = 0.1
	decrease_option_prob = False # if False the random option probability is set to the initial value and is constant
	explore_options_episodes = 2000 # the number of episodes under which to perform the deacrease of the option probabilities

	"""The probability of executing a random action in the environment, entripy coefficient """
	final_random_action_prob = 0.01

	"""The number test episodes to execute, over which to average results"""
	nb_test_ep = 100

	"""The maximum length of episodes in the environment"""
	max_length = 3000

	"""Clipping the gradient norm"""
	gradient_clip_norm_value = 40
	"""Option to clip the gradient by value. Not sure it still works."""
	clip_option_grad_by_value = False
	clip_by_value = 5

	"""Whether to include option corresponding to primitive actions in the environment or keep only temporally abstract ones"""
	include_primitive_options = True

	"""Summary intervals every specific number of steps"""
	summary_interval = 100
	checkpoint_interval = 100
	step_summary_interval = 1000
	# episode_eval_interval = 10

	"""Whether to print logs in the environment"""
	logging = False

	"""Whether to perform evaluation. Pretty costly"""
	evaluation = False

	"""Flags for performing multi task continual learning scenario. Mostly added for switching in case of running the experiments on any Atari environments"""
	multi_task = True

	return locals()


def linear_sf():
	"""Load configuration options from default and override or add new ones"""
	locals().update(default())

	"""The kind of agent to use in the environment"""
	target_agent = LinearSFAgent
	"""The number of agents to run asyncronously"""
	num_agents = 2

	"""The kind of network to use for function approximation"""
	network = LinearSFNetwork

	"""Learning rate"""
	lr = 1e-3
	"""Discount factor"""
	discount = 0.985

	"""The maximum number of steps to execute in the environment"""
	steps = -1  # 1M

	"""The interval of timesteps to wait between each summary tensorboard update"""
	summary_interval = 10
	"""The insterval of timesteps to wait between each checkpoint save"""
	checkpoint_interval = 1
	"""The interval of timesteps to wait between each evaluation"""
	eval_interval = 1

	"""In the case of dynamic computation of eigendirections this corresponds to the number of successor representation/features matrix buffer size"""
	sf_transition_matrix_size = 1e3

	return locals()


def dynamic_SR():
	"""Load configuration options from default and override or add new ones"""
	locals().update(default())

	"""The kind of agent to use in the environment"""
	target_agent = DynSRAgent
	"""The number of agents to run asyncronously"""
	num_agents = 2

	"""The kind of network to use for function approximation"""
	network = DynSRNetwork

	"""If the history size is 1 than it uses grayscale images. Otherwise, if  > 1 it stacks the last number of grayscale frames"""
	history_size = 1

	"""Configuration for the neural network"""
	fc_layers = 128, # the number of layers and units in each layer mapping from input space to latent state representation
	sf_layers = 128, # the number of layers and units mapping from the latent representation obtained from next frame prediction to the latent successor representation space
	aux_fc_layers = 169, # the number of layers and units in the decoder network mapping the latent space to the predicted next frame

	"""Learning rate"""
	lr = 1e-3
	"""Discount factor"""
	discount = 0.985


	batch_size = 16

	"""The size of the experience replay buffer"""
	memory_size = 500000
	"""Warm start of training after the observation_steps"""
	observation_steps = 100

	"""The maximum number of steps to execute in the environment"""
	steps = -1  # 10M

	"""The interval of timesteps to wait between each summary tensorboard update"""
	summary_interval = 10
	"""The insterval of timesteps to wait between each checkpoint save"""
	checkpoint_interval = 1



	return locals()


def oc():
	"""Load configuration options from default and override or add new ones"""
	locals().update(default())
	"""The number of options to use"""
	nb_options = 4

	"""The number of agents to run asyncronously"""
	num_agents = 2

	"""If the history size is 1 than it uses grayscale images. Otherwise, if  > 1 it stacks the last number of grayscale frames"""
	history_size = 1

	"""The kind of agent to use in the environment"""
	target_agent = EigenOCAgent

	"""Flag which switches between the use of guidance from the latent successor representation space or not. If set to False this defaults to the classic option critic"""
	use_eigendirections = False

	"""The kind of network to use for function approximation"""
	network = EignOCNetwork

	"""Configuration for the neural network"""
	fc_layers = 128,	 # the number of layers and units in each layer mapping from input space to latent state representation
	sf_layers = 128, 	# the number of layers and units mapping from the latent representation obtained from next frame prediction to the latent successor representation space
	aux_fc_layers = 13 * 13,	 # the number of layers and units in the decoder network mapping the latent space to the predicted next frame

	"""Learning rate"""
	lr = 1e-3

	"""When performing next frame predition with experience replay buffer, this is the size of the batch size used for sampling the buffer"""
	batch_size = 16

	"""The size of the experience replay buffer"""
	memory_size = 100000

	"""Warm start of training after the observation_steps"""
	observation_steps = 16 * 4

	"""The maximum number of steps to execute in the environment"""
	steps = -1  # 1M

	"""The maximum length of episodes in the environment"""
	max_length = 1000

	"""The maximum length of episodes in the environment when performing evaluation"""
	max_length_eval = 1000

	"""The size of the successor matrix. If it it larger than the state space it is done dynamically with a buffer"""
	sf_matrix_size = 169

	"""Whether to use the successor matrix buffer or construct the matrix statically for each state of the 169"""
	sr_matrix = None

	"""Whether to include primitive actions as being options"""
	include_primitive_options = False

	"""The goal positions corresponding the the specific reward position on the grid. In the case of multiple elements in the list this corresponds to the continual learning case in which the reward signal is in constant change every "move_goal_nb_of_ep" episodes"""
	goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2)]
	episodes = len(goal_locations)

	"""Move to the next task specified in the goal_locations after the specfied number of episodes"""
	move_goal_nb_of_ep = 1000

	"""The environment in which to run experiments. Experiments were performed in the 4 Rooms domain"""
	env = functools.partial(
		GridWorld, goal_locations, "./mdps/4rooms.mdp")

	"""Discount factor"""
	discount = 0.985

	"""The probability of taking the random option in the environment, instead of the gready option. """
	final_random_option_prob = 0.1
	initial_random_option_prob = 0.1
	decrease_option_prob = False # if False the random option probability is set to the initial value and is constant
	explore_options_episodes = 2000 # the number of episodes under which to perform the deacrease of the option probabilities

	"""The margin that the current option has to be better than the expected value of the state in order keep it and not increase the probability of termination"""
	delib_margin = 0.0
	"""The deliberation cost we have to pay at each time step for terminating options"""
	delib_cost = 0.0

	"""Summary intervals every specific number of steps"""
	summary_interval = 100
	step_summary_interval = 1000
	checkpoint_interval = 100

	return locals()


def eigenoc():
	"""Load configuration options from default and override or add new ones"""
	locals().update(default())
	"""The nuber of options to use"""
	nb_options = 8

	"""If the history size is 1 than it uses grayscale images. Otherwise, if  > 1 it stacks the last number of grayscale framefs"""
	history_size = 1

	"""The kind of agent to use in the environment"""
	target_agent = EigenOCAgent

	"""Flag which switches between the use of guidance from the latent successor representation space or not. If set to False this defaults to the classic option critic"""
	use_eigendirections = True

	"""The kind of network to use for function approximation"""
	network = EignOCNetwork

	"""Configuration for the neural network"""
	fc_layers = 128,  # the number of layers and units in each layer mapping from input space to latent state representation
	sf_layers = 128,  # the number of layers and units mapping from the latent representation obtained from next frame prediction to the latent successor representation space
	aux_fc_layers = 13 * 13,  # the number of layers and units in the decoder network mapping the latent space to the predicted next frame

	"""Learning rate"""
	lr = 1e-3

	"""Weighting coeficients for individual losses when summing out the final loss of the agent"""
	sf_coef = 1  # corresponds to the TD error of the successor representation component of the loss

	"""When performing next frame predition with experience replay buffer, this is the size of the batch size used for sampling the buffer"""
	batch_size = 16

	"""The size of the experience replay buffer"""
	memory_size = 100000

	"""Warm start of training after the observation_steps"""
	observation_steps = 16 * 4

	"""The maximum number of steps to execute in the environment"""
	steps = -1  # 1M

	"""The weight put on the internal reward signal when constructing the mixture reward used to train the intra-option policies."""
	alpha_r = 0.75

	"""The maximum length of episodes in the environment"""
	max_length = 1000

	"""The maximum length of episodes in the environment when performing evaluation"""
	max_length_eval = 1000

	"""From which eigenvector to start using the eigen directions as basis for the options."""
	first_eigenoption = 1

	"""Whether to include primitive actions as being options"""
	include_primitive_options = True

	"""The size of the successor matrix. If it it larger than the state space it is done dynamically with a buffer"""
	sf_matrix_size = 169

	"""Wheather to use a dynamic buffer for the SR matrix or to construct it statically for the whole environment"""
	sr_matrix = "static"

	"""The goal positions corresponding the the specific reward position on the grid. In the case of multiple elements in the list this corresponds to the continual learning case in which the reward signal is in constant change every "move_goal_nb_of_ep" episodes"""
	goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2)]
	episodes = len(goal_locations)

	"""Move to the next task specified in the goal_locations after the specfied number of episodes"""
	move_goal_nb_of_ep = 1000

	"""The environment in which to run experiments. Experiments were performed in the 4 Rooms domain"""
	env = functools.partial(
		GridWorld, goal_locations, "./mdps/4rooms.mdp")

	"""Discount factor"""
	discount = 0.985

	"""The probability of taking the random option in the environment, instead of the gready option. """
	final_random_option_prob = 0.1
	initial_random_option_prob = 0.1
	decrease_option_prob = False # if False the random option probability is set to the initial value and is constant

	"""The margin that the current option has to be better than the expected value of the state in order keep it and not increase the probability of termination"""
	delib_margin = 0.005
	"""The deliberation cost we have to pay at each time step for terminating options"""
	delib_cost = 0.005

	return locals()


def eigenoc_dyn():
	"""Load configuration options from eigenoc agent and override or add new ones"""
	locals().update(eigenoc())

	"""The nuber of options to use"""
	nb_options = 8

	"""The kind of agent to use in the environment"""
	target_agent = EigenOCAgentDyn

	"""The size of the successor matrix. If it it larger than the state space it is done dynamically with a buffer"""
	sf_matrix_size = 5000

	"""Wheather to use a dynamic buffer for the SR matrix or to construct it statically for the whole environment"""
	sr_matrix = "dynamic"

	"""Whether to include primitive actions as being options"""
	include_primitive_options = True

	"""Flag which switches between the use of guidance from the latent successor representation space or not. If set to False this defaults to the classic option critic"""
	use_eigendirections = True

	return locals()


def embedding():
	"""Load configuration options from eigenoc_dyn and override or add new ones"""
	locals().update(eigenoc_dyn())

	"""The kind of agent to use in the environment"""
	target_agent = EmbeddingAgent

	"""The kind of network to use for function approximation"""
	network = EmbeddingNetwork

	return locals()

def attention():
	"""Load configuration options from eigenoc_dyn and override or add new ones"""
	locals().update(eigenoc_dyn())
	fc_layers = 169,  # the number of layers and units in each layer mapping from input space to latent state representation
	sf_layers = 169,
	nb_options = 8
	num_agents = 8
	"""The maximum length of episodes in the environment"""
	max_length = 1000
	max_clusters = 32
	test_random_action = False
	sr_matrix = None
	use_eigendirections = False
	use_clustering = True
	temperature = 1e-5
	summary_interval = 1
	checkpoint_interval = 1
	cluster_interval = 1
	cold_start_episodes = 10
	"""The kind of agent to use in the environment"""
	target_agent = AttentionAgent
	"""The number test episodes to execute, over which to average results"""
	nb_test_ep = 1
	"""Move to the next task specified in the goal_locations after the specfied number of episodes"""
	move_goal_nb_of_ep = 50
	"""The kind of network to use for function approximation"""
	network = AttentionNetwork
	goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2), (9, 11), (2, 7)]

	return locals()

def attention_w_term():
	"""Load configuration options from eigenoc_dyn and override or add new ones"""
	locals().update(attention())
	fc_layers = 169,  # the number of layers and units in each layer mapping from input space to latent state representation
	sf_layers = 169,
	nb_options = 8
	num_agents = 8
	"""The maximum length of episodes in the environment"""
	max_length = 1000
	max_clusters = 32
	test_random_action = False
	sr_matrix = None
	use_eigendirections = False
	use_clustering = True
	summary_interval = 1
	checkpoint_interval = 1
	cluster_interval = 1
	cold_start_sf_episodes = 10
	"""The kind of agent to use in the environment"""
	target_agent = AttentionWTermAgent
	"""The number test episodes to execute, over which to average results"""
	nb_test_ep = 1
	"""Move to the next task specified in the goal_locations after the specfied number of episodes"""
	move_goal_nb_of_ep = 50
	"""The kind of network to use for function approximation"""
	network = AttentionWTermNetwork
	goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2), (9, 11), (2, 7)]

	return locals()

def attention_feudal():
	"""Load configuration options from eigenoc_dyn and override or add new ones"""
	locals().update(attention())
	fc_layers = 169,  # the number of layers and units in each layer mapping from input space to latent state representation
	sf_layers = 169,
	nb_options = 4
	num_agents = 8
	"""The maximum length of episodes in the environment"""
	max_length = 1000
	max_clusters = 32
	test_random_action = False
	sr_matrix = None
	use_eigendirections = False
	use_clustering = True
	summary_interval = 1
	checkpoint_interval = 1
	cluster_interval = 1
	initial_random_goal_prob = 0.1
	final_random_goal_prob = 0
	temperature = 1e-5
	c = 2
	cold_start_episodes = 10
	"""The kind of agent to use in the environment"""
	target_agent = AttentionFeudalAgent
	"""The number test episodes to execute, over which to average results"""
	nb_test_ep = 1
	"""Move to the next task specified in the goal_locations after the specfied number of episodes"""
	move_goal_nb_of_ep = 50
	"""The kind of network to use for function approximation"""
	network = AttentionFeudalNetwork
	goal_locations = [(11, 7), (5, 2), (1, 10), (2, 2), (6, 2), (9, 11), (2, 7)]

	return locals()


def lstm():
	"""Load configuration options from embedding and override or add new ones"""
	locals().update(embedding())

	"""The kind of agent to use in the environment"""
	target_agent = LSTMAgent

	"""The kind of network to use for function approximation"""
	network = LSTMNetwork

	return locals()
