# Copyright 2017 The TensorFlow Agents Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example configurations using the PPO algorithm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=unused-variable

from agents import AOCAgent
import networks


def default():
  """Default configuration for PPO."""
  # General
  # algorithm = ppo.PPOAlgorithm
  num_agents = 10
  eval_episodes = 25
  use_gpu = False
  # Network
  weight_summaries = dict(
      all=r'.*',
      policy=r'.*/policy/.*',
      value=r'.*/value/.*')
  policy_layers = 200, 100
  value_layers = 200, 100
  init_mean_factor = 0.05
  init_logstd = -1
  # Optimization
  update_every = 25
  policy_optimizer = 'AdamOptimizer'
  value_optimizer = 'AdamOptimizer'
  update_epochs_policy = 50
  update_epochs_value = 50
  policy_lr = 1e-4
  value_lr = 3e-4
  # Losses
  discount = 0.985
  kl_target = 1e-2
  kl_cutoff_factor = 2
  kl_cutoff_coef = 1000
  kl_init_penalty = 1
  return locals()

def aoc():
  locals().update(default())
  agent = AOCAgent
  num_agents = 2
  use_gpu = False
  # Network
  network = networks.AOCNetwork
  weight_summaries = dict(
      all=r'.*',
      conv=r'.*/conv/.*',
      fc=r'.*/fc/.*',
      term=r'.*/option_term/.*',
      q_val=r'.*/q_val/.*',
      policy=r'.*/i_o_policies/.*')

  conv_layers = (8, 4, 16), (4, 2, 32)
  fc_layers = 256,
  # Optimization
  network_optimizer = 'AdamOptimizer'
  lr = 0.0007
  # Losses
  discount = 0.99
  entropy_coef = 0.01
  critic_coef = 0.5
  clip_gradient_value = 40
  nb_options = 2
  update_every = 2
  env = 'Breakout-v4'
  max_update_freq = 25
  min_update_freq = 5
  steps = 1e6  # 1M
  explore_steps = 1e5
  final_random_action_prob = 0.05
  initial_random_action_prob = 1.0
  delib_cost = 0
  gradient_clip_value = 50
  return locals()


def pendulum():
  """Configuration for the pendulum classic control task."""
  locals().update(default())
  # Environment
  env = 'Pendulum-v0'
  max_length = 200
  steps = 1e6  # 1M
  return locals()


def cheetah():
  """Configuration for MuJoCo's half cheetah task."""
  locals().update(default())
  # Environment
  env = 'HalfCheetah-v1'
  max_length = 1000
  steps = 1e7  # 10M
  return locals()


def walker():
  """Configuration for MuJoCo's walker task."""
  locals().update(default())
  # Environment
  env = 'Walker2d-v1'
  max_length = 1000
  steps = 1e7  # 10M
  return locals()


def reacher():
  """Configuration for MuJoCo's reacher task."""
  locals().update(default())
  # Environment
  env = 'Reacher-v1'
  max_length = 1000
  steps = 1e7  # 10M
  return locals()


def hopper():
  """Configuration for MuJoCo's hopper task."""
  locals().update(default())
  # Environment
  env = 'Walker2d-v1'
  max_length = 1000
  steps = 2e7  # 20M
  return locals()


def ant():
  """Configuration for MuJoCo's ant task."""
  locals().update(default())
  # Environment
  env = 'Ant-v1'
  max_length = 1000
  steps = 5e7  # 50M
  return locals()


def humanoid():
  """Configuration for MuJoCo's humanoid task."""
  locals().update(default())
  # Environment
  env = 'Humanoid-v1'
  max_length = 1000
  steps = 5e7  # 50M
  return locals()
