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
  if config.history_size == 3:
    env = wrappers.FrameResize(env, config.input_size)
  else:
    env = wrappers.FrameHistoryGrayscaleResize(env, config.input_size)

  # env = tools.wrappers.ClipAction(env)
  env = wrappers.ConvertTo32Bit(env)
  return env