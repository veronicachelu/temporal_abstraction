import argparse
import tensorflow as tf
from tensorflow.python.summary import summary_iterator as ea

from matplotlib import pyplot as plt
from matplotlib import colors as colors
import seaborn as sns
import numpy as np
import os
import random
sns.set(style="darkgrid")
sns.set_context("paper")

def plot(params):
  ''' beautify tf log
      Use better library (seaborn) to plot tf event file'''
  comp_exp = ['direction_option_critic_dyn_exploration', 'option_critic']
  log_path = params['logdir']
  experiments = [os.path.join(log_path, f) for f in os.listdir(log_path)]
  exps = []
  exp_names = []
  for experiment in experiments:
    name_exp = os.path.basename(experiment)
    if name_exp in comp_exp:
      exp_names.append(name_exp)
      log_files = [os.path.join(experiment, f) for f in os.listdir(experiment)]
      # smooth_space = params['smooth']
      color_code = params['color']

      y_list = []
      x_list = []
      for event_log in log_files:
        x = []
        y = []
        for e in tf.train.summary_iterator(event_log):
          if len(x) == 500:
            break
          for v in e.summary.value:
            if v.tag == 'Perf/Length':
              x.append(e.step)
              y.append(v.simple_value)
        x_list.append(x)
        y_list.append(y)

      y_array = np.array(y_list)
      y_mean = np.mean(y_array, axis=0)
      y_std = np.std(y_array, axis=0)

      exps.append((x, y_mean, y_std))

  plt.figure(0)
  plt.subplot(111)
  plt.title('Episode length')
  for i in range(len(exps)):
    color_code = np.random.rand(3,)
    plt.plot(exps[i][0], exps[i][1], label=exp_names[i], color=color_code, linewidth=1.5)
    max = exps[i][1]+exps[i][2]
    min = exps[i][1]-exps[i][2]
    min[min < 0] = 0
    plt.fill_between(exps[i][0], max, min, color=color_code, alpha=0.4)

  plt.legend()
  plt.show()

  # x_list = []
  # y_list = []
  # x_list_raw = []
  # y_list_raw = []
  # for tag in scalar_list:
  #   x = [int(s.step) for s in acc.Scalars(tag)]
  #   y = [s.value for s in acc.Scalars(tag)]
  #
  #   # smooth curve
  #   x_ = []
  #   y_ = []
  #   for i in range(0, len(x), smooth_space):
  #     x_.append(x[i])
  #     y_.append(sum(y[i:i+smooth_space]) / float(smooth_space))
  #   x_.append(x[-1])
  #   y_.append(y[-1])
  #   x_list.append(x_)
  #   y_list.append(y_)

    # # raw curve
    # x_list_raw.append(x)
    # y_list_raw.append(y)


  # for i in range(len(x_list)):
  #   plt.figure(i)
  #   plt.subplot(111)
  #   plt.title(scalar_list[i])
  #   plt.plot(x_list_raw[i], y_list_raw[i], color=colors.to_rgba(color_code, alpha=0.4))
  #   plt.plot(x_list[i], y_list[i], color=color_code, linewidth=1.5)
  # plt.show()


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', default='./experiments/comparatie_eigenoc_vs_oc', type=str, help='logdir to event file')
  parser.add_argument('--smooth', default=100, type=float, help='window size for average smoothing')
  parser.add_argument('--color', default='#4169E1', type=str, help='HTML code for the figure')

  args = parser.parse_args()
  params = vars(args) # convert to ordinary dict

  plot(params)