import csv
from scipy.interpolate import spline
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

csv_eigenoc = "/Users/ioanaveronicachelu/RL/AOC/eigenoc.csv"
csv_oc = "/Users/ioanaveronicachelu/RL/AOC/oc.csv"

step_eigenoc = []
step_oc = []
value_eigenoc = []
value_oc = []

xnew = np.linspace(0, 100000, 300) #300 represents number of points to make between T.min and T.max

with open(csv_eigenoc, 'rt') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',')
  for i, row in enumerate(spamreader):
    if i == 0:
      continue
    step_eigenoc.append(int(row[1]))
    value_eigenoc.append(int(float(row[2])))

with open(csv_oc, 'rt') as csvfile:
  spamreader = csv.reader(csvfile, delimiter=',')
  for i, row in enumerate(spamreader):
    if i == 0:
      continue
    step_oc.append(int(row[1]))
    value_oc.append(int(float(row[2])))

step_eigenoc = np.array(step_eigenoc)
step_oc = np.array(step_oc)
value_eigenoc = np.array(value_eigenoc)
value_oc = np.array(value_oc)

# smooth_eigenoc = spline(step_eigenoc, value_eigenoc, xnew)
# smooth_oc = spline(step_oc, value_oc, xnew)
value_eigenoc[value_eigenoc > 100] = 100
value_oc[value_oc > 100] = 100

f_eigenoc = interp1d(step_eigenoc, value_eigenoc, kind='cubic')
f_oc = interp1d(step_oc, value_oc, kind='cubic')
plt.yticks(np.arange(0, value_eigenoc.max(), 10))
plt.plot(xnew, value_eigenoc)
plt.plot(xnew, value_oc)

plt.legend(['eigenoc', 'oc'], loc='upper left')
plt.show()

