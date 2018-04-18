import csv
import matplotlib.pyplot as plt;
import numpy as np;
import scipy.optimize as opt;
x_eoc = []
x_oc = []
y_eoc = []
y_oc = []

with open('to_plot.csv', 'rt') as f:
    reader = csv.reader(f)
    for i, row in enumerate(reader):
        if i == 0:
            continue
        else:
            x_eoc.append(int(row[0]))
            y_eoc.append(int(row[1]))
            x_oc.append(int(row[2]))
            y_oc.append(int(row[3]))

# plt.plot(x_eoc, y_eoc, ".", label="EOC")
# plt.plot(x_oc, y_oc, "*", label="OC")
from scipy.interpolate import interp1d
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
xnew = np.linspace(0, max(x_eoc), 40)
ius = InterpolatedUnivariateSpline(x_eoc, y_eoc)

yi = ius(xnew)

f2 = interp1d(x_eoc, y_eoc, kind='cubic', bounds_error=False, fill_value=1)

# xvals1 = np.linspace(0, max(x_eoc), 100)
# yinterp1 = np.interp(xvals1, x_eoc, y_eoc)
from scipy import interpolate
t = [np.pi/2-.1, np.pi/2+.1, 3*np.pi/2-.1, 3*np.pi/2+.1]
# s = interpolate.LSQUnivariateSpline(x_eoc, y_eoc, t, k=2)

# plt.plot(x_eoc, y_eoc, 'r-', label="EOC")
plt.plot(x_oc, y_oc, 'b-', label="OC")

# xvals2 = np.linspace(0, max(x_oc), 100)
# yinterp2 = np.interp(xvals2, x_oc, y_oc)
plt.plot(xnew, f2(xnew), '-', label="EOC")

# # The actual curve fitting happens here
# optimizedParameters, pcov = opt.curve_fit(func, x_eoc, y_eoc);
#
# # Use the optimized parameters to plot the best fit
# plt.plot(xdata, func(xdata, *optimizedParameters), label="fit");

# Show the graph

plt.legend(loc='best')
plt.show(block=True)