#!/usr/bin/env python

from numpy import insert, cumsum, sqrt, linspace, exp, full_like, geomspace, rint, mean, std, array
from numpy.random import seed, normal
from matplotlib import pyplot as plt

# a
def brownian_motion(time_horizon, num_steps, random_seed=1108):
	seed(random_seed)
	increments = normal(0, sqrt(time_horizon / num_steps), size=num_steps)
	path = insert(cumsum(increments), 0, 0)
	return linspace(0, time_horizon, num_steps + 1), path

def stratonovich_integral(func, time, process):
	integrand = func(time, process)
	dx = process[1:] - process[:-1]
	return insert(cumsum((integrand[1:] + integrand[:-1]) / 2 * dx), 0, 0)

def ito_integral(func, time, process):
	integrand = func(time, process)
	dx = process[1:] - process[:-1]
	return insert(cumsum(integrand[:-1] * dx), 0, 0)

def ito_to_stratonovich(func, time, process, delta=1e-6):
	grad_fx = ((func(time, process + delta) - func(time, process - delta)) / (2 * delta))[:-1]
	dt = time[1:] - time[:-1]
	return ito_integral(func, time, process) + insert(cumsum(grad_fx * dt), 0, 0) / 2

time, process = brownian_motion(10, 100)
args = (lambda t, x: x, time, process)
plt.plot(time, ito_integral(*args), label='Ito')
plt.plot(time, stratonovich_integral(*args), label='Stratonovich')
plt.plot(time, ito_to_stratonovich(*args), label='Ito to Stratonovich')
plt.legend()
plt.show()

# b
def geometric_brownian_motion(drift, volatility, time_horizon, num_steps, random_seed=1108):
	time, wiener_process = brownian_motion(time_horizon, num_steps, random_seed)
	return time, exp(drift * time + volatility * wiener_process)

plt.plot(*geometric_brownian_motion(0.1, 0.2, 10, 100))
plt.show()

# c
time, process = geometric_brownian_motion(0.1, 0.2, 10, 100)
args = (lambda t, x: full_like(t, 1), time, process)
plt.plot(time, ito_integral(*args))
plt.show()

# d
plt.plot(time, stratonovich_integral(*args))
plt.show()

# e
ito_means = []
stratonovich_means = []
ito_stds = []
stratonovich_stds = []
num_steps_list = rint(geomspace(10, 1e4, 6)).astype(int)
for num_steps in num_steps_list:
	ito_final_vals = []
	stratonovich_final_vals = []
	for i in range(500):
		time, process = geometric_brownian_motion(0.1, 0.2, 10, num_steps, random_seed=i)
		args = (lambda t, x: full_like(t, 1), time, process)
		ito_final_vals.append(ito_integral(*args)[-1])
		stratonovich_final_vals.append(stratonovich_integral(*args)[-1])
	ito_means.append(mean(ito_final_vals))
	stratonovich_means.append(mean(stratonovich_final_vals))
	ito_stds.append(std(ito_final_vals))
	stratonovich_stds.append(std(stratonovich_final_vals))

fig, axes = plt.subplots(2, 2)
axes[0,0].plot(num_steps_list, ito_means)
axes[0,0].set_xscale('log')
axes[0,0].set_title('Ito Mean')
axes[0,1].plot(num_steps_list, stratonovich_means)
axes[0,1].set_xscale('log')
axes[0,1].set_title('Stratonovich Mean')
axes[1,0].plot(num_steps_list, ito_stds)
axes[1,0].set_xscale('log')
axes[1,0].set_title('Ito Std')
axes[1,1].plot(num_steps_list, stratonovich_stds)
axes[1,1].set_xscale('log')
axes[1,1].set_title('Stratonovich Std')
plt.show()

# fc
time, process = geometric_brownian_motion(0.1, 0.2, 10, 100)
args = (lambda t, x: x**2, time, process)
plt.plot(time, ito_integral(*args))
plt.show()

# fd
plt.plot(time, stratonovich_integral(*args))
plt.show()

# fe
ito_means = []
stratonovich_means = []
ito_stds = []
stratonovich_stds = []
num_steps_list = rint(geomspace(10, 1e4, 6)).astype(int)
for num_steps in num_steps_list:
	ito_final_vals = []
	stratonovich_final_vals = []
	for i in range(500):
		time, process = geometric_brownian_motion(0.1, 0.2, 10, num_steps, random_seed=i)
		args = (lambda t, x: x**2, time, process)
		ito_final_vals.append(ito_integral(*args)[-1])
		stratonovich_final_vals.append(stratonovich_integral(*args)[-1])
	ito_means.append(mean(ito_final_vals))
	stratonovich_means.append(mean(stratonovich_final_vals))
	ito_stds.append(std(ito_final_vals))
	stratonovich_stds.append(std(stratonovich_final_vals))

fig, axes = plt.subplots(2, 2)
axes[0,0].plot(num_steps_list, ito_means)
axes[0,0].set_xscale('log')
axes[0,0].set_title('Ito Mean')
axes[0,1].plot(num_steps_list, stratonovich_means)
axes[0,1].set_xscale('log')
axes[0,1].set_title('Stratonovich Mean')
axes[1,0].plot(num_steps_list, ito_stds)
axes[1,0].set_xscale('log')
axes[1,0].set_title('Ito Std')
axes[1,1].plot(num_steps_list, stratonovich_stds)
axes[1,1].set_xscale('log')
axes[1,1].set_title('Stratonovich Std')
plt.show()

# g
# Actually C(t, tau) = <F(t) F(t+tau)> depends on t in addition to tau
# (because <F(t)> ~ exp(3 mu t) depends on t, also notice that F(0) = 0 always),
# but the worksheet seems to assume that it does not.
# I will adopt C(tau) = <F(tau)> instead (i.e. C(tau) ~ C(0, tau) / F(0)).
time_step = 0.1
num_steps = 300
time_vals = array([5, 10, 20, 30])

time_indices = rint(time_vals / time_step).astype(int)
correlation_vals = []
for i in range(500):
	time_series, process_series = geometric_brownian_motion(0.1, 0.2, time_step * num_steps, num_steps, random_seed=i)
	func_series = stratonovich_integral(lambda t, x: x**2, time_series, process_series)
	correlation_vals.append([func_series[j] for j in time_indices])
correlation_vals = mean(correlation_vals, axis=0)

fig, ax = plt.subplots()
ax.plot(time_vals, correlation_vals)
ax.set_yscale('log')
plt.show()
