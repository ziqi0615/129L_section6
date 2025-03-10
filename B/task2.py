#!/usr/bin/env python

from numpy import sqrt, insert, cumsum, linspace
from numpy.random import seed, normal
from matplotlib import pyplot as plt

def brownian_motion(time_horizon, num_steps, random_seed=1108):
	seed(random_seed)
	increments = normal(0, sqrt(time_horizon / num_steps), size=num_steps)
	path = insert(cumsum(increments), 0, 0)
	return linspace(0, time_horizon, num_steps + 1), path

time_horizon = 10
num_steps = 100
initial_velocity = 2
diffusion_coeff = 0.1
damping_coeff = 1

time_step = time_horizon / num_steps
time, brownian_path = brownian_motion(time_horizon, num_steps)
brownian_sq = brownian_path**2
noise_term = (brownian_path[1:] - brownian_path[:-1]) / time_step
velocity = [initial_velocity]

for i in range(1, num_steps):
	acceleration = -damping_coeff * velocity[-1] + sqrt(2 * diffusion_coeff) * noise_term[i]
	velocity.append(velocity[-1] + acceleration * time_step)

plt.plot(time[:-1], velocity)
plt.show()
