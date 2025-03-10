#!/usr/bin/env python

from numpy import exp, sin, pi, tile, eye, zeros, ones, arange
from numpy.linalg import eigh, norm, matrix_power
from scipy.linalg import null_space, logm
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt

# 1
# copied from section 4
def is_adjacent_hopping(num_sites, state_a, state_b):
	if state_a == state_b:
		return 0
	diff = state_a ^ state_b  # find sites where the spin differs
	if state_a & diff == 0 or state_b & diff == 0:  # one of the states has spin up on those sites
		return 0
	return int(diff % 3 == 0 and diff // 3 & diff // 3 - 1 == 0) + int(diff == (1 << num_sites - 1) + 1)  # condition 1: adjacent; condition 2: periodic

def get_site_spin(num_sites, state, index):
	return 1 / 2 - (state >> index % num_sites & 1)

def heisenberg_xxx_element(num_sites, state_a, state_b):
	result = 0
	if state_a == state_b:
		for i in range(num_sites):
			result += 1 / 4 - get_site_spin(num_sites, state_a, i) * get_site_spin(num_sites, state_a, i + 1)
	result -= is_adjacent_hopping(num_sites, state_a, state_b) / 2
	return result

def heisenberg_xxx_hamiltonian(num_sites):
	return [[heisenberg_xxx_element(num_sites, i, j) for j in range(1 << num_sites)] for i in range(1 << num_sites)]

def boltzmann_distribution(energy_levels, temperature):
	numerator = exp(-energy_levels / temperature)
	return numerator / sum(numerator)

def transition_matrix(num_sites, temperature=1):
	hamiltonian = heisenberg_xxx_hamiltonian(num_sites)
	energy_levels, eigenvectors = eigh(hamiltonian)
	probabilities = boltzmann_distribution(energy_levels, temperature)
	basis_amp = (eigenvectors.conj() * eigenvectors).real
	return basis_amp.T @ tile(probabilities, (1 << num_sites, 1)) @ basis_amp

num_spins = 3
transition_probabilities = transition_matrix(num_spins)
print(transition_probabilities)

# 2
stationary_distribution = null_space(transition_probabilities.T - eye(1 << num_spins)).T
stationary_distribution /= stationary_distribution.sum(axis=1, keepdims=True)
if len(stationary_distribution) != 1:
	raise ValueError('Stationary distribution is not unique.')
stationary_distribution = stationary_distribution[0]
print(stationary_distribution)

# 3
def markov_iteration(transition, initial_state, max_steps=1000, tolerance=1e-6):
	current_state = initial_state
	for i in range(max_steps):
		next_state = current_state @ transition
		if norm(next_state - current_state) < tolerance:
			return next_state
		current_state = next_state
	raise ValueError('Did not converge in {} iterations.'.format(max_steps))

initial_state = zeros(1 << num_spins)
initial_state[0b000] = 1
print(markov_iteration(transition_probabilities, initial_state))

initial_state = zeros(1 << num_spins)
initial_state[0b000] = 0.5
initial_state[0b101] = 0.5
print(markov_iteration(transition_probabilities, initial_state))

initial_state = ones(1 << num_spins) / (1 << num_spins)
print(markov_iteration(transition_probabilities, initial_state))

# 4
def transition_matrix_magnon(num_sites, temperature=1):
	momentum_indices = arange(num_sites)
	energy_levels = 2 * sin(pi * momentum_indices / num_sites) ** 2
	probabilities = boltzmann_distribution(energy_levels, temperature)
	return tile(probabilities, (num_sites, 1))

transition_probabilities_magnon = transition_matrix_magnon(num_spins)
print(transition_probabilities_magnon)

# The difference between the transition matrix in site basis and in magnon basis
# is that they have different dimensions.
# The probability in the magnon transition matrix
# is the conditional probability for a magnon to transition to another magnon,
# neglecting the possibility of transitioning to non-magnon states.

# 5
stationary_distribution_magnon = null_space(transition_probabilities_magnon.T - eye(num_spins)).T
stationary_distribution_magnon /= stationary_distribution_magnon.sum(axis=1, keepdims=True)
if len(stationary_distribution_magnon) != 1:
	raise ValueError('Stationary distribution is not unique.')
stationary_distribution_magnon = stationary_distribution_magnon[0]
print(stationary_distribution_magnon)

# 6
initial_state = zeros(num_spins)
initial_state[1] = 1
print(markov_iteration(transition_probabilities_magnon, initial_state))

initial_state = zeros(num_spins)
initial_state[1] = 0.5
initial_state[2] = 0.5  # According to the worksheet, it should be initial_state[4], but num_spins=3 forbids such large k.
print(markov_iteration(transition_probabilities_magnon, initial_state))

initial_state = ones(num_spins) / num_spins
print(markov_iteration(transition_probabilities_magnon, initial_state))

# 7
time_steps = 100
rate_matrix = logm(matrix_power(transition_probabilities_magnon, time_steps)) / time_steps
initial_state = zeros(num_spins)
initial_state[1] = 1
solution = solve_ivp(lambda t, y: rate_matrix.T @ y, (0, 20), initial_state)
for k in range(num_spins):
	plt.plot(solution.t, solution.y[k], label=r'$\pi_{}$'.format(k))
plt.xlabel('$t$')
plt.legend()
plt.show()
