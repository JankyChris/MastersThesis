"""
Title:              Langevin Expectation Maximization (LEM) Algorithm
Description:        The LEM class is designed to implement the Langevin Expectation Maximization (LEM) algorithm, 
                    a method for generating samples from a posterior distribution in Bayesian inverse problems. 
                    The algorithm is an extension of the Expectation-Maximization (EM) algorithm which also approximates the posterior distribution
                    of the parameter of the noise model.
Author:             Christoph Jankowsky
Last Modified:      2024-09-05
Usage:              Instantiate the LEM class with appropriate parameters and call the `optimize` method to perform the sampling. 
                    The class requires an initial ensemble and parameters like initial noise parameters and number of iterations to function.
Additional Notes:   This software is developed as part of the requirements for the Master's thesis at Freie Universität Berlin. 
                    It is intended for academic purposes.
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal

from aldi import ALDI
from potential import Potential

class LEM:
    def __init__(self, observations, samples_per_observation, forward_map, forward_gradient, prior_mean, prior_covariance, number_of_iterations, initial_noise_estimate, dimensions=(2, 1), sampler_steps=1000, gradient_free=False, verbose=False) -> None:
        self.observations = observations
        self.samples_per_observation = samples_per_observation
        self.forward_map = forward_map
        self.forward_gradient = forward_gradient
        self.prior_mean = prior_mean
        self.prior_covariance = prior_covariance
        self.number_of_iterations = number_of_iterations # t
        self.noise_estimate = initial_noise_estimate
        self.initial_noise_estimate = initial_noise_estimate
        self.dimensions = dimensions
        self.gradient_free = gradient_free
        self.verbose = verbose

        self.ensembles = None
        self.ensemble = None
        self.potential = None
        self.dimension = self.dimensions[0]
        self.output_dimension = self.dimensions[1]
        self.number_of_observations = np.shape(self.observations)[0]
        self.number_of_particles = self.samples_per_observation * self.number_of_observations
        self.step_size = 0.01
        self.number_of_langevin_steps = sampler_steps
        self.sampler = None
        self.single_problem = False
        self.elbo = 0
        self.max_step_size = 0.01

    def generate_initial_ensembles(self):
        self.ensembles = multivariate_normal.rvs(mean=self.prior_mean, cov=self.prior_covariance*np.eye(self.dimension), size=self.number_of_particles).T

    def set_potential(self, observation):
        self.potential = Potential(self.forward_map, (self.dimension, self.output_dimension), self.prior_mean, self.prior_covariance, observation)

    def update_potential(self):
        self.potential.update_error_sigma(self.noise_estimate)

    def update_sampler(self):
        self.update_potential()
        self.sampler = ALDI(step_size=self.step_size, 
                            number_of_iterations=self.number_of_langevin_steps, 
                            gradient_free=self.gradient_free, 
                            ensemble=self.ensemble.copy(), 
                            potential=self.potential, 
                            verbose=True)
        self.sampler.step_size = self.max_step_size
        
    def generate_samples(self):
        return self.sampler.generate_samples()
    
    def get_residuals(self):
        residuals = np.zeros((self.output_dimension, self.number_of_particles))
        for i, particle in enumerate(self.ensemble.T):
            residuals[:, i] = self.potential.observation - self.potential.forward_map(particle.T)
        return residuals
    
    def estimate_cov(self, observation, mean):
        K = self.output_dimension
        cov = np.zeros((K, K))
        for i in range(K):
            for j in range(K):
                cov[i, j] += (observation[i] - mean[i]) * (observation[j] - mean[j])
        return cov
    

    def update_noise_estimate(self):
        number_of_samples = self.number_of_observations * self.samples_per_observation
        c = 0
        for i, observation in enumerate(self.observations):
            start = i * self.samples_per_observation
            end = (i + 1) * self.samples_per_observation
            current_ensemble = self.ensembles[:, start:end]
            for particle in current_ensemble.T: # i = 1 .. K
                observation = np.array(observation)
                evaluation = np.array(self.forward_map(particle))
                c += np.linalg.norm((observation-evaluation))**2
                        
        self.noise_estimate = c / (self.output_dimension * number_of_samples)
        self.elbo = -(self.output_dimension * number_of_samples)/2 * np.log(2*np.pi*self.noise_estimate) - 1/(2*self.noise_estimate)*c

        #number_of_samples = self.number_of_observations * self.samples_per_observation
        #cov = np.zeros((self.output_dimension, self.output_dimension))
        #for i, observation in enumerate(self.observations):
        #    start = i * self.samples_per_observation
        #    end = (i + 1) * self.samples_per_observation
        #    current_ensemble = self.ensembles[:, start:end]
        #    for particle in current_ensemble.T: # i = 1 .. K
        #        observation = np.array(observation)
        #        evaluation = np.array(self.forward_map(particle))
        #        #deviation = observation - evaluation
        #        cov += self.estimate_cov(observation, evaluation)
        #self.noise_estimate = 1/number_of_samples * cov + np.eye(self.output_dimension) * 0.000001
        
        #log_likelihood = 0
        #for i, observation in enumerate(self.observations):
        #    start = i * self.samples_per_observation
        #    end = (i + 1) * self.samples_per_observation
        #    current_ensemble = self.ensembles[:, start:end]
        #    for particle in current_ensemble.T: # i = 1 .. K
        #        observation = np.array(observation)
        #        evaluation = np.array(self.forward_map(particle))
        #        deviation = observation - evaluation
        #        log_likelihood += np.dot(deviation.T, np.dot(np.linalg.inv(self.noise_estimate), deviation))
        
        #self.elbo = - 0.5 * np.log(np.linalg.det(self.noise_estimate)) - 0.5 * log_likelihood
        

    def E_step(self):
        for i, observation in enumerate(self.observations):
            if self.output_dimension == 1:
                observation = observation.item()
            self.set_potential(observation)
            self.update_potential()
            if self.gradient_free is False:
                self.potential.set_forward_gradient(self.forward_gradient)
            start = i * self.samples_per_observation
            end = (i + 1) * self.samples_per_observation
            self.ensemble = self.ensembles[:, start:end].copy()
            self.update_sampler()
            samples = self.generate_samples()
            self.ensembles[:, start:end] = samples.copy()

    def M_step(self):
        noise_memory = self.noise_estimate
        elbo_prev = self.elbo
        self.update_noise_estimate()
        if self.elbo < elbo_prev or self.noise_estimate > self.initial_noise_estimate: # check if ELBO is actually increased
            self.noise_estimate = noise_memory

    def EM(self, verbose: bool = False, return_archive: bool = False):
        if self.ensembles is None:
            self.generate_initial_ensembles()
        
        ensemble_archive = np.zeros((self.dimension, self.number_of_particles, self.number_of_iterations))
        noise_archive = np.zeros((self.number_of_iterations))

        for i in range(self.number_of_iterations):
            if self.verbose:
                print(" "*20, end="\r")
                print(f"Iteration {i+1}...")
                print("Generating samples...")
            self.E_step()
            if self.verbose:
                print("Updating noise estimate...")
            self.M_step()
            if self.verbose:
                print(f"Current noise estimate: {self.noise_estimate}")# or approximately {np.sqrt(self.noise_estimate):.2f}^2. \n")
            if return_archive:
                #ensemble_archive[:,:,i] = self.ensemble
                noise_archive[i] = self.noise_estimate
        if return_archive:
            return self.ensembles, noise_archive
        else:
            return self.ensembles, self.noise_estimate
        
    def visualize_ensembles(self):
        plt.axis([-2, 2, -2, 2])
        plt.scatter(*self.ensembles, color="teal")
        plt.show()


if __name__ == "__main__":
    d = 2                               # parameter space is 2-dimensional
    G = lambda x: x[0]**2 + 2 * x[1]  # forward map G : R^d -> R
    G_prime = lambda x: np.array([2*x[0], 2.])
    error_mean = 0.                     # observational noise is centered
    error_covariance = 4.               # observational noise (co)variance
    prior_mean = np.array([0., 0.])
    prior_covariance = 5. * np.eye(d)

    number_of_observations = 10         # how many measurements are made
    sample_point = np.zeros((d, 1))     # true parameter generating the measurements
    k = np.shape(G(sample_point))[0]    # output dimension of the forward model
    observational_noise = np.random.normal(error_mean, error_covariance, size=(k, number_of_observations))
    observations = G(sample_point) + observational_noise
    observation = observations[:,0].item()
    potential = Potential(forward_map=G, dimension=d, prior_mean=prior_mean, prior_covariance=prior_covariance, observation=observation)
    ensemble = np.random.multivariate_normal(mean=prior_mean, cov=prior_covariance, size=100).T
    number_of_iterations = 50
    noise_estimate = 8.
    potential.set_forward_gradient(G_prime)

    
    lem = LEM(ensemble, potential, number_of_iterations, noise_estimate, gradient_free=False)
    _, noise_archive = lem.EM(return_archive=True) # 100 iterations of E and M
    lem.visualize_ensemble()

    noise_errors = []
    for i in range(number_of_iterations):
        noise_estimate = noise_archive[:,:,i]
        noise_error = np.linalg.norm(noise_estimate - error_covariance)
        noise_errors.append(noise_error)
    plt.plot([i for i in range(number_of_iterations)], noise_errors)
    plt.show()

