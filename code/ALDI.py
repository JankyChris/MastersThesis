"""
Title:              ALDI Class Implementation
Description:        The ALDI class is designed to implement the Affine Invariant Langevin Dynamics (ALDI) algorithm, 
                    a method for generating samples from a posterior distribution in Bayesian inverse problems. 
Author:             Christoph Jankowsky
Date Created:       2024-04-29
Last Modified:      2024-09-05
Usage:              Instantiate the ALDI class with appropriate parameters and call the `aldi` method to perform the sampling. 
                    The class requires an initial ensemble and parameters like step size and number of iterations to function.
Additional Notes:   This software is developed as part of the requirements for the Master's thesis at Freie Universität Berlin. 
                    It is intended for academic purposes.
"""

import numpy as np  # Importing the NumPy library for numerical operations, especially on arrays.
import matplotlib.pyplot as plt

from tqdm import tqdm

class ALDI:
    def __init__(self, step_size: float, number_of_iterations: int, gradient_free: bool, ensemble: np.ndarray, potential, verbose: bool = False) -> None:
        # Initialize class attributes with the provided parameters and set others to None for later computation.
        self.max_step_size = step_size
        self.number_of_iterations = number_of_iterations
        self.gradient_free = gradient_free
        self.ensemble = ensemble.copy() # D x N
        self.starting_ensemble = ensemble.copy()
        self.dimension = np.shape(self.ensemble)[0] # D
        self.number_of_particles = np.shape(self.ensemble)[1] # N
        self.potential = potential
        self.verbose = verbose

        self.ensemble_mean = None
        self.ensemble_covariance = None
        self.sqrt_ensemble_covariance = None
        self.ensemble_deviations = None
        self.path = None
        self.show_steps = False

    def set_ensemble_mean(self):
        """
        Calculates and sets the mean of the ensemble.
        """
        # Calculate the mean across the ensemble (axis=1 for column-wise mean if ensemble is 2D).
        self.ensemble_mean = np.mean(self.ensemble, axis=1)

    def set_ensemble_covariance(self):
        """
        Calculates and sets the covariance matrix of the ensemble.
        """
        self.ensemble_covariance = np.cov(self.ensemble, bias=True)

    def set_sqrt_ensemble_covariance(self):
        """
        Calculates and sets a square root of the ensemble covariance matrix.
        """
        # Calculate the square root of the ensemble covariance matrix.
        self.set_ensemble_deviations()
        self.sqrt_ensemble_covariance = self.ensemble_deviations / np.sqrt(self.number_of_particles) # (D x N)

    def set_ensemble_deviations(self):
        self.ensemble_deviations = self.ensemble - (self.ensemble_mean[:, np.newaxis])

    def get_empirical_cross_correlation(self) -> np.ndarray:
        """
        Placeholder for method to calculate empirical cross-correlation.
        """
        self.set_ensemble_mean
        self.set_ensemble_deviations

        forward_evaluations = np.array([self.potential.forward_map(particle) for particle in self.ensemble.T]).squeeze()
        forward_deviations = forward_evaluations - np.mean(forward_evaluations)
        #forward_deviations = self.potential.forward_map(self.ensemble) - np.mean(self.potential.forward_map(self.ensemble))

        cross_correlation = np.dot(self.ensemble_deviations, forward_deviations)
        return 1/self.number_of_particles * cross_correlation # D x K
    
    def reset_ensemble(self):
        """
        Resets the ensemble to its starting configuration.
        """
        self.ensemble = self.starting_ensemble

    def aldi_step(self) -> np.ndarray:
        """
        Performs a single step of the ALDI algorithm.
        
        Returns:
        - np.ndarray: The update to apply to the ensemble based on the ALDI algorithm.
        """
        if self.gradient_free:
            # Calculate empirical cross-correlation if gradient-free approach is used.
            empirical_cross_correlation = self.get_empirical_cross_correlation()
            if self.potential.output_dimension != 1:
                step_A_1 = np.dot(empirical_cross_correlation, np.dot(1/self.potential.error_sigma * np.eye(self.potential.output_dimension), (np.array([self.potential.forward_map(particle) for particle in self.ensemble.T]) - self.potential.observation).T))
            else:
                step_A_1 = empirical_cross_correlation * 1/self.potential.error_sigma * (np.array([self.potential.forward_map(particle) for particle in self.ensemble.T]) - self.potential.observation)
            step_A_2 = np.dot(self.ensemble_covariance, np.dot(self.potential.prior_sigma * np.eye(self.dimension), (self.ensemble - self.potential.prior_mean[:, np.newaxis])))
            step_A = - (step_A_1 + step_A_2)
        else:
            # Calculate step_A using the gradient of the potential.
            ensemble_gradients = np.zeros((self.dimension, self.number_of_particles)) # D x N
            for i, particle in enumerate(self.ensemble.T):
                gradient_at_particle = self.potential.get_potential_gradient(particle.T)
                ensemble_gradients[:, i] = gradient_at_particle.T
            #ensemble_gradient = np.array([self.potential.evaluate_potential_gradient(particle.T) for particle in self.ensemble.T]) # (N x D)
            step_A = - np.dot(self.ensemble_covariance, ensemble_gradients) # (D x D) @ (D x N) = D x N
        
        # The following two lines prevent overflow by reducing step size, if needed.
        max_value = np.max(np.abs(step_A))
        if max_value == 0: max_value=0.0001
        max_step_size = self.max_step_size
        self.step_size = min(max_step_size / max_value, 0.999)

        # Calculate step_B which is a correction term based on the ensemble mean.
        step_B = (self.dimension + 1)/self.number_of_particles * (self.ensemble - self.ensemble_mean[:, np.newaxis]) # D x N
        
        # Generate random diffusion term for step_C.
        diffusion = np.random.randn(self.number_of_particles, self.number_of_particles) # (N x N)
        step_C = np.sqrt(2 * self.step_size) * np.dot(self.sqrt_ensemble_covariance, diffusion.T) # (D x N) @ (N x N) = (D x N)

        # Return the combined update step.
        return step_A * self.step_size + step_B * self.step_size + step_C # (D x N)
    
    def generate_samples(self) -> np.ndarray:
        """
        Executes the ALDI algorithm for the specified number of iterations.
        
        Returns:
        - np.ndarray: The final ensemble after all iterations of the ALDI algorithm.
        """
        for i in tqdm(range(self.number_of_iterations), disable = not self.verbose, desc="ALDI"):
            # Update the ensemble parameters
            self.set_ensemble_mean()
            self.set_ensemble_covariance()
            self.set_sqrt_ensemble_covariance()
            # Perform a single ALDI step and update the ensemble.
            step = self.aldi_step()

            if self.show_steps:
                plt.axis([-10, 10, -10, 10])
                plt.scatter(*self.ensemble)
                self.potential.visualize_posterior(10, 10)
                for i, particle in enumerate(self.ensemble.T):
                    dx, dy = self.potential.evaluate_potential_gradient(particle.T)
                    plt.arrow(particle[0], particle[1], -dx * self.step_size, -dy*self.step_size, width=0.1,color="teal")
                    plt.arrow(particle[0], particle[1], step[1, i], step[0, i], width=0.1, color="orange")
                plt.show(block=False)
                plt.pause(1.)
                plt.close()
            
            self.ensemble += step
            #if np.isnan(self.ensemble).any():
            #    raise ValueError("NaN value detected in the ensemble.")
        # Return the updated ensemble after all iterations.
        return self.ensemble
    
    def generate_sample_path(self) -> np.ndarray:
        """
        Executes the ALDI algorithm for the specified number of iterations.
        
        Returns:
        - np.ndarray: The path the ensemble takes during the ALDI algorithm.
        """

        if self.verbose:
            print("Generating sample path...")
            if self.gradient_free:
                print("Gradient-free approach selected.")
            else:
                print("Gradient-based approach selected.")
            print(f"Performing {self.number_of_iterations} iterations...")

        self.path = np.zeros((self.dimension, self.number_of_particles, self.number_of_iterations))

        for i in range(self.number_of_iterations):
            if self.verbose:
                if i % 100 == 0:
                    dots = (i // 100) % 4  # Cycle through 0 to 3
                    print(" "*20, end="\r")
                    print(f"Iteration {i}" + "."*dots, end="\r")
            # Update the ensemble parameters
            self.set_ensemble_mean()
            self.set_ensemble_covariance()
            self.set_sqrt_ensemble_covariance()
            # Perform a single ALDI step and update the ensemble.
            step = self.aldi_step()
            self.ensemble += step
            self.path[:,:,i] = self.ensemble

        if self.verbose:
            print(f"Iteration {i+1}...")
            print("Sample path generated.")
        # Return the path.
        return self.path

class GaussianPotential: 
    def __init__(self, forward_map, forward_gradient, error_covariance, prior_covariance, prior_mean, observation) -> None:
        """
        Initializes the GaussianPotential class with specified parameters.
        
        Parameters:
        - forward_map: A function representing the forward map.
        - forward_gradient: A function representing the gradient of the forward map.
        - error_covariance: The covariance matrix of the observation error.
        - prior_covariance: The covariance matrix of the prior distribution.
        - prior_mean: The mean of the prior distribution.
        """
        self.forward_map = forward_map              # G
        self.forward_gradient = forward_gradient
        self.error_covariance = error_covariance    # R
        self.sqrt_error_covariance = None
        self.prior_covariance = prior_covariance    # P_0
        self.sqrt_prior_covariance = None
        self.prior_mean = prior_mean                # mu_0
        self.observation = observation              # y

    def set_sqrt_covariance(self):
        """
        Placeholder for method to calculate square root of covariance matrices.
        """
        raise NotImplementedError
        
    def least_squares_misfit(self, particle, observation):
        """
        Calculates the least squares misfit for a given particle and observation.
        
        Parameters:
        - particle: The particle for which to calculate the misfit.
        - observation: The observed data.
        
        Returns:
        - float: The least squares misfit value.
        """
        return 0.5 * np.linalg.norm(1/self.sqrt_error_covariance * (observation - self.forward_map(particle)))**2
    
    def prior_potential(self, particle):
        """
        Calculates the prior potential for a given particle.
        
        Parameters:
        - particle: The particle for which to calculate the prior potential.
        
        Returns:
        - float: The prior potential value.
        """
        return 0.5 * np.linalg.norm(1/self.sqrt_prior_covariance * (particle - self.prior_mean))**2
    
    def potential(self, particle, observation):
        """
        Calculates the total potential for a given particle and observation.
        
        Parameters:
        - particle: The particle for which to calculate the potential.
        - observation: The observed data.
        
        Returns:
        - float: The total potential value.
        """
        return self.least_squares_misfit(particle, observation) + self.prior_potential(particle)
    
    def lsm_gradient(self, particle):
        """
        Calculates the gradient of the least squares misfit for a given particle.
        
        Parameters:
        - particle: The particle for which to calculate the gradient.
        
        Returns:
        - np.ndarray: The gradient of the least squares misfit.
        """
        # TODO include observation
        return -1/self.error_covariance**2 * (self.observation - self.forward_map(particle)) * self.forward_gradient(particle)

    def prior_gradient(self, particle):
        """
        Calculates the gradient of the prior potential for a given particle.
        
        Parameters:
        - particle: The particle for which to calculate the gradient.
        
        Returns:
        - np.ndarray: The gradient of the prior potential.
        """
        inverse_covariance = np.linalg.inv(self.prior_covariance)
        return np.dot(inverse_covariance, (particle - self.prior_mean))
    
    def potential_gradient(self, particle):
        """
        Calculates the total gradient of the potential for a given particle.
        
        Parameters:
        - particle: The particle for which to calculate the gradient.
        
        Returns:
        - np.ndarray: The total gradient of the potential.
        """
        # TODO include observation
        return self.lsm_gradient(particle) + self.prior_gradient(particle)
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.stats import multivariate_normal

    D = 2
    N = 50
    n_iter = 200
    prior_mean = [0., 7.]
    prior_sigma = 5.
    error_mean = 0.
    error_sigma = 5.

    bound = 10
    x1 = np.arange(-bound, bound, 0.1)
    x2 = np.arange(-bound, bound, 0.1)

    ensemble = multivariate_normal.rvs(prior_mean, prior_sigma*np.eye(D), size=N).T

    def G(x):
    # the forward map G : R^2 -> R
        return x[0]**2 + 5*x[1]
    
    def grad_G(x):
        return np.array([2*x[0], 5])

    def prior(x, prior_mean, prior_covariance):
        # pi_0
        return np.exp(-1/(2 * prior_covariance**2) * np.linalg.norm(x-prior_mean)**2)

    def likelihood(x, error_mean, error_covariance):
        return np.exp(-1/(2 * error_covariance**2) * (G(x) - error_mean)**2)

    def posterior(x, prior_mean, prior_covariance, error_mean, error_covariance):
        # Bayes Theorem
        return prior(x, prior_mean, prior_covariance) * likelihood(x, error_mean, error_covariance)

    potential = GaussianPotential(forward_map=G, forward_gradient=grad_G, error_sigma=error_sigma, prior_covariance=prior_covariance, prior_mean=prior_mean)

    sampler = ALDI(0.1, n_iter, gradient_free=False, ensemble=ensemble.copy(), potential=potential, verbose=False)

    sampler.reset_ensemble()
    sample_path = sampler.generate_sample_path()
    final_ensemble = sample_path[:,:,-1]
    #plt.scatter(*ensemble)
    plt.scatter(*final_ensemble)
    plt.show()
