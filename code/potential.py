import numpy as np
import matplotlib.pyplot as plt

class Potential():
    """
    A class to represent the potential function with prior and likelihood distribution.
    """

    def __init__(self, forward_map, dimensions, prior_mean, prior_sigma, observation=None) -> None:
        """
        Initializes the Potential object with the given parameters.

        Parameters
        ----------
        forward_map : function
            Forward model G.
        dimension : tuple
            Dimensions of the input and output space of the forward model.
        prior_mean : np.ndarray
            Mean of the prior distribution.
        prior_covariance : np.ndarray
            Covariance matrix of the prior distribution.
        observation : np.ndarray, optional
            Observed data; if not provided, it defaults to the forward map evaluated at zeros.
        """
        self.forward_map = forward_map
        self.dimension = dimensions[0]
        self.output_dimension = dimensions[1]
        self.prior_mean = prior_mean
        self.prior_sigma = prior_sigma

        self.forward_gradient = None
        self.prior = None
        self.prior_gradient = None
        self.likelihood = None
        self.likelihood_gradient = None
        self.error_sigma = None
        
        if observation is not None:
            self.observation = observation
        else:
            self.observation = self.forward_map(np.zeros(self.dimension))

    def invert(self, value):
        if np.isscalar(value):
            return 1 / value
        else:
            return np.linalg.inv(value)

    def set_error_sigma(self, sigma):
        """
        Sets the observational covariance matrix.

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix of the observational noise.
        """
        self.error_sigma = sigma

    def update_error_sigma(self, sigma):
        """
        Updates the observational covariance and reconfigures the likelihood.

        Parameters
        ----------
        covariance : np.ndarray
            Covariance matrix of the observational noise.
        """
        self.set_error_sigma(sigma)

    def get_prior(self, position):
        """
        Sets the prior probability density function (PDF).
        """
        return np.exp(- self.get_prior_energy(position))
    
    def get_prior_energy(self, position):
        deviation = position - self.prior_mean.T
        return np.dot(deviation.T, np.dot(self.invert(self.prior_sigma), deviation)) / 2

    def get_prior_energy_gradient(self, position):
        """
        Sets the gradient of the prior probability density function (PDF).
        """
        deviation = position  - self.prior_mean.T
        return np.dot(self.invert(self.prior_sigma), deviation)
        #return deviation / self.prior_sigma**2
    
    def get_likelihood(self, position):
        return np.exp(- self.get_likelihood_energy(position))

    def get_likelihood_energy(self, position):
        """
        Sets the likelihood probability density function (PDF) based on the observational covariance.
        """
        deviation = self.observation - self.forward_map(position)
        return np.dot(deviation.T, np.dot(self.invert(self.error_sigma), deviation)) / 2
        #return deviation**2 / (2 * self.error_sigma**2)

    def set_forward_gradient(self, gradient):
        """
        Sets the gradient of the forward model.

        Parameters
        ----------
        gradient : function
            Gradient of the forward model.
        """
        self.forward_gradient = gradient

    def get_likelihood_energy_gradient(self, position):
        """
        Sets the gradient of the likelihood probability density function (PDF) based on the observational covariance and forward gradient.
        """
        if self.forward_gradient is None:
            raise Exception("Forward gradient not set")
        deviation = self.forward_map(position) - self.observation
        return np.dot(self.forward_gradient(position).T, np.dot(self.invert(self.error_sigma), deviation))
        #return self.forward_gradient(position) * deviation / self.error_sigma**2

    def evaluate_posterior(self, position):
        """
        Evaluates the posterior probability density function (PDF) at a given position.

        Parameters
        ----------
        position : np.ndarray
            Position at which to evaluate the posterior.

        Returns
        -------
        float
            Value of the posterior PDF at the given position.
        """
        if self.error_sigma is None:
            raise Exception("Observational covariance not set")
        return self.get_prior(position) * self.get_likelihood(position)

    def get_potential_gradient(self, position):
        """
        Evaluates the gradient of the potential function at a given position.

        Parameters
        ----------
        position : np.ndarray
            Position at which to evaluate the potential gradient.

        Returns
        -------
        np.ndarray
            Gradient of the potential at the given position.
        """
        return self.get_likelihood_energy_gradient(position) + self.get_prior_energy_gradient(position)
    
    def visualize_posterior(self, x_range, y_range):
        """
        Visualizes the posterior distribution over the specified range.

        Parameters
        ----------
        x_range : float
            Range for the x-axis.
        y_range : float
            Range for the y-axis.
        """
        xs = np.arange(-x_range, x_range, 0.1)
        ys = np.arange(-y_range, y_range, 0.1)
        grid = np.zeros((len(xs), len(ys)))
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                grid[i, j] = self.evaluate_posterior([y, -x])
        
        plt.imshow(grid, extent=[-x_range, x_range, -y_range, y_range], cmap="PuRd")
        plt.show()

    def visualize_potential_gradient(self, x_range, y_range):
        """
        Visualizes the gradient of the potential function over the specified range.

        Parameters
        ----------
        x_range : float
            Range for the x-axis.
        y_range : float
            Range for the y-axis.
        """
        # Create a meshgrid
        xs = np.arange(-x_range, x_range, 0.1)
        ys = np.arange(-y_range, y_range, 0.1)
        X, Y = np.meshgrid(xs, ys)
        
        # Initialize gradient arrays
        U = np.zeros(X.shape)
        V = np.zeros(Y.shape)
        
        # Compute the gradient at each point
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                gradient = self.evaluate_potential_gradient([X[i, j], Y[i, j]])
                norm = np.linalg.norm(gradient)
                if norm != 0:
                    U[i, j] = gradient[0] / norm * 100
                    V[i, j] = gradient[1] / norm * 100
        
        # Plot the vector field using quiver
        plt.quiver(X, Y, U, V)
        plt.xlim(-x_range, x_range)
        plt.ylim(-y_range, y_range)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Potential Gradient Vector Field')
        plt.show()

if __name__ == "__main__":
    d = 2                               # parameter space is 2-dimensional
    N = 1
    G = lambda x: x[0] ** 2 + 2 * x[1]  # forward map G : R^d -> R
    G_prime = lambda x: np.array([2 * x[0], 2])
    error_mean = 0.                     # observational noise is centered
    error_covariance = 1.               # observational noise (co)variance
    prior_mean = np.array([0., 7.])
    prior_covariance = 15.

    sample_point = [0., 0.]
    observation = G(sample_point) + (np.random.random_sample() - 0.5) # noisy measurement

    potential = Potential(forward_map=G, dimension=d, prior_mean=prior_mean, prior_sigma=prior_covariance, observation=observation)
    potential.set_error_sigma(error_covariance)
    potential.set_forward_gradient(G_prime)
    potential.visualize_posterior(10, 10)
