import numpy as np

class WeatherOracle:
    """
    A statistical utility class that models weather transitions and calculates
    maintenance window probabilities.
    """

    def __init__(self, transition_matrix=None, wind_bins=None, weibull_shape=None, weibull_scale=None):
        """
        Initialize the WeatherOracle with either a Markov transition matrix or Weibull parameters.

        Args:
            transition_matrix (np.ndarray, optional): A 2D array representing the Markov transition matrix.
            wind_bins (np.ndarray, optional): A 1D array representing the wind speed bins.
            weibull_shape (float, optional): The shape parameter (k) for the Weibull distribution.
            weibull_scale (float, optional): The scale parameter (A) for the Weibull distribution.
        """
        self.transition_matrix = transition_matrix
        self.wind_bins = wind_bins
        self.weibull_shape = weibull_shape
        self.weibull_scale = weibull_scale

    def get_maintenance_window_probability(self, current_wind_speed: float, lookahead_days: int) -> float:
        """
        Calculates the probability that the wind speed will remain strictly below 10 m/s
        for the requested lookahead duration.

        Args:
            current_wind_speed (float): The current wind speed in m/s.
            lookahead_days (int): The number of days to look ahead.

        Returns:
            float: The probability of the wind speed remaining < 10 m/s.
        """
        if current_wind_speed >= 10.0:
            return 0.0

        if self.transition_matrix is not None and self.wind_bins is not None:
            # Find the closest wind bin to the current wind speed
            current_state_idx = int(np.argmin(np.abs(np.array(self.wind_bins) - current_wind_speed)))

            valid_states = np.array(self.wind_bins) < 10.0

            if not valid_states[current_state_idx]:
                return 0.0

            valid_indices = np.where(valid_states)[0]
            T_restricted = np.array(self.transition_matrix)[np.ix_(valid_indices, valid_indices)]

            local_idx = np.where(valid_indices == current_state_idx)[0][0]
            initial_vector = np.zeros(len(valid_indices))
            initial_vector[local_idx] = 1.0

            T_power = np.linalg.matrix_power(T_restricted, lookahead_days)
            final_vector = initial_vector.dot(T_power)

            return float(np.sum(final_vector))

        elif self.weibull_shape is not None and self.weibull_scale is not None:
            # Weibull CDF: P(X < x) = 1 - exp(-(x / scale)^shape)
            p_single_day = 1.0 - np.exp(-((10.0 / self.weibull_scale) ** self.weibull_shape))
            return float(p_single_day ** lookahead_days)

        else:
            raise ValueError("Must provide either (transition_matrix, wind_bins) or (weibull_shape, weibull_scale)")
