import numpy as np
from scipy.special import j0

class DynamicChannelSimulator:
    def __init__(self, num_users, carrier_freq_hz=2e9, user_speed_kmh=30, tti_duration_s=1e-3):
        self.num_users = num_users
        speed_ms = user_speed_kmh * (1000 / 3600)
        wavelength = 3e8 / carrier_freq_hz
        fd = speed_ms / wavelength 
        self.rho = j0(2 * np.pi * fd * tti_duration_s)
        self.H_current = self._generate_rayleigh_matrix()

    def _generate_rayleigh_matrix(self):
        return (np.random.randn(self.num_users, self.num_users) + 
                1j * np.random.randn(self.num_users, self.num_users)) / np.sqrt(2)

    def step(self):
        innovation = self._generate_rayleigh_matrix()
        self.H_current = self.rho * self.H_current + np.sqrt(1 - self.rho**2) * innovation
        return self.H_current