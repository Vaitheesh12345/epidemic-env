"""
SEIR Disease Model for Epidemic Containment RL Environment
Simulates disease spread: Susceptible → Exposed → Infected → Recovered/Dead
"""

import numpy as np


class SEIRModel:
    def __init__(self, population=1_000_000, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.N = population
        self.S = population - 100   # Susceptible
        self.E = 50                  # Exposed
        self.I = 50                  # Infected
        self.R = 0                   # Recovered
        self.D = 0                   # Dead

        # Model parameters
        self.gamma = 0.1             # Recovery rate (~10 days)
        self.delta = 0.2             # Incubation rate (~5 days)
        self.mortality_rate = 0.02   # 2% infection fatality rate
        self.hospital_capacity = 50_000  # Max patients

        # History tracking
        self.history = []

    def step(self, beta, vaccination_coverage=0.0):
        """
        Simulate one week (7 days) of disease spread.

        Args:
            beta: Transmission rate (controlled by agent actions)
            vaccination_coverage: Fraction of population vaccinated

        Returns:
            dict with current stats
        """
        # Run 7 daily sub-steps for accuracy
        for _ in range(7):
            effective_susceptible = self.S * (1 - vaccination_coverage)

            # SEIR differential equations
            new_exposed   = beta * effective_susceptible * self.I / self.N
            new_infected  = self.delta * self.E
            new_recovered = self.gamma * self.I * (1 - self.mortality_rate)
            new_dead      = self.gamma * self.I * self.mortality_rate

            # Add small noise for realism
            noise = np.random.normal(0, 0.001)
            new_exposed = max(0, new_exposed + noise * self.N)

            # Update compartments
            self.S = max(0, self.S - new_exposed)
            self.E = max(0, self.E + new_exposed - new_infected)
            self.I = max(0, self.I + new_infected - new_recovered - new_dead)
            self.R = self.R + new_recovered
            self.D = self.D + new_dead

        stats = self.get_stats()
        self.history.append(stats)
        return stats

    def get_stats(self):
        return {
            "susceptible":        int(self.S),
            "exposed":            int(self.E),
            "infected":           int(self.I),
            "recovered":          int(self.R),
            "dead":               int(self.D),
            "infection_rate":     self.I / self.N,
            "hospitalized":       min(self.I, self.hospital_capacity),
            "hospital_overflow":  max(0, self.I - self.hospital_capacity),
            "hospital_load":      min(1.0, self.I / self.hospital_capacity),
            "total_cases":        int(self.E + self.I + self.R + self.D),
        }

    def reset(self):
        self.__init__(self.N)