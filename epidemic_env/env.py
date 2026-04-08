"""
EpidemicEnv — OpenEnv-compatible Reinforcement Learning Environment

An agent acts as a government policy-maker during a disease outbreak,
choosing interventions to minimize deaths while preserving the economy.

OpenEnv Spec:
- Gymnasium-compatible API (reset, step, render)
- Defined task, grader, and reward logic
- LLM-gradable episode summaries
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .disease_model import SEIRModel
from .reward import calculate_reward, ACTION_NAMES
from .grader import grade_episode, llm_grade_prompt


class EpidemicEnv(gym.Env):
    """
    Epidemic Containment RL Environment

    Observation Space (6 variables, all normalized 0-1):
        [infection_rate, hospital_load, vaccination_coverage,
         economic_index, public_compliance, time_progress]

    Action Space (Discrete 6):
        0 → Do Nothing
        1 → Partial Lockdown
        2 → Full Lockdown
        3 → Launch Vaccination Drive
        4 → Restrict Interstate Travel
        5 → Ease Restrictions

    Reward:
        Balances health outcomes, economic impact, and hospital capacity
        Range: approximately -100 to +80 per step

    Episode:
        180 simulated days (26 weekly steps)
        Ends early if deaths exceed 100,000
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    # Action → transmission rate (beta)
    ACTION_BETA = {
        0: 0.50,   # Do Nothing      — high spread
        1: 0.30,   # Partial Lockdown
        2: 0.10,   # Full Lockdown   — very low spread
        3: 0.45,   # Vaccination     — slight reduction
        4: 0.35,   # Travel Restrict
        5: 0.60,   # Ease Restrict   — higher spread
    }

    # Action → weekly economic impact
    ACTION_ECONOMY = {
        0:  0.000,
        1: -0.050,
        2: -0.120,
        3: -0.020,
        4: -0.040,
        5: +0.030,
    }

    def __init__(self, population=1_000_000, render_mode=None):
        super().__init__()

        self.population = population
        self.render_mode = render_mode

        # Spaces
        self.action_space = spaces.Discrete(6)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # Internal state
        self.model = None
        self.day = 0
        self.economic_index = 1.0
        self.vaccination_coverage = 0.0
        self.public_compliance = 0.8
        self.prev_infection_rate = None

        # Episode tracking
        self.episode_reward = 0.0
        self.actions_taken = []
        self.step_logs = []

        self.reset()

    # ── GYM API ─────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.model = SEIRModel(population=self.population, seed=seed)
        self.day = 0
        self.economic_index = 1.0
        self.vaccination_coverage = 0.0
        self.public_compliance = 0.8
        self.prev_infection_rate = None
        self.episode_reward = 0.0
        self.actions_taken = []
        self.step_logs = []

        return self._get_obs(), {}

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"

        # Get transmission rate for chosen action
        beta = self.ACTION_BETA[action]

        # Adjust beta for public compliance
        beta *= (0.5 + 0.5 * self.public_compliance)

        # Update economy
        self.economic_index = float(np.clip(
            self.economic_index + self.ACTION_ECONOMY[action], 0.0, 1.0
        ))

        # Vaccination progress
        if action == 3:
            self.vaccination_coverage = min(
                1.0, self.vaccination_coverage + 0.05
            )

        # Update compliance (lockdown fatigue)
        if action == 2:
            self.public_compliance = max(0.3, self.public_compliance - 0.03)
        elif action == 5:
            self.public_compliance = min(1.0, self.public_compliance + 0.02)

        # Run disease model for 1 week
        stats = self.model.step(beta, self.vaccination_coverage)

        # Calculate reward
        reward, reward_breakdown = calculate_reward(
            stats=stats,
            economic_index=self.economic_index,
            vaccination_coverage=self.vaccination_coverage,
            action=action,
            prev_infection_rate=self.prev_infection_rate,
        )

        self.prev_infection_rate = stats["infection_rate"]
        self.day += 7
        self.episode_reward += reward
        self.actions_taken.append(action)

        # Log step
        self.step_logs.append({
            "day": self.day,
            "action": action,
            "action_name": ACTION_NAMES[action],
            "reward": reward,
            "reward_breakdown": reward_breakdown,
            "stats": stats,
            "economic_index": self.economic_index,
            "vaccination_coverage": self.vaccination_coverage,
        })

        # Check termination
        terminated = (self.day >= 180) or (stats["dead"] > 100_000)
        truncated = False

        info = {
            "day": self.day,
            "stats": stats,
            "reward_breakdown": reward_breakdown,
            "economic_index": self.economic_index,
            "vaccination_coverage": self.vaccination_coverage,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            self._print_state()

    # ── GRADING ─────────────────────────────────────────────────────

    def grade(self):
        """Grade the completed episode. Call after episode ends."""
        final_stats = self.model.get_stats()
        return grade_episode(
            total_reward=self.episode_reward,
            final_stats=final_stats,
            final_economic_index=self.economic_index,
            final_vaccination_coverage=self.vaccination_coverage,
            actions_taken=self.actions_taken,
            episode_length=len(self.actions_taken),
        )

    def get_llm_grade_prompt(self):
        """Return LLM grading prompt for this episode."""
        summary = self._build_episode_summary()
        return llm_grade_prompt(summary)

    # ── HELPERS ─────────────────────────────────────────────────────

    def _get_obs(self):
        stats = self.model.get_stats()
        return np.array([
            stats["infection_rate"],
            stats["hospital_load"],
            self.vaccination_coverage,
            self.economic_index,
            self.public_compliance,
            self.day / 180.0,
        ], dtype=np.float32)

    def _print_state(self):
        stats = self.model.get_stats()
        print(f"\n{'='*55}")
        print(f"  Day {self.day:>3} | Ep Reward: {self.episode_reward:>8.1f}")
        print(f"{'='*55}")
        print(f"  Infected:    {stats['infected']:>10,}  ({stats['infection_rate']*100:.2f}%)")
        print(f"  Dead:        {stats['dead']:>10,}")
        print(f"  Recovered:   {stats['recovered']:>10,}")
        print(f"  Hosp Load:   {stats['hospital_load']*100:>9.1f}%")
        print(f"  Economy:     {self.economic_index*100:>9.1f}%")
        print(f"  Vaccination: {self.vaccination_coverage*100:>9.1f}%")
        print(f"  Compliance:  {self.public_compliance*100:>9.1f}%")
        if self.actions_taken:
            print(f"  Last Action: {ACTION_NAMES[self.actions_taken[-1]]}")
        print(f"{'='*55}")

    def _build_episode_summary(self):
        final = self.model.get_stats()
        action_counts = {ACTION_NAMES[i]: self.actions_taken.count(i) for i in range(6)}
        return f"""
Episode Summary:
- Duration: {self.day} days ({len(self.actions_taken)} steps)
- Total Deaths: {final['dead']:,}
- Total Recovered: {final['recovered']:,}
- Final Infection Rate: {final['infection_rate']*100:.2f}%
- Final Economic Index: {self.economic_index*100:.1f}%
- Vaccination Coverage: {self.vaccination_coverage*100:.1f}%
- Total Reward: {self.episode_reward:.1f}
- Actions Used: {action_counts}
"""