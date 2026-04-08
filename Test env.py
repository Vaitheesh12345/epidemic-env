"""
Tests for EpidemicEnv — OpenEnv Hackathon
Run with: python -m pytest tests/ -v
"""

import pytest
import numpy as np
from epidemic_env import EpidemicEnv, SEIRModel, ACTION_NAMES


class TestSEIRModel:
    def test_init(self):
        model = SEIRModel(population=1_000_000)
        assert model.N == 1_000_000
        assert model.S + model.E + model.I + model.R + model.D == model.N

    def test_step_returns_stats(self):
        model = SEIRModel()
        stats = model.step(beta=0.3)
        assert "infection_rate" in stats
        assert "dead" in stats
        assert "hospital_load" in stats
        assert 0 <= stats["infection_rate"] <= 1

    def test_full_lockdown_reduces_spread(self):
        m1 = SEIRModel(seed=42)
        m2 = SEIRModel(seed=42)
        s1 = m1.step(beta=0.5)   # No lockdown
        s2 = m2.step(beta=0.1)   # Full lockdown
        assert s2["infected"] <= s1["infected"]

    def test_history_tracked(self):
        model = SEIRModel()
        for _ in range(5):
            model.step(beta=0.3)
        assert len(model.history) == 5


class TestEpidemicEnv:
    def setup_method(self):
        self.env = EpidemicEnv()

    def test_observation_space(self):
        obs, _ = self.env.reset()
        assert obs.shape == (6,)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_action_space(self):
        assert self.env.action_space.n == 6

    def test_step_returns_correct_format(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_all_actions_valid(self):
        for action in range(6):
            self.env.reset()
            obs, reward, _, _, info = self.env.step(action)
            assert obs is not None
            assert reward is not None

    def test_full_episode(self):
        obs, _ = self.env.reset()
        done = False
        steps = 0
        total_reward = 0

        while not done:
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1

        assert steps <= 26   # Max 180 days / 7 days per step
        assert steps >= 1

    def test_vaccination_increases(self):
        self.env.reset()
        self.env.step(3)  # Launch vaccination drive
        assert self.env.vaccination_coverage > 0

    def test_full_lockdown_harms_economy(self):
        self.env.reset()
        initial_economy = self.env.economic_index
        self.env.step(2)  # Full lockdown
        assert self.env.economic_index < initial_economy

    def test_ease_restrictions_helps_economy(self):
        self.env.reset()
        self.env.step(1)   # Worsen first
        mid_economy = self.env.economic_index
        self.env.step(5)   # Ease
        assert self.env.economic_index >= mid_economy

    def test_grade_after_episode(self):
        obs, _ = self.env.reset()
        done = False
        while not done:
            action = self.env.action_space.sample()
            obs, _, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

        grade = self.env.grade()
        assert "total_score" in grade
        assert "grade" in grade
        assert 0 <= grade["total_score"] <= 100
        assert grade["grade"] in ["A+", "A", "B", "C", "D", "F"]

    def test_llm_grade_prompt(self):
        obs, _ = self.env.reset()
        done = False
        while not done:
            obs, _, terminated, truncated, _ = self.env.step(0)
            done = terminated or truncated
        prompt = self.env.get_llm_grade_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 50

    def test_reset_clears_state(self):
        self.env.reset()
        for _ in range(5):
            self.env.step(2)
        self.env.reset()
        assert self.env.day == 0
        assert self.env.economic_index == 1.0
        assert self.env.vaccination_coverage == 0.0
        assert len(self.env.actions_taken) == 0


class TestActionNames:
    def test_all_actions_named(self):
        for i in range(6):
            assert i in ACTION_NAMES
            assert len(ACTION_NAMES[i]) > 0