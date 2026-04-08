"""
Reward Function for Epidemic Containment RL Environment

Balances three competing objectives:
1. Health outcomes (minimize infections & deaths)
2. Economic impact (minimize lockdown damage)
3. Healthcare capacity (prevent hospital overflow)
"""


# Action names for logging
ACTION_NAMES = {
    0: "Do Nothing",
    1: "Partial Lockdown",
    2: "Full Lockdown",
    3: "Vaccination Drive",
    4: "Travel Restriction",
    5: "Ease Restrictions",
}


def calculate_reward(stats, economic_index, vaccination_coverage, action, prev_infection_rate=None):
    """
    Calculate reward for a given step.

    Args:
        stats: dict from SEIRModel.get_stats()
        economic_index: float 0-1 (current economic health)
        vaccination_coverage: float 0-1
        action: int (0-5)
        prev_infection_rate: float or None (previous week's infection rate)

    Returns:
        float: reward value
        dict: reward breakdown for logging
    """
    reward = 0.0
    breakdown = {}

    infection_rate = stats["infection_rate"]
    hospital_load  = stats["hospital_load"]
    dead           = stats["dead"]
    hospital_overflow = stats["hospital_overflow"]

    # ── HEALTH REWARDS ──────────────────────────────────────────────
    # Reward for keeping infection low
    if infection_rate < 0.01:
        r = +30
    elif infection_rate < 0.05:
        r = +15
    elif infection_rate < 0.10:
        r = +5
    else:
        r = -10
    reward += r
    breakdown["infection_control"] = r

    # Reward for infection trending down
    if prev_infection_rate is not None:
        if infection_rate < prev_infection_rate:
            r = +10
        else:
            r = -5
        reward += r
        breakdown["infection_trend"] = r

    # Penalty for hospital overflow
    if hospital_overflow > 0:
        r = -30
        reward += r
        breakdown["hospital_overflow"] = r
    elif hospital_load < 0.5:
        r = +10
        reward += r
        breakdown["hospital_safe"] = r
    else:
        breakdown["hospital_safe"] = 0

    # Penalty for deaths
    death_penalty = -min(dead / 1000, 30)
    reward += death_penalty
    breakdown["death_penalty"] = round(death_penalty, 2)

    # ── ECONOMIC REWARDS ────────────────────────────────────────────
    if economic_index > 0.8:
        r = +15
    elif economic_index > 0.6:
        r = +8
    elif economic_index > 0.4:
        r = +2
    else:
        r = -15
    reward += r
    breakdown["economic_health"] = r

    # Penalty for aggressive lockdowns
    if action == 2:  # Full lockdown
        r = -8
        reward += r
        breakdown["lockdown_cost"] = r
    else:
        breakdown["lockdown_cost"] = 0

    # ── VACCINATION REWARDS ─────────────────────────────────────────
    if vaccination_coverage > 0.7:
        r = +15
    elif vaccination_coverage > 0.4:
        r = +8
    elif vaccination_coverage > 0.2:
        r = +3
    else:
        r = 0
    reward += r
    breakdown["vaccination_reward"] = r

    breakdown["total"] = round(reward, 2)
    return round(reward, 2), breakdown