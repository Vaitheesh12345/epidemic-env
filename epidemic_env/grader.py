"""
Grader for Epidemic Containment RL Environment (OpenEnv spec)

Evaluates agent performance across:
- Health outcomes
- Economic outcomes
- Policy efficiency
"""


def grade_episode(
    total_reward,
    final_stats,
    final_economic_index,
    final_vaccination_coverage,
    actions_taken,
    episode_length,
):
    """
    Grade a full episode out of 100.

    Args:
        total_reward: float — cumulative reward
        final_stats: dict — final SEIRModel stats
        final_economic_index: float 0-1
        final_vaccination_coverage: float 0-1
        actions_taken: list of ints — all actions in episode
        episode_length: int — number of steps taken

    Returns:
        dict with score and breakdown
    """
    score = 0
    breakdown = {}

    total_dead = final_stats["dead"]
    final_infection_rate = final_stats["infection_rate"]
    hospital_overflows = sum(1 for _ in [final_stats] if final_stats["hospital_overflow"] > 0)

    # ── HEALTH SCORE (40 pts) ────────────────────────────────────
    if total_dead < 2_000:
        h = 40
    elif total_dead < 10_000:
        h = 30
    elif total_dead < 30_000:
        h = 15
    elif total_dead < 60_000:
        h = 5
    else:
        h = 0
    score += h
    breakdown["health_score"] = h

    # ── ECONOMIC SCORE (25 pts) ──────────────────────────────────
    if final_economic_index > 0.8:
        e = 25
    elif final_economic_index > 0.6:
        e = 18
    elif final_economic_index > 0.4:
        e = 10
    else:
        e = 0
    score += e
    breakdown["economic_score"] = e

    # ── CONTAINMENT SCORE (20 pts) ───────────────────────────────
    if final_infection_rate < 0.01:
        c = 20
    elif final_infection_rate < 0.05:
        c = 14
    elif final_infection_rate < 0.10:
        c = 7
    else:
        c = 0
    score += c
    breakdown["containment_score"] = c

    # ── VACCINATION SCORE (10 pts) ───────────────────────────────
    if final_vaccination_coverage > 0.7:
        v = 10
    elif final_vaccination_coverage > 0.4:
        v = 6
    elif final_vaccination_coverage > 0.2:
        v = 3
    else:
        v = 0
    score += v
    breakdown["vaccination_score"] = v

    # ── EFFICIENCY BONUS (5 pts) ─────────────────────────────────
    # Penalize over-reliance on full lockdown (action 2)
    full_lockdowns = actions_taken.count(2)
    lockdown_ratio = full_lockdowns / max(episode_length, 1)
    if lockdown_ratio < 0.1:
        ef = 5
    elif lockdown_ratio < 0.3:
        ef = 2
    else:
        ef = 0
    score += ef
    breakdown["efficiency_bonus"] = ef

    breakdown["total_score"] = score
    breakdown["total_reward"] = round(total_reward, 2)
    breakdown["grade"] = _letter_grade(score)

    return breakdown


def _letter_grade(score):
    if score >= 90: return "A+"
    if score >= 80: return "A"
    if score >= 70: return "B"
    if score >= 60: return "C"
    if score >= 50: return "D"
    return "F"


def llm_grade_prompt(episode_summary):
    """
    Generate a prompt for LLM-based grading (OpenEnv spec).
    """
    return f"""
You are an expert epidemiologist and AI policy evaluator.

Evaluate the following epidemic containment episode:

{episode_summary}

Grade the agent's performance on:
1. Was the outbreak contained effectively?
2. Were economic impacts minimized?
3. Were policy decisions proportional and timely?
4. Was the vaccination strategy effective?

Provide a score out of 100 and explain your reasoning.
"""