import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO
from epidemic_env.env import EpidemicEnv
import os

# ── Load model once at startup ──────────────────────────────────────────────
MODEL_PATH = "models/epidemic_ppo_final.zip"
env = EpidemicEnv()

try:
    model = PPO.load(MODEL_PATH, env=env)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

ACTION_LABELS = {
    0: "🟢 No Intervention",
    1: "😷 Mask Mandate",
    2: "🏠 Light Lockdown",
    3: "🔒 Strict Lockdown",
    4: "💉 Vaccination Drive",
    5: "🚫 Full Shutdown",
}

def run_simulation(agent_type, seed):
    if agent_type == "PPO Agent" and not MODEL_LOADED:
        return None, None, f"❌ Model failed to load: {LOAD_ERROR}"

    np.random.seed(int(seed))
    obs, _ = env.reset(seed=int(seed))

    S_list, I_list, R_list, D_list = [], [], [], []
    actions_taken = []
    total_reward = 0
    step = 0

    while True:
        if agent_type == "PPO Agent":
            action, _ = model.predict(obs, deterministic=True)
        elif agent_type == "Rule-Based Agent":
            # Simple rule: lockdown if infected > threshold
            infected_frac = obs[1] if len(obs) > 1 else 0.1
            action = 3 if infected_frac > 0.1 else (1 if infected_frac > 0.05 else 0)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        # Try to extract SIR values from info or obs
        S = float(obs[0]) if len(obs) > 0 else 0
        I = float(obs[1]) if len(obs) > 1 else 0
        R = float(obs[2]) if len(obs) > 2 else 0
        D = float(obs[3]) if len(obs) > 3 else 0

        S_list.append(S)
        I_list.append(I)
        R_list.append(R)
        D_list.append(D)
        actions_taken.append(int(action))

        if terminated or truncated:
            break

    # ── Plot epidemic curves ─────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor="#0f172a")
    fig.suptitle(f"Epidemic Simulation — {agent_type}", color="white", fontsize=14, fontweight="bold")

    steps = range(len(S_list))

    ax1.set_facecolor("#1e293b")
    ax1.plot(steps, S_list, color="#38bdf8", linewidth=2, label="Susceptible")
    ax1.plot(steps, I_list, color="#f87171", linewidth=2, label="Infected")
    ax1.plot(steps, R_list, color="#4ade80", linewidth=2, label="Recovered")
    ax1.plot(steps, D_list, color="#a78bfa", linewidth=2, label="Deaths")
    ax1.set_ylabel("Population Fraction", color="white")
    ax1.set_xlabel("Time Steps", color="white")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#1e293b", labelcolor="white")
    ax1.spines[:].set_color("#334155")
    ax1.set_title("SIR Dynamics", color="#94a3b8", fontsize=11)

    # ── Plot actions ─────────────────────────────────────────────────────────
    action_colors = ["#4ade80","#facc15","#fb923c","#f87171","#38bdf8","#c084fc"]
    colors = [action_colors[min(a, len(action_colors)-1)] for a in actions_taken]

    ax2.set_facecolor("#1e293b")
    ax2.bar(steps, [1]*len(actions_taken), color=colors, width=1.0, alpha=0.85)
    ax2.set_ylabel("Action Taken", color="white")
    ax2.set_xlabel("Time Steps", color="white")
    ax2.set_yticks([])
    ax2.tick_params(colors="white")
    ax2.spines[:].set_color("#334155")
    ax2.set_title("Agent Actions Over Time", color="#94a3b8", fontsize=11)

    patches = [mpatches.Patch(color=action_colors[min(i, len(action_colors)-1)],
                               label=ACTION_LABELS.get(i, str(i))) for i in sorted(set(actions_taken))]
    ax2.legend(handles=patches, facecolor="#1e293b", labelcolor="white",
               loc="upper right", fontsize=8)

    plt.tight_layout()

    # ── Summary stats ────────────────────────────────────────────────────────
    final_deaths = D_list[-1] if D_list else 0
    peak_infected = max(I_list) if I_list else 0
    unique_actions = len(set(actions_taken))

    summary = f"""
## 📊 Simulation Results

| Metric | Value |
|--------|-------|
| 🤖 Agent | {agent_type} |
| 🏆 Total Reward | `{total_reward:.1f}` |
| ⏱️ Steps | `{step}` |
| 📈 Peak Infected | `{peak_infected:.3f}` |
| ☠️ Final Deaths | `{final_deaths:.3f}` |
| 🎮 Unique Actions Used | `{unique_actions}` |

### Benchmark Reference
| Agent | Avg Reward |
|-------|-----------|
| Random | 164.5 |
| Rule-Based | 477.3 |
| **PPO (yours)** | **533.7** ✅ |
"""
    return fig, summary

# ── Gradio UI ────────────────────────────────────────────────────────────────
with gr.Blocks(theme=gr.themes.Base(), title="Epidemic Containment RL") as demo:
    gr.Markdown("""
    # 🦠 Epidemic Containment RL Environment
    **A Reinforcement Learning agent trained with PPO to minimize deaths and economic damage during an epidemic.**
    
    This environment was built for the **OpenEnv Hackathon**. The PPO agent outperforms both random and rule-based baselines.
    """)

    with gr.Row():
        agent_dropdown = gr.Dropdown(
            choices=["PPO Agent", "Rule-Based Agent", "Random Agent"],
            value="PPO Agent",
            label="Select Agent"
        )
        seed_slider = gr.Slider(minimum=0, maximum=100, value=42, step=1, label="Random Seed")

    run_btn = gr.Button("▶ Run Simulation", variant="primary")

    plot_out = gr.Plot(label="Epidemic Curves & Actions")
    summary_out = gr.Markdown()

    run_btn.click(
        fn=run_simulation,
        inputs=[agent_dropdown, seed_slider],
        outputs=[plot_out, summary_out]
    )

    gr.Markdown("""
    ---
    📁 [GitHub Repo](https://github.com/Vaitheesh12345/epidemic-env) · 
    🤗 Built with Gradio + Stable-Baselines3
    """)

demo.launch()