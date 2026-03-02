

# GridWorld Q-Learning (Pygame)

A compact **tabular Q-learning** project built around a custom **GridWorld** environment and a **Pygame** UI. The agent learns to navigate mazes with walls, traps, and collectible bonuses, then demonstrates its learned behavior via greedy inference.

This repo is designed to be easy to run, easy to demo, and easy to explain in a course report.

---

## What this project does

- Provides **5 fixed 10×10 maps** (Map 1–5).
- Trains a **separate Q-table per map** (tabular Q-learning does not generalize across different layouts).
- Runs an **automatic curriculum**:
  1. Train **Map 1 → Map 4**, **50 episodes per map**.
  2. Automatically switch to **Map 5** and run **greedy inference** continuously.
- Saves/loads Q-tables to/from disk so you can resume training or demo without retraining.

---

## Environment definition

### Cell types

| Code | Name  | Behavior |
|------|-------|----------|
| 0 | Free  | Walkable |
| 1 | Wall  | Not walkable (invalid move) |
| 2 | Trap  | Terminal (episode ends) |
| 3 | Bonus | **Collectible**: gives reward and **disappears** for the rest of the episode |
| 4 | Start | Episode reset position |
| 5 | Goal  | Terminal success (episode ends) |

### State space (tabular)

The state is the agent position \((x,y)\) encoded as:

\[
\text{state\_id} = y \cdot W + x
\]

### Action space

4 discrete moves:

- Up (↑), Down (↓), Left (←), Right (→)

Invalid moves (into walls or out of bounds) keep the agent in place and apply a penalty.

---

## Reward function (defaults)

The default reward settings in `GridWorld` are:

- Step cost: `-0.05`
- Invalid move (wall/boundary): `-1.0`
- Trap (terminal): `-10.0`
- Goal (terminal): `+10.0`
- Bonus collected: `+0.2` (then the bonus cell becomes Free until next reset)
- Episode timeout: `max_steps = 200`

### Optional reward shaping

A Manhattan-distance shaping term exists in the environment (`use_shaping=True`), but the current MVP keeps shaping **off** by default.

---

## Q-learning

This project uses the standard tabular Q-learning update:

\[
Q(s,a) \leftarrow Q(s,a) + \alpha\Big(r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Big)
\]

Default hyperparameters:

- Learning rate: `alpha = 0.2`
- Discount factor: `gamma = 0.95`
- ε-greedy exploration: `eps_start = 1.0`, `eps_end = 0.05`, `eps_decay = 0.995`

### Greedy tie-breaking (important)

When acting greedily (ε=0), if multiple actions share the same maximum Q-value (common on untrained states), the agent **breaks ties randomly** rather than always choosing the first action. This prevents “stuck” behavior when a row of Q-values is all zeros.

---

## Saving and loading Q-tables

Q-tables are stored in the `q_tables/` directory:

- `q_tables/q_table_map1.npy`
- `q_tables/q_table_map2.npy`
- `q_tables/q_table_map3.npy`
- `q_tables/q_table_map4.npy`
- `q_tables/q_table_map5.npy`

On startup, the program loads any existing files automatically.

---

## Installation

### Requirements

- Python 3.8+
- `numpy`, `pygame`, `matplotlib`

Install:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy pygame matplotlib
```

---

## Run

```bash
python main.py
```

A Pygame window opens and the program starts training the curriculum.

---

## Controls (Keyboard)

| Key | Description |
|-----|-------------|
| **T** | Toggle pause (and keeps training enabled while in Maps 1–4) |
| **S** | Toggle speed (normal / fast) |
| **R** | Reset the current episode (does not clear Q-table) |
| **Y** | Start greedy inference (test) from the current map |
| **P** | Save the current map’s Q-table to `q_tables/` |
| **1–5** | Switch to Map 1–5 manually |
| **ESC** | Quit |

> Note: The default flow is automatic training on Maps 1–4 and greedy inference on Map 5. Manual switching is mainly for demos.

---

## How to demo (suggested)

1. Run `python main.py`.
2. Let it train through Map 1–4 (50 episodes each).
3. Watch it switch to Map 5 and perform greedy inference.
4. Press **1–5** to show different maps; press **Y** to force greedy inference; press **P** to save.

---

## Project structure

```text
gridworld_qlearn/
├── main.py
├── README.md
└── q_tables/           # generated (saved Q-tables)
```

---

## Notes / extensions

If you want to extend this project:

- Enable reward shaping and compare learning curves.
- Add more maps or generate procedural maps.
- Log metrics to disk (CSV/NPZ) for report-grade analysis.
- Replace tabular Q-learning with a function approximator (e.g., DQN) to study generalization.

---

## License

This repository is intended for educational use. Add a license if required by your course.