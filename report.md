

# GridWorld Maze Game Based on Tabular Q-Learning

**Demo (video):** _<add your YouTube link here>_  
**Code (GitHub):** _<https://github.com/puff-l/gridworld_qlearn>_

---

## 1. Introduction

This project implements a GridWorld maze game and trains an agent with **tabular Q-learning** to navigate from a **Start** cell to a **Goal** cell while avoiding **Walls** and **Traps** and optionally collecting **Bonus** cells. The main objective is to demonstrate how a value-based reinforcement learning algorithm can learn a near-optimal policy through trial-and-error interactions with an environment.

The project is designed to be easy to reproduce and easy to present in a course demo:

- A **Pygame** user interface renders the environment in real time and displays key signals (mode, action, step reward, cumulative return, epsilon, etc.).
- The implementation supports **five fixed maps** (Map 1–5). Because tabular Q-learning does not naturally generalize across different transition dynamics, the project maintains a **separate Q-table per map**.
- The program runs a **curriculum**: it trains on **Map 1–4** (50 episodes per map) and then switches to **Map 5** for greedy inference at a cinematic display rate (24 FPS) to demonstrate behavior on an unseen layout.
- After each training map finishes, the project automatically saves **visualizations** (episode return / steps / success rate) to disk for inclusion in the final report.

**Technology stack:** Python, NumPy, Pygame, Matplotlib.

---

## 2. Game Design

### 2.1 Rules of the Game

The game takes place on a 2D grid of size \(H\times W\) (default \(10\times 10\)). Each episode begins with the agent at the **Start** position and ends when one of the termination conditions is met.

**Cell types** are encoded as integers:

- **0 = Free:** walkable cell
- **1 = Wall:** not walkable (attempting to move into a wall results in an invalid move)
- **2 = Trap:** terminal cell (episode ends with penalty)
- **3 = Bonus:** walkable collectible (reward is granted and the bonus disappears until the next reset)
- **4 = Start:** initial cell
- **5 = Goal:** terminal success cell (episode ends with reward)

**Actions.** The agent selects one of four discrete actions each step:

\[
\mathcal{A}=\{\text{Up},\text{Down},\text{Left},\text{Right}\}
\]

If an action would move the agent outside the grid boundary or into a wall, the move is **invalid**: the agent stays in place and receives an invalid-move penalty.

**Episode termination.** An episode ends when:

1. The agent reaches the **Goal** (success), or
2. The agent steps on a **Trap** (failure), or
3. The agent exceeds a maximum number of steps (**timeout**, default 200).

### 2.2 State Space and Encoding

This project uses a compact tabular state representation based on the agent’s position \((x,y)\). The 2D position is mapped to a single integer index:

\[
\text{state\_id} = y\cdot W + x
\]

This yields \(|\mathcal{S}| = H\cdot W\) states per map. Because the environment layout differs across maps (walls/traps/bonuses), a separate Q-table is maintained per map.

### 2.3 Reward Function

The reward design is intentionally simple and interpretable:

- **Step cost:** \(-0.05\) per valid step (encourages shorter paths)
- **Invalid move (wall/boundary):** \(-1.0\)
- **Trap (terminal):** additional \(-10.0\)
- **Goal (terminal):** additional \(+10.0\)
- **Bonus collected:** additional \(+0.2\), then the bonus cell becomes a Free cell until the next reset

This reward structure balances efficiency (shorter routes) with safety (avoid traps/walls). A Manhattan-distance shaping term is implemented in the environment but disabled by default to keep the baseline behavior easy to interpret.

### 2.4 Map Design (Five Levels)

The project includes **five fixed 10×10 maps** with different wall layouts and hazard placements:

- **Map 1–4:** used for curriculum training (50 episodes each)
- **Map 5:** used for final greedy inference demo (unseen during training)

This design supports both (i) learning within a fixed Markov Decision Process (MDP) and (ii) illustrating limitations of tabular RL when transition dynamics change across environments.

### 2.5 UI Design (Pygame)

The UI renders the grid with distinct colors for each cell type (e.g., walls, traps, bonuses, goal) and draws the agent as a colored circle. An information panel at the bottom displays:

- **Mode:** TRAIN or TEST (GREEDY)
- **Ep(map):** number of episodes completed on the current map
- **Step:** step count within the current episode
- **Action:** arrow symbol (↑ ↓ ← →)
- **Reward / Return:** step reward and cumulative return
- **Epsilon:** exploration rate
- **Done / InvalidMove:** termination and invalid-move indicator

During Map 5 inference, the display is fixed to **24 FPS** to produce a smoother “demo-like” viewing experience.

---

## 3. Implementation of Q-Learning

### 3.1 Q-Learning Update Rule

Tabular Q-learning learns a state–action value function \(Q(s,a)\) and selects actions using an exploration strategy. The update rule used in this project is:

\[
Q(s,a)\leftarrow Q(s,a) + \alpha\Big(r + \gamma\max_{a'}Q(s',a') - Q(s,a)\Big)
\]

where:

- \(\alpha\) is the learning rate
- \(\gamma\) is the discount factor
- \(r\) is the immediate reward
- \(s'\) is the next state

### 3.2 Exploration Strategy (ε-greedy)

The agent uses an ε-greedy strategy:

- With probability \(\varepsilon\), select a random action (exploration).
- Otherwise, select the action with the highest Q-value (exploitation).

Hyperparameters:

- \(\alpha = 0.2\)
- \(\gamma = 0.95\)
- \(\varepsilon\) decays from 1.0 to 0.05 with a multiplicative decay factor 0.995.

### 3.3 Greedy Tie-Breaking

A subtle but important implementation detail is **tie-breaking** during greedy selection. In many states (especially unseen states), multiple actions can share the same maximum Q-value (often all zeros). If the agent always picks the first argmax action, it can become stuck repeatedly attempting an invalid move.

To avoid this, greedy selection breaks ties **randomly** among actions with maximal Q-value. This ensures the agent continues to move even when Q-values are not informative.

### 3.4 Curriculum Training and Persistence

Training proceeds sequentially:

1. Train **Map 1** for 50 episodes (Q-table saved)
2. Train **Map 2** for 50 episodes
3. Train **Map 3** for 50 episodes
4. Train **Map 4** for 50 episodes
5. Switch to **Map 5** and run **greedy inference** continuously

Each map has its own Q-table saved under:

- `q_tables/q_table_map1.npy` … `q_tables/q_table_map5.npy`

On startup, any existing Q-table files are loaded automatically.

---

## 4. Experimental Results and Saved Visualizations

To provide reproducible evidence of learning, the program saves training curves for each trained map (Map 1–4) when training on that map completes.

### 4.1 Metrics

For each episode, the following are recorded:

- **Episode Return:** cumulative reward over the episode
- **Episode Steps:** number of steps taken until termination
- **Success:** 1 if the Goal is reached, 0 otherwise

### 4.2 Visualizations

At the end of training on each map, the program saves:

- `*_return.png`: moving average of episode return
- `*_steps.png`: moving average of episode steps
- `*_success.png`: moving average of success rate
- `*_metrics.npz`: raw arrays for post-processing

All outputs are written to the `results/` folder with a timestamped prefix such as:

- `results/map3_YYYYMMDD_HHMMSS_return.png`

These figures can be inserted directly into the final PDF report.

### 4.3 Interpretation

In successful learning runs, episode return increases over time while the number of steps decreases, indicating that the agent discovers shorter, safer routes to the goal. The success rate typically increases as exploration decays and the learned policy becomes more stable.

---

## 5. Challenges and Solutions

### 5.1 Reward Hacking (Bonus Farming)

An early version of the environment allowed the bonus reward to be collected repeatedly, which created a reward loophole: the agent preferred looping around bonus cells instead of pursuing the goal. This is a common RL issue known as **reward hacking**.

**Solution:** bonus cells were redesigned as **collectibles** that disappear after being collected within an episode (they reappear on reset). This prevents infinite reward accumulation and restores the intended objective.

### 5.2 Multi-Map Evaluation vs. Tabular Generalization

Using different maps for training and testing raises a fundamental limitation of tabular methods: changing the layout changes the MDP transition dynamics. A single Q-table trained on one map generally does not transfer to a different map.

**Solution:** maintain **one Q-table per map**, and use a curriculum to demonstrate learning across multiple environments. Map 5 is kept as a demo map for qualitative evaluation.

### 5.3 Greedy Policy Getting Stuck

When testing on an unseen map with an untrained Q-table, greedy action selection can become degenerate if argmax always picks the same action when Q-values tie.

**Solution:** implement random tie-breaking among actions with identical maximal Q-values.

### 5.4 Demo Usability (Speed Control)

Fast rendering is useful during training but makes demos difficult to follow.

**Solution:** enforce **24 FPS** inference speed on Map 5 for smooth visualization, while allowing accelerated training speed on Maps 1–4.

---

## 6. Conclusion

This project demonstrates a complete reinforcement learning pipeline for a discrete GridWorld maze:

- A clear MDP formulation (state, action, reward, termination)
- A working tabular Q-learning implementation with ε-greedy exploration
- A real-time UI that visualizes state transitions and learning signals
- A multi-map curriculum with persistent Q-tables
- Automatic saving of training curves for report-ready evaluation

Overall, the trained agent learns efficient navigation strategies within each map, and the project highlights both the strengths (simplicity, interpretability) and limitations (lack of cross-map generalization) of tabular Q-learning.

---

## 7. References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
2. Watkins, C. J. C. H., & Dayan, P. (1992). Q-learning. *Machine Learning*, 8, 279–292.
3. Pygame Documentation. https://www.pygame.org/docs/