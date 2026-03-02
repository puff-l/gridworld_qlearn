import sys
import time
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pygame
import matplotlib.pyplot as plt
import os


# -------------------------
# 1) Environment: GridWorld
# -------------------------
@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool
    info: dict


class GridWorld:
    """
    Cell types:
      0 = Free
      1 = Wall
      2 = Trap (terminal)
      3 = Bonus (collectible; disappears after collection within an episode)
      4 = Start
      5 = Goal (terminal)
    """
    ACTIONS = {
        0: (0, -1),  # Up
        1: (0,  1),  # Down
        2: (-1, 0),  # Left
        3: (1,  0),  # Right
    }

    def __init__(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int],
                 step_cost: float = -0.05,
                 wall_penalty: float = -1.0,
                 trap_penalty: float = -10.0,
                 goal_reward: float = 10.0,
                 bonus_reward: float = 0.2,
                 use_shaping: bool = False,
                 shaping_lambda: float = 0.1,
                 max_steps: int = 200):
        # Keep an immutable copy of the initial map so we can reset collectibles each episode
        self.original_grid = grid.copy()
        # Work on a mutable per-episode grid (bonus cells can disappear when collected)
        self.grid = grid.copy()
        self.H, self.W = grid.shape
        self.start = start
        self.goal = goal

        self.step_cost = step_cost
        self.wall_penalty = wall_penalty
        self.trap_penalty = trap_penalty
        self.goal_reward = goal_reward
        self.bonus_reward = bonus_reward

        self.use_shaping = use_shaping
        self.shaping_lambda = shaping_lambda
        self.max_steps = max_steps

        self.pos = start
        self.steps = 0

    def reset(self) -> int:
        # Restore the grid so collectibles (bonus cells) are available again each episode
        self.grid = self.original_grid.copy()
        self.pos = self.start
        self.steps = 0
        return self._state_id(self.pos)

    def set_map(self, grid: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]):
        """Switch to a new map (same HxW assumed). Resets collectibles and position."""
        self.original_grid = grid.copy()
        self.grid = grid.copy()
        self.H, self.W = grid.shape
        self.start = start
        self.goal = goal
        self.reset()

    def _state_id(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        return y * self.W + x

    def _manhattan(self, pos: Tuple[int, int]) -> int:
        x, y = pos
        gx, gy = self.goal
        return abs(x - gx) + abs(y - gy)

    def step(self, action: int) -> StepResult:
        self.steps += 1
        x, y = self.pos
        dx, dy = self.ACTIONS[action]
        nx, ny = x + dx, y + dy

        info = {"action": action, "invalid_move": False, "cell_type": None}

        # invalid move: boundary or wall
        if not (0 <= nx < self.W and 0 <= ny < self.H) or self.grid[ny, nx] == 1:
            info["invalid_move"] = True
            reward = self.wall_penalty
            done = False
            next_pos = (x, y)
        else:
            next_pos = (nx, ny)
            cell = self.grid[ny, nx]
            info["cell_type"] = int(cell)

            # baseline reward
            reward = self.step_cost
            done = False

            if cell == 2:  # Trap
                reward += self.trap_penalty
                done = True
            elif cell == 5:  # Goal
                reward += self.goal_reward
                done = True
            elif cell == 3:  # Bonus
                reward += self.bonus_reward
                # Bonus disappears after being collected
                self.grid[ny, nx] = 0

        # shaping (optional): encourage moving closer to goal
        if self.use_shaping:
            d_before = self._manhattan((x, y))
            d_after = self._manhattan(next_pos)
            reward += self.shaping_lambda * (d_before - d_after)

        self.pos = next_pos

        # max steps termination
        if self.steps >= self.max_steps:
            done = True
            info["timeout"] = True

        return StepResult(
            next_state=self._state_id(self.pos),
            reward=reward,
            done=done,
            info=info
        )


def build_maps() -> List[Tuple[str, np.ndarray, Tuple[int, int], Tuple[int, int]]]:
    """Return 5 fixed 10x10 maps: (name, grid, start, goal)."""

    def make_base() -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        H, W = 10, 10
        grid = np.zeros((H, W), dtype=np.int32)
        start = (0, 0)
        goal = (9, 9)
        grid[start[1], start[0]] = 4
        grid[goal[1], goal[0]] = 5
        return grid, start, goal

    maps: List[Tuple[str, np.ndarray, Tuple[int, int], Tuple[int, int]]] = []

    # Map 1: original MVP map
    grid, start, goal = make_base()
    walls = [
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
        (5, 5), (5, 6), (5, 7),
        (7, 2), (7, 3), (7, 4), (7, 5),
        (1, 7), (2, 7), (3, 7), (4, 7),
    ]
    for x, y in walls:
        if (x, y) not in [start, goal]:
            grid[y, x] = 1

    traps = [(4, 2), (6, 1), (8, 6)]
    for x, y in traps:
        if (x, y) not in [start, goal]:
            grid[y, x] = 2

    bonuses = [(1, 3), (4, 6), (8, 2)]
    for x, y in bonuses:
        if (x, y) not in [start, goal]:
            grid[y, x] = 3

    maps.append(("Map 1", grid, start, goal))

    # Map 2: corridor + detour
    grid, start, goal = make_base()
    walls = [
        (1, 1), (2, 1), (3, 1), (4, 1), (5, 1),
        (5, 2), (5, 3), (5, 4),
        (3, 3), (3, 4), (3, 5),
        (7, 6), (8, 6),
        (1, 7), (2, 7), (3, 7), (4, 7),
    ]
    for x, y in walls:
        if (x, y) not in [start, goal]:
            grid[y, x] = 1

    traps = [(6, 2), (7, 3), (2, 6)]
    for x, y in traps:
        if (x, y) not in [start, goal]:
            grid[y, x] = 2

    bonuses = [(1, 4), (6, 5), (8, 1)]
    for x, y in bonuses:
        if (x, y) not in [start, goal]:
            grid[y, x] = 3

    maps.append(("Map 2", grid, start, goal))

    # Map 3: central block with two openings
    grid, start, goal = make_base()
    walls = []
    for x in range(2, 8):
        walls.append((x, 4))
    for y in range(2, 8):
        walls.append((4, y))
    # openings
    walls.remove((4, 3))
    walls.remove((6, 4))
    for x, y in walls:
        if (x, y) not in [start, goal]:
            grid[y, x] = 1

    traps = [(2, 2), (7, 7), (6, 2)]
    for x, y in traps:
        if (x, y) not in [start, goal]:
            grid[y, x] = 2

    bonuses = [(1, 5), (5, 1), (8, 5)]
    for x, y in bonuses:
        if (x, y) not in [start, goal]:
            grid[y, x] = 3

    maps.append(("Map 3", grid, start, goal))

    # Map 4: zig-zag walls
    grid, start, goal = make_base()
    walls = [
        (1, 2), (2, 2), (3, 2),
        (3, 3), (3, 4),
        (2, 4), (1, 4),
        (5, 5), (6, 5), (7, 5),
        (7, 6), (7, 7),
        (6, 7), (5, 7),
    ]
    for x, y in walls:
        if (x, y) not in [start, goal]:
            grid[y, x] = 1

    traps = [(4, 3), (6, 6), (2, 8)]
    for x, y in traps:
        if (x, y) not in [start, goal]:
            grid[y, x] = 2

    bonuses = [(0, 5), (4, 6), (9, 3)]
    for x, y in bonuses:
        if (x, y) not in [start, goal]:
            grid[y, x] = 3

    maps.append(("Map 4", grid, start, goal))

    # Map 5: harder test map
    grid, start, goal = make_base()
    walls = [
        (2, 0), (2, 1), (2, 2),
        (4, 2), (5, 2), (6, 2),
        (6, 3), (6, 4),
        (1, 5), (2, 5), (3, 5),
        (8, 6), (8, 7),
        (4, 7), (5, 7), (6, 7),
    ]
    for x, y in walls:
        if (x, y) not in [start, goal]:
            grid[y, x] = 1

    traps = [(3, 1), (7, 4), (5, 8)]
    for x, y in traps:
        if (x, y) not in [start, goal]:
            grid[y, x] = 2

    bonuses = [(1, 3), (7, 1), (3, 8)]
    for x, y in bonuses:
        if (x, y) not in [start, goal]:
            grid[y, x] = 3

    maps.append(("Map 5", grid, start, goal))

    return maps


# -------------------------
# 2) Tabular Q-Learning Agent
# -------------------------
class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int,
                 alpha: float = 0.2, gamma: float = 0.95,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: float = 0.995,
                 seed: int = 42):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        random.seed(seed)
        np.random.seed(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def act(self, state: int, greedy: bool = False) -> int:
        """Select an action using ε-greedy. If choosing greedily, break ties randomly."""
        if greedy or random.random() > self.eps:
            q_row = self.Q[state]
            max_q = float(np.max(q_row))
            best_actions = np.flatnonzero(q_row == max_q)
            return int(random.choice(best_actions))
        return random.randint(0, self.n_actions - 1)

    def update(self, s: int, a: int, r: float, s_next: int, done: bool):
        best_next = 0.0 if done else float(np.max(self.Q[s_next]))
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.eps = max(self.eps_end, self.eps * self.eps_decay)


# -------------------------
# 3) Pygame UI
# -------------------------
class GridUI:
    def __init__(self, env: GridWorld, cell_size: int = 52, pad: int = 10):
        pygame.init()
        self.env = env
        self.cell = cell_size
        self.pad = pad
        self.font = pygame.font.SysFont("Menlo", 16)
        self.small = pygame.font.SysFont("Menlo", 13)

        w = env.W * cell_size + pad * 2
        h = env.H * cell_size + pad * 2 + 160
        self.screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption("GridWorld Q-Learning (MVP)")
        self.clock = pygame.time.Clock()

    def _cell_rect(self, x: int, y: int) -> pygame.Rect:
        return pygame.Rect(self.pad + x * self.cell, self.pad + y * self.cell, self.cell, self.cell)

    def draw(self, info_panel: dict):
        self.screen.fill((245, 245, 245))

        # draw grid
        for y in range(self.env.H):
            for x in range(self.env.W):
                rect = self._cell_rect(x, y)
                cell = int(self.env.grid[y, x])

                if cell == 1:      color = (60, 60, 60)      # wall
                elif cell == 2:    color = (220, 80, 80)     # trap
                elif cell == 3:    color = (120, 200, 120)   # bonus
                elif cell == 5:    color = (120, 140, 240)   # goal
                elif cell == 4:    color = (240, 210, 120)   # start
                else:              color = (230, 230, 230)   # free

                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (200, 200, 200), rect, 1)

        # draw agent
        ax, ay = self.env.pos
        arect = self._cell_rect(ax, ay)
        pygame.draw.circle(self.screen, (40, 120, 200), arect.center, self.cell // 3)

        # info panel
        base_y = self.pad + self.env.H * self.cell + 12
        lines = [
            f"Mode: {info_panel.get('mode','')}   Ep(map): {info_panel.get('episode',0)}   Step: {info_panel.get('step',0)}",
            f"Action: {info_panel.get('action_str','')}   Reward: {info_panel.get('reward',0):+.3f}   Return: {info_panel.get('return',0):+.3f}",
            f"Epsilon: {info_panel.get('epsilon',0):.3f}   Done: {info_panel.get('done',False)}   InvalidMove: {info_panel.get('invalid',False)}"
        ]
        for i, txt in enumerate(lines):
            surf = self.font.render(txt, True, (30, 30, 30))
            self.screen.blit(surf, (self.pad, base_y + i * 28))

        help_txt = "Keys: [T] Train/Pause  [Y] Test (greedy)  [S] Speed  [R] Reset  [P] Save Q  [1-5] Switch Map  [ESC] Quit"
        surf2 = self.small.render(help_txt, True, (60, 60, 60))
        self.screen.blit(surf2, (self.pad, base_y + 96))

        pygame.display.flip()

    def tick(self, fps: int):
        self.clock.tick(fps)

    def quit(self):
        pygame.quit()


# -------------------------
# 4) Training / Testing Loop
# -------------------------
def action_to_str(a: int) -> str:
    return {0: "↑", 1: "↓", 2: "←", 3: "→"}.get(a, str(a))


def main():
    maps = build_maps()  # 5 maps: (name, grid, start, goal)
    current_map_idx = 0
    map_name, grid, start, goal = maps[current_map_idx]

    # Auto curriculum: train Map 1-4, then run greedy inference on Map 5
    TRAIN_MAP_COUNT = 4          # train Map 1..4
    EPISODES_PER_MAP = 50        # episodes per training map

    env = GridWorld(
        grid=grid, start=start, goal=goal,
        use_shaping=False,
        shaping_lambda=0.1,
        max_steps=200
    )

    n_states = env.H * env.W
    n_actions = 4

    # One Q-table per map
    q_tables: List[np.ndarray] = [np.zeros((n_states, n_actions), dtype=np.float32) for _ in range(len(maps))]

    agent = QLearningAgent(
        n_states=n_states, n_actions=n_actions,
        alpha=0.2, gamma=0.95,
        eps_start=1.0, eps_end=0.05, eps_decay=0.995
    )
    agent.Q = q_tables[current_map_idx]

    # Save / Load Q-tables
    Q_DIR = "q_tables"
    os.makedirs(Q_DIR, exist_ok=True)

    def q_path(idx: int) -> str:
        return os.path.join(Q_DIR, f"q_table_map{idx+1}.npy")

    for i in range(len(maps)):
        path = q_path(i)
        if os.path.exists(path):
            q_tables[i] = np.load(path)
            print(f"Loaded Q table for Map {i+1} from {path}")

    agent.Q = q_tables[current_map_idx]

    ui = GridUI(env)

    training = True
    paused = False
    fast = False

    ep_returns: List[float] = []
    ep_steps: List[int] = []
    ep_success: List[int] = []

    # Per-map episode counters
    episodes_seen: List[int] = [0 for _ in range(len(maps))]

    state = env.reset()
    ep_return = 0.0
    step_in_ep = 0

    pending_test = False
    test_state = None
    test_return = 0.0
    test_step = 0

    def end_episode(done_reason: str):
        nonlocal state, ep_return, step_in_ep, current_map_idx, map_name, grid, start, goal, training, pending_test, test_state, test_return, test_step, paused

        success = 1 if env.pos == env.goal else 0
        ep_returns.append(ep_return)
        ep_steps.append(step_in_ep)
        ep_success.append(success)

        episodes_seen[current_map_idx] += 1
        ep_on_map = episodes_seen[current_map_idx]

        agent.decay_epsilon()

        # Save after episode 1 and episode 50 (per map)
        if ep_on_map == 1 or ep_on_map % 50 == 0:
            path = q_path(current_map_idx)
            np.save(path, agent.Q)
            print(f"Auto-saved Q table for Map {current_map_idx+1} to {path} at map-episode {ep_on_map}")

        # Curriculum switching
        if training and current_map_idx < TRAIN_MAP_COUNT and ep_on_map >= EPISODES_PER_MAP:
            # Final save for this training map
            path = q_path(current_map_idx)
            np.save(path, agent.Q)
            print(f"Finished training {map_name}: saved Q to {path}")

            # Advance to next map
            current_map_idx += 1
            map_name, grid, start, goal = maps[current_map_idx]
            env.set_map(grid, start, goal)
            agent.Q = q_tables[current_map_idx]

            # Arrive at Map 5 => greedy inference
            if current_map_idx >= TRAIN_MAP_COUNT:
                agent.eps = 0.0
                training = False
                paused = False
                pending_test = True
                test_state = env.reset()
                test_return = 0.0
                test_step = 0

                state = test_state
                ep_return = 0.0
                step_in_ep = 0
                print(f"Switched to {map_name} for greedy inference")
                return

            # Otherwise keep training on next map
            state = env.reset()
            ep_return = 0.0
            step_in_ep = 0
            print(f"Switched to {map_name} for training")
            return

        state = env.reset()
        ep_return = 0.0
        step_in_ep = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ui.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    ui.quit()
                    sys.exit(0)

                if event.key == pygame.K_t:
                    # Toggle pause. Only allow forcing training while in training maps.
                    if current_map_idx < TRAIN_MAP_COUNT:
                        training = True
                    paused = not paused

                if event.key == pygame.K_s:
                    fast = not fast

                if event.key == pygame.K_r:
                    state = env.reset()
                    ep_return = 0.0
                    step_in_ep = 0

                if event.key == pygame.K_y:
                    agent.eps = 0.0
                    training = False
                    pending_test = True
                    test_state = env.reset()
                    test_return = 0.0
                    test_step = 0

                if event.key == pygame.K_p:
                    path = q_path(current_map_idx)
                    np.save(path, agent.Q)
                    print(f"Saved Q table for Map {current_map_idx+1} to {path}")

                # manual map switch (still supported)
                if pygame.K_1 <= event.key <= pygame.K_5:
                    idx = event.key - pygame.K_1
                    if 0 <= idx < len(maps):
                        current_map_idx = idx
                        map_name, grid, start, goal = maps[current_map_idx]
                        env.set_map(grid, start, goal)
                        agent.Q = q_tables[current_map_idx]
                        state = env.reset()
                        ep_return = 0.0
                        step_in_ep = 0
                        if not training:
                            agent.eps = 0.0
                        print(f"Switched to {map_name}")

        if training and not paused:
            a = agent.act(state, greedy=False)
            res = env.step(a)
            agent.update(state, a, res.reward, res.next_state, res.done)

            state = res.next_state
            ep_return += res.reward
            step_in_ep += 1

            if res.done:
                end_episode("done")

            panel = {
                "mode": f"TRAIN {map_name}" + (" (FAST)" if fast else ""),
                "episode": episodes_seen[current_map_idx],
                "step": step_in_ep,
                "action_str": action_to_str(a),
                "reward": res.reward,
                "return": ep_return,
                "epsilon": agent.eps,
                "done": res.done,
                "invalid": res.info.get("invalid_move", False)
            }

        elif (not training) and pending_test:
            a = agent.act(test_state, greedy=True)
            res = env.step(a)
            test_state = res.next_state
            test_return += res.reward
            test_step += 1

            if res.done:
                # continuous inference demo
                test_state = env.reset()
                test_return = 0.0
                test_step = 0

            panel = {
                "mode": f"TEST (GREEDY) {map_name}",
                "episode": episodes_seen[current_map_idx],
                "step": test_step,
                "action_str": action_to_str(a),
                "reward": res.reward,
                "return": test_return,
                "epsilon": agent.eps,
                "done": res.done,
                "invalid": res.info.get("invalid_move", False)
            }
        else:
            panel = {
                "mode": f"PAUSED {map_name}" if paused else f"IDLE {map_name}",
                "episode": episodes_seen[current_map_idx],
                "step": step_in_ep,
                "action_str": "-",
                "reward": 0.0,
                "return": ep_return,
                "epsilon": agent.eps,
                "done": False,
                "invalid": False
            }

        ui.draw(panel)

        fps = 60 if not fast else 600
        ui.tick(fps)


if __name__ == "__main__":
    main()