import sys
import time
import random
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import pygame
import matplotlib.pyplot as plt


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
      3 = Bonus (small reward, repeatable in MVP)
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
        self.grid = grid
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
        self.pos = self.start
        self.steps = 0
        return self._state_id(self.pos)

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


def build_default_map() -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    """
    A simple 10x10 maze.
    """
    H, W = 10, 10
    grid = np.zeros((H, W), dtype=np.int32)

    start = (0, 0)
    goal = (9, 9)

    grid[start[1], start[0]] = 4
    grid[goal[1], goal[0]] = 5

    # Walls
    walls = [
        # vertical-ish barrier
        (2, 0), (2, 1), (2, 2), (2, 3), (2, 4),
        (5, 5), (5, 6), (5, 7),
        (7, 2), (7, 3), (7, 4), (7, 5),
        (1, 7), (2, 7), (3, 7), (4, 7),
    ]
    for x, y in walls:
        if (x, y) not in [start, goal]:
            grid[y, x] = 1

    # Traps (terminal)
    traps = [(4, 2), (6, 1), (8, 6)]
    for x, y in traps:
        if (x, y) not in [start, goal]:
            grid[y, x] = 2

    # Bonus cells
    bonuses = [(1, 3), (4, 6), (8, 2)]
    for x, y in bonuses:
        if (x, y) not in [start, goal]:
            grid[y, x] = 3

    return grid, start, goal


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
        if greedy or random.random() > self.eps:
            return int(np.argmax(self.Q[state]))
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
        self.font = pygame.font.SysFont("Menlo", 18)
        self.small = pygame.font.SysFont("Menlo", 14)

        w = env.W * cell_size + pad * 2
        h = env.H * cell_size + pad * 2 + 120
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

                # Colors (keep simple)
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
        center = arect.center
        pygame.draw.circle(self.screen, (40, 120, 200), center, self.cell // 3)

        # info panel
        base_y = self.pad + self.env.H * self.cell + 20
        lines = [
            f"Mode: {info_panel.get('mode','')}   Episode: {info_panel.get('episode',0)}   Step: {info_panel.get('step',0)}",
            f"Action: {info_panel.get('action_str','')}   Reward: {info_panel.get('reward',0):+.3f}   Return: {info_panel.get('return',0):+.3f}",
            f"Epsilon: {info_panel.get('epsilon',0):.3f}   Done: {info_panel.get('done',False)}   InvalidMove: {info_panel.get('invalid',False)}"
        ]
        for i, txt in enumerate(lines):
            surf = self.font.render(txt, True, (30, 30, 30))
            self.screen.blit(surf, (self.pad, base_y + i * 28))

        help_txt = "Keys: [T] Train/Pause  [Y] Test(1 episode)  [S] Toggle Speed  [R] Reset  [ESC] Quit"
        surf2 = self.small.render(help_txt, True, (60, 60, 60))
        self.screen.blit(surf2, (self.pad, base_y + 90))

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
    grid, start, goal = build_default_map()
    env = GridWorld(
        grid=grid, start=start, goal=goal,
        use_shaping=False,  # MVP: 先关掉，跑通后你再打开
        shaping_lambda=0.1,
        max_steps=200
    )

    n_states = env.H * env.W
    n_actions = 4
    agent = QLearningAgent(
        n_states=n_states, n_actions=n_actions,
        alpha=0.2, gamma=0.95,
        eps_start=1.0, eps_end=0.05, eps_decay=0.995
    )

    ui = GridUI(env)

    # toggles
    training = True
    paused = False
    fast = False

    # metrics
    ep_returns: List[float] = []
    ep_steps: List[int] = []
    ep_success: List[int] = []

    episode = 0
    state = env.reset()
    ep_return = 0.0
    step_in_ep = 0

    def end_episode(done_reason: str):
        nonlocal episode, state, ep_return, step_in_ep
        # success if on goal
        success = 1 if env.pos == env.goal else 0
        ep_returns.append(ep_return)
        ep_steps.append(step_in_ep)
        ep_success.append(success)

        episode += 1
        agent.decay_epsilon()
        state = env.reset()
        ep_return = 0.0
        step_in_ep = 0

    # for testing mode
    pending_test = False
    test_state = None
    test_return = 0.0
    test_step = 0

    while True:
        # handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ui.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    ui.quit()
                    sys.exit(0)
                if event.key == pygame.K_t:
                    training = True
                    paused = not paused
                if event.key == pygame.K_s:
                    fast = not fast
                if event.key == pygame.K_r:
                    # reset env + stats (keep Q table)
                    state = env.reset()
                    ep_return = 0.0
                    step_in_ep = 0
                if event.key == pygame.K_y:
                    # run one greedy test episode
                    training = False
                    pending_test = True
                    test_state = env.reset()
                    test_return = 0.0
                    test_step = 0

        # decide one step
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
                "mode": "TRAIN" + (" (FAST)" if fast else ""),
                "episode": episode,
                "step": step_in_ep,
                "action_str": action_to_str(a),
                "reward": res.reward,
                "return": ep_return,
                "epsilon": agent.eps,
                "done": res.done,
                "invalid": res.info.get("invalid_move", False)
            }

        elif (not training) and pending_test:
            # test step: greedy
            a = agent.act(test_state, greedy=True)
            res = env.step(a)
            test_state = res.next_state
            test_return += res.reward
            test_step += 1

            if res.done:
                pending_test = False

            panel = {
                "mode": "TEST (GREEDY)",
                "episode": episode,
                "step": test_step,
                "action_str": action_to_str(a),
                "reward": res.reward,
                "return": test_return,
                "epsilon": agent.eps,
                "done": res.done,
                "invalid": res.info.get("invalid_move", False)
            }
        else:
            # idle panel
            panel = {
                "mode": "PAUSED" if paused else "IDLE",
                "episode": episode,
                "step": step_in_ep,
                "action_str": "-",
                "reward": 0.0,
                "return": ep_return,
                "epsilon": agent.eps,
                "done": False,
                "invalid": False
            }

        ui.draw(panel)

        # speed control
        fps = 60 if not fast else 600  # fast mode trains faster
        ui.tick(fps)

        # optional: stop after enough episodes, auto-plot
        if training and episode == 400:
            break

    ui.quit()

    # Plot metrics (for report)
    returns = np.array(ep_returns, dtype=np.float32)
    steps = np.array(ep_steps, dtype=np.float32)
    success = np.array(ep_success, dtype=np.float32)

    def moving_avg(x, k=20):
        if len(x) < k:
            return x
        return np.convolve(x, np.ones(k)/k, mode="valid")

    plt.figure()
    plt.title("Episode Return (moving avg)")
    plt.plot(moving_avg(returns, 20))
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.show()

    plt.figure()
    plt.title("Episode Steps (moving avg)")
    plt.plot(moving_avg(steps, 20))
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.show()

    plt.figure()
    plt.title("Success Rate (moving avg)")
    plt.plot(moving_avg(success, 20))
    plt.xlabel("Episode")
    plt.ylabel("Success(Goal=1)")
    plt.ylim(-0.05, 1.05)
    plt.show()


if __name__ == "__main__":
    main()