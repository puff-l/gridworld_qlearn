# GridWorld Q-Learning

基于 **表格型 Q-Learning** 的网格世界强化学习演示项目，使用 Pygame 可视化训练与测试过程。

## 功能概览

- **GridWorld 环境**：自定义网格地图，支持墙壁、陷阱、奖励格与终点
- **Q-Learning 智能体**：ε-greedy 策略、可调学习率与折扣因子
- **Pygame 界面**：实时显示网格、智能体位置与当前 episode 信息
- **训练结束后**：自动绘制 Episode 回报、步数、成功率（含滑动平均）

## 环境要求

- Python 3.7+
- 依赖：`numpy`、`pygame`、`matplotlib`

安装依赖：

```bash
pip install numpy pygame matplotlib
```

## 快速开始

```bash
python main.py
```

启动后会打开 Pygame 窗口，默认进入**训练模式**，训练满 400 个 episode 后自动关闭窗口并弹出三张 matplotlib 曲线图。

## 环境说明（GridWorld）

### 格子类型

| 类型码 | 含义       | 说明                    |
|--------|------------|-------------------------|
| 0      | 空地       | 可通行，每步有步进代价  |
| 1      | 墙         | 不可通行，撞墙有惩罚    |
| 2      | 陷阱       | 进入即终止，大额负奖励  |
| 3      | 奖励格     | 小额正奖励，可重复经过  |
| 4      | 起点       | 每局重置位置            |
| 5      | 终点       | 到达即终止，正奖励      |

### 动作与奖励

- **动作**：上(↑)、下(↓)、左(←)、右(→)，共 4 个离散动作
- **默认奖励**：
  - 每步：`step_cost = -0.05`
  - 撞墙：`wall_penalty = -1.0`
  - 陷阱：`trap_penalty = -10.0`
  - 到达终点：`goal_reward = 10.0`
  - 经过奖励格：`bonus_reward = 0.2`
- **单局上限**：`max_steps = 200`，超时也会终止

可选 **reward shaping**（基于到终点的曼哈顿距离），在 `GridWorld` 中通过 `use_shaping=True`、`shaping_lambda=0.1` 开启；当前 MVP 默认关闭。

### 默认地图

`build_default_map()` 提供 10×10 迷宫：起点 (0,0)，终点 (9,9)，含墙壁、若干陷阱与奖励格。

## 键盘操作

| 按键   | 功能                     |
|--------|--------------------------|
| **T**  | 切换训练 / 暂停          |
| **Y**  | 运行一次贪婪测试 episode |
| **S**  | 切换速度（正常 / 加速）  |
| **R**  | 重置当前 episode（保留 Q 表） |
| **ESC**| 退出程序                 |

## 算法与参数

- **算法**：表格型 Q-Learning，更新式  
  `Q(s,a) += α * (r + γ * max_a' Q(s',a') - Q(s,a))`
- **典型参数**（在 `main()` 中）：
  - `alpha=0.2`，`gamma=0.95`
  - `eps_start=1.0`，`eps_end=0.05`，`eps_decay=0.995`

训练满 400 个 episode 后，程序会绘制：

1. Episode Return（20 步滑动平均）
2. Episode Steps（20 步滑动平均）
3. Success Rate（是否到达终点，20 步滑动平均）

## 项目结构

```
gridworld_qlearn/
├── main.py      # 环境、Q-Learning 智能体、Pygame UI、训练/测试主循环
└── README.md    # 本说明
```

## 许可证

按项目需要自行添加。
