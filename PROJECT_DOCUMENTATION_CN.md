# G1 Deploy Mujoco 项目详细说明文档

> 本文档为项目新手提供完整的技术说明，帮助你快速理解项目结构、工作原理和使用方法。

---

## 📌 项目概述

### 什么是这个项目？

**G1 Deploy Mujoco** 是一个轻量级的机器人强化学习策略部署工具。它的主要功能是：

1. 将在服务器上使用 **IsaacLab + Unitree RL Lab** 训练的机器人行走策略
2. 无需安装 IsaacSim、IsaacLab 等重型依赖
3. 直接在本地通过 **Mujoco** 物理仿真器进行可视化验证

### 应用场景

```
服务器（训练） ──► 导出策略 ──► 本地（Mujoco 仿真验证）
```

- ✅ 在高性能服务器上训练强化学习策略
- ✅ 将训练好的模型导出为 JIT/ONNX 格式
- ✅ 在本地轻松使用 Mujoco 查看机器人行走效果

---

## 🤖 什么是 Unitree G1？

**Unitree G1** 是宇树科技（Unitree Robotics）生产的一款人形机器人。本项目支持的是 **29 自由度（29-DoF）** 版本：

| 部位 | 关节数量 | 说明 |
|------|----------|------|
| 髋部（Hip） | 6 | 左右各3个（pitch/roll/yaw） |
| 膝盖（Knee） | 2 | 左右各1个 |
| 踝关节（Ankle） | 4 | 左右各2个（pitch/roll） |
| 腰部（Waist） | 3 | yaw/roll/pitch |
| 肩膀（Shoulder） | 6 | 左右各3个（pitch/roll/yaw） |
| 手肘（Elbow） | 2 | 左右各1个 |
| 手腕（Wrist） | 6 | 左右各3个（roll/pitch/yaw） |
| **合计** | **29** | 全身自由度 |

---

## 📁 项目结构详解

```
g1_deploy_mujoco/
├── deploy_mujoco.py          # 🎮 主程序：在 Mujoco 中运行策略
├── environment.yml           # 📦 Conda 环境配置文件
├── README.md                 # 项目说明
├── PROJECT_DOCUMENTATION_CN.md  # 本文档
│
├── configs/                  # ⚙️ 配置文件目录
│   └── g1_29dof_walk.yaml   #    G1 机器人的完整配置
│
├── checkpoint/               # 💾 策略模型存储目录
│   └── policy.pt            #    预训练的行走策略（JIT格式）
│
├── scripts/                  # 🔧 工具脚本
│   ├── convert_jit_to_onnx.py   # JIT → ONNX 格式转换
│   └── batch_processing.py      # 批量处理训练 checkpoint
│
├── g1_xml/                   # 🦿 Mujoco 机器人模型
│   ├── scene_29dof.xml      #    场景描述文件
│   ├── g1_29dof.xml         #    机器人本体描述
│   └── meshes/              #    64个3D网格模型文件
│
└── *.gif                     # 演示动画
```

---

## 🔧 核心文件详解

### 1. `deploy_mujoco.py` - 主部署程序

这是项目的核心，负责在 Mujoco 中运行 RL 策略。

#### 主要功能模块：

```python
# 1. 加载配置文件
with open("configs/g1_29dof_walk.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# 2. 加载 Mujoco 机器人模型
m = mujoco.MjModel.from_xml_path(xml_path)
d = mujoco.MjData(m)

# 3. 加载 PyTorch JIT 策略
policy = torch.jit.load(policy_path)

# 4. 仿真循环
with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        # 计算 PD 控制力矩
        tau = pd_control(target_dof_pos, d.qpos[7:], kps, ...)
        # 执行物理仿真步
        mujoco.mj_step(m, d)
        # 策略推理（每隔一定步数）
        if counter % control_decimation == 0:
            action = policy(obs_tensor)
```

#### 关键概念说明：

| 概念 | 说明 |
|------|------|
| **PD 控制** | 位置-速度反馈控制，计算关节力矩 |
| **观测（Observation）** | 策略的输入，包含角速度、姿态、关节位置/速度、上一帧动作 |
| **动作（Action）** | 策略输出的目标关节角度偏移量 |
| **Frame Stack** | 堆叠5帧观测，提供时序信息（共 96×5=480 维） |

---

### 2. `configs/g1_29dof_walk.yaml` - 配置文件

包含机器人部署所需的所有参数：

```yaml
# === 仿真参数 ===
simulation_duration: 60.0     # 仿真时长（秒）
simulation_dt: 0.002          # 物理仿真步长（2ms）
control_decimation: 10        # 控制频率 = 1/(0.002*10) = 50Hz

# === PD 增益 ===
kps: [100.0, 100.0, ...]      # 比例增益（刚度）
kds: [2.0, 2.0, ...]          # 微分增益（阻尼）

# === 默认关节角度（弧度） ===
default_angles: [-0.1, -0.1, 0.0, ...]

# === 观测/动作缩放因子 ===
ang_vel_scale: 0.2            # 角速度缩放
dof_pos_scale: 1.0            # 关节位置缩放
dof_vel_scale: 0.05           # 关节速度缩放
action_scale: 0.25            # 动作输出缩放

# === 尺寸信息 ===
num_actions: 29               # 动作维度
num_obs: 96                   # 单帧观测维度

# === 运动命令 ===
cmd_init: [0.5, 0, 0]         # [前进速度, 侧向速度, 转向速度]

# === 关节顺序映射 ===
policy_joints:                # 策略使用的关节顺序
  - left_hip_pitch_joint
  - right_hip_pitch_joint
  - ...
```

#### 重要：关节顺序映射

由于策略训练时的关节顺序与 Mujoco XML 中的顺序不同，需要进行映射：

```
策略顺序 ──policy_to_xml──► Mujoco 顺序
Mujoco 顺序 ──xml_to_policy──► 策略顺序
```

---

### 3. `scripts/batch_processing.py` - 批量处理脚本

将 RSL-RL 训练的 checkpoint 转换为可部署的模型。

#### 核心功能：

1. **智能加载**：支持 state_dict 和 TorchScript 两种格式
2. **导出 JIT**：可在任何设备上运行的优化模型
3. **导出 ONNX**：跨平台通用格式

#### 使用方法：

```bash
# 处理单个文件
python scripts/batch_processing.py --input_path logs/model_1000.pt --output_path ./

# 处理整个目录
python scripts/batch_processing.py --input_path logs/ --output_path ./

# 使用通配符
python scripts/batch_processing.py --input_path "logs/model_*.pt" --output_path ./
```

---

### 4. `scripts/convert_jit_to_onnx.py` - 格式转换

将 JIT 格式策略转换为 ONNX 格式。

```bash
python scripts/convert_jit_to_onnx.py \
    --jit-path checkpoint/policy.pt \
    --onnx-path exported/policy.onnx \
    --input-shape 1 480
```

---

## 📊 观测空间结构

策略的输入是一个 **480 维向量**（5帧 × 96维/帧）：

| 组成部分 | 维度 | 说明 |
|----------|------|------|
| 角速度（ω） | 3 | 机器人躯干角速度 |
| 重力方向 | 3 | 投影到机器人坐标系的重力向量 |
| 命令 | 3 | 目标速度命令 [vx, vy, ωz] |
| 关节位置 | 29 | 当前关节角度（相对默认值） |
| 关节速度 | 29 | 当前关节角速度 |
| 上一帧动作 | 29 | 策略上一次输出的动作 |
| **单帧合计** | **96** | |
| **5帧堆叠** | **480** | 最终输入维度 |

---

## 🚀 快速开始

### 1. 创建环境

```bash
# 方法1：使用 environment.yml
conda env create -f environment.yml
conda activate g1_deploy_mujoco_py

# 方法2：手动安装
pip install torch mujoco numpy pyyaml glfw onnx onnxruntime rsl-rl-lib gymnasium
```

### 2. 运行预置策略

```bash
python deploy_mujoco.py
```

### 3. 运行自己的策略

```bash
python deploy_mujoco.py --policy /path/to/your/policy.pt
```

### 4. 键盘交互

在 Mujoco 可视化窗口中：
- **空格键**：暂停/继续仿真
- **ESC**：退出
- **鼠标拖拽**：旋转视角
- **滚轮**：缩放

---

## 🔄 完整工作流程

```
┌─────────────────────────────────────────────────────────────────┐
│                     服务器端（训练）                              │
├─────────────────────────────────────────────────────────────────┤
│  1. 安装 IsaacLab + Unitree RL Lab                              │
│  2. 训练 G1 29-DoF 行走策略                                      │
│  3. 得到 checkpoint: logs/model_XXXX.pt                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     模型转换                                     │
├─────────────────────────────────────────────────────────────────┤
│  python scripts/batch_processing.py \                           │
│      --input_path logs/model_XXXX.pt \                          │
│      --output_path ./                                           │
│                                                                 │
│  → 输出: exported/policy.pt (JIT) + policy.onnx (ONNX)          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     本地端（验证）                               │
├─────────────────────────────────────────────────────────────────┤
│  1. 安装轻量级环境 (无需 IsaacLab)                               │
│  2. python deploy_mujoco.py --policy exported/policy.pt        │
│  3. 在 Mujoco 中观察机器人行走效果                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧠 技术细节

### PD 控制器

```python
def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd
```

- **target_q**: 目标关节角度（策略输出 + 默认角度）
- **q**: 当前关节角度
- **kp**: 比例增益（控制刚度）
- **target_dq**: 目标关节速度（通常为 0）
- **dq**: 当前关节速度
- **kd**: 微分增益（控制阻尼）

### 重力方向计算

```python
def get_gravity_orientation(quaternion):
    # 将世界坐标系的重力向量转换到机器人坐标系
    # 用于让策略感知机器人的倾斜状态
```

---

## 📝 常见问题

### Q1: 为什么需要关节顺序映射？

IsaacLab/RSL-RL 训练时的关节顺序与 Mujoco XML 定义的顺序不同。如果不进行映射，动作会作用到错误的关节上。

### Q2: 如何修改机器人运动速度？

编辑 `configs/g1_29dof_walk.yaml` 中的 `cmd_init` 参数：

```yaml
cmd_init: [0.5, 0, 0]  # [前进, 侧向, 转向] 单位：m/s 或 rad/s
```

### Q3: 支持其他机器人吗？

目前仅支持 Unitree G1 29-DoF。如需支持其他机器人，需要：
1. 准备 Mujoco XML 模型文件
2. 创建对应的配置文件
3. 确保关节顺序映射正确

### Q4: 如何调整仿真速度？

修改配置文件中的 `simulation_dt` 和 `control_decimation` 参数。

---

## 📚 依赖库说明

| 库名 | 版本 | 用途 |
|------|------|------|
| torch | 2.7.0 | PyTorch 深度学习框架 |
| mujoco | 3.2.6 | 物理仿真引擎 |
| numpy | - | 数值计算 |
| pyyaml | - | YAML 配置解析 |
| onnx | - | ONNX 模型格式支持 |
| onnxruntime | - | ONNX 推理运行时 |
| rsl-rl-lib | 2.3.1 | RSL-RL 训练框架 |
| gymnasium | 0.29.1 | 强化学习环境接口 |

---

## 📞 相关资源

- [Unitree RL Lab](https://github.com/unitreerobotics/unitree_rl_lab) - 训练框架
- [MuJoCo 官方文档](https://mujoco.readthedocs.io/) - 物理仿真器
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) - 强化学习库

---

*文档创建时间：2026-01-31*
