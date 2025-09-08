# G1 Deploy Mujoco


| IsaacLab | this project | Mujoco |
|:--------:|:--------:|:------:|
| ![G1机器人演示](./isaaclab.gif) | <span style="font-size: 64px;">⟹</span>| ![G1机器人演示](./mujoco_deploy.gif) |



## ✨ 概览

Unitree_RL_Lab 自带 C++ 版本的 Mujoco 部署，本仓库是对 Python 版本 Mujoco 部署的补充。它可以帮助你将 `unitree_rl_lab` 训练出的结果更容易地部署到 Mujoco 环境中。我们提供了一个基础的 G1 29 自由度行走策略(`checkpoint/policy.pt`)供你尝试，你也可以将其替换为自己训练的策略。

Unitree_RL_Lab comes with a C++ implementation of Mujoco deployment, and this repository serves as a Python-based supplement. It helps you more easily deploy the results trained with unitree_rl_lab into the Mujoco environment. We provide a basic G1 29-DoF walking policy (checkpoint/policy.pt) for you to try out, and you can also replace it with your own trained policy.

## 🛠️ 步骤（中文版）

1. 参考 [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) ，训练出 **29-DoF Unitree G1** 行走策略并导出`exported/policy.pt`

2. 克隆本仓库：
   ```bash
   git clone https://github.com/RoboCubPilot/g1_deploy_mujoco.git
    ```
3. 安装必要环境（如果已安装 Isaac Lab 环境可跳过）：
   ```bash
    conda env create -f environment.yml
    conda activate g1_deploy
    ```
4. 在 Mujoco 模拟器中运行 Sim2Sim，默认策略路径为  `checkpoint/policy.pt`：
   ```bash
    python deploy_mujoco.py --policy YOUR_POLICY_PATH
    ```
5. （可选）如需将 JIT 格式策略转换为 ONNX 格式：
   ```bash
    python convert_jit_to_onnx.py --jit-path YOUR_POLICY_PATH --onnx-path OUTPUT_ONNX_PATH
    ```


## 🛠️ Steps (in English)

1. **Train a policy**  
   Train the **29-DoF Unitree G1** locomotion policy in [unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab) and export 
   `
   exported/policy.pt
   `

2. **Clone this repository**
   ```bash
   git clone https://github.com/RoboCubPilot/g1_deploy_mujoco.git
   ```

3. **Install environment** (skip if Isaac Lab is already installed)
   ```bash
   conda env create -f environment.yml
   conda activate g1_deploy
   ```

4. **Run deployment**  
   Launch Sim2Sim in Mujoco with the default policy path `checkpoint/policy.pt`:
   ```bash
   python deploy_mujoco.py --policy YOUR_POLICY_PATH
   ```

5. **(Optional) Convert JIT → ONNX**  
   ```bash
   python convert_jit_to_onnx.py --jit-path YOUR_POLICY_PATH --onnx-path OUTPUT_ONNX_PATH
   ```

---

## 🎉 Features

- 🏃‍♂️ Deploy RL policies to Mujoco in seconds  
- 🔄 JIT → ONNX conversion supported  
- 🔌 Seamless integration with Unitree RL Lab  

---