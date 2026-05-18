# HANDOFF

## 2026-05-15 → 17 · GapOnly + FlatWalkParkourCmd 系列实验

### 任务目标
让 Go2 用深度视觉学会跨 gap，最终验证视觉对 parkour 决策的作用。

### 关键发现/改进

#### 1. **`reward_lin_vel_xy_command_tracking` 是 base-frame**
- 公式：`exp(-4 × |cmd_vel - root_lin_vel_b|²)`
- 是机体坐标系下的速度跟踪，policy 朝任何方向跑都能拿满分
- 跟教师的 `tracking_goal_vel`（world frame，朝 goal 方向投影）**完全不同**
- 用它当主驱动会让 policy 朝歪了走也拿高分 — 跨过 gap 后无法纠正方向

#### 2. **删 + 调 reward 找平衡（v8→v9）**
- 删 `lin_vel_xy_command_tracking` → mean_reward 从 +34 跌到 0.6（信号过弱，policy 走太慢）
- 加回 weight=1.0 + 加 `ang_vel_yaw_command_tracking` weight=0.5 + tracking_yaw 0.5→0.8 → 平衡

#### 3. **PIEVelocity vs ParkourCommand**
- Walking baseline `model_10999.pt` 是 PIEVelocity 训的（cmd 直接给 yaw_rate=0）
- ParkourCommand 用 heading_target → 算 yaw_rate cmd（heading_control）
- Resume 切 Cmd 类型后，policy 看到的 cmd 输入分布完全变了 → 需要 1000-3000 轮适应
- 关键 reward：`ang_vel_yaw_command_tracking` 强迫 policy 跟 cmd_yaw_rate（不加的话 cmd 只是参考）

#### 4. **goal 设计陷阱**
- 之前 GapOnly 第一个 goal 在起点平台中间 → 跟 spawn 重合 → robot 出生就拿到 goal_idx=1（虚假完成）
- 修复：goal 只放 gap **之后**的平台中间
- num_goals 也要跟 num_gaps 对齐 — `num_goals > num_gaps` 时多余 goal 重复填最后一个 → 多个 goal 重叠

#### 5. **horizontal_scale 离散化**
- 默认 0.08m，5cm gap 离散化后变成 8cm（脚卡进去）
- 改 0.04m 后 5cm = 4cm gap，脚能跨过去
- 代价：height_field 像素数 ×4，训练略慢

#### 6. **Reward 形状（exp 函数让"慢"也有 reward）**
- `reward_lin_vel_xy_command_tracking = exp(-4 × |error|²)`：error=0.5 还能拿 0.37 的 reward
- policy 选择"慢慢蹭"也能拿正 reward → 不主动跑快
- 真正"达到才给"的形状：教师 `tracking_goal_vel` 用 `min(actual, cmd)`，慢就直接低分

#### 7. **新增 `reward_goal_reached`**
- 检测 `cur_goal_idx` 增加 → 一次性 +10 bonus
- 稀疏但强信号，比 dense reward 更能突破"刹车在 gap 边缘"局部最优
- 在 `parkour_isaaclab/envs/mdp/rewards.py` 末尾，class-based with state（保存 previous_goal_idx）

### 训练结果（v9 中段）

`logs/rsl_rl/unitree_go2_pie_parkour/2026-05-17_15-29-17/`
- iter 15178（4000 轮 resume from walking baseline）
- mean_reward 36, terrain_levels 5.97, goal_reached 0.0015
- 估计 11% episode 跨 gap，朝向偏 23°
- 速度跟踪 92%（达到 cmd），朝向跟踪 66%

### 当前 GapOnly 配置（v9）

**地形：**
- tile 8×4m, hs=0.04, vs=0.005, slope_threshold=0.3
- num_rows=10, num_cols=5, num_goals=1, difficulty_range=(0,0.15)
- start_platform=3m, plat_len=1.5m fixed, gap_width=5cm-15.5cm, gap_depth=2m
- num_gaps=1（1 个 gap → 1 个 goal）

**Command（ParkourCommand）：**
- lin_vel_x=(0.5, 1.0), heading=(0.0, 0.0)
- heading_control_stiffness=0.8

**Reward (14 项)：**
- 正向：tracking_goal_vel +1.5, tracking_yaw +0.8, lin_vel_xy_cmd +1.0, ang_vel_yaw_cmd +0.5, goal_reached +10
- 负向：lin_vel_z -0.5, ang_vel_xy -0.05, orientation -1.0, hip_default_pos -0.5, dof_acc -2.5e-7, action_rate -0.01, joint_power -2e-5, collision -10, feet_edge -1.0

**Termination（教师风格 `TerminationsCfg`）：**
- `total_terminates`：time_out / reach_all_goals / |roll|>1.5 / |pitch|>1.5 / root_z<-0.25

**Episode：** 15s（750 step）

**Runner：** 800 env, save_interval=500, clip_actions=1.2, init_noise_std=0.30

### 修改文件清单（本次）

- `parkour_isaaclab/envs/mdp/rewards.py` — 新增 `reward_goal_reached` class
- `parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrains_cfg.py` — `GapOnlyTerrainCfg` 调参
- `parkour_isaaclab/terrains/extreme_parkour/config/gap_only.py` — tile 12→8m, num_goals=1, hs=0.04
- `parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrians.py` — `gap_only_terrain` goal 放置
- `parkour_tasks/.../config/go2/parkour_mdp_cfg.py` — `GapOnlyRewardsCfg` 14 项, `FlatWalkParkourCmdRewardsCfg` 19 项
- `parkour_tasks/.../config/go2/parkour_pie_cfg.py` — `UnitreeGo2PIEGapOnlyEnvCfg` + `UnitreeGo2PIEFlatWalkParkourCmdEnvCfg`
- `parkour_tasks/.../config/go2/__init__.py` — 注册 `Isaac-PIE-FlatWalkParkourCmd-Unitree-Go2-v0`
- `parkour_tasks/.../config/go2/agents/rsl_pie_ppo_cfg.py` — save_interval 1000→500
- `scripts/rsl_rl/play.py` — 加 OpenCV 深度图可视化（带 try/except）

### 抽过的弯路

1. **`failure_terminal_penalty -250` 太重** — policy 怕掉坑直接不动，删
2. **`heading=(-1, 1)` + `lin_vel_x=(0.5, 1.5)`** — 高速 + 大转向太激进，policy 趴下加速
3. **`base_height_below_target` weight=-50** — 太重 policy 不敢动，调到 -10
4. **`gap_width_min=30cm`** — 起点太宽 policy 学不会，降到 5cm
5. **`gap_depth=0.3m`** — 浅坑 robot 卡里挣扎 collision 爆炸，回到 2m 让快速终止
6. **goal 跟 spawn 重合** — robot 出生就完成，没动力前进
7. **num_goals=2 + num_gaps=1** — 第二个 goal 重复填，跟第一个重叠

### 下一步

1. 让 v9 跑完 20000 轮（约 3-6 小时）
2. 等 `reward_goal_reached > 0.01`（≈ 100% 过 gap）
3. play 验证步态 + 朝向
4. 之后可以加 num_gaps=2-3 或 difficulty_range 上限提高

---

## 2026-05-14 · StairsOnly v2→v3: 碗形地形 + 随机阶高 + 视觉审计

### 改动概要

1. **StairsOnlyRewardsCfg 加 3 项 reward**（`parkour_tasks/.../parkour_mdp_cfg.py`）：
   - `reward_failure_terminal_penalty` weight=-250（只对 `illegal_body_contact`）
   - `reward_collision` weight=-10（base/hip/thigh/calf/head 非脚接触）
   - `reward_feet_air_time` weight=+1.0 threshold=0.2（Isaac Lab `feet_air_time`）
   - 原因：iter 11100 时 policy 卡死（terrain_levels=0, illegal=0.45），没有"摔=灾难"的梯度

2. **地形改为 IsaacLab 风格倒金字塔碗**（`extreme_parkour_terrians.py`）：
   - 从"沿 +x 一维 up→top→down"改为"四面嵌套矩形环向中心递降"
   - Robot spawn 在碗底（中心平台 2.5m 宽，保证 origin 落在最低点）
   - Tile 尺寸 12×12m，num_cols=5（减少总碗数）

3. **阶数增加到 15**（`extreme_parkour_terrains_cfg.py`）：
   - `num_steps_up=15, num_steps_down=15`
   - `step_depth=0.28m`

4. **竖直面修正**：`slope_threshold` 从 1.5 降到 0.3（在 hs=0.08/vs=0.005 下，≥2.5cm 的台阶面变成完全竖直）

5. **去掉 goal marker**：`self.parkours.base_parkour.debug_vis = False`（碗形无有意义 waypoint）

6. **v2 训练（固定阶高碗）**：
   - Run: `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-13_19-56-21/`
   - 配置：800 env / 15000 iter / 从 walking baseline model_10999 resume
   - 结果（iter 25998）：`mean_reward=88.6`, `terrain_levels=4.75`（difficulty≈0.53, 每阶 8.2cm）, `illegal=0.018`, `collision=-0.05`
   - 结论：policy 学会了稳定爬 8cm 均匀台阶，摔倒率从 45% 降到 1.8%

7. **视觉审计**（`scripts/audit_pie_estimator_pipeline.py` 对 model_25998）：
   - `depth_shuffle → z_m` RMS = **0.006**（几乎无影响）
   - `depth_shuffle → action`（间接）≈ **0.0004**
   - `proprio_shuffle → v_hat` = **0.132**（主要依赖 proprio）
   - `zero_v_hat` action RMS = **0.283**（actor 最大依赖）
   - `attn_entropy_norm=0.623`, `attn_max=0.26`（attention 变尖锐但对 action 贡献弱）
   - **结论：视觉未起作用**。碗形对称地形下 policy 学到了"盲爬"策略，靠 proprio 感知碰到台阶边缘后抬脚

8. **v3 改为随机阶高**（逼视觉依赖）：
   - `step_height_min=0.05m`（固定下限 5cm）
   - `step_height="0.05 + 0.10 * difficulty"`（上限 5-15cm）
   - 每环从 `[min, max]` 均匀随机采样，同一碗内每阶高度不同
   - 训练：800 env / 10000 iter / 从 model_25998 resume / save_interval=1000
   - Run: `logs/rsl_rl/unitree_go2_pie_parkour/pie_stairsonly_v3_randomh_resume_25998_800_10000.stdout.log`
   - PID: 395579，进行中（iter 31390/35998）
   - 中间指标：`mean_reward=91.1`, `terrain_levels=4.41`（difficulty≈0.49, 每阶 [5cm, 9.9cm]）, `illegal=0.005`

9. **play.py 加深度图可视化**：
   - 在 play 循环里用 OpenCV 实时显示 env 0 的深度图（灰度，4x 放大）
   - 窗口名 "Depth Camera (env 0)"

10. **save_interval 改为 1000**（`rsl_pie_ppo_cfg.py` 的 `UnitreeGo2PIEGentleLoadFixPPORunnerCfg`）

### 关键发现

- **碗形地形不适合训练视觉依赖**：四面对称 → 深度图无方向信息 → policy 用 proprio 盲爬就够
- **要逼出视觉**需要：不对称地形（直线走廊）、不可预测障碍（gap/随机缺失台阶）、或 proprio dropout
- **z_mu 仍然很弱**（mean_abs=0.006, kl_per_dim=0.00003），actor 主要靠 v_hat 和 z_m
- **走歪原因**：碗形四面等价 + 训练时 ang_vel_yaw=0 但无强 yaw 纠偏 reward
- **z=z_mu（actor 侧）不需要加噪声**：posterior_std≈1.0，sample 会产生 0.97 RMS 的灾难性扰动

### 下一步 TODO

1. 等 v3 训练跑完（ETA ~4.5h），再跑审计看 `depth_shuffle` 敏感性是否提升
2. 如果视觉仍无用 → 改地形为**直线走廊 + 随机阶高 + 偶尔 gap**
3. 考虑 proprio dropout 或去掉 z_m 强制 actor 用 z_mu
4. 长期：gap terrain 需要先学跳跃动作

### 文件改动清单

- `parkour_tasks/.../config/go2/parkour_mdp_cfg.py` — StairsOnlyRewardsCfg 加 3 项
- `parkour_tasks/.../config/go2/parkour_pie_cfg.py` — UnitreeGo2PIEStairsOnlyEnvCfg 去 goal marker
- `parkour_tasks/.../config/go2/agents/rsl_pie_ppo_cfg.py` — save_interval 500→1000
- `parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrians.py` — stairs_only_terrain 改为碗形+随机阶高
- `parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrains_cfg.py` — 15阶/slope_threshold/step_height_min
- `parkour_isaaclab/terrains/extreme_parkour/config/stairs_only.py` — tile 12×12m, num_cols=5, slope_threshold=0.3
- `scripts/rsl_rl/play.py` — 加 OpenCV 深度图可视化

---

## 2026-05-13 · StairsBeam waypoint aggressive spacing

- 修改 `parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrians.py` 的 `stairs_beam_terrain()` full-mode goal 生成：`num_goals=8` 时不再从 9 个 key waypoint 里 `np.linspace(...).astype(int)` 裁剪，改为显式 8 点。新分布只保留 1 个早期台阶点 + 1 个顶平台点，把 4 个 waypoint 分配到独木桥上，减少楼梯密集 waypoint 导致 `target_vel_dir` 频繁翻转，同时保留桥尾引导。
- difficulty=0 / full mode / 默认 `hs=0.08` 下，goal x 约为 `[1.52, 2.64, 3.92, 4.72, 5.52, 6.24, 7.04, 8.08]`；旧分布为 `[1.52, 2.64, 2.96, 3.28, 3.92, 4.88, 5.84, 8.08]`，旧版丢掉了 beam 后段点且楼梯点过密。
- 验证：`python3 -m py_compile parkour_isaaclab/terrains/extreme_parkour/extreme_parkour_terrians.py` 通过。下一步训练/播放 StairsBeam 时重点观察 `current_goal_idx` 是否更平滑、桥尾是否少出现目标跳到 landing 后方向突变。

## 2026-05-12 (part 2 · stable walking baseline)

**最终 baseline**: `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-12_20-51-33/model_10999.pt`（FlatWalk 平地任务 + FlatWalk 专用 reward 组，512 env / seed 1，iter 0→1000 从 0 训、1000→10999 resume 加 stand_still 续训）。play 观察四腿正常 trot、不弹跳不三脚、静止前倾消失，`mean_reward=84`、`error_vel_xy=0.13`、`illegal_body_contact=0.005`、`time_out=0.99`。**请勿再修改该 checkpoint 对应的 cfg**；后续所有实验从它出发或基于它的 cfg 分叉。

走到这个 baseline 经过的 ablation（按时间顺序）：

1. **zfix gait 副作用**：zfix（decoder 断 z_m shortcut + 关闭训练期采样）让 `z_mu` 活过来但 policy 学成三脚走（RR contact=0.027）。gaitfix 第一版（`reward_feet_min_force_share=-5.0` + `reward_feet_air_time=+1.0 threshold=0.5` + push + 双向 command）让 iter 1500 时 RR force_share 恢复到 0.229；但 iter 2000+ 课程推进后 RR 再退化，play 是 **pronk（四腿同步弹跳）**。`logs/.../2026-05-12_14-53-57/model_2999.pt`（作 pronk 对比用）。

2. **foot_mirror_diagonal ablation**：新增 `reward_foot_mirror_diagonal`（对角对 thigh+calf offset 平方差，hip 不约束）在 `parkour_isaaclab/envs/mdp/rewards.py`。weight=-0.5 压住弹跳但 mean_reward 掉 1/3；weight=-0.1 约束不住。**已注释掉**，保留代码备用。

3. **action_scale=0.25 ablation**：`PIEActionsCfg.scale 1.0→0.25`。500 iter 后四腿 contact 0.96-0.99 均衡但 policy 几乎不动（pwr 0.2-0.7 vs 之前 6-15）。DreamWaQ 用 scale=0.25 时配 init_noise_std=1.0 + clip=100，有效 joint 偏移 ±0.75 rad；单改 scale 不等价于 DreamWaQ。**已回滚到 scale=1.0**。

4. **FlatWalk + no_fly + air_time threshold=0.2**：切到 `Isaac-PIE-FlatWalk-Unitree-Go2-GentleLoadFix-v0`（纯 `parkour_flat apply_flat=True`、`difficulty=0`、curriculum 关）。Reward 改动：
   - 新增 `reward_no_fly(sensor_cfg, contact_threshold=1.0)` in `parkour_isaaclab/envs/mdp/rewards.py`（四脚同时 vertical force < 1N → 1.0 否则 0.0）
   - `PIEGentleLoadFixRewardsCfg` 加 `reward_no_fly weight=-1.0`
   - `reward_feet_air_time` threshold 从 0.5 降到 0.2（Isaac Lab 的实现不 clip，0.5 threshold 会奖励长腾空即 pronk；降到 0.2 让 trot 短步也拿得到 reward）
   - `UnitreeGo2PIEFlatWalkEnvCfg.__post_init__`: `lin_vel_x=(0, 1.0) lin_vel_y=(0,0) ang_vel_yaw=(-0.3, 0.3)`；`events.push_by_setting_velocity=None`；`commands.base_velocity.debug_vis=True`、`parkours.base_parkour.debug_vis=False`（FlatWalk 命令不跟 parkour goal 耦合，开 goal marker 在 play 会误导）
   
   512 env / 3000 iter / seed 1 run `logs/.../2026-05-12_19-58-48`：iter 1000 `mean_reward=81`、`illegal=0.018`、四腿 contact 0.69-0.81 正常 trot，**play 观察为正常走路**，但静止时前倾。

5. **stand_still resume**：从 `model_1000.pt` resume 加 `reward_stand_still`（legged_gym 标准实现）训到 iter 10000。
   - 新增 `reward_stand_still(asset_cfg, command_name, command_threshold=0.1)` in `parkour_isaaclab/envs/mdp/rewards.py`（仅当 `|v_cmd| < threshold` 时惩罚 `sum(|dof_pos - default|)`）
   - `PIEGentleLoadFixRewardsCfg` 加 `reward_stand_still weight=-0.5 command_threshold=0.1`
   - 训练命令：`--resume --load_run 2026-05-12_19-58-48 --checkpoint model_1000.pt --reset_optimizer_on_resume --num_envs 512 --max_iterations 10000 --seed 1`
   - run 目录 `logs/.../2026-05-12_20-51-33`，跑到 `model_10999.pt` 自然结束
   - iter 10999：`mean_reward=84`、`error_vel_xy=0.13`、`illegal=0.005`、`bad_base_orientation=0`、`low_base_height=0.008`、`reward_no_fly≈-0.004`（不弹跳）、`reward_stand_still` 偶有 -0.011（生效但低频）
   - **play 观察**：四腿正常 trot、不弹跳、不三脚、静止不前倾，走路质量明显高于之前任何 run

**当前 `PIEGentleLoadFixRewardsCfg` 活跃 reward 清单**（17 项，`foot_mirror_diagonal` 注释）：
- Tracking (+): `lin_vel_xy +4.0`、`ang_vel_yaw +0.5`
- Posture (-): `ang_vel_xy -0.1`、`lin_vel_z -2.0`、`orientation -2.0`、`hip_default_pos -1.0`、`joint_default_pos -0.03`
- Smoothness (-): `dof_acc -5e-7`、`joint_power -2e-5`、`action_rate -0.05`、`action_smoothness -0.01`、`delta_torques -1e-7`
- Safety (-): `collision -10.0`、`failure_terminal_penalty -250`、`bad_orientation_terminal_penalty -250`、`low_base_terminal_penalty -250`
- Gait (-): `feet_stumble -1.0`、`feet_slip -0.5`、`feet_min_force_share -5.0`、`no_fly -1.0`、`stand_still -0.5`
- Gait (+): `feet_air_time +1.0 threshold=0.2`

**下一步 TODO**（按优先级）：
1. 用 `scripts/audit_pie_estimator_pipeline.py` 审计 `model_10999.pt`，看 walking baseline 下 `z_t`/`z_mu` 是否真的被 actor 用（对比 2026-05-11 zfix long run 的审计数字）
2. 从 `model_10999.pt` resume 切回 parkour 任务（`Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0`）+ 恢复 push + 双向 command，看能否保留 trot 同时学过障
3. 如果 2 下 RR 再偷腿，考虑 `reward_foot_mirror_diagonal` weight=-0.2 做 warmup
4. sim-to-sim / sim-to-real（用 `mujoco_isaaclab_deploy_v2` 或 `DreamWaq_train_go2/deploy/`）

---

## 2026-05-12

- 对上一次（2026-05-11）PIE zfix 修复做 512 env / 3000 iter 长训并做逐 checkpoint 审计，任务 `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0`。
  - 改动：`UnitreeGo2PIEGentleLoadFixPPORunnerCfg.save_interval` 从 `4000` 临时改为 `500`，方便中途审计多个 checkpoint。
  - run 目录：`logs/rsl_rl/unitree_go2_pie_parkour/2026-05-11_21-22-31`。stdout：`logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_zfix_from0_512_3000.stdout.log`。命令：`--num_envs 512 --max_iterations 3000 --seed 1`，不 resume。
  - 对 `model_500/1000/1500/2000/2500/2999.pt` 都用训练时语义 `--policy_action_limit 1.2 --clip_actions_override 1.2 --num_envs 128 --steps 256 --warmup_steps 80` 跑 `scripts/audit_pie_estimator_pipeline.py`，summary 存 `/tmp/audit_zfix_long_<iter>_summary.txt`。pipeline 自检（builtin vs manual actor obs / policy prefix / hidden reset / finite failures）所有 checkpoint 均为 0。
  - 训练指标时序（iter 1065 / 1535 / 2076 / 2558 / 2999）：`terrain_levels` `2.39 / 2.92 / 3.37 / 3.33 / 3.69`、`current_goal_idx` `0.72 / 0.79 / 0.87 / 0.94 / 0.86`、`mean_reward` `29 / 41 / 43 / 52 / 44`、`episode_length` `603 / 726 / 737 / 820 / 711`、`illegal_body_contact` `0.30 / 0.19 / 0.23 / 0.22 / 0.23`、`low_base_height` `0.15 / 0.09 / 0.14 / 0.10 / 0.12`、`bad_base_orientation` ≤ 0.02、`loss_kl` `0.0012 / 0.0007 / 0.0005 / 0.0003 / 0.0003`、`loss_next_proprio` `0.0036 / 0.0028 / 0.0023 / 0.0017 / 0.0018`。
  - 审计时序（iter 500 / 1000 / 1500 / 2000 / 2500 / 2999）：
    - `z_mu_mean_abs`：`0.0708 / 0.0336 / 0.0236 / 0.0202 / 0.0151 / 0.0146`（iter 500 峰值后缓降，但仍比 5-08 的 KL-mean baseline 同期高数倍）。
    - `z_mu_batch_std_mean`：`0.030 / 0.043 / 0.032 / 0.026 / 0.019 / 0.019`。
    - `kl_per_dim_mean`：`0.00397 / 0.00102 / 0.00055 / 0.00038 / 0.00021 / 0.00021`（vs 5-06 mask_prev_action 末尾 `0.00014`、5-06 KL-mean `0.00003`；posterior 未塌回先验）。
    - `zero_z_mu`（actor action RMS）：`0.097 / 0.048 / 0.044 / 0.044 / 0.035 / 0.039`（vs 5-06 baseline `0.003-0.008`；5-13×）。
    - `zero_z_m`：`0.237 / 0.298 / 0.288 / 0.257 / 0.248 / 0.221`。
    - `zero_v_hat`：`0.111 / 0.118 / 0.121 / 0.168 / 0.196 / 0.169`。
    - `zero_all_estimator`：`0.239 / 0.336 / 0.337 / 0.298 / 0.309 / 0.285`。
    - `v_hat_rmse`：iter 1000 以后稳定在 `[0.03, 0.03, 0.03]` 附近；`h_f_hat_rmse` RR 有抖动（2999 时 RL `0.053`、RR `0.019`）；`height_hat_rmse` 稳在 `0.08-0.12`。
    - `attn_max` `0.040 → 0.103`（均匀值 0.0185 的 5.6×）、`attn_entropy_norm` `0.944 → 0.873`、`highway_beta_mean` `0.528 → 0.653`、`grf_gate_mean` 稳在 `0.52-0.57`；cross-attention / GRU / highway 都持续在工作且越训越尖锐/偏向 GRU 路径。
    - actor 第一层 mean_col_l2 同步上升：`z_m 0.890→0.996`、`z_mu 0.881→1.063`、`v_hat 1.075→1.314`、`h_f_hat 0.900→0.972`；`z_mu` 的 actor 权重 norm 在 iter 2000 开始超过 `z_m`，说明 actor 对 z_mu 每一维敏感度提高，但 z_mu 的数值本身因 KL 被继续压低，两者抵消后 `zero_z_mu` 没继续涨。
    - 输入敏感性：`depth_shuffle→z_mu` 稳在 `0.004-0.008`、`proprio_shuffle→z_mu` 稳在 `0.024-0.049`；`depth_shuffle→z_m` 升到 `0.04-0.05`，地形信息主要仍进入 `z_m`，`z_mu` 承担小幅 residual。
  - 结论：zfix（断 z_m shortcut + 关闭训练期采样）彻底有效；训练曲线良性，`terrain_levels` 从 0 爬到 3.69、`current_goal_idx` 最高 0.95；z_mu 存活但在 `detach_pie_actor_features=True` 下承担的是 residual，monitor 到缓降但未塌回先验。下一步 play 验证步态（重点看 RR）再决定是否做 `pie_joint_actor_estimator=True` ablation 或切换到 paper-aligned task（`Isaac-PIE-Parkour-Unitree-Go2-v0` / `-FullFix-v0`）做二阶段课程。

## 2026-05-11

- 诊断 PIE `z_t` 几乎不起作用的根因并做最小结构修复，改动文件：
  - `scripts/rsl_rl/modules/feature_extractors/pie_estimator.py`
  - `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/agents/rsl_pie_ppo_cfg.py`
  - `scripts/audit_pie_estimator_pipeline.py`、`parkour_test/test_pie_estimator.py`
  - 根因定位：`next_proprio_decoder(cat(z, z_m, v_hat, h_f_hat))` 让 z_m 成为 decoder shortcut，且 z_m 没有 KL 约束，decoder 会完全靠 z_m 重构 next_proprio，梯度不回到 z_mu_head；训练期 `sample_latent_in_training=True` 让 decoder 一开始就把 z 当成噪声忽略；两者叠加导致 VAE posterior 塌到先验，推理期 `z_mu≈0`，actor 第一层 z_mu 段权重非零但输入数值接近 0，所以 z 对 action 无影响。
  - 修复：`next_proprio_decoder` 输入从 `cat(z, z_m, v_hat, h_f_hat)`（84 维）改为 `cat(z, v_hat, h_f_hat)`（39 维），断掉 z_m shortcut；`ParkourRslRlPIEEstimatorCfg.sample_latent_in_training=True → False`，训练/推理都用确定性 `z=z_mu`。actor 仍使用 `("z_m","z_mu","v_hat","h_f_hat")`，actor_obs 维度 116 不变；`z_m_dim`、`latent_dim` 不动；KL per-dim mean、`detach_pie_actor_features=True`、`pie_joint_actor_estimator=False` 均不动。同步更新 audit 脚本里的手算 decoder forward 和单测 `in_features` 断言。旧 checkpoint 的 `next_proprio_decoder` 权重 shape 已变，不能 resume，只能从 0 重训。
  - `scripts/audit_pie_estimator_pipeline.py` 新增 `--summary_out <path>` 和 `emit()`，将 summary 同时镜像到文件，避免 `env.close()` 前 Isaac 接管 stdout 导致看不到审计摘要。
  - 验证：`py_compile` 通过；`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_rollout_storage.py parkour_test/test_pie_estimator_ppo_update.py -q` 通过（17 passed）。
  - 短训验证 run：`logs/rsl_rl/unitree_go2_pie_parkour/2026-05-11_20-41-11`（task `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0`，512 env / 500 iter / seed 1 / 不 resume），stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_zfix_from0_512_500.stdout.log`。末轮约 `mean_reward=53.18`、`episode_length=839.87`、`loss=0.0106`、`loss_v=0.0019`、`loss_hf=0.0001`、`loss_height=0.0008`、`loss_next_proprio=0.0062`、`loss_kl=0.0016`、`illegal_body_contact=0.234`、`bad_base_orientation=0.002`、`low_base_height=0.057`、`time_out=0.710`、`terrain_levels=0.0059`、`current_goal_idx=0`、`goal_reached=0`。`loss_kl` 从旧 mask-prev-action 末期 `0.0001` 升到 `0.0016`，不再归零。
  - 对 `model_499.pt` 用训练时语义 `--num_envs 128 --steps 256 --warmup_steps 80 --policy_action_limit 1.2 --clip_actions_override 1.2` 审计（summary 见 `/tmp/audit_zfix_499_summary.txt`）：`actor_obs_builtin_vs_manual_rms=0`、`actor_obs_policy_prefix_rms=0`、`z_vs_z_mu_rms=0`、`hidden_reset_max_abs=0`、`finite_prediction_failures=0`，pipeline 仍自洽。latent posterior：`z_mu_mean_abs=0.0531`（旧 mask-prev-action 500 `0.00968`，×5.5）、`z_mu_max_abs=0.3331`（旧 `0.0850`）、`z_mu_batch_std_mean=0.0337`（旧 `0.01031`）、`posterior_std_mean=0.9996`、`kl_sum_mean=0.0726`（旧 `0.00449`）、`kl_per_dim_mean=0.00227`（旧 `0.000140`）。actor ablation：`zero_z_mu=0.0389`（旧 `0.00788`，×5）、`zero_z_m=0.0420`（旧 `0.179`，塌降）、`zero_v_hat=0.175`、`zero_h_f_hat=0.00486`、`zero_all_estimator=0.132`，`sample_actor_latent_from_posterior=0.389`；actor 第一层权重 norm `z_m=0.890`、`z_mu=0.881`、`v_hat=1.075`、`h_f_hat=0.900`，和 z 段输入数值一起起来。输入敏感性：`proprio_shuffle→z_mu=0.0470`、`depth_shuffle→z_mu=0.0032`，z_mu 开始承担 proprio 相关 residual；`proprio_shuffle→v_hat=0.0691`、`depth_shuffle→z_m=0.0098`。内部通路：`attn_entropy_norm=0.944`、`attn_max=0.038`（54 tokens 均匀值 0.0185，仍偏均匀）、`grf_gate_mean=0.567`、`highway_beta_mean=0.528`、`highway_beta_low/high_frac≈0`，cross-attention/GRU/highway 均在起作用。target RMSE：`v_hat=[0.063,0.031,0.036]`、`h_f_hat FL/FR/RL/RR=0.006/0.015/0.007/0.009`、`height=0.0328`、`next_proprio=0.1343`（比之前带 z_m shortcut 时高，属 shortcut 断开后 decoder 的预期代价；未伤 episode_length/reward）。
  - 结论：之前 `z_t` 在 actor 不起作用不是 actor 侧 shortcut，而是 VAE posterior collapse。修复后 z_mu 起来且 actor 开始用它，但 500 iter 尚未收敛，需要长训确认 parkour 指标（`terrain_levels`、`current_goal_idx`、`how_far`）随 z_mu 使用率一起提高。下一步做 512 env / 3000-5000 iter 长训并在 250/500/1000/2000/... checkpoint 复跑审计。

- 参考本地 DreamWaQ / legged_gym 实现，将 PIE proprioception 里的 command 也改为固定尺度缩放，改动文件：
  - `parkour_isaaclab/envs/mdp/observations.py`
  - 新增 `PIE_COMMAND_SCALE = (2.0, 2.0, 0.25)`；`pie_proprioception()` 现在拼接 `commands * [2.0, 2.0, 0.25]`，对齐 DreamWaQ 的 `cmd_scale=[2, 2, 0.25]` 风格。
  - 保持 `root_ang_vel * 0.25`、`joint_vel * 0.05`、`joint_pos-default`、`previous_action` 不变；仍不启用 RSL-RL running mean/std（`empirical_normalization=False`）。
  - 影响范围：policy proprio、proprioception_history、critic 前 45 维、`next_proprioception` auxiliary target 都会使用同一 command 固定尺度；reward command tracking 和 base velocity target 仍使用原始物理量。
  - 验证：`py_compile parkour_isaaclab/envs/mdp/observations.py` 通过；`parkour_test/pie_estimator_env_smoke.py --task Isaac-PIE-FlatWalk-Unitree-Go2-GentleLoadFix-v0 --headless --num_envs 2` 通过。

- 按“只训基础走路先切平面”的方向新增独立 FlatWalk 任务，不覆盖现有 `StableEasyHeight-GentleLoadFix` parkour/easy terrain baseline。改动文件：
  - `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_pie_cfg.py`
  - `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/__init__.py`
  - 新增 `UnitreeGo2PIEFlatWalkEnvCfg`，继承 `StableEasyHeightGentleLoadFix` 的 command/reward/termination/action 设置，但将 terrain generator 改为 `curriculum=False`、`random_difficulty=False`、`difficulty_range=(0.0, 0.0)`，并设置仅 `parkour_flat.proportion=1.0`、其他子地形 `proportion=0.0`；同时关闭 roughness/noise，`max_init_terrain_level=0`。
  - 注册新任务 `Isaac-PIE-FlatWalk-Unitree-Go2-GentleLoadFix-v0`，runner 沿用 `UnitreeGo2PIEGentleLoadFixPPORunnerCfg`，方便直接从 0 训练平面走路并和原 `GentleLoadFix` 对比。
  - 验证：`py_compile parkour_pie_cfg.py __init__.py` 通过；`parkour_test/pie_estimator_env_smoke.py --task Isaac-PIE-FlatWalk-Unitree-Go2-GentleLoadFix-v0 --headless --num_envs 2` 通过，RewardManager 仍为当前 GentleLoadFix 16 个 reward term，Observation/Estimator shape 正常。

- 按用户决定修正 PIE estimator 的 `next_proprio` 重建目标，改动文件：
  - `scripts/rsl_rl/modules/feature_extractors/pie_estimator_loss.py`
  - 新增 `NEXT_PROPRIO_STATE_DIM = 33`，对应 PIE proprio 前 33 维 `[base_ang_vel, projected_gravity, command, joint_pos_offset, joint_vel]`。
  - `loss_next_proprio` 现在只对 `predictions["next_proprio_hat"][..., :33]` 和 target 前 33 维计算 MSE；不再监督最后 12 维 `previous_action`，因为 `obs_{t+1}.previous_action` 实际是当前 policy sampled action `a_t`，不是干净的环境状态预测目标。
  - 网络输出维度、observation 结构、actor 输入均未改变；`next_proprio_hat` 仍输出 45 维，只是最后 12 维不参与该 auxiliary loss。
  - 验证：`/home/zzysh/miniconda3/envs/env_isaaclab/bin/python -m py_compile scripts/rsl_rl/modules/feature_extractors/pie_estimator_loss.py` 通过；`/home/zzysh/miniconda3/envs/env_isaaclab/bin/python -m pytest parkour_test/test_pie_estimator_loss.py -q` 通过（5 passed）。

- 为验证上述改动是否改善 `z_mu`，从 0 跑了 512 env / 500 iter 短训。最初带 `--enable_cameras` 的两次启动均在 Isaac/RTX 初始化阶段崩溃（`librtx.scenedb.plugin.so` / `libomni.hydra.rtx.plugin.so`），未进入训练；去掉 `--enable_cameras` 后可正常训练，因为 PIE depth 使用 `RayCasterCameraCfg`，非渲染 headless kit 仍提供 `depth_camera=(2,58,87)`。
  - 成功 run：`logs/rsl_rl/unitree_go2_pie_parkour/2026-05-11_18-03-25_pie_gentle_loadfix_maskprevaction_zmu_from0_512_500_nocam`，checkpoint：`model_499.pt`。
  - 最终第 499 轮约 `mean_reward=23.45`、`mean_episode_length=476.97`、`pie_estimator/loss=0.0150`、`loss_v=0.0029`、`loss_next_proprio=0.0076`、`loss_kl=0.0001`、`terrain_levels=0.3276`、`current_goal_idx=0.0632`、`goal_reached=0`。短训能形成低难度走路，但不能说明 parkour 能力。
  - 同时修改 `scripts/audit_pie_estimator_pipeline.py`，移除硬编码 `args_cli.enable_cameras = True`，避免审计脚本强制加载 RTX rendering kit；需要渲染时仍可显式传 `--enable_cameras`。验证：`py_compile scripts/audit_pie_estimator_pipeline.py` 通过。
  - 对新 `model_499.pt` 用 `--headless --num_envs 128 --steps 256 --warmup_steps 80 --policy_action_limit 1.2 --clip_actions_override 1.2` 审计：`z_mu_mean_abs=0.009684`、`z_mu_max_abs=0.08498`、`z_mu_batch_std_mean=0.01031`、`kl_sum_mean=0.00449`、`kl_per_dim_mean=0.000140`、`zero_z_mu` action RMS `0.00788`；但 `zero_z_m=0.17855`、`zero_v_hat=0.05967`、`zero_all_estimator=0.17504`，actor 仍主要依赖 `z_m/v_hat`，`z_mu` 影响较小。
  - 用同一审计条件对旧 `2026-05-08_15-17-42_pie_gentle_loadfix_klmean_zmuactor_from0_512_1000_save250/model_500.pt` 做公平对比：旧 `z_mu_mean_abs=0.01433`、`z_mu_batch_std_mean=0.01747`、`zero_z_mu=0.01491`，均高于新 500 轮；旧 `model_999.pt` 也为 `z_mu_mean_abs=0.01268`、`zero_z_mu=0.00874`。结论：mask 掉 `next_proprio.previous_action` 让监督目标更干净，且新模型的 `v_hat/next_proprio/height` RMSE 有改善趋势，但没有让 `z_mu` 变强或让 actor 更依赖 `z_mu`。下一步若目标是提高 `z_mu` 使用率，应考虑结构性限制 actor 侧捷径（例如先做 `actor_feature_keys=("z_mu",)` 或至少去掉/降权 `z_m`），而不是继续只靠这个 loss mask 长训。

## 2026-05-08

- 按要求准备长训配置：`StableEasyHeight-GentleLoadFix` 专用 runner 改为 `num_envs=512` 训练时使用，`max_iterations` 由启动命令传入 `20000`；`UnitreeGo2PIEGentleLoadFixPPORunnerCfg` 覆盖为 `save_interval=4000`、`clip_actions=1.2`，并新增 `ParkourRslRlPIEGentleLoadFixActorCriticCfg` 将该任务 `policy.action_limit=1.2`，保留 Gentle 的 `init_noise_std=0.30`。
  - 已停止旧 16 env play，并启动长训：PID `204437`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-08_21-38-10_pie_gentle_loadfix_hipslip_action1p2_from0_512_20000_save4000`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_hipslip_action1p2_from0_512_20000_save4000.stdout.log`。
  - 参数快照确认：`max_iterations=20000`、`clip_actions=1.2`、`save_interval=4000`、`policy.action_limit=1.2`、`init_noise_std=0.3`。

- 按要求重做 `StableEasyHeight-GentleLoadFix` 的 walking reward 正则项，改动文件：
  - `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py`
  - `parkour_isaaclab/envs/mdp/rewards.py`
  - 从 `PIEGentleLoadFixRewardsCfg` 移除四脚受力/接触平衡思路的 `reward_feet_min_force_share` 与 `reward_feet_vertical_force_balance`。
  - 新增/覆盖 reward：`reward_hip_default_pos=-1.0`（只压 `.*_hip_joint` 偏离默认姿态）、`reward_joint_default_pos=-0.03`（轻压全关节默认姿态偏离）、`reward_lin_vel_z=-2.0`、`reward_action_rate=-0.05`（当前 raw action 与上一 raw action 的 norm 差）、`reward_dof_acc=-5e-7`（关节速度差除以 dt）、`reward_delta_torques=-1e-7`、`reward_feet_stumble=-1.0`、`reward_feet_slip=-0.5`。
  - 新增 `reward_feet_slip()`，按脚部水平接触力相对竖直力超过阈值的平方惩罚；`reward_feet_stumble()` 增加可配置的水平/竖直力比例阈值。
  - `reward_action_rate`、`reward_dof_acc`、`reward_delta_torques` 的 reset 逻辑补齐 `env_ids=None` 情况，避免全量 reset 时索引异常。
  - 验证：`py_compile` 通过；2 env headless smoke 通过，RewardManager 显示 16 个 active terms，并确认 `reward_feet_min_force_share`/`reward_feet_vertical_force_balance` 已不在 GentleLoadFix reward 列表中，新项 `reward_hip_default_pos`、`reward_joint_default_pos`、`reward_delta_torques`、`reward_feet_stumble`、`reward_feet_slip` 已生效。

- 按要求调整 `StableEasyHeight-GentleLoadFix` 的 reward 配置，改动文件：
  - `parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py`
  - 在 `PIEGentleLoadFixRewardsCfg` 内新增/覆盖：
    - `reward_feet_vertical_force_balance`，`weight=-0.4`，参数沿用 `contact_forces` 的 `.*_foot`、`ema_alpha=0.02`、`min_total_force=20.0`
    - `reward_orientation` 从默认 `-1.0` 覆盖为 `-2.0`
    - `reward_ang_vel_xy` 从默认 `-0.05` 覆盖为 `-0.1`
- 验证：
  - `/home/zzysh/miniconda3/envs/env_isaaclab/bin/python -m py_compile parkour_tasks/parkour_tasks/extreme_parkour_task/config/go2/parkour_mdp_cfg.py` 通过。
  - 2 env headless smoke 通过：`parkour_test/pie_estimator_env_smoke.py --task Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0 --headless --num_envs 2`。
  - smoke 中 RewardManager 显示 13 个 active terms，并确认：
    - `reward_ang_vel_xy=-0.1`
    - `reward_orientation=-2.0`
    - `reward_feet_vertical_force_balance=-0.4`

## 2026-05-06

- 继续按复现审计检查 estimator 链路：新增 `scripts/audit_pie_estimator_pipeline.py`，加载 checkpoint 后逐步对比 `build_pie_actor_observations()` 与手工 `policy + estimator(z_m,z,v_hat,h_f_hat)` 拼接、检查 policy prefix、latent eval 确定性、GRU hidden reset、feature clamp 和监督 target RMSE。验证：`py_compile` 通过；对 `action0p6/model_3499.pt` 使用训练时语义 `--policy_action_limit 0.6 --clip_actions_override 0.6` 审计，`actor_obs_builtin_vs_manual_rms=0`、`actor_obs_policy_prefix_rms=0`、`z_vs_z_mu_rms=0`、`finite_prediction_failures=0`、`hidden_reset_max_abs=0`，说明 inference actor 输入链路没有发现拼接错位或 hidden reset 错误。监督头误差为 `v_hat_rmse=[0.036,0.021,0.025]`、`h_f_hat_rmse=FL0.0036/FR0.0040/RL0.0034/RR0.0091`、`height_hat_rmse=0.0375`、`next_proprio_hat_rmse=0.0349`；actor feature 无 clip 饱和，`z_m max=1.37`、`z max=0.014`、`v_hat max=0.797`、`h_f_hat max=0.185`。结论：当前 checkpoint 的 estimator 推理链路整体自洽，右后腿/站姿问题暂不优先归因于 estimator 输出错位。
- 扩展 `scripts/audit_pie_estimator_pipeline.py`，新增 latent posterior 统计和在线输入敏感性检查，用于解释 `z` 为什么很小：统计 `z_mu`、`z_logvar`、posterior std、KL、训练期采样期望幅度，并分别 shuffle `depth_camera`/`proprioception_history` 后观察 `z_mu/z_m/v_hat/h_f_hat` 变化。验证：`py_compile` 通过；对 `action0p6/model_3499.pt` 复跑，结果 `z_mu_mean_abs=0.00136`、`z_mu_max_abs=0.0061`、`z_logvar_mean≈0.00012`、`posterior_std_mean≈1.00006`、`kl_sum_mean≈0.000074`、`expected_sampled_z_abs_mean≈0.798`。结论：posterior 基本贴到标准正态先验，eval/play 中 `z=z_mu≈0`，但训练期若 estimator 处于 train mode 且 `sample_latent_in_training=True`，actor 看到的 sampled `z` 理论均值幅度约 0.8，存在 train/eval latent 分布不一致和 `z` 被 actor 忽略的风险。敏感性结果显示 shuffle depth/proprio 对 `z_mu` 影响很小（约 `0.0008/0.0012` RMS），但对 `z_m` 有明显影响（约 `0.028/0.103` RMS），说明当前隐式地形信息主要进了 `z_m`，不是进了 VAE latent `z`。
- 再扩展 `scripts/audit_pie_estimator_pipeline.py`，加入 actor feature ablation 和 actor 第一层输入权重 norm：在同一观测下分别置零 `z/z_m/v_hat/h_f_hat/all estimator features`，以及用 posterior sampled `z` 替代 eval `z_mu`，统计 action RMS 变化。验证：`py_compile` 通过；对 `action0p6/model_3499.pt` 复跑，`zero_z` action RMS 仅 `0.00015`，几乎无影响；`zero_z_m` 为 `0.139`，`zero_v_hat` 为 `0.021`，`zero_h_f_hat` 为 `0.0028`，`zero_all_estimator` 为 `0.129`，`sample_z_from_posterior` 为 `0.0765`。actor 第一层权重 norm 虽然 `z` 段不为零（约 `0.77`），但由于 eval `z` 数值接近 0，实际行为影响几乎为 0；结论是当前 actor 主要使用 `z_m`，几乎不使用 VAE latent `z`，且训练期 sampled `z` 与 eval `z_mu≈0` 的分布差异会明显扰动 action。
- 修复 VAE latent `z` 训练尺度问题：`_gaussian_kl()` 从“对 32 维 latent 求和再 batch mean”改为“所有 latent 维度和 batch 统一 mean”，等价于参考 DreamWaQ/HIM 类实现把 KL 除以 latent 维度数，避免 reconstruction loss 按维度 mean 而 KL 按维度 sum 造成 KL 相对过强；同时将 `ParkourRslRlPIEEstimatorCfg.sample_latent_in_training` 改为 `False`，使训练和 play 都使用确定性 `z=z_mu`，消除训练期 sampled `z` 与推理期 `z_mu≈0` 的分布不一致。新增单测 `test_gaussian_kl_averages_latent_dimensions()`，用 `mu=1, logvar=0` 的 32 维样本断言 KL 为 `0.5` 而不是 `16.0`。验证：`py_compile` 通过；`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_ppo_update.py -q` 通过（14 passed）。下一次从 0 训练时需重点看 `loss_kl` 不应再 250 轮内快速归零、`z_mu_mean_abs` 是否高于此前约 `0.0013`，以及 `zero_z` ablation 对 action 的影响是否明显增大。
- 随后按 VAE 语义修正上一条的采样策略：恢复 `ParkourRslRlPIEEstimatorCfg.sample_latent_in_training=True`，使 estimator reconstruction/next-proprio decoder 训练仍使用 reparameterization sampled `z`；但 actor feature 改为 `("z_m", "z_mu", "v_hat", "h_f_hat")`，并同步 `PPOWithExtractor` 默认 feature key，使 actor 训练和 play 都消费确定性的 posterior mean `z_mu`，不再直接吃 sampled `z`。更新 `scripts/audit_pie_estimator_pipeline.py` 支持 `z_mu` 作为 actor latent feature、`zero_z_mu` 和 posterior sample ablation；更新 PPO 相关单测里的 actor feature key。验证：`py_compile` 通过；`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_ppo_update.py -q` 通过（14 passed）。当前最终方案是：KL 按 latent 维度 mean，VAE decoder 路径继续采样，actor 路径使用 `z_mu`。
- 已按 KL-mean + actor 使用 `z_mu` 的最终方案启动从 0 短训 ablation：PID `156434`，task `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-08_15-17-42_pie_gentle_loadfix_klmean_zmuactor_from0_512_1000_save250`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_klmean_zmuactor_from0_512_1000_save250.stdout.log`。命令为 512 env、1000 iter、seed 1、不 resume；启动时 `setsid` 脱离会话运行。日志确认已进入训练循环，约 iteration 6-15 时 `pie_estimator/loss_kl` 为 `0.0029 -> 0.0021`（注意该 KL 已从 sum 改为 per-dim mean，不能和旧 run 的 KL 数值直接比较）。后续应在 `model_250.pt`、`model_500.pt` 跑 `audit_pie_estimator_pipeline.py`，重点看 `z_mu_mean_abs`、`z_mu_batch_std_mean`、`kl_per_dim_mean`、`zero_z_mu` action RMS 和 `sample_actor_latent_from_posterior` action RMS。
- KL-mean + actor `z_mu` 短训已完成到 `model_999.pt`；末尾日志约 `mean_reward=63.37`、`mean_episode_length=911.75`、`pie_estimator/loss=0.0116`、`loss_next_proprio=0.0087`、`loss_height=0.0011`、`episode_time_out=0.710`、`bad_base_orientation=0.0005`、`low_base_height=0.064`、`illegal_body_contact=0.227`，但 parkour 推进弱（`current_goal_idx≈0.023`、`how_far≈1.87`）。对 `model_999.pt` 跑 `audit_pie_estimator_pipeline.py --policy_action_limit 0.8 --clip_actions_override 0.8`：pipeline 拼接和 hidden reset 仍为 0 误差；`z_mu_mean_abs=0.00414`、`z_mu_max_abs=0.0931`、`z_mu_batch_std_mean=0.00519`、`kl_sum_mean=0.000866`、`kl_per_dim_mean=0.000027`，相比旧 `action0p6/model_3499` 的 `z_mu_mean_abs≈0.0013` 有改善但仍偏弱。actor ablation 显示 `zero_z_mu` action RMS 仅 `0.00285`，而 `zero_z_m=0.296`、`zero_v_hat=0.128`、`zero_all_estimator=0.285`，说明 actor 仍主要使用 `z_m/v_hat`，几乎不使用 `z_mu`；把 actor latent 替换为 posterior sample 会造成 `0.391` action RMS 扰动，进一步说明不应让 actor 直接吃 sampled `z`。结论：KL mean 修复缓解了 `z_mu` 绝对塌缩，但尚未让 actor 依赖 VAE latent；若目标是论文式 `z` 有效，需要增加 decoder/actor 对 `z_mu` 的使用压力，而不是继续只长训。
- 为回答 cross attention / GRU / gate 是否实际起作用，扩展 `scripts/audit_pie_estimator_pipeline.py`：复现 estimator 内部 forward，统计 cross-attention entropy/max/std、GRF gate、highway gate beta、GRU hidden norm，并加入内部 ablation：`zero_cross_attention`、`zero_gru_hidden`、`highway_f_only`、`highway_gru_only`，分别看 actor action RMS 和 `z_mu/z_m/v_hat/h_f_hat` head RMS。对 KL-mean `model_999.pt` 审计结果：cross attention 起作用，`zero_cross_attention` 造成 actor action RMS `0.127`，head RMS 中 `z_m=0.281`、`v_hat=0.066`；但 attention 不算很尖锐，`attn_entropy_norm=0.891`、`attn_max=0.056`（54 个 visual tokens 均匀值约 0.0185）。GRU hidden 起作用但比 cross attention 弱，`zero_gru_hidden` actor RMS `0.0296`，主要影响 `v_hat=0.0537`，`z_m=0.0487`。highway gate 明显参与融合，`beta_mean=0.510`、`beta_std=0.178`、低/高饱和比例各约 `0.05`，不是死在 0 或 1；强制只用 current fused feature/action RMS `0.170`，只用 GRU path/action RMS `0.209`，说明最终输出确实依赖二者混合。GRF residual gate 也很强，`grf_gate_mean=0.662`、`grf_residual_ratio=4.48`。结论：cross attention、GRU、gate 均在数值上有效；当前 `z_mu` 弱不是因为这些模块完全没工作，而是 `z_mu` head/decoder/actor 使用压力不足。
- 处理 PIE 姿态稳定与课程难度：新增 `reward_base_height_below_target` 和 `reward_contact_force_above_threshold`，用于把低身高和非足部接触作为连续 shaping 信号；新增 `PIEPostureTerminalPenaltyRewardsCfg`、`PIEPostureHeightRewardsCfg`、`PIEPostureHeightWarmupRewardsCfg`、`PIEWarmupTerminationsCfg`，并注册 `StableEasy`、`StableEasyHeight`、`StableWarmup` 等可分阶段 ablation 任务。默认 `Isaac-PIE-Parkour-Unitree-Go2-v0` 未改动，仍保留论文 reward/termination 对齐路径。
- 新增 `UnitreeGo2PIEGentlePPORunnerCfg`：`init_noise_std=0.15`、`action_limit=0.4`、`clip_actions=0.4`、`entropy_coef=0.002`，并注册 `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-Gentle-v0`，用于从低噪声/小动作幅度启动姿态稳定课程。
- 短训结论：`StableEasy` 压住了坏姿态/接触但退化为低底盘早终止（`low_base_height avg20=0.9976`）；`StableEasyHeight` 压住低底盘和姿态，但退化为非法接触终止（`illegal_body_contact avg20=0.9991`）；`StableWarmup` 放开接触终止后仍会在“坏姿态”和“低身高”之间切换；完全无失败终止会变成摔倒后长 episode timeout，不建议作为最终 warmup。
- `StableEasyHeight-Gentle` 是目前最好的方向：短训到 79 iter 后被手动 kill，最后值 `mean_reward=2.17`、`mean_episode_length=981.6`、`bad_base_orientation=0.0156`、`low_base_height=0`、`illegal_body_contact=0.0169`、`reward_base_height_below_target=-0.0060`、`current_goal_idx=0.0547`；近 20 轮均值仍受中段非法接触峰值影响（`illegal_body_contact avg20=0.4753`），但趋势已从失败恢复到稳定 timeout。建议下一次正式 warmup 使用该 task/runner 长训，而不是默认 0.5/1.2 动作探索。
- 将 `UnitreeGo2PIEGentlePPORunnerCfg.save_interval` 改为 `500`。此前 512 env/3000 iter 的 `StableEasyHeight-Gentle` 运行到 481 iter 后进程退出，只留下 `model_0.pt`，因为继承默认 `save_interval=2000`；后续 warmup 每 500 轮保存一次，避免中途退出导致训练进展不可恢复。
- 新增 `StableEasyHeight-Bridge` 过渡配置，用于从 `StableEasyHeight-Gentle/model_2999.pt` 之后继续放开动作与课程：`PIEBridgeCommandsCfg` 将 `lin_vel_x` 放到 `(0.3, 1.2)`、`ang_vel_yaw` 放到 `(-0.8, 0.8)`；`PIEBridgeRewardsCfg` 保留姿态/碰撞安全项但把 `reward_base_height_below_target` 从 `-50` 降到 `-20`；`UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridge` 将课程难度设为 `(0.0, 0.8)` 并降低 flat/demo 占比；`UnitreeGo2PIEBridgePPORunnerCfg` 使用 `clip_actions=0.8`、`action_limit=0.8`、`init_noise_std=0.25`、`entropy_coef=0.005`、`save_interval=500`；注册新任务 `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-Bridge-v0`。验证：相关配置文件 `py_compile` 通过；单独实例化 IsaacLab 配置仍需 AppLauncher，普通 Python 会因缺 `carb/omni` 无法加载传感器模块。
- 已从 `2026-05-07_10-25-17_pie_stable_easy_height_gentle_512_3000_save500/model_2999.pt` 启动 bridge 续训，PID `4124007`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_13-23-08_pie_stable_easy_height_bridge_from_gentle2999_512_3000_save500`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_stable_easy_height_bridge_from_gentle2999_512_3000_save500.stdout.log`。由于 resume 后保留了旧 checkpoint 中约 `0.09` 的 action std，`init_noise_std=0.25` 不会立即生效；前几轮 bridge 指标为负 reward、短 episode、`low_base_height` 上升，需观察到 3000+100/3000+500 再决定是否改为更缓的 bridge 或重置 actor std。
- 新增 `StableEasyHeight-BridgeGaitFix`，用于修复 `Bridge/model_5998.pt` 可正常移动和避让小台阶但退化为右后脚几乎不用、三脚支撑 gait 的问题。新增 `PIEBridgeGaitRewardsCfg`，在 `PIEBridgeRewardsCfg` 基础上加入轻量四脚使用约束：`reward_feet_contact_balance=-0.2`、`reward_feet_vertical_force_balance=-0.4`，均使用 `contact_forces` 的 `.*_foot` body 和 `ema_alpha=0.02`；新增 `UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridgeGaitFix` 并注册任务 `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-BridgeGaitFix-v0`，runner 复用 `UnitreeGo2PIEBridgePPORunnerCfg`。验证：相关配置文件 `py_compile` 通过。建议优先从 `Gentle/model_2999.pt` 而不是 `Bridge/model_5998.pt` 开始 gait-fix bridge 训练，避免在三脚 gait 已固化后再纠正。
- 已停止 bridge play PID `4154123`，并从 `Gentle/model_2999.pt` 启动 `BridgeGaitFix` 续训，PID `4156855`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_15-23-01_pie_stable_easy_height_bridge_gaitfix_from_gentle2999_512_3000_save500`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_stable_easy_height_bridge_gaitfix_from_gentle2999_512_3000_save500.stdout.log`。启动日志确认 `reward_feet_contact_balance` 和 `reward_feet_vertical_force_balance` 生效；前十轮仍有 bridge 初期冲击，`low_base_height` 偏高，需观察到 `3500` checkpoint 再评估 gait 和课程推进。
- 新增 `scripts/diagnose_pie_leg_usage.py`，用于对 PIE checkpoint 做四腿使用诊断：同一 task/checkpoint 下分别运行 zero-action baseline 和 policy rollout，输出每条腿的接触占比、平均垂直力、接触时垂直力、载荷占比、action RMS/abs、关节位置/速度 RMS、torque RMS/abs，并打印 foot/action/joint 映射。对 `BridgeGaitFix/model_5998.pt` 诊断结果：zero-action 时 RL/RR 对称（RR `force_z=32.65N`、`force_share=0.222`），说明默认姿态、接触传感器和右后腿模型不是根因；policy rollout 时 RR `action_rms=0.502` 最大但 `force_z=8.92N`、`force_z_ct=12.98N`、`force_share=0.068`、`torque_rms=4.466` 明显偏低，判断更像策略学到右后轻触/拖腿的局部最优，而不是 action/joint 映射完全错误。
- 扩展 `scripts/diagnose_pie_leg_usage.py`，除垂直载荷外新增机体系前向/水平接触力、接触时脚底滑动速度、每条腿正向/绝对关节功率统计。当前 `contact_forces` 水平分量在诊断中几乎为 0，不适合判断推进；更可靠的推进参考是 `joint_pwr_pos/joint_pwr_abs` 和 play 视觉。对 GentleLoadFix checkpoint 回测：`model_3499.pt` 的 RR 垂直载荷较好（`force_share=0.185`）但正向关节功率偏低（`joint_pwr_pos=0.216`），与 play 中“右腿不提供速度”一致；`model_3250.pt` 的 RR 正向关节功率明显更高（`joint_pwr_pos=0.815`、`joint_pwr_abs=1.024`），但 RR 垂直载荷低于 3499（`force_share=0.156`）。下一步优先 play `model_3250.pt`，不要只按垂直载荷选择 checkpoint。
- 新增 Gentle load-fix 分支：`reward_feet_min_force_share` 使用四脚垂直力 EMA 计算长期载荷占比，对任一脚 `force_share < 0.14` 的缺口做平方惩罚（`min_total_force=20.0` 时启用）；新增 `PIEGentleLoadFixRewardsCfg`、`UnitreeGo2PIEParkourEnvCfg_StableEasyHeightGentleLoadFix` 和 gym task `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0`，runner `UnitreeGo2PIEGentleLoadFixPPORunnerCfg` 继承 Gentle 配置但 `save_interval=250`。`scripts/rsl_rl/train.py` 新增 `--reset_optimizer_on_resume`，用于只加载 checkpoint 权重但重置 PPO optimizer。验证：相关文件 `py_compile` 通过；2 env headless 诊断确认新 task 可注册、`model_2500.pt` 可加载，RewardManager 显示 `reward_feet_min_force_share=-1.0`。
- 新增 Bridge load-fix 分支：`PIEBridgeLoadFixRewardsCfg` 在 `PIEBridgeRewardsCfg` 基础上保留 `reward_feet_min_force_share=-1.0`，新增 `UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridgeLoadFix`、gym task `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-BridgeLoadFix-v0` 和 runner `UnitreeGo2PIEBridgeLoadFixPPORunnerCfg(save_interval=250)`，用于从 GentleLoadFix 的健康 checkpoint 进入 Bridge 时继续约束长期最低载荷。验证：相关文件 `py_compile` 通过；2 env headless 诊断确认新 task 可注册、`model_3250.pt` 可加载，RewardManager 显示 Bridge 高度项 `reward_base_height_below_target=-20.0` 和 `reward_feet_min_force_share=-1.0`。
- 已从 `StableEasyHeight-Gentle/model_2500.pt` 启动 GentleLoadFix 续训 1000 iter，PID `4193901`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_17-48-47_pie_gentle_loadfix_from2500_512_1000_save250`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_from2500_512_1000_save250.stdout.log`。命令使用 `--num_envs 512 --max_iterations 1000 --reset_optimizer_on_resume`，保存间隔由 `UnitreeGo2PIEGentleLoadFixPPORunnerCfg.save_interval=250` 提供；启动日志确认加载 `model_2500.pt`，RewardManager 含 `reward_feet_min_force_share=-1.0`，目前已进入 `Learning iteration 2500/3500`。后续重点诊断 `model_2750.pt`、`model_3000.pt`、`model_3250.pt`、`model_3500.pt` 的四腿载荷、how_far/current_goal_idx/terrain_levels、失败类型和 estimator loss。
- 已按用户要求从 GentleLoadFix 健康 checkpoint `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_17-48-47_pie_gentle_loadfix_from2500_512_1000_save250/model_3250.pt` 启动 BridgeLoadFix 长训，PID `14728`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_19-00-25_pie_bridge_loadfix_from_gentleloadfix3250_512_3000_save250`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_bridge_loadfix_from_gentleloadfix3250_512_3000_save250.stdout.log`。命令使用 `--num_envs 512 --max_iterations 3000 --reset_optimizer_on_resume`，从 `model_3250.pt` 续训并重置 PPO optimizer，保存间隔由 `UnitreeGo2PIEBridgeLoadFixPPORunnerCfg.save_interval=250` 提供；启动日志确认已进入 `Learning iteration 3256/6250`，早期 ETA 约 `01:55`，后续重点观察 `model_3500.pt`、`model_3750.pt`、`model_4000.pt` 的 RR 正向关节功率、四脚载荷、`current_goal_idx`/`terrain_levels` 和失败类型。
- 右后腿不发力问题转为非 reward 链路诊断：扩展 `scripts/diagnose_pie_leg_usage.py`，新增 hip/thigh/calf per-joint 的 action mean/RMS、关节偏移、关节速度、torque 和正/绝对关节功率输出；新增 `scripts/probe_pie_leg_control.py`，在不加载策略的情况下对 RL/RR 后腿施加相同 thigh/calf 正弦动作，比较控制链路响应。验证：`py_compile` 通过；`model_6249.pt` per-joint 诊断显示 zero-action 下 RL/RR 对称，但 policy 下 RR `contact=0.123`、`force_share=0.031`、`joint_pwr_pos=1.806`，RR calf `action_mean=0.792` 且 `vel_rms=1.362`、`pwr_pos=1.553`，对比 RL calf `vel_rms=5.589`、`pwr_pos=43.353`，说明策略把 RR calf 顶在几乎不参与推进的构型；脚本动作探针中 `RL_ONLY` 与 `RR_ONLY` 镜像对称，`RL_RR_BOTH` 下 RL/RR `force_share=0.244/0.240`、`pwr_pos_mean=2.898/2.972`，基本排除 RR 物理/actuator/control mapping 本身弱。
- 新增 `scripts/diagnose_pie_action_estimator.py`，用于同时检查 PIE actor 观测中的 `previous_action` 与 action term 实际 delay 后 `raw_actions` 的偏差，以及 estimator `h_f_hat` 对四个 foot-clearance target 的 per-foot 误差。验证：`py_compile` 通过；在 `model_6249.pt` 上，`obs_prev - executed` 的 leg 平均 RMS 为 FL `0.165`、FR `0.251`、RL `0.210`、RR `0.054`，说明 previous-action 确实不是实际执行动作，但 RR 偏差最小，不是右后腿失效主因；foot-clearance estimator 误差很小，RR `target_mean=0.070`、`pred_mean=0.069`、`rmse=0.012`、`abs_error=0.007`，基本排除 `h_f_hat[RR]` 系统性预测错误。当前更像 actor 自身学到了 RR calf 近似常量高输出/低速度的局部模式，而不是 estimator 或 actuator 链路错误。
- 按用户决定从 0 重新训练，避免从已固化 RR 坏 gait 的 checkpoint 继续纠偏。已启动 `Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0` 从零训练，PID `39360`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_20-58-56_pie_gentle_loadfix_from0_512_3500_save250`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_from0_512_3500_save250.stdout.log`。命令使用 `--num_envs 512 --max_iterations 3500 --seed 1`，不带 `--resume`，保存间隔由 GentleLoadFix runner 的 `save_interval=250` 提供；启动日志已进入 `Learning iteration 6/3500`，初期 `bad_base_orientation/low_base_height/illegal_body_contact` 均为 `0`，ETA 约 2 小时。后续重点在 `model_250/500/1000/1500/2500/3000/3250/3499` 上跑 per-joint 四腿诊断，尽早看 RR calf 是否开始常量高输出。
- `model_500.pt` 已做 per-joint 四腿诊断：RR 没有早期锁死，policy 下 RR `contact=1.000`、`force_share=0.144`、`joint_pwr_pos=0.150`，RR calf `action_mean=0.040`、`action_rms=0.071`、`pwr_pos=0.429`；对比 RL `contact=0.978`、`force_share=0.119`、RL calf `action_mean=0.315`、`pwr_pos=0.760`。结论是 500 轮时后腿整体仍弱，但没有出现之前 `model_6249` 那种 RR calf 高 action/低速度/低接触锁死。为诊断已停止旧 PID `39360`，随后从 `model_500.pt` 续训，当前 PID `45813`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_21-23-14_pie_gentle_loadfix_from0_resume500_512_to3500_save250`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_from0_resume500_512_to3500_save250.stdout.log`，命令为 `--resume --load_run 2026-05-07_20-58-56_pie_gentle_loadfix_from0_512_3500_save250 --checkpoint model_500.pt --max_iterations 3000`，已进入 `Learning iteration 506/3500`。
- 用户观察 `model_750.pt` play 后认为当前动作幅度过小，不再新建基础地形任务，改为保留 `StableEasyHeight-GentleLoadFix` 地形/奖励链路，仅放大 Gentle runner 动作：`ParkourRslRlPIEGentleActorCriticCfg.init_noise_std` 从 `0.15` 改为 `0.25`，`action_limit` 从 `0.4` 改为 `0.6`，`UnitreeGo2PIEGentlePPORunnerCfg.clip_actions` 从 `0.4` 改为 `0.6`。已结束 `model_750.pt` play PID `49706` 和此前训练 PID `48618`，并从 0 启动新训练 PID `51013`，run 目录 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_21-54-20_pie_gentle_loadfix_action0p6_from0_512_3500_save250`，stdout `logs/rsl_rl/unitree_go2_pie_parkour/pie_gentle_loadfix_action0p6_from0_512_3500_save250.stdout.log`。命令使用 `--num_envs 512 --max_iterations 3500 --seed 1`，不带 `--resume`；启动日志确认 `Mean action noise std: 0.25`，已进入 `Learning iteration 1/3500`，需重点观察 250/500/750 checkpoint 的动作幅度、右后腿是否发力，以及是否因幅度增大带来 `illegal_body_contact` 或 `low_base_height` 回升。
- `action0p6` 从 0 到 3500 完整训练已跑完，最终 checkpoint 为 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-07_21-54-20_pie_gentle_loadfix_action0p6_from0_512_3500_save250/model_3499.pt`；末尾指标约为 `mean_reward=22.28`、`episode_length=860`、`terrain_levels=3.48`、`current_goal_idx=0.262`、`time_out=0.798`、`bad_base_orientation=0.023`、`low_base_height=0.002`、`illegal_body_contact=0.179`。play 观察显示整体能站稳但缺少明显运动趋势，且前腿/后腿出现横向倾斜/扭姿；per-joint 诊断显示 zero-action RL/RR 对称，policy 下存在明显 hip/calf 静态偏置和 RR 载荷偏低（RR `force_share=0.130`，RR hip `action_mean=0.125`，RL hip `action_mean=-0.187`，FL/FR hip 正偏，calf 多数大正偏）。注意该诊断是在后续配置已改到 action0p8 后加载旧 checkpoint 做的，动作幅度绝对值会受新 cfg 影响，但偏置趋势仍可作为线索。
- 在用户指出应停止盲目改训练前，曾做过但未启动训练的未验证配置改动：`PIEStableCommandsCfg` 改为 `lin_vel_x=(0.5, 1.0)`、`ang_vel_yaw=(-0.3, 0.3)`；`PIEGentleLoadFixRewardsCfg` 改为继承 `PIEFailureTerminalPenaltyRewardsCfg`，取消强姿态/高度 dense shaping 继承，并把 `reward_lin_vel_xy_command_tracking` 权重改为 `4.0`；`ParkourRslRlPIEGentleActorCriticCfg` 改为 `init_noise_std=0.30`、`action_limit=0.8`，`UnitreeGo2PIEGentlePPORunnerCfg.clip_actions=0.8`。这些改动未经过训练验证，后续做复现审计时应先决定保留、回滚或拆成单独 ablation，避免与已完成的 `action0p6` 结果混淆。
- 开始按复现审计第 1 项检查 action/joint 映射：新增 `scripts/audit_pie_action_mapping.py`，不加载策略，逐个 action 维度施加正/负单关节动作，打印 action index、目标 joint、asset joint id/name、scale/default offset、实际 joint delta、其他关节泄漏和对应 foot 在 base frame 的位移。验证：`py_compile` 通过；hip-only 和全 12 维审计均完成。结论：12 维 action 顺序与 asset joint id/name 一致，`0-3=FL/FR/RL/RR hip`、`4-7=FL/FR/RL/RR thigh`、`8-11=FL/FR/RL/RR calf`，没有发现 action 维度错接到错误腿或错误关节；hip 正/负脚端侧向响应成左右镜像，thigh/calf 也按预期主要影响对应腿。`model_3499.pt` 的前后腿横向扭姿更像策略输出静态偏置/目标函数局部最优或后续观测/推理链路问题，不像底层 action mapping 错误。注意当前审计是在未验证 walk-drive cfg 下运行，action mapping 结论可用，但 reward/command 表已不是 `action0p6` 训练时配置。
- 继续按复现审计第 2 项检查 action 处理链路：新增 `scripts/audit_pie_action_pipeline.py`，加载 PIE checkpoint 后逐步记录 `policy_action -> wrapper clipped action -> action_history_buf -> delayed raw_action -> processed joint target -> actual joint_pos`，同时检查 obs 中 `previous_action` 与 action history 是否一致。验证：`py_compile` 通过；对 `action0p6/model_3499.pt` 使用训练时语义 `--policy_action_limit 0.6 --clip_actions_override 0.6` 审计，global checks 为 `obs_previous_action_vs_history_last_rms=0`、`raw_vs_delayed_expected_rms=0`、`processed_delta_vs_raw_rms≈0.0055`、`joint_pos_vs_processed_target_rms≈0.115`，说明 previous_action、delay、raw/processed target 链路自洽；但 policy 本身输出明显静态偏置，如 RR hip `policy_mean≈0.214`、RR thigh `≈0.211`、FL/FR calf `≈0.379/0.340`。同一旧 checkpoint 在当前未验证 cfg 的 `action_limit/clip=0.8` 下审计，policy/target 幅度被放大，`joint_pos_vs_processed_target_rms≈0.156`，RR hip `policy_mean≈0.290`、RR thigh `≈0.323`、FL/FR calf `≈0.448/0.449`。结论：action processing 链路没有发现错位，但 train/play/诊断配置不一致会显著改变旧 checkpoint 行为，后续所有 play/诊断必须显式固定 checkpoint 训练时的 `action_limit/clip_actions/command/reward` 配置或回滚到对应代码版本。
- 继续按复现审计第 3 项检查 PIE 45 维 proprio observation：新增 `scripts/audit_pie_observation.py`，使用 scripted action 跑环境，逐项对照 `policy` obs 的 `[0:3]=root_ang_vel_b*0.25`、`[3:6]=projected_gravity_b`、`[6:9]=command`、`[9:21]=joint_pos-default`、`[21:33]=joint_vel*0.05`、`[33:45]=previous_action`，并检查 `proprioception_history` 最新帧、`estimator_targets.next_proprioception`、critic 前 45 维/base_velocity/height_scan 一致性。验证：`py_compile` 通过；运行 `audit_pie_observation.py --task Isaac-PIE-Parkour-Unitree-Go2-StableEasyHeight-GentleLoadFix-v0 --num_envs 16 --steps 240`，所有 policy slice RMS/max 均为 0，`policy_vs_history_latest`、`policy_vs_target_next_proprio`、`policy_vs_critic_prefix`、`critic_base_velocity`、`critic_height_scan` 均为 0。唯一差异是 `previous_action_vs_raw_delayed_rms≈0.0177`、`max≈0.0308`，原因是 observation 的 `previous_action` 取 `action_history_buf[:, -1]`（上一条 policy command），而 actuator `raw_action` 经过 1-step delay 后执行；该语义与当前实现自洽，但若论文期望 actor 看到“实际执行动作”，这里是潜在不对齐点。
- 新增低探索噪声 PIE 消融：`ParkourRslRlPIELowNoiseActorCriticCfg.init_noise_std=0.5`，并注册 `Isaac-PIE-Parkour-Unitree-Go2-LowNoise-v0`。该任务复用默认 PIE 环境和 reward，只替换 runner 为 `UnitreeGo2PIELowNoisePPORunnerCfg`，用于验证早期 `illegal_body_contact`/`bad_base_orientation` 是否主要由 `scale=1.0` 配合初始 std=1.0 的探索过猛导致。
- 运行低噪声短训 `pie_low_noise_0p5_no_ppo_estimator_grad_128_500` 至 80 iter 后停止：`Mean action noise std` 从 0.50 降到约 0.38，`illegal_body_contact avg20=0.0031`、`bad_base_orientation avg20=0`，但 `low_base_height avg20=0.9984`、`mean_episode_length avg20=10.04`、`goal_reached=0`。结论是低噪声单独解决乱撞，但会立刻收敛到低底盘早终止，下一步需要同时处理低底盘早死激励或动作幅度/站立约束。
- 新增组合消融任务 `Isaac-PIE-Parkour-Unitree-Go2-LowNoise-ClipReward-v0`：环境使用 `UnitreeGo2PIEParkourEnvCfg_ClipReward`，runner 使用 `UnitreeGo2PIELowNoisePPORunnerCfg`，用于验证低噪声能否压住接触/姿态终止，同时 `clip_total_reward=True` 能否避免低底盘早死激励。
- 运行组合短训 `pie_low_noise_0p5_clip_reward_no_ppo_estimator_grad_128_500` 至 134 iter 后停止：`Mean reward avg20=0.0121`、`Mean episode length avg20=33.37`、`low_base_height avg20=0.0635`，低底盘早死明显缓解；但 `illegal_body_contact avg20=0.6539`、`bad_base_orientation avg20=0.3071`、`goal_reached avg20=0`、`terrain_levels avg20=0.0051`，且 `value_function avg20=8.3e-06`，说明 reward clip 让 value/reward 信号几乎变平，episode 延长不等于学到过障。该实验不建议继续长训。
- 新增 `Isaac-PIE-Parkour-Unitree-Go2-LowerNoise-v0`，runner 为 `UnitreeGo2PIELowerNoisePPORunnerCfg`，只把 actor 初始探索噪声降到 `init_noise_std=0.3`，不改 reward、不做全局 reward clip。短训 `pie_lower_noise_0p3_no_ppo_estimator_grad_128_500` 至 60 iter 后停止：`low_base_height avg20=0.9288`、`illegal_body_contact avg20=0.1042`、`bad_base_orientation avg20=0`、`mean_episode_length avg20=12.81`、`goal_reached=0`。结论：更低噪声会压住乱撞，但仍快速学成低底盘早终止。
- 新增 `Isaac-PIE-Parkour-Unitree-Go2-LimitedAction-v0`，runner 为 `UnitreeGo2PIELimitedActionPPORunnerCfg`，使用 `init_noise_std=0.5`，并把 actor `action_limit` 与 wrapper `clip_actions` 都收紧到 `0.6`。短训 `pie_limited_action_0p6_noise0p5_no_ppo_estimator_grad_128_500` 至 64 iter 后停止：`low_base_height avg20=0.9966`、`illegal_body_contact avg20=0.0157`、`bad_base_orientation avg20=0`、`mean_episode_length avg20=11.05`、`goal_reached=0`。结论：限制动作幅度也只是把失败从“撞/翻”转成“低底盘早死”，根因更像失败 termination 与负回报结构形成的 early-death incentive，而不是单纯动作幅度或探索噪声。
- 新增 `reward_failure_terminal_penalty` 与 `PIEFailureTerminalPenaltyRewardsCfg`，只对 `bad_base_orientation`、`low_base_height`、`illegal_body_contact` 这三类失败终止给一次性惩罚（`weight=-250`，在 `dt=0.02` 下约等于单次 `-5`），不惩罚 timeout 或 `goal_reached`；新增 `Isaac-PIE-Parkour-Unitree-Go2-LowNoise-TerminalPenalty-v0`，环境用该 reward 组，runner 继续用 `UnitreeGo2PIELowNoisePPORunnerCfg`。
- 运行 `pie_low_noise_0p5_terminal_penalty_no_ppo_estimator_grad_128_500` 至 81 iter 后停止：`low_base_height avg20=0.0276`，确认 failure terminal penalty 打掉了低底盘早死；但 `bad_base_orientation avg20=0.6533`、`illegal_body_contact avg20=0.3387`、`goal_reached=0`、`current_goal_idx=0`、`mean_episode_length avg20=27.0`，说明失败转为姿态翻倒/非法接触，尚不适合长训。下一步应处理姿态稳定/课程难度，而不是继续只调噪声或动作上限。
- 将 PIE 默认训练切换为“PPO 只更新 actor/critic，estimator 只由 supervised PIE regression/VAE loss 更新”：`ParkourRslRlPIEEstimatorCfg.detach_pie_actor_features=True`、`pie_joint_actor_estimator=False`。actor 仍使用 estimator 输出的 `z_m/z/v_hat/h_f_hat`，但 PPO actor loss 不再反传进 estimator，用于排查早期 `illegal_body_contact` 占满时 PPO 噪声塑形 estimator 的风险。
- 诊断 PIE 低底盘失败：零动作保持 80 步时 reset 相对地形高度约 0.398m，80 步后约 0.300m，`low_frac=0`，确认默认站姿/PD/`minimum_height=0.18` 不是低底盘主因；失败更可能来自负 reward 下低底盘早终止局部最优。新增 `Isaac-PIE-Parkour-Unitree-Go2-ClipReward-v0` ablation task，仅继承默认 PIE 并设置 `clip_total_reward=True`，用于验证不新增 reward 项时是否能消除早死激励。

- 继续修复 PIE 与论文训练设定的 5 个关键不对齐点：默认 `UnitreeGo2PIEParkourEnvCfg` 改用拆分的 `PIETerminationsCfg`，避免失败终止和 timeout 混在 legacy `total_terminates` 中；`ParkourRewardManager` 新增按 env cfg 控制的总 reward 裁剪，`UnitreeGo2PIEParkourEnvCfg.clip_total_reward=False`，使 PIE 负惩罚不再被全局 `clip(min=0)` 吞掉；新增 `PIEActionsCfg`，PIE action 直接表示关节目标偏移弧度，`scale=1.0`、clip/action_limit/wrapper clip 改为 `±1.2`；PIE estimator KL 改为按 latent 维度求和后 batch mean，更接近 VAE KL 公式；PIE proprio 中对角速度和关节速度做固定尺度缩放（0.25/0.05），并让 next proprio target 使用同一尺度。
- 按 PIE 论文对齐默认 reward：`PIERewardsCfg` 删除此前额外加入的 `reward_dof_error`、`reward_hip_pos`、`reward_feet_contact_balance` 三个轻量 gait/posture 正则项，默认 `Isaac-PIE-Parkour-Unitree-Go2-v0` 现在只保留论文 Table I 风格的 10 个 reward term；按用户要求已停止旧 512 env/5000 iter 训练 PID `3963260`，并启动新训练 PID `3990216`，run 目录为 `logs/rsl_rl/unitree_go2_pie_parkour/2026-05-06_14-49-40_pie_paper_reward_vae_kl1_512_30000_save2000`，参数为 512 env、30000 iter、每 2000 iter 保存 checkpoint。
- 进一步贴近 PIE 论文 estimator 设计：在 `ParkourRslRlPIEEstimatorCfg` 中显式启用 `sample_latent_in_training=True`，让训练期 `z` 使用 VAE 重参数采样而不是固定取 posterior mean；同时将 PIE estimator KL loss 权重从 `1e-3` 调整为 `1.0`，使辅助监督目标更接近论文中的 `DKL + MSE(next_proprio) + MSE(height_map) + MSE(base_velocity) + MSE(foot_clearance)` 未缩放形式。
- 针对 PIE 训练中姿态异常、FR/RL 受力偏低和 actor action 饱和的问题做可控修复：保留 legacy `TerminationsCfg` 供默认 PIE/teacher/student 使用，新增 PIE 专用 `PIETerminationsCfg`，把 timeout、goal、姿态、低底盘和非脚接触拆开记录/触发，避免后续 ablation 难以归因。
- 新增 termination 实现：`time_out`、`goal_reached`、`bad_base_orientation(max_roll/max_pitch=0.9)`、`base_height_below_terrain(minimum_height=0.18)` 和 `illegal_body_contact(threshold=5.0)`；低底盘判定使用 `height_scanner` ray hit 的局部地形 median z，避免继续用世界系 `root_z < -0.25`；这些严格 termination 只在 `TermFix`/`FullFix` PIE task 中启用。
- 新增 `PIERegularizedRewardsCfg` 作为 FullFix reward 组：将 `ang_vel_xy` 从 `-0.05` 加强到 `-0.1`、`orientation` 从 `-1.0` 加强到 `-2.0`、`collision` 从 `-10.0` 加强到 `-20.0`，新增 `reward_action_magnitude=-0.002`，并把 `reward_feet_contact_balance` 从 `-0.2` 加强到 `-0.5`；默认 `PIERewardsCfg` 保持轻量版本。
- 新增 `reward_feet_vertical_force_balance`，对四脚垂直支撑力做 EMA 后约束长期 load share 接近均衡，避免只用 contact duty 导致“轻触地但不承重”的腿逃过惩罚；该项只在 `PIERegularizedRewardsCfg` 中启用，默认 PIE 任务不启用。
- 修复 action/feature 尺度问题：`SimpleActorCritic` 新增 `action_limit`，PIE 默认 `action_limit=4.8`，actor mean 通过 `tanh` 限幅；PIE runner 默认 `clip_actions=4.8`；`PPOWithExtractor` 新增 `pie_actor_feature_clip=5.0`，训练和推理构造 PIE actor obs 时都会 clamp `z_m/z/v_hat/h_f_hat`，避免未归一化 estimator latent 把 actor action mean 推爆。
- 验证：`py_compile` 通过；`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_ppo_update.py -q` 通过（13 passed）；2 env headless runtime smoke 确认 TerminationManager 为 5 个 term、RewardManager 为 15 个 term，step 正常；旧 `model_39998.pt` 加载/前向 smoke 通过，`action_abs_max=4.799` 且 PIE feature clamp 生效。
- 将上述修复拆成可做 ablation 的三组配置，避免一次性全量改动导致难以归因：默认 `Isaac-PIE-Parkour-Unitree-Go2-v0` 现在是 A 组，只保留 action limit / wrapper clip / PIE feature clamp，并恢复 legacy `TerminationsCfg` 与轻量 `PIERewardsCfg`；新增 `Isaac-PIE-Parkour-Unitree-Go2-TermFix-v0` 作为 B 组，在 A 组基础上启用 `PIETerminationsCfg`；新增 `Isaac-PIE-Parkour-Unitree-Go2-FullFix-v0` 作为 C 组，在 B 组基础上启用 `PIERegularizedRewardsCfg`（更强姿态/碰撞、action magnitude、四脚垂直力均衡）。
- 进一步验证：`py_compile` 通过；`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_ppo_update.py -q` 通过（13 passed）；headless 配置解析确认 A/B/C 三个 task 的 reward/termination 组合分别为预期，且均使用 `clip_actions=4.8`、`action_limit=4.8`、`pie_actor_feature_clip=5.0`。

## 2026-05-04

- 为 `scripts/rsl_rl/play.py` 新增 `--follow_robot_view` 参数：play 时可强制主 viewport 相机跟随 env_0 的 `robot` root，避免默认 viewport/controller 没对准时看不到 Go2；相机初始设置后每 5 个 step 根据机器人 root 坐标刷新一次。
- 按 `scripts/rsl_rl/demo.py` 的可视化方式重构 `--follow_robot_view`：不再只调用 `sim.set_camera_view`，而是创建 `/World/PlayFollowCamera`，将 Viewport active camera 切到该 camera，并用 `ViewportCameraState` 每帧根据 robot root pose 更新第三人称相机。
- `scripts/rsl_rl/play.py` 的 PIE 任务创建路径改为直接实例化 `ParkourManagerBasedRLEnv(cfg=env_cfg)`，对齐 `demo.py`，避免 PIE play 经 `gym.make`/registry 包装后 GUI 中看不到 Go2 或窗口提前退出；非 PIE 任务仍保留原 `gym.make` 路径。
- 根据 `model_29999.pt` 之前可正常观察的 play 路径，回退上述 `--follow_robot_view` 和 direct-env PIE play 改动：`scripts/rsl_rl/play.py` 恢复为原始 `gym.make(...)+ParkourRslRlVecEnvWrapper` 逻辑，避免额外相机/direct-env 分支干扰 GUI Stage 中 Go2 显示。

## 2026-05-01

- 将 PIE 环境奖励切换为论文 Table I 风格：新增 `PIERewardsCfg` 并让 `UnitreeGo2PIEParkourEnvCfg.rewards` 使用它，保留原 `TeacherRewardsCfg` 不变以避免影响 teacher/student 任务。
- 新增论文奖励函数实现：command-frame `lin_vel_xy` / `yaw_rate` 指数跟踪、paper 版本 `lin_vel_z` 与 orientation 惩罚、`joint_power=|tau|*|joint_vel|`、平方 action rate、二阶 action smoothness；PIE reward 移除 `feet_edge`、`feet_stumble`、`dof_error`、`hip_pos`、`delta_torques` 等非论文项。
- PIE reward 权重对齐论文：`lin_vel_xy=1.5`、`yaw_rate=0.5`、`lin_vel_z=-1.0`、`ang_vel_xy=-0.05`、`orientation=-1.0`、`dof_acc=-2.5e-7`、`joint_power=-2e-5`、`collision=-10.0`、`action_rate=-0.01`、`smoothness=-0.01`；collision 以“非 foot contact 计数”为正值函数配负权重实现惩罚。
- 验证：`py_compile` 通过；1 env headless PIE env smoke 成功，RewardManager 显示 10 个论文 reward term，step reward finite。
- 对 `model_29999.pt` 做 gait 诊断后，确认怪姿态主要表现为 FL/RR 接触占比高、FR/RL 接触偏低，RL 动作幅度和 FR/RL 关节偏离默认位较大；基于此在 PIE reward 上增加轻量姿态/接触正则：`reward_dof_error=-0.01`、`reward_hip_pos=-0.1`、`reward_feet_contact_balance=-0.2`。
- 新增 `reward_feet_contact_balance`，用四足 contact 的 EMA duty 差值平方作为惩罚，避免某两只脚长期主支撑而不固定 gait phase；验证通过 `py_compile` 和 1 env headless PIE env smoke，RewardManager 显示 13 个 reward term 且 step reward finite。

## 2026-04-30

- 将 PIE RSL-RL 配置的 `save_interval` 从 100 调整为 2000，用于从 `2026-04-30_12-57-49/model_999.pt` 继续长训时每 2000 iteration 保存 checkpoint。
- 实现 PIE actor-estimator 联训：`PPOWithExtractor` 新增 `pie_joint_actor_estimator` 路径，PPO update 时不再直接使用 rollout 中 inference-mode 缓存的 116 维 actor obs，而是从 storage 内的 policy 前 45 维 obs 和 `PIEEstimatorRolloutStorage` 中保存的 depth/proprio 序列按时间重算 `z_m/z/v_hat/h_f_hat`，再拼回 actor obs；该路径保留 GRU 时间顺序并在 done 后清零 hidden，使 surrogate/entropy 等 actor loss 可以反传到 PIE estimator。
- 联训 optimizer 接入：PPO minibatch backward 时同时 zero/clip/step policy optimizer 和 estimator optimizer；保留 rollout 后的 supervised PIE estimator loss 更新，因此 estimator 现在同时接收 actor loss 梯度和 `v/h_f/height/next_proprio/KL` 辅助监督。新增日志标量 `pie_actor_estimator_joint`，训练 smoke 中为 `1.0000`；同时覆盖 multi-GPU reduce/broadcast，使联训时 estimator 参数也会同步。
- 更新 PIE 默认配置：`detach_pie_actor_features=False`、`pie_joint_actor_estimator=True`、`pie_policy_obs_dim=45`、`pie_actor_estimator_grad_scale=1.0`，actor 输入仍为 116 维，critic 仍为 180 维 privileged obs。
- 补充联训单测：新增 “监督 loss 全置 0 时，PPO actor update 仍会改变 estimator 参数” 测试，确认 estimator 参数变化来自 actor loss 路径；验证通过 `py_compile`、`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_rollout_storage.py parkour_test/test_pie_estimator_ppo_update.py -q`（16 passed）、`parkour_test/pie_rsl_runner_smoke.py --headless --num_envs 1 --num_steps 4`，以及 4 env / 5 iteration 的 `scripts/rsl_rl/train.py --task Isaac-PIE-Parkour-Unitree-Go2-v0 --headless --enable_cameras --num_envs 4 --max_iterations 5`。
- 对齐论文 PIE 的 asymmetric critic 输入：新增 `pie_critic_observation = [proprioception(45), true base_velocity(3), true height_scan(132)]`，并在 `PieObservationsCfg` 增加 `critic` observation group（180 维）。`OnPolicyRunnerWithExtractor` 原有逻辑会自动检测 `extras["observations"]["critic"]`，因此 actor 仍使用 116 维 PIE feature obs，critic/value/return 改用 180 维 privileged obs；真实 runner smoke 中确认 `Simple Critic MLP` 第一层为 `Linear(in_features=180, ...)`，`privileged_obs_type=critic`。
- 补充 critic 路径验证：更新 env smoke 检查 `critic` shape，runner smoke 检查 `num_privileged_obs=180`；验证通过 `py_compile`、`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_rollout_storage.py parkour_test/test_pie_estimator_ppo_update.py -q`（15 passed）、`parkour_test/pie_estimator_env_smoke.py --headless --num_envs 2`、`parkour_test/pie_rsl_runner_smoke.py --headless --num_envs 1 --num_steps 4`，以及 4 env / 5 iteration 的 `scripts/rsl_rl/train.py --task Isaac-PIE-Parkour-Unitree-Go2-v0 --headless --enable_cameras --num_envs 4 --max_iterations 5`。
- 将 PIE estimator 的训练路径改为按 rollout 时间序列训练 GRU：`PIEEstimatorRolloutStorage` 新增 `sequence_mini_batch_generator`，按 env 维切 minibatch 且保留 `(T, B, ...)` 时间维；`PPOWithExtractor.update_pie_estimator_from_storage()` 默认启用 `pie_train_gru_sequence=True`，在每个 sequence minibatch 内从 `t=0` 逐步 forward estimator、把 `rnn_hidden` 传到下一步，并在当前 transition 的 `done` 后将对应 env hidden 清零，使训练期 GRU memory 和推理期在线 memory 更一致；保留 flat minibatch fallback。
- 补充 sequence storage/updater 单测：覆盖 sequence minibatch 保序、1 env 时 minibatch 数自动退化、以及 estimator update 中后续 step 确实收到非空 hidden；验证通过 `py_compile`、`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_rollout_storage.py parkour_test/test_pie_estimator_ppo_update.py -q`（15 passed）、`parkour_test/pie_rsl_runner_smoke.py --headless --num_envs 1 --num_steps 4`，以及 4 env / 5 iteration 的 `scripts/rsl_rl/train.py --task Isaac-PIE-Parkour-Unitree-Go2-v0 --headless --enable_cameras --num_envs 4 --max_iterations 5`。
- 进一步贴近论文 PIE 的 latent/decoder 结构：`PIEEstimator` 新增真实 latent 输出 `z`/`z_t`（默认使用 posterior mean，保留可选训练期采样开关），`height_hat` 改为从 `z_m` decode，`next_proprio_hat` 改为从 `[z, z_m, v_hat, h_f_hat]` decode；PIE actor feature 默认改为 `("z_m", "z", "v_hat", "h_f_hat")`，actor 输入维度从 84 扩到 116，配置中 KL 权重从 0 调为 `1e-3`。注意旧 84 维 checkpoint（如 `model_2999.pt`）不能直接加载到新 116 维 actor。
- 同步更新 PIE shape/unit tests 和 env smoke 断言：新增 `z`/`z_t` 输出 shape 检查，覆盖 `height_decoder`/`next_proprio_decoder` 输入维度以及 116 维 actor obs 拼接路径；验证通过 `py_compile`、`pytest parkour_test/test_pie_estimator.py parkour_test/test_pie_estimator_loss.py parkour_test/test_pie_estimator_rollout_storage.py parkour_test/test_pie_estimator_ppo_update.py -q`（14 passed）、`parkour_test/pie_rsl_runner_smoke.py --headless --num_envs 1 --num_steps 4`，以及 4 env / 5 iteration 的 `scripts/rsl_rl/train.py --task Isaac-PIE-Parkour-Unitree-Go2-v0 --headless --enable_cameras --num_envs 4 --max_iterations 5`。
- 修复 PIE actor 推理链路：新增 `PIEActorInferenceWrapper`，推理时从 env `obs_dict["policy"]`、`depth_camera`、`proprioception_history` 构造与训练一致的 84 维 actor obs，并维护/按 done reset PIE estimator GRU hidden；`OnPolicyRunnerWithExtractor.get_inference_policy()` 在 PIE actor feature 开启时自动返回该 wrapper，另提供显式 `get_pie_inference_policy()`。
- 更新 `scripts/rsl_rl/play.py` 和 `scripts/rsl_rl/evaluation.py`：PIE 非蒸馏任务走新的 obs_dict wrapper，不再读取 PIE estimator cfg 中不存在的 `num_prop/num_scan/num_priv_explicit`，也不再走旧的 45 维 obs privileged-state 覆写路径；play 中 PIE 跳过旧 teacher exporter，避免导出不完整的 84 维 actor-only 策略。
- 增强 `evaluation.py` 的可观测指标：除 reward/episode length/waypoint 外，新增 raw goal idx、terrain level、done rate、terminated rate 输出，并在 episode done 后重置 episode length 累计。
- 新增 `Isaac-PIE-Parkour-Unitree-Go2-Eval-v0` 注册和 `UnitreeGo2PIEParkourEnvCfg_EVAL`，用于直接跑 PIE evaluation；补充单测覆盖 PIE inference wrapper 使用 obs_dict 构造 action 和 done hidden reset。
- 验证：`python -m py_compile` 已通过相关改动文件；当前默认 Python 环境缺少 `torch` 和 `pytest`，未能执行 PIE 单测函数。
- 运行 PIE play 时发现 `torch.inference_mode()` 生成的 GRU hidden 在 step 后 reset 触发 “Inplace update to inference tensor outside InferenceMode” 报错；`reset_pie_actor_hidden(dones)` 改为 clone 后替换 hidden，再清零 done 环境，避免对 inference tensor 直接原地写入。

## 2026-04-29

- 将 PIE estimator 的确定性输出接入 actor 输入：`PPOWithExtractor` 新增 `build_pie_actor_observations`，把 45 维 policy proprio obs 与 `detach(z_m, v_hat, h_f_hat)` 拼接成 84 维 actor obs；同时为该路径维护 PIE GRU hidden，env done 时清零对应环境 hidden，estimator 更新后重置 hidden，保证 actor 特征路径使用在线 memory 且 PPO loss 不反传到 estimator。
- 更新 `OnPolicyRunnerWithExtractor.learn_rl`：PPO storage 的 actor observation shape 使用 policy 配置中的 `num_actor_obs`，rollout 采样 action 前先构造 PIE actor obs，critic/value/return 仍使用原 45 维 observation；step 后继续写入 transition-aligned estimator storage 并重置 done 环境的 PIE actor hidden。
- 更新 PIE RSL-RL 配置：`ParkourRslRlPIEActorCriticCfg.num_actor_obs=84`，`ParkourRslRlPIEEstimatorCfg` 启用 `use_pie_actor_features=True`、`detach_pie_actor_features=True`，默认 actor features 为 `("z_m", "v_hat", "h_f_hat")`；补充单测覆盖 84 维 actor/45 维 critic 前向和 PIE actor obs 拼接/hidden reset。

## 2026-04-28

- 加固 `parkour_isaaclab/envs/mdp/observations.py` 的 depth 图像归一化：归一化前先用相机最大距离替换 `nan/+inf`、用 0 替换 `-inf`，再 clamp 到 `[0, max_distance]`，最后缩放到 `[-0.5, 0.5]`，避免异常深度值污染 PIE/depth encoder 输入。
- 重构 `scripts/rsl_rl/modules/feature_extractors/pie_estimator.py` 的跨模态融合：移除 `TransformerEncoder` self-attention 路径，改为 CReF-style `MultiheadAttention`，由完整 10 帧 proprio history token 查询 54 个 depth visual tokens；新增 depth/proprio LayerNorm、GRF gated residual fusion、GRU recurrent projector 和 highway output gate，所有估计头改为从 highway 后的 `fusion_dim*2` 表示输出；默认 depth feature map 改为 `(6, 9)`，GRU hidden size 改为 256；同步更新 PIE estimator shape test 和 env smoke 的 rnn hidden shape 断言。
- 新增 PIE estimator target 观测组 `estimator_targets`，以非拼接 dict 暴露 `base_velocity(3)`、`foot_clearance(4)`、`height_scan(132)`、`next_proprioception(45)`；其中 foot clearance 通过脚端位置匹配最近 height-scanner ray hit 估计，next proprio 先提供当前 45 维 proprio target，后续接 rollout 时按 transition 对齐为 next target；新增 `parkour_test/pie_estimator_targets_smoke.py` 检查 reset/step 后 target shape 和 device。
- 将 PIE foot clearance target 从“匹配最近 height-scanner ray hit”的近似方法改为 4 个专用 foot downward RayCaster：`foot_scanner_fl/fr/rl/rr` 分别挂在四个 foot prim，单点向下 raycast 到 `/World/ground`，`pie_foot_clearance_target` 直接用 foot ray origin z 减 hit z，减少台阶/边缘/gap 附近的 target 污染；target smoke 增加 finite 检查，避免 ray miss 被 shape 检查掩盖。
- 新增 `scripts/rsl_rl/modules/feature_extractors/pie_estimator_loss.py`，提供纯 PyTorch `compute_pie_estimator_loss` 和 `PIEEstimatorLoss`，覆盖 `v_hat/base_velocity`、`h_f_hat/foot_clearance`、`height_hat/height_scan`、`next_proprio_hat/next_proprioception` 的 MSE，以及可加权 `z_mu/z_logvar` KL（默认权重 0）；新增随机 tensor backward test 和 PIE env forward+target+loss+backward smoke，仍未接 PPO、runner、storage 或 optimizer。
- 处理 PIE `next_proprioception` 的 transition 对齐：新增 `cache_pie_estimator_targets` 在 env step 前冻结 obs_t target，并新增 `build_pie_transition_targets` 用 obs_{t+1} 的 `policy` proprioception 覆盖 `next_proprioception`，使 estimator 输入 obs_t 对应到真正的 next proprio target；更新随机 tensor 测试和 env loss smoke，当前仍未接 rollout/storage。
- 为 PIE `next_proprioception` loss 增加 done/reset mask：`build_pie_transition_targets` 可接收 `dones` 或 `terminated/truncated` 并输出 `next_proprioception_mask=(~done)`，`compute_pie_estimator_loss` 对 `next_proprio_hat` 使用 masked MSE，全 mask 时返回 0 避免 NaN；更新随机 tensor mask 测试和 env loss smoke。
- 新增 `PIEEstimatorRolloutStorage`，独立缓存 PIE estimator rollout 数据：step 前 clone `obs_t` 的 `depth_camera`、`proprioception_history` 和当前 targets，step 后用 `obs_{t+1}` 构造 transition-aligned targets 和 `next_proprioception_mask`，并提供 `get`/`flatten`/`mini_batch_generator`；新增随机 tensor storage 测试和真实 PIE env rollout-storage smoke，仍未把 estimator loss 接进 PPO update。
- 将 PIE estimator rollout storage 接到 `PPOWithExtractor`/`OnPolicyRunnerWithExtractor` 的可选辅助路径：当 estimator 是 `PIEEstimator` 或配置启用 `use_pie_estimator_rollout` 时，runner 在 rollout 中缓存 obs_t estimator 输入并在 step 后写入 transition-aligned target；PPO update 中跳过旧 RMA privileged-state estimator loss，改为从 PIE storage minibatch 训练 `PIEEstimatorLoss`，但仍不把 estimator 输出接进 actor；新增纯 PyTorch PPO PIE updater 单测。
- 新增 PIE 专用 RSL-RL 训练配置和最小 flat policy：`SimpleActorCritic` 直接消费 45 维 PIE `policy` proprio obs，`rsl_pie_ppo_cfg.py` 注册 `PIEEstimator` + `PPOWithExtractor` 辅助 estimator loss，PIE Gym task 现在带 `rsl_rl_cfg_entry_point`；新增 runner smoke 脚本用于验证 registry/config/wrapper/runner 初始化和一次短 rollout/update。
- 修复 `OnPolicyRunnerWithExtractor.learn_rl` 在 `log_dir=None` smoke/测试场景下仍调用 `store_code_state` 的问题：保存代码状态前先检查 `self.log_dir is not None`。

## 2026-04-27

- 在 `scripts/rsl_rl/train.py` 中为当前 IsaacLab 版本添加本地 `dump_pickle` 兼容实现；当前 IsaacLab 只从 `isaaclab.utils.io` 导出 `dump_yaml`，训练脚本原先的 `dump_pickle` 导入会失败。
- 新增 `parkour_isaaclab/__init__.py`，使 `parkour_isaaclab` 成为可导入的 Python 包；否则训练脚本导入 `parkour_isaaclab.envs` 会失败。
- 将 `ParkourViewportCameraController.__del__` 的键盘事件取消订阅 API 从旧的 `unsubscribe_from_keyboard_events` 更新为当前 Isaac Sim 使用的 `unsubscribe_to_keyboard_events`，并加空值保护。
- 新增 PIE 输入观测通路：`pie_proprioception` 输出 45 维本体感知，`image_features` 支持返回 2 帧深度历史；新增 `PieObservationsCfg` 暴露 `policy`、`proprioception_history` 和 `depth_camera` 三个观测组。
- 新增 `parkour_pie_cfg.py` 和 Gym 任务注册 `Isaac-PIE-Parkour-Unitree-Go2-v0` / `Isaac-PIE-Parkour-Unitree-Go2-Play-v0`，用于验证 PIE 输入数据形状，尚未接入 PIE PPO/actor。
- 新增 `UniformPIEVelocityCommand` / `PIEVelocityCommandCfg`，使 PIE 环境的 `base_velocity` 直接采样 `c_t=[v_x, v_y, omega_yaw]`，范围为 `v_x=[0.0, 1.5]`、`v_y=[0.0, 0.0]`、`omega_yaw=[-1.2, 1.2]`，不再通过 heading target 间接生成 yaw rate。
- 新增 `scripts/rsl_rl/modules/feature_extractors/pie_estimator.py`，实现 PIEEstimator 最小骨架：depth CNN encoder、proprio MLP encoder、MLP/Transformer fusion 占位、GRU、`v_hat`/`h_f_hat`/`z_m`/`z_mu`/`z_logvar` heads，以及 `height_hat`/`next_proprio_hat` decoders；forward 会适配输入 tensor/单 term dict/flatten shape 和 hidden state device/dtype；新增随机 tensor shape test 和 PIE env forward smoke 脚本，并兼容本地 `parkour_tasks` 源码导入布局，失败时显式非零退出；Transformer 占位也复用本地 activation resolver。
- 调整 PIEEstimator 融合结构：depth CNN 保留 2D feature map 并拆成 visual tokens，proprio history 的每个时间步各自编码为 proprio token，shared Transformer encoder 只在单时间步内做 visual/proprio 跨模态推理；Transformer 输出的 proprio token 和 pooled visual tokens concat 成时间序列后再送入 GRU，使 GRU 专注跨时间建模；shape test 新增 cross-modal sequence 断言，Transformer 内部激活默认使用 `gelu`。
- 再次调整 PIEEstimator 融合语义：10 帧 proprio history 先 concat 成 `450` 维整体特征并编码为单个 proprio token，和 depth 2D feature map 拆出的 visual tokens 一次性进入 shared Transformer；Transformer 输出 concat 后作为长度为 1 的 fused step 输入 GRU，GRU 只承担跨 forward 调用的在线 memory；shape test 更新为 `(B, 1, gru_input_size)` 并覆盖 flattened proprio history 输入。
