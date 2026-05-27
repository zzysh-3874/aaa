from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs.mdp import events as isaac_events

from parkour_isaaclab.envs.mdp import parkour_commands
from parkour_isaaclab.envs import ParkourManagerBasedRLEnvCfg
from parkour_tasks.default_cfg import CAMERA_USD_CFG, VIEWER

from .parkour_mdp_cfg import (
    CommandsCfg,
    EventCfg,
    PIEActionsCfg,
    PIEBridgeGaitRewardsCfg,
    PIEBridgeLoadFixRewardsCfg,
    PIEBridgeRewardsCfg,
    PIEFailureTerminalPenaltyRewardsCfg,
    PIEGentleLoadFixRewardsCfg,
    PIEPostureHeightRewardsCfg,
    PIEPostureHeightWarmupRewardsCfg,
    PIEPostureTerminalPenaltyRewardsCfg,
    ParkourEventsCfg,
    PIERegularizedRewardsCfg,
    PIERewardsCfg,
    PIETerminationsCfg,
    PIEWarmupTerminationsCfg,
    PieObservationsCfg,
    StairsBeamRewardsCfg,
    StairsOnlyRewardsCfg,
    GapOnlyRewardsCfg,
    TeacherRewardsCfg,
    FlatStageOneRewardsCfg,
    TerminationsCfg,
)
from .parkour_student_cfg import ParkourStudentSceneCfg


FOOT_RAY_PATTERN_CFG = patterns.GridPatternCfg(
    resolution=0.1,
    size=(0.0, 0.0),
    direction=(0.0, 0.0, -1.0),
)


@configclass
class PIEParkourSceneCfg(ParkourStudentSceneCfg):
    foot_scanner_fl = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FL_foot",
        attach_yaw_only=False,
        pattern_cfg=FOOT_RAY_PATTERN_CFG,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
        debug_vis=False,
    )
    foot_scanner_fr = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/FR_foot",
        attach_yaw_only=False,
        pattern_cfg=FOOT_RAY_PATTERN_CFG,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
        debug_vis=False,
    )
    foot_scanner_rl = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RL_foot",
        attach_yaw_only=False,
        pattern_cfg=FOOT_RAY_PATTERN_CFG,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
        debug_vis=False,
    )
    foot_scanner_rr = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/RR_foot",
        attach_yaw_only=False,
        pattern_cfg=FOOT_RAY_PATTERN_CFG,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
        debug_vis=False,
    )


@configclass
class PIECommandsCfg:
    """PIE command c_t = [v_x_cmd, v_y_cmd, omega_yaw_cmd]."""

    base_velocity = parkour_commands.PIEVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        ranges=parkour_commands.PIEVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 1.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_yaw=(-1.2, 1.2),
        ),
    )


@configclass
class PIEStableCommandsCfg:
    """Reduced command range for early forward-walking bootstrapping."""

    base_velocity = parkour_commands.PIEVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        ranges=parkour_commands.PIEVelocityCommandCfg.Ranges(
            lin_vel_x=(0.5, 1.0),
            lin_vel_y=(0.0, 0.0),
            ang_vel_yaw=(-0.3, 0.3),
        ),
    )


@configclass
class PIEBridgeCommandsCfg:
    """Moderate command range for moving from stable walking to obstacle approach."""

    base_velocity = parkour_commands.PIEVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        ranges=parkour_commands.PIEVelocityCommandCfg.Ranges(
            lin_vel_x=(0.3, 1.2),
            lin_vel_y=(0.0, 0.0),
            ang_vel_yaw=(-0.8, 0.8),
        ),
    )


@configclass
class PIEStableWarmupCommandsCfg:
    """Narrow command range for contact-avoidance warmup."""

    base_velocity = parkour_commands.PIEVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0, 6.0),
        ranges=parkour_commands.PIEVelocityCommandCfg.Ranges(
            lin_vel_x=(0.1, 0.5),
            lin_vel_y=(0.0, 0.0),
            ang_vel_yaw=(-0.3, 0.3),
        ),
    )


@configclass
class UnitreeGo2PIEParkourEnvCfg(ParkourManagerBasedRLEnvCfg):
    """Default PIE task aligned with the paper-style reward and training signals."""

    clip_total_reward = False

    scene: PIEParkourSceneCfg = PIEParkourSceneCfg(num_envs=192, env_spacing=1.0)
    observations: PieObservationsCfg = PieObservationsCfg()
    actions: PIEActionsCfg = PIEActionsCfg()
    commands: PIECommandsCfg = PIECommandsCfg()
    rewards: PIERewardsCfg = PIERewardsCfg()
    terminations: PIETerminationsCfg = PIETerminationsCfg()
    parkours: ParkourEventsCfg = ParkourEventsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**18
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.scene.height_scanner.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_fl.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_fr.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_rl.update_period = self.sim.dt * self.decimation
        self.scene.foot_scanner_rr.update_period = self.sim.dt * self.decimation
        self.scene.contact_forces.update_period = self.sim.dt * self.decimation
        self.scene.terrain.terrain_generator.curriculum = True
        self.actions.joint_pos.use_delay = True
        self.actions.joint_pos.history_length = 8


@configclass
class UnitreeGo2PIEParkourEnvCfg_TermFix(UnitreeGo2PIEParkourEnvCfg):
    """Ablation B: action/feature scale fix plus split stricter terminations."""

    terminations: PIETerminationsCfg = PIETerminationsCfg()


@configclass
class UnitreeGo2PIEParkourEnvCfg_FullFix(UnitreeGo2PIEParkourEnvCfg_TermFix):
    """Ablation C: termination fix plus stronger posture/contact regularization."""

    rewards: PIERegularizedRewardsCfg = PIERegularizedRewardsCfg()


@configclass
class UnitreeGo2PIEParkourEnvCfg_ClipReward(UnitreeGo2PIEParkourEnvCfg):
    """Ablation D: keep PIE terms but clip negative total reward to avoid early-death incentives."""

    clip_total_reward = True


@configclass
class UnitreeGo2PIEParkourEnvCfg_TerminalPenalty(UnitreeGo2PIEParkourEnvCfg):
    """Ablation E: preserve reward signs but penalize failure terminations once."""

    rewards: PIEFailureTerminalPenaltyRewardsCfg = PIEFailureTerminalPenaltyRewardsCfg()


@configclass
class UnitreeGo2PIEParkourEnvCfg_StableEasy(UnitreeGo2PIEParkourEnvCfg):
    """Ablation F: easier initial curriculum plus stronger posture-stability signals."""

    commands: PIEStableCommandsCfg = PIEStableCommandsCfg()
    rewards: PIEPostureTerminalPenaltyRewardsCfg = PIEPostureTerminalPenaltyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.difficulty_range = (0.0, 0.6)
            for key, sub_terrain in terrain_generator.sub_terrains.items():
                if key in ("parkour_flat", "parkour_demo"):
                    sub_terrain.proportion = 0.3
                else:
                    sub_terrain.proportion = 0.1
                if hasattr(sub_terrain, "apply_roughness"):
                    sub_terrain.apply_roughness = False
                if hasattr(sub_terrain, "noise_range"):
                    sub_terrain.noise_range = (0.0, 0.0)
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_by_setting_velocity = None


@configclass
class UnitreeGo2PIEParkourEnvCfg_StableEasyHeight(UnitreeGo2PIEParkourEnvCfg_StableEasy):
    """Ablation G: stable easy curriculum plus explicit base-height stabilization."""

    rewards: PIEPostureHeightRewardsCfg = PIEPostureHeightRewardsCfg()


@configclass
class UnitreeGo2PIEParkourEnvCfg_StableEasyHeightGentleLoadFix(UnitreeGo2PIEParkourEnvCfg_StableEasyHeight):
    """Gentle warmup with a minimum long-term load-share constraint."""

    rewards: PIEGentleLoadFixRewardsCfg = PIEGentleLoadFixRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Make the commands omnidirectional so the policy cannot consistently
        # leave one hind leg idle; forward-only commands biased the gait toward
        # "front-leg propulsion + rear-leg drag" which we observed as RR dragging.
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_yaw = (-1.0, 1.0)
        # Re-enable periodic pushes. The parent StableEasy class disabled
        # push_by_setting_velocity; we turn it back on with modest ±0.5 m/s
        # velocity perturbations every 8 s so three-legged gaits get knocked
        # off balance during training.
        self.events.push_by_setting_velocity = EventTerm(
            func=isaac_events.push_by_setting_velocity,
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
            interval_range_s=(8.0, 8.0),
            is_global_time=True,
            mode="interval",
        )


@configclass
class UnitreeGo2PIEFlatWalkEnvCfg(UnitreeGo2PIEParkourEnvCfg_StableEasyHeightGentleLoadFix):
    """Pure flat-ground walking task for gait bootstrapping."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.curriculum = False
            terrain_generator.random_difficulty = False
            terrain_generator.difficulty_range = (0.0, 0.0)
            for key, sub_terrain in terrain_generator.sub_terrains.items():
                sub_terrain.proportion = 1.0 if key == "parkour_flat" else 0.0
                if hasattr(sub_terrain, "apply_roughness"):
                    sub_terrain.apply_roughness = False
                if hasattr(sub_terrain, "noise_range"):
                    sub_terrain.noise_range = (0.0, 0.0)
                # Force goals onto the centerline (y_range≈0) so that
                # `target_yaw` for reward_tracking_yaw stays close to 0 when
                # the robot is on-center. Otherwise the default y_range=(-0.4,
                # 0.4) creates a goal direction that fights ParkourCommand's
                # heading=0 instruction, and the policy learns a chronic
                # right/left bias instead of true straight-line walking.
                # Use (0, 0.1) instead of (0, 0) because the underlying
                # terrain generator does np.random.randint(low, high) which
                # rejects empty ranges; 0.1m at horizontal_scale=0.08 yields
                # a non-zero discretised range while still keeping target_yaw
                # well below 0.05 rad.
                if hasattr(sub_terrain, "y_range") and key == "parkour_flat":
                    sub_terrain.y_range = (0.0, 0.1)
        # Simplify the task to a pure forward-walking baseline: no reverse, no
        # sidestep, no external pushes. Small yaw range keeps turning trainable.
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_yaw = (-0.3, 0.3)
        self.events.push_by_setting_velocity = None
        # Enable visualization markers so play shows the command arrow / goal.
        self.commands.base_velocity.debug_vis = True
        # parkour goal marker is NOT wired to the command signal in this
        # flat-walk task (command is sampled from uniform ranges), so leaving
        # the goal marker on makes it look like the robot is ignoring the
        # blue sphere. Keep it off so the only visual is the velocity arrow.
        self.parkours.base_parkour.debug_vis = False


@configclass
class UnitreeGo2PIEStairsBeamEnvCfg(UnitreeGo2PIEFlatWalkEnvCfg):
    """Stairs + balance-beam parkour built on top of the FlatWalk reward stack.

    Uses PIEVelocityCommandCfg (fixed forward velocity, no heading mode).
    The policy decides HOW to traverse obstacles purely from depth + proprio;
    the only external signal is "go forward at vx m/s".
    """

    rewards: StairsBeamRewardsCfg = StairsBeamRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Swap in the stairs+beam terrain generator.
        from parkour_isaaclab.terrains.extreme_parkour.config.stairs_beam import (
            STAIRS_BEAM_TERRAINS_CFG,
        )
        self.scene.terrain.terrain_generator = STAIRS_BEAM_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.curriculum = True
            terrain_generator.difficulty_range = (0.0, 1.0)
            terrain_generator.random_difficulty = False
        # Pure forward velocity command. No heading mode, no goal-based
        # direction. The terrain is a straight line so "go forward" is all
        # the policy needs; it learns obstacle traversal from depth vision.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_yaw = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = True
        self.parkours.base_parkour.debug_vis = True
        self.parkours.base_parkour.num_future_goal_obs = 6
        # Enable ray caster visualization during play so you can see the
        # depth scan pattern projected onto the terrain.
        self.scene.depth_camera.debug_vis = True
        # Relax illegal_body_contact to only terminate on base/head contact.
        self.terminations.illegal_body_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["base", "Head_upper", "Head_lower"],
        )


@configclass
class UnitreeGo2PIEStairsOnlyEnvCfg(UnitreeGo2PIEFlatWalkEnvCfg):
    """Pure staircase task on an IsaacLab-style inverted pyramid (bowl).

    The robot spawns at the centre of the bowl (lowest plateau) and must
    climb up the concentric stair rings in any direction. Rewards are the
    StairsOnly minimal set; no goal markers are displayed during play.
    Only base_contact terminates (fell over).
    """

    rewards: StairsOnlyRewardsCfg = StairsOnlyRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        from parkour_isaaclab.terrains.extreme_parkour.config.stairs_only import (
            STAIRS_ONLY_TERRAINS_CFG,
        )
        self.scene.terrain.terrain_generator = STAIRS_ONLY_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        # Forward velocity command only.
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_yaw = (0.0, 0.0)
        self.commands.base_velocity.debug_vis = True
        # Hide goal markers — the bowl has no meaningful waypoints; the
        # policy just needs to climb out.
        self.parkours.base_parkour.debug_vis = False
        self.parkours.base_parkour.num_future_goal_obs = 6
        # Minimal terminations: only base contact (fell over).
        # Remove low_base_height and bad_base_orientation — both are normal
        # during stair climbing.
        self.terminations.low_base_height = None
        self.terminations.bad_base_orientation = None
        self.terminations.illegal_body_contact.params["sensor_cfg"] = SceneEntityCfg(
            "contact_forces",
            body_names=["base"],
        )
        self.terminations.illegal_body_contact.params["threshold"] = 1.0
        # Enable depth camera visualization for play.
        self.scene.depth_camera.debug_vis = True


@configclass
class UnitreeGo2PIEGapOnlyEnvCfg(UnitreeGo2PIEFlatWalkEnvCfg):
    """Gap-crossing corridor task.

    Straight corridor with random-width gaps. Forces vision dependence:
    the policy must detect gaps from depth and jump or adjust stride.
    Uses ParkourCommand (heading control) like the teacher so the robot
    automatically turns toward the next goal.
    """

    rewards: GapOnlyRewardsCfg = GapOnlyRewardsCfg()
    # Match the teacher: a single combined termination (time_out OR
    # roll/pitch>1.5 rad OR root_z<-0.25 OR all goals reached).
    terminations: TerminationsCfg = TerminationsCfg()
    # Use teacher-style heading command instead of PIEVelocityCommand.
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        from parkour_isaaclab.terrains.extreme_parkour.config.gap_only import (
            GAP_ONLY_TERRAINS_CFG,
        )
        self.scene.terrain.terrain_generator = GAP_ONLY_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        # 3 gaps with ~2m spacing, 12m tile — 30s gives enough time even
        # if the policy hesitates at each gap edge.
        self.episode_length_s = 30.0
        # ParkourCommand heading control: heading=0 means "face forward".
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_control_stiffness = 0.8
        self.commands.base_velocity.clips.lin_vel_clip = 0.2
        self.commands.base_velocity.clips.ang_vel_clip = 0.4
        self.commands.base_velocity.debug_vis = True
        self.parkours.base_parkour.debug_vis = True
        self.parkours.base_parkour.num_future_goal_obs = 6
        # Disable all ray caster debug visualizations (height scanner, foot
        # scanners, depth camera) so play stays clean of red/green ray hit
        # markers. Only the velocity arrow + goal markers remain visible.
        self.scene.height_scanner.debug_vis = False
        self.scene.foot_scanner_fl.debug_vis = False
        self.scene.foot_scanner_fr.debug_vis = False
        self.scene.foot_scanner_rl.debug_vis = False
        self.scene.foot_scanner_rr.debug_vis = False
        self.scene.depth_camera.debug_vis = False


@configclass
class UnitreeGo2PIEFullParkourEnvCfg(UnitreeGo2PIEGapOnlyEnvCfg):
    """Teacher-style full-parkour env with PIE network and a gap_corridor slot.

    Goals
    -----
    Reproduce the Teacher policy training environment (5 mixed parkour
    sub-terrains with goal y-offsets) while keeping the PIE network stack
    (PIE estimator + depth camera + actor with proprio/depth features +
    privileged critic). Adds a 6th sub-terrain ``gap_corridor`` (3-gap
    corridor from the GapOnly experiment) so the policy still sees the
    multi-gap layout we previously fine-tuned.

    Differences vs ``UnitreeGo2PIEGapOnlyEnvCfg``
    ---------------------------------------------
    - Terrain: 6-way mix (parkour_gap, parkour_hurdle, parkour_flat,
      parkour_step, parkour, gap_corridor). 16x4m tiles, 10 rows x 40 cols.
    - Reward: ``FlatStageOneRewardsCfg`` (Teacher reward + Stage1
      anti-cheat additions: feet_air_time, dof_pos_limits, collision
      filter without base). Carries the same anti-cheat constraints
      that broke the FlatStage1 prone-shake failure mode so the policy
      cannot regress to the same trick on the obstacle mix.
    - Difficulty range: (0.0, 1.0) instead of (0.0, 0.15).
    - Episode length: 20s (matches Teacher).
    - num_goals: defaults to 8 (terrain generator default).
    - Termination: same ``terminate_episode`` function as Stage1 with
      slightly relaxed cutoffs (1.0 rad vs 0.7 rad orientation, 0.20 m
      vs 0.22 m height) so jumping over hurdles or landing on tilted
      stones is not aborted, while prone gaits still get reset.

    Network / observation / actions are inherited from PIE so the existing
    ``OnPolicyRunnerWithExtractor`` + ``PPOWithExtractor`` + ``PIEEstimator``
    pipeline runs unchanged. Trains from random init on this env produces
    a Teacher-equivalent policy that keeps PIE's depth-vision deployment
    path.
    """

    rewards: FlatStageOneRewardsCfg = FlatStageOneRewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        from parkour_isaaclab.terrains.extreme_parkour.config.full_parkour_with_gap import (
            FULL_PARKOUR_WITH_GAP_TERRAINS_CFG,
        )
        self.scene.terrain.terrain_generator = FULL_PARKOUR_WITH_GAP_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        # Match Teacher's episode budget; 20s gives the policy enough time
        # to traverse the 16m corridor at the typical 0.5-0.8 m/s commanded
        # speed without leaning on time-out resets.
        self.episode_length_s = 20.0
        # ParkourCommand: heading=0 (straight ahead), v_x range copied from
        # the default CommandsCfg (0.3 - 0.8 m/s) which matches Teacher.
        # Heading rotation in (-1.6, 1.6) is also the Teacher default; the
        # GapOnly child class narrowed this to 0, so re-open it here.
        self.commands.base_velocity.ranges.lin_vel_x = (0.3, 0.8)
        self.commands.base_velocity.ranges.heading = (-1.6, 1.6)
        # Teacher uses the default heading_control_stiffness=0.8, which
        # CommandsCfg already provides; explicitly state it for clarity.
        self.commands.base_velocity.heading_control_stiffness = 0.8
        self.commands.base_velocity.clips.lin_vel_clip = 0.2
        self.commands.base_velocity.clips.ang_vel_clip = 0.4
        self.commands.base_velocity.debug_vis = True
        self.parkours.base_parkour.debug_vis = True
        self.parkours.base_parkour.num_future_goal_obs = 6
        # All ray caster debug visualisations stay off (set by the parent
        # GapOnly class). No additional changes needed for play.

        # Re-enable Teacher-style domain randomisation, with smaller
        # magnitudes than Teacher's defaults. The PIE parent chain disabled
        # all three (StableEasy zero-out + FlatWalk further nulling
        # push_by_setting_velocity) which is appropriate for static-walking
        # warmup but leaves a from-scratch parkour policy too brittle.
        from isaaclab.envs.mdp.events import randomize_rigid_body_mass as _rrm
        from parkour_isaaclab.envs.mdp import events as _proj_events
        # base mass ±10–30% of the ~5 kg Go2 base.
        self.events.randomize_rigid_body_mass = EventTerm(
            func=_rrm,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-0.5, 1.5),
                "operation": "add",
            },
        )
        # base COM jitter ±1 cm (half of Teacher's ±2 cm).
        self.events.randomize_rigid_body_com = EventTerm(
            func=_proj_events.randomize_rigid_body_com,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "com_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (-0.01, 0.01),
                },
            },
        )
        # Periodic ±0.3 m/s push every 10 s (vs Teacher's ±0.5 / 8 s).
        self.events.push_by_setting_velocity = EventTerm(
            func=isaac_events.push_by_setting_velocity,
            params={"velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)}},
            interval_range_s=(10.0, 10.0),
            is_global_time=True,
            mode="interval",
        )

        # Termination cutoffs: tighter than Teacher defaults (1.5 rad / -0.25 m)
        # to stop the policy from sliding through episodes in degenerate
        # poses, but slightly looser than Stage 1 since obstacle landings
        # legitimately produce more roll / pitch and brief crouching.
        # Stage 1 was 0.7 / 0.7 / 0.22; Stage 2 uses 1.0 / 1.0 / 0.20 so
        # hurdle and gap landings are not aborted while prone-shake still
        # gets caught.
        self.terminations.total_terminates.params["max_roll"] = 1.0
        self.terminations.total_terminates.params["max_pitch"] = 1.0
        self.terminations.total_terminates.params["minimum_height"] = 0.20


@configclass
class UnitreeGo2PIEFullParkourStage2WarmEnvCfg(UnitreeGo2PIEFullParkourEnvCfg):
    """Stage 2 warm-up environment: full obstacle mix, no domain randomisation.

    The full FullParkour env adds Teacher-style domain randomisation
    (mass +-1.5 kg startup, COM +-1 cm startup, periodic +-0.3 m/s push
    every 10 s). When fine-tuning a Stage 1 walker that has only ever
    seen a perfectly nominal robot on a smooth floor, those three
    perturbations push the policy into states it has never sampled,
    and the robot literally cannot stand within two steps even on the
    smooth row 0 flat sub-terrain.

    This warm-up cfg keeps everything that Stage 2 needs - obstacle
    mix, smooth row 0 (noise_range=(0.0, 0.06)), Stage 1 anti-cheat
    reward set, the slightly relaxed termination cutoffs (1.0 rad /
    0.20 m) - but disables mass / COM / push events. Once the policy
    is again standing and moving forward on the obstacle mix (Stage
    2a), the next stage (Stage 2b) flips back to the full FullParkour
    env to re-introduce robustness perturbations.
    """

    def __post_init__(self):
        super().__post_init__()
        # Disable startup mass / COM jitter and the periodic 10 s push that
        # the FullParkour parent re-enables. This makes the Stage 2 warm-up
        # env a strict superset of Stage 1 (Stage 1 already had these off):
        # the only differences from Stage 1 are obstacle mix + relaxed
        # termination cutoffs + ParkourCommand heading range, all of which
        # the Stage 1 policy can handle.
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_by_setting_velocity = None

        # Lock heading_target to 0 during Stage 2a warm-up. ParkourCommand's
        # heading_target is independent of the goal direction; reward_tracking_yaw
        # rewards turning toward the goal (target_yaw, computed from goal
        # position) but vel_command_b[2] is the omega command driven by
        # heading_target. When the two disagree (which they do whenever the
        # goal is offset in y), the policy gets a contradictory signal: the
        # reward says turn toward the goal, the command says turn toward
        # heading_target. On flat single-corridor terrain the Stage 1 policy
        # learned to ignore omega_cmd; on the obstacle mix it cannot ignore
        # it any more and ends up staggering backwards. Fix: collapse
        # heading_target to 0 so omega_cmd = 0.8 * (0 - robot_yaw) (i.e. the
        # command tells the robot to face +x), aligning with the goal heading
        # which is also approximately +x for the flat corridor layout.
        # Stage 2b can re-open the heading range.
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)

        # Add an explicit illegal-body-contact termination. The default
        # ``terminate_episode`` only resets when world-frame z < 0.20 m,
        # which never fires when the robot is wedged against a 10-15 cm
        # hurdle (base z ~0.30 m, robot stuck pushing forward, accumulating
        # tracking_goal_vel reward without progressing). This term resets
        # the episode when any of base / hip / thigh / head sustains a
        # contact force above 50 N - i.e. genuinely stuck pushing into an
        # obstacle. Threshold 50 N tolerates the brief landing impulse of
        # clearing a hurdle (transient ~30-40 N) while still catching
        # persistent wedging (50-100 N).
        from isaaclab.managers import TerminationTermCfg as _DoneTerm
        from parkour_isaaclab.envs.mdp import terminations as _term_funcs
        self.terminations.illegal_body_contact = _DoneTerm(
            func=_term_funcs.illegal_body_contact,
            time_out=False,
            params={
                "threshold": 50.0,
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[
                        "base", ".*_hip", ".*_thigh",
                        "Head_upper", "Head_lower",
                    ],
                ),
            },
        )

        # Relax pitch cutoff from 1.0 rad to 1.4 rad (~80 deg). Successful
        # traversal of a 30 cm hurdle peaks at ~0.5-0.7 rad pitch in flight,
        # and hurdle landing can transiently hit 0.9-1.0 rad. Keeping the
        # cutoff at 1.0 rad caused the policy to abort the takeoff because
        # the moment it pitched forward to clear the obstacle it was reset.
        # Roll cutoff stays at 1.0 rad - there's no legitimate parkour
        # situation that produces large lateral roll.
        self.terminations.total_terminates.params["max_pitch"] = 1.4


@configclass
class UnitreeGo2PIEFullParkourEasyEnvCfg(UnitreeGo2PIEFullParkourStage2WarmEnvCfg):
    """Easy variant of Stage 2 warm-up: every obstacle starts at 5 cm.

    Layout, rewards, terminations and command stack are identical to
    ``Stage2Warm``. The only change is the per sub-terrain difficulty
    expression, so that ``difficulty=0`` produces a 5 cm obstacle that
    a normal trot stride can clear (Go2 calf is ~15 cm), and
    ``difficulty=1`` reaches roughly half of the regular Stage2Warm peak.
    Use this to bridge a Stage 1 flat-only walker to the multi-corridor
    obstacle layout without the policy being immediately killed by 10-15
    cm hurdles / steps on row 0.

    Difficulty progressions (vs Stage2Warm parent in parens):
        gap_size:    0.05 + 0.30*d   (parent: 0.05 + 0.65*d)
        hurdle_h:    (0.05+0.05*d, 0.05+0.10*d)  (parent: (0.1+0.1*d, 0.15+0.25*d))
        step_h:      0.05 + 0.10*d   (parent: 0.10 + 0.35*d)
        incline_h:   0.05 + 0.05*d   (parent: 0.25*d)
        last_inc_h:  0.05 + 0.05*d   (parent: 0.10 - 0.10*d + incline)

    parkour_flat is left untouched because apply_flat=True already masks
    its built-in hurdles to a flat strip.
    """

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        if gen is None:
            return

        gap = gen.sub_terrains.get("parkour_gap")
        if gap is not None:
            gap.gap_size = "0.05 + 0.30*difficulty"

        hurdle = gen.sub_terrains.get("parkour_hurdle")
        if hurdle is not None:
            # numpy.random.randint requires low < high. At difficulty=0
            # the previous "(0.05, 0.05)" gave low==high → ValueError. Add
            # a permanent 1 cm gap between low and high so randint stays
            # well-defined and the policy still sees small variation.
            hurdle.hurdle_height_range = "0.05 + 0.05*difficulty, 0.06 + 0.10*difficulty"

        step = gen.sub_terrains.get("parkour_step")
        if step is not None:
            step.step_height = "0.05 + 0.10*difficulty"

        parkour = gen.sub_terrains.get("parkour")
        if parkour is not None:
            parkour.incline_height = "0.05 + 0.05*difficulty"
            parkour.last_incline_height = "0.05 + 0.05*difficulty"

        # Loosen the "reached goal" radius from default 0.2 m to 0.35 m so a
        # Stage 1 walker (which has only seen y_range=(-0.2, 0.2)) does not
        # have to perfectly aim each goal during the Easy finetune. Larger
        # radius means: more goal_reached bonus per episode, less penalty
        # for slight off-line tracking, faster terrain_levels promotion.
        # Stage 2b (full difficulty) can drop it back to 0.2.
        self.parkours.base_parkour.next_goal_threshold = 0.35


@configclass
class UnitreeGo2PIEFlatParkourEnvCfg(UnitreeGo2PIEFullParkourEnvCfg):
    """Flat-only stage 1 of the FullParkour curriculum.

    Same network / obs / command stack as ``UnitreeGo2PIEFullParkourEnvCfg``
    so checkpoints transfer cleanly to FullParkour for stage 2 finetuning.
    What changes:

    - Reward: ``FlatStageOneRewardsCfg`` (Teacher reward + feet_air_time
      and Unitree-rough-style rebalanced torque/action_rate weights). This
      stops the policy from learning the "drag two rear calves at action
      limit" failure mode that the bare TeacherRewardsCfg tolerated on
      flat ground.
    - Terrain: 100% ``parkour_flat`` (one of the Teacher 5-terrain mix slots,
      already shaped as 'goals on a flat strip with low roughness').
      ``apply_roughness`` is also turned off so the surface is true flat.
    - Difficulty range narrowed to (0.0, 0.3); the parkour_flat sub-terrain's
      hurdle_height_range is irrelevant because apply_flat=True replaces
      hurdles with a flat strip.
    - Domain randomisation disabled (mass / COM / push) so the from-scratch
      walking policy isn't fighting unmodelled dynamics during bootstrap.
    - episode_length_s = 20s (inherited from FullParkour) is enough for goals
      on a 16m tile at 0.3-0.8 m/s commanded velocity.

    Use this as stage 1; once the policy walks reliably (e.g.
    `how_far_from_start > 5m`, `terrain_levels` rising), resume into
    ``UnitreeGo2PIEFullParkourEnvCfg`` for the full obstacle mix.
    """

    rewards: FlatStageOneRewardsCfg = FlatStageOneRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # All sub-terrains -> proportion 0 except parkour_flat.
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            for key, sub_terrain in terrain_generator.sub_terrains.items():
                sub_terrain.proportion = 1.0 if key == "parkour_flat" else 0.0
                # Even the parkour_flat slot has hidden hurdle bumps
                # depending on its cfg; keep apply_flat True (its default
                # in FULL_PARKOUR_WITH_GAP_TERRAINS_CFG) and turn off the
                # roughness perturbation so this stage is genuinely flat.
                if hasattr(sub_terrain, "apply_roughness"):
                    sub_terrain.apply_roughness = False
                if hasattr(sub_terrain, "noise_range"):
                    sub_terrain.noise_range = (0.0, 0.0)
                # Open up goal y offset so the policy actually learns to
                # turn toward goals instead of sliding straight along +x.
                # Teacher uses (-0.4, 0.4); we use (-0.2, 0.2) as a
                # halfway compromise: target_yaw can reach ~12 deg, which
                # is enough that reward_tracking_yaw becomes a real
                # gradient signal but well below Teacher's ~22 deg so
                # walking + turning can still both be learned at once.
                if hasattr(sub_terrain, "y_range") and key == "parkour_flat":
                    sub_terrain.y_range = (-0.2, 0.2)
            # Narrow difficulty so the curriculum resampler always gives easy
            # goals; no hidden bumps mean difficulty doesn't really matter
            # for parkour_flat, but we keep it bounded for clarity.
            terrain_generator.difficulty_range = (0.0, 0.3)
        self.scene.terrain.max_init_terrain_level = 0

        # Disable domain randomisation during the walking-bootstrap stage. The
        # parent FullParkour cfg explicitly turned these on; we re-disable.
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_by_setting_velocity = None

        # Tighten orientation cutoffs for ``terminate_episode`` so prone /
        # heavily-tilted gaits get reset instead of riding out the 20 s
        # episode and banking a fake-positive Train/mean_reward. The
        # default 1.5 rad (~86 deg) is meant for parkour landings on
        # tilted obstacles; on flat terrain a normal trot is well under
        # 0.2 rad of roll/pitch, so 0.7 rad (~40 deg) is a tight signal
        # for "policy is failing to stand".
        # Also add an absolute base-z floor at 0.22 m. Default Go2 stance
        # is ~0.32 m, so 0.22 m gives 10 cm crouch room while terminating
        # any "prone-shake" gait long before it accumulates a Train reward.
        self.terminations.total_terminates.params["max_roll"] = 0.7
        self.terminations.total_terminates.params["max_pitch"] = 0.7
        self.terminations.total_terminates.params["minimum_height"] = 0.22


@configclass
class UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridge(UnitreeGo2PIEParkourEnvCfg_StableEasy):
    """Bridge curriculum after Gentle warmup: faster commands and less height shaping."""

    commands: PIEBridgeCommandsCfg = PIEBridgeCommandsCfg()
    rewards: PIEBridgeRewardsCfg = PIEBridgeRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.difficulty_range = (0.0, 0.8)
            for key, sub_terrain in terrain_generator.sub_terrains.items():
                if key == "parkour_flat":
                    sub_terrain.proportion = 0.15
                elif key == "parkour_demo":
                    sub_terrain.proportion = 0.05
                else:
                    sub_terrain.proportion = 0.2


@configclass
class UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridgeLoadFix(UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridge):
    """Bridge curriculum with a minimum long-term load-share constraint."""

    rewards: PIEBridgeLoadFixRewardsCfg = PIEBridgeLoadFixRewardsCfg()


@configclass
class UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridgeGaitFix(UnitreeGo2PIEParkourEnvCfg_StableEasyHeightBridge):
    """Bridge curriculum with light four-foot usage regularization."""

    rewards: PIEBridgeGaitRewardsCfg = PIEBridgeGaitRewardsCfg()


@configclass
class UnitreeGo2PIEParkourEnvCfg_StableWarmup(UnitreeGo2PIEParkourEnvCfg):
    """Ablation H: easy curriculum with posture/height termination and contact reward shaping."""

    commands: PIEStableWarmupCommandsCfg = PIEStableWarmupCommandsCfg()
    rewards: PIEPostureHeightWarmupRewardsCfg = PIEPostureHeightWarmupRewardsCfg()
    terminations: PIEWarmupTerminationsCfg = PIEWarmupTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = 0
        terrain_generator = self.scene.terrain.terrain_generator
        if terrain_generator is not None:
            terrain_generator.difficulty_range = (0.0, 0.3)
            for key, sub_terrain in terrain_generator.sub_terrains.items():
                if key in ("parkour_flat", "parkour_demo"):
                    sub_terrain.proportion = 0.4
                else:
                    sub_terrain.proportion = 0.05
                if hasattr(sub_terrain, "apply_roughness"):
                    sub_terrain.apply_roughness = False
                if hasattr(sub_terrain, "noise_range"):
                    sub_terrain.noise_range = (0.0, 0.0)
        self.events.randomize_rigid_body_mass = None
        self.events.randomize_rigid_body_com = None
        self.events.push_by_setting_velocity = None


@configclass
class UnitreeGo2PIEParkourEnvCfg_EVAL(UnitreeGo2PIEParkourEnvCfg):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 256
        self.episode_length_s = 20.0
        self.scene.depth_camera_usd = CAMERA_USD_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.debug_vis = True
        self.commands.base_velocity.resampling_time_range = (60.0, 60.0)
        self.parkours.base_parkour.debug_vis = True
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.random_difficulty = True
            self.scene.terrain.terrain_generator.difficulty_range = (0.0, 1.0)
            for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
                if key in ("parkour_flat", "parkour_demo"):
                    sub_terrain.proportion = 0.0
                else:
                    sub_terrain.proportion = 0.25
                    sub_terrain.noise_range = (0.02, 0.02)
        self.events.randomize_rigid_body_com = None
        self.events.randomize_rigid_body_mass = None
        if self.events.push_by_setting_velocity is not None:
            self.events.push_by_setting_velocity.interval_range_s = (6.0, 6.0)


@configclass
class UnitreeGo2PIEParkourEnvCfg_PLAY(UnitreeGo2PIEParkourEnvCfg):
    viewer = VIEWER

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 60.0
        self.scene.depth_camera_usd = CAMERA_USD_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.commands.base_velocity.debug_vis = True
        self.parkours.base_parkour.debug_vis = True
        # Disable ray caster debug visualizations (height scanner, foot scanners,
        # depth camera) so play doesn't render the red/green ray hit markers.
        self.scene.height_scanner.debug_vis = False
        self.scene.foot_scanner_fl.debug_vis = False
        self.scene.foot_scanner_fr.debug_vis = False
        self.scene.foot_scanner_rl.debug_vis = False
        self.scene.foot_scanner_rr.debug_vis = False
        self.scene.depth_camera.debug_vis = False
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.difficulty_range = (0.7, 1.0)
        self.events.push_by_setting_velocity = None
        for key, sub_terrain in self.scene.terrain.terrain_generator.sub_terrains.items():
            if key == "parkour_flat":
                sub_terrain.proportion = 0.0
            else:
                sub_terrain.proportion = 0.25
                sub_terrain.noise_range = (0.02, 0.02)


