from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.envs.mdp.events import ( 
randomize_rigid_body_mass,
apply_external_force_torque,
reset_joints_by_scale

)
from isaaclab.envs.mdp.rewards import undesired_contacts
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import rewards as isaac_mdp
from parkour_isaaclab.envs.mdp.parkour_actions import DelayedJointPositionActionCfg 
from parkour_isaaclab.envs.mdp import terminations, rewards, parkours, events, observations, parkour_commands

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = parkour_commands.ParkourCommandCfg(
        asset_name="robot",
        resampling_time_range=(6.0,6.0 ),
        heading_control_stiffness=0.8,
        ranges=parkour_commands.ParkourCommandCfg.Ranges(
            lin_vel_x=(0.3, 0.8), 
            heading=(-1.6, 1.6)
        ),
        clips= parkour_commands.ParkourCommandCfg.Clips(
            lin_vel_clip = 0.2,
            ang_vel_clip = 0.4
        )
    )

@configclass
class ParkourEventsCfg:
    """Command specifications for the MDP."""
    base_parkour = parkours.ParkourEventsCfg(
        asset_name = 'robot',
        )

@configclass
class TeacherObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={            
            "asset_cfg":SceneEntityCfg("robot"),
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "parkour_name":'base_parkour',
            "history_length": 10
            },
            clip= (-100,100)
        )
    policy: PolicyCfg = PolicyCfg()

@configclass
class StudentObservationsCfg:

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        extreme_parkour_observations = ObsTerm(
            func=observations.ExtremeParkourObservations,
            params={            
            "asset_cfg":SceneEntityCfg("robot"),
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "parkour_name":'base_parkour',
            "history_length": 10,
            },
            clip= (-100,100)
        )

    @configclass
    class DepthCameraPolicyCfg(ObsGroup):
        depth_cam = ObsTerm(
            func=observations.image_features,
            params={            
            "sensor_cfg":SceneEntityCfg("depth_camera"),
            "resize": (58, 87),
            "buffer_len": 2,
            "debug_vis":True
            },
        )

    @configclass
    class DeltaYawOkPolicyCfg(ObsGroup):
        deta_yaw_ok =  ObsTerm(
            func=observations.obervation_delta_yaw_ok,
            params={            
            "parkour_name":'base_parkour',
            'threshold': 0.6
            },
        )
    policy: PolicyCfg = PolicyCfg()
    depth_camera: DepthCameraPolicyCfg = DepthCameraPolicyCfg()
    delta_yaw_ok: DeltaYawOkPolicyCfg = DeltaYawOkPolicyCfg()


@configclass
class PieObservationsCfg:
    """Observation groups used by the PIE one-stage estimator."""

    @configclass
    class PolicyCfg(ObsGroup):
        proprioception = ObsTerm(
            func=observations.pie_proprioception,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "action_name": "joint_pos",
            },
            clip=(-100, 100),
        )

    @configclass
    class ProprioceptionHistoryCfg(ObsGroup):
        proprioception = ObsTerm(
            func=observations.pie_proprioception,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "action_name": "joint_pos",
            },
            history_length=10,
            flatten_history_dim=False,
            clip=(-100, 100),
        )

    @configclass
    class DepthCameraCfg(ObsGroup):
        depth = ObsTerm(
            func=observations.image_features,
            params={
                "sensor_cfg": SceneEntityCfg("depth_camera"),
                "resize": (58, 87),
                "buffer_len": 2,
                "debug_vis": False,
                "return_history": True,
            },
        )

    @configclass
    class EstimatorTargetsCfg(ObsGroup):
        base_velocity = ObsTerm(
            func=observations.pie_base_velocity_target,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        foot_clearance = ObsTerm(
            func=observations.pie_foot_clearance_target,
            params={
                "sensor_names": (
                    "foot_scanner_fl",
                    "foot_scanner_fr",
                    "foot_scanner_rl",
                    "foot_scanner_rr",
                ),
            },
        )
        height_scan = ObsTerm(
            func=observations.pie_height_scan_target,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )
        next_proprioception = ObsTerm(
            func=observations.pie_proprioception,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "action_name": "joint_pos",
            },
            clip=(-100, 100),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CriticCfg(ObsGroup):
        critic = ObsTerm(
            func=observations.pie_critic_observation,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "base_velocity",
                "action_name": "joint_pos",
                "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            },
            clip=(-100, 100),
        )

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    proprioception_history: ProprioceptionHistoryCfg = ProprioceptionHistoryCfg()
    depth_camera: DepthCameraCfg = DepthCameraCfg()
    estimator_targets: EstimatorTargetsCfg = EstimatorTargetsCfg()


@configclass
class StudentRewardsCfg:
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-0., 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base",".*_calf",".*_thigh"]),
        },
    )
    

@configclass
class TeacherRewardsCfg:
    """Reward terms for the MDP.
    ['base', 
    'FL_hip', 
    'FL_thigh', 
    'FL_calf', 
    'FL_foot', 
    'FR_hip', 
    'FR_thigh', 
    'FR_calf', 
    'FR_foot', 
    'Head_upper', 
    'Head_lower', 
    'RL_hip', 
    'RL_thigh', 
    'RL_calf', 
    'RL_foot', 
    'RR_hip', 
    'RR_thigh', 
    'RR_calf',
    'RR_foot']
    """
# Available Body strings: 
    reward_collision = RewTerm(
        func=rewards.reward_collision, 
        weight=-10., 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=["base",".*_calf",".*_thigh"]),
        },
    )
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge, 
        weight=-1.0, 
        params={
            "asset_cfg":SceneEntityCfg(name="robot", body_names=["FL_foot","FR_foot","RL_foot","RR_foot"]),
            "sensor_cfg":SceneEntityCfg(name="contact_forces", body_names=".*_foot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_torques = RewTerm(
        func=rewards.reward_torques, 
        weight=-0.00001, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_dof_error = RewTerm(
        func=rewards.reward_dof_error, 
        weight=-0.04, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_hip_pos = RewTerm(
        func=rewards.reward_hip_pos, 
        weight=-0.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy, 
        weight=-0.05, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate, 
        weight=-0.1, 
        params={
          "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc, 
        weight=-2.5e-7, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z, 
        weight=-1.0, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation, 
        weight=-1.0, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour',
        },
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble, 
        weight=-1.0, 
        params={
            "sensor_cfg":SceneEntityCfg("contact_forces", body_names=".*_foot"),
        },
    )
    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel, 
        weight=1.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour'
        },
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw, 
        weight=0.5, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
            "parkour_name":'base_parkour'
        },
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques, 
        weight=-1.0e-7, 
        params={
            "asset_cfg":SceneEntityCfg("robot"),
        },
    )

@configclass
class PIERewardsCfg:
    """PIE Table I reward terms.

    Collision is implemented as a positive non-foot contact count with a
    negative weight. This keeps the effective reward contribution penalizing
    non-foot collisions.
    """

    reward_lin_vel_xy_command_tracking = RewTerm(
        func=rewards.reward_lin_vel_xy_command_tracking,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )
    reward_ang_vel_yaw_command_tracking = RewTerm(
        func=rewards.reward_ang_vel_yaw_command_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z_paper,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_joint_power = RewTerm(
        func=rewards.reward_joint_power,
        weight=-2.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
        },
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate_squared,
        weight=-0.01,
        params={
            "action_name": "joint_pos",
        },
    )
    reward_action_smoothness = RewTerm(
        func=rewards.reward_action_smoothness,
        weight=-0.01,
        params={
            "action_name": "joint_pos",
        },
    )


@configclass
class PIERegularizedRewardsCfg(PIERewardsCfg):
    """Ablation C reward set with stronger posture/contact/action regularization."""

    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
        },
    )
    reward_action_magnitude = RewTerm(
        func=rewards.reward_action_magnitude,
        weight=-0.002,
        params={
            "action_name": "joint_pos",
        },
    )
    reward_feet_contact_balance = RewTerm(
        func=rewards.reward_feet_contact_balance,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ema_alpha": 0.02,
        },
    )
    reward_feet_vertical_force_balance = RewTerm(
        func=rewards.reward_feet_vertical_force_balance,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ema_alpha": 0.02,
            "min_total_force": 20.0,
        },
    )


@configclass
class PIEFailureTerminalPenaltyRewardsCfg(PIERewardsCfg):
    """PIE reward ablation with a one-shot penalty on failure terminations."""

    reward_failure_terminal_penalty = RewTerm(
        func=rewards.reward_failure_terminal_penalty,
        weight=-250.0,
        params={
            "term_names": ("bad_base_orientation", "low_base_height", "illegal_body_contact"),
        },
    )


@configclass
class PIEPostureTerminalPenaltyRewardsCfg(PIEFailureTerminalPenaltyRewardsCfg):
    """Failure-penalty reward set with stronger posture stabilization."""

    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
        },
    )
    reward_bad_orientation_terminal_penalty = RewTerm(
        func=rewards.reward_failure_terminal_penalty,
        weight=-250.0,
        params={
            "term_names": ("bad_base_orientation",),
        },
    )


@configclass
class PIEPostureHeightRewardsCfg(PIEPostureTerminalPenaltyRewardsCfg):
    """Posture-stability reward set with explicit base-height shaping."""

    reward_base_height_below_target = RewTerm(
        func=rewards.reward_base_height_below_target,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.30,
        },
    )
    reward_low_base_terminal_penalty = RewTerm(
        func=rewards.reward_failure_terminal_penalty,
        weight=-250.0,
        params={
            "term_names": ("low_base_height",),
        },
    )


@configclass
class PIEGentleLoadFixRewardsCfg(PIEFailureTerminalPenaltyRewardsCfg):
    """Forward-walking warmup reward with posture and slip regularization."""

    reward_lin_vel_xy_command_tracking = RewTerm(
        func=rewards.reward_lin_vel_xy_command_tracking,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )

    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z_paper,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_hip_default_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
        },
    )
    reward_joint_default_pos = RewTerm(
        func=rewards.reward_dof_error,
        weight=-0.03,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "action_name": "joint_pos",
        },
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc,
        weight=-5.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_delta_torques = RewTerm(
        func=rewards.reward_delta_torques,
        weight=-1.0e-7,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "horizontal_force_ratio": 4.0,
            "min_vertical_force": 1.0,
        },
    )
    reward_feet_slip = RewTerm(
        func=rewards.reward_feet_slip,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "horizontal_force_ratio": 2.0,
            "min_vertical_force": 5.0,
            "contact_force_threshold": 2.0,
        },
    )

    reward_feet_min_force_share = RewTerm(
        func=rewards.reward_feet_min_force_share,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ema_alpha": 0.02,
            "min_share": 0.15,
            "min_total_force": 20.0,
        },
    )

    # Legged-gym / DreamWaQ style "long step" reward. Isaac Lab's implementation
    # is unclipped so large air_time gives unbounded positive reward which
    # incentivises pronk. Lower the threshold to 0.2 s so short steps already
    # count, and pair it with reward_no_fly below to forbid all-four-in-air
    # frames.
    reward_feet_air_time = RewTerm(
        func=isaac_mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.2,
        },
    )

    # Directly penalise every frame where all four feet are in the air. This
    # kills pronk / flying trot while leaving normal trot (where at least one
    # diagonal pair is in contact) untouched. Does NOT reward any particular
    # gait shape, so it does not produce the bouncing failure mode that
    # feet_air_time did.
    reward_no_fly = RewTerm(
        func=rewards.reward_no_fly,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "contact_threshold": 1.0,
        },
    )

    # Force the policy to return to the default standing pose when the
    # commanded velocity is near zero. Without this the policy leans forward
    # at rest because the base force distribution (battery in front) is only
    # 55%/45% front/back, and nothing else penalises that static tilt.
    reward_stand_still = RewTerm(
        func=rewards.reward_stand_still,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "command_threshold": 0.1,
        },
    )

    # Encourage diagonal-trot symmetry so the policy cannot settle into pronk
    # (all four legs synchronised) or other non-trot gaits. Compared against the
    # default joint offset so front/back thigh bias does not produce a constant
    # penalty; hip joints excluded so yaw commands still work.
    # Disabled for the action-scale ablation: the reward conflicts with parkour
    # traversal (it compresses the action space in a way that also hurts
    # terrain_levels). Re-enable only if a separate warmup stage requires it.
    # reward_foot_mirror_diagonal = RewTerm(
    #     func=rewards.reward_foot_mirror_diagonal,
    #     weight=-0.1,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #     },
    # )


@configclass
class StairsBeamRewardsCfg(PIEGentleLoadFixRewardsCfg):
    """Reward set tuned for the stairs + balance-beam parkour task.

    Relative to the walking baseline (PIEGentleLoadFixRewardsCfg):
      - weaken posture / contact / gait regularisers that fight normal stair
        climbing (feet must leave default pose, hit step risers, etc.)
      - keep velocity tracking strong
      - add reward_feet_clearance_stairs: reward lifting swing feet enough to
        clear the upcoming step
      - add reward_tracking_goal_vel: reward moving along the goal direction,
        so the policy is pushed toward the next waypoint even when the
        velocity command is only weakly aligned.
    """

    # --- weight overrides -------------------------------------------------
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.05,  # was -0.1
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z_paper,
        weight=-0.5,  # was -2.0, stairs require vertical velocity
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-1.0,  # was -2.0, slight pitch when climbing stairs is OK
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_default_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-1.0,  # keep hip tight, straight-line stairs don't need hip spread
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.01,  # was -0.05, allow more dynamic action changes
        params={"asset_cfg": SceneEntityCfg("robot"), "action_name": "joint_pos"},
    )
    reward_feet_stumble = RewTerm(
        func=rewards.reward_feet_stumble,
        weight=-0.3,  # was -1.0, feet hitting step risers is expected
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "horizontal_force_ratio": 4.0,
            "min_vertical_force": 1.0,
        },
    )
    reward_feet_min_force_share = RewTerm(
        func=rewards.reward_feet_min_force_share,
        weight=-0.5,  # was -5.0, on a narrow beam one foot may briefly dangle
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ema_alpha": 0.02,
            "min_share": 0.15,
            "min_total_force": 20.0,
        },
    )
    reward_no_fly = RewTerm(
        func=rewards.reward_no_fly,
        weight=-0.3,  # was -1.0, some aerial phase is unavoidable on tall steps
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "contact_threshold": 1.0,
        },
    )

    # --- new rewards for parkour ----------------------------------------
    # Reward lifting swing feet above the terrain. Target clearance 0.10 m
    # covers easy-difficulty stair steps (0.05 m) and lets the policy know it
    # must pick the foot up over the step riser.
    # Weight 0.3 (was 1.0): at 1.0 the policy exploits this by lifting feet
    # in place without advancing. 0.3 keeps the signal but makes it secondary
    # to velocity tracking.
    reward_feet_clearance_stairs = RewTerm(
        func=rewards.reward_feet_clearance_stairs,
        weight=0.3,
        params={
            "foot_sensor_names": (
                "foot_scanner_fl",
                "foot_scanner_fr",
                "foot_scanner_rl",
                "foot_scanner_rr",
            ),
            "contact_sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=".*_foot"
            ),
            "target_clearance": 0.10,
            "contact_threshold": 1.0,
        },
    )

    # No reward_tracking_goal_vel: the task uses pure velocity command (no
    # heading mode / goal waypoints). Forward progress is driven by
    # reward_lin_vel_xy_command_tracking (+4.0) which rewards matching the
    # commanded vx=0.5-1.0 m/s.

    # Direct reward for forward distance traveled in +x. Cannot be exploited
    # by standing still or lifting feet in place. Only positive progress
    # counts (going backwards gives 0).
    reward_forward_distance = RewTerm(
        func=rewards.reward_forward_distance,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class StairsOnlyRewardsCfg:
    """Minimal reward set for pure staircase climbing, inspired by IsaacLab
    official Go2 rough terrain config. Only tracking + physics penalties +
    forward distance + foot clearance. No gait shapers."""

    reward_lin_vel_xy_command_tracking = RewTerm(
        func=rewards.reward_lin_vel_xy_command_tracking,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )
    reward_ang_vel_yaw_command_tracking = RewTerm(
        func=rewards.reward_ang_vel_yaw_command_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )
    reward_forward_distance = RewTerm(
        func=rewards.reward_forward_distance,
        weight=2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_feet_clearance_stairs = RewTerm(
        func=rewards.reward_feet_clearance_stairs,
        weight=0.3,
        params={
            "foot_sensor_names": (
                "foot_scanner_fl", "foot_scanner_fr",
                "foot_scanner_rl", "foot_scanner_rr",
            ),
            "contact_sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "target_clearance": 0.10,
            "contact_threshold": 1.0,
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z_paper,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_default_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot"), "action_name": "joint_pos"},
    )
    reward_joint_power = RewTerm(
        func=rewards.reward_joint_power,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- Safety signals (added after iter 11100 showed policy stuck at
    # terrain_levels=0 with illegal_body_contact~0.46; no strong "falling is
    # catastrophic" gradient). -------------------------------------------
    # One-shot heavy penalty when the episode ends via illegal contact (the
    # only failure termination kept in StairsOnly). Matches walking baseline.
    reward_failure_terminal_penalty = RewTerm(
        func=rewards.reward_failure_terminal_penalty,
        weight=-250.0,
        params={
            "term_names": ("illegal_body_contact",),
        },
    )
    # Continuous per-frame penalty for non-foot contacts (base / hip / thigh /
    # calf / head). This gives the policy a gradient *before* it falls over,
    # not just at the terminal step.
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
        },
    )
    # Reward long swing phases (threshold=0.2 s so a normal trot step already
    # counts). Discourages shuffle / tiny-step gaits that the current policy
    # seems to settle into on stairs.
    reward_feet_air_time = RewTerm(
        func=isaac_mdp.feet_air_time,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.2,
        },
    )


@configclass
class GapOnlyRewardsCfg:
    """Reward set for gap-crossing corridor.

    Key design: goal-based tracking to push the robot toward the next
    platform, heavy failure penalty for falling into gaps, collision
    penalty for non-foot contacts, feet_edge to avoid landing on rims.
    """

    reward_tracking_goal_vel = RewTerm(
        func=rewards.reward_tracking_goal_vel,
        weight=1.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    reward_tracking_yaw = RewTerm(
        func=rewards.reward_tracking_yaw,
        weight=0.8,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
        },
    )
    # Velocity command tracking (base frame). Forces the policy to actually
    # reach the commanded forward speed, not just walk slowly toward goal.
    reward_lin_vel_xy_command_tracking = RewTerm(
        func=rewards.reward_lin_vel_xy_command_tracking,
        weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )
    # Yaw rate command tracking. ParkourCommand computes a corrective yaw rate
    # from heading error; this reward forces the policy to actually follow it,
    # so the robot self-corrects when it drifts off heading.
    reward_ang_vel_yaw_command_tracking = RewTerm(
        func=rewards.reward_ang_vel_yaw_command_tracking,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
        },
    )
    # One-shot bonus when the robot crosses a gap (advances cur_goal_idx).
    # Sparse but strong signal that "actually getting to the next platform"
    # is what matters, not just facing the goal direction.
    reward_goal_reached = RewTerm(
        func=rewards.reward_goal_reached,
        weight=10.0,
        params={
            "parkour_name": "base_parkour",
        },
    )
    reward_lin_vel_z = RewTerm(
        func=rewards.reward_lin_vel_z_paper,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_hip_default_pos = RewTerm(
        func=rewards.reward_hip_pos,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")},
    )
    reward_dof_acc = RewTerm(
        func=rewards.reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    reward_action_rate = RewTerm(
        func=rewards.reward_action_rate,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot"), "action_name": "joint_pos"},
    )
    # Gap-aware joint-power penalty. Outside the padded gap zone the weight
    # is 4x stronger than the legacy -2e-5; inside the zone the function
    # internally rescales by 0.25 so the effective penalty stays at -2e-5
    # (matching the value v9 trained under, which DID solve gap crossing
    # but produced a "jump-everywhere" gait on flat ground). Net effect:
    # tighter energy budget on flat ground → encourages trot, while the
    # gap region keeps the latitude needed for the leap.
    reward_joint_power = RewTerm(
        func=rewards.reward_joint_power_gap_aware,
        weight=-8.0e-5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "parkour_name": "base_parkour",
            "in_zone_scale": 0.25,
        },
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
        },
    )
    # Penalise feet landing on terrain edges (gap rims). Forces the policy
    # to place feet squarely on platforms rather than teetering on the edge.
    reward_feet_edge = RewTerm(
        func=rewards.reward_feet_edge,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(name="robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            "sensor_cfg": SceneEntityCfg(name="contact_forces", body_names=".*_foot"),
            "parkour_name": "base_parkour",
        },
    )


@configclass
class PIEBridgeRewardsCfg(PIEPostureHeightRewardsCfg):
    """Bridge reward set that keeps posture safety while allowing larger motion."""

    reward_base_height_below_target = RewTerm(
        func=rewards.reward_base_height_below_target,
        weight=-20.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.30,
        },
    )


@configclass
class PIEBridgeLoadFixRewardsCfg(PIEBridgeRewardsCfg):
    """Bridge reward set that keeps a minimum long-term load share for every foot."""

    reward_feet_min_force_share = RewTerm(
        func=rewards.reward_feet_min_force_share,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ema_alpha": 0.02,
            "min_share": 0.14,
            "min_total_force": 20.0,
        },
    )


@configclass
class PIEBridgeGaitRewardsCfg(PIEBridgeRewardsCfg):
    """Bridge reward set with light four-foot usage regularization."""

    reward_feet_contact_balance = RewTerm(
        func=rewards.reward_feet_contact_balance,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ema_alpha": 0.02,
        },
    )
    reward_feet_vertical_force_balance = RewTerm(
        func=rewards.reward_feet_vertical_force_balance,
        weight=-0.4,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ema_alpha": 0.02,
            "min_total_force": 20.0,
        },
    )


@configclass
class PIEPostureHeightWarmupRewardsCfg(PIEPostureHeightRewardsCfg):
    """Early curriculum reward set that penalizes non-foot contact without terminating it."""

    reward_ang_vel_xy = RewTerm(
        func=rewards.reward_ang_vel_xy,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_orientation = RewTerm(
        func=rewards.reward_orientation_paper,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reward_collision = RewTerm(
        func=rewards.reward_collision,
        weight=-40.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
        },
    )
    reward_nonfoot_contact_force = RewTerm(
        func=rewards.reward_contact_force_above_threshold,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
            "threshold": 5.0,
        },
    )
    reward_failure_terminal_penalty = None
    reward_bad_orientation_terminal_penalty = None
    reward_base_height_below_target = RewTerm(
        func=rewards.reward_base_height_below_target,
        weight=-80.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "target_height": 0.32,
        },
    )
    reward_low_base_terminal_penalty = None


@configclass
class TerminationsCfg:
    """Legacy termination terms shared by teacher/student and baseline PIE."""

    total_terminates = DoneTerm(
        func=terminations.terminate_episode,
        time_out=True,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class PIETerminationsCfg:
    """Ablation B/C split termination terms for easier failure attribution."""

    episode_time_out = DoneTerm(
        func=terminations.time_out,
        time_out=True,
        params={},
    )
    goal_reached = DoneTerm(
        func=terminations.goal_reached,
        time_out=False,
        params={
            "parkour_name": "base_parkour",
        },
    )
    bad_base_orientation = DoneTerm(
        func=terminations.bad_base_orientation,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_roll": 0.9,
            "max_pitch": 0.9,
        },
    )
    low_base_height = DoneTerm(
        func=terminations.base_height_below_terrain,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
            "minimum_height": 0.18,
        },
    )
    illegal_body_contact = DoneTerm(
        func=terminations.illegal_body_contact,
        time_out=False,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["base", ".*_hip", ".*_thigh", ".*_calf", "Head_upper", "Head_lower"],
            ),
            "threshold": 5.0,
        },
    )


@configclass
class PIEWarmupTerminationsCfg(PIETerminationsCfg):
    """Warmup terminations: learn stability from dense penalties instead of early resets."""

    bad_base_orientation = None
    low_base_height = None
    illegal_body_contact = None
    
@configclass
class EventCfg:
    ### Modified origin events, plz see relative issue https://github.com/isaac-sim/IsaacLab/issues/1955
    """Configuration for events."""
    reset_root_state = EventTerm(
        func= events.reset_root_state,
        params = {'offset': 3.},
        mode="reset",
    )
    reset_robot_joints = EventTerm(
        func= reset_joints_by_scale, 
        params={
            "position_range": (0.95, 1.05),
            "velocity_range": (0.0, 0.0),
        },
        mode="reset",
    )
    physics_material = EventTerm( # Okay
        func=events.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "friction_range": (0.6, 2.0),
            "num_buckets": 64,
        },
    )

    ## we don't use this event, If you use this, you will get a bad result
    # randomize_actuator_gains = EventTerm(
    #     func= events.randomize_actuator_gains,
    #     params={
    #         "asset_cfg" :SceneEntityCfg("robot", joint_names=".*"),
    #         "stiffness_distribution_params": (0.975, 1.025),  
    #         "damping_distribution_params": (0.975, 1.025),
    #         "operation": "scale",
    #         },
    #     mode="startup",
    # )
    randomize_rigid_body_mass = EventTerm(
        func= randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1., 3.0),
            "operation": "add",
            },
    )
    randomize_rigid_body_com = EventTerm(
        func= events.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {'x':(-0.02, 0.02),'y':(-0.02, 0.02),'z':(-0.02, 0.02)}
            },
    )
    random_camera_position = EventTerm(
        func= events.random_camera_position,
        mode="startup",
        params={'sensor_cfg':SceneEntityCfg("depth_camera"),
                'rot_noise_range': {'pitch':(-5, 5)},
                'convention':'ros',
                },
    )
    push_by_setting_velocity = EventTerm( # Okay
        func = events.push_by_setting_velocity, 
        params={'velocity_range':{"x":(-0.5, 0.5), "y":(-0.5, 0.5)}},
        interval_range_s = (8. ,8. ),
        is_global_time= True, 
        mode="interval",
    )
    base_external_force_torque = EventTerm(  # Okay
        func=apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

@configclass
class ActionsCfg:
    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot", 
        joint_names=[".*"], 
        scale=0.25, 
        use_default_offset=True,
        action_delay_steps = [1, 1],
        delay_update_global_steps = 24 * 8000,
        history_length = 8,
        use_delay = True,
        clip = {'.*': (-4.8,4.8)}
        )


@configclass
class PIEActionsCfg:
    """PIE action directly represents the joint-position offset in radians."""

    joint_pos = DelayedJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
        action_delay_steps=[1, 1],
        delay_update_global_steps=24 * 8000,
        history_length=8,
        use_delay=True,
        clip={".*": (-1.2, 1.2)},
    )
