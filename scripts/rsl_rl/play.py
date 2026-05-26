# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--show_depth",
    action="store_true",
    default=False,
    help="Pop up an OpenCV window with the live depth image grid (8-env tile).",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from scripts.rsl_rl.modules.on_policy_runner_with_extractor import OnPolicyRunnerWithExtractor

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from parkour_tasks.extreme_parkour_task.config.go2.agents.parkour_rl_cfg import ParkourRslRlOnPolicyRunnerCfg

from scripts.rsl_rl.exporter import (
export_teacher_policy_as_jit, 
export_teacher_policy_as_onnx,
export_deploy_policy_as_jit, 
export_deploy_policy_as_onnx,
)
from scripts.rsl_rl.vecenv_wrapper import ParkourRslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    agent_cfg: ParkourRslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    # Optionally turn on the depth-camera ObsTerm debug_vis (cv2.imshow grid).
    # Stays off by default because training runs headless on the server.
    if args_cli.show_depth:
        try:
            depth_obs = env_cfg.observations.depth_camera.depth
            depth_obs.params["debug_vis"] = True
            print("[INFO] --show_depth: enabled OpenCV depth grid window.")
        except AttributeError:
            print("[WARN] --show_depth requested but env_cfg has no observations.depth_camera.depth term.")
        # Snap the Isaac Sim viewer onto env 0's robot so the depth grid's
        # "env 0" tile matches what you're looking at in the simulator.
        try:
            from isaaclab.envs import ViewerCfg
            env_cfg.viewer = ViewerCfg(
                eye=(-0.0, 2.6, 1.6),
                lookat=(0.0, 0.0, 0.0),
                asset_name="robot",
                origin_type="asset_root",
                env_index=0,
            )
            print("[INFO] --show_depth: viewer locked to env 0 robot.")
        except Exception as exc:
            print(f"[WARN] viewer override failed: {exc}")

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = ParkourRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunnerWithExtractor(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)
    print(ppo_runner)
    # obtain the trained policy for inference

    use_pie_inference = (
        agent_cfg.algorithm.class_name != "DistillationWithExtractor"
        and getattr(ppo_runner.alg, "use_pie_actor_features", False)
    )
    estimator = None
    if not use_pie_inference:
        estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if agent_cfg.algorithm.class_name == "DistillationWithExtractor":
        policy = ppo_runner.get_inference_depth_policy(device=env.unwrapped.device)
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        policy_nn = ppo_runner.alg.depth_actor
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_deploy")
        export_deploy_policy_as_jit(policy_nn, 
                                    estimator,
                                    depth_encoder,
                                    ppo_runner.obs_normalizer, 
                                    path=export_model_dir, 
                                    filename="policy.pt")
        export_deploy_policy_as_onnx(
                            policy_nn, 
                            estimator,
                            depth_encoder,
                            agent_cfg,
                            normalizer=ppo_runner.obs_normalizer, 
                            path=export_model_dir, 
                            filename="policy.onnx"
                        )

    else:
        if use_pie_inference:
            policy = ppo_runner.get_pie_inference_policy(device=env.unwrapped.device)
            print("[INFO] Using PIE actor inference wrapper with obs_dict depth/proprioception features.")
        else:
            policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)
            policy_nn = ppo_runner.alg.policy
            export_model_dir = os.path.join(os.path.dirname(resume_path), "exported_teacher")
            export_teacher_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
            export_teacher_policy_as_onnx(
                policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
            )

    dt = env.unwrapped.step_dt
    if not use_pie_inference:
        estimator_paras = agent_cfg.to_dict()["estimator"]
        num_prop = estimator_paras["num_prop"]
        num_scan = estimator_paras["num_scan"]
        num_priv_explicit = estimator_paras["num_priv_explicit"]

    # --- Depth visualization setup ---
    show_depth = True
    try:
        import cv2
    except ImportError:
        print("[WARN] OpenCV not installed, depth visualization disabled.")
        show_depth = False

    # reset environment
    obs, extras = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        if agent_cfg.algorithm.class_name != "DistillationWithExtractor":
            with torch.inference_mode():
                # agent stepping
                if use_pie_inference:
                    actions = policy(extras["observations"], hist_encoding=True)
                else:
                    obs[:, num_prop+num_scan:num_prop+num_scan+num_priv_explicit] = estimator.inference(obs[:, :num_prop])
                    actions = policy(obs, hist_encoding = True)
            # env stepping
        else:
            depth_camera = extras["observations"]['depth_camera'].to(env.device)
            with torch.inference_mode():
                if env.unwrapped.common_step_counter %5 == 0:
                    obs_student = obs[:, :num_prop].clone()
                    obs_student[:, 6:8] = 0
                    depth_latent_and_yaw = depth_encoder(depth_camera, obs_student)
                    depth_latent = depth_latent_and_yaw[:, :-2]
                    yaw = depth_latent_and_yaw[:, -2:]
                obs[:, 6:8] = 1.5*yaw
                actions = policy(obs, hist_encoding=True, scandots_latent=depth_latent)
        obs, _, dones, extras = env.step(actions)
        if use_pie_inference:
            policy.reset(dones)

        # --- Show depth image from env 0 ---
        if show_depth and "depth_camera" in extras.get("observations", {}):
            try:
                depth_obs = extras["observations"]["depth_camera"]
                depth_flat = depth_obs[0].detach().cpu().float().reshape(-1).numpy()
                h, w = 58, 87
                if depth_flat.size >= h * w:
                    frame = depth_flat[-(h * w):].reshape(h, w)
                    frame_min, frame_max = frame.min(), frame.max()
                    if frame_max - frame_min > 1e-6:
                        frame_norm = ((frame - frame_min) / (frame_max - frame_min) * 255).astype("uint8")
                    else:
                        frame_norm = (frame * 0).astype("uint8")
                    frame_display = cv2.resize(frame_norm, (w * 4, h * 4), interpolation=cv2.INTER_NEAREST)
                    cv2.imshow("Depth Camera (env 0)", frame_display)
                    cv2.waitKey(1)
            except Exception as e:
                if timestep == 0:
                    print(f"[WARN] Depth visualization failed: {e}")
                    show_depth = False

        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
