import argparse
import os
import pickle
import shutil
from importlib import metadata

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from h12env import H1_2Env



def get_train_cfg(exp_name, max_iterations):
    train_cfg_dict = {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "seed": 1,
    }

    return train_cfg_dict


def get_cfgs():
    env_cfg = {
        "num_actions": 21,
        # joint/link names / tt changement done une erreur de log
        "default_joint_angles": {  # [rad]
            'left_hip_yaw_joint': 0,
            'left_hip_roll_joint': 0,
            'left_hip_pitch_joint': -0.16,
            'left_knee_joint': 0.36,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            'right_hip_yaw_joint': 0,
            'right_hip_roll_joint': 0,
            'right_hip_pitch_joint': -0.16,
            'right_knee_joint': 0.36,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            'torso_joint': 0,
            'left_shoulder_pitch_joint': 0.4,
            'left_shoulder_roll_joint': 0,
            'left_shoulder_yaw_joint': 0,
            'left_elbow_joint': 0.3,
            'right_shoulder_pitch_joint': 0.4,
            'right_shoulder_roll_joint': 0,
            'right_shoulder_yaw_joint': 0,
            'right_elbow_joint': 0.3,
       



            
        },
        "joint_names": [
            'left_hip_yaw_joint',
            'left_hip_roll_joint',
            'left_hip_pitch_joint',
            'left_knee_joint',
            'left_ankle_pitch_joint',
            'left_ankle_roll_joint',
            'right_hip_yaw_joint',
            'right_hip_roll_joint',
            'right_hip_pitch_joint',
            'right_knee_joint',
            'right_ankle_pitch_joint',
            'right_ankle_roll_joint',
            'torso_joint',
            'left_shoulder_pitch_joint',
            'left_shoulder_roll_joint',
            'left_shoulder_yaw_joint',
            'left_elbow_joint',
            'right_shoulder_pitch_joint',
            'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint',
            'right_elbow_joint',
	
        ],
        # PD
        "joint_kps":{
        	'left_hip_yaw_joint': 50,
        	'left_hip_roll_joint': 50.,
        	'left_hip_pitch_joint': 50.,
        	'left_knee_joint': 20,
        	'left_ankle_pitch_joint': 10.,
        	'left_ankle_roll_joint': 10.,

        	'right_hip_yaw_joint': 50.,
        	'right_hip_roll_joint': 50.,
        	'right_hip_pitch_joint': 50.,
        	'right_knee_joint': 20,
        	'right_ankle_pitch_joint': 10.,
        	'right_ankle_roll_joint': 10.,

       		'torso_joint': 100.,

        	'left_shoulder_pitch_joint': 100.,
        	'left_shoulder_roll_joint': 80.,
        	'left_shoulder_yaw_joint': 60.,
        	'left_elbow_joint': 120.,

        	'right_shoulder_pitch_joint': 100.,
        	'right_shoulder_roll_joint': 80.,
        	'right_shoulder_yaw_joint': 60.,
        	'right_elbow_joint': 120.,},
        
        "joint_kds": {
	    	'left_hip_yaw_joint': 2.5,
        	'left_hip_roll_joint': 2.5,
        	'left_hip_pitch_joint': 2.5,
        	'left_knee_joint': 4,
        	'left_ankle_pitch_joint': 2.0,
        	'left_ankle_roll_joint': 2.0,

        	'right_hip_yaw_joint': 2.5,
        	'right_hip_roll_joint': 2.5,
        	'right_hip_pitch_joint': 2.5,
        	'right_knee_joint': 4,
        	'right_ankle_pitch_joint': 2.0,
        	'right_ankle_roll_joint': 2.0,

        	'torso_joint': 10.,

        	'left_shoulder_pitch_joint': 2.0,
        	'left_shoulder_roll_joint': 1.5,
        	'left_shoulder_yaw_joint': 1.0,
        	'left_elbow_joint': 3.0,

        	'right_shoulder_pitch_joint': 2.0,
        	'right_shoulder_roll_joint': 1.5,
        	'right_shoulder_yaw_joint': 1.0,
        	'right_elbow_joint': 3.0,
   		 },
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 1.04],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.2,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "num_obs": 72,
        "obs_scales": {
            "lin_vel": 1.0,
            "ang_vel": -0.05,
            "dof_pos": 1.0,
            "dof_vel": -1e-3,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 1.04,
        "feet_height_target": 0.0,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": 3.0,
            "base_height": -1.0,
            "action_rate": -0.01,
            "similar_to_default": -0.0,
	    "forward_progress": 1.0,
            "alive": 2.0,
            "energy_penalty": -0.0001,
	    "no_jump": 2.0,
            "stable_height": 0.0,
            #"forward_walk": 2.0,
	    "tracking_velocity": 0.0,
	    "upright_orientation": 2.0,
	    "dof_vel":-0.0001,
	    "dof_acc":-0.0,
 	    "action_smoothness":0.0,
	    "hip_pos" :-1.0,

	    
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.2, 0.6],
        "lin_vel_y_range": [-0.0, 0.0],
        "ang_vel_range": [0.2 ,0.3],
    }

    return env_cfg, obs_cfg, reward_cfg, command_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="h1_2-walking")
    parser.add_argument("-B", "--num_envs", type=int, default=4096)
    parser.add_argument("--max_iterations", type=int, default=101)
    args = parser.parse_args()

    gs.init( logging_level="warning")

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp_name, args.max_iterations)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump(
        [env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg],
        open(f"{log_dir}/cfgs.pkl", "wb"),
    )

    env = H1_2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()

"""
# training
python h12_train.py
"""
