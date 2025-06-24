# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2024 Beijing RobotEra TECHNOLOGY CO.,LTD. All rights reserved.


from humanoid.envs.base.base_config import BaseConfig


class LeggedRobotCfg(BaseConfig):
    """
    Configuration class for the humanoid robot.
    """
    class env:
        # change the observation dim
        frame_stack = 5
        c_frame_stack = 3
        num_single_obs = 47
        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 113
        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 12
        num_envs = 4096
        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 24  # episode length in seconds
        use_ref_actions = True

    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85

    class asset:
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/MOSC0516/MOSC_UH_point_feet.urdf'  
        name = "MOSC"
        foot_name = "foot"
        # terminate_after_contacts_on = ['base_link','hip_yaw','hip_roll','thigh','calf']
        terminate_after_contacts_on = ['body']
        penalize_contacts_on = ["body"]
        collapse_fixed_joints = True # merge bodies connected by fixed joints. Specific fixed joints can be kept by adding " <... dont_collapse="true">
        self_collisions = 1  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        test_ref_dof = False
        fix_base_link = False
        disable_gravity = False
        default_dof_drive_mode = 3 # see GymDofDriveModeFlags (0 is none, 1 is pos tgt, 2 is vel tgt, 3 effort)
        density = 0.001
        angular_damping = 0.
        linear_damping = 0.
        max_angular_velocity = 1000.
        max_linear_velocity = 1000.
        armature = 0.
        thickness = 0.01

    class terrain:
        # rough terrain only:
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

        #mesh_type = 'plane'
        mesh_type = 'trimesh'
              
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 30  # number of terrain rows (levels)
        num_cols = 30  # number of terrain cols (types)
        horizontal_scale = 0.1 # [m]
        vertical_scale = 0.005 # [m]
        border_size = 25 # [m]
        selected = False
        max_init_terrain_level = 2  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.50, 0.00, 0.50, 0.0, 0.0, 0.0]
        slope_treshold = 0.75 # slopes above this threshold will be corrected to vertical surfaces
        terrain_kwargs = None # Dict of arguments for selected terrain
        
    class commands:
        num_commands = 3
        resampling_time = 4  # time before command are changed[s]
        # show command direction
        use_debug_lines = True
        class ranges:
            lin_vel_x = [-1.0, 1.5]   # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_z = [-1.0, 1.0]   # min max [rad/s]


    class noise:
        add_noise = True
        noise_level = 0.6    # scales other values

        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.8
            ang_vel = 0.2
            lin_vel = 0.1
            quat = 0.03

    class init_state:
        pos = [0.0, 0.0, 0.51]
        rot = [0.0, 0.0, 0.0, 1.0] # x,y,z,w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x,y,z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x,y,z [rad/s]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'b_Lh':0.0, 'b_Rh':0.0,
            'Lh_Ll':0.0, 'Rh_Rl':0.0,
            'Ll_Ll1':0.0, 'Rl_Rl1':0.0, 
            'Ll1_Ll2':0.0, 'Rl1_Rl2':0.0, 
            'Ll2_La':0.0, 'Rl2_Ra':0.0, 
            'La_Lf':0.0, 'Ra_Rf':0.0
        }

    class viewer:
        ref_env = 0
        pos = [10, 0, 6]  # [m]
        lookat = [0., 0., 0.]  # [m]
        debug_viz = False

    class control:
        # PD Drive parameters:
        stiffness = {'b_Lh':100, 'b_Rh':100,
                    'Lh_Ll':100, 'Rh_Rl':100,
                    'Ll_Ll1':100, 'Rl_Rl1':100, 
                    'Ll1_Ll2':100, 'Rl1_Rl2':100, 
                    'Ll2_La':50, 'Rl2_Ra':50, 
                    'La_Lf':25, 'Ra_Rf':25}
                     
        damping = {'b_Lh':2.0, 'b_Rh':2.0,
                    'Lh_Ll':2.0, 'Rh_Rl':2.0,
                    'Ll_Ll1':2.0, 'Rl_Rl1':2.0, 
                    'Ll1_Ll2':2.0, 'Rl1_Rl2':2.0, 
                    'Ll2_La':1.5, 'Rl2_Ra':1.5, 
                    'La_Lf':0.5, 'Ra_Rf':0.5}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25

        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10 # 50hz

    class sim:
        dt = 0.002  # 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
        gravity = [0., 0. ,-9.81]  # [m/s^2]

        class physx:
            num_threads = 20
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand:
        randomize_friction = True
        friction_range = [0.3, 0.8]
        restitution_range = [0.0, 1.0]
        rand_init_pos = [-0.01,0.01]
        rand_init_rot = [-0.01,0.01]

        randomize_base_mass = True
        add_mass = [-6., 2.]
        add_com_x = [-0.040,0.040]
        add_com_y = [-0.040,0.040]
        add_com_z = [-0.040,0.040]
        add_link_mass_rate = [0.5,1.5]
        
        action_randomization = 0.02
        action_rand_filter_rate = 0.05

        kp_rand_ratio = 0.2
        kd_rand_ratio = 0.1
        torque_rand_ratio = 0.0 # 98% -100% torque
        default_pos_rand = 0.02
        init_joint_pos_r = 0.05
        joint_friction = [0.0,0.0] # 0 -0.1
        joint_armature = [0.0,0.0]
        joint_max_delay = 10
        joint_min_delay = 5
                
        push_robots = True
        push_prop = 1
        push_interval_s = 4
        max_push_vel_xy = 0.3 * 1.2
        max_push_ang_vel = 0.6 * 1.2




    class rewards:
        #base_height_target = 0.33
        min_dist = 0.3
        max_dist = 0.6
        # put some settings here for LLM parameter tuning
        target_joint_pos_scale = 0.5        # rad
        target_feet_height = 0.05            # m
        ref_pos_dir = [-1, 1, -1, -1, 1, -1]
        cycle_time = 0.8                      # sec
        double_stand_phase = 0.5              # 
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        max_contact_force = 300  # forces above this value are penalized
        tracking_sigma = 0.5 # 0.25

        class scales:
            termination = -200
            
            # reference motion tracking
            # stage I
            joint_pos = 1.2 * 1.1
            feet_orientation = 1.
            feet_clearance = 2.
            # tracking_lin_vel = 10.0
            # tracking_ang_vel = 4.0
            tracking_lin_vel = 3.0 * 1.5
            tracking_ang_vel = 0.5 * 1.5
            #symmetry_act = 0

            # gait
            feet_distance = 0.2
            # feet_air_time = 0.1
            foot_slip = -0.05

            # contact
            feet_contact_forces = -0.01

            
            # base pos
            orientation = 1.0 * 1.2

            # energy
            action_smoothness = -1e-2
            torques = -1e-5
            dof_vel = -1e-5
            dof_acc = -1e-9


    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
        clip_observations = 18.
        clip_actions = 18.

class LeggedRobotCfgPPO(BaseConfig):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]
        
    class algorithm():
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.001
        learning_rate = 1e-3
        num_learning_epochs = 5
        schedule = 'adaptive' 
        gamma = 0.99
        lam = 0.95
        num_mini_batches = 4
        desired_kl = 0.01
        max_grad_norm = 1.

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 2000  # number of policy updates

        # logging
        save_interval = 400  # check for potential saves every this many iterations
        experiment_name = 'MOSC'
        run_name = ''
        
        # load and resume
        resume = True
        
        load_run = 'Jun20_18-06-04_v3' # -1 = last run
        checkpoint = 3600 # -1 = last saved model
        
        # load_run = 'Jun23_21-26-19_v3_s2' # -1 = last run
        # checkpoint = -1 # -1 = last saved model
        
        resume_path = None  # updated from load_run 
