python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t20 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:6" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.feet_clearance=1.0

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t21 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:6" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.feet_clearance=0.0

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t22 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:6" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.feet_clearance=0.0 \
  rewards.scales.foot_slip=0.0

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t23 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:6" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.feet_clearance=0.0 \
  rewards.scales.foot_slip=0.0 \
  rewards.scales.feet_contact_forces=0.0

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t24 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:6" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.feet_clearance=0.0 \
  rewards.scales.foot_slip=0.0 \
  rewards.scales.feet_contact_forces=0.0 \
  rewards.scales.feet_distance=0

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t25 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:6" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.feet_clearance=0.0 \
  rewards.scales.feet_distance=0