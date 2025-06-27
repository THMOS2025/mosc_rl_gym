python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t26 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:5" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.cycle_time=0.5

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t27 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:5" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.cycle_time=0.6

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t28 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:5" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.cycle_time=0.7

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t29 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:5" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.cycle_time=0.8

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t30 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:5" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.cycle_time=0.9

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t31 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:5" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.cycle_time=1.0