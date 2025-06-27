python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t14 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:7" \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.joint_pos=0

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t15 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:7" \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.joint_pos=0.2

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t16 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:7" \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.joint_pos=0.4

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t17 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:7" \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.joint_pos=0.6

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t18 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:7" \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.joint_pos=0.8

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t19 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:7" \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.0 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4 \
  rewards.scales.joint_pos=1.0
