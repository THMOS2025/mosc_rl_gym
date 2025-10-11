# python humanoid-gym/humanoid/scripts/train.py \
#   runner.run_name=v3_t8 \
#   runner.max_iterations=2000 \
#   headless=true \
#   rl_device="cuda:0" \
#   rewards.scales.joint_pos=1.2 \
#   rewards.scales.termination=-150 \
#   rewards.scales.tracking_lin_vel=4.5 \
#   domain_rand.max_push_vel_xy=0.2 \
#   domain_rand.max_push_ang_vel=0.4

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t9 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=4.5 \
  domain_rand.max_push_vel_xy=0.1 \
  domain_rand.max_push_ang_vel=0.2

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t10 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=3.5 \
  domain_rand.max_push_vel_xy=0.2 \
  domain_rand.max_push_ang_vel=0.4

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t11 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=3.5 \
  domain_rand.max_push_vel_xy=0.1 \
  domain_rand.max_push_ang_vel=0.2

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t12 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=2.0 \
  commands.ranges.lin_vel_x="[-0.4,0.8]" \
  commands.ranges.lin_vel_y="[-0.3,0.3]" \
  commands.ranges.ang_vel_z="[-1.0,1.0]"

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v3_t13 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \
  rewards.scales.joint_pos=1.2 \
  rewards.scales.termination=-150 \
  rewards.scales.tracking_lin_vel=1.5 \
  commands.ranges.lin_vel_x="[-0.4,0.8]" \
  commands.ranges.lin_vel_y="[-0.3,0.3]" \
  commands.ranges.ang_vel_z="[-1.0,1.0]"
