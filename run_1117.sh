python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v5_t4 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v5_t4 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \
  resume=false \

python humanoid-gym/humanoid/scripts/train.py \
  runner.run_name=v5_t4 \
  runner.max_iterations=2000 \
  headless=true \
  rl_device="cuda:0" \
  resume=false \
  noise.noise_scales.quat=0.1
