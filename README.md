# THMOS-RL
## install

### conda env create

```bash
conda create -n mevita python=3.8
conda activate mevita
```

```bash
pip install torch torchvision torchaudio
conda install numpy=1.23
pip install hydra-core
```


### rl pkg install
Isaac Gym Preview 4 && humanoid_gym

```bash
pip install -e ./isaacgym/python/
pip install -e ./humanoid-gym
```


### train and play

train

```shell
python humanoid-gym/humanoid/scripts/train.py --task=humanoid_ppo --run_name v1 --headless
```


play

```shell
python humanoid-gym/humanoid/scripts/play.py --task=humanoid_ppo --run_name v1
```

sim2sim

```shell
python humanoid-gym/humanoid/scripts/sim2sim.py --run_name v1 
```

```bash
python humanoid-gym/humanoid/scripts/train.py runner.run_name=v3_s5 runner.max_iterations=2000 headless=true rl_device="cuda:0"
python humanoid-gym/humanoid/scripts/train.py runner.run_name=v3_t8 runner.max_iterations=2000 headless=false rl_device="cuda:0" runner.load_run="Jun27_22-03-51_v3_t8" runner.checkpoint=-1
python humanoid-gym/humanoid/scripts/play.py rl_device="cuda:0" headless=false runner.resume=true runner.load_run=-1 runner.checkpoint=-1 env.num_envs=20 runner.run_name=v3_t7 
```