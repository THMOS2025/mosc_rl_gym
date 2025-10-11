# THMOS-RL-DRIBBLING
## install

### conda env create

```shell
conda create -n MOSC_RL_GYM python=3.8
conda activate MOSC_RL_GYM
```
use conda install torch:
```shell
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```
pip:
```
pip install torch==1.13.1+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
or use the whl in https://THMOS.quickconnect.cn/d/f/xvTLJtM3xg7cpiLfs7uasLdebDBgw4vR

`conda install numpy=1.23`

### rl pkg install

Isaac Gym Preview 4

- `cd isaacgym/python/ && pip install -e .`

humanoid_gym 
+

- `cd humanoid-gym && pip install -e .`

pip install hydra-core

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