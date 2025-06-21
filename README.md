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

- `cd humanoid-gym && pip install -e .`

### train and play

train

```shell
python humanoid-gym/humanoid/scripts/train.py --task=humanoid_ppo --run_name v1 --headless --num_envs 4096
```


play

```shell
python humanoid-gym/humanoid/scripts/play.py --task=humanoid_ppo --run_name v1
```

sim2sim

```shell
python humanoid-gym/humanoid/scripts/sim2sim.py --run_name v1 
```

