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


```python
# gym joints indexes ***
joint b_Lh's id is 0
joint Lh_Ll's id is 1
joint Ll_Ll1's id is 2
joint Ll1_Ll2's id is 3
joint Ll2_La's id is 4
joint La_Lf's id is 5
joint b_Rh's id is 6
joint Rh_Rl's id is 7
joint Rl_Rl1's id is 8
joint Rl1_Rl2's id is 9
joint Rl2_Ra's id is 10
joint Ra_Rf's id is 11

# mujoco joints indexes
['', '', '', '', 'b_Rh', 'Rh_Rl', 'Rl_Rl1', 'Rl1_Rl2', 'Rl2_Ra', 'Ra_Rf', 'b_Lh', 'Lh_Ll', 'Ll_Ll1', 'Ll1_Ll2', 'Ll2_La', 'La_Lf']

# model joints indexes (tau limit pd obs in mujoco.py)
['b_Lh', 'b_Rh',
'Lh_Ll', 'Rh_Rl',
'Ll_Ll1', 'Rl_Rl1', 
'Ll1_Ll2', 'Rl1_Rl2', 
'Ll2_La', 'Rl2_Ra', 
'La_Lf', 'Ra_Rf']
```