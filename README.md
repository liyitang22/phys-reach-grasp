<h1 align="center"> Learning Physics-Based Full-Body Human Reaching and Grasping from Brief Walking References </h1>

<div align="center">

[[Website]](https://liyitang22.github.io/phys-reach-grasp/)
[[Arxiv]](https://arxiv.org/abs/2503.07481)
[[Video]](https://www.youtube.com/watch?v=eJ2G_tpUE8Y)

<img src="static/images/teaser.png" style="height:350px;" />




[![IsaacGym](https://img.shields.io/badge/IsaacGym-Preview4-b.svg)](https://developer.nvidia.com/isaac-gym) [![Linux platform](https://img.shields.io/badge/Platform-linux--64-orange.svg)](https://ubuntu.com/blog/tag/22-04-lts) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)]()

</div>

## TODO
- [x] Release low-level policy training code
- [ ] Release high-level policy training code
- [ ] Release data generation code
- [ ] Release low-level policy tuning code

## IsaacGym Conda Env

```bash
conda create -n walkgrasp python=3.8
conda activate walkgrasp
```
### Install IsaacGym

Download [IsaacGym](https://developer.nvidia.com/isaac-gym/download) and extract:

```bash
wget https://developer.nvidia.com/isaac-gym-preview-4
tar -xvzf isaac-gym-preview-4
```

Install IsaacGym Python API:

```bash
pip install -e isaacgym/python
```

Test installation:

```bash
python 1080_balls_of_solitude.py  # or
python joint_monkey.py
```

For libpython error:

- Check conda path:
    ```bash
    conda info -e
    ```
- Set LD_LIBRARY_PATH:
    ```bash
    export LD_LIBRARY_PATH=</path/to/conda/envs/your_env/lib>:$LD_LIBRARY_PATH
    ```


## Installation 
Download Isaac Gym from the website, then follow the installation instructions.

Once Isaac Gym is installed, install the external dependencies for this repo:
```
pip install -r requirements.txt
```

## Train Low-level Policy

First, an initial low-level policy can be trained to imitate a dataset of walking clips using the following command:
```
python ase/run.py --task FullbodyAMPGetup --cfg_env ase/data/cfg/fullbody_ase_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/motion_cfg/walk.yaml --max_iterations 20000  --headless
```
`--motion_file` can be used to specify a dataset of motion clips that the model should imitate. 
The task `FullbodyAMPGetup` will train a model to imitate a dataset of motion clips and get up after falling.
`--headless` is used to disable visualizations. If you want to view the
simulation, simply remove this flag. To test a trained model, use the following command:
```
python ase/run.py --test --task FullbodyAMPGetup --cfg_env ase/data/cfg/fullbody_ase_getup.yaml --cfg_train ase/data/cfg/train/rlg/ase_humanoid.yaml --motion_file ase/data/motions/motion_cfg/walk.yaml --checkpoint [path_to_ase_checkpoint]
```