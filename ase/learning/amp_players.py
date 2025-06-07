# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import torch 

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState
import code
import os

import learning.common_player as common_player

class AMPPlayerContinuous(common_player.CommonPlayer):
    def __init__(self, config):
        self._normalize_amp_input = config.get('normalize_amp_input', True)
        self._disc_reward_scale = config['disc_reward_scale']
        super().__init__(config)
        self.is_save = False
        self.save_path = "/share1/mingxian/mms_dataset/rollout_amp"
        self.save_motions_max_num = 1000
        return

    def restore(self, fn):
        if (fn != 'Base'):
            super().restore(fn)
            if self._normalize_amp_input:
                checkpoint = torch_ext.load_checkpoint(fn)
                self._amp_input_mean_std.load_state_dict(checkpoint['amp_input_mean_std'])
        return
    
    def _build_net(self, config):
        super()._build_net(config)
        
        if self._normalize_amp_input:
            self._amp_input_mean_std = RunningMeanStd(config['amp_input_shape']).to(self.device)
            self._amp_input_mean_std.eval()  
        
        return
    
    def save_motions(self, rotations_stack, root_translations_stack, done_labels, done, done_locomotion, done_grasp, done_point_index, steps):
        # get skeleton tree and fps from example_motion.npy
        curr_motion = SkeletonMotion.from_file("example_motion.npy")
        self.skeleton_tree = curr_motion.skeleton_tree
        self.fps = curr_motion.fps
        
        if 'llc_checkpoint' in self.config:
            exp_dir = os.path.join(self.save_path, self.config["llc_checkpoint"].split("/")[1], self.config["full_experiment_name"])
        else:
            exp_dir = os.path.join(self.save_path, self.config["full_experiment_name"])
            
        # save motions at specific path
        exp_loco_dir = os.path.join(exp_dir, "locomotion")
        exp_grasp_dir = os.path.join(exp_dir, "grasp")
        exp_overall_dir = os.path.join(exp_dir, "overall")
        os.makedirs(exp_loco_dir, exist_ok=True)
        os.makedirs(exp_grasp_dir, exist_ok=True)
        os.makedirs(exp_overall_dir, exist_ok=True)
        
        for i in range(rotations_stack.shape[0]):
            rotations = rotations_stack[i]
            root_translations = root_translations_stack[i]
            skeleton_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree=self.skeleton_tree, r=rotations[0:done_point_index[i].item(), ...].detach().cpu(), t=root_translations[0:done_point_index[i].item(), ...].detach().cpu(), is_local=False)
            skeleton_motion = SkeletonMotion.from_skeleton_state(skeleton_state, self.fps)

            if done_locomotion[i].item() == 0:
                output_path = os.path.join(exp_loco_dir, f"{done+i}_{done_labels[i].item()}.npy")
                skeleton_motion.to_file(output_path)
            else:
                output_path = os.path.join(exp_overall_dir, f"{done+i}_{done_labels[i].item()}.npy")
                skeleton_motion.to_file(output_path)
                loco_skeleton_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree=self.skeleton_tree, r=rotations[0:done_locomotion[i].item(), ...].detach().cpu(), t=root_translations[0:done_locomotion[i].item(), ...].detach().cpu(), is_local=False)
                loco_skeleton_motion = SkeletonMotion.from_skeleton_state(loco_skeleton_state, self.fps)
                loco_skeleton_motion.to_file(os.path.join(exp_loco_dir, f"{done+i}_{done_labels[i].item()}.npy"))
                done_drasp_index = done_point_index[i].item() if (done_grasp[i].item()==0) else done_grasp[i].item()
                if done_drasp_index <= done_locomotion[i].item():
                    code.interact(local=dict(globals(), **locals()))
                else:
                    grasp_skeleton_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree=self.skeleton_tree, r=rotations[done_locomotion[i].item():done_drasp_index, ...].detach().cpu(), t=root_translations[done_locomotion[i].item():done_drasp_index, ...].detach().cpu(), is_local=False)
                    grasp_skeleton_motion = SkeletonMotion.from_skeleton_state(grasp_skeleton_state, self.fps)
                    grasp_skeleton_motion.to_file(os.path.join(exp_grasp_dir, f"{done+i}_{done_labels[i].item()}.npy"))
    
                
    def _post_step(self, info):
        super()._post_step(info)
        if (self.env.task.viewer):
            self._amp_debug(info)
        return

    def _build_net_config(self):
        config = super()._build_net_config()
        if (hasattr(self, 'env')):
            config['amp_input_shape'] = self.env.amp_observation_space.shape
        else:
            config['amp_input_shape'] = self.env_info['amp_observation_space']
        return config

    def _amp_debug(self, info):
        with torch.no_grad():
            amp_obs = info['amp_obs']
            amp_obs = amp_obs[0:1]
            disc_pred = self._eval_disc(amp_obs)
            amp_rewards = self._calc_amp_rewards(amp_obs)
            disc_reward = amp_rewards['disc_rewards']

            disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
            disc_reward = disc_reward.cpu().numpy()[0, 0]
            print("disc_pred: ", disc_pred, disc_reward)

        return

    def _preproc_amp_obs(self, amp_obs):
        if self._normalize_amp_input:
            amp_obs = self._amp_input_mean_std(amp_obs)
        return amp_obs

    def _eval_disc(self, amp_obs):
        proc_amp_obs = self._preproc_amp_obs(amp_obs)
        return self.model.a2c_network.eval_disc(proc_amp_obs)

    def _calc_amp_rewards(self, amp_obs):
        disc_r = self._calc_disc_rewards(amp_obs)
        output = {
            'disc_rewards': disc_r
        }
        return output

    def _calc_disc_rewards(self, amp_obs):
        with torch.no_grad():
            disc_logits = self._eval_disc(amp_obs)
            prob = 1 / (1 + torch.exp(-disc_logits)) 
            disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.device)))
            disc_r *= self._disc_reward_scale
        return disc_r
