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

from rl_games.algos_torch import players
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.player import BasePlayer

import numpy as np

class CommonPlayer(players.PpoPlayerContinuous):
    def __init__(self, config):
        BasePlayer.__init__(self, config)
        self.network = config['network']
        
        self._setup_action_space()
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.is_save = False
        self.is_task = False
        net_config = self._build_net_config()
        self._build_net(net_config)   
        
        return

    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_determenistic = self.is_determenistic
        sum_rewards = 0
        sum_steps = 0
        sum_fs_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        n_games = 1000
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True

        if has_masks_func:
            has_masks = self.env.has_action_mask()
        self.device='cpu'
        need_init_rnn = self.is_rnn
        for _ in range(n_games):
            if games_played >= n_games:
                break

            obs_dict = self.env_reset()
            
            batch_size = 1
            batch_size = self.get_batch_size(obs_dict['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            steps = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
            fs = torch.zeros([batch_size, 5000, 1], dtype=torch.int64)
            point_index = torch.zeros(batch_size, dtype=torch.int64)
            rotations = torch.zeros([batch_size, 500, 15, 4], dtype=torch.float32)
            root_translations = torch.zeros([batch_size, 500, 3], dtype=torch.float32)
            stages = torch.zeros(batch_size, dtype=torch.int64)
            stages_2 = torch.zeros(batch_size, dtype=torch.int64)

            print_game_res = False

            done_indices = []
            self.max_steps = 5000
            for n in range(self.max_steps):
                obs_dict = self.env_reset(done_indices)

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(obs_dict, masks, is_determenistic)
                else:
                    action = self.get_action(obs_dict, is_determenistic)
                obs_dict, r, done, info =  self.env_step(self.env, action)
                if hasattr(self.env.task, 'rigid_body_states'):
                    current_right_foot_xy_pos = self.env.task.rigid_body_states[:, 34, 0:2].detach().cpu()
                    current_left_foot_xy_pos = self.env.task.rigid_body_states[:, 37, 0:2].detach().cpu()
                    if n==0:
                        pre_right_foot_xy_pos = current_right_foot_xy_pos
                        pre_left_foot_xy_pos = current_left_foot_xy_pos
                    right_foot_speed = torch.norm((current_right_foot_xy_pos-pre_right_foot_xy_pos), dim=-1)
                    left_foot_speed = torch.norm((current_left_foot_xy_pos-pre_left_foot_xy_pos), dim=-1)
                    pre_right_foot_xy_pos = current_right_foot_xy_pos
                    pre_left_foot_xy_pos = current_left_foot_xy_pos

                    current_right_foot_z_pos = self.env.task.rigid_body_states[:, 34, 2].detach().cpu()
                    current_left_foot_z_pos = self.env.task.rigid_body_states[:, 37, 2].detach().cpu()
                    foot_pos_diff = torch.where(current_right_foot_z_pos<current_left_foot_z_pos, right_foot_speed, left_foot_speed)

                    fs[:, n, 0] = torch.where(foot_pos_diff>0.01, torch.zeros(batch_size, dtype=torch.int64)+1, torch.zeros(batch_size, dtype=torch.int64))
                    if self.is_save == True:
                        batch_indices = torch.arange(rotations.shape[0])
                        rotations[batch_indices, point_index.long()] = self.env.task.rigid_body_states[:, self.env.task.body_rb_ids, 3:7].detach().cpu()
                        root_translations[batch_indices, point_index.long()] = self.env.task.rigid_body_states[:, 0, 0:3].detach().cpu()
                    
                        point_index = point_index + 1
                        # code.interact(local=dict(globals(), **locals()))
                        stages = torch.where(self.env.task.envs_stages_buf.detach().cpu() == 0, stages+1, stages)
                        stages_2 = torch.where(self.env.task.envs_stages_buf.detach().cpu() != 4, stages_2+1, stages_2)
                cr += r
                steps += 1
  
                self._post_step(info)

                if render:
                    self.env.render(mode = 'human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0:
                    done_index_flat = done_indices.clone().detach().flatten()
                    done_labels = torch.zeros_like(done_index_flat)
                    if self.is_rnn:
                        for s in self.states:
                            s[:,all_done_indices,:] = s[:,all_done_indices,:] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps
                    if hasattr(self.env.task, 'rigid_body_states'):
                        sum_fs_steps += (fs[done_indices, :, 0] == 1).sum().item()
                    # code.interact(local=dict(globals(), **locals()))
                        fs = fs * (1.0 - done.float()).unsqueeze(1).unsqueeze(1).repeat(1, 5000, 1)
                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)
                        if self.is_save == True:
                            done_rotations = rotations[done_index_flat, :, ...]
                            done_root_translations = root_translations[done_index_flat, :, ...]
                            # code.interact(local=dict(globals(), **locals()))
                            done_locomotion = stages[done_index_flat]
                            done_grasp = stages_2[done_index_flat]
                            done_point_index = point_index[done_index_flat]
                        # code.interact(local=dict(globals(), **locals()))
                            if games_played < self.save_motions_max_num:
                                self.save_motions(done_rotations, done_root_translations, done_labels, games_played, done_locomotion, done_grasp, done_point_index, steps)
                            rotations[done_index_flat, :, ...] = 0
                            root_translations[done_index_flat, :, ...] = 0
                    point_index = point_index * (1 - done.long())
                    stages = stages* (1 - done.long())
                    stages_2 = stages_2* (1 - done.long())
                    games_played += done_count
                    if self.print_stats:
                        if print_game_res:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count, 'w:', game_res)
                        else:
                            print('reward:', cur_rewards/done_count, 'steps:', cur_steps/done_count)

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
                
                done_indices = done_indices[:, 0]

        print(sum_rewards)

        if self.is_task ==False:
            if hasattr(self.env.task, 'rigid_body_states'):
                print("Post-Reset average consecutive successes(grasp) = {:.5f}".format(
                self.env.task.total_grasp_successes / self.env.task.total_resets))
                print("Post-Reset average consecutive successes(goal) = {:.5f}".format(
                self.env.task.total_goal_successes / self.env.task.total_resets))
                print("Post-Reset average consecutive successes(grasp) list = {}".format(
                self.env.task.total_grasp_successes_list / self.env.task.total_resets_list))
                print("Post-Reset average consecutive successes(goal) list = {}".format(
                self.env.task.total_goal_successes_list / self.env.task.total_resets_list))
                print("fs", sum_fs_steps / sum_steps)
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps / games_played * n_game_life)

        return

    def obs_to_torch(self, obs):
        obs = super().obs_to_torch(obs)
        obs_dict = {
            'obs': obs
        }
        return obs_dict

    def get_action(self, obs_dict, is_determenistic = False):
        output = super().get_action(obs_dict['obs'], is_determenistic)
        return output

    def env_step(self, env, actions):
        if not self.is_tensor_obses:
            actions = actions.cpu().numpy()
        obs, rewards, dones, infos = env.step(actions)

        if hasattr(obs, 'dtype') and obs.dtype == np.float64:
            obs = np.float32(obs)
        if self.value_size > 1:
            rewards = rewards[0]
        if self.is_tensor_obses:
            return obs, rewards.to(self.device), dones.to(self.device), infos
        else:
            if np.isscalar(dones):
                rewards = np.expand_dims(np.asarray(rewards), 0)
                dones = np.expand_dims(np.asarray(dones), 0)
            return self.obs_to_torch(obs), torch.from_numpy(rewards), torch.from_numpy(dones), infos

    def _build_net(self, config):
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        if self.normalize_input:
            obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
            self.running_mean_std = RunningMeanStd(obs_shape).to(self.device)
            self.running_mean_std.eval() 
        return

    def env_reset(self, env_ids=None):
        obs = self.env.reset(env_ids)
        return self.obs_to_torch(obs)

    def _post_step(self, info):
        return

    def _build_net_config(self):
        obs_shape = torch_ext.shape_whc_to_cwh(self.obs_shape)
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents
        } 
        return config

    def _setup_action_space(self):
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        return