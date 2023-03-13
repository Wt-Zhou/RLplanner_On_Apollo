# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
np.set_printoptions(suppress=True)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import os
import time
import argparse
import random
import math
from torch_geometric.data import DataLoader, DataListLoader, Data
from tqdm import tqdm
from Agent.world_model.self_attention.modeling.predmlp import TrajPredMLP
from Agent.world_model.self_attention.modeling.selfatten import SelfAttentionLayer
from Agent.world_model.transition_model import make_transition_model
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.actions import LaneAction


class GNN_World_Model(object):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        state_space_dim,
        env,
        transition_model_lr=0.001,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        decoder_lr=0.001,
        decoder_weight_lambda=0.0,
    ):
        self.device = device
        self.discount = discount
        self.state_space_dim = state_space_dim
        self.action_shape = action_shape

        # Planner
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 
        
        # Ego vehicle model
        self.ego_transition_model = make_transition_model(
            transition_model_type='probabilistic', encoder_feature_dim=5, action_shape=action_shape).to(device)

        # GNN transition model
        decay_lr_factor = 0.3
        decay_lr_every = 5
        self.env_transition_model = Attention_GNN(5, 5).to(device)
        self.env_trans_optimizer = optim.Adam(self.env_transition_model.parameters(), lr=transition_model_lr)
        self.trans_scheduler = optim.lr_scheduler.StepLR(self.env_trans_optimizer, step_size=decay_lr_every, gamma=decay_lr_factor)

        # Reward model
        self.reward_decoder = nn.Sequential(
        nn.Linear(state_space_dim + action_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)).to(device)

        self.reward_decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.ego_transition_model.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )
        self.env = env

        args = self.parse_args()
        self.replay_buffer = World_Buffer(obs_shape=env.observation_space.shape,
            action_shape=[1], # discrete, 1 dimension!
            capacity= args.replay_buffer_capacity,
            batch_size= args.batch_size,
            device=device)

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # FIXME:Should be the same with CP
        parser.add_argument("--decision_count", type=int, default=1, help="how many steps for a decision")

        # environment
        parser.add_argument('--domain_name', default='carla')
        parser.add_argument('--task_name', default='run')
        parser.add_argument('--image_size', default=84, type=int)
        parser.add_argument('--action_repeat', default=1, type=int)
        parser.add_argument('--frame_stack', default=1, type=int) #3
        parser.add_argument('--resource_files', type=str)
        parser.add_argument('--eval_resource_files', type=str)
        parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
        parser.add_argument('--total_frames', default=1000, type=int)
        # replay buffer
        parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
        # train
        parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
        parser.add_argument('--init_steps', default=1, type=int)
        parser.add_argument('--num_train_steps', default=1000, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
        parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
        parser.add_argument('--load_encoder', default=None, type=str)
        # eval
        parser.add_argument('--eval_freq', default=1000, type=int)  # TODO: master had 10000
        parser.add_argument('--num_eval_episodes', default=20, type=int)
        # critic
        parser.add_argument('--critic_lr', default=1e-3, type=float)
        parser.add_argument('--critic_beta', default=0.9, type=float)
        parser.add_argument('--critic_tau', default=0.005, type=float)
        parser.add_argument('--critic_target_update_freq', default=2, type=int)
        # actor
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--actor_beta', default=0.9, type=float)
        parser.add_argument('--actor_log_std_min', default=-10, type=float)
        parser.add_argument('--actor_log_std_max', default=2, type=float)
        parser.add_argument('--actor_update_freq', default=2, type=int)
        # encoder/decoder
        parser.add_argument('--encoder_type', default='pixelCarla098', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity'])
        parser.add_argument('--encoder_feature_dim', default=50, type=int)
        parser.add_argument('--encoder_lr', default=1e-3, type=float)
        parser.add_argument('--encoder_tau', default=0.005, type=float)
        parser.add_argument('--encoder_stride', default=1, type=int)
        parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
        parser.add_argument('--decoder_lr', default=1e-3, type=float)
        parser.add_argument('--decoder_update_freq', default=1, type=int)
        parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
        parser.add_argument('--num_layers', default=4, type=int)
        parser.add_argument('--num_filters', default=32, type=int)
        # sac
        parser.add_argument('--discount', default=0.99, type=float)
        parser.add_argument('--init_temperature', default=0.01, type=float)
        parser.add_argument('--alpha_lr', default=1e-3, type=float)
        parser.add_argument('--alpha_beta', default=0.9, type=float)
        # misc
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--work_dir', default='.', type=str)
        parser.add_argument('--save_tb', default=False, action='store_true')
        parser.add_argument('--save_model', default=True, action='store_true')
        parser.add_argument('--save_buffer', default=True, action='store_true')
        parser.add_argument('--save_video', default=False, action='store_true')
        parser.add_argument('--transition_model_type', default='probabilistic', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
        parser.add_argument('--render', default=False, action='store_true')
        parser.add_argument('--port', default=2000, type=int)
        args = parser.parse_args()
        return args

    def update(self, replay_buffer, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        obs_with_action = torch.cat([obs, action], dim=1)
        
        # Reward loss
        pred_next_reward = self.reward_decoder(obs_with_action)
        reward_loss = F.mse_loss(pred_next_reward, reward)

        # Transition loss
        self.env_transition_model.train()
        # ego_obs = torch.take(obs, torch.tensor([[0,1,2,3,4],[0,1,2,3,4]]).to(device=self.device))
        ego_obs = torch.take(obs, torch.tensor([[0,1,2,3,4]]).to(device=self.device))
        
        # print("debug",ego_obs, obs)
        ego_obs_with_action = torch.cat([ego_obs, action], dim=1)
        # next_ego_obs = torch.take(next_obs, torch.tensor([[0,1,2,3,4],[0,1,2,3,4]]).to(device=self.device))
        next_ego_obs = torch.take(next_obs, torch.tensor([[0,1,2,3,4]]).to(device=self.device))
        pred_next_latent_mu, pred_next_latent_sigma = self.ego_transition_model(ego_obs_with_action)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)
        diff = (pred_next_latent_mu - next_ego_obs.detach()) / pred_next_latent_sigma
        ego_trans_loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))    
        # print("ego_trans_loss",ego_trans_loss)
        # print("next_ego_obs",next_ego_obs)
        # print("pred_next_latent_mu",pred_next_latent_mu)
        
        # obs_with_action = torch.cat([obs, action, action, action, action, action], dim=1)
        # x = torch.reshape(obs_with_action, [6,5])
        x = torch.reshape(obs, [2,5])
        # edge_index = torch.tensor([[0, 1], [1, 0], [1,2], [2,1], [0,2], [2,0]], dtype=torch.long)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        valid_len = torch.tensor([[2], [2]], dtype=torch.float) # Zwt: Useless, Set to None in GNN
        obs_with_action = Data(x=x, edge_index=edge_index,  valid_len=valid_len).to(device=self.device)
        next_env_state = self.env_transition_model(obs_with_action)
        y = torch.reshape(next_obs, [2,5])
        env_trans_loss = F.mse_loss(y, next_env_state)

        # print("env_trans_loss",y,next_env_state)

        total_loss = ego_trans_loss + env_trans_loss + reward_loss
        # print("total_loss", ego_trans_loss, env_trans_loss, reward_loss)
        self.reward_decoder_optimizer.zero_grad()
        self.env_trans_optimizer.zero_grad()
        # total_loss.backward()
        ego_trans_loss.backward()
        env_trans_loss.backward()
        reward_loss.backward()
        self.reward_decoder_optimizer.step()
        self.env_trans_optimizer.step()
        # print("[World_Model] : Updated all models! Step:",step)

    def train_world_model(self, env, load_step, train_step):
        args = self.parse_args()
        self.make_dir(args.work_dir)
        model_dir = self.make_dir(os.path.join(args.work_dir, 'world_model'))

        # Collected data and train
        episode, episode_reward, done = 0, 0, True
        start_time = time.time()
        
        fig, ax = plt.subplots()
        try:
            self.load(model_dir, load_step)
            print("[World_Model] : Load learned model successful, step=",load_step)

        except:
            load_step = 0
            print("[World_Model] : No learned model, Creat new model")

        for step in range(train_step + 1):
            if done:
                if step > 0:
                    start_time = time.time()

                obs = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward = 0   
                random_policy = random.randint(0,1)
            
            # save agent periodically
            if step % args.eval_freq == 0:
                if args.save_model:
                    print("[World_Model] : Saved Model! Step:",step + load_step)
                    self.save(model_dir, step + load_step)
                # if args.save_buffer:
                #     self.replay_buffer.save(buffer_dir)
                #     print("[World_Model] : Saved Buffer!")

            # run training update
            if step >= args.init_steps:
                num_updates = args.init_steps if step == args.init_steps else 10
                for _ in range(num_updates):
                    self.update(self.replay_buffer, step) # Updated Transition and Reward Module


            obs = np.array(obs)
            curr_reward = reward
            
            # Draw Plot
            ax.cla() 

            angle = obs.tolist()[4]/math.pi*180
            rect = plt.Rectangle((obs.tolist()[0],-obs.tolist()[1]),2.5,6,angle=-angle+90, facecolor="red")
            ax.add_patch(rect)

            angle2 = obs.tolist()[9]/math.pi*180 

            # rect = plt.Rectangle((obs.tolist()[5],-obs.tolist()[6]),2,5,angle=obs.tolist()[9]/math.pi*180)
            rect = plt.Rectangle((obs.tolist()[5],-obs.tolist()[6]),2.5,6,angle=-angle2+90)
            ax.add_patch(rect)
            # ax.axis([200, 315,-145,-30])# Town03 highway
            ax.axis([-92,-13,-199,-137])# Town03 cut in
            ax.legend()
            plt.pause(0.001)
            # print("Obs",obs)

            # Rule-based Planner
            self.dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, action = self.trajectory_planner.trajectory_update(self.dynamic_map)
            # if random_policy == 1 and obs[1] < 85: # For car follow case
            #     action = 0
            # if random_policy == 1: # For cut in case
            #     action = 4
            # else:
            #     action = 0

            action = np.array(random.randint(0,6)) #FIXME:Action space
            print("Action",action)
            # Control
            trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory)
            for i in range(args.decision_count):
                control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                output_action = [control_action.acc, control_action.steering]
                new_obs, reward, done, info = env.step(output_action)
                if done:
                    break
                self.dynamic_map.update_map_from_obs(new_obs, env)

            print("Predicted Reward:",self.get_reward_prediction(obs, action))
            print("Actual Reward:",reward, step)
            print("Predicted State:",self.get_trans_prediction(obs, action)* (env.observation_space.high - env.observation_space.low) + env.observation_space.low)
            print("Actual State:",new_obs)
            episode_reward += reward
            normal_new_obs = (new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            normal_obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            self.replay_buffer.add(normal_obs, action, curr_reward, reward, normal_new_obs, done)

            obs = new_obs
            episode_step += 1

    def get_reward_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)

        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1)
            return self.reward_decoder(obs_with_action)

    def get_trans_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)
        predict_state = obs.copy()

        ego_obs = obs[0:5]
        np_ego_obs = np.empty((1, 5), dtype=np.float32)
        np.copyto(np_ego_obs[0], ego_obs)   
        ego_obs = torch.as_tensor(np_ego_obs, device=self.device).float()

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()

        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)

        with torch.no_grad():
            ego_obs_with_action = torch.cat([ego_obs, action], dim=1)
            next_ego_state = self.ego_transition_model(ego_obs_with_action)[0] # Return both mean and variance
            
            x = torch.reshape(obs, [2,5])
            edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
            valid_len = torch.tensor([[2], [2]], dtype=torch.float) # Zwt: Useless, Set to None in GNN
            obs_with_action = Data(x=x, edge_index=edge_index,  valid_len=valid_len).to(device=self.device)
            next_env_state = self.env_transition_model(obs_with_action)
        next_ego_state = next_ego_state[0].cpu().numpy()
        next_env_state = next_env_state[1].cpu().numpy()

        predict_state[0] = next_ego_state[0]
        predict_state[1] = next_ego_state[1]
        predict_state[2] = next_ego_state[2]
        predict_state[3] = next_ego_state[3]
        predict_state[4] = next_ego_state[4]
        predict_state[5] = next_env_state[0]
        predict_state[6] = next_env_state[1]
        predict_state[7] = next_env_state[2]
        predict_state[8] = next_env_state[3]
        predict_state[9] = next_env_state[4]

        return predict_state
            
    def save(self, model_dir, step):
        self.replay_buffer.save(os.path.join(model_dir, 'world_buffer'))

        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.env_transition_model.state_dict(),
            '%s/env_transition_model%s.pt' % (model_dir, step)
        )
        torch.save(
            self.ego_transition_model.state_dict(),
            '%s/ego_transition_model%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):
        # self.replay_buffer.load(os.path.join(model_dir, 'world_buffer'))

        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        )
        self.env_transition_model.load_state_dict(
            torch.load('%s/env_transition_model%s.pt' % (model_dir, step))
        )
        self.ego_transition_model.load_state_dict(
            torch.load('%s/ego_transition_model%s.pt' % (model_dir, step))
        )

    def reset(self):    
        self.s_0 = obs_e # Start State
        print("Start State",self.s_0)
        self.current_s = self.s_0
        self.simulation_step = 0

        if not self.world_model:
            self.load_world_model()

        return self.s_0

    def step(self, action):

        # Step reward
        reward = self.get_reward_prediction(self.current_s, action)

        # State
        self.current_s = self.get_trans_prediction(self.current_s, action)
        
        
        p1 = np.array([self.current_s[0] , self.current_s[1]]) # ego
        p2 = np.array([self.current_s[5] , self.current_s[6]]) # the env vehicle
        p3 = p2 - p1
        p4 = math.hypot(p3[0],p3[1])


        # If finish
        done = False
        print("p4",p4)
        if p4 < 5:# Collision check
            print("Imagine: Collision")
            done = True
        
        elif (self.current_s[0] + 73) ** 2 + (self.current_s[1] - 167) ** 2 < 100:
            print("Imagine: Pass")
            done = True

        elif self.current_s[0] > -20 or self.current_s[0] < -82 or self.current_s[1] > 200 or self.current_s[1] < 160:
            print("Imagine: Out of Area")
            done = True

        self.simulation_step += 1
        return self.current_s, reward, done, None

    def make_dir(self, dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass
        return dir_path

    def try_inference(self, env, load_step):
        args = self.parse_args()
        self.make_dir(args.work_dir)
        model_dir = self.make_dir(os.path.join(args.work_dir, 'world_model'))
       
        self.load(model_dir, load_step)
        print("[World_Model] : Load learned model successful, step=",load_step)
        done = True
        
        fig, ax = plt.subplots()

        for step in range(args.num_train_steps + 1):
            if done:
                if step > 0:
                    start_time = time.time()


                obs = [-32.99572754, 194.    ,     -15.08877638 , -0.00000804 , -3.14159212, -51.  ,       187.80000305 ,  0.   ,        0.    ,       2.61799335] # start state of town03 cut in old 
                done = False
                stay_too_long = 0

                random_policy = random.randint(0,1)
            
            obs = np.array(obs)
            print("obs",obs)

            # Draw Plot
            ax.cla() 

            angle = obs.tolist()[4]/math.pi*180
            rect = plt.Rectangle((obs.tolist()[0],-obs.tolist()[1]),2.5,6,angle=angle+90, facecolor="red")
            ax.add_patch(rect)

            angle2 = obs.tolist()[9]/math.pi*180 

            # rect = plt.Rectangle((obs.tolist()[5],-obs.tolist()[6]),2,5,angle=obs.tolist()[9]/math.pi*180)
            rect = plt.Rectangle((obs.tolist()[5],-obs.tolist()[6]),2.5,6,angle=angle2+90)
            ax.add_patch(rect)
            # ax.axis([200, 315,-145,-30])# Town03 highway
            ax.axis([-92,-13,-199,-137])# Town03 cut in
            ax.legend()
            plt.pause(0.001)

            # Rule-based Planner
            self.dynamic_map.update_map_from_obs(obs, env)
            rule_trajectory, action = self.trajectory_planner.trajectory_update(self.dynamic_map)
            # if random_policy == 1 and obs[1] < 80:
            #   action = 0 
            # if random_policy == 1: # For cut in case
            #     action = 4
            # else:
            #     action = 0
            action = np.array(random.randint(0,6)) #FIXME:Action space
            # print("Action",action)
            # Control
            trajectory = self.trajectory_planner.trajectory_update_CP(action, rule_trajectory)
            for i in range(args.decision_count):
                control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                output_action = [control_action.acc, control_action.steering]
                new_obs = self.get_trans_prediction(obs, action)
                if obs[1] < 45 or stay_too_long > 50:
                    done = True
                if abs(obs[3]) < 0.2:
                    stay_too_long += 1 
                if done:
                    break
                new_obs_recover = new_obs *(env.observation_space.high - env.observation_space.low) + env.observation_space.low

                self.dynamic_map.update_map_from_obs(new_obs_recover, env)

            obs = new_obs_recover
            # new_obs_recover[6] += 0.75


class Attention_GNN(nn.Module):
    """
    Self_attention GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, global_graph_width=8, traj_pred_mlp_width=8):
        super(Attention_GNN, self).__init__()
        self.polyline_vec_shape = in_channels
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width, need_scale=False)
        self.traj_pred_mlp = TrajPredMLP(
            global_graph_width, out_channels, traj_pred_mlp_width)

    def forward(self, obs_with_action):
        """
        args: 
            data (Data): [x, y, cluster, edge_index, valid_len]

        """

        valid_lens = obs_with_action.valid_len 
        out = self.self_atten_layer(obs_with_action.x, valid_lens)
        # print("gnn_out",out.squeeze(0)[:, ].squeeze(1))
        pred = self.traj_pred_mlp(out.squeeze(0)[:, ].squeeze(1))
        # print("pred",pred)
        return pred


class World_Buffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end

