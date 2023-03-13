import copy
import glob
import math
import os
import os.path as osp
import random
import sys
import time
from math import atan2

import numba
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Agent.long_tail_planner.transition_model.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel
from Agent.long_tail_planner.transition_model.predmlp import (TrajPredGaussion,
                                                              TrajPredMLP)
from numba import jit
from numpy import clip, cos, sin, tan


@jit(nopython=True)
def _kinematic_model(vehicle_num, obs, future_frame, action_list, throttle_scale, steer_scale, dt):
    # vehicle model parameter
    wheelbase = 2.96
    max_steer = np.deg2rad(60)
    c_r = 0.01
    c_a = 0.05

    path_list = []

    for j in range(1, vehicle_num):
        x = obs[j][0]
        y = obs[j][1]
        velocity = math.sqrt(obs[j][2]**2 + obs[j][3]**2)
        yaw = obs[j][4]
        
        path = []
        x_list = [x]
        y_list = [y]
        yaw_list = [yaw]
    
        for k in range(future_frame):
            throttle = action_list[j*2*future_frame + 2*k] * throttle_scale
            delta = action_list[j*2*future_frame + 2*k+1] * steer_scale
            f_load = velocity * (c_r + c_a * velocity)

            velocity += (dt) * (throttle - f_load)
            if velocity <= 0:
                velocity = 0
            
            # Compute the radius and angular velocity of the kinematic bicycle model
            if delta >= max_steer:
                delta = max_steer
            elif delta <= -max_steer:
                delta = -max_steer
            # Compute the state change rate
            x_dot = velocity * cos(yaw)
            y_dot = velocity * sin(yaw)
            omega = velocity * tan(delta) / wheelbase

            # Compute the final state using the discrete time model
            x += x_dot * dt
            y += y_dot * dt
            yaw += omega * dt
            yaw = atan2(sin(yaw), cos(yaw))
                
            x_list.append(x)
            y_list.append(y)
            yaw_list.append(yaw)

        path.append(x_list)
        path.append(y_list)
        path.append(yaw_list)
        path_list.append(path)
        
    return path_list

# @jit(nopython=True)
def _colli_check_acc(ego_x_list, ego_y_list, ego_yaw_list, rollout_trajectory, future_frame, move_gap, check_radius, time_expansion_rate):
    
    for i in range(future_frame):
        ego_x = ego_x_list[i]
        ego_y = ego_y_list[i]
        ego_yaw = ego_yaw_list[i]
        
        ego_front_x = ego_x+np.cos(ego_yaw)*move_gap
        ego_front_y = ego_y+np.sin(ego_yaw)*move_gap
        ego_back_x = ego_x-np.cos(ego_yaw)*move_gap
        ego_back_y = ego_y-np.sin(ego_yaw)*move_gap
        
        for j in range(len(rollout_trajectory)):
            one_vehicle_path = rollout_trajectory[j]
            obst_x = one_vehicle_path[0][i]
            obst_y = one_vehicle_path[1][i]
            obst_yaw = one_vehicle_path[2][i]
            
            obst_front_x = obst_x+np.cos(obst_yaw)*move_gap
            obst_front_y = obst_y+np.sin(obst_yaw)*move_gap
            obst_back_x = obst_x-np.cos(obst_yaw)*move_gap
            obst_back_y = obst_y-np.sin(obst_yaw)*move_gap
            d = (ego_front_x - obst_front_x)**2 + (ego_front_y - obst_front_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            d = (ego_front_x - obst_back_x)**2 + (ego_front_y - obst_back_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            d = (ego_back_x - obst_front_x)**2 + (ego_back_y - obst_front_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            d = (ego_back_x - obst_back_x)**2 + (ego_back_y - obst_back_y)**2
            if d <= (2*check_radius+i*time_expansion_rate)**2: 
                return True
            
    return False
    

class DCP_Agent():
    def __init__(self, training=False):
        
        # transition model parameter        
        self.ensemble_num = 5
        self.used_ensemble_num = 5
        self.history_frame = 1
        self.future_frame = 20 # Note that the length of candidate trajectories should larger than future frame
        self.obs_scale = 10
        self.obs_bias_x = 130
        self.obs_bias_y = 200
        self.throttle_scale = 0.5
        self.steer_scale = 0.1
        self.agent_dimension = 5  # x,y,vx,vy,yaw
        self.agent_num = 2
        self.rollout_times = 1
        self.dt = 0.1
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type(torch.DoubleTensor)
        
        self.ensemble_transition_model = DCP_Transition_Model(self.ensemble_num, self.history_frame, 
                                                              self.future_frame, self.agent_dimension, self.agent_num,
                                                              self.obs_bias_x, self.obs_bias_y, self.obs_scale, 
                                                              self.throttle_scale, self.steer_scale, 
                                                              self.device, training, self.dt)
        self.history_obs_list = []
        self.rollout_trajectory_tuple = []
                
        # collision checking parameter
        self.robot_radius = 5.0
        self.move_gap = 2.5
        self.time_expansion_rate = 0.09
        
        self.check_radius = self.robot_radius
      
    def act(self, candidate_trajectories_tuple, dynamic_map):
        obs = self.wrap_state(dynamic_map)
        print("obs",obs)
        rollout_trajectory_tuple = self.generate_imagined_trajectories(obs, candidate_trajectories_tuple)

        worst_Q_list, used_worst_Q_list = self.calculate_worst_Q_value(candidate_trajectories_tuple, rollout_trajectory_tuple)
        dcp_action = np.where(used_worst_Q_list==np.max(used_worst_Q_list))[0] 
        print("worst_Q_list",worst_Q_list,used_worst_Q_list)
        print("dcp_action",dcp_action)
        return dcp_action

    def wrap_state(self, dynamic_map):
        state  = []

        ego_vehicle_state = [dynamic_map.ego_vehicle.x,
                                dynamic_map.ego_vehicle.y,
                                dynamic_map.ego_vehicle.vx,
                                dynamic_map.ego_vehicle.vy,
                                dynamic_map.ego_vehicle.yaw / 180.0 * math.pi]
        state.append(ego_vehicle_state)


        # Obs state    
        i = 0    
        for vehicle in dynamic_map.vehicles:
            if i < self.agent_num-1:
                vehicle_state = [vehicle.x,
                                vehicle.y,
                                vehicle.vx,
                                vehicle.vy,
                                vehicle.yaw / 180.0 * math.pi]
                state.append(vehicle_state)
                i += 1
            else:
                break
            
        
        if self.agent_num - len(state) > 0:
            for i in range(self.agent_num - len(state)):
                vehicle_state = [-999,-999,0,0,0]
                state.append(vehicle_state)
        
        return state

    def generate_imagined_trajectories(self, obs, candidate_trajectories_tuple, fixed_ego_traj=True):
        if fixed_ego_traj:
            action = candidate_trajectories_tuple[0]
            rollout_trajectory_tuple = []
            for ensemble_index in range(self.ensemble_num):
                ego_trajectory, rollout_trajectory = self.ensemble_transition_model.rollout(obs, action, ensemble_index)
                rollout_trajectory_tuple.append(rollout_trajectory)
            return rollout_trajectory_tuple
        else:
            return None

    def calculate_worst_Q_value(self, candidate_trajectories_tuple, rollout_trajectory_tuple):
        worst_Q_list = []
        used_worst_Q_list = []
        worst_Q_list.append(-500) # -500 reward for brake action, low reward but bigger than collision trajectories
        used_worst_Q_list.append(-500) # -500 reward for brake action, low reward but bigger than collision trajectories
        for action in candidate_trajectories_tuple:
            worst_Q_value = 999
            for ensemble_index in range(self.ensemble_num):
                q_value_for_a_head = self.q_value_for_a_head(rollout_trajectory_tuple[ensemble_index], action, ensemble_index)
                if q_value_for_a_head < worst_Q_value:
                    worst_Q_value = q_value_for_a_head
                if ensemble_index == self.used_ensemble_num-1:
                    used_worst_Q_list.append(worst_Q_value)
            worst_Q_list.append(worst_Q_value)       

        return worst_Q_list, used_worst_Q_list

    def q_value_for_a_head(self, rollout_trajectory_one_head, ego_trajectory, ensemble_index):
        g_value_list = []
        for i in range(self.rollout_times):
            g_value = self.calculate_g_value(ego_trajectory, rollout_trajectory_one_head)
            g_value_list.append(g_value)
            
        q_value = np.mean(g_value_list)
        return q_value
        
    def calculate_g_value(self, ego_trajectory, rollout_trajectory): 
        
        ego_x_list = np.array(ego_trajectory[0].x)
        ego_y_list = np.array(ego_trajectory[0].y)
        ego_yaw_list = np.array(ego_trajectory[0].yaw)
        rollout_trajectory = np.array(rollout_trajectory)
        
        if _colli_check_acc(ego_x_list, ego_y_list, ego_yaw_list, rollout_trajectory, self.future_frame, self.move_gap, self.check_radius, self.time_expansion_rate):
            g_colli = -500
        else:
            g_colli = 0
 
        g_ego = -ego_trajectory[1] # fp.cf # cost of whole trajectory
        g_value = g_colli + g_ego
        
        return g_value
    
    def colli_check(self, ego_trajectory, rollout_trajectory):
        for i in range(self.future_frame):
            ego_x = ego_trajectory[0].x[i]
            ego_y = ego_trajectory[0].y[i]
            ego_yaw = ego_trajectory[0].yaw[i]
            
            for one_vehicle_path in rollout_trajectory:
                obst_x = one_vehicle_path.x[i]
                obst_y = one_vehicle_path.y[i]
                obst_yaw = one_vehicle_path.yaw[i]
                
                if self.colli_between_vehicle(ego_x, ego_y, ego_yaw, obst_x, obst_y, obst_yaw):
                    return True
                
        return False
    
    def colli_between_vehicle(self, ego_x, ego_y, ego_yaw, obst_x, obst_y, obst_yaw):
        ego_front_x = ego_x+np.cos(ego_yaw)*self.move_gap
        ego_front_y = ego_y+np.sin(ego_yaw)*self.move_gap
        ego_back_x = ego_x-np.cos(ego_yaw)*self.move_gap
        ego_back_y = ego_y-np.sin(ego_yaw)*self.move_gap
        
        obst_front_x = obst_x+np.cos(obst_yaw)*self.move_gap
        obst_front_y = obst_y+np.sin(obst_yaw)*self.move_gap
        obst_back_x = obst_x-np.cos(obst_yaw)*self.move_gap
        obst_back_y = obst_y-np.sin(obst_yaw)*self.move_gap
        
        d = (ego_front_x - obst_front_x)**2 + (ego_front_y - obst_front_y)**2
        if d <= self.check_radius**2: 
            return True
        d = (ego_front_x - obst_back_x)**2 + (ego_front_y - obst_back_y)**2
        if d <= self.check_radius**2: 
            return True
        d = (ego_back_x - obst_front_x)**2 + (ego_back_y - obst_front_y)**2
        if d <= self.check_radius**2: 
            return True
        d = (ego_back_x - obst_back_x)**2 + (ego_back_y - obst_back_y)**2
        if d <= self.check_radius**2: 
            return True
        
        return False
        
       
class DCP_Transition_Model():
    def __init__(self, ensemble_num, history_frame, future_frame, agent_dimension, agent_num, obs_bias_x, obs_bias_y, 
                 obs_scale, throttle_scale, steer_scale, device, training, dt):
        super(DCP_Transition_Model, self).__init__()
        
        self.ensemble_num = ensemble_num
        self.history_frame = history_frame
        self.future_frame = future_frame
        
        self.obs_bias_x = obs_bias_x
        self.obs_bias_y = obs_bias_y
        self.obs_scale = obs_scale
        self.throttle_scale = throttle_scale
        self.steer_scale = steer_scale
        self.agent_dimension = agent_dimension  # x,y,vx,vy,yaw
        self.agent_num = agent_num

        self.ensemble_models = []
        self.ensemble_optimizer = []
        self.device = device
        
        for i in range(self.ensemble_num):
            env_transition = TrajPredGaussion(self.history_frame * self.agent_dimension * self.agent_num,
                                              self.future_frame * 2 * self.agent_num, hidden_unit=128)
            env_transition.to(self.device)
            env_transition.apply(self.weight_init)
            if training:
                env_transition.train()
  
            self.ensemble_models.append(env_transition)
            self.ensemble_optimizer.append(torch.optim.Adam(env_transition.parameters(), lr=0.005, weight_decay=0))
            
        # transition vehicle model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(80)
        self.dt = dt
        self.c_r = 0.01
        self.c_a = 0.05
        self.kbm = KinematicBicycleModel(
            self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)
        
        # dataset
        self.data = []
        self.trained_data = []
        self.one_trajectory = []
        self.infer_obs_list = []
    
        self.rollout_times = 30   

    def rollout(self, state, candidate_trajectory, ensemble_index):
    
        if ensemble_index > self.ensemble_num-1:
            print("[Warning]: Ensemble Index out of index!")
            return None
        
        rollout_trajectory = []
        history_obs = state
        obs = copy.deepcopy(state)
        vehicle_num = 0
        
        for i in range(len(history_obs)):
            if history_obs[i][0] != -999: # use -100 as signal, very unstable
                vehicle_num += 1
        
        
        history_obs = self.normalize_state(history_obs)        
        history_obs = torch.tensor(history_obs).to(self.device)

        predict_action, sigma = self.ensemble_models[ensemble_index](history_obs)
        predict_action = predict_action.cpu().detach().numpy()
        
        time1 = time.time()

        for j in range(1, vehicle_num):  # exclude ego vehicle
            # one_path = Frenet_path()
            # one_path.t = [t for t in np.arange(0.0, 0.1 * self.future_frame, 0.1)]
            # one_path.c = j  # use the c to indicate which vehicle
            # one_path.cd = ensemble_index  # use the cd to indicate which ensemble model
            # one_path.cf = self.ensemble_num  # use the cf to indicate heads num
            path = []
            x_list = []
            y_list = []
            yaw_list = []
            x = obs[j][0]
            y = obs[j][1]
            velocity = math.sqrt(obs[j][2]**2 + obs[j][3]**2)
            yaw = obs[j][4]
            for k in range(0, self.future_frame):
                throttle = predict_action[j*2*self.future_frame + 2*k] * self.throttle_scale
                delta = predict_action[j*2*self.future_frame + 2*k+1] * self.steer_scale
                x, y, yaw, velocity, _, _ = self.kbm.kinematic_model(x, y, yaw, velocity, throttle, delta)
                x_list.append(x)
                y_list.append(y)
                yaw_list.append(yaw)
                
            path.append(x_list)
            path.append(y_list)
            path.append(yaw_list)
            rollout_trajectory.append(path)

        time2 = time.time()
        # print("time",time2-time1)

    
        ego_trajectory = candidate_trajectory

        
        return ego_trajectory, rollout_trajectory
    
    def rollout_jit_acc(self, state, candidate_trajectory, ensemble_index):
        time1 = time.time()

        if ensemble_index > self.ensemble_num-1:
            print("[Warning]: Ensemble Index out of index!")
            return None
        
        rollout_trajectory = []
        history_obs = state
        obs = copy.deepcopy(state)
        

        vehicle_num = 0
        for i in range(len(history_obs)):
            if history_obs[i][0] != -999: # use -100 as signal, very unstable
                vehicle_num += 1

        history_obs = self.normalize_state(history_obs)        
        history_obs = torch.tensor(history_obs).to(self.device)
        time2 = time.time()

        predict_action, sigma = self.ensemble_models[ensemble_index](history_obs)
        time3 = time.time()

        predict_action = predict_action.cpu().detach().numpy()
        print("obs",obs,len(predict_action))
        rollout_trajectory = _kinematic_model(vehicle_num, numba.typed.List(obs), self.future_frame, predict_action, self.throttle_scale, 
                                              self.steer_scale, self.dt)

        time4 = time.time()
        # print("time_consume",time4-time3, time3-time2 , time2-time1)             
        ego_trajectory = candidate_trajectory

        return ego_trajectory, rollout_trajectory
  
    # transition_training functions
    def add_training_data(self, obs, done):
        trajectory_length = self.history_frame + self.future_frame
        if not done:
            obs = np.array(obs)
            self.one_trajectory.append(obs)
            if len(self.one_trajectory) >= trajectory_length:
                self.data.append(self.one_trajectory[0:trajectory_length])
                self.one_trajectory.pop(0)
        else:
            self.one_trajectory = []
 
    def normalize_state(self, history_obs):
        normalize_state = []
        obs = history_obs
        normalize_obs = copy.deepcopy(obs)
        obs_length = self.agent_num
        for i in range(self.agent_num):
            if obs[i][0] == -999:
                normalize_obs[i][0] = 20
                normalize_obs[i][1] = 0
            else:
                normalize_obs[i][0] = obs[i][0] - self.obs_bias_x
                normalize_obs[i][1] = obs[i][1] - self.obs_bias_y
        normalize_state.append(normalize_obs)
        
        return (np.array(normalize_state).flatten()/self.obs_scale) # flatten to list
    
    def update_model(self):
        if len(self.data) > 0:
            # take data
            for k in range(1):
                # one_trajectory = self.data[random.randint(0, len(self.data)-1)]
                one_trajectory = self.data[0]

                history_obs = one_trajectory[0:self.history_frame] 
                history_obs = self.normalize_state(history_obs)
                history_obs = torch.tensor(history_obs).to(self.device)

                # target: output action
                target_action = self.get_target_action_from_obs(one_trajectory)
                target_action = np.array(target_action).flatten().tolist()
                target_action = torch.tensor(target_action).to(self.device)

                for i in range(self.ensemble_num):
                    # compute loss
                    predict_action, sigma = self.ensemble_models[i](history_obs)
                    # print("target_action",target_action[40:80])
                    # print("predict_action",predict_action[40:80])
                    # print("sigma", sigma)
                    # sigma = torch.ones_like(predict_action)
                    diff = (predict_action - target_action) / sigma
                    loss = torch.mean(0.5 * torch.pow(diff, 2) + torch.log(sigma))  
                    # loss = F.mse_loss(predict_action, target_action)
                    print("------------loss", loss)

                    # train
                    self.ensemble_optimizer[i].zero_grad()
                    loss.backward()
                    self.ensemble_optimizer[i].step()

                # closed loop test
                # candidate_trajectory = 1
                # ego_trajectory, rollout_trajectory = self.rollout(one_trajectory[0:self.history_frame] , candidate_trajectory, 0)
                # dx = (rollout_trajectory[0].x[-1] - one_trajectory[-1][1][0])
                # dy = (rollout_trajectory[0].y[-1] - one_trajectory[-1][1][1]) 
                # fde = math.sqrt(dx*dx + dy*dy)
                # print("dx",rollout_trajectory[0].x[-1], one_trajectory[-1][1][0])
                # print("dy",rollout_trajectory[0].y[-1], one_trajectory[-1][1][1])
                # print("fde", fde)

                self.trained_data.append(one_trajectory)
                self.data.pop(0)

        return None
 
    def get_target_action_from_obs(self, one_trajectory):

        vehicle_num = 0
        for i in range(len(one_trajectory[0])):
            if one_trajectory[0][i][0] != -999: # use -100 as signal, very unstable
                vehicle_num += 1
                
        action_list = []
        for j in range(0, vehicle_num):
            vehicle_action = []
            # print("one__trajecctory")
            for i in range(0, self.future_frame):
                x1 = one_trajectory[self.history_frame-1+i][j][0]
                y1 = one_trajectory[self.history_frame-1+i][j][1]
                yaw1 = one_trajectory[self.history_frame-1+i][j][4]
                v1 = math.sqrt(one_trajectory[self.history_frame-1+i][j][2]
                               ** 2 + one_trajectory[self.history_frame-1+i][j][3] ** 2)
                x2 = one_trajectory[self.history_frame+i][j][0]
                y2 = one_trajectory[self.history_frame+i][j][1]
                yaw2 = one_trajectory[self.history_frame+i][j][4]
                v2 = math.sqrt(one_trajectory[self.history_frame+i][j][2]
                               ** 2 + one_trajectory[self.history_frame+i][j][3] ** 2)
                throttle, delta = self.kbm.calculate_a_from_data(
                    x1, y1, yaw1, v1, x2, y2, yaw2, v2)
                # print("get_target_action1",x1, y1, yaw1)
                # print("get_target_action2",x2, y2, yaw2)
                # print("get_target_action3",throttle, delta)

                vehicle_action.append(throttle/self.throttle_scale)
                vehicle_action.append(delta/self.steer_scale)
            action_list.append(vehicle_action)

            # check this action calculation
            # x = one_trajectory[self.history_frame-1][j][0]
            # y = one_trajectory[self.history_frame-1][j][1]
            # yaw = one_trajectory[self.history_frame-1][j][4]
            # velocity = math.sqrt(one_trajectory[self.history_frame-1][j][2]
            #                 ** 2 + one_trajectory[self.history_frame-1][j][3] ** 2)
            
            # for k in range(0, self.future_frame):
            #     throttle = vehicle_action[2*k] * self.throttle_scale
            #     delta = vehicle_action[2*k+1] * self.steer_scale
            #     x, y, yaw, velocity, _, _ = self.kbm.kinematic_model(
            #         x, y, yaw, velocity, throttle, delta)
            # print("dx",x-one_trajectory[-1][j][0])
            # print("dy",y-one_trajectory[-1][j][1])
                
        for k in range (self.agent_num - vehicle_num):
            vehicle_action = []
            for i in range(0, self.future_frame):
                vehicle_action.append(0) 
                vehicle_action.append(0) 
            action_list.append(vehicle_action)

        return action_list 
 
    def weight_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.uniform_(m.weight, a=-0.1, b=0.1)
            # nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)
        # 也可以判断是否为conv2d，使用相应的初始化方式
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        # 是否为批归一化层
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def load(self, load_step):
        try:
            for i in range(self.ensemble_num):

                self.ensemble_models[i].load_state_dict(
                    torch.load('DCP_models/ensemble_models_%s_%s.pt' %
                               (load_step, i))
                )
            print("[DCP] : Load Learned Model, Step=", load_step)
        except:
            load_step = 0
            print("[DCP] : No Learned Model, Creat New Model")
        return load_step
   
    def save(self, train_step):
        for i in range(self.ensemble_num):
            torch.save(
                self.ensemble_models[i].state_dict(),
                'DCP_models/ensemble_models_%s_%s.pt' % (train_step, i)
            )


if __name__ == '__main__':
    a=1
