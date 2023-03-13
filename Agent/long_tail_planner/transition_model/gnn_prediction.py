import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.frenet import Frenet_path
from Agent.zzz.JunctionTrajectoryPlanner_simple_predict import \
    JunctionTrajectoryPlanner
from Agent.zzz.prediction.agent_model.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel
from Agent.zzz.prediction.coordinates import Coordinates
from Agent.zzz.prediction.KinematicBicycleModel.kinematic_model import \
    KinematicBicycleModel
from Agent.zzz.prediction.predmlp import TrajPredGaussion, TrajPredMLP
from Agent.zzz.prediction.selfatten import SelfAttentionLayer
from results import Results
from tqdm import tqdm

Use_Gaussion_Output = False


class GNN_Prediction_Model(nn.Module):
    """
    Self_attention GNN with trajectory prediction MLP
    """

    def __init__(self, in_channels, out_channels, obs_scale, use_gaussion, global_graph_width=64, traj_pred_mlp_width=64):
        super(GNN_Prediction_Model, self).__init__()
        self.polyline_vec_shape = in_channels
        self.self_atten_layer = SelfAttentionLayer(
            self.polyline_vec_shape, global_graph_width)
        self.self_atten_layer_2 = SelfAttentionLayer(
            global_graph_width, global_graph_width)

        self.obs_scale = obs_scale
        self.use_gaussion = use_gaussion
        if self.use_gaussion:
            self.traj_pred_gau = TrajPredGaussion(
                global_graph_width, out_channels, traj_pred_mlp_width)
        else:
            self.traj_pred_mlp = TrajPredMLP(
                global_graph_width, out_channels, traj_pred_mlp_width)

    def forward(self, obs):
        out = self.self_atten_layer(obs)
        # out = self.self_atten_layer_2(out)
        if self.use_gaussion:
            pred_action, sigma = self.traj_pred_gau(out)
            return pred_action, sigma

        else:
            pred_action = self.traj_pred_mlp(out)
            return pred_action

class Prediction_Model_Training():

    def __init__(self):
        self.data = []
        self.trained_data = []
        self.one_trajectory = []
        self.infer_obs_list = []

        self.ensemble_models = []
        self.ensemble_optimizer = []
        self.train_step = 0

        # Parameters of Prediction Model
        self.heads_num = 10
        self.history_frame = 5
        self.future_frame = 30
        self.obs_scale = 5
        self.action_scale = 5
        self.agent_dimension = 5  # x,y,vx,vy,yaw

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        torch.set_default_tensor_type(torch.DoubleTensor)

        for i in range(self.heads_num):
            predition_model = GNN_Prediction_Model(
                self.history_frame * 5, self.future_frame * 2, self.obs_scale, Use_Gaussion_Output).to(self.device)
            predition_model.apply(self.weight_init)
            self.ensemble_models.append(predition_model)
            self.ensemble_optimizer.append(torch.optim.Adam(
                predition_model.parameters(), lr=0.001))
            predition_model.train()

        # Vehicle Model
        self.wheelbase = 2.96
        self.max_steer = np.deg2rad(30)
        self.dt = 0.1
        self.c_r = 0.01
        self.c_a = 0.05
        self.kbm = KinematicBicycleModel(
            self.wheelbase, self.max_steer, self.dt, self.c_r, self.c_a)

    def add_data(self, obs, done):
        trajectory_length = self.history_frame + self.future_frame
        if not done:
            self.one_trajectory.append(obs)
            if len(self.one_trajectory) > trajectory_length:
                self.data.append(self.one_trajectory[0:trajectory_length])
                self.one_trajectory.pop(0)
        else:
            self.one_trajectory = []

    def learn(self, env, load_step, train_episode=1):
        # Create Agent
        trajectory_planner = JunctionTrajectoryPlanner()
        controller = Controller()
        dynamic_map = DynamicMap()
        target_speed = 30/3.6

        results = Results(self.history_frame)

        pass_time = 0
        task_time = 0

        # Load_model
        self.load_prediction_model(load_step)

        # Collect Data from CARLA and train model
        for episode in tqdm(range(load_step, train_episode + load_step), unit='episodes'):

            # Reset environment and get initial state
            obs = env.reset()
            episode_reward = 0
            done = False
            decision_count = 0

            # Loop over steps
            while True:
                obs = np.array(obs)
                dynamic_map.update_map_from_list_obs(obs, env)
                rule_trajectory, action = trajectory_planner.trajectory_update(
                    dynamic_map)
                rule_trajectory = trajectory_planner.trajectory_update_CP(
                    action, rule_trajectory)
                # Control

                control_action = controller.get_control(
                    dynamic_map,  rule_trajectory.trajectory, rule_trajectory.desired_speed)
                action = [control_action.acc, control_action.steering]
                new_obs, reward, done, _ = env.step(action)

                self.add_data(new_obs, done)
                self.train_model()

                obs = new_obs
                episode_reward += reward
                self.train_step += 1

                if done:
                    trajectory_planner.clear_buff(clean_csp=False)
                    task_time += 1
                    if reward > 0:
                        pass_time += 1
                    break

        # Calculate Prediction Results in the end
        results.calculate_predition_results(
            self.trained_data, self.predict_future_paths, self.history_frame)

    def train_model(self):
        if len(self.data) > 0:
            # take data
            one_trajectory = self.data[0]
            
            vehicle_num = 0
            for i in range(len(one_trajectory[0])):
                if one_trajectory[0][i][0] != -999: # use -100 as signal, very unstable
                    vehicle_num += 1
                    
            # input: transfer to ego
            history_obs = one_trajectory[0:self.history_frame]
            # print("one_trajectory",one_trajectory)

            history_data = []
            for j in range(0, vehicle_num):
                vehicle_state = []
                # Use the first frame ego vehicle as center
                ego_vehicle_coordiate = Coordinates(
                    history_obs[0][0][0], history_obs[0][0][1], history_obs[0][0][4])
                for obs in history_obs:
                    x_t, y_t, vx_t, vy_t, yaw_t = ego_vehicle_coordiate.transfer_coordinate(obs[j][0], obs[j][1],
                                                                                            obs[j][2], obs[j][3], obs[j][4])
                    scale_state = [
                        x / self.obs_scale for x in [x_t, y_t, vx_t, vy_t, yaw_t]]
                    vehicle_state.extend(scale_state)
                history_data.append(vehicle_state)
            history_data = torch.tensor(history_data).to(
                self.device).unsqueeze(0)
            # target: output action
            target_action = self.get_target_action_from_obs(one_trajectory)
            target_action = torch.tensor(
                target_action).to(self.device).unsqueeze(0)

            for i in range(self.heads_num):
                # compute loss
                if Use_Gaussion_Output:
                    predict_action, sigma = self.ensemble_models[i](
                        history_data)

                    diff = (predict_action - target_action) / sigma
                    loss = torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
                else:
                    predict_action = self.ensemble_models[i](history_data)
                    loss = F.mse_loss(target_action, predict_action)

                # print("target_action",target_action)
                # print("predict_action",predict_action)
                # print("sigma",sigma)
                print("------------loss", loss)

                # train
                self.ensemble_optimizer[i].zero_grad()
                loss.backward()

                self.ensemble_optimizer[i].step()

            # closed loop test
            self.predict_future_paths(one_trajectory[0], done=False)
            self.predict_future_paths(one_trajectory[1], done=False)
            self.predict_future_paths(one_trajectory[2], done=False)
            self.predict_future_paths(one_trajectory[3], done=False)
            paths_of_all_models = self.predict_future_paths(
                one_trajectory[4], done=False)
            # print("paths_of_all_models.x",paths_of_all_models[0].x)
            # print("paths_of_all_models.y",paths_of_all_models[0].y)
            # print("realPath.x",one_trajectory[5][1][0])
            # print("realPath.x",one_trajectory[6][1][0])
            # print("realPath.x",one_trajectory[7][1][0])
            # print("realPath.x",one_trajectory[8][1][0])
            # print("realPath.y",one_trajectory[5][1][1])
            dx = paths_of_all_models[0].x[-1] - one_trajectory[-1][1][0]
            dy = paths_of_all_models[0].y[-1] - one_trajectory[-1][1][1]
            fde = math.sqrt(dx**2 + dy**2)
            # print("len:paths_of_all_models",len(paths_of_all_models))
            # print("paths_of_all_models",paths_of_all_models[0].x)
            # print("paths_of_all_models",paths_of_all_models[1].x)
            # print("paths_of_all_models",paths_of_all_models[2].x)
            # print("one_trajectory",one_trajectory[4][0])
            # print("one_trajectory",one_trajectory[4][1])
            print("fde", fde)

            self.trained_data.append(one_trajectory)
            self.data.pop(0)

        if self.train_step % 10000 == 0:
            self.save_prediction_model(self.train_step)

        return None

    def get_target_action_from_obs(self, one_trajectory):
        action_list = []
        vehicle_num = 0
        for i in range(len(one_trajectory[0])):
            if one_trajectory[0][i][0] != -999: # use -100 as signal, very unstable
                vehicle_num += 1
        action_list = []

        for j in range(0, vehicle_num):
            vehicle_action = []
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

                vehicle_action.append(throttle/self.action_scale)
                vehicle_action.append(delta/self.action_scale)
            action_list.append(vehicle_action)

        return action_list

    def predict_future_paths(self, obs, done):

        if done:
            self.infer_obs_list = []

        self.infer_obs_list.append(obs)
        if len(self.infer_obs_list) >= self.history_frame:
            # modify history data and transfer to ego coordiate
            history_obs = self.infer_obs_list
            history_data = []
            vehicle_num = 0
            for i in range(len(history_obs[0])):
                if history_obs[0][i][0] != -999: # use -100 as signal, very unstable
                    vehicle_num += 1

            for j in range(0, vehicle_num):
                vehicle_state = []
                # Use the first frame ego vehicle as center
                ego_vehicle_coordiate = Coordinates(
                    history_obs[0][0][0], history_obs[0][0][1], history_obs[0][0][4])
                for obs in history_obs:
                    x_t, y_t, vx_t, vy_t, yaw_t = ego_vehicle_coordiate.transfer_coordinate(obs[j][0], obs[j][1],
                                                                                            obs[j][2], obs[j][3], obs[j][4])
                    scale_state = [
                        x / self.obs_scale for x in [x_t, y_t, vx_t, vy_t, yaw_t]]
                    vehicle_state.extend(scale_state)
                history_data.append(vehicle_state)
            history_data = torch.tensor(history_data).to(self.device).unsqueeze(0)

            # infer using prediction model
            paths_of_all_models = []
            for i in range(self.heads_num):
                # paths_of_one_model = []
                if Use_Gaussion_Output:
                    predict_action, sigma = self.ensemble_models[i](
                        history_data)
                    predict_action = predict_action.cpu().detach().numpy()
                else:
                    predict_action = self.ensemble_models[i](
                        history_data).cpu().detach().numpy()

                for j in range(1, vehicle_num):  # exclude ego vehicle
                    one_path = Frenet_path()
                    one_path.t = [t for t in np.arange(
                        0.0, 0.1 * self.future_frame, 0.1)]
                    one_path.c = j  # use the c to indicate which vehicle
                    one_path.cd = i  # use the cd to indicate which ensemble model
                    one_path.cf = self.heads_num  # use the cf to indicate heads num
                    x = obs[j][0]
                    y = obs[j][1]
                    velocity = math.sqrt(obs[j][2]**2 + obs[j][3]**2)
                    yaw = obs[j][4]
                    for k in range(0, self.future_frame):

                        throttle = predict_action[0][j][2*k] * self.action_scale
                        delta = predict_action[0][j][2*k+1] * self.action_scale
                        x, y, yaw, velocity, _, _ = self.kbm.kinematic_model(
                            x, y, yaw, velocity, throttle, delta)
                        one_path.x.append(x)
                        one_path.y.append(y)
                    # paths_of_one_model.append(one_path)
                    paths_of_all_models.append(one_path)

            self.infer_obs_list.pop(0)

            return paths_of_all_models

        return None

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

    def save_prediction_model(self, step):
        for i in range(self.heads_num):
            torch.save(
                self.ensemble_models[i].state_dict(),
                'save_model/ensemble_models_%s_%s.pt' % (step, i)
            )

    def load_prediction_model(self, load_step):
        try:
            for i in range(self.heads_num):

                self.ensemble_models[i].load_state_dict(
                    torch.load('save_model/ensemble_models_%s_%s.pt' %
                               (load_step, i))
                )
            print(
                "[Prediction_Model] : Load learned model successful, step=", load_step)
        except:
            load_step = 0
            print("[Prediction_Model] : No learned model, Creat new model")
        return load_step


class Prediction():
    def __init__(self, considered_obs_num, maxt, dt, robot_radius, radius_speed_ratio, move_gap):

        self.maxt = maxt
        self.dt = dt
        self.robot_radius = robot_radius
        self.radius_speed_ratio = radius_speed_ratio
        self.move_gap = move_gap
        self.considered_obs_num = considered_obs_num

        self.gnn_predictin_model = Prediction_Model_Training()
        self.gnn_predictin_model.load_prediction_model(50000)

    def update_prediction(self, dynamic_map):
        self.dynamic_map = dynamic_map
        self.check_radius = self.robot_radius + \
            self.radius_speed_ratio * self.dynamic_map.ego_vehicle.v

        self.predict_paths = self.gnn_predictin_model.predict_future_paths(
            dynamic_map.real_time_obs, dynamic_map.done)
        if self.predict_paths is None:
            interested_vehicles = self.found_interested_vehicles(
                self.considered_obs_num)
            self.predict_paths = self.prediction_obstacle_uniform_speed(
                interested_vehicles, self.maxt, self.dt)

    def check_collision(self, fp):

        if len(self.predict_paths) == 0 or len(fp.t) < 2:
            return True

        # two circles for a vehicle
        fp_front = copy.deepcopy(fp)
        fp_back = copy.deepcopy(fp)

        fp_front.x = (np.array(fp.x)+np.cos(np.array(fp.yaw))
                      * self.move_gap).tolist()
        fp_front.y = (np.array(fp.y)+np.sin(np.array(fp.yaw))
                      * self.move_gap).tolist()

        fp_back.x = (np.array(fp.x)-np.cos(np.array(fp.yaw))
                     * self.move_gap).tolist()
        fp_back.y = (np.array(fp.y)-np.sin(np.array(fp.yaw))
                     * self.move_gap).tolist()

        for path in self.predict_paths:

            len_predict_t = min(len(fp.x)-1, len(path.t)-1)
            predict_step = 2
            start_predict = 2
            for t in range(start_predict, len_predict_t, predict_step):
                d = (path.x[t] - fp_front.x[t])**2 + \
                    (path.y[t] - fp_front.y[t])**2
                if d <= self.check_radius**2:
                    return False
                d = (path.x[t] - fp_back.x[t])**2 + \
                    (path.y[t] - fp_back.y[t])**2
                if d <= self.check_radius**2:
                    return False

        return True

    def found_interested_vehicles(self, interested_vehicles_num=3):

        interested_vehicles = []

        # Get interested vehicles by distance
        distance_tuples = []
        ego_loc = np.array([self.dynamic_map.ego_vehicle.x,
                           self.dynamic_map.ego_vehicle.y])

        for vehicle_idx, vehicle in enumerate(self.dynamic_map.vehicles):
            vehicle_loc = np.array([vehicle.x, vehicle.y])
            d = np.linalg.norm(vehicle_loc - ego_loc)

            distance_tuples.append((d, vehicle_idx))

        sorted_vehicle = sorted(
            distance_tuples, key=lambda vehicle_dis: vehicle_dis[0])

        for _, vehicle_idx in sorted_vehicle:
            interested_vehicles.append(self.dynamic_map.vehicles[vehicle_idx])
            if len(interested_vehicles) >= interested_vehicles_num:
                break
        return interested_vehicles

    def prediction_obstacle_uniform_speed(self, vehicles, max_prediction_time, delta_t):
        predict_paths = []
        for vehicle in vehicles:

            predict_path_front = Frenet_path()
            predict_path_back = Frenet_path()
            predict_path_front.t = [t for t in np.arange(
                0.0, max_prediction_time, delta_t)]
            predict_path_back.t = [t for t in np.arange(
                0.0, max_prediction_time, delta_t)]
            ax = 0  # one_ob[9]
            ay = 0  # one_ob[10]
            # print("vehicle information",vehicle.x, vehicle.y, vehicle.vx, vehicle.vy, vehicle.yaw)

            vx_predict = vehicle.vx*np.ones(len(predict_path_front.t))
            vy_predict = vehicle.vy*np.ones(len(predict_path_front.t))

            x_predict = vehicle.x + \
                np.arange(len(predict_path_front.t))*delta_t*vx_predict
            y_predict = vehicle.y + \
                np.arange(len(predict_path_front.t))*delta_t*vy_predict

            predict_path_front.x = (x_predict + math.cos(vehicle.yaw)
                                    * np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_front.y = (y_predict + math.sin(vehicle.yaw)
                                    * np.ones(len(predict_path_front.t))*self.move_gap).tolist()
            predict_path_back.x = (x_predict - math.cos(vehicle.yaw)
                                   * np.ones(len(predict_path_back.t))*self.move_gap).tolist()
            predict_path_back.y = (y_predict - math.sin(vehicle.yaw)
                                   * np.ones(len(predict_path_back.t))*self.move_gap).tolist()

            predict_paths.append(predict_path_front)
            predict_paths.append(predict_path_back)

        return predict_paths
