
import sys

sys.path.append("..")

import math

import numpy as np

from Agent.zzz.controller import Controller
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.frenet import Frenet_path
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from TestScenario_Town03_Waymo_long_tail import CarEnv_03_Waymo_Long_Tail

if __name__ == '__main__':

    # Create environment 
    env = CarEnv_03_Waymo_Long_Tail()

    # Create Agent
    trajectory_planner = JunctionTrajectoryPlanner()
    controller = Controller()
    dynamic_map = DynamicMap()
    dynamic_map.update_ref_path(env)
    
    obs = env.reset()
    done = False
    
    for i in range(10):
        while True:
            obs = np.array(obs)
            dynamic_map.update_map_from_list_obs(obs)

            candidate_trajectories_tuple = trajectory_planner.generate_candidate_trajectories(dynamic_map)

            chosen_action_id = 1
            chosen_trajectory = trajectory_planner.trajectory_update_CP(chosen_action_id)
           
            control_action =  controller.get_control(dynamic_map,  chosen_trajectory.trajectory, chosen_trajectory.desired_speed)
            action = [control_action.acc , control_action.steering]
            new_obs, reward, done, collision_signal = env.step(action)   
            obs = new_obs

            
            if done:
                print("done!")
                trajectory_planner.clear_buff(clean_csp=False)
                obs = env.reset()
                break
                    

            


            
        

         
            

    