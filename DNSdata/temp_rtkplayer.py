#!/usr/bin/env python3

###############################################################################
# Copyright 2017 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
"""
Generate Planning Path
"""

import argparse
import atexit
import logging
import math
import os
import sys
import time
import numpy as np
import _thread

from numpy import genfromtxt
import scipy.signal as signal

from cyber.python.cyber_py3 import cyber
from cyber.python.cyber_py3 import cyber_time
from modules.tools.common.logger import Logger
from modules.canbus.proto import chassis_pb2
from modules.common.configs.proto import vehicle_config_pb2
from modules.common.proto import drive_state_pb2
from modules.common.proto import pnc_point_pb2
from modules.control.proto import pad_msg_pb2
from modules.localization.proto import localization_pb2
from modules.planning.proto import planning_pb2
from modules.prediction.proto import  prediction_obstacle_pb2
import modules.tools.common.proto_utils as proto_utils
from modules.dreamview.proto import chart_pb2

## local package
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.frenet import Frenet_path
from Agent.zzz.JunctionTrajectoryPlanner_simple_predict import JunctionTrajectoryPlanner_SP

from Agent.zzz.dynamic_map import Lane, Lanepoint, Vehicle
from scipy.spatial.transform import Rotation as R
from modules.routing.proto import routing_pb2
from modules.perception.proto import perception_obstacle_pb2
from matplotlib import pyplot as plt




# TODO(all): hard-coded path temporarily. Better approach needed.
APOLLO_ROOT = "/apollo"

target_v_list = [[0,30/3.6],[21,20/3.6],[120,25/3.6],[300,10/3.6],[340,25/3.6],[560,10/3.6],[610,40/3.6]] #eath item [s,v] after each s, the target velocity change to v
intersection_s_list = [340] # the obs 

dns_speed_list=[15/3.6, 20/3.6, 25/3.6,30/3.6]  #四个位置 0
pub_dns_map_data = True 

class RtkPlayer(object):
    """
    rtk player class
    """

    def __init__(self, record_file, node):
        """Init player."""
        self.firstvalid = False
        self.logger = Logger.get_logger(tag="RtkPlayer")
        self.logger.info("Load record file from: %s" % record_file)
        try:
            file_handler = open(record_file, 'r')
        except (FileNotFoundError, IOError) as ex:
            self.logger.error("Error opening {}: {}".format(record_file, ex))
            sys.exit(1)

        self.data = genfromtxt(file_handler, delimiter=',', names=True)
        file_handler.close()
        self.starttime = cyber_time.Time.now().to_sec()
        self.localization = localization_pb2.LocalizationEstimate()
        self.prediction = prediction_obstacle_pb2.PredictionObstacles()
        self.perception = perception_obstacle_pb2.PerceptionObstacles()
        self.chassis = chassis_pb2.Chassis()
        self.padmsg = pad_msg_pb2.PadMessage()
        self.localization_received = False
        self.chassis_received = False

        self.planning_pub = node.create_writer('/apollo/planning',
                                               planning_pb2.ADCTrajectory)

        self.speedmultiplier = 1
        self.terminating = False
        self.sequence_num = 0

        b, a = signal.butter(6, 0.05, 'low')
        self.data['acceleration'] = signal.filtfilt(b, a,
                                                    self.data['acceleration'])

        self.start = 0
        self.end = 0
        self.closestpoint = 0
        self.logger.info("Planning Ready")

        # init
        self.carx = 0
        self.cary = 0
        self.yaw = 0
        self.carvx = 0
        self.carvy = 0
        self.obss = None

        # stop
        self.stop_count = 0
        self.brake_trajectory = None

        
    def localization_callback(self, data):
        """
        New localization Received
        """

        self.localization.CopyFrom(data)
        self.carx = self.localization.pose.position.x
        self.cary = self.localization.pose.position.y
        self.carz = self.localization.pose.position.z
        self.carvx = self.localization.pose.linear_velocity.x
        self.carvy = self.localization.pose.linear_velocity.y
        
        # self.qx = self.localization.pose.orientation.qx
        # self.qy = self.localization.pose.orientation.qy
        # self.qz = self.localization.pose.orientation.qz
        # self.qw = self.localization.pose.orientation.qw
       
        self.localization_received = True
        # self.yaw =  self.xyzw2yaw()
        self.yaw = self.localization.pose.heading

    def prediction_callback(self, data):
        self.prediction.CopyFrom(data)
        # self.logger.info(self.prediction)

    def perception_callback(self, data):
        self.perception.CopyFrom(data)
        self.obss = self.perception.perception_obstacle    #list
        # self.logger.info(self.obss)

        # message PerceptionObstacles {
        # repeated PerceptionObstacle perception_obstacle = 1;  // An array of obstacles
        # optional apollo.common.Header header = 2;             // Header
        # optional apollo.common.ErrorCode error_code = 3 [default = OK];
        # optional LaneMarkers lane_marker = 4;
        # optional CIPVInfo cipv_info = 5;  // Closest In Path Vehicle (CIPV)
        # }

    def routing_callback(self, data):
        self.routing.CopyFrom(data)
        # print(self.rounting)
        # self.logger.info(self.rounting)

    def chassis_callback(self, data):
        """
        New chassis Received
        """
        self.chassis.CopyFrom(data)
        self.automode = (self.chassis.driving_mode
                         == chassis_pb2.Chassis.COMPLETE_AUTO_DRIVE)
        self.chassis_received = True

    def publish_planningmsg(self):
        """
        Generate New Path
        """
        if not self.localization_received:
            self.logger.warning(
                "localization not received yet when publish_planningmsg")
            return

        planningdata = planning_pb2.ADCTrajectory()
        now = cyber_time.Time.now().to_sec()
        planningdata.header.timestamp_sec = now
        planningdata.header.module_name = "planning"
        planningdata.header.sequence_num = self.sequence_num
        self.sequence_num = self.sequence_num + 1

        self.start = 0
        self.end = len(self.data) - 1


        planningdata.total_path_length = self.data['s'][self.end] - \
            self.data['s'][self.start]
        # self.logger.info("total number of planning data point: %d" %
        #                  (self.end - self.start))
        planningdata.total_path_time = self.data['time'][self.end] - \
            self.data['time'][self.start]
        planningdata.gear = 1
        planningdata.engage_advice.advice = \
            drive_state_pb2.EngageAdvice.READY_TO_ENGAGE

        for i in range(self.start, self.end):
            adc_point = pnc_point_pb2.TrajectoryPoint()
            adc_point.path_point.x = self.data['x'][i]
            adc_point.path_point.y = self.data['y'][i]
            adc_point.path_point.z = self.data['z'][i]
            adc_point.v = self.data['speed'][i] * self.speedmultiplier
            adc_point.a = self.data['acceleration'][i] * self.speedmultiplier
            adc_point.path_point.kappa = self.data['curvature'][i]
            adc_point.path_point.dkappa = self.data['curvature_change_rate'][i]
            adc_point.path_point.theta = self.data['theta'][i]
            adc_point.path_point.s = self.data['s'][i]


            time_diff = self.data['time'][i] - \
                self.data['time'][0]

            adc_point.relative_time = time_diff  - (
                now - self.starttime)

            planningdata.trajectory_point.extend([adc_point])

        planningdata.estop.is_estop = False
        self.planning_pub.write(planningdata)
        # self.logger.debug("Generated Planning Sequence: "
        #                   + str(self.sequence_num - 1))

    def publish_planningmsg_start(self):
        """
        Generate New Path
        """
        if not self.localization_received:
            self.logger.warning(
                "localization not received yet when publish_planningmsg")
            return

        planningdata = planning_pb2.ADCTrajectory()
        now = cyber_time.Time.now().to_sec()
        planningdata.header.timestamp_sec = now
        planningdata.header.module_name = "planning"
        planningdata.header.sequence_num = self.sequence_num
        self.sequence_num = self.sequence_num + 1

        self.start = 0
        self.end = len(self.data) - 1


        planningdata.total_path_length = self.data['s'][self.end] - \
            self.data['s'][self.start]
        # self.logger.info("total number of planning data point: %d" %
        #                  (self.end - self.start))
        planningdata.total_path_time = self.data['time'][self.end] - \
            self.data['time'][self.start]
        planningdata.gear = 1
        planningdata.engage_advice.advice = \
            drive_state_pb2.EngageAdvice.READY_TO_ENGAGE

        for i in range(self.start, self.start+50):
            adc_point = pnc_point_pb2.TrajectoryPoint()
            adc_point.path_point.x = self.data['x'][i]
            adc_point.path_point.y = self.data['y'][i]
            adc_point.path_point.z = self.data['z'][i]
            adc_point.v = self.data['speed'][i] * self.speedmultiplier
            adc_point.a = self.data['acceleration'][i] * self.speedmultiplier
            adc_point.path_point.kappa = self.data['curvature'][i]
            adc_point.path_point.dkappa = self.data['curvature_change_rate'][i]
            adc_point.path_point.theta = self.data['theta'][i]
            adc_point.path_point.s = self.data['s'][i]


            time_diff = self.data['time'][i] - \
                self.data['time'][0]

            adc_point.relative_time = time_diff  - (
                now - self.starttime)

            planningdata.trajectory_point.extend([adc_point])

    

        planningdata.estop.is_estop = False
        #########plot debug"
        # print(planningdata.trajectory_point.path_point)
        # plt.plot(planningdata.trajectory_point.path_point.x,planningdata.trajectory_point.path_point.y)
        # plt.savefig("./plot rout")
        # print("save fig")
        self.planning_pub.write(planningdata)
        # self.logger.debug("Generated Planning Sequence: "
        #                   + str(self.sequence_num - 1))

    def publish_planningmsg_trajectory(self, trajectory, action_id):
        # print("trajectory.trajectory",trajectory.trajectory)
        if not self.localization_received:
            self.logger.warning(
                "localization not received yet when publish_planningmsg")
            return

        planningdata = planning_pb2.ADCTrajectory()
        now = cyber_time.Time.now().to_sec()
        planningdata.header.timestamp_sec = now
        planningdata.header.module_name = "planning"
        planningdata.header.sequence_num = self.sequence_num
        self.sequence_num = self.sequence_num + 1


        planningdata.total_path_length = self.data['s'][self.end] - \
            self.data['s'][self.start]
        # self.logger.info("total number of planning data point: %d" %
        #                 (self.end - self.start))
        planningdata.total_path_time = self.data['time'][self.end] - \
            self.data['time'][self.start]
        planningdata.gear = 1
        planningdata.engage_advice.advice = \
            drive_state_pb2.EngageAdvice.READY_TO_ENGAGE

        
        # to fix trajectory when continuously brakes
        if action_id == 0:
            if self.stop_count < 1:
                self.brake_trajectory = trajectory
                self.stop_count += 1
            else:
                self.stop_count += 1 
        
            trajectory = self.brake_trajectory
            start_time = 0.1 * self.stop_count
        else:
            self.brake_trajectory = None
            self.stop_count = 0
            start_time = 0


        
        for i in range(len(trajectory.trajectory)-2):
            adc_point = pnc_point_pb2.TrajectoryPoint()
            adc_point.path_point.x = trajectory.trajectory[i][0]
            adc_point.path_point.y = trajectory.trajectory[i][1]
            adc_point.path_point.z = 0
            adc_point.path_point.kappa = -trajectory.trajectory[i][6]*0.1 #curvature
            adc_point.path_point.dkappa = (trajectory.trajectory[i+1][6]-trajectory.trajectory[i][6])/0.1 #curvature_change_rate
            adc_point.v =  trajectory.trajectory[i][3]
            adc_point.a =  trajectory.trajectory[i][5]
            adc_point.path_point.theta =  trajectory.trajectory[i][2]
            adc_point.path_point.s = trajectory.trajectory[i][4]


            time_diff = 0.1*i - 0.1 -start_time
            adc_point.relative_time = time_diff 

            planningdata.trajectory_point.extend([adc_point])

        if pub_dns_map_data:
            dns_path =  pnc_point_pb2.Path()
            dns_path.name= "dns test path"
            xlist=458500+np.array([75.5 ,78.81,72,67.2,75.5])
            ylist=4400000+np.array([55.5,54.03,23,28.8,55.5])
            for i in range(len(xlist)):
                path_point = pnc_point_pb2.PathPoint()
                path_point.x = xlist[i]
                path_point.y = ylist[i]
                path_point.z = 0

                # path_point.kappa = self.data['curvature'][i]
                # path_point.dkappa = self.data['curvature_change_rate'][i]
                # path_point.theta = self.data['theta'][i]
                # path_point.s = self.data['s'][i]
                dns_path.path_point.extend([path_point])
                planningdata.debug.planning_data.path.extend([dns_path])

        if pub_dns_map_data:
            dns_path2 =  pnc_point_pb2.Path()
            dns_path2.name= "dns test path2"
            xlist = [458356.829,458363.37]
            ylist = [4400197.71, 4400212.257283324]
            for i in range(len(xlist)):
                path_point = pnc_point_pb2.PathPoint()
                path_point.x = xlist[i]
                path_point.y = ylist[i]
                path_point.z = 0

                dns_path2.path_point.extend([path_point])
                planningdata.debug.planning_data.path.extend([dns_path2])
        planningdata.estop.is_estop = False
    
        self.planning_pub.write(planningdata)
        # self.logger.debug("Generated Planning Sequence: "
        #                 + str(self.sequence_num - 1))

    def shutdown(self):
        """
        shutdown cyber
        """
        self.terminating = True
        self.logger.info("Shutting Down...")
        time.sleep(0.2)

    def quit(self, signum, frame):
        """surrouding_obs
        shutdown the keypress thread
        """
        sys.exit(0)

    def get_obs(self):
        obs = []
        ego_obs =[self.carx, self.cary, self.carvx, self.carvy, self.yaw] #FIXME: The angle should be modified
        if self.carx == 0 and self.cary == 0:
            return []
        # print("ego_yaw",self.carx, self.cary, self.yaw)
        obs.append(ego_obs)
        if self.obss is not None:  
            for i in range(len(self.obss)):
                obs_info = self.obss[i]
                surrouding_obs = [obs_info.position.x,
                                                    obs_info.position.y,
                                                    obs_info.velocity.x,
                                                    obs_info.velocity.y,
                                                    obs_info.theta,
                                                    obs_info.type]
                obs.append(surrouding_obs)
        return obs

class Werling_planner_SP():
    def  __init__(self,):
        self.trajectory_planner = JunctionTrajectoryPlanner_SP()
        self.dynamic_map = DynamicMap()
        self.dynamic_map.add_target_v_on_s(target_v_list)
        self.read_ref_path_from_file()

        # for traffic light
        _thread.start_new_thread(self.get_keyboard,())
        self.traffic_light_obs = True

        self.using_prdiction = True

        #dns_flag    0     1    2     3
        #                    [10, 15, 20, 25] 
        self.speed_index = 2 
    
        self.direction = 0
    
    def update_path(self, obs, done):# TODO: represent a obs
        if done or len(obs)<1:
            self.trajectory_planner.clear_buff(clean_csp=False)
            return None, None
        else:
            self.dynamic_map.update_map_from_list_obs(obs)
            if self.trajectory_planner.csp is not None and self.traffic_light_obs:
                self.dynamic_map.generate_intersection_obs(intersection_s_list, self.trajectory_planner.csp)
            """
            DNS_______________________________________________________________________________________
            """

            self.trajectory_planner.generate_low_level_trajectory_for_dns(v_target=dns_speed_list[self.speed_index], direction=self.direction, dynamic_map=self.dynamic_map)

            """
            DNS  end_______________________________________________________________________________________
            """
            if self.using_prdiction:
                trajectory_action, index = self.trajectory_planner.trajectory_update(self.dynamic_map)
            else:
                candidate_trajectories_tuple = self.trajectory_planner.generate_candidate_trajectories(self.dynamic_map)
                index = 2

            chosen_action_id = index
            chosen_trajectory = self.trajectory_planner.trajectory_update_CP(chosen_action_id)

            # print("[Action_ID]",chosen_action_id)
            # print("[Traffic_Light]",self.traffic_light_obs)

            return chosen_trajectory, chosen_action_id
    
    def read_ref_path_from_file(self):
        record_file = os.path.join(APOLLO_ROOT, 'data/log/garage4_length.csv')
        # record_file = os.path.join(APOLLO_ROOT, 'data/log/yizhuang_indside.csv')

        try:
            file_handler = open(record_file, 'r')
        except (FileNotFoundError, IOError) as ex:
            self.logger.error("Error opening {}: {}".format(record_file, ex))
            sys.exit(1)

        self.data = genfromtxt(file_handler, delimiter=',', names=True)
        file_handler.close()
        t_array = []
        self.ref_path = Lane()



        for i in range(0,len(self.data)//100): # The Apollo record data is too dense!
            lanepoint = Lanepoint()
            lanepoint.position.x = self.data['x'][i*90]
            lanepoint.position.y = self.data['y'][i*90]
            # print("ref path", lanepoint.position.x, lanepoint.position.y)
            self.ref_path.central_path.append(lanepoint)
            t_array.append(lanepoint)
        self.ref_path.central_path_array = np.array(t_array)
        self.ref_path.speed_limit = 60/3.6 # m/s
        self.dynamic_map.update_ref_path_from_routing(self.ref_path) 


    def get_keyboard(self):
        while(1):
            keyboard_input = input("Keyboard_Init")
            print("Received Key board Signal----------------",keyboard_input)
            if keyboard_input == "c":
                if self.traffic_light_obs:
                    self.traffic_light_obs = False
                    print("Traffic_light:False")
                else:
                    self.traffic_light_obs = True
                    print("Traffic_light:True")
            if keyboard_input == "w":
                if  self.speed_index< len(dns_speed_list)-1:
                    self.speed_index = self.speed_index +1
                    print("DNS::speed up  up up up, target speed:",dns_speed_list[self.speed_index]*3.6 ) #
                else:
                    print("DNS::Reach Max speed!",dns_speed_list[self.speed_index]*3.6 )

            if keyboard_input == "s":
                if  self.speed_index>0:
                    self.speed_index = self.speed_index -1
                    print("DNS::speed down  down down down, target speed:",dns_speed_list[self.speed_index]*3.6 )
                else:
                    print("DNS::Reach Min speed!",dns_speed_list[self.speed_index]*3.6 )

            if keyboard_input == "a":
                if   self .direction>-1:
                    self .direction = self .direction - 1
                    print("DNS::Turn Left  Left Left  Left" )

            if keyboard_input == "d":
                if self.direction <1:
                    self .direction = self .direction+1
                    print("DNS::Turn Right  Right Right Right" )


            

def main():
    """
    Main cyber
    """

    node = cyber.Node("rtk_player")

    Logger.config(
        log_file=os.path.join(APOLLO_ROOT, 'data/log/rtk_player.log'),
        use_stdout=True,
        log_level=logging.DEBUG)

    record_file = os.path.join(APOLLO_ROOT, 'data/log/garage4_length.csv')
    # record_file = os.path.join(APOLLO_ROOT, 'data/log/yizhuang1.csv')


    player = RtkPlayer(record_file, node)
    
    planner = Werling_planner_SP()

    atexit.register(player.shutdown)

    node.create_reader('/apollo/canbus/chassis', chassis_pb2.Chassis,
                       player.chassis_callback)

    node.create_reader('/apollo/localization/pose',
                       localization_pb2.LocalizationEstimate,
                       player.localization_callback)
    
    node.create_reader('/apollo/prediction', 
                        prediction_obstacle_pb2.PredictionObstacles,
                        player.prediction_callback)
    
    
    node.create_reader('/apollo/perception/obstacles',
                        perception_obstacle_pb2.PerceptionObstacles,
                        player.perception_callback)

    while not cyber.is_shutdown():
        now = cyber_time.Time.now().to_sec()
        # New add
        obs =  player.get_obs()
        trajectory, action_id = planner.update_path(obs, done=0)
        if trajectory is not None:
            player.publish_planningmsg_trajectory(trajectory, action_id)
        

        # player.publish_planningmsg()
        # player.publish_planningmsg_start()
        sleep_time = 0.1 - (cyber_time.Time.now().to_sec() - now)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == '__main__':
    cyber.init()
    main()



    # plt.plot(adc_point.path_point.x,adc_point.path_point.y)

    cyber.shutdown()
