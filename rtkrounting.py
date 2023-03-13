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


## local package
from Agent.zzz.dynamic_map import DynamicMap
from Agent.zzz.frenet import Frenet_path
from Agent.zzz.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.zzz.dynamic_map import Lane, Lanepoint, Vehicle
from scipy.spatial.transform import Rotation as R
from modules.routing.proto import routing_pb2
# TODO(all): hard-coded path temporarily. Better approach needed.
APOLLO_ROOT = "/apollo"

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
        
        self.qx = self.localization.pose.orientation.qx
        self.qy = self.localization.pose.orientation.qy
        self.qz = self.localization.pose.orientation.qz
        self.qw = self.localization.pose.orientation.qw
       
        self.localization_received = True
        self.yaw =  self.xyzw2yaw()

    def  xyzw2yaw(self,):
        Rq=[self.qx, self.qy, self.qz, self.qw]
        r = R.from_quat(Rq)
        Rm = r.as_matrix()
        euler0 = r.as_euler('xyz', degrees=True)
        return euler0[2]

    def prediction_callback(self, data):
        self.prediction.CopyFrom(data)

    def routing(self,data): # dns Fixme add routing
        self.routing.CopyFrom(data)

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
        self.logger.info("total number of planning data point: %d" %
                         (self.end - self.start))
        planningdata.total_path_time = self.data['time'][self.end] - \
            self.data['time'][self.start]
        planningdata.gear = 1
        planningdata.engage_advice.advice = \
            drive_state_pb2.EngageAdvice.READY_TO_ENGAGE

        for i in range(self.start, self.end//100):
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
        self.logger.debug("Generated Planning Sequence: "
                          + str(self.sequence_num - 1))

    def publish_planningmsg_trajectory(self, trajectory):
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
                self.logger.info("total number of planning data point: %d" %
                                (self.end - self.start))
                planningdata.total_path_time = self.data['time'][self.end] - \
                    self.data['time'][self.start]
                planningdata.gear = 1
                planningdata.engage_advice.advice = \
                    drive_state_pb2.EngageAdvice.READY_TO_ENGAGE

                for i in range(len(trajectory.trajectory)-1):
                    adc_point = pnc_point_pb2.TrajectoryPoint()
                    adc_point.path_point.x = trajectory.trajectory[i][0]
                    adc_point.path_point.y = trajectory.trajectory[i][1]
                    adc_point.path_point.z = 0
                    adc_point.v =  trajectory.trajectory[i][3]
                    adc_point.a =  trajectory.trajectory[i][5]
                    adc_point.path_point.kappa = 0
                    adc_point.path_point.dkappa =0
                    adc_point.path_point.theta =  trajectory.trajectory[i][2]
                    adc_point.path_point.s = trajectory.trajectory[i][4]


                    time_diff = self.data['time'][i] - \
                        self.data['time'][0]

                    adc_point.relative_time = time_diff  - (
                        now - self.starttime)

                    planningdata.trajectory_point.extend([adc_point])

                planningdata.estop.is_estop = False

                self.planning_pub.write(planningdata)
                self.logger.debug("Generated Planning Sequence: "
                                + str(self.sequence_num - 1))

    def shutdown(self):
        """
        shutdown cyber
        """
        self.terminating = True
        self.logger.info("Shutting Down...")
        time.sleep(0.2)

    def quit(self, signum, frame):
        """
        shutdown the keypress thread
        """
        sys.exit(0)

    def get_obs(self):
        obs =[ [self.carx, self.cary, self.carvx, self.carvy, 1.5],[0,0,0,0,0]] #FIXME: The angle should be modified
        print("obs",obs)
        return obs

# Werling Planner Init
class Werling_planner():
    def  __init__(self,):
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.dynamic_map = DynamicMap()
        self.read_ref_path_from_file()
    
    def update_path(self, obs, done):# TODO: represent a obs
        if done or obs[0][0]==0:
            self.trajectory_planner.clear_buff(clean_csp=False)
        else:
            self.dynamic_map.update_map_from_list_obs(obs)

            candidate_trajectories_tuple = self.trajectory_planner.generate_candidate_trajectories(self.dynamic_map)
            chosen_action_id = 3
            chosen_trajectory = self.trajectory_planner.trajectory_update_CP(chosen_action_id)

            return chosen_trajectory
    
    def read_ref_path_from_file(self):
        record_file = os.path.join(APOLLO_ROOT, 'data/log/garage.csv')
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



def main():
    """
    Main cyber
    """

    node = cyber.Node("rtk_player")

    Logger.config(
        log_file=os.path.join(APOLLO_ROOT, 'data/log/rtk_player.log'),
        use_stdout=True,
        log_level=logging.DEBUG)

    record_file = os.path.join(APOLLO_ROOT, 'data/log/garage.csv')

    player = RtkPlayer(record_file, node)
    
    planner = Werling_planner()

    atexit.register(player.shutdown)

    node.create_reader('/apollo/canbus/chassis', chassis_pb2.Chassis,
                       player.chassis_callback)

    node.create_reader('/apollo/localization/pose',
                       localization_pb2.LocalizationEstimate,
                       player.localization_callback)
    node.create_reader('/apollo/prediction', 
                        prediction_obstacle_pb2.PredictionObstacles,
                        player.prediction_callback)
    node.create_reader('/apollo/routing', 
                        routing_pb2.RoutingResponse,
                        player.routing_callback)



    while not cyber.is_shutdown():
        now = cyber_time.Time.now().to_sec()

        # New add
        obs =  player.get_obs()
        trajectory = planner.update_path(obs, done=0)
        if trajectory is not None:
            player.publish_planningmsg_trajectory(trajectory)
        
        
        # player.publish_planningmsg()
        sleep_time = 0.1 - (cyber_time.Time.now().to_sec() - now)
        if sleep_time > 0:
            time.sleep(sleep_time)


if __name__ == '__main__':
    cyber.init()
    main()
    cyber.shutdown()
