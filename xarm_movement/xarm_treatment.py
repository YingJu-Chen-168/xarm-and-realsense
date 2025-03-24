# relative motion
import sys
import math
import time
import queue
import datetime
import random
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI
import cv2

class RobotTreatment(object):
    """Robot Main Class"""
    def __init__(self, robot, width, repeat_time, **kwargs):
        self.alive = True
        self._arm = robot
        self._width = width
        self._repeat_time = repeat_time
        self._tcp_speed = 50
        self._tcp_acc = 500
        self._angle_speed = 20
        self._angle_acc = 500
        self._M2_diameter = 2
        self._vars = {}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    def check_plasma_cycle(self, accumulated_plasma_cycle):
        self._tcp_speed = 30
        rest_time = 11
        plasma_space = 105
        countdown = rest_time - (plasma_space / self._tcp_speed) - 1
        try:
            if accumulated_plasma_cycle % 30 == 0:
                time.sleep(0.2)
                code = self._arm.set_position(y = plasma_space, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                if not self._check_code(code, 'set_position'):
                    return
                time.sleep(countdown)
            elif accumulated_plasma_cycle % 30 == 15:
                code = self._arm.set_position(y = -plasma_space, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                if not self._check_code(code, 'set_position'):
                    return
                time.sleep(countdown)
            else:
                time.sleep(rest_time)
            self._tcp_speed = 50
        except Exception as e:
            self.pprint('MainException: {}'.format(e))

    # Robot Main Run
    def run(self):
        try:
            self._tcp_speed = 50
            loop = 2 * int(self._repeat_time)   # 看要不要改成 *2，讓路徑比較貼合真實範圍
            working_diatance = 153            
            accumulated_distance = 0
            accumulated_plasma_cycle = 0
            for i in range(loop):
                if not self.is_alive:
                    break
                ##
                if i % 4 == 0:
                    remain_diatance = working_diatance - accumulated_distance
                    if remain_diatance >= self._width:
                        code = self._arm.set_position(y = -self._width, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_distance = accumulated_distance + self._width   
                    elif remain_diatance < self._width:
                        code = self._arm.set_position(y = -remain_diatance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)   
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_plasma_cycle = accumulated_plasma_cycle + 1
                        self.check_plasma_cycle(accumulated_plasma_cycle)
                        # 加上手臂位移
                        if (self._width - remain_diatance) >= working_diatance:
                            repeat = self._width // working_diatance
                            for i in range(repeat):
                                code = self._arm.set_position(y = -working_diatance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)
                                if not self._check_code(code, 'set_position'):
                                    return
                                accumulated_plasma_cycle = accumulated_plasma_cycle + 1
                                self.check_plasma_cycle(accumulated_plasma_cycle)
                            accumulated_distance = self._width - remain_diatance - repeat * working_diatance   
                            code = self._arm.set_position(y = -accumulated_distance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                            if not self._check_code(code, 'set_position'):
                                return                     
                        elif (self._width - remain_diatance) < working_diatance:
                            accumulated_distance = self._width - remain_diatance
                            code = self._arm.set_position(y = -accumulated_distance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                            if not self._check_code(code, 'set_position'):
                                return                
                ##
                elif i % 4 == 1:
                    remain_diatance = working_diatance - accumulated_distance
                    if remain_diatance >= self._M2_diameter:
                        code = self._arm.set_position(x = - self._M2_diameter, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_distance = accumulated_distance + self._M2_diameter   
                    elif remain_diatance < self._M2_diameter:
                        code = self._arm.set_position(x = - remain_diatance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)   
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_plasma_cycle = accumulated_plasma_cycle + 1
                        self.check_plasma_cycle(accumulated_plasma_cycle)
                        accumulated_distance = self._M2_diameter - remain_diatance  
                        code = self._arm.set_position(x = -accumulated_distance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)
                        if not self._check_code(code, 'set_position'):
                            return     
                ## 
                elif i % 4 == 2:
                    remain_diatance = working_diatance - accumulated_distance
                    if remain_diatance >= self._width:
                        code = self._arm.set_position(y = self._width, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_distance = accumulated_distance + self._width   
                    elif remain_diatance < self._width:
                        code = self._arm.set_position(y = remain_diatance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True) 
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_plasma_cycle = accumulated_plasma_cycle + 1
                        self.check_plasma_cycle(accumulated_plasma_cycle)
                        if (self._width - remain_diatance) >= working_diatance:
                            repeat = self._width // working_diatance
                            for i in range(repeat):
                                code = self._arm.set_position(y = working_diatance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)
                                if not self._check_code(code, 'set_position'):
                                    return
                                accumulated_plasma_cycle = accumulated_plasma_cycle + 1
                                self.check_plasma_cycle(accumulated_plasma_cycle)
                            accumulated_distance = self._width - remain_diatance - repeat * working_diatance   
                            code = self._arm.set_position(y = accumulated_distance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                            if not self._check_code(code, 'set_position'):
                                return                     
                        elif (self._width - remain_diatance) < working_diatance:
                            accumulated_distance = self._width - remain_diatance
                            code = self._arm.set_position(y = accumulated_distance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                            if not self._check_code(code, 'set_position'):
                                return              
                ##
                elif i % 4 == 3:
                    remain_diatance = working_diatance - accumulated_distance
                    if remain_diatance > self._M2_diameter:
                        code = self._arm.set_position(x = - self._M2_diameter, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)                            
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_distance = accumulated_distance + self._M2_diameter   
                    elif remain_diatance <= self._M2_diameter:
                        code = self._arm.set_position(x = - remain_diatance, radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)    
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_plasma_cycle = accumulated_plasma_cycle + 1
                        self.check_plasma_cycle(accumulated_plasma_cycle)
                        code = self._arm.set_position(x = - (self._M2_diameter - remain_diatance), radius=-1, speed=self._tcp_speed, mvacc=self._tcp_acc, relative=True, wait=True)
                        if not self._check_code(code, 'set_position'):
                            return
                        accumulated_distance = self._M2_diameter - remain_diatance  
        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'release_count_changed_callback'):
            self._arm.release_count_changed_callback(self._count_changed_callback)

def treatment_run(repeat, width):
    arm = XArmAPI('192.168.1.222', baud_checkset=False)
    robot_main = RobotTreatment(arm, repeat, width)
    robot_main.run()