import sys
import random
import numpy as np
from collections import deque

from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.models import load_model
from keras.callbacks import LambdaCallback

# from keras.callbacks import TensorBoard
from keras import optimizers

import matplotlib.pyplot as plt
import copy
import test_algorithm as ta
import datetime
import os
from SumTree import SumTree
from time import time


# from Graph import Graph_simple
from Graph import Graph_simple_100
from Graph import Graph_jeju
from Graph import Graph_jejusi
from Graph import Graph_simple_39

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


EPISODES = 2000
N_REQ = 150
EPS_DC = 0.9994
UNITtimecost = 8
ECRate = 0.16
Step_SOC = 0.15
Base_SOC = 0.6
Final_SOC = 0.9
N_SOC = int((Final_SOC-Base_SOC)/Step_SOC)+1
TRAIN = True
NCS= 5
# TRAIN = False

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

def default_setting_random_value():
    t_start = np.random.uniform(0, 1200)
    soc = np.random.uniform(0.2, 0.4)
    while soc <= 0.0 or soc > 1.0:
        soc = np.random.uniform(0.2, 0.4)
    return t_start, soc

def reset_CS_info(graph):
    CS_list = []
    profit = np.random.uniform(0.7, 1.3, len(graph.cs_info))
    # for l in graph.cs_info:
    for i, l in enumerate(graph.cs_info):
        cs = CS(l, profit[i], graph.cs_info[l]['long'], graph.cs_info[l]['lat'])
        CS_list.append(cs)
    return CS_list

class CPOLE:
    def __init__(self, id, csid, price):
        self.id = id
        self.csid = csid
        self.avail_time = 0.0
        self.charging_ev = []
        self.curr_charging_ev = None
        self.chargingpower = 60 #kw
        self.price = price

    def update_cpole(self, cur_time):
        if self.avail_time<cur_time:
            self.avail_time = cur_time
            self.curr_charging_ev = None

    def set_charging(self, ev):
        # print('set charging EV:{} CS:{}, CP:{}'.format(ev.id, self.csid, self.id))
        charging_energy = ev.maxBCAPA * (ev.req_SOC - ev.curr_SOC)
        charging_duration = (charging_energy/(self.chargingpower * ev.charging_effi))
        ev.cscharingenergy = charging_energy
        ev.true_charging_duration = charging_duration*60

        self.charging_ev.append(ev)
        self.curr_charging_ev = ev
        # print("  1===setCharging AvailTime:", self.avail_time)
        if self.avail_time < ev.true_arrtime:
            ev.true_waitingtime = 0.0
            ev.cschargingstarttime = ev.true_arrtime + ev.true_waitingtime
            self.avail_time = ev.true_arrtime + ev.true_charging_duration
        else:
            ev.true_waitingtime = self.avail_time - ev.true_arrtime
            ev.cschargingstarttime = self.avail_time
            self.avail_time = self.avail_time + ev.true_charging_duration

        ev.time_diff_WT = ev.true_waitingtime - ev.ept_waitingtime
        ev.curr_time = ev.cschargingstarttime + ev.true_charging_duration
        ev.charging_finish_time = ev.curr_time
        ev.curr_SOC = ev.req_SOC
        ev.cschargingprice = self.price[int(ev.cschargingstarttime%1440/5)]
        # print("  2===setCharging AvailTime:", self.avail_time)
        # print("  3===setCharging EVID: {0:.2f}  truWT: {1:.2f}  truChDur: {2:.2f}".format(ev.id, ev.true_waitingtime, ev.true_charging_duration))



class CS:
    def __init__(self, node_id, profit, long, lat):
        self.id = node_id
        self.price = list()
        self.waittime = list()
        self.chargingpower = 60 # kw
        self.homechargingpower = 3.3
        self.num_sorket = 2
        self.cplist = []



        self.charing_ev = [] # pair (pev, start, endtime)
        self.waiting_ev = []
        self.reserve_ev = [] # pair (pev, arrtime, endtime)
        self.x = long
        self.y = lat
        self.profit = profit
        self.TOU_price = [0.15575 for i in range(24)]

        for i in range(288):
            p = np.random.normal(self.TOU_price[int(i/12)], 0.30 * self.TOU_price[int(i/12)])
            while p < 0:
                p = np.random.normal(self.TOU_price[int(i/12)], 0.30 * self.TOU_price[int(i/12)])
            self.price.append(p*self.profit)

        for i in range(self.num_sorket):
            cp = CPOLE(i, self.id, self.price)
            self.cplist.append(cp)

    def update_avail_time(self, cur_time, graph):#현재시간을 기준으로 도착한 예약들을 처리.
        self.reserve_ev.sort(key=lambda element: element[2])
        for rev, eptTarr, _ in self.reserve_ev:
            # print('befor reserve_ev list CS ID:', self.id, 'reseved ev:', rev.id, 'eptTarr:', eptTarr, 'TruTarr:', rev.true_arrtime,'curTime:', cur_time)
            if rev.true_arrtime <= cur_time:
                self.waiting_ev.append((rev, rev.true_arrtime))
                # print(' ', self.reserve_ev)
                self.reserve_ev = self.reserve_ev[1:]
                # print(' ', self.reserve_ev)

        # for rev, eptTarr, _ in self.reserve_ev:
            # print('after reserve_ev list CS ID:', self.id, 'reseved ev:', rev.id, 'eptTarr:', eptTarr, 'TruTarr:', rev.true_arrtime,'curTime:', cur_time)

        self.waiting_ev.sort(key=lambda e: e[1])
        tmpcp = []
        for cp in self.cplist:
            tmpcp.append((cp, cp.avail_time))

        for i, tp_wev in enumerate(self.waiting_ev):
            tmpcp.sort(key=lambda element: element[1])
            cp, at = tmpcp[0]
            wev, true_arrtime = tp_wev
            cp.set_charging(wev)
            ta.finishi_trip(wev, cp, graph)
            tmpcp[0] = cp, cp.avail_time

        self.waiting_ev = []



    def get_ept_avail_time(self, ept_arrtime): # 예상 정보를 가지고
        self.reserve_ev.sort(key=lambda element: element[1])
        tmp_rsv = copy.deepcopy(self.reserve_ev)
        tmp_wtev_list = []
        ept_extedWT_list = []
        for rev, eptTarr,_ in tmp_rsv:
            # print('CS ID:',self.id, 'reseved ev:', rev.id, 'eptTarr:', eptTarr,'TruTarr:', rev.true_arrtime)
            if eptTarr <= ept_arrtime:
                tmp_wtev_list.append((rev, eptTarr))
            else:
                # print('need to recalculate for charged WT')
                ept_extedWT_list.append((rev, eptTarr))

        tmp_wtev_list.sort(key=lambda e:e[1])

        tmp_atime = []
        for cp in self.cplist:
            tmp_atime.append(cp.avail_time)

        for i, tp_wev in enumerate(tmp_wtev_list):
            tmp_atime.sort()
            at = tmp_atime[0]
            wev, eptTarr = tp_wev
            if at < wev.ept_arrtime:
                eptWT = 0.0
                # wev.eptcschargingstarttime = wev.ept_arrtime + eptWT
                at = wev.ept_arrtime + wev.ept_charging_duration
            else:
                eptWT = at - wev.ept_arrtime
                # wev.eptcschargingstarttime = wev.ept_arrtime + eptWT
                at = at + wev.ept_charging_duration
            tmp_atime[0] = at


        # tmp_exted_atime = copy.deepcopy(tmp_atime)
        # for i, te_wev in enumerate(ept_extedWT_list):
        #     tmp_exted_atime.sort()
        #     at = tmp_exted_atime[0]
        #     wev, eptTarr = te_wev
        #     if at < wev.ept_arrtime:
        #         eptWT = 0.0
        #         eptChstart = wev.ept_arrtime + eptWT
        #         wev.eptcschargingstarttime = wev.ept_arrtime + eptWT
        #         at = wev.ept_arrtime + wev.ept_charging_duration
        #     else:
        #         eptWT = at - wev.ept_arrtime
        #         eptChstart = wev.ept_arrtime + eptWT
        #         wev.eptcschargingstarttime = wev.ept_arrtime + eptWT
        #         at = at + wev.ept_charging_duration
        #     tmp_exted_atime[0] = at



        return tmp_atime

    def recieve_request(self, ev):
        self.reserve_ev.append((ev, ev.ept_arrtime, ev.true_arrtime))

        self.reserve_ev.sort(key=lambda e:e[1])

        tmp_wtev_list = []
        ept_extedWT_list = []

        for rev, eptTarr, _ in self.reserve_ev:
            # print('CS ID:',self.id, 'reseved ev:', rev.id, 'eptTarr:', eptTarr,'TruTarr:', rev.true_arrtime)
            if eptTarr <= ev.ept_arrtime:
                tmp_wtev_list.append((rev, eptTarr))
            else:
                # print('need to recalculate for charged WT')
                ept_extedWT_list.append((rev, eptTarr))

        diff_ch_start = []
        tmp_atime = []
        for cp in self.cplist:
            tmp_atime.append(cp.avail_time)

        for i, (wev, _, _) in enumerate(self.reserve_ev):
            tmp_atime.sort()
            at = tmp_atime[0]

            if at < wev.ept_arrtime:
                eptWT = 0.0
                ept_ch_start = wev.ept_arrtime + eptWT
                diff = ept_ch_start - wev.eptcschargingstarttime
                diff_ch_start.append(diff)
                # wev.eptcschargingstarttime = ept_ch_start
                at = wev.ept_arrtime + wev.ept_charging_duration
            else:
                eptWT = at - wev.ept_arrtime
                ept_ch_start = wev.ept_arrtime + eptWT
                diff = ept_ch_start - wev.eptcschargingstarttime
                diff_ch_start.append(diff)
                # wev.eptcschargingstarttime = ept_ch_start
                at = at + wev.ept_charging_duration
            tmp_atime[0] = at

        # for i, (rev, _, _) in enumerate(self.reserve_ev):
        #     print(ev.curr_time, rev.id, diff_ch_start[i])

        return sum(diff_ch_start)



    def sim_finish(self, graph):
        self.update_avail_time(3000, graph)

    def get_ept_WT(self, ept_arrtime, cur_time, graph):

        self.reserve_ev.sort(key=lambda element: element[1])
        self.update_avail_time(cur_time, graph)
        at_list = self.get_ept_avail_time(ept_arrtime)  # charging finish time for ev being charged

        at_list.sort()
        minAT = at_list[0]
        if minAT > ept_arrtime:
            eptWT = minAT - ept_arrtime
        else:
            eptWT = 0

        return eptWT



    def get_price(self, sim_time):
        return self.price[int(sim_time/5)]

class EV:
    def __init__(self, id,  source, destination, soc, reqsoc, t_start):
        self.id = id
        self.t_start = t_start
        self.curr_time = t_start
        self.curr_day = 0

        self.charging_effi = 0.9
        self.curr_SOC = soc
        self.init_SOC = soc
        self.final_soc = reqsoc
        self.req_SOC = reqsoc
        self.before_charging_SOC=0.0
        self.source = source
        self.destination = destination
        self.maxBCAPA= 50  # kw
        self.curr_location = source
        self.next_location = source
        # self.ECRate = 0.2 # kwh/km
        self.ECRate = ECRate # kwh/km
        self.charged = 0
        self.cs = None
        self.csid = -1

        self.totalenergyconsumption = 0.0
        self.totaldrivingdistance = 0.0
        self.totaldrivingtime = 0.0
        self.expense_cost_part = 0.0
        self.expense_time_part = 0.0
        self.totalcost = 0.0

        self.true_arrtime = 0.0
        self.true_waitingtime = 0.0
        self.true_charging_duration = 0.0
        self.ept_arrtime = 0.0
        self.ept_waitingtime = 0.0
        self.ept_charging_duration = 0.0

        self.charging_finish_time = 0.0

        self.time_diff_WT = 0.0

        self.cschargingtime = 0.0
        self.cschargingcost = 0.0
        self.cschargingwaitingtime = 0.0
        self.cscharingenergy = 0.0
        self.cschargingprice = 0.0
        self.cschargingstarttime = 0.0
        self.eptcschargingstarttime = 0.0
        self.csdistance = 0
        self.csdrivingtime = 0
        self.cssoc = 0

        self.homechargingtime = 0.0
        self.homechargingcost = 0.0
        self.homechargingenergy = 0.0
        self.homechargingprice = 0.0
        self.homechargingstarttime = 0.0
        self.homedrivingdistance = 0.0
        self.homedrivingtime = 0.0
        self.homesoc = 0.0

        self.fdist=0
        self.rdist=0
        self.path=[]
        self.front_path = []
        self.rear_path = []

        self.predic_totaltraveltime = 0.0

        self.weightvalue = 0.0


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)



# 카트폴 예제에서의 DQN 에이전트



class DQNAgent:
    def __init__(self, state_size, action_size, load_state):
        self.render = False
        self.load_model = load_state



        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = EPS_DC
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 5000
        self.B = 30

        # 리플레이 메모리, 최대 크기 2000
        self.memory_size = 40000
        # self.memory = deque(maxlen=4000)
        self.memory = Memory(self.memory_size)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()
        # log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # self.tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if self.load_model:
            # self.model.load_weights("dqn_model.h5")
            self.model = load_model("dqn_model.h5")
            self.epsilon = 0.0

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        print('in agent', state_size, action_size)
        model = Sequential()
        model.add(Dense(200, input_dim=self.state_size, kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(100, kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        # model.add(Dense(400, kernel_initializer='he_uniform'))
        # # model.add(BatchNormalization())
        # model.add(Activation('relu'))
        #
        # model.add(Dense(200, kernel_initializer='he_uniform'))
        # # model.add(BatchNormalization())
        # model.add(Activation('relu'))

        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model



    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            # print(action)
        else:
            # state = np.reshape(state, [1, self.state_size])
            q_value = self.model.predict(state)
            action = np.argmax(q_value[0])
            # print()
            # print(action, q_value)
        return action

    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        if self.epsilon == 1:
            done = True

            # TD-error 를 구해서 같이 메모리에 저장
        target = self.model.predict([state])
        old_val = target[0][action]
        target_val = self.target_model.predict([next_state])
        if done:
            target[0][action] = reward
        else:
            target[0][action] = reward + self.discount_factor * (
                np.amax(target_val[0]))
        error = abs(old_val - target[0][action])

        # self.memory.append((state, action, reward, next_state, done))
        self.memory.add(error, (state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        # if self.epsilon > self.epsilon_min:
        #     self.epsilon *= self.epsilon_decay

            # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = self.memory.sample(self.batch_size)

        errors = np.zeros(self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][1][0]
            actions.append(mini_batch[i][1][1])
            rewards.append(mini_batch[i][1][2])
            next_states[i] = mini_batch[i][1][3]
            dones.append(mini_batch[i][1][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            old_val = target[i][actions[i]]
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))
            # TD-error를 저장
            errors[i] = abs(old_val - target[i][actions[i]])

        # TD-error로 priority 업데이트
        for i in range(self.batch_size):
            idx = mini_batch[i][0]
            self.memory.update(idx, errors[i])

        # self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0, callbacks=[self.tensorboard])
        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)




class Env:
    def __init__(self, graph, state_size, action_size):
        self.graph = graph
        self.source_node_list = list(self.graph.source_node_set)
        self.destination_node_list = list(self.graph.destination_node_set)
        self.num_request = N_REQ
        self.request_be_EV = []
        self.request_ing_EV = []
        self.request_ed_EV = []

        self.path_info = []
        self.sim_time=0
        self.CS_list = []
        self.pev = None

        # self.target = -1
        self.state_size = state_size
        self.action_size = action_size


        self.CS_list = reset_CS_info(self.graph)





    def reset(self):
        # self.graph.reset_traffic_info()
        self.CS_list = reset_CS_info(self.graph)
        self.request_be_EV = init_request(self.num_request, self.graph)


        self.pev = self.request_be_EV[0]
        # self.request_be_EV = self.request_be_EV[1:]
        # self.request_ing_EV.append(self.pev)

        self.path_info = []
        self.sim_time = self.pev.t_start
        self.timeIDX = int(self.sim_time / 5)

        self.path_info = ta.get_feature_state_fleet(self.sim_time, self.pev, self.CS_list, self.graph, NCS)

        state = [self.pev.curr_location / self.graph.num_node, self.pev.destination / self.graph.num_node, self.pev.curr_SOC, (self.pev.t_start/288)/288]

        for path in self.path_info:
            (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
             ept_front_d_time,
             ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
             ept_home_charging_cost, ept_arrtime) = path

            for cp in cs.cplist:
                state += [(cp.avail_time-self.pev.curr_time)/60]
            # state += [ept_WT/60, ept_charduration/60, ept_driving_cost/10, ept_cs_charging_cost/10]
        state = np.reshape(state, [1, self.state_size])

        return state

    def test_reset(self, EV_list, CS_list, graph):

        self.CS_list = CS_list
        self.request_be_EV = EV_list
        self.graph = graph

        self.pev = self.request_be_EV[0]
        self.path_info = []
        self.sim_time = self.pev.t_start
        self.timeIDX = int(self.sim_time / 5)

        self.path_info = ta.get_feature_state_fleet(self.sim_time, self.pev, self.CS_list, self.graph, NCS)

        state = [self.pev.curr_location / self.graph.num_node, self.pev.destination / self.graph.num_node,
                 self.pev.curr_SOC, (self.pev.t_start / 288) / 288]

        for path in self.path_info:
            (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
             ept_front_d_time,
             ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
             ept_home_charging_cost, ept_arrtime) = path

            for cp in cs.cplist:
                state += [(cp.avail_time - self.pev.curr_time) / 60]
            # state += [ept_WT / 60, ept_charduration / 60, ept_driving_cost / 10, ept_cs_charging_cost / 10]
        state = np.reshape(state, [1, self.state_size])

        return state


    def step(self, action):
        (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, ept_front_d_time,
         ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost,
         ept_home_charging_cost, ept_arrtime) = self.path_info[action]

        pev = self.pev
        pev.front_path = front_path
        pev.rear_path = rear_path
        pev.path = front_path + rear_path[1:]
        pev.ept_arrtime = ept_arrtime
        pev.true_arrtime = ta.get_true_arrtime(pev, cs, self.graph)
        pev.ept_waitingtime = ept_WT
        pev.eptcschargingstarttime = pev.ept_arrtime + ept_WT
        pev.ept_charging_duration = ept_charduration
        pev.cs = cs
        diff_ch = cs.recieve_request(pev)

        # reward = diff_ch
        reward = -weight
        # reward = -pev.ept_waitingtime/10
        done = 0

        if pev.id+1<N_REQ:
            next_pev = self.request_be_EV[pev.id+1]
            self.path_info = ta.get_feature_state_fleet(next_pev.t_start, next_pev, self.CS_list, self.graph, 0)
            next_state = [next_pev.curr_location / self.graph.num_node,
                          next_pev.destination / self.graph.num_node, next_pev.curr_SOC, (next_pev.t_start/288)/288]
            for path in self.path_info:
                (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                 ept_front_d_time, ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration,
                 ept_cs_charging_cost, ept_home_charging_cost, ept_arrtime) = path

                for cp in cs.cplist:
                    next_state += [(cp.avail_time-pev.curr_time)/60]

                # next_state += [ept_WT/60, ept_charduration/60, ept_driving_cost/10, ept_cs_charging_cost/10]
            next_state = np.reshape(next_state, [1, self.state_size])
            return next_state, next_pev, reward, done

        else:

            for cs in self.CS_list:
                cs.sim_finish(self.graph)
            tot_wt = 0
            tot_cost = 0
            for pev in self.request_be_EV:
                tot_wt += pev.true_waitingtime
                tot_cost += pev.totalcost
            done = 1
            reward = -(tot_cost/len(self.request_be_EV))
            return np.zeros((1, self.state_size)), -1, reward, done

def init_request(num_request, graph):
    request_be_EV = []

    source_node_list = list(graph.source_node_set)
    destination_node_list = list(graph.destination_node_set)

    request_time = np.random.uniform(360, 1200, num_request)  # 06:00 ~ 20:00
    request_time.sort()

    for i in range(num_request):
        s = source_node_list[np.random.randint(0, len(source_node_list) - 1)]
        d = destination_node_list[np.random.randint(0, len(destination_node_list) - 1)]
        soc = np.random.uniform(0.2, 0.4)
        req_soc = np.random.uniform(0.7, 0.9)
        t_start = request_time[i]
        request_be_EV.append(EV(i, s, d, soc, req_soc, t_start))

    return request_be_EV


def gen_test_envir_simple(graph):

    graph.reset_traffic_info()
    CS_list = reset_CS_info(graph)
    EV_list = init_request()

    return EV_list, CS_list, graph

def test_dqn(EV_list_DQN_REF, CS_list_DQN_REF, graph, env, agent ):

    agent.epsilon = 0
    episcore = 0
    done = False
    state = env.test_reset(EV_list_DQN_REF, CS_list_DQN_REF, graph)
    print("\nEpi:", e, agent.epsilon)


    while not done:
        action = agent.get_action(state)
        next_state, next_node, reward, done = env.step(action)

        state = next_state
        env.pev = next_pev







if __name__ == "__main__":
    # npev = 39
    # EV_list, CS_list, graph = gen_test_envir_simple(npev)

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02} {5} {6} {7}'.format(now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second,TRAIN, EPISODES, EPS_DC)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    graph_train = Graph_jejusi()
    graph_test = Graph_jejusi()

    action_size = len(graph_train.cs_info)
    state_size = action_size*2+4

    print('S:{} A:{}'.format(state_size, action_size))



    if TRAIN:
        episcores, episodes, eptscores, finalscores = [], [], [], []
        n_step = 0
        train_step = 0
        agent = DQNAgent(state_size, action_size, False)
        step=  0
        for e in range(EPISODES):
            episcore = 0
            epistep = 0
            env = Env(graph_train, state_size, action_size)
            print("\nEpi:", e, agent.epsilon, 'episcore:', episcore)
            done = False
            final_score = 0
            ept_score = 0
            state = env.reset()
            while not done:
                print(env.pev.id, end=' ')
                action = agent.get_action(state)
                next_state, next_pev, reward, done = env.step(action)
                agent.append_sample(state, action, reward, next_state, done)
                step += 1

                if step >= agent.train_start:
                    agent.train_model()
                    train_step += 1


                state = next_state
                env.pev = next_pev
                if train_step > agent.B:
                    agent.update_target_model()
                    train_step = 0
                if done:
                    final_score = reward
                    print('Done: {}, Total cost: {}'.format(done, reward))
                else:
                    ept_score += reward

            episodes.append(e)
            eptscores.append(ept_score)
            finalscores.append(final_score)

            if e % 50 == 49:
                now = datetime.datetime.now()
                training_time = now - now_start

                agent.model.save("{}/dqn_model.h5".format(resultdir))
                agent.model.save("dqn_model.h5")

                plt.title('Training eptscores: {}'.format(training_time))
                plt.xlabel('Epoch')
                plt.ylabel('score')
                plt.plot(episodes, eptscores, 'b')
                fig = plt.gcf()
                fig.savefig('{}/train eptscores.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
                plt.clf()
                plt.title('Training finalscores: {}'.format(training_time))
                plt.xlabel('Epoch')
                plt.ylabel('step')
                plt.plot(episodes, finalscores, 'r')
                fig = plt.gcf()
                fig.savefig('{}/train finalscores.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
                plt.clf()

        now = datetime.datetime.now()
        training_time = now - now_start

        agent.model.save("{}/dqn_model.h5".format(resultdir))
        agent.model.save("dqn_model.h5")

        plt.title('Training eptscores: {}'.format(training_time))
        plt.xlabel('Epoch')
        plt.ylabel('score')
        plt.plot(episodes, eptscores, 'b')
        fig = plt.gcf()
        fig.savefig('{}/train eptscores.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
        plt.clf()
        plt.title('Training finalscores: {}'.format(training_time))
        plt.xlabel('Epoch')
        plt.ylabel('step')
        plt.plot(episodes, finalscores, 'r')
        fig = plt.gcf()
        fig.savefig('{}/train finalscores.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
        plt.clf()



###############################  performance evaluation  #############

    # graph_test = Graph_jeju('data/20191001_5Min_modified.csv')
    graph_test = Graph_simple_39()

    testagent = DQNAgent(state_size, action_size, True)

    print(N_SOC, len(graph_test.cs_info), action_size, state_size)
    for i in range(1):

        npev = 1000
        EV_list, CS_list, graph_test = gen_test_envir_simple(graph_test)
        # graph = graph_train

        env = Env(graph_test, state_size, action_size)

        EV_list_DQN_REF = copy.deepcopy(EV_list)
        CS_list_DQN_REF = copy.deepcopy(CS_list)
        test_dqn(EV_list_DQN_REF, CS_list_DQN_REF, graph_test, env, testagent)


        EV_list_Greedy = copy.deepcopy(EV_list)
        CS_list_Greedy = copy.deepcopy(CS_list)
        ta.get_greedy_fleet(EV_list_Greedy, CS_list_Greedy, graph_test)

        tot_wt = 0
        tot_cost = 0
        for pev in EV_list_DQN_REF:
            tot_wt += pev.true_waitingtime
            tot_cost += pev.totalcost
        print('==========DQN===================')
        print('Avg. total waiting time: ', tot_wt / len(EV_list_DQN_REF))
        print('Total cost: ', tot_cost)

        tot_wt = 0
        tot_cost = 0
        for pev in EV_list_Greedy:
            tot_wt += pev.true_waitingtime
            tot_cost += pev.totalcost
        print('==========EV_list_Greedy===================')
        print('Avg. total waiting time: ', tot_wt / len(EV_list_Greedy))
        print('Total cost: ', tot_cost)



        # ta.sim_result_text_ref(i, CS_list, graph, resultdir, EV_list_DQN_REF=EV_list_DQN_REF, EV_list_Astar_ref=EV_list_Astar_ref, EV_list_Astar_shortest=EV_list_Astar_shortest)
        # ta.sim_result_text_last(i, CS_list, graph_test, resultdir, EV_list_DQN_REF=EV_list_DQN_REF, EV_list_Greedy=EV_list_Greedy, EV_list_Greedy_short=EV_list_Greedy_short)
