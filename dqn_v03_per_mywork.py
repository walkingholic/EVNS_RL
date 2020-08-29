import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from keras import optimizers
import matplotlib.pyplot as plt
import routing as rt
import copy
import test_algorithm as ta
import datetime
import os
from SumTree import SumTree
from time import time
from keras.callbacks import TensorBoard


from Graph import Graph_simple
from Graph import Graph_simple_100
from Graph import Graph_jeju
from Graph import Graph_simple_39




def one_hot(x):
    return np.identity(100)[x:x + 1]
def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')
class CS:
    def __init__(self, node_id, long, lat, alpha):
        self.id = node_id
        self.price = list()
        self.waittime = list()
        self.chargingpower = 60 # kw
        self.alpha = alpha
        self.x = long
        self.y = lat

        for i in range(288):
            p = np.random.normal(alpha, 0.15 * alpha)
            while p < 0:
                p = np.random.normal(alpha, 0.15 * alpha)
            self.price.append(p)

        for i in range(288):
            waittime = np.random.normal(-1200 * (self.price[i] - 0.07), 20)
            while waittime < 0:
                waittime = 0
            self.waittime.append(waittime/60)

class EV:
    def __init__(self, id, t_start, soc, source, destination):
        self.id = id
        self.t_start = t_start
        self.charging_effi = 0.9
        self.SOC = soc
        self.init_SOC = soc
        self.req_SOC = 1.0
        self.before_charging_SOC=soc
        self.source = source
        self.destination = destination
        self.maxBCAPA= 60  # kw
        self.curr_location = source
        self.next_location = source
        # self.ECRate = 0.2 # kwh/km
        self.ECRate = 0.147 # kwh/km
        self.traveltime = 0 # hour
        self.charged = 0
        self.cs = None
        self.csid = -1
        self.energyconsumption = 0.0
        self.chargingtime = 0.0
        self.chargingcost = 0.0
        self.waitingtime = 0.0
        self.csstayingtime = 0.0
        self.drivingdistance = 0.0
        self.drivingtime = 0.0
        self.charingenergy = 0.0
        self.pev.cschargingprice = 0.0

        self.fdist=0
        self.rdist=0
        self.path=[]
        self.predic_totaltraveltime = 0.0
        self.totalcost=0.0
        self.chargingstarttime = 0.0

        self.to_cs_dist = 0
        self.to_cs_driving_time = 0
        self.to_cs_charging_time = 0
        self.to_cs_waiting_time = 0
        self.to_cs_soc = 0

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


def gen_test_envir_simple(num_evs):

    graph = Graph_simple_39()

    EV_list = []

    for e in range(num_evs):
        t_start =  np.random.uniform(0, 1200)
        soc = np.random.uniform(0.4, 0.6)
        while soc <= 0.0 or soc > 1.0 :
            soc = np.random.uniform(0.4, 0.6)
        graph.source_node_set = list(graph.source_node_set)
        graph.destination_node_set = list(graph.destination_node_set)

        # source = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]
        # while source in graph.cs_info.keys():
        #     source = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]
        source = e+1
        destination = graph.destination_node_set[np.random.random_integers(0, len(graph.destination_node_set) - 1)]

        ev = EV(e, t_start, soc, source, destination)
        EV_list.append(ev)

    CS_list = []
    for l in graph.cs_info:
        # print('gen cs')
        # alpha = np.random.uniform(0.03, 0.07)
        alpha = np.random.uniform(0.03, 0.07)

        cs = CS(l, graph.cs_info[l]['long'], graph.cs_info[l]['lat'], alpha)
        CS_list.append(cs)

    return EV_list, CS_list, graph

# 카트폴 예제에서의 DQN 에이전트

EPISODES = 800
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.render = False
        self.load_model = False

        # 상태와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # DQN 하이퍼파라미터
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.9994
        self.epsilon_min = 0.01
        self.batch_size = 16
        self.train_start = 2000

        # 리플레이 메모리, 최대 크기 2000
        self.memory_size = 2000
        # self.memory = deque(maxlen=4000)
        self.memory = Memory(self.memory_size)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model(0)
        log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

        if self.load_model:
            self.model.load_weights("dqn_model.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(512, activation='relu',
                        kernel_initializer='he_uniform'))

        # model.add(Dense(32, activation='relu',
        #                 kernel_initializer='he_uniform'))

        model.add(Dense(self.action_size, activation='linear',
                        kernel_initializer='he_uniform'))
        model.summary()

        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))



        return model



    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self, e):
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
        # np.random.seed(10)
        self.graph = graph
        self.graph.reset_traffic_info()
        # self.graph = Graph_jeju('data/20191001_5Min_modified.csv')
        self.source = 0
        self.graph.source_node_set = list(self.graph.source_node_set)
        self.graph.destination_node_set = list(self.graph.destination_node_set)

        self.path = []
        self.path_info = []
        self.sim_time=0
        self.CS_list = []
        self.pev = None
        # self.target = -1
        self.state_size = state_size
        self.action_size = action_size

        self.reset_CS_info()

    def reset_CS_info(self):
        self.CS_list = []
        for l in self.graph.cs_info:
            alpha = np.random.uniform(0.03, 0.07)
            cs = CS(l, self.graph.cs_info[l]['long'], self.graph.cs_info[l]['lat'], alpha)
            self.CS_list.append(cs)

    def reset(self):
        self.graph.reset_traffic_info()
        self.reset_CS_info()
        t_start = np.random.uniform(0, 1200)
        soc = np.random.uniform(0.4, 0.6)
        while soc <= 0.0 or soc > 1.0:
            soc = np.random.uniform(0.4, 0.6)
        self.graph.source_node_set = list(self.graph.source_node_set)
        source = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]
        self.graph.destination_node_set = list(self.graph.destination_node_set)
        destination = self.graph.destination_node_set[np.random.random_integers(0, len(self.graph.destination_node_set) - 1)]


        self.path_info = []
        self.pev = EV(e, t_start, soc, source, destination)
        self.sim_time = self.pev.t_start
        self.path_info = rt.get_feature_state_refer(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)

        # print('\npev soc', self.pev.SOC)
        state =[self.pev.source, self.pev.SOC]
        for path in self.path_info:
            cs, pev_SOC, path, road_cost, waiting_time, charging_time, charging_cost = path
            # next_state += [front_d_time, rear_d_time, waiting_time, charging_time]
            state += [road_cost, charging_cost, waiting_time]
        state = np.reshape(state, [1, self.state_size])

        return state, source, destination

    def test_reset(self, pev, graph, CS_list):

        self.graph = graph
        self.path_info = []
        self.CS_list = CS_list
        self.pev = pev
        self.sim_time = self.pev.t_start
        self.path_info = rt.get_feature_state_refer(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)

        state = [self.pev.source, self.pev.curr_SOC]
        for path in self.path_info:
            cs, pev_SOC, path, road_cost, waiting_time, charging_time, charging_cost = path
            # state += [front_d_time, rear_d_time, waiting_time, charging_time]
            state += [road_cost, charging_cost, waiting_time]
        state = np.reshape(state, [1, self.state_size])

        return state, self.pev.source, self.pev.destination


    def step(self, action):
        cs, pev_SOC, path, road_cost, waiting_time, charging_time, charging_cost = self.path_info[action]

        self.pev.path.append(self.pev.curr_location)

        if len(path)>1:
            next_node = path[1]
            self.sim_time, time = rt.update_ev(self.pev, self.graph, self.pev.curr_location, next_node, self.sim_time)
            if self.sim_time == 0 and time == 0:
                done = 1
                reward = -20
                return np.zeros((1, self.state_size)), -1, reward, done
            if pev_SOC <= 0.0:
                done = 1
                reward = -20
                return np.zeros((1, self.state_size)), -1, reward, done


            done = 0
            reward = -1*road_cost
            # print(reward)
            # reward = -time

            self.pev.curr_location = next_node
            self.path_info = rt.get_feature_state_refer(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)
            next_state = [self.pev.curr_location, self.pev.curr_SOC]

            for path in self.path_info:
                cs, pev_SOC, path, road_cost, waiting_time, charging_time, charging_cost = path
                # next_state += [front_d_time, rear_d_time, waiting_time, charging_time]
                next_state += [road_cost, charging_cost, waiting_time]
            next_state = np.reshape(next_state, [1, self.state_size])
            return next_state, next_node, reward, done

        elif self.pev.curr_location == cs.id:

            self.pev.before_charging_SOC = self.pev.curr_SOC
            self.pev.cscharingenergy = self.pev.maxBCAPA * self.pev.req_SOC - self.pev.curr_SOC * self.pev.maxBCAPA
            self.pev.cschargingcost = self.pev.cscharingenergy * cs.price[int(self.sim_time / 5)]
            self.pev.curr_SOC = self.pev.req_SOC
            self.pev.cschargingtime = (self.pev.cscharingenergy / (cs.chargingpower * self.pev.charging_effi))

            self.pev.cschargingwaitingtime = cs.waittime[int(self.sim_time / 5)]
            self.pev.charged = 1
            self.pev.cs = cs
            self.pev.csid = cs.id
            self.pev.cschargingstarttime = self.sim_time
            self.pev.cschargingprice = cs.price[int(self.sim_time / 5)]

            self.pev.fdist = self.pev.totaldrivingdistance
            self.pev.csdrivingtime = self.pev.totaldrivingtime
            self.pev.csdistance = self.pev.totaldrivingdistance
            self.pev.cschargingwaitingtime = self.pev.cschargingwaitingtime
            self.pev.cschargingtime = self.pev.cschargingtime
            self.pev.cssoc = self.pev.curr_SOC
            self.pev.totalcost = self.pev.totaldrivingtime * 0.75 + self.pev.totaldrivingdistance * self.pev.ECRate * cs.price[int(self.sim_time / 5)] + self.pev.cschargingcost + self.pev.cschargingwaitingtime * 0.75

            self.sim_time += self.pev.cschargingtime * 60
            self.sim_time += self.pev.cschargingwaitingtime * 60


            done = 1
            # reward = -1 * (self.pev.waitingtime + self.pev.chargingtime + rear_d_time + (rear_consump_energy/(cs.chargingpower*self.pev.charging_effi)))
            reward = -1 * (self.pev.cschargingcost + self.pev.cschargingwaitingtime * 0.75)
            # print(self.sim_time, reward)

            return np.zeros((1,self.state_size)), -1, reward, done

        else:
            print("???")
            input()


def test_dqn(EV_list_DQN_REF, CS_list_DQN_REF, graph, env, agent ):

    agent.epsilon = 0


    for e, pev in enumerate(EV_list_DQN_REF):
        done = False
        score = 0
        path=[]
        state, source, destination = env.test_reset(pev, graph, CS_list_DQN_REF)
        print("\nEpi:", e, agent.epsilon)
        print(source,'->', destination)
        print('sim time:', env.sim_time)
        path.append(source)

        while not done:
            action = agent.get_action(state)
            next_state, next_node, reward, done = env.step(action)
            score += reward
            state = next_state
            path.append(next_node)

            if done:
                print('sim time:', env.sim_time)
                print('Distance:', pev.totaldrivingdistance)
                print('Driving time:', pev.totaldrivingtime)
                print(pev.charged, pev.curr_location, pev.init_SOC, pev.curr_SOC)
                print(path)

        while pev.curr_location != pev.destination:
            came_from, cost_so_far = rt.a_star_search(graph, pev.curr_location, pev.destination)
            path = rt.reconstruct_path(came_from, pev.curr_location, pev.destination)
            path_distance = graph.get_path_distance(path)
            # print("evcango: {}  path dist: {}".format(evcango, path_distance))
            pev.next_location = path[1]
            pev.path.append(pev.next_location)
            env.sim_time, time = rt.update_ev(pev, graph, pev.curr_location, pev.next_location, env.sim_time)
            pev.curr_location = pev.next_location





if __name__ == "__main__":
    # npev = 39
    # EV_list, CS_list, graph = gen_test_envir_simple(npev)

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    action_size = 3
    state_size = action_size*3+2
    graph = Graph_simple_39()

    print('S:{} A:{}'.format(state_size, action_size))

    agent = DQNAgent(state_size, action_size)
    scores, episodes, steps= [], [], []
    n_step = 0
    train_step = 0

    agent.load_model = False
    # agent.load_model = True
    if not agent.load_model:
        for e in range(EPISODES):
            episcore = 0
            epistep = 0
            numsucc, numfail = 0 , 0
            env = Env(graph, state_size, action_size)
            print("\nEpi:", e, agent.epsilon, 'episcore:', episcore)

            for n in range(39):
                done = False
                path=[]
                score, step = 0, 0
                state, source = env.reset(n+1)
                current_node = source
                path.append(source)

                while not done:
                    action = agent.get_action(state)
                    next_state, next_node, reward, done = env.step(action)
                    step += 1
                    n_step += 1
                    agent.append_sample(state, action, reward, next_state, done)
                    if n_step >= agent.train_start:
                        agent.train_model()
                        train_step += 1
                    score += reward
                    state = next_state
                    path.append(next_node)
                    if train_step > 20:
                        agent.update_target_model(e)
                        train_step = 0

                    if done:
                        if env.pev.curr_SOC >= 1.0:
                            numsucc += 1
                        else:
                            numfail += 1

                        print('({}, {})'.format(env.pev.csid, step), end='   ')
                        if n % 10 == 9:
                            print(' ->  ', numsucc, numfail)

                        episcore += score
                        epistep += step

            print("\nEpi:", e, agent.epsilon, 'episcore:', episcore)
            episodes.append(e)
            steps.append(epistep)
            scores.append(episcore)

        now = datetime.datetime.now()
        training_time = now - now_start

        agent.model.save_weights("{}/dqn_model.h5".format(resultdir))
        plt.title('Training Scores: {}'.format(training_time))
        plt.plot(episodes, scores, 'b')
        plt.show()

        plt.title('Training Steps: {}'.format(training_time))
        plt.plot(episodes, steps, 'r')
        plt.show()

        plt.title('Training Scores: {}'.format(training_time))
        plt.xlabel('Epoch')
        plt.ylabel('score')
        plt.plot(episodes, scores, 'b')
        fig = plt.gcf()
        fig.savefig('{}/train score.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
        plt.clf()
        plt.title('Training Steps: {}'.format(training_time))
        plt.xlabel('Epoch')
        plt.ylabel('step')
        plt.plot(episodes, steps, 'r')
        fig = plt.gcf()
        fig.savefig('{}/train step.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
        plt.clf()

###############################  performance evaluation  #############

    for i in range(5):

        npev=39
        EV_list, CS_list, graph = gen_test_envir_simple(npev)

        agent.epsilon = 0
        env = Env(graph, state_size, action_size)

        EV_list_DQN_REF = copy.deepcopy(EV_list)
        CS_list_DQN_REF = copy.deepcopy(CS_list)
        test_dqn(EV_list_DQN_REF, CS_list_DQN_REF, graph, env, agent)


        EV_list_Astar_ref = copy.deepcopy(EV_list)
        CS_list_Astar_ref = copy.deepcopy(CS_list)
        ta.every_time_check_refer(EV_list_Astar_ref, CS_list_Astar_ref, graph)

        EV_list_Astar_shortest = copy.deepcopy(EV_list)
        CS_list_Astar_shortest = copy.deepcopy(CS_list)
        ta.every_time_check_refer_shortest(EV_list_Astar_shortest, CS_list_Astar_shortest, graph)

        ta.sim_result_text_last(i, resultdir, EV_list_DQN_REF=EV_list_DQN_REF, EV_list_Astar_ref=EV_list_Astar_ref, EV_list_Astar_shortest=EV_list_Astar_shortest)
        ta.sim_result_general_presentation_last(i, graph, resultdir, npev, EV_list_DQN_REF=EV_list_DQN_REF, EV_list_Astar_ref=EV_list_Astar_ref, EV_list_Astar_shortest=EV_list_Astar_shortest)

