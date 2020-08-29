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


from Graph import Graph_simple
from Graph import Graph_simple_100
from Graph import Graph_jeju




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
        self.req_SOC = 0.8
        self.before_charging_SOC=soc
        self.source = source
        self.destination = destination
        self.maxBCAPA= 60  # kw
        self.curr_location = source
        self.next_location = source
        self.ECRate = 0.2 # kwh/km
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
        self.fdist=0
        self.rdist=0
        self.path=[]
        self.predic_totaltraveltime = 0.0
        self.totalcost=0.0

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

    np.random.seed(10)
    graph = Graph_simple_100()

    EV_list = []

    for e in range(num_evs):
        t_start =  np.random.uniform(0, 1200)
        soc = np.random.uniform(0.3, 0.5)
        while soc <= 0.0 or soc > 1.0 :
            soc = np.random.uniform(0.3, 0.5)
        graph.source_node_set = list(graph.source_node_set)
        graph.destination_node_set = list(graph.destination_node_set)

        source = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]
        while source in graph.cs_info.keys():
            source = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]

        destination = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]
        while destination in graph.cs_info.keys():
            destination = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]
        # source = 4080021700
        # destination = 4070008103
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


def gen_test_envir_jeju(traffic_data_path, num_evs):

    graph = Graph_jeju(traffic_data_path)

    EV_list = []

    for e in range(num_evs):
        t_start =  np.random.uniform(0, 1200)
        soc = np.random.uniform(0.3, 0.5)
        while soc <= 0.0 or soc > 1.0 :
            soc = np.random.uniform(0.3, 0.5)
        graph.source_node_set = list(graph.source_node_set)
        graph.destination_node_set = list(graph.destination_node_set)
        source = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]

        destination = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]

        while destination in graph.cs_info.keys():
            destination = graph.source_node_set[np.random.random_integers(0, len(graph.source_node_set) - 1)]
        # source = 4080021700
        # destination = 4070008103
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

EPISODES = 300
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
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.01
        self.batch_size = 64
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

        if self.load_model:
            self.model.load_weights("dqn_model.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(256, activation='relu',
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
            action = random.choice(range(self.action_size))
            # print(action)
        else:
            # state = np.reshape(state, [1, self.state_size])
            q_value = self.model.predict(state)
            action = np.argmax(q_value[0])

            # print(q_value, q_value.shape, q_value[0].shape, q_value[0], action)
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

        self.model.fit(states, target, batch_size=self.batch_size, epochs=1, verbose=0)


class Env:
    def __init__(self, state_size, action_size):
        # np.random.seed(10)
        # self.graph = Graph_simple()
        self.graph = Graph_simple_100()
        # self.graph = Graph_jeju('data/20191001_5Min_modified.csv')
        self.source = 0
        self.destination = 99
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

        for l in self.graph.cs_info:
            alpha = np.random.uniform(0.03, 0.07)
            cs = CS(l, self.graph.cs_info[l]['long'], self.graph.cs_info[l]['lat'], alpha)
            self.CS_list.append(cs)

    def reset(self):

        t_start = np.random.uniform(0, 1200)
        soc = np.random.uniform(0.3, 0.5)
        while soc <= 0.0 or soc > 1.0:
            soc = np.random.uniform(0.3, 0.5)
        self.graph.source_node_set = list(self.graph.source_node_set)
        self.graph.destination_node_set = list(self.graph.destination_node_set)

        source = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]
        while source in self.graph.cs_info.keys():
            source = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]

        destination = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]
        while destination in self.graph.cs_info.keys():
            destination = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]

        self.path_info = []
        self.pev = EV(e, t_start, soc, source, destination)
        self.sim_time = self.pev.t_start
        self.path_info = rt.get_feature_state(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)

        # print('\npev soc', self.pev.SOC)
        state = [self.pev.source, self.pev.destination, self.pev.SOC]
        for path in self.path_info:
            cs, pev_SOC, front_path,  rear_path, front_path_distance, rear_path_distance,  front_d_time, rear_d_time, rear_consump_energy, waiting_time, charging_time = path
            state += [front_d_time, rear_d_time, waiting_time, charging_time]
            # state += [front_path_distance, rear_path_distance]
        state = np.reshape(state, [1, self.state_size])

        return state, source, destination

    def reset_group(self, source):

        t_start = np.random.uniform(0, 1200)
        soc = np.random.uniform(0.3, 0.7)
        while soc <= 0.0 or soc > 1.0:
            soc = np.random.uniform(0.3, 0.7)
        self.graph.source_node_set = list(self.graph.source_node_set)
        self.graph.destination_node_set = list(self.graph.destination_node_set)

        source = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]
        destination = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]

        self.path_info = []
        self.pev = EV(e, t_start, soc, source, destination)
        self.sim_time = self.pev.t_start
        self.path_info = rt.get_feature_state(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)

        # print('\npev soc', self.pev.SOC)
        state =[self.pev.source, self.pev.destination, self.pev.SOC]
        for path in self.path_info:
            cs, pev_SOC, front_path,  rear_path, front_path_distance, rear_path_distance,  front_d_time, rear_d_time, rear_consump_energy, waiting_time, charging_time = path
            state += [front_d_time, rear_d_time, waiting_time, charging_time]
            # state += [front_path_distance, rear_path_distance]
        state = np.reshape(state, [1, self.state_size])

        return state, source, destination

    def test_reset(self, pev, CS_list):

        self.path_info = []
        self.CS_list = CS_list
        self.pev = pev
        self.sim_time = self.pev.t_start
        self.path_info = rt.get_feature_state(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)


        state = [self.pev.source, self.pev.destination, self.pev.curr_SOC]
        for path in self.path_info:
            cs, pev_SOC, front_path,  rear_path,  front_path_distance, rear_path_distance,  front_d_time, rear_d_time, rear_consump_energy, waiting_time, charging_time = path
            state += [front_d_time, rear_d_time, waiting_time, charging_time]
            # state += [front_path_distance, rear_path_distance]
        state = np.reshape(state, [1, self.state_size])

        return state, self.pev.source, self.pev.destination


    def step(self, action, done):
        cs, pev_SOC, front_path,  rear_path, front_path_distance, rear_path_distance,  front_d_time, rear_d_time, rear_consump_energy, waiting_time, charging_time = self.path_info[action]
        # print(self.sim_time, self.pev.curr_location, cs.id, pev_SOC)
        self.pev.path.append(self.pev.curr_location)

        if len(front_path)>1:
            next_node = front_path[1]
            self.sim_time, time = rt.update_ev(self.pev, self.graph, self.pev.curr_location, next_node, self.sim_time)
            if self.sim_time == 0 and time == 0:
                done = 1
                reward = -10
                return np.zeros((1, self.state_size)), -1, reward, done

            if pev_SOC <= 0.0:
                done = 1
                reward = -10
                return np.zeros((1, self.state_size)), -1, reward, done


            done = 0
            reward = -1
            # reward = -time

            self.pev.curr_location = next_node
            self.path_info = rt.get_feature_state(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)
            next_state = [self.pev.source, self.pev.destination, self.pev.curr_SOC]

            for path in self.path_info:
                cs, pev_SOC, front_path,  rear_path, front_path_distance, rear_path_distance,  front_d_time, rear_d_time, rear_consump_energy, waiting_time, charging_time = path
                # next_state += [front_d_time, rear_d_time, waiting_time, charging_time]
                next_state += [front_path_distance, rear_path_distance]
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

            self.sim_time += self.pev.cschargingtime * 60
            self.sim_time += self.pev.cschargingwaitingtime * 60
            # print(waiting_time, waiting_time * 60)

            self.pev.fdist = self.pev.totaldrivingdistance
            self.pev.csdrivingtime = self.pev.totaldrivingtime
            self.pev.csdistance = self.pev.totaldrivingdistance
            self.pev.cschargingwaitingtime = self.pev.cschargingwaitingtime
            self.pev.cschargingtime = self.pev.cschargingtime
            self.pev.cssoc = self.pev.curr_SOC

            done = 1
            # reward = -1 * (self.pev.waitingtime + self.pev.chargingtime + rear_d_time + (rear_consump_energy/(cs.chargingpower*self.pev.charging_effi)))
            reward = -1 * len(rear_path)


            # print(done, '충전소야')

            return np.zeros((1,self.state_size)), -1, reward, done
        else:
            print("???")
            input()





if __name__ == "__main__":

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)
    action_size = 4
    state_size = action_size*2+3
    env = Env(state_size, action_size)
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
            print("\nEpi:", e, agent.epsilon, 'episcore:', episcore)
            for n in range(100):
                done = False
                path=[]
                score=0
                state, source, destination = env.reset_group(n)
                current_node = source
                step = 0

                # print(source,'->', destination)
                # print('{} sim time: {} - {}'.format(n, env.sim_time, env.pev.SOC))
                path.append(source)

                while not done:
                    # print(state)
                    action = agent.get_action(state)
                    next_state, next_node, reward, done = env.step(action, done)
                    step += 1
                    # if step>30:
                    #     done = 1
                    # print('{} sim time: {} - {}'.format(n, env.sim_time, env.pev.SOC))
                    n_step += 1
                    agent.append_sample(state, action, reward, next_state, done)
                    if n_step >= agent.train_start:
                        agent.train_model()
                        train_step += 1
                    score += reward
                    state = next_state
                    path.append(next_node)
                    # taget_cs.append(env.target.id)

                    if train_step >20:
                        agent.update_target_model(e)
                        train_step = 0

                    if done:
                        # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                        # if e % 1 == 0:

                        if env.pev.curr_SOC>=0.8:
                            numsucc+=1
                        else:
                            numfail += 1

                        print(env.pev.csid, end='   ')
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
        plt.plot(episodes, scores, 'b')
        plt.show()
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

    scores, episodes, steps = [], [], []
    npev=100
    EV_list, CS_list, graph = gen_test_envir_simple(npev)


    agent.epsilon = 0
    EV_list_DQN = copy.deepcopy(EV_list)
    CS_list_DQN = copy.deepcopy(CS_list)
    for e, pev in enumerate(EV_list_DQN):
        done = False
        score = 0
        path=[]
        # taget_cs = []
        state, source, destination = env.test_reset(pev, CS_list_DQN)
        current_node = source
        step = 0
        print("\nEpi:", e, agent.epsilon)
        print(source,'->', destination)
        print('sim time:', env.sim_time)
        path.append(source)

        while not done:
            action = agent.get_action(state)
            next_state, next_node, reward, done = env.step(action, done)
            score += reward
            state = next_state
            path.append(next_node)
            # taget_cs.append(env.target.id)
            step += 1

            if done:
                print('Score:', score)
                print('Step:', step)
                print('sim time:', env.sim_time)
                print('Distance:', env.pev.totaldrivingdistance)
                print('Driving time:', env.pev.totaldrivingtime)
                print(env.pev.charged, env.pev.curr_location, env.pev.init_SOC, env.pev.curr_SOC)
                print(path)
                # print(taget_cs)
                scores.append(score)
                episodes.append(e)
                steps.append(step)

        pev = env.pev

        while pev.curr_location != pev.destination:
            came_from, cost_so_far = rt.a_star_search(graph, pev.curr_location, pev.destination)
            path = rt.reconstruct_path(came_from, pev.curr_location, pev.destination)
            path_distance = graph.get_path_distance(path)
            # print("evcango: {}  path dist: {}".format(evcango, path_distance))
            pev.next_location = path[1]
            pev.path.append(pev.next_location)
            env.sim_time, time = rt.update_ev(pev, graph, pev.curr_location, pev.next_location, env.sim_time)
            pev.curr_location = pev.next_location


    EV_list_TA = copy.deepcopy(EV_list)
    CS_list_TA = copy.deepcopy(CS_list)
    # ta.one_time_check_timeweight(EV_list_TA, CS_list_TA, graph, 313)
    ta.one_time_check_distweight(EV_list_TA, CS_list_TA, graph, 313)

    EV_list_TEA = copy.deepcopy(EV_list)
    CS_list_TEA = copy.deepcopy(CS_list)
    # ta.every_time_check_timeweight(EV_list_TEA, CS_list_TEA, graph, 313)
    ta.every_time_check_distweight(EV_list_TEA, CS_list_TEA, graph, 313)

    ta.sim_result_text(resultdir, EV_list_DQN=EV_list_DQN, EV_list_TA=EV_list_TA, EV_list_TEA=EV_list_TEA)
    ta.sim_result_general_presentation(graph, resultdir, npev, EV_list_DQN=EV_list_DQN, EV_list_TA=EV_list_TA, EV_list_TEA=EV_list_TEA)

