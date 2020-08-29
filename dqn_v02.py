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


from Graph import Graph_simple

EPISODES = 10000

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

def one_hot(x):
    return np.identity(100)[x:x + 1]

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
        self.cs=None
        self.csid = None
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

# 카트폴 예제에서의 DQN 에이전트
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
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.train_start = 1000

        # 리플레이 메모리, 최대 크기 2000
        self.memory = deque(maxlen=4000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model(0)

        # if self.load_model:
        #     self.model.load_weights("cartpole_dqn_trained.h5")

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        model = Sequential()

        # model.add(Conv2D(input_shape=(self.state_size.shape[0], self.state_size.shape[1], self.state_size.shape[2]), filters=50, kernel_size=(3, 3),
        #            strides=(1, 1), padding='same'))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        #
        # # prior layer should be flattend to be connected to dense layers
        # model.add(Flatten())
        # # dense layer with 50 neurons
        # model.add(Dense(50, activation='relu'))
        # # final layer with 10 neurons to classify the instances
        # model.add(Dense(self.action_size, activation='softmax'))
        #
        # adam = optimizers.Adam(lr=self.learning_rate)
        # model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


        model.add(Dense(64, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(24, activation='relu',
                        kernel_initializer='he_uniform'))
        # model.add(Dense(300, activation='relu',
        #                 kernel_initializer='he_uniform'))
        # model.add(Dense(400, activation='relu',
        #                 kernel_initializer='he_uniform'))
        # model.add(Dense(200, activation='relu',
        #                 kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax',
                        kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model



    # 타깃 모델을 모델의 가중치로 업데이트
    def update_target_model(self, e):
        if self.epsilon > self.epsilon_min and e>500:
            self.epsilon *= self.epsilon_decay
        self.target_model.set_weights(self.model.get_weights())

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state, current, graph):

        # action_list = graph.neighbors(current)
        # print(state.shape)
        if np.random.rand() <= self.epsilon:
            # return random.choice(action_list)
            action =  random.choice(range(4))
        else:
            state = np.reshape(state, [1, self.state_size])
            # print(state.shape)
            q_value = self.model.predict(state)
            # print('q value', q_value.shape, q_value[0].shape, np.argmax(q_value[0]))
            # print(q_value)
            # if np.argmax(q_value[0]) not in action_list:
            #     print('error')
            #     input()
            action = np.argmax(q_value[0])

        return action




    # 샘플 <s, a, r, s'>을 리플레이 메모리에 저장
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):


        # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        mini_batch = random.sample(self.memory, self.batch_size)


        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []
        # print(states.shape)
        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])

        # 현재 상태에 대한 모델의 큐함수
        # 다음 상태에 대한 타깃 모델의 큐함수
        target = self.model.predict(states)
        target_val = self.target_model.predict(next_states)

        # 벨만 최적 방정식을 이용한 업데이트 타깃
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = rewards[i]
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (
                    np.amax(target_val[i]))

        self.model.fit(states, target, batch_size=self.batch_size,
                       epochs=1, verbose=0)




class Env:
    def __init__(self):
        np.random.seed(10)
        self.graph = Graph_simple()

        self.source = 0
        self.destination = 99

        self.graph.source_node_set = list(self.graph.source_node_set)
        self.graph.destination_node_set = list(self.graph.destination_node_set)
        self.path = []

    def reset(self):
        self.path = []
        self.graph.source_node_set = list(self.graph.source_node_set)
        self.graph.destination_node_set = list(self.graph.destination_node_set)
        self.source = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]
        # self.source = self.graph.source_node_set[s]
        self.destination = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]

        while self.destination == self.source:
            self.destination = self.graph.source_node_set[np.random.random_integers(0, len(self.graph.source_node_set) - 1)]

        # self.source = 0
        # self.destination = 44

        # s = one_hot(self.source)
        # d = one_hot(self.destination)
        xs, ys = self.graph.nodes_xy(self.source)
        xd, yd = self.graph.nodes_xy(self.destination)

        # state = np.zeros((5, 5, 2))
        # state[int(xs), int(ys), 0] = 1
        # state[int(xd), int(yd), 1] = 1

        cur = np.zeros((5, 5))
        dest = np.zeros((5, 5))
        cur[int(xs), int(ys)] = 1
        dest[int(xd), int(yd)] = 1
        cur = np.reshape(cur, [1, 25])
        dest = np.reshape(dest, [1, 25])
        state = np.concatenate((cur, dest), axis=0)
        state = np.reshape(state, [1, 50])

        return state, self.source, self.destination

    def next_node(self, curr, action):

        x, y = self.graph.nodes_xy(curr)
        nodes = self.graph.neighbors(curr)
        arr = {}
        for n in nodes:
            nx, ny = self.graph.nodes_xy(n)
            # print(n, nx, ny)
            if x>nx:
                arr[UP] = n
            elif x<nx:
                arr[DOWN] = n
            else:
                if y>ny:
                    arr[LEFT] = n
                elif y<ny:
                    arr[RIGHT] = n

        if action not in arr.keys():
            return curr
        else:
            return arr[action]



    def step(self, current, action):
        action_list = self.graph.neighbors(current)
        next_node = self.next_node(current, action)
        # print(next_node)

        if next_node == current:
            reward = -10
            done = 0
        elif next_node in self.path:
            reward = -2
            done = 0
        else:
            if next_node == self.destination:
                done = 1
                reward = 1
            else:
                done = 0
                reward = -0.01


        xs, ys = self.graph.nodes_xy(next_node)
        xd, yd = self.graph.nodes_xy(self.destination)

        # next_state = np.zeros((5, 5, 2))
        # next_state[int(xs), int(ys), 0] = 1
        # next_state[int(xd), int(yd), 1] = 1


        cur = np.zeros((5, 5))
        dest = np.zeros((5, 5))
        cur[int(xs), int(ys)] = 1
        dest[int(xd), int(yd)] = 1
        cur = np.reshape(cur, [1, 25])
        dest = np.reshape(dest, [1, 25])
        next_state = np.concatenate((cur, dest), axis=0)
        next_state = np.reshape(next_state, [1, 50])
        self.path.append(next_node)
        return next_state, next_node, reward, done


if __name__ == "__main__":
    # CartPole-v1 환경, 최대 타임스텝 수가 500
    # env = gym.make('CartPole-v1')
    # state_size = env.observation_space.shape[0]
    # action_size = env.action_space.n

    env = Env()
    # state_size = np.zeros((5, 5, 2))
    state_size = 50
    action_size = 4

    agent = DQNAgent(state_size, action_size)

    # # DQN 에이전트 생성
    # agent = DQNAgent(state_size, action_size)
    #
    scores, episodes, steps= [], [], []
    for e in range(EPISODES):

        done = False
        score = 0
        # env 초기화
        path=[]
        state, start, destination = env.reset()
        # state = np.reshape(state, [1, state_size])
        current_node = start
        step = 0
        print("\nEpi:", e, agent.epsilon)
        print(start,'->', destination)
        path.append(start)
        while not done:
            action = agent.get_action(state, current_node, env.graph)
            # 선택한 행동으로 환경에서 한 타임스텝 진행
            next_state, next_node, reward, done = env.step(current_node, action)
            # print(current_node, action, next_node)
            # next_state = np.reshape(next_state, [1, state_size])
            # 에피소드가 중간에 끝나면 -100 보상

            # 리플레이 메모리에 샘플 <s, a, r, s'> 저장
            # state_reshape = np.reshape(state, [1, state.shape[0]*state.shape[1]*state.shape[2] ] )
            # next_state_reshape = np.reshape(state, [1, next_state.shape[0]*state.shape[1]*state.shape[2] ] )
            # print(state.shape)
            agent.append_sample(state, action, reward, next_state, done)
            #


            # 매 타임스텝마다 학습
            if len(agent.memory) >= agent.train_start:
                agent.train_model()

            if score < -40:
                done = 1


            # print(current_node, reward, done)
            score += reward
            state = next_state
            current_node = next_node
            path.append(next_node)
            step += 1

            if done:
                # 각 에피소드마다 타깃 모델을 모델의 가중치로 업데이트
                if e % 1 == 0:
                    agent.update_target_model(e)
                    print('update model')
                print(path)
                print('last node:', current_node)
                print('Score:', score)
                print('Step:', step)
                scores.append(score)
                episodes.append(e)
                steps.append(step)

    plt.plot(episodes, scores)
    plt.show()
    plt.plot(episodes, steps, 'r')
    plt.show()