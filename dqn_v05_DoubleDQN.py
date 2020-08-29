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


from Graph import Graph_simple
from Graph import Graph_simple_100
from Graph import Graph_jeju
from Graph import Graph_simple_39

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'


EPISODES = 1500
EPS_DC = 0.9996
UNITtimecost = 8
ECRate = 0.16
Step_SOC = 0.15
Base_SOC = 0.6
Final_SOC = 0.9
N_SOC = int((Final_SOC-Base_SOC)/Step_SOC)+1
TRAIN = True
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

class CS:
    def __init__(self, node_id, profit, long, lat):
        self.id = node_id
        self.price = list()
        self.waittime = list()
        self.chargingpower = 60 # kw
        self.homechargingpower = 3.3
        self.x = long
        self.y = lat
        self.profit = profit


        # self.price = [0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.1748, 0.174, 0.174, 0.174, 0.174, 0.174, 0.174, 0.174, 0.174, 0.174, 0.174, 0.174, 0.174, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1724, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1735, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1601, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1321, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1618, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.1616, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.161, 0.161, 0.161, 0.161, 0.161, 0.161, 0.161, 0.161, 0.161, 0.161, 0.161, 0.161, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.1635, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.165, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1633, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1749, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1808, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1753, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1739, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1717, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1823, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786, 0.1786]
        # self.waittime = [0.3159,0.2307,0.3822,0.4749,0.3528,0.4735,0.1971,0.3783,0.4607,0.3992,0.4557,0.3294,0.4403,0.4590,0.2981,0.3891,0.3215,0.4613,0.3711,0.5208,0.4827,0.4757,0.4797,0.2796,0.3116,0.3992,0.4076,0.4489,0.4174,0.4063,0.3936,0.5251,0.3674,0.3566,0.4226,0.3560,0.4122,0.2921,0.2739,0.2923,0.3191,0.4183,0.3879,0.4939,0.3287,0.4576,0.4660,0.2593,0.4611,0.3035,0.4362,0.3509,0.4703,0.5681,0.4405,0.3755,0.3299,0.3427,0.4404,0.5371,0.5407,0.4423,0.4230,0.5321,0.4501,0.3971,0.3894,0.4253,0.5598,0.3422,0.1688,0.4409,0.4654,0.3112,0.4317,0.5211,0.3803,0.4241,0.4233,0.4078,0.4360,0.4090,0.3598,0.3690,0.4024,0.3636,0.2758,0.4755,0.3453,0.4930,0.4456,0.3565,0.3267,0.4188,0.2886,0.3617,0.4206,0.3294,0.3296,0.3406,0.3210,0.4533,0.4186,0.5008,0.4922,0.5021,0.4470,0.3758,0.4862,0.3849,0.2900,0.5458,0.3725,0.4174,0.3753,0.4968,0.2240,0.3586,0.2819,0.4385,0.4559,0.3370,0.4037,0.2974,0.3064,0.5119,0.4494,0.5884,0.3926,0.4368,0.4147,0.3312,0.5641,0.3283,0.2829,0.4689,0.4729,0.3672,0.5488,0.3224,0.2546,0.4055,0.5259,0.3265,0.4290,0.3452,0.3440,0.5260,0.4598,0.4478,0.4163,0.2667,0.1862,0.4671,0.4462,0.3923,0.2992,0.3437,0.2978,0.4633,0.3563,0.3805,0.3578,0.2498,0.2673,0.3651,0.3005,0.3771,0.4484,0.3530,0.4620,0.4828,0.4287,0.4525,0.3460,0.4551,0.4078,0.3666,0.2957,0.3249,0.3726,0.4391,0.2875,0.3319,0.4592,0.3242,0.3910,0.2487,0.3846,0.4402,0.4143,0.2350,0.3820,0.5937,0.4546,0.4485,0.4302,0.3886,0.4650,0.5457,0.3914,0.4016,0.3718,0.3041,0.4923,0.3852,0.4196,0.4298,0.3875,0.4117,0.4279,0.4596,0.4948,0.5727,0.3628,0.3619,0.3475,0.3198,0.5290,0.3238,0.3863,0.3741,0.4545,0.5760,0.4331,0.3883,0.4263,0.4659,0.4033,0.3616,0.3820,0.5245,0.4045,0.4542,0.2432,0.3457,0.3685,0.1991,0.4416,0.3721,0.4077,0.4908,0.3609,0.3947,0.3069,0.4584,0.4867,0.4628,0.3819,0.3733,0.4329,0.4091,0.4221,0.4033,0.3406,0.3566,0.3703,0.3137,0.3955,0.3851,0.3734,0.3855,0.2748,0.3519,0.4983,0.4315,0.3999,0.4824,0.4933,0.5352,0.2207,0.3719,0.3129,0.5666,0.3927,0.4336,0.3781,0.3067,0.4340,0.3496,0.4598,0.4787,0.1922,0.3125,0.4136,0.3921,0.2404,0.4437]

        # self.price = [0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736, 0.1736]
        # self.waittime = [0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307, 0.2307]

        # self.TOU_price = [[0.1736, 0.1601, 0.1748, 0.174, 0.1724, 0.1735, 0.1601, 0.1736, 0.1321, 0.1618, 0.1616, 0.1650, 0.161, 0.1635, 0.1650, 0.1633, 0.1749, 0.1808, 0.1808, 0.1753, 0.1739, 0.1717, 0.1823, 0.1786],
        #                 [155.75, 155.52, 150.71, 151.27, 149.81, 152.74, 154.38, 154.59, 179.44,  173.95, 175.59, 165.2, 158.13, 162.17, 166.25, 163.31, 176.85, 166.25, 233.37, 233.37, 233.37, 178.22, 160.1, 155.75],
        #                 [162.44, 49.56, 157.15, 150.11, 152.53, 155.15, 155.04, 158.33, 156.71, 168.12, 171.56, 172.24, 170.18, 172.59, 173.45, 168.47, 170.7, 230.47, 230.47, 230.47, 230.47, 171.9, 165.96, 163.68],
        #                 [151.48, 152.59, 151.96, 149.56, 149.56, 151.96, 152.59, 164.8, 156.61, 156.61, 161.38, 168.74, 157.97, 170.18, 171.85, 173.67, 227.71, 227.71, 227.71, 233.62, 227.71, 227.71, 172.81, 176.07],
        #                 [153.87, 154.13, 154.62, 152.3, 152.3, 152.64, 154.62, 154.44, 163.61, 169.35, 170.38, 170.72, 170.38, 165.15, 166.52, 169.01, 223.6, 223.6, 230.02,230.02, 230.02, 223.6, 168.66, 159.69],
        #                 [154.44, 153.32, 154.44, 53.17, 151.88,1 53.32, 152.3, 158.86, 162.17, 167.91, 169.29, 169.46, 168.77, 164.06, 164.1, 166.88, 211.32, 217.74, 232.85,224.34, 231.64, 211.32, 169.11, 162.96]]

        self.TOU_price = [0.15575, .015552, 0.15071, 0.15127, 0.14981, 0.15274, 0.15438, 0.15459, 0.17944, 0.17395, 0.17559, 0.1652, 0.15813, 0.16217, 0.16625, 0.16331, 0.17685, 0.16625, 0.23337, 0.23337, 0.23337, 0.17822, 0.1601, 0.15575]


        # for i in range(288):
        #     p = self.TOU_price[int(i/12)]*self.profit
        #     self.price.append(p)
        for i in range(288):
            p = np.random.normal(self.TOU_price[int(i/12)], 0.30 * self.TOU_price[int(i/12)])
            while p < 0:
                p = np.random.normal(self.TOU_price[int(i/12)], 0.30 * self.TOU_price[int(i/12)])
            self.price.append(p*self.profit)


        for i in range(288):
            waittime = np.random.normal(self.profit-0.7, 0.4*(self.profit-0.7))
            while waittime < 0:
                waittime = 0
            self.waittime.append(waittime)


        # for i in range(288):
        #     p = np.random.normal(self.TOU_price[int(i/12)], 0.20 * self.TOU_price[int(i/12)])
        #     while p < 0:
        #         p = np.random.normal(self.TOU_price[int(i/12)], 0.20 * self.TOU_price[int(i/12)])
        #     self.price.append(p)
        #
        # for i in range(288):
        #     waittime = np.random.normal(0.4, 0.1)
        #     while waittime < 0:
        #         waittime = 0
        #     self.waittime.append(waittime)

class EV:
    def __init__(self, id, t_start, soc, source, destination):
        self.id = id
        self.t_start = t_start
        self.curr_time = t_start
        self.curr_day = 0

        self.charging_effi = 0.9
        self.curr_SOC = soc
        self.init_SOC = soc
        self.final_soc = Final_SOC
        self.req_SOC = 0.0
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

        self.cschargingtime = 0.0
        self.cschargingcost = 0.0
        self.cschargingwaitingtime = 0.0
        self.cscharingenergy = 0.0
        self.cschargingprice = 0.0
        self.cschargingstarttime = 0.0
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
        self.rear_path = []

        self.predic_totaltraveltime = 0.0

        self.weightvalue = 0.0




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
        self.B = 20

        # 리플레이 메모리, 최대 크기 2000
        # self.memory_size = 40000
        # self.memory = deque(maxlen=4000)
        self.memory = deque(maxlen=4000)

        # 모델과 타깃 모델 생성
        self.model = self.build_model()
        self.target_model = self.build_model()

        # 타깃 모델 초기화
        self.update_target_model()


        if self.load_model:
            # self.model.load_weights("dqn_model.h5")
            self.model = load_model("double_dqn_model.h5")
            self.epsilon = 0.0

    # 상태가 입력, 큐함수가 출력인 인공신경망 생성
    def build_model(self):
        print('in agent', state_size, action_size)
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(480, kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

        model.add(Dense(250, kernel_initializer='he_uniform'))
        # model.add(BatchNormalization())
        model.add(Activation('relu'))

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
        self.memory.append((state, action, reward, next_state, done))


    # 리플레이 메모리에서 무작위로 추출한 배치로 모델 학습
    def train_model(self):
        if len(self.memory) < self.train_start:
            return
            # 메모리에서 배치 크기만큼 무작위로 샘플 추출
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.state_size))
        action, reward, done = [], [], []

        for i in range(batch_size):
            update_input[i] = mini_batch[i][0]
            action.append(mini_batch[i][1])
            reward.append(mini_batch[i][2])
            update_target[i] = mini_batch[i][3]
            done.append(mini_batch[i][4])

        target = self.model.predict(update_input)
        target_next = self.model.predict(update_target)
        target_val = self.target_model.predict(update_target)

        for i in range(self.batch_size):
            # like Q Learning, get maximum Q value at s'
            # But from target model
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                # the key point of Double DQN
                # selection of action is from model
                # update is from target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])

        # make minibatch which includes target q value and predicted q value
        # and do the model fit!
        self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)




class Env:
    def __init__(self, graph, state_size, action_size):
        self.graph = graph
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

        self.CS_list = reset_CS_info(self.graph)

    # def reset_CS_info(self):
    #     self.CS_list = []
    #     for l in self.graph.cs_info:
    #         cs = CS(l, self.graph.cs_info[l]['long'], self.graph.cs_info[l]['lat'])
    #         self.CS_list.append(cs)



    def reset(self):
        self.graph.reset_traffic_info()
        self.CS_list = reset_CS_info(self.graph)
        t_start, soc = default_setting_random_value()
        self.graph.source_node_set = list(self.graph.source_node_set)
        source = self.graph.source_node_set[np.random.randint(0, len(self.graph.source_node_set) - 1)]
        self.graph.destination_node_set = list(self.graph.destination_node_set)
        destination = self.graph.destination_node_set[np.random.randint(0, len(self.graph.destination_node_set) - 1)]
        while source==destination  or destination in self.graph.cs_info:
            destination = self.graph.destination_node_set[np.random.randint(0, len(self.graph.destination_node_set) - 1)]

        self.path_info = []
        self.pev = EV(e, t_start, soc, source, destination)
        self.sim_time = self.pev.t_start
        self.path_info = ta.get_feature_state(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)

        # print('\npev soc', self.pev.SOC)
        state =[self.pev.curr_location/self.graph.num_node, self.pev.destination/self.graph.num_node, self.pev.curr_SOC, self.sim_time/288]
        for path in self.path_info:
            cs, req_soc, driving_cost, fpath, rpath, fp_dist, rp_dist, fp_dt, rp_dt, fp_w, r_w, wtime, ctime, cs_char_cost, home_char_cost = path
            state += [wtime, ctime, driving_cost/10, cs_char_cost/10, home_char_cost/10]

        state = np.reshape(state, [1, self.state_size])

        return state, source, destination

    def test_reset(self, pev, graph, CS_list):
        self.graph = graph
        self.path_info = []
        self.CS_list = CS_list
        self.pev = pev
        self.sim_time = self.pev.t_start
        self.path_info = ta.get_feature_state(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)

        state = [self.pev.curr_location/self.graph.num_node, self.pev.destination/self.graph.num_node, self.pev.curr_SOC,  self.sim_time/288]
        for path in self.path_info:
            cs, req_soc, driving_cost, fpath, rpath, fp_dist, rp_dist, fp_dt, rp_dt, fp_w, r_w, wtime, ctime, cs_char_cost, home_char_cost = path
            state += [wtime, ctime, driving_cost/10, cs_char_cost/10, home_char_cost/10]
        state = np.reshape(state, [1, self.state_size])


        return state, self.pev.source, self.pev.destination


    def step(self, action):
        cs, req_soc, driving_cost, fpath, rpath, fp_dist, rp_dist, fp_dt, rp_dt, fp_w, r_w, wtime, ctime, cs_char_cost, home_char_cost = self.path_info[action]
        self.pev.path.append(self.pev.curr_location)

        if len(fpath)>1:
            next_node = fpath[1]
            chargingprice = cs.price[int(self.sim_time / 5)]
            distance_onetwo = self.graph.get_path_distance(fpath[0:2])
            velocity_onetwo = self.graph.velocity(fpath[0], fpath[1], int(self.sim_time / 5))
            croad = self.pev.ECRate * distance_onetwo * chargingprice
            troad = UNITtimecost * distance_onetwo / velocity_onetwo
            road_cost = croad + troad


            self.sim_time, time = ta.update_ev(self.pev, self.graph, self.pev.curr_location, next_node, self.sim_time)
            if self.sim_time == 0 and time == 0:
                done = 1
                reward = -50
                return np.zeros((1, self.state_size)), -1, reward, done
            if self.pev.curr_SOC <= 0.0:
                done = 1
                reward = -50
                return np.zeros((1, self.state_size)), -1, reward, done

            done = 0
            reward = -1*road_cost

            self.path_info = ta.get_feature_state(self.sim_time, self.pev, self.CS_list, self.graph, self.action_size)

            next_state = [self.pev.curr_location / self.graph.num_node, self.pev.destination / self.graph.num_node, self.pev.curr_SOC,  self.sim_time/288]
            for path in self.path_info:
                cs, req_soc, driving_cost, fpath, rpath, fp_dist, rp_dist, fp_dt, rp_dt, fp_w, r_w, wtime, ctime, cs_char_cost, home_char_cost = path
                next_state += [wtime, ctime, driving_cost/10, cs_char_cost/10, home_char_cost/10]
            next_state = np.reshape(next_state, [1, self.state_size])

            return next_state, next_node, reward, done

        elif self.pev.curr_location == cs.id:
            self.pev.charged = 1
            self.pev.cs = cs
            self.pev.csid = cs.id
            self.pev.req_SOC = req_soc
            self.pev.rear_path = rpath

            self.pev.cssoc = self.pev.curr_SOC
            self.pev.before_charging_SOC = self.pev.curr_SOC
            self.pev.cscharingenergy = self.pev.maxBCAPA * self.pev.req_SOC - self.pev.curr_SOC * self.pev.maxBCAPA
            self.pev.cschargingwaitingtime = cs.waittime[int(self.sim_time / 5)]
            self.pev.cschargingstarttime = self.sim_time + self.pev.cschargingwaitingtime * 60
            self.pev.cschargingprice = cs.price[int(self.pev.cschargingstarttime / 5)]
            self.pev.cschargingcost = self.pev.cscharingenergy * self.pev.cschargingprice

            self.pev.csdrivingtime = self.pev.totaldrivingtime
            self.pev.csdistance = self.pev.totaldrivingdistance
            self.pev.cschargingwaitingtime = self.pev.cschargingwaitingtime
            self.pev.cschargingtime = (self.pev.cscharingenergy / (cs.chargingpower * self.pev.charging_effi))

            self.pev.curr_SOC = self.pev.req_SOC
            self.sim_time += self.pev.cschargingwaitingtime * 60
            self.sim_time += self.pev.cschargingtime * 60

            self.pev.curr_time = self.sim_time

            rpath = self.pev.rear_path
            # print(rpath)
            for i in range(len(rpath) - 1):
                fnode = rpath[i]
                tnode = rpath[i + 1]
                # print(fnode, tnode, self.pev.curr_location)
                self.pev.path.append(tnode)
                self.sim_time, homedtime = ta.update_ev(self.pev, self.graph, fnode, tnode, self.sim_time)
                self.pev.homedrivingtime += homedtime

            # print(self.pev.curr_location,self.pev.destination)
            if self.pev.curr_location == self.pev.destination:
                self.pev.homesoc = self.pev.curr_SOC
                self.pev.homechargingstarttime = self.sim_time
                self.pev.homechargingenergy = self.pev.maxBCAPA * self.pev.final_soc - self.pev.curr_SOC * self.pev.maxBCAPA
                self.pev.homechargingtime =  (self.pev.homechargingenergy/(cs.homechargingpower*self.pev.charging_effi))
                self.pev.homedrivingdistance = self.graph.get_path_distance(rpath)
                self.pev.homechargingprice = cs.TOU_price[int(self.sim_time/60)]
                self.pev.homechargingcost = self.pev.homechargingenergy * self.pev.homechargingprice

                self.pev.curr_SOC = self.pev.final_soc
                self.sim_time += self.pev.homechargingtime
                self.pev.curr_time = self.sim_time

                self.pev.expense_time_part = (self.pev.totaldrivingtime + self.pev.cschargingwaitingtime + self.pev.cschargingtime) * UNITtimecost
                self.pev.expense_cost_part = self.pev.totaldrivingdistance * self.pev.ECRate * self.pev.cschargingprice + self.pev.cschargingcost + self.pev.homechargingcost

                self.pev.totalcost = self.pev.expense_time_part + self.pev.expense_cost_part

                done = 1
                # reward = -1 * ((self.pev.cschargingwaitingtime + self.pev.cschargingtime + rp_dt) * UNITtimecost + self.pev.homechargingcost)
                reward = -1 * self.pev.totalcost

            else:
                print('error final destination')
                input()


            return np.zeros((1,self.state_size)), -1, reward, done
        else:
            done = 1
            reward = -50
            return np.zeros((1, self.state_size)), -1, reward, done

def gen_test_envir_simple(num_evs, graph):

    # graph = Graph_simple_39()
    graph.reset_traffic_info()

    EV_list = []

    for e in range(num_evs):

        t_start, soc = default_setting_random_value()
        # CS_list = []
        CS_list = reset_CS_info(graph)
        # for l in graph.cs_info:
        #     # print('gen cs', l)
        #     # alpha = np.random.uniform(0.03, 0.07)
        #     cs = CS(l, graph.cs_info[l]['long'], graph.cs_info[l]['lat'])
        #     CS_list.append(cs)

        graph.source_node_set = list(graph.source_node_set)
        graph.destination_node_set = list(graph.destination_node_set)

        source = graph.source_node_set[np.random.randint(0, len(graph.source_node_set) - 1)]
        destination = graph.destination_node_set[np.random.randint(0, len(graph.destination_node_set) - 1)]

        while source==destination or destination in graph.cs_info:
            destination = graph.destination_node_set[np.random.randint(0, len(graph.destination_node_set) - 1)]


        # print(source, destination)

        ev = EV(e, t_start, soc, source, destination)
        EV_list.append(ev)



    return EV_list, CS_list, graph

def test_dqn(EV_list_DQN_REF, CS_list_DQN_REF, graph, env, agent ):

    agent.epsilon = 0
    episcore = 0
    for e, pev in enumerate(EV_list_DQN_REF):
        done = False
        score = 0

        state, source, destination = env.test_reset(pev, graph, CS_list_DQN_REF)
        print("\nEpi:", e, agent.epsilon)
        print(source,'->', destination)
        # print('sim time:', env.sim_time)

        while not done:
            action = agent.get_action(state)
            next_state, next_node, reward, done = env.step(action)
            score += reward
            state = next_state

            episcore += reward

        print('charged: ',pev.charged)
    # print('episcore: ', episcore)




if __name__ == "__main__":
    # npev = 39
    # EV_list, CS_list, graph = gen_test_envir_simple(npev)

    now_start = datetime.datetime.now()
    resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02} {5} {6} {7}'.format(now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second,TRAIN, EPISODES, EPS_DC)
    basepath = os.getcwd()
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)


    graph_train = Graph_simple_39()
    # graph_train = Graph_simple_100()

    action_size = len(graph_train.cs_info)*N_SOC
    state_size = action_size*5+4
    # graph_train = Graph_jeju('data/20191001_5Min_modified.csv')
    # graph_train = Graph_simple_100()

    print('S:{} A:{}'.format(state_size, action_size))



    scores, episodes, steps= [], [], []
    n_step = 0
    train_step = 0

    if TRAIN:
        agent = DQNAgent(state_size, action_size, False)
        for e in range(EPISODES):
            episcore = 0
            epistep = 0
            numsucc, numfail = 0 , 0
            env = Env(graph_train, state_size, action_size)
            print("\nEpi:", e, agent.epsilon, 'episcore:', episcore)

            for n in range(100):
                done = False
                score, step = 0, 0
                state, source, destination = env.reset()
                current_node = source
                # print(source, destination)
                while not done:
                    # print(state)
                    action = agent.get_action(state)
                    next_state, next_node, reward, done = env.step(action)
                    step += 1
                    n_step += 1
                    # print(done, next_node, reward)

                    agent.append_sample(state, action, reward, next_state, done)

                    if n_step >= agent.train_start:
                        agent.train_model()
                        train_step += 1
                    score += reward
                    state = next_state
                    if train_step > agent.B:
                        agent.update_target_model()
                        train_step = 0
                    if done:
                        if env.pev.charged >= 1.0:
                            numsucc += 1
                        else:
                            numfail += 1
                        step = len(env.pev.path)
                        print('({},{},{})'.format(env.pev.csid, env.pev.req_SOC, step), end='   ')
                        # print('(cs:{},S:{},SOC:{},Step:{})'.format(env.pev.csid, env.pev.source, env.pev.req_SOC, step))
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

        agent.model.save("{}/double_dqn_model.h5".format(resultdir))
        agent.model.save("double_dqn_model.h5")

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

    graph_test = Graph_simple_39()
    # graph_test = Graph_simple_100()

    # action_size = len(graph_test.cs_info) * N_SOC
    # state_size = action_size * 5 + 3

    testagent = DQNAgent(state_size, action_size, True)

    print(N_SOC, len(graph_test.cs_info), action_size, state_size)
    for i in range(10):

        npev = 1000
        EV_list, CS_list, graph_test = gen_test_envir_simple(npev, graph_test)
        # graph = graph_train

        env = Env(graph_test, state_size, action_size)

        EV_list_DQN_REF = copy.deepcopy(EV_list)
        CS_list_DQN_REF = copy.deepcopy(CS_list)
        test_dqn(EV_list_DQN_REF, CS_list_DQN_REF, graph_test, env, testagent)


        EV_list_Greedy = copy.deepcopy(EV_list)
        CS_list_Greedy = copy.deepcopy(CS_list)
        ta.greedy_total_cost_search(EV_list_Greedy, CS_list_Greedy, graph_test)

        EV_list_Greedy_short = copy.deepcopy(EV_list)
        CS_list_Greedy_short = copy.deepcopy(CS_list)
        ta.greedy_shortest_search(EV_list_Greedy_short, CS_list_Greedy_short, graph_test)
        #
        # EV_list_optimal_ref = copy.deepcopy(EV_list)
        # CS_list_optimal_ref = copy.deepcopy(CS_list)
        # EV_list_optimal_ref = ta.optimal_solution(EV_list_optimal_ref, CS_list_optimal_ref, graph_test)
        #
        # EV_list_Astar_time = copy.deepcopy(EV_list)
        # CS_list_Astar_time = copy.deepcopy(CS_list)
        # ta.every_time_check_refer_time(EV_list_Astar_time, CS_list_Astar_time, graph)
        #
        # EV_list_Astar_shortest = copy.deepcopy(EV_list)
        # CS_list_Astar_shortest = copy.deepcopy(CS_list)
        # ta.every_time_check_refer_shortest(EV_list_Astar_shortest, CS_list_Astar_shortest, graph)



        # ta.sim_result_text_ref(i, CS_list, graph, resultdir, EV_list_DQN_REF=EV_list_DQN_REF, EV_list_Astar_ref=EV_list_Astar_ref, EV_list_Astar_shortest=EV_list_Astar_shortest)
        ta.sim_result_text_last(i, CS_list, graph_test, resultdir, EV_list_DQN_REF=EV_list_DQN_REF, EV_list_Greedy=EV_list_Greedy, EV_list_Greedy_short=EV_list_Greedy_short)
        # ta.sim_result_general_presentation_ref(i, graph, resultdir, npev,  EV_list_DQN_REF=EV_list_DQN_REF, EV_list_Greedy=EV_list_Greedy)


        # ta.sim_result_general_presentation_ref(i, graph, resultdir, npev, EV_list_DQN_REF=EV_list_DQN_REF,EV_list_Astar_ref=EV_list_Astar_ref, EV_list_Astar_shortest=EV_list_Astar_shortest)
        # ta.sim_result_general_presentation_ref(i, graph, resultdir, npev, EV_list_Astar_time=EV_list_Astar_time,EV_list_Astar_ref=EV_list_Astar_ref, EV_list_Astar_shortest=EV_list_Astar_shortest)

