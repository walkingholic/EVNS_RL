import numpy as np
import random as rnd
from dqn_v06_EV_fleet import EV,CS, Env

import dqn_v06_EV_fleet as dqn

# from dqn_v03 import Env
import matplotlib.pyplot as plt
import datetime
# import gurobipy as gp
# from gurobipy import GRB, Model, quicksum
from Graph import Graph_simple_39
from Graph import Graph_jejusi
import copy
import os
import datetime
import test_algorithm as ta





now_start = datetime.datetime.now()
resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(now_start.month, now_start.day, now_start.hour, now_start.minute, now_start.second)
basepath = os.getcwd()
dirpath = os.path.join(basepath, resultdir)
ta.createFolder(dirpath)

for i in range(2):

    graph_test = Graph_jejusi()
    npev = 300
    EV_list, CS_list, graph_test = dqn.gen_test_envir_simple(npev, graph_test)
    # graph = graph_train


    EV_list_Greedy = copy.deepcopy(EV_list)
    CS_list_Greedy = copy.deepcopy(CS_list)
    ta.get_greedy_time_cost_fleet(EV_list_Greedy, CS_list_Greedy, graph_test)


    tot_wt = 0
    tot_cost = 0
    for pev in EV_list_Greedy:
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost
    print('==========EV_list_Greedy===================')
    print('Avg. total waiting time: ', tot_wt / len(EV_list_Greedy))
    print('Total cost: ', tot_cost)


    EV_list_short_Greedy = copy.deepcopy(EV_list)
    CS_list_short_Greedy = copy.deepcopy(CS_list)
    ta.get_greedy_shortest_fleet(EV_list_short_Greedy, CS_list_short_Greedy, graph_test)


    tot_wt = 0
    tot_cost = 0
    for pev in EV_list_short_Greedy:
        tot_wt += pev.true_waitingtime
        tot_cost += pev.totalcost
    print('==========EV_list_Greedy===================')
    print('Avg. total waiting time: ', tot_wt / len(EV_list_short_Greedy))
    print('Total cost: ', tot_cost)



    ta.sim_result_text_fleet(i, CS_list, graph_test, resultdir, EV_list_Greedy=EV_list_Greedy, EV_list_short_Greedy=EV_list_short_Greedy)












#
#
# num_request = 150
# request_be_EV = []
# request_ing_EV = []
# charging = []
#
# graph = Graph_jejusi()
# arr_time = np.random.uniform(360, 1200, num_request)
# arr_time.sort()
#
# for i in range(num_request):
#     s = np.random.randint(1, 40)
#     d =  np.random.randint(1, 40)
#     soc = np.random.uniform(0.2, 0.4)
#     req_soc = np.random.uniform(0.7, 0.9)
#     t_start = arr_time[i]
#     request_be_EV.append(EV(i, s, d, soc, req_soc, t_start))
#
# CS_list = []
# profit = np.random.uniform(0.7, 1.3, len(graph.cs_info))
#
# for i, l in enumerate(graph.cs_info):
#     cs = CS(l, profit[i], graph.cs_info[l]['long'], graph.cs_info[l]['lat'])
#     CS_list.append(cs)
#
#
# # CS_list=CS_list[:1]
# for pev in request_be_EV:
#     # charging_energy = pev.maxBCAPA * (pev.req_SOC - pev.curr_SOC)
#     # charging_duration = (charging_energy / (60 * pev.charging_effi))
#     # pev.ept_charging_duration = charging_duration * 60
#
#     print('=====================================================================')
#     print('be ID:{0:3} S:{1:3} D:{2:3} CurSOC:{3:0.2f} ReqSOC:{4:0.2f} Tstart:{5:0.2f} Tarr:{6:0.2f}'
#           .format(pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime))
#     candi=[]
#     candi = ta.get_feature_state_fleet(pev.t_start, pev, CS_list, graph, 0)
#
#     for cs,_,_,_,_,_,_,_,_,_,_,eptWT,ept_charduration,_,_,ept_arrtime in candi:
#         print('ID: {0:3}  eptWT: {1:.2f}   eptCharduration: {3:.2f}   Len_reserv: {2:.2f}'.format(cs.id, eptWT, len(cs.reserve_ev), ept_charduration))
#
#     candi.sort(key=lambda e: e[1])
#
#     (cs, weight, ept_driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,ept_front_d_time,
#      ept_rear_d_time, fpath_weight, rpath_weight, ept_WT, ept_charduration, ept_cs_charging_cost, ept_home_charging_cost,
#      ept_arrtime )= candi[0]
#
#     pev.front_path = front_path
#     pev.rear_path = rear_path
#     pev.path = front_path + rear_path[1:]
#     pev.ept_arrtime = ept_arrtime
#     pev.true_arrtime = ta.get_true_arrtime(pev, cs, graph)
#     pev.ept_waitingtime = ept_WT
#     pev.ept_charging_duration = ept_charduration
#     pev.cs = cs
#     pev.cschargingprice = cs.price[int(pev.cschargingstarttime/5)]
#     cs.recieve_request(pev)
#
#
# for cs in CS_list:
#     cs.sim_finish(graph)
#     print('=====================================================================')
#
# tot_wt = 0
# tot_cost=0
# for pev in request_be_EV:
#     print('result,  ID: {0:3},  CSID: {1:3},  CurSOC: {2:.2f},  ReqSOC: {3:.2f},  Tstart: {4:5.2f},  EptTarr: {5:5.2f},  TruTarr: {10:5.2f},  diffTarr: {11:5.2f},  WT: {6:5.2f},  eptWT: {8:5.2f},  diffWT: {9:5.2f},  ChaStart: {7:5.2f},  finTime: {12:5.2f}'
#           .format(pev.id, pev.cs.id, pev.curr_SOC, pev.req_SOC, pev.t_start, pev.ept_arrtime, pev.true_waitingtime, pev.cschargingstarttime, pev.ept_waitingtime, pev.time_diff_WT, pev.true_arrtime, pev.true_arrtime-pev.ept_arrtime, pev.curr_time))
#     tot_wt += pev.true_waitingtime
#     tot_cost+= pev.totalcost
#
# print('Avg. total waiting time: ', tot_wt/num_request)
# print('Total cost: ', tot_cost)





# for cs in CS_list:
#     for cp in cs.cplist:
#         print('[ CS: {}\tPole: {} ]'.format(cs.id, cp.id))
#         for ev in cp.charging_ev:
#             print('(ID: {0:3}\tS: {5:3}\tD: {6:3}\tRsvT: {4:4.2f}\tTrueTarr: {1:4.2f}\tTchStart: {2:4.2f}\tTchEnd: {3:4.2f})'.format(ev.id, ev.true_arrtime, ev.cschargingstarttime, ev.cschargingstarttime+ev.true_charging_duration, ev.t_start, ev.source, ev.destination))
#             print('--> path: ', ev.path)
#             print('--> cost: ', ev.totalcost, ev.expense_cost_part, ev.expense_time_part)
#
#         print()
#     print('=====================================================================')





#
#
#
#
# charging.sort(key= lambda element: element[1])



#
# for pev, s in charging:
#     print('{} - {} {} {} {} {} {} {}'.format(s, pev.id, pev.source, pev.destination, pev.curr_SOC, pev.req_SOC, pev.t_start,
#                                             pev.path))








# print(arr_time)
#
# for i, pev in zip(range(len(arr_time)), arr_time):
#     print(i, pev)
#     arr_time = arr_time[1:]
#     print(arr_time)


# waittime = []
# for i in range(10):
#     waittime.append(int(np.random.normal(10, 10*0.2)))
#     print(waittime)
#
# rnd = np.random
# rnd.seed(0)
#
# n = 39
# xc = rnd.rand(n+1)*200
# yc = rnd.rand(n+1)*100
# print(xc)
# print(yc)
# plt.plot(xc[0], yc[0], c='r', marker='s')
# plt.scatter(xc[1:],yc[1:])
#
#
# N = [i for i in range(1, n+1)]
# V = [0] + N
# A = [(i, j) for i in V for j in V if i != j]
# c = {(i, j): np.hypot(xc[i]-xc[j], yc[i]-yc[j]) for i, j in A}
# Q = 20
# q = {i: rnd.randint(1, 10) for i in N}
#
#
# model = Model('Test')
# x = model.addVars(A, vtype=GRB.BINARY)
# u = model.addVars(N, vtype=GRB.CONTINUOUS)
#
# model.modelSense = GRB.MINIMIZE
# model.setObjective(quicksum( x[i,j]*c[i,j]  for i,j in A ))
#
# model.addConstrs(quicksum (x[i,j] for j in V if j!=i) ==1 for i in N );
# model.addConstrs(quicksum(x[i,j] for i in V if i!=j)==1 for j in N );
# model.addConstrs((x[i,j]==1) >> (u[i]+q[j] == u[j]) for i,j in A if i!=0 and j!=0);
# model.addConstrs(u[i]>=q[i] for i in N);
# model.addConstrs(u[i]<=Q for i in N);
#
# # model.Params.MIPGap = 0.1
# # model.Params.TimeLimit = 30  # seconds
# # model.optimize()
#
# model.optimize()
#
#
# active_arcs = [a for a in A if x[a].x > 0.99]

#
# for i, j in active_arcs:
#     plt.plot([xc[i], xc[j]], [yc[i], yc[j]], c='g', zorder=0)
# plt.plot(xc[0], yc[0], c='r', marker='s')
# plt.scatter(xc[1:], yc[1:], c='b')
# plt.show()


#
# class CS:
#     def __init__(self, node_id, long, lat, alpha):
#         self.id = node_id
#         self.price = list()
#         self.waittime = list()
#         self.chargingpower = 60 # kw
#         self.alpha = alpha
#         self.x = long
#         self.y = lat
#
#         for i in range(288):
#             p = np.random.normal(alpha, 0.15 * alpha)
#             while p < 0:
#                 p = np.random.normal(alpha, 0.15 * alpha)
#             self.price.append(p)
#
#         for i in range(288):
#             waittime = np.random.normal(-1200 * (self.price[i] - 0.07), 20)
#             while waittime < 0:
#                 waittime = 0
#             self.waittime.append(waittime/60)
#
# class Env:
#     def __init__(self, pev):
#         self.pev = pev
#
# class Ev:
#     def __init__(self, s, d, soc):
#         self.source = s
#         self.destination = d
#         self.soc = soc
#
#
#
# graph = Graph_simple_39()
# CS_list = []
# for l in graph.cs_info:
#     # print('gen cs')
#     # alpha = np.random.uniform(0.03, 0.07)
#     alpha = np.random.uniform(0.03, 0.07)
#
#     cs = CS(l, graph.cs_info[l]['long'], graph.cs_info[l]['lat'], alpha)
#     CS_list.append(cs)
#
# #
# # for i in range(288):
# #     waittime = np.random.normal(0.4, 0.08)
# #     while waittime < 0:
# #         waittime = 0
# #     waittime.append(waittime)
# TOU_price = [0.1736, 0.1601, 0.1748, 0.174, 0.1724, 0.1735, 0.1601, 0.1736, 0.1321, 0.1618, 0.1616, 0.1650, 0.161, 0.1635, 0.1650, 0.1633, 0.1749, 0.1808, 0.1808, 0.1753, 0.1739, 0.1717, 0.1823, 0.1786, 0.1823, 0.1321, 0.1687]
# wait, price = [], []
# for i in range(288):
#     # price.append(TOU_price[int(i/12)])
#     wait.append(0.2307)
#     price.append(0.1736)
#
# print(wait)
# print(price)


#
# wa = np.random.normal(0.4, 0.08, 288)
# for w in wa:
#     print('{0:2.4f}'.format(w), end=',')
#     print('0.2307', end=',')
# fig, ax = plt.subplot()
# # fig, ax = plt.subplots(2, figsize=(8,8))
# # ax[0].plot(xx, yy)
# # ax[1].plot(xx, yy)
# ax.plot(xx, yy)
# plt.show()

#
# plt.title('from {} '.format(pev.source))
# plt.xlim(graph.minx, graph.maxx)
# plt.ylim(graph.miny, graph.maxy)
#
#


#
# xx = [1, 2, 3, 4]
# yy = [11, 12, 8, 12]
#
# fig = plt.figure()
# ax1 = fig.add_subplot(2, 2, 1)
# ax1.plot(xx, yy)
#
# ax1 = fig.add_subplot(2, 2, 3)
# ax1.set_title('2')
# ax1.plot(xx, yy)
#
# ax1 = fig.add_subplot(2, 2, 4)
# key = ['a', 'b', 'c']
# data = [sum(xx), sum(yy), sum(xx)]
# ax1.bar(key,data)
# plt.show()
# #
# plt.title('from {} '.format(pev.source))
# plt.xlim(graph.minx, graph.maxx)
# plt.ylim(graph.miny, graph.maxy)
# plt.legend()
# fig = plt.gcf()
# fig.savefig('{}/route_{}.png'.format(resultdir, i), facecolor='#eeeeee', dpi=300)
# plt.clf()
#
# try:
#     # Create a new  model
#     m = gp.Model("mip1")
#
#     #Create  variables
#     x = m.addVar(vtype=GRB.BINARY , name="x")
#     y = m.addVar(vtype=GRB.BINARY , name="y")
#     z = m.addVar(vtype=GRB.BINARY , name="z")
#
#     # Set  objective
#     m.setObjective(x + y + 2 * z, GRB.MAXIMIZE)
#
#     # Add  constraint: x + 2 y + 3 z  <= 4
#     m.addConstr(x + 2 * y + 3 * z  <= 4, "c0")
#
#     # Add  constraint: x + y  >= 1
#     m.addConstr(x + y  >= 1, "c1")
#
#     # Optimize  model
#     m.optimize ()
#
#     for v in m.getVars ():
#       print('%s %g' % (v.varName , v.x))
#     print('Obj: %g' % m.objVal)
# except gp.GurobiError  as e:
#     print('Error  code'  +str(e.errno) + ':' +str(e))
#
# except AttributeError:
#     print('Encountered  an  attribute  error ')
#
#


#
# nowa = datetime.datetime.now()
# resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(nowa.month, nowa.day, nowa.hour, nowa.minute, nowa.second)
#
# print(resultdir)
#
# input()
#
# now = datetime.datetime.now()
# resultdir = '{0:02}-{1:02} {2:02}-{3:02} {4:02}'.format(now.month, now.day, now.hour, now.minute, now.second)
#
# print(now - nowa)

#
# epi = [1,2,3,4,5,6,7,8]
#
# print(sum(epi))
# sco = [5,3,1,2,3,4,6,7]
#
# for i, e in enumerate(epi):
#     plt.plot(i, e, '.')
#     print(i, e)
# plt.show()

# state = [11, 22]
# for i in range(3):
#     state += [i, 222, 333]
#
# state = np.reshape(state, [1, len(state)])
# print(state, state.shape)

# t = {}
#
# for i in range(10):
#     t[i] = random.uniform(0,100)
#
# print(t)
# print(t[7])


# e = Env()
# state, source, destination, CS_list = e.reset()
# cs = CS_list[0]
# print(cs.price)
#
#
# state, source, destination, CS_list = e.reset()
# cs = CS_list[0]
# print(cs.price)

# state, source, destination, CS_list = e.reset(1)
# cs = CS_list[0]
# print(cs.price)




#
# cur = np.zeros((5, 5))
# dest = np.zeros((5, 5))
# cur[0, 0] = 1
# dest[4, 4] = 1
# next_state = np.concatenate((cur, dest), axis=0)
# # cur = np.reshape(cur, [1, 25])
# # dest = np.reshape(dest, [1, 25])
# print(next_state)
# next_state = np.concatenate((cur, dest), axis=1)
# # cur = np.reshape(cur, [1, 25])
# # dest = np.reshape(dest, [1, 25])
# print(next_state)
#
# next_state = np.reshape(next_state, [1, 50])
# print(next_state)
#
# print(next_state)


# print(next_state.T)

# test = np.zeros((10,10,2))
#
# test[1,1,0]=1
# test[5,5,1]=1
#
# print(test[:,:,0])
# print(test[:,:,1])
#
# test = np.reshape(test, [1, test.shape[0],test.shape[1],test.shape[2] ] )
# print(test.shape)


#
# test = np.array(range(100))
# test = np.reshape(test, (10,10))
# print(test)
# print(np.shape(test))
#
# print(test[2,1])

# test = np.array(range(100))
# print(test)
# print(np.shape(test))
#
#
# print(test)
# print(np.shape(test))
#
# print(test.shape)
# print(np.shape(test))

# test  = [2,4,3,5,8]
#
# if 1 in test:
#     print('dd')
# def one_hot(x):
#     return np.identity(100)[x:x + 1]
# s = one_hot(0)
# d = one_hot(99)
#
# print(s+d)

# aa = []
#
# aa.append((0, 11))
# aa.append((2, 22))
# aa.append((3, 33))
#
# print(random.choice(range(4)))

