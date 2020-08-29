import sys

import pylab
import random
import numpy as np

import matplotlib.pyplot as plt
import copy
import test_algorithm as ta
import datetime
import os
from SumTree import SumTree
from time import time
from Graph import Graph_simple_6
from haversine import haversine
from dqn_v04_my_work_per import *
import heapq
import queue


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]



class State:
    def __init__(self, n, time, path):
        self.node = n
        self.vlist = path
        self.sim_time = time


def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b

    return abs(x1 - x2) + abs(y1 - y2)

def dijkstra_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.weight(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def a_star_search(graph, start, goal):

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    while not frontier.empty():
        current = frontier.get()
        # print('frontier.get()', current)
        if current == goal:
            break
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.weight(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                # print('frontier.put()', next, priority)
                came_from[next] = current

    return came_from, cost_so_far
def heuristic_astar(a, b):
    (x1, y1) = a
    (x2, y2) = b

    x1, y1 = a
    x2, y2 = b

    dist = haversine((y1, x1), (y2, x2), unit='km')
    return dist
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        # print('path.append(current)', current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

def update_envir_ref_weight(cs, graph, sim_time):

    time_idx = int(sim_time / 5)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        croad = (ECRate * graph.link_data[l_id]['LENGTH'] )* cs.price[time_idx]
        troad = (graph.link_data[l_id]['LENGTH']/velo )* UNITtimecost
        graph.link_data[l_id]['WEIGHT'] = croad+troad


#
# def optimal_solution(EV_list, CS_list, graph):
#     pev_list = []
#     for i, pev in enumerate(EV_list):
#         print(i, pev.source)
#         path = [pev.source]
#         pev.path = path
#         bestsolution = PriorityQueue()
#
#         for cs in CS_list:
#             ev = copy.deepcopy(pev)
#             # print()
#             # print(cs.id, ev.curr_time, ev.SOC, ev.drivingdistance, ev.path)
#             bucket = queue.Queue()
#             bucket.put(ev)
#             bestPev = ev
#             bestPev.totalcost = 10000
#
#             while bucket.qsize():
#                 ev = bucket.get()
#                 if ev.curr_location == cs.id:
#                     sim_time = ev.curr_time
#                     ev.charged = 1
#                     ev.before_charging_SOC = ev.SOC
#
#                     ev.charingenergy = ev.maxBCAPA * ev.req_SOC - ev.SOC * ev.maxBCAPA
#                     ev.chargingtime = (ev.charingenergy / (cs.chargingpower * ev.charging_effi))
#                     ev.waitingtime = cs.waittime[int(sim_time / 5)]
#                     ev.SOC = ev.req_SOC
#                     ev.cs = cs
#                     ev.fdist = ev.drivingdistance
#                     ev.chargingstarttime = sim_time + ev.waitingtime * 60
#                     ev.chargingprice = cs.price[int(ev.chargingstarttime / 5)]
#                     ev.chargingcost = ev.charingenergy * ev.chargingprice
#
#                     ev.to_cs_driving_time = ev.drivingtime
#                     ev.to_cs_dist = ev.drivingdistance
#                     ev.to_cs_waiting_time = ev.waitingtime
#                     ev.to_cs_charging_time = ev.chargingtime
#                     ev.to_cs_soc = ev.SOC
#
#                     ev.expense_time_part = (ev.drivingtime + ev.waitingtime) * UNITtimecost
#                     ev.expense_cost_part = ev.drivingdistance * ev.ECRate * ev.chargingprice + ev.chargingcost
#                     ev.totalcost = ev.expense_time_part + ev.expense_cost_part
#
#                     sim_time += ev.chargingtime * 60
#                     sim_time += cs.waittime[int(sim_time / 5)] * 60
#                     # print(bestPev.totalcost, ev.totalcost)
#                     if ev.totalcost < bestPev.totalcost:
#                         bestPev = ev
#                 else:
#                     cnode = ev.curr_location
#                     for next in graph.neighbors(cnode):
#                         newpath = copy.copy(ev.path)
#                         if next not in newpath:
#                             ctime = ev.curr_time
#                             newpath.append(next)
#                             newev = copy.deepcopy(ev)
#                             newev.path = newpath
#                             newev.curr_time, _ = ta.update_ev(newev, graph, cnode, next, ctime)
#
#                             charingenergy = newev.maxBCAPA * (newev.req_SOC - newev.SOC)
#                             cost = newev.drivingtime * UNITtimecost + newev.energyconsumption *  min(cs.price)
#                             # cost += max(cs.price) * charingenergy
#                             if newev.SOC > 0.0 and cost <= bestPev.totalcost:
#                                 # print(cost, bestPev.totalcost)
#                                 bucket.put(newev)
#                             # else:
#                             #     print(newev.SOC, 'SOC not enough')
#                             #     print(cost, bestPev.totalcost)
#
#             bestsolution.put(bestPev, bestPev.totalcost)
#
#         pev = bestsolution.get()
#         pev_list.append(pev)
#     return pev_list

def optimal_solution_v02(EV_list, CS_list, graph):
    pev_list = []
    for i, pev in enumerate(EV_list):
        print(i, pev.source)
        path = [pev.source]
        pev.path = path
        bestsolution = PriorityQueue()

        for cs in CS_list:
            ev = copy.deepcopy(pev)
            bucket = queue.Queue()
            bucket.put(ev)
            bestPev = ev
            bestPev.totalcost = 10000

            while bucket.qsize():
                ev = bucket.get()
                if ev.curr_location == cs.id:
                    sim_time = ev.curr_time
                    ev.charged = 1
                    ev.before_charging_SOC = ev.curr_SOC

                    ev.cscharingenergy = ev.maxBCAPA * ev.req_SOC - ev.curr_SOC * ev.maxBCAPA
                    ev.cschargingtime = (ev.cscharingenergy / (cs.chargingpower * ev.charging_effi))
                    ev.cschargingwaitingtime = cs.waittime[int(sim_time / 5)]
                    ev.curr_SOC = ev.req_SOC
                    ev.cs = cs
                    ev.fdist = ev.totaldrivingdistance
                    ev.cschargingstarttime = sim_time + ev.cschargingwaitingtime * 60
                    ev.cschargingprice = cs.price[int(ev.cschargingstarttime / 5)]
                    ev.cschargingcost = ev.cscharingenergy * ev.cschargingprice

                    ev.csdrivingtime = ev.totaldrivingtime
                    ev.csdistance = ev.totaldrivingdistance
                    ev.cschargingwaitingtime = ev.cschargingwaitingtime
                    ev.cschargingtime = ev.cschargingtime
                    ev.cssoc = ev.curr_SOC

                    ev.expense_time_part = (ev.totaldrivingtime + ev.cschargingwaitingtime) * UNITtimecost
                    ev.expense_cost_part = ev.totaldrivingdistance * ev.ECRate * ev.cschargingprice + ev.cschargingcost
                    ev.totalcost = ev.expense_time_part + ev.expense_cost_part

                    sim_time += ev.cschargingtime * 60
                    sim_time += cs.waittime[int(sim_time / 5)] * 60
                    if ev.totalcost < bestPev.totalcost:
                        bestPev = ev
                else:
                    cnode = ev.curr_location
                    for next in graph.neighbors(cnode):
                        newpath = copy.copy(ev.path)
                        if next not in newpath:
                            ctime = ev.curr_time
                            newpath.append(next)
                            newev = copy.deepcopy(ev)
                            newev.path = newpath
                            newev.curr_time, _ = ta.update_ev(newev, graph, cnode, next, ctime)

                            charingenergy = newev.maxBCAPA * (newev.req_SOC - newev.curr_SOC)
                            cost = newev.totaldrivingtime * UNITtimecost + newev.totalenergyconsumption * min(cs.price)
                            cost += min(cs.price) * charingenergy
                            if newev.curr_SOC > 0.0 and cost <= bestPev.totalcost:
                                # print(cost, bestPev.totalcost)
                                bucket.put(newev)
                            # else:
                            #     print(newev.SOC, 'SOC not enough')
                            #     print(cost, bestPev.totalcost)

            bestsolution.put(bestPev, bestPev.totalcost)

        pev = bestsolution.get()
        pev_list.append(pev)
    return pev_list


def optimal_current_solution(EV_list, CS_list, graph):
    pev_list = []
    for i, pev in enumerate(EV_list):
        print(i, pev.source)
        path = [pev.source]
        pev.path = path
        # bestsolution = []
        bestsolution = PriorityQueue()

        for cs in CS_list:
            ev = copy.deepcopy(pev)
            bucket = queue.Queue()
            bucket.put(ev)
            bestPev = ev
            bestPev.totalcost = 10000

            while bucket.qsize():
                ev = bucket.get()
                ctime = ev.curr_time
                if ev.curr_location == cs.id:

                    ev.charged = 1
                    ev.before_charging_SOC = ev.curr_SOC

                    ev.cscharingenergy = ev.maxBCAPA * ev.req_SOC - ev.curr_SOC * ev.maxBCAPA
                    ev.cschargingtime = (ev.cscharingenergy / (cs.chargingpower * ev.charging_effi))
                    ev.cschargingwaitingtime = cs.waittime[int(ctime / 5)]
                    ev.curr_SOC = ev.req_SOC
                    ev.cs = cs
                    ev.fdist = ev.totaldrivingdistance
                    ev.cschargingstarttime = ctime + ev.cschargingwaitingtime * 60
                    ev.cschargingprice = cs.price[int(ctime / 5)]
                    ev.cschargingcost = ev.cscharingenergy * ev.cschargingprice

                    ev.csdrivingtime = ev.totaldrivingtime
                    ev.csdistance = ev.totaldrivingdistance
                    ev.cschargingwaitingtime = ev.cschargingwaitingtime
                    ev.cschargingtime = ev.cschargingtime
                    ev.cssoc = ev.curr_SOC

                    ev.expense_time_part = (ev.totaldrivingtime + ev.cschargingwaitingtime) * UNITtimecost
                    ev.expense_cost_part = ev.totaldrivingdistance * ev.ECRate * ev.cschargingprice + ev.cschargingcost
                    ev.totalcost = ev.expense_time_part + ev.expense_cost_part

                    if ev.totalcost < bestPev.totalcost:
                        bestPev = ev

                else:
                    cnode = ev.curr_location
                    for next in graph.neighbors(cnode):
                        newpath = copy.copy(ev.path)
                        if next not in newpath:

                            newpath.append(next)
                            newev = copy.deepcopy(ev)
                            newev.path = newpath
                            _, _ = ta.update_ev(newev, graph, cnode, next, ctime)
                            charingenergy = newev.maxBCAPA *(newev.req_SOC - newev.curr_SOC)
                            cost = newev.totaldrivingtime * UNITtimecost + newev.totalenergyconsumption * cs.price[int(ctime / 5)]
                            cost += cs.price[int(ctime / 5)] * charingenergy
                            if newev.curr_SOC > 0.0 and cost < bestPev.totalcost:
                                # print(cost, bestPev.totalcost)
                                bucket.put(newev)
                            # else:
                            #     print(newev.SOC, 'SOC not enough')
                            #     print(cost, bestPev.totalcost)
            bestsolution.put(bestPev, bestPev.totalcost)
            # bestsolution.append(bestPev)

        print(len(bestsolution.elements))
        pev = bestsolution.get()
        # print('\nCS', pev.cs.id)
        # print('totalcost', pev.totalcost)
        pev_list.append(pev)
    return pev_list

def astar_solution(EV_list, CS_list, graph):
    pev_list = []
    for i, ev in enumerate(EV_list):

        sim_time = ev.t_start
        start = ev.curr_location
        end = ev.destination
        bestsolution = PriorityQueue()


        for cs in CS_list:
            pev = copy.deepcopy(ev)
            update_envir_ref_weight(cs, graph, sim_time)

            pev.cs = cs
            came_from, cost_so_far = a_star_search(graph, start, cs.id)
            front_path = reconstruct_path(came_from, start, cs.id)
            front_path_distance = graph.get_path_distance(front_path)

            pev.charged = 1
            pev.before_charging_SOC = pev.curr_SOC
            pev.path = front_path

            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            pev.cschargingcost = charging_energy * cs.price[int(sim_time / 5)]
            charging_time = (charging_energy / (cs.chargingpower * pev.charging_effi))
            pev.cschargingwaitingtime = cs.waittime[int(sim_time / 5)]
            pev.totaldrivingtime = graph.get_path_drivingtime(front_path, int(sim_time / 5))
            pev.totaldrivingdistance = graph.get_path_distance(front_path)
            pev.cschargingprice = cs.price[int(sim_time / 5)]

            pev.expense_time_part = (pev.totaldrivingtime + pev.cschargingwaitingtime) * UNITtimecost
            pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost
            pev.totalcost = pev.expense_time_part + pev.expense_cost_part

            bestsolution.put(pev, pev.totalcost)
            # bestsolution.append(bestPev)

        pev = bestsolution.get()
        # print('\nCS', pev.cs.id)
        # print('totalcost', pev.totalcost)
        pev_list.append(pev)
    return pev_list
def dijkstra_solution(EV_list, CS_list, graph):
    pev_list = []
    for i, ev in enumerate(EV_list):

        sim_time = ev.t_start
        start = ev.curr_location
        end = ev.destination
        bestsolution = PriorityQueue()


        for cs in CS_list:
            pev = copy.deepcopy(ev)
            update_envir_ref_weight(cs, graph, sim_time)

            pev.cs = cs
            came_from, cost_so_far = dijkstra_search(graph, start, cs.id)
            front_path = reconstruct_path(came_from, start, cs.id)
            front_path_distance = graph.get_path_distance(front_path)

            pev.charged = 1
            pev.before_charging_SOC = pev.curr_SOC
            pev.path = front_path

            remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
            charging_energy = pev.maxBCAPA * pev.req_SOC - remainenergy

            pev.cschargingcost = charging_energy * cs.price[int(sim_time / 5)]
            charging_time = (charging_energy / (cs.chargingpower * pev.charging_effi))
            pev.cschargingwaitingtime = cs.waittime[int(sim_time / 5)]
            pev.totaldrivingtime = graph.get_path_drivingtime(front_path, int(sim_time / 5))
            pev.totaldrivingdistance = graph.get_path_distance(front_path)
            pev.cschargingprice = cs.price[int(sim_time / 5)]

            pev.expense_time_part = (pev.totaldrivingtime + pev.cschargingwaitingtime) * UNITtimecost
            pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost
            pev.totalcost = pev.expense_time_part + pev.expense_cost_part

            bestsolution.put(pev, pev.totalcost)
            # bestsolution.append(bestPev)

        pev = bestsolution.get()
        # print('\nCS', pev.cs.id)
        # print('totalcost', pev.totalcost)
        pev_list.append(pev)
    return pev_list

if __name__ == "__main__":
    # graph_train = Graph_simple()
    graph_train = Graph_simple_39()



    # optimal = optimal_solution(EV_list, CS_list, graph)
    #
    # for pev in optimal:
    #     print(pev.totalcost, end='\t')

    # optimal_v02 = optimal_solution_v02(EV_list, CS_list, graph)
    # for pev in optimal_v02:
    #     print(pev.totalcost, end='\t')

    npev = 10
    EV_list, CS_list, graph = gen_test_envir_simple(npev, graph_train)
    # opti_list = optimal_current_solution(EV_list, CS_list, graph)

    EV_list_Greedy = copy.deepcopy(EV_list)
    CS_list_Greedy = copy.deepcopy(CS_list)
    ta.greedy_total_cost_search(EV_list_Greedy, CS_list_Greedy, graph)
    for pev in EV_list_Greedy:
        print(pev.totalcost, pev.req_SOC)

    print('================================================================================')
    EV_list_Optimal = copy.deepcopy(EV_list)
    CS_list_Optimal = copy.deepcopy(CS_list)
    EV_list_Optimal = ta.optimal_solution(EV_list_Optimal, CS_list_Optimal, graph)
    for pev in EV_list_Optimal:
        print(pev.totalcost, pev.req_SOC)

    # time_optimal, time_dijk, time_astar = [], [], []
    #
    # for i in range(20):
    #     npev = (i+1)*5
    #     EV_list, CS_list, graph = gen_test_envir_simple(npev, graph_train)
    #
    #     now_start = datetime.datetime.now()
    #     opti_list = optimal_current_solution(EV_list, CS_list, graph)
    #     now = datetime.datetime.now()
    #     opti_time = now - now_start
    #     time_optimal.append(opti_time)
    #
    #     now_start = datetime.datetime.now()
    #     astar_list = astar_solution(EV_list, CS_list, graph)
    #     now = datetime.datetime.now()
    #     astar_time = now - now_start
    #     time_astar.append(astar_time)
    #
    #     now_start = datetime.datetime.now()
    #     dijkstra_list = dijkstra_solution(EV_list, CS_list, graph)
    #     now = datetime.datetime.now()
    #     dij_time = now - now_start
    #     time_dijk.append(dij_time)
    #
    #
    # for time in time_optimal:
    #     print(time, end='\t')
    # print()
    # for time in time_dijk:
    #     print(time, end='\t')
    # print()
    # for time in time_astar:
    #     print(time, end='\t')
    # print()
