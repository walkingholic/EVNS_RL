import numpy as np
import matplotlib.pyplot as plt
import heapq
from Graph import Graph_simple
# from Graph import Graph_jeju
import pprint as pp
import copy
import datetime
import os
import csv
from haversine import haversine
import dqn_v04_my_work_per as dqn_sim
import queue
# from queue import PriorityQueue

UNITtimecost = dqn_sim.UNITtimecost
ECRate = dqn_sim.ECRate
Step_SOC = dqn_sim.Step_SOC
Base_SOC = dqn_sim.Base_SOC
N_SOC = dqn_sim.N_SOC

def createFolder(directory):
    try:
        if not os.path.isdir(directory):
            os.makedirs(directory)
    except OSError:
        print('error')

class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):

        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


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
                priority = new_cost + heuristic_astar(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                # print('frontier.put()', next, priority)
                came_from[next] = current

    return came_from, cost_so_far


def a_star_search_optimal(graph, start, goal):

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
                priority = new_cost + heuristic_astar(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                # print('frontier.put()', next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def update_envir_timeweight(CS_list, graph, sim_time):

    time_idx = int(sim_time/5)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        avg_price += cs.price[time_idx]
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH']/velo

    return avg_price

def update_envir_costweight(CS_list, pev, graph, sim_time):
    time_idx = int(sim_time/5)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        # gen_evcs_random(cs)
        avg_price += cs.price[time_idx]
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH'] * pev.ECRate * avg_price

    return avg_price

def update_envir_costtimeweight(CS_list, pev, graph, sim_time):
    time_idx = int(sim_time/5)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        avg_price += cs.price[time_idx]
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        roadtime = graph.link_data[l_id]['LENGTH'] / velo
        roadcost = graph.link_data[l_id]['LENGTH'] * pev.ECRate * avg_price
        graph.link_data[l_id]['WEIGHT'] = roadtime*UNITtimecost + roadcost

    return avg_price

def update_envir_distweight(CS_list, graph, sim_time):

    time_idx = int(sim_time/5)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        avg_price += cs.price[time_idx]
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH']

    return avg_price

def update_envir_ref_weight(CS_list, graph, sim_time):
    # print(int(sim_time / 5))

    time_idx = int(sim_time / 5)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        avg_price += cs.price[time_idx]
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        croad = ECRate*graph.link_data[l_id]['LENGTH']*cs.price[time_idx]
        troad = graph.link_data[l_id]['LENGTH']/velo*UNITtimecost
        graph.link_data[l_id]['WEIGHT'] = croad+troad

    return avg_price

def update_envir_weight(cs, graph, sim_time):
    # print(int(sim_time / 5))

    time_idx = int(sim_time / 5)
    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        croad = ECRate*graph.link_data[l_id]['LENGTH']*cs.TOU_price[int(time_idx/ 12)]
        troad = graph.link_data[l_id]['LENGTH']/velo*UNITtimecost
        graph.link_data[l_id]['WEIGHT'] = croad+troad


def update_envir_weight_shortestpath(cs, graph, sim_time):
    # print(int(sim_time / 5))

    time_idx = int(sim_time / 5)
    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        # croad = ECRate*graph.link_data[l_id]['LENGTH']*cs.TOU_price[int(time_idx/ 12)]
        # troad = graph.link_data[l_id]['LENGTH']/velo*UNITtimecost
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH']



def update_envir_ref_dist_weight(CS_list, graph, sim_time):

    time_idx = int(sim_time/5)
    count = 0
    avg_price = 0.0


    for cs in CS_list:
        avg_price += cs.price[time_idx]
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
        graph.link_data[l_id]['WEIGHT'] = graph.link_data[l_id]['LENGTH']

    return avg_price



def heuristic_astar(a, b):
    (x1, y1) = a
    (x2, y2) = b

    x1, y1 = a
    x2, y2 = b

    dist = haversine((y1, x1), (y2, x2), unit='km')
    return dist

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

def get_feature_state_refer(sim_time, pev, CS_list, graph, ncandi):

    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_ref_weight(CS_list, graph, sim_time)
        evcs_id = cs.id
        # came_from, cost_so_far = a_star_search(graph, start, evcs_id)
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)

        final_path = front_path
        dist = graph.get_path_distance(final_path)

        front_d_time = graph.get_path_drivingtime(front_path, int(sim_time / 5))
        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
        charging_energy = pev.maxBCAPA*pev.req_SOC - remainenergy
        chargingprice = cs.price[int(sim_time / 5)]

        charging_cost = charging_energy*chargingprice
        charging_time = (charging_energy/(cs.chargingpower*pev.charging_effi))
        waiting_time = cs.waittime[int(sim_time / 5)]


        if len(front_path)>1:
            distance_onetwo = graph.get_path_distance(front_path[0:2])
            velocity_onetwo = graph.velocity(front_path[0], front_path[1], int(sim_time / 5))
            croad = pev.ECRate * distance_onetwo * chargingprice
            troad = UNITtimecost * distance_onetwo / velocity_onetwo

        else:
            distance_onetwo = 0
            velocity_onetwo = 0
            croad = 0
            troad = 0


        # print('c:',croad,'t:', troad)
        road_cost = croad+troad

        info.append((cs, pev.curr_SOC, final_path, road_cost, waiting_time, charging_time, charging_cost))

    return info


def get_feature_state_optimal_refer(sim_time, pev, CS_list, graph, ncandi):

    bestsolution = []
    path = [pev.curr_location]
    for cs in CS_list:
        # print('cs', cs.id)
        ev = copy.deepcopy(pev)
        ev.path = path
        bucket = queue.Queue()
        bucket.put(ev)
        bestPev = ev
        bestPev.totalcost = 10000

        while bucket.qsize():
            ev = bucket.get()
            ctime = ev.curr_time
            if ev.curr_location == cs.id:

                ev.charged = 1

                ev.cscharingenergy = ev.maxBCAPA * ev.req_SOC - ev.curr_SOC * ev.maxBCAPA
                ev.cschargingtime = (ev.cscharingenergy / (cs.chargingpower * ev.charging_effi))
                ev.cschargingwaitingtime = cs.waittime[int(ctime / 5)]
                ev.cs = cs
                ev.fdist = ev.totaldrivingdistance
                ev.cschargingstarttime = ctime + ev.cschargingwaitingtime * 60
                ev.cschargingprice = cs.price[int(ctime / 5)]
                ev.cschargingcost = ev.cscharingenergy * ev.cschargingprice
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
                        _, _ = update_ev(newev, graph, cnode, next, ctime)
                        charingenergy = newev.maxBCAPA * (newev.req_SOC - newev.curr_SOC)
                        cost = newev.totaldrivingtime * UNITtimecost + newev.totalenergyconsumption * cs.price[int(ctime / 5)]
                        cost += cs.price[int(ctime / 5)] * charingenergy
                        if newev.curr_SOC > 0.0 and cost < bestPev.totalcost:
                            bucket.put(newev)

        front_path = bestPev.path
        # print(front_path, bestPev.SOC, ev.charged )
        if len(front_path) > 1:
            distance_onetwo = graph.get_path_distance(front_path[0:2])
            velocity_onetwo = graph.velocity(front_path[0], front_path[1], int(sim_time / 5))
            croad = pev.ECRate * distance_onetwo * bestPev.chargingprice
            troad = UNITtimecost * distance_onetwo / velocity_onetwo
        else:
            distance_onetwo = 0
            velocity_onetwo = 0
            croad = 0
            troad = 0
        road_cost = croad + troad

        info = (cs, bestPev.SOC, bestPev.path, road_cost, bestPev.waitingtime, bestPev.chargingtime, bestPev.chargingcost)
        bestsolution.append(info)

    return bestsolution


def every_time_check_refer(EV_list, CS_list, graph):
    # 시뮬레이션 시간은 유닛은 minute, 매 노드마다 경로를 다시 탐색한다.
    # Min total travel time
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================every_time_check_refer sim: {}==================================".format(sim_n))
        print("ID {}  S:{}  Time:{}".format(pev.id, pev.source, pev.t_start))

        start = pev.source
        pev.curr_location = start
        pev.path.append(start)

        while pev.charged != 1:
            here = pev.curr_location
            paths_info = PriorityQueue()
            avg_price = update_envir_ref_weight(CS_list, graph, sim_time)
            evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate

            if pev.curr_SOC <= 0:
                print('Real Error!!')
                input()
            for cs in CS_list:
                evcs_id = cs.id
                came_from, cost_so_far = a_star_search(graph, here, evcs_id)
                front_path = reconstruct_path(came_from, here, evcs_id)
                front_path_distance = graph.get_path_distance(front_path)
                if front_path_distance > evcango:
                    continue
                final_path = front_path
                path_weight = graph.get_path_weight(final_path)

                dist = graph.get_path_distance(final_path)
                d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5))
                remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
                c_energy = pev.maxBCAPA * pev.req_SOC - remainenergy
                c_time = (c_energy / (cs.chargingpower * pev.charging_effi))

                w = path_weight+c_energy*cs.price[int(sim_time/5)]+cs.waittime[int(sim_time / 5)]*UNITtimecost

                totaltraveltime = d_time + cs.waittime[int(sim_time / 5)] + c_time
                paths_info.put((remainenergy, final_path, front_path, front_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)
                # print(cs.id, w)

            remainenergy, path, fpath, fpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
            # print(evcs.id)
            pev.cs = evcs
            if pev.curr_location == evcs.id:
                pev.charged = 1
                print('CS', pev.charged, evcs.id)
                pev.before_charging_SOC = pev.curr_SOC
                pev.cscharingenergy = pev.maxBCAPA * pev.req_SOC - pev.curr_SOC * pev.maxBCAPA

                pev.cschargingwaitingtime = evcs.waittime[int(sim_time / 5)]
                pev.cschargingstarttime = sim_time + pev.cschargingwaitingtime * 60
                pev.cschargingprice = evcs.price[int(pev.cschargingstarttime / 5)]
                pev.cschargingcost = pev.cscharingenergy * pev.cschargingprice

                pev.curr_SOC = pev.req_SOC
                pev.cschargingtime = (pev.cscharingenergy / (evcs.chargingpower * pev.charging_effi))

                pev.fdist = pev.totaldrivingdistance
                pev.csid = evcs.id

                pev.csdrivingtime = pev.totaldrivingtime
                pev.csdistance = pev.totaldrivingdistance
                pev.cschargingwaitingtime = pev.cschargingwaitingtime
                pev.cschargingtime = pev.cschargingtime
                pev.cssoc = pev.curr_SOC
                pev.expense_time_part = (pev.totaldrivingtime + pev.cschargingwaitingtime) * UNITtimecost
                pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost
                pev.totalcost = pev.expense_time_part + pev.expense_cost_part

                sim_time += pev.cschargingtime * 60
                sim_time += evcs.waittime[int(sim_time / 5)] * 60

            else:
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time, _ = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
        sim_n += 1


def every_time_check_refer_shortest(EV_list, CS_list, graph):
    # 시뮬레이션 시간은 유닛은 minute, 매 노드마다 경로를 다시 탐색한다.
    # Min total travel time
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================every_time_check_refer_shortest sim: {}==================================".format(sim_n))
        print("ID {}  S:{}  Time:{}".format(pev.id, pev.source, pev.t_start))

        start = pev.source
        pev.curr_location = start
        pev.path.append(start)

        while pev.charged != 1:
            here = pev.curr_location
            paths_info = PriorityQueue()
            avg_price = update_envir_distweight(CS_list, graph, sim_time)
            evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate

            if pev.curr_SOC <= 0:
                print('Real Error!!')
                input()
            for cs in CS_list:
                evcs_id = cs.id
                came_from, cost_so_far = a_star_search(graph, here, evcs_id)
                front_path = reconstruct_path(came_from, here, evcs_id)
                front_path_distance = graph.get_path_distance(front_path)
                if front_path_distance > evcango:
                    continue
                final_path = front_path
                path_weight = graph.get_path_weight(final_path)

                dist = graph.get_path_distance(final_path)
                d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5))
                remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
                c_energy = pev.maxBCAPA * pev.req_SOC - remainenergy
                c_time = (c_energy / (cs.chargingpower * pev.charging_effi))
                w = path_weight
                totaltraveltime = d_time + cs.waittime[int(sim_time / 5)] + c_time
                paths_info.put((remainenergy, final_path, front_path, front_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)
                # print(cs.id, w)

            remainenergy, path, fpath, fpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
            # print(evcs)
            pev.cs = evcs
            if pev.curr_location == evcs.id:
                pev.charged = 1
                print('CS', pev.charged, evcs.id)
                pev.before_charging_SOC = pev.curr_SOC
                pev.cscharingenergy = pev.maxBCAPA * pev.req_SOC - pev.curr_SOC * pev.maxBCAPA

                pev.cschargingwaitingtime = evcs.waittime[int(sim_time / 5)]
                pev.cschargingstarttime = sim_time + pev.cschargingwaitingtime * 60
                pev.cschargingprice = evcs.price[int(pev.cschargingstarttime / 5)]
                pev.cschargingcost = pev.cscharingenergy * pev.cschargingprice

                pev.curr_SOC = pev.req_SOC
                pev.cschargingtime = (pev.cscharingenergy / (evcs.chargingpower * pev.charging_effi))

                pev.fdist = pev.totaldrivingdistance
                pev.csid = evcs.id

                pev.csdrivingtime = pev.totaldrivingtime
                pev.csdistance = pev.totaldrivingdistance
                pev.cschargingwaitingtime = pev.cschargingwaitingtime
                pev.cschargingtime = pev.cschargingtime
                pev.cssoc = pev.curr_SOC
                pev.expense_time_part = (pev.totaldrivingtime + pev.cschargingwaitingtime) * UNITtimecost
                pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost
                pev.totalcost = pev.expense_time_part + pev.expense_cost_part

                sim_time += pev.cschargingtime * 60
                sim_time += evcs.waittime[int(sim_time / 5)] * 60

            else:
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time, _ = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
        sim_n += 1

def every_time_check_refer_time(EV_list, CS_list, graph):
    # 시뮬레이션 시간은 유닛은 minute, 매 노드마다 경로를 다시 탐색한다.
    # Min total travel time
    sim_n = 0

    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================every_time_check_refer_time sim: {}==================================".format(sim_n))
        print("ID {}  S:{}  Time:{}".format(pev.id, pev.source, pev.t_start))

        start = pev.source
        pev.curr_location = start
        pev.path.append(start)

        while pev.charged != 1:
            here = pev.curr_location
            paths_info = PriorityQueue()
            avg_price = update_envir_timeweight(CS_list, graph, sim_time)
            evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate

            if pev.curr_SOC <= 0:
                print('Real Error!!')
                input()
            for cs in CS_list:
                evcs_id = cs.id
                came_from, cost_so_far = a_star_search(graph, here, evcs_id)
                front_path = reconstruct_path(came_from, here, evcs_id)
                front_path_distance = graph.get_path_distance(front_path)
                if front_path_distance > evcango:
                    continue
                final_path = front_path
                path_weight = graph.get_path_weight(final_path)

                dist = graph.get_path_distance(final_path)
                d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5))
                remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate
                c_energy = pev.maxBCAPA * pev.req_SOC - remainenergy
                c_time = (c_energy / (cs.chargingpower * pev.charging_effi))

                w = d_time + cs.waittime[int(sim_time / 5)] + c_time

                totaltraveltime = d_time + cs.waittime[int(sim_time / 5)] + c_time
                paths_info.put((remainenergy, final_path, front_path, front_path_distance, cs, d_time, c_time, c_energy, totaltraveltime), w)
                # print(cs.id, w)

            remainenergy, path, fpath, fpath_dist, evcs, dtime, c, c_energy, totaltraveltime = paths_info.get()
            # print(evcs.id)
            pev.cs = evcs
            if pev.curr_location == evcs.id:
                pev.charged = 1
                print('CS', pev.charged, evcs.id)
                pev.before_charging_SOC = pev.curr_SOC
                pev.cscharingenergy = pev.maxBCAPA * pev.req_SOC - pev.curr_SOC * pev.maxBCAPA

                pev.cschargingwaitingtime = evcs.waittime[int(sim_time / 5)]
                pev.cschargingstarttime = sim_time + pev.cschargingwaitingtime * 60
                pev.cschargingprice = evcs.price[int(pev.cschargingstarttime / 5)]
                pev.cschargingcost = pev.cscharingenergy * pev.cschargingprice

                pev.curr_SOC = pev.req_SOC
                pev.cschargingtime = (pev.cscharingenergy / (evcs.chargingpower * pev.charging_effi))

                pev.fdist = pev.totaldrivingdistance
                pev.csid = evcs.id

                pev.csdrivingtime = pev.totaldrivingtime
                pev.csdistance = pev.totaldrivingdistance
                pev.cschargingwaitingtime = pev.cschargingwaitingtime
                pev.cschargingtime = pev.cschargingtime
                pev.cssoc = pev.curr_SOC
                pev.expense_time_part = (pev.totaldrivingtime + pev.cschargingwaitingtime) * UNITtimecost
                pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost
                pev.totalcost = pev.expense_time_part + pev.expense_cost_part

                sim_time += pev.cschargingtime * 60
                sim_time += evcs.waittime[int(sim_time / 5)] * 60
            else:
                pev.next_location = path[1]
                pev.path.append(pev.next_location)
                sim_time , _ = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
        sim_n += 1

def optimal_solution_source_to_cs(EV_list, CS_list, graph):
    pev_list = []
    for i, pev in enumerate(EV_list):
        print(
            "\n===========================optimal_solution sim: {}==================================".format(i))
        print("ID {}  S:{}  Time:{}".format(pev.id, pev.source, pev.t_start))
        path = [pev.source]
        pev.path = path
        bestsolution = PriorityQueue()

        for cs in CS_list:
            ev = copy.deepcopy(pev)
            # print()
            # print(cs.id, ev.curr_time, ev.SOC, ev.drivingdistance, ev.path)
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
                    # print(bestPev.totalcost, ev.totalcost)
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
                            newev.curr_time, _ = update_ev(newev, graph, cnode, next, ctime)

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



def update_ev(pev, simple_graph, fnode, tnode, sim_time):

    # if sim_time >= 1440:
    #     print('if sim time', sim_time)
    #     sim_time = sim_time%1440
    #     print('if sim time', sim_time)
    #     pev.curr_day += 1


    time_idx = int(sim_time%1440 / 5)
    # print(sim_time, time_idx)


    dist = simple_graph.distance(fnode, tnode)
    velo = simple_graph.velocity(fnode, tnode, time_idx)

    # print(' curTime: {}   curLoca: {}    sim: {}  velo: {}'.format(pev.curr_time, pev.curr_location, sim_time, velo))

    time_diff = (dist/velo)*60
    # pev.traveltime = pev.traveltime + time_diff
    pev.totaldrivingtime += time_diff
    pev.totaldrivingdistance += dist
    soc_before = pev.curr_SOC
    pev.curr_SOC = pev.curr_SOC - (dist * pev.ECRate) / pev.maxBCAPA
    pev.totalenergyconsumption = pev.totalenergyconsumption + dist * pev.ECRate

    new_sim_time = sim_time + time_diff

    # if new_sim_time >= 1440:
    #     new_sim_time = new_sim_time - 1440
    #     pev.curr_day += 1

    time_idx = int(new_sim_time%1440 / 5)
    # print("fnode {} tnode {} dist {} velo {} soc_b {} soc_a {}".format(fnode, tnode, dist, velo, soc_before, pev.SOC))

    if pev.charged != 1:
        pev.fdist += dist
    pev.curr_time = new_sim_time
    pev.curr_location = tnode
    # pev.path.append(pev.curr_location)
    return new_sim_time, time_diff


def pruning_evcs(pev, CS_list,graph, ncandi):
    cslist = []
    for cs in CS_list:
        x1, y1 = graph.nodes_xy(pev.curr_location)
        x2, y2 = graph.nodes_xy(cs.id)
        x3, y3 = graph.nodes_xy(pev.destination)

        f = haversine((y1, x1),(y2, x2), unit='km')
        r = haversine((y2, x2),(y3, x3), unit='km')
        airdist = f+r
        cslist.append((cs, airdist))
    cslist.sort(key=lambda element:element[1])
    return cslist[:ncandi]


def get_feature_state(sim_time, pev, CS_list, graph, ncandi):

    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight(cs, graph, sim_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)
        fpath_weight = graph.get_path_weight(front_path)
        rpath_weight = graph.get_path_weight(rear_path)
        rear_consump_energy = rear_path_distance * pev.ECRate
        front_d_time = graph.get_path_drivingtime(front_path, int(sim_time / 5))
        rear_d_time = graph.get_path_drivingtime(rear_path, int(sim_time / 5))
        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
        waiting_time = cs.waittime[int(sim_time / 5)]

        driving_cost = graph.get_path_weight(final_path)

        for i in range(0, N_SOC):
            req_soc = i*Step_SOC+Base_SOC
            charging_energy = pev.maxBCAPA*req_soc - remainenergy
            if charging_energy<=0:
                print('charging_energy error')
                input()
            chargingprice = cs.price[int(sim_time / 5)]
            charging_time = (charging_energy/(cs.chargingpower*pev.charging_effi))
            cs_charging_cost = charging_energy * chargingprice+charging_time*UNITtimecost
            athome_remainE = (pev.maxBCAPA*req_soc-rear_consump_energy)
            athome_soc = athome_remainE/pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA*pev.final_soc-athome_remainE)*cs.TOU_price[int(sim_time/60)]

            info.append((cs, req_soc, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, waiting_time,
                         charging_time, cs_charging_cost, home_charging_cost))

            # print('actions: ',cs.id, req_soc)

    return info

def get_true_arrtime(pev, cs, graph):
    # print('get_true_arrtime', pev.id)
    for i in range(len(pev.front_path) - 1):
        fnode = pev.front_path[i]
        tnode = pev.front_path[i + 1]

        _, time = update_ev(pev, graph, pev.curr_location, tnode, pev.curr_time)

        if pev.curr_SOC <= 0.0:
            print('No soc')
            input()
    # print(' curTime: {}   curLoca: {} '.format(pev.curr_time, pev.curr_location))

    return pev.curr_time

def finishi_trip(pev, cp, graph):
    # print('fin ev', pev.id, pev.cs.id, cp.id, pev.curr_time, pev.true_waitingtime)
    # print(pev.curr_location, pev.rear_path)
    for i in range(len(pev.rear_path) - 1):
        fnode = pev.rear_path[i]
        tnode = pev.rear_path[i + 1]

        _, time = update_ev(pev, graph, pev.curr_location, tnode, pev.curr_time)

        if pev.curr_SOC <= 0.0:
            print('No soc')
            input()
    # print(' curTime: {}   curLoca: {} '.format(pev.curr_time, pev.curr_location))

    pev.cschargingcost = pev.cscharingenergy * pev.cschargingprice
    pev.expense_time_part = (pev.totaldrivingtime + pev.true_waitingtime + pev.true_charging_duration)/60 * UNITtimecost
    pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost

    pev.totalcost = pev.expense_time_part + pev.expense_cost_part

    return pev.curr_time



def get_feature_state_fleet(cur_time, pev, CS_list, graph, ncandi):

    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    for cs in CS_list:
        update_envir_weight(cs, graph, cur_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)

        fpath_weight = graph.get_path_weight(front_path)
        rpath_weight = graph.get_path_weight(rear_path)

        rear_consump_energy = rear_path_distance * pev.ECRate
        front_d_time = graph.get_path_drivingtime(front_path, int(cur_time / 5))*60
        rear_d_time = graph.get_path_drivingtime(rear_path, int(cur_time / 5))*60
        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate

        ept_arrtime = cur_time+front_d_time
        # print(cur_time, ept_arrtime)

        driving_cost = graph.get_path_weight(final_path)
        charging_energy = pev.maxBCAPA*pev.req_SOC - remainenergy

        if charging_energy<=0:
            print('charging_energy error')
            input()

        chargingprice = cs.price[int(cur_time / 5)]
        ept_charging_duration = (charging_energy/(cs.chargingpower*pev.charging_effi))*60

        cs_charging_cost = charging_energy * chargingprice + ept_charging_duration*UNITtimecost

        athome_remainE = (pev.maxBCAPA*pev.req_SOC - rear_consump_energy)
        athome_soc = athome_remainE/pev.maxBCAPA
        home_charging_cost = (pev.maxBCAPA*pev.final_soc-athome_remainE)*cs.TOU_price[int(cur_time/60)]

        ept_WT = cs.get_ept_WT(ept_arrtime, cur_time, graph)

        weight = cs_charging_cost+driving_cost+home_charging_cost
        info.append((cs, weight, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                     front_d_time, rear_d_time, fpath_weight, rpath_weight, ept_WT,
                     ept_charging_duration, cs_charging_cost, home_charging_cost, ept_arrtime))

            # print('actions: ',cs.id, req_soc)

    return info


def get_feature_state_jeju(sim_time, pev, CS_list, graph, ncandi):

    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []

    clist = pruning_evcs(pev, CS_list, graph, ncandi)
    for cs,_ in clist:
        update_envir_weight(cs, graph, sim_time)
        evcs_id = cs.id
        came_from, cost_so_far = dijkstra_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = dijkstra_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        total_distance = graph.get_path_distance(final_path)
        fpath_weight = graph.get_path_weight(front_path)
        rpath_weight = graph.get_path_weight(rear_path)
        rear_consump_energy = rear_path_distance * pev.ECRate
        front_d_time = graph.get_path_drivingtime(front_path, int(sim_time / 5))
        rear_d_time = graph.get_path_drivingtime(rear_path, int(sim_time / 5))
        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
        waiting_time = cs.waittime[int(sim_time / 5)]

        driving_cost = graph.get_path_weight(final_path)

        for i in range(0, N_SOC):
            req_soc = i*Step_SOC+Base_SOC
            charging_energy = pev.maxBCAPA*req_soc - remainenergy
            if charging_energy<=0:
                print('charging_energy error')
                input()
            chargingprice = cs.price[int(sim_time / 5)]
            charging_time = (charging_energy/(cs.chargingpower*pev.charging_effi))
            cs_charging_cost = charging_energy * chargingprice+charging_time*UNITtimecost
            athome_remainE = (pev.maxBCAPA*req_soc-rear_consump_energy)
            athome_soc = athome_remainE/pev.maxBCAPA
            home_charging_cost = (pev.maxBCAPA*pev.final_soc-athome_remainE)*cs.TOU_price[int(sim_time/60)]

            info.append((cs, req_soc, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance,
                         front_d_time, rear_d_time, fpath_weight, rpath_weight, waiting_time,
                         charging_time, cs_charging_cost, home_charging_cost))

            # print('actions: ',cs.id, req_soc)

    return info



def greedy_total_cost_search(EV_list, CS_list, graph):

    sim_n = 0
    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================greedy_total_cost_search sim: {}==================================".format(sim_n))
        print("ID {}  S:{} D: {} Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))


        pev.path.append(pev.source)

        while pev.charged != 1:
            paths_info = PriorityQueue()
            evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate

            if pev.curr_SOC <= 0:
                print('Real Error!!')
                input()
            for cs in CS_list:
                update_envir_weight(cs, graph, sim_time)
                evcs_id = cs.id
                came_from, cost_so_far = dijkstra_search(graph, pev.curr_location, evcs_id)
                front_path = reconstruct_path(came_from, pev.curr_location, evcs_id)
                front_path_distance = graph.get_path_distance(front_path)
                if front_path_distance > evcango:
                    continue
                came_from, cost_so_far = dijkstra_search(graph, evcs_id, pev.destination)
                rear_path = reconstruct_path(came_from, evcs_id, pev.destination)
                rear_path_distance = graph.get_path_distance(rear_path)

                final_path = front_path + rear_path[1:]
                fpath_weight = graph.get_path_weight(front_path)
                rpath_weight = graph.get_path_weight(rear_path)
                rear_consump_energy = rear_path_distance * pev.ECRate
                front_d_time = graph.get_path_drivingtime(front_path, int(sim_time / 5))
                rear_d_time = graph.get_path_drivingtime(rear_path, int(sim_time / 5))
                remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

                cs_chargingwaitingtime = cs.waittime[int(sim_time / 5)]
                totaldrivingdistance = graph.get_path_distance(final_path)

                totaldriving_cost = graph.get_path_weight(final_path)
                totaldrivingtime = graph.get_path_drivingtime(final_path, int(sim_time / 5))

                for i in range(0, N_SOC):
                    req_soc = i * Step_SOC +Base_SOC
                    cs_charging_energy = pev.maxBCAPA * req_soc - remainenergy
                    if cs_charging_energy <= 0:
                        print('charging_energy error')
                        input()
                    cs_chargingprice = cs.price[int(sim_time / 5)]
                    cs_charging_time = (cs_charging_energy / (cs.chargingpower * pev.charging_effi))
                    cs_charging_cost = cs_charging_energy * cs_chargingprice + cs_charging_time * UNITtimecost
                    athome_remainE = (pev.maxBCAPA * req_soc - rear_consump_energy)
                    athome_soc = athome_remainE / pev.maxBCAPA
                    home_charging_cost = (pev.maxBCAPA*pev.final_soc - athome_remainE) * cs.TOU_price[int(sim_time / 60)]
                    expense_time_part = (totaldrivingtime + cs_chargingwaitingtime + cs_charging_time) * UNITtimecost
                    expense_cost_part = totaldrivingdistance * pev.ECRate * cs_chargingprice + cs_charging_cost + home_charging_cost
                    totalcost = expense_time_part + expense_cost_part
                    w = totalcost
                    paths_info.put((i, cs, req_soc, totaldriving_cost, front_path, rear_path, front_path_distance, rear_path_distance, front_d_time, rear_d_time, fpath_weight, rpath_weight, cs_chargingwaitingtime, cs_charging_time, cs_charging_cost, home_charging_cost), w)

            _, evcs, req_soc, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, front_d_time, rear_d_time, fpath_weight, rpath_weight, waiting_time, charging_time, cs_charging_cost, home_charging_cost = paths_info.get()


            pev.cs = evcs
            pev.req_SOC = req_soc
            if pev.curr_location == evcs.id:
                pev.charged = 1
                print('CS', pev.charged, evcs.id)

                pev.cssoc = pev.curr_SOC
                pev.before_charging_SOC = pev.curr_SOC
                pev.cscharingenergy = pev.maxBCAPA * pev.req_SOC - pev.curr_SOC * pev.maxBCAPA

                pev.cschargingwaitingtime = evcs.waittime[int(sim_time / 5)]
                pev.cschargingstarttime = sim_time + pev.cschargingwaitingtime * 60
                pev.cschargingprice = evcs.price[int(pev.cschargingstarttime / 5)]
                pev.cschargingcost = pev.cscharingenergy * pev.cschargingprice
                pev.csdrivingtime = pev.totaldrivingtime
                pev.csdistance = pev.totaldrivingdistance
                pev.cschargingwaitingtime = pev.cschargingwaitingtime
                pev.cschargingtime = (pev.cscharingenergy / (evcs.chargingpower * pev.charging_effi))

                pev.fdist = pev.totaldrivingdistance
                pev.csid = pev.cs.id

                pev.curr_SOC = pev.req_SOC
                sim_time += pev.cschargingwaitingtime * 60
                sim_time += pev.cschargingtime * 60
                pev.curr_time = sim_time

            else:
                pev.next_location = front_path[1]
                pev.path.append(pev.next_location)
                sim_time, _ = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)


        while pev.curr_location != pev.destination:
            came_from, cost_so_far = dijkstra_search(graph, pev.curr_location, pev.destination)
            path = reconstruct_path(came_from, pev.curr_location, pev.destination)
            path_distance = graph.get_path_distance(path)
            print('path', path)
            pev.next_location = path[1]
            pev.path.append(pev.next_location)
            pev.homedrivingdistance += graph.distance(pev.curr_location, pev.next_location)
            sim_time, homedtime = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
            pev.homedrivingtime += homedtime


            if pev.curr_location == pev.destination:
                pev.homesoc = pev.curr_SOC
                pev.homechargingstarttime = sim_time
                pev.homechargingenergy = pev.maxBCAPA * pev.final_soc - pev.curr_SOC * pev.maxBCAPA
                pev.homechargingtime = pev.homechargingenergy / (cs.homechargingpower * pev.charging_effi)
                pev.homechargingprice = cs.TOU_price[int(sim_time / 60)]
                pev.homechargingcost = pev.homechargingenergy * pev.homechargingprice

                pev.curr_SOC = pev.final_soc
                sim_time += pev.homechargingtime
                pev.curr_time = sim_time

                pev.expense_time_part = (pev.totaldrivingtime + pev.cschargingwaitingtime + pev.cschargingtime) * UNITtimecost
                pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost + pev.homechargingcost

                pev.totalcost = pev.expense_time_part + pev.expense_cost_part

        sim_n += 1

def greedy_shortest_search(EV_list, CS_list, graph):

    sim_n = 0
    for pev in EV_list:
        sim_time = pev.t_start
        print("\n===========================greedy_shortest_search sim: {}==================================".format(sim_n))
        print("ID {}  S:{} D: {} Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))


        pev.path.append(pev.source)

        while pev.charged != 1:
            # paths_info = PriorityQueue()
            paths_info = []
            evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate

            if pev.curr_SOC <= 0:
                print('Real Error!!')
                input()
            for cs in CS_list:
                # update_envir_weight(cs, graph, sim_time)
                update_envir_weight_shortestpath(cs, graph, sim_time)
                evcs_id = cs.id
                came_from, cost_so_far = dijkstra_search(graph, pev.curr_location, evcs_id)
                front_path = reconstruct_path(came_from, pev.curr_location, evcs_id)
                front_path_distance = graph.get_path_distance(front_path)
                if front_path_distance > evcango:
                    continue
                came_from, cost_so_far = dijkstra_search(graph, evcs_id, pev.destination)
                rear_path = reconstruct_path(came_from, evcs_id, pev.destination)
                rear_path_distance = graph.get_path_distance(rear_path)

                final_path = front_path + rear_path[1:]
                fpath_weight = graph.get_path_weight(front_path)
                rpath_weight = graph.get_path_weight(rear_path)
                rear_consump_energy = rear_path_distance * pev.ECRate
                front_d_time = graph.get_path_drivingtime(front_path, int(sim_time / 5))
                rear_d_time = graph.get_path_drivingtime(rear_path, int(sim_time / 5))
                remainenergy = pev.maxBCAPA * pev.init_SOC - front_path_distance * pev.ECRate

                cs_chargingwaitingtime = cs.waittime[int(sim_time / 5)]
                totaldrivingdistance = graph.get_path_distance(final_path)

                totaldriving_cost = graph.get_path_weight(final_path)
                totaldrivingtime = graph.get_path_drivingtime(final_path, int(sim_time / 5))

                for i in range(0, N_SOC):
                    req_soc = i * Step_SOC +Base_SOC
                    cs_charging_energy = pev.maxBCAPA * req_soc - remainenergy
                    if cs_charging_energy <= 0:
                        print('charging_energy error')
                        input()
                    cs_chargingprice = cs.price[int(sim_time / 5)]
                    cs_charging_time = (cs_charging_energy / (cs.chargingpower * pev.charging_effi))
                    cs_charging_cost = cs_charging_energy * cs_chargingprice + cs_charging_time * UNITtimecost
                    athome_remainE = (pev.maxBCAPA * req_soc - rear_consump_energy)
                    athome_soc = athome_remainE / pev.maxBCAPA
                    home_charging_cost = (pev.maxBCAPA*pev.final_soc - athome_remainE) * cs.TOU_price[int(sim_time / 60)]
                    expense_time_part = (totaldrivingtime + cs_chargingwaitingtime + cs_charging_time) * UNITtimecost
                    expense_cost_part = totaldrivingdistance * pev.ECRate * cs_chargingprice + cs_charging_cost + home_charging_cost
                    totalcost = expense_time_part + expense_cost_part

                    w = graph.get_path_weight(final_path)

                    # print('len Q', len(paths_info.elements))
                    # # print('Q', paths_info.elements)
                    # for e in paths_info.elements:
                    #     print(e)
                    #
                    # print(w, cs, req_soc, totaldriving_cost, front_path, rear_path, front_path_distance,
                    #                 rear_path_distance, front_d_time, rear_d_time, fpath_weight, rpath_weight,
                    #                 cs_chargingwaitingtime, cs_charging_time, cs_charging_cost, home_charging_cost)

                    # paths_info.put((i, req_soc, cs,  totaldriving_cost, front_path, rear_path, front_path_distance, rear_path_distance, front_d_time, rear_d_time, fpath_weight, rpath_weight, cs_chargingwaitingtime, cs_charging_time, cs_charging_cost, home_charging_cost), w)
                    paths_info.append(((i, req_soc, cs,  totaldriving_cost, front_path, rear_path, front_path_distance, rear_path_distance, front_d_time, rear_d_time, fpath_weight, rpath_weight, cs_chargingwaitingtime, cs_charging_time, cs_charging_cost, home_charging_cost), w))
                    paths_info.sort(key=lambda element: element[1])

            data, weight = paths_info[0]
            _, req_soc, evcs, driving_cost, front_path, rear_path, front_path_distance, rear_path_distance, front_d_time, rear_d_time, fpath_weight, rpath_weight, waiting_time, charging_time, cs_charging_cost, home_charging_cost = data

            pev.cs = evcs
            pev.req_SOC = req_soc
            if pev.curr_location == evcs.id:
                pev.charged = 1
                print('CS', pev.charged, evcs.id)

                pev.cssoc = pev.curr_SOC
                pev.before_charging_SOC = pev.curr_SOC
                pev.cscharingenergy = pev.maxBCAPA * pev.req_SOC - pev.curr_SOC * pev.maxBCAPA

                pev.cschargingwaitingtime = evcs.waittime[int(sim_time / 5)]
                pev.cschargingstarttime = sim_time + pev.cschargingwaitingtime * 60
                pev.cschargingprice = evcs.price[int(pev.cschargingstarttime / 5)]
                pev.cschargingcost = pev.cscharingenergy * pev.cschargingprice
                pev.csdrivingtime = pev.totaldrivingtime
                pev.csdistance = pev.totaldrivingdistance
                pev.cschargingwaitingtime = pev.cschargingwaitingtime
                pev.cschargingtime = (pev.cscharingenergy / (evcs.chargingpower * pev.charging_effi))

                pev.fdist = pev.totaldrivingdistance
                pev.csid = pev.cs.id

                pev.curr_SOC = pev.req_SOC
                sim_time += pev.cschargingwaitingtime * 60
                sim_time += pev.cschargingtime * 60
                pev.curr_time = sim_time

            else:
                pev.next_location = front_path[1]
                pev.path.append(pev.next_location)
                sim_time, _ = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)


        while pev.curr_location != pev.destination:
            came_from, cost_so_far = dijkstra_search(graph, pev.curr_location, pev.destination)
            path = reconstruct_path(came_from, pev.curr_location, pev.destination)
            path_distance = graph.get_path_distance(path)
            # print('path', path)
            pev.next_location = path[1]
            pev.path.append(pev.next_location)
            pev.homedrivingdistance += graph.distance(pev.curr_location, pev.next_location)
            sim_time, homedtime = update_ev(pev, graph, pev.curr_location, pev.next_location, sim_time)
            pev.homedrivingtime += homedtime


            if pev.curr_location == pev.destination:
                pev.homesoc = pev.curr_SOC
                pev.homechargingstarttime = sim_time
                pev.homechargingenergy = pev.maxBCAPA * pev.final_soc - pev.curr_SOC * pev.maxBCAPA
                pev.homechargingtime = pev.homechargingenergy / (cs.homechargingpower * pev.charging_effi)
                pev.homechargingprice = cs.TOU_price[int(sim_time / 60)]
                pev.homechargingcost = pev.homechargingenergy * pev.homechargingprice

                pev.curr_SOC = pev.final_soc
                sim_time += pev.homechargingtime
                pev.curr_time = sim_time

                pev.expense_time_part = (pev.totaldrivingtime + pev.cschargingwaitingtime + pev.cschargingtime) * UNITtimecost
                pev.expense_cost_part = pev.totaldrivingdistance * pev.ECRate * pev.cschargingprice + pev.cschargingcost + pev.homechargingcost

                pev.totalcost = pev.expense_time_part + pev.expense_cost_part

        sim_n += 1


def optimal_solution(EV_list, CS_list, graph):
    pev_list = []
    for i, pev in enumerate(EV_list):
        print(
            "\n===========================optimal_solution sim: {}==================================".format(i))
        print("ID {}  S:{} D:{} Time:{}".format(pev.id, pev.source, pev.destination, pev.t_start))
        path = [pev.source]
        pev.path = path
        # bestsolution = PriorityQueue()
        bestsolution = []

        for cs in CS_list:
            for i in range(0, N_SOC):
                req_soc = i * Step_SOC + Base_SOC
                ev = copy.deepcopy(pev)
                ev.req_SOC = req_soc
                bucket = queue.Queue()
                bucket.put(ev)
                bestPev = ev
                bestPev.totalcost = 10000

                while bucket.qsize():
                    ev = bucket.get()

                    if ev.curr_location == cs.id:
                        ev.rear_path.append(ev.curr_location)
                        ev.charged = 1
                        ev.before_charging_SOC = ev.curr_SOC

                        ev.cscharingenergy = ev.maxBCAPA * ev.req_SOC - ev.curr_SOC * ev.maxBCAPA
                        ev.cschargingtime = (ev.cscharingenergy / (cs.chargingpower * ev.charging_effi))
                        ev.cschargingwaitingtime = cs.waittime[int(ev.curr_time / 5)]
                        ev.cs = cs
                        ev.fdist = ev.totaldrivingdistance
                        ev.cschargingstarttime = ev.curr_time + ev.cschargingwaitingtime * 60
                        ev.cschargingprice = cs.price[int(ev.cschargingstarttime / 5)]
                        ev.cschargingcost = ev.cscharingenergy * ev.cschargingprice

                        ev.csdrivingtime = ev.totaldrivingtime
                        ev.csdistance = ev.totaldrivingdistance
                        ev.cssoc = ev.curr_SOC

                        ev.curr_SOC = ev.req_SOC
                        ev.curr_time += cs.waittime[int(ev.curr_time / 5)] * 60
                        ev.curr_time += ev.cschargingtime * 60

                    elif ev.charged == 0 :
                        cnode = ev.curr_location
                        for next in graph.neighbors(cnode):
                            newpath = copy.copy(ev.path)
                            if next not in newpath:
                                ctime = ev.curr_time
                                newpath.append(next)
                                newev = copy.deepcopy(ev)
                                newev.path = newpath
                                newev.curr_time, _ = update_ev(newev, graph, cnode, next, ctime)
                                charingenergy = newev.maxBCAPA * (newev.req_SOC - newev.curr_SOC)
                                cost = newev.totaldrivingtime * UNITtimecost + newev.totalenergyconsumption * min(cs.price)
                                cost += min(cs.price) * charingenergy
                                if newev.curr_SOC > 0.0 and cost <= bestPev.totalcost:
                                    bucket.put(newev)

                    elif ev.charged == 1 :

                        if ev.curr_location == ev.destination:
                            ev.homesoc = ev.curr_SOC
                            ev.homechargingstarttime = ev.curr_time
                            ev.homechargingenergy = ev.maxBCAPA * ev.final_soc - ev.curr_SOC * ev.maxBCAPA
                            ev.homechargingtime = ev.homechargingenergy / (cs.homechargingpower * ev.charging_effi)
                            ev.homechargingprice = cs.TOU_price[int(ev.homechargingstarttime / 60)]
                            ev.homechargingcost = ev.homechargingenergy * ev.homechargingprice

                            ev.curr_SOC = ev.final_soc
                            ev.curr_time += ev.homechargingtime
                            ev.expense_time_part = (ev.totaldrivingtime + ev.cschargingwaitingtime + ev.cschargingtime) * UNITtimecost
                            ev.expense_cost_part = ev.totaldrivingdistance * ev.ECRate * ev.cschargingprice + ev.cschargingcost + ev.homechargingcost
                            ev.totalcost = ev.expense_time_part + ev.expense_cost_part
                            if ev.totalcost < bestPev.totalcost:
                                bestPev = ev

                        else:
                            cnode = ev.curr_location
                            for next in graph.neighbors(cnode):
                                rear_path = copy.copy(ev.rear_path)
                                path = copy.copy(ev.path)
                                if next not in rear_path:
                                    path.append(next)
                                    rear_path.append(next)
                                    newev = copy.deepcopy(ev)
                                    newev.path = path
                                    newev.rear_path = rear_path
                                    newev.curr_time, _ = update_ev(newev, graph, cnode, next, newev.curr_time)
                                    expense_time_part = (ev.totaldrivingtime + ev.cschargingwaitingtime + ev.cschargingtime) * UNITtimecost
                                    expense_cost_part = (ev.totaldrivingdistance * ev.ECRate * ev.cschargingprice) + ev.cschargingcost
                                    cost = expense_time_part + expense_cost_part
                                    if newev.curr_SOC > 0.0 and cost <= bestPev.totalcost:
                                        bucket.put(newev)

                # bestsolution.put(bestPev, bestPev.totalcost)
                bestsolution.append((bestPev, bestPev.totalcost))
                bestsolution.sort(key=lambda element:element[1])

        pev, _ = bestsolution[0]
        # pev = bestsolution.get()
        pev_list.append(pev)
    return pev_list





#
# def sim_result_general_presentation(graph, resultdir, numev, **results):
#     print('makeing figures...')
#     keyname = ''
#     for key in results.keys():
#         keyname += '_'+ key
#     basepath = os.getcwd()
#     resultdir = resultdir+'/result{}'.format(keyname)
#     print(os.path.join(basepath, resultdir))
#     dirpath = os.path.join(basepath, resultdir)
#     createFolder(dirpath)
#
#     keylist = list(results.keys())
#
#     # plt.figure(figsize=(12, 12), dpi=300)
#     # for i in range(numev):
#     #     kth_result=1
#     #     for key, EVlist in results.items():
#     #         pev = EVlist[i]
#     #         xx = []
#     #         yy = []
#     #         nth=0
#     #         for nid in pev.path:
#     #             x, y = graph.nodes_xy(nid)
#     #             plt.text(x+kth_result*0.1,y,str(nth))
#     #             xx.append(x)
#     #             yy.append(y)
#     #             nth+=1
#     #         plt.plot(xx, yy, label=key)
#     #
#     #         cs_x, cs_y = graph.nodes_xy(pev.cs.id)
#     #         plt.plot(cs_x, cs_y, 'D', label=key+' EVCS')
#     #         kth_result+=1
#     #
#     #     s_x, s_y = graph.nodes_xy(pev.source)
#     #     plt.plot(s_x, s_y, 'p', label='Source')
#     #     d_x, d_y = graph.nodes_xy(pev.destination)
#     #     plt.plot(d_x, d_y, 'h', label='Destination')
#     #
#     #     plt.xlim(graph.minx, graph.maxx)
#     #     plt.ylim(graph.miny, graph.maxy)
#     #     plt.legend()
#     #     fig = plt.gcf()
#     #     fig.savefig('{}/route_{}.png'.format(resultdir, i), facecolor='#eeeeee', dpi=300)
#     #     plt.clf()
#
#     plt.figure(figsize=(12, 12), dpi=300)
#     for key, EVlist in results.items():
#
#         for i, pev in enumerate(EVlist):
#             xx = []
#             yy = []
#             nth = 0
#             for nid in pev.path:
#                 x, y = graph.nodes_xy(nid)
#                 plt.text(x, y, str(nth))
#                 xx.append(x)
#                 yy.append(y)
#                 nth += 1
#             plt.plot(xx, yy, label=key)
#             cs_x, cs_y = graph.nodes_xy(pev.cs.id)
#             plt.plot(cs_x, cs_y, 'D', label=key + ' EVCS')
#
#             s_x, s_y = graph.nodes_xy(pev.source)
#             plt.plot(s_x, s_y, 'p', label='Source')
#             d_x, d_y = graph.nodes_xy(pev.destination)
#             plt.plot(d_x, d_y, 'h', label='Destination')
#
#             plt.xlim(graph.minx, graph.maxx)
#             plt.ylim(graph.miny, graph.maxy)
#             plt.legend()
#             fig = plt.gcf()
#             fig.savefig('{}/route_{}_{}.png'.format(resultdir, key, i), facecolor='#eeeeee', dpi=300)
#             plt.clf()
#
#
#
#     plt.title('Selected EVCS')
#     plt.xlabel('EV index')
#     plt.ylabel('EVCS ID')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cs.id)
#         plt.plot(range(len(r1_list)), r1_list,'x',  label=key)
#     plt.legend()
#     # plt.xlim(graph.minx, graph.maxx)
#     # plt.ylim(graph.miny, graph.maxy)
#     fig = plt.gcf()
#     fig.savefig('{}/EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     linetype = ['-', '--', ':', '-.']
#
#     plt.figure(figsize=(12, 6), dpi=300)
#     plt.title('Charging Cost')
#     plt.xlabel('EV ID')
#     plt.ylabel('Cost($)')
#     cnt=0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cschargingcost)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt+=1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Charging Cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Driving distance')
#     plt.xlabel('EV ID')
#     plt.ylabel('Distance(km)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.totaldrivingdistance)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Driving distance.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Distance from S to EVCS')
#     plt.xlabel('EV ID')
#     plt.ylabel('Distance(km)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.fdist)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Distance from S to EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Driving time')
#     plt.xlabel('EV ID')
#     plt.ylabel('Time(h)')
#     cnt = 0
#     for key, EVlist in results.items():
#         numev = len(EVlist)
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.totaldrivingtime)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Driving time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Charging energy')
#     plt.xlabel('EV ID')
#     plt.ylabel('Energy(kWh)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cscharingenergy)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Charging energy.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Charging time')
#     plt.xlabel('EV ID')
#     plt.ylabel('Time(h)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cschargingtime)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Charging time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Waiting time')
#     plt.xlabel('EV ID')
#     plt.ylabel('Time(h)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cschargingwaitingtime)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Waiting time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Total travel time')
#     plt.xlabel('EV ID')
#     plt.ylabel('Time(h)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Total travel time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('Total travel cost')
#     plt.xlabel('EV ID')
#     plt.ylabel('Cost($)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.totalcost)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/Total travel cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     plt.title('EV SOC')
#     plt.xlabel('EV ID')
#     plt.ylabel('SOC(%)')
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.init_SOC)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     cnt = 0
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.curr_SOC)
#         plt.plot(r1_list, linetype[cnt], label=key)
#         cnt += 1
#     plt.legend()
#     fig = plt.gcf()
#     fig.savefig('{}/ev SOC.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
#     plt.clf()
#
#     #=======================================================================================
#
# def sim_result_text(resultdir, **results):
#     print('makeing documents...')
#     keyname = ''
#     for key in results.keys():
#         keyname += key + '_'
#     print(keyname)
#
#     fw = open('{}/result_{}.txt'.format(resultdir, keyname), 'w', encoding='UTF8')
#
#     fw.write('Source\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.source)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('Destination\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.destination)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('init_SOC\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.init_SOC)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('cs soc\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cssoc)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('cs req soc\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.req_SOC)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('home soc\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.homesoc)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('t_start\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.t_start)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('cs\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.csid)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('Charging energy\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cscharingenergy)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('to_cs_dist\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.csdistance)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('fdist\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.fdist)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('to_cs_driving_time\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.csdrivingtime)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('to_cs_charging_time\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cschargingtime)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('to_cs_waiting_time\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cschargingwaitingtime)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('to home driving distance\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.homedrivingdistance)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('to home driving time\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.homedrivingtime)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('homechargingenergy\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.homechargingenergy)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('homechargingcost\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.homechargingcost)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#
#     fw.write('Total Driving distance\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.totaldrivingdistance)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('Total Driving time\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.totaldrivingtime)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#     fw.write('Total travel time\n')
#     for key, EVlist in results.items():
#         r1_list = []
#         for ev in EVlist:
#             r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
#         fw.write(key + '\t')
#         for value in r1_list:
#             fw.write(str(value) + '\t')
#         fw.write('\n')
#
#
#
#     fw.close()
#


def sim_result_general_presentation_last(nth, graph, resultdir, numev, **results):
    print('makeing figures...')
    keyname = ''
    for key in results.keys():
        keyname += '_'+ key
    basepath = os.getcwd()
    resultdir = resultdir+'/result{}_{}'.format(keyname, nth)
    print(os.path.join(basepath, resultdir))
    dirpath = os.path.join(basepath, resultdir)
    createFolder(dirpath)

    keylist = list(results.keys())
    linetype = ['-', '--', ':', '-.']

    fig = plt.figure(figsize=(12, 12), dpi=300)
    for i in range(numev):
        kth_result=1
        cnt = 0
        totcost = []
        for key, EVlist in results.items():
            pev = EVlist[i]
            xx = []
            yy = []
            nth=0
            ax = fig.add_subplot(2, 2, cnt+1)
            ax.set_title(key)
            for nid in pev.path:
                x, y = graph.nodes_xy(nid)
                plt.text(x,y,str(nth))
                xx.append(x)
                yy.append(y)
                nth+=1
            ax.plot(xx, yy, linetype[cnt])
            cnt += 1
            if pev.cs != None:
                cs_x, cs_y = graph.nodes_xy(pev.cs.id)
                ax.plot(cs_x, cs_y, 'D', label=key+' EVCS')
            kth_result+=1

            s_x, s_y = graph.nodes_xy(pev.source)
            # ax.set_xlim(graph.minx - 1, graph.maxx + 1)
            # ax.set_ylim(graph.miny - 1, graph.maxy + 1)
            ax.plot(s_x, s_y, 'p', label='Source')

            d_x, d_y = graph.nodes_xy(pev.destination)
            ax.set_xlim(graph.minx-1, graph.maxx+1)
            ax.set_ylim(graph.miny-1, graph.maxy+1)
            ax.plot(d_x, d_y, 'p', label='Destination')
            plt.legend()

            totcost.append(pev.totalcost)

        ax = fig.add_subplot(2, 2, 4)
        ax.set_title('total cost')
        ax.bar(keylist, totcost)

        fig = plt.gcf()
        fig.savefig('{}/route_{}.png'.format(resultdir, i), facecolor='#eeeeee', dpi=300)
        plt.clf()


    plt.title('Selected EVCS')
    plt.xlabel('EV index')
    plt.ylabel('EVCS ID')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cs.id)
        plt.plot(range(len(r1_list)), r1_list,'x',  label=key)
    plt.legend()
    # plt.xlim(graph.minx, graph.maxx)
    # plt.ylim(graph.miny, graph.maxy)
    fig = plt.gcf()
    fig.savefig('{}/EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()


    plt.figure(figsize=(12, 6), dpi=300)

    plt.title('CS Charging Cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    cnt=0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingcost)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt+=1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/CS Charging Cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Home Charging Cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    cnt=0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingcost)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt+=1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Home Charging Cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()


    plt.title('Driving distance')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving distance.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Distance from S to EVCS')
    plt.xlabel('EV ID')
    plt.ylabel('Distance(km)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.fdist)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Distance from S to EVCS.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Driving time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        numev = len(EVlist)
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Driving time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('CS Charging energy')
    plt.xlabel('EV ID')
    plt.ylabel('Energy(kWh)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cscharingenergy)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/CS Charging energy.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Home Charging energy')
    plt.xlabel('EV ID')
    plt.ylabel('Energy(kWh)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingenergy)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Home Charging energy.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('CS Charging time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/CS Charging time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Waiting time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Waiting time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Total travel time')
    plt.xlabel('EV ID')
    plt.ylabel('Time(h)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Total travel time.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('Total travel cost')
    plt.xlabel('EV ID')
    plt.ylabel('Cost($)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totalcost)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/Total travel cost.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    plt.title('EV SOC')
    plt.xlabel('EV ID')
    plt.ylabel('SOC(%)')
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.init_SOC)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    cnt = 0
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.curr_SOC)
        plt.plot(r1_list, linetype[cnt], label=key)
        cnt += 1
    plt.legend()
    fig = plt.gcf()
    fig.savefig('{}/ev SOC.png'.format(resultdir), facecolor='#eeeeee', dpi=300)
    plt.clf()

    #=======================================================================================

def sim_result_text_last(nth, CS_list, graph, resultdir, **results):
    print('makeing documents...')
    keyname = ''
    for key in results.keys():
        keyname += key + '_'
    print(keyname)

    fw = open('{}/data_{}_{}.txt'.format(resultdir, keyname, nth), 'w', encoding='UTF8')

    fw.write('\ncs price\n')
    for cs in CS_list:
        r1_list = []
        for p in cs.price:
            r1_list.append(p)
        fw.write(str(cs.id) + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs wait\n')
    for cs in CS_list:
        r1_list = []
        for p in cs.waittime:
            r1_list.append(p)
        fw.write(str(cs.id) + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nlink velocity\n')
    for l_id in graph.link_data.keys():
        fw.write(str(l_id) + '\t' + str(graph.traffic_info[l_id]) + '\n')

    fw.close()



    fw = open('{}/result_{}_{}.txt'.format(resultdir, keyname, nth), 'w', encoding='UTF8')


    fw.write('\nSum total travel cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totalcost)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total expense_time_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_time_part)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total expense_cost_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_cost_part)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')


    fw.write('\nSum total travel time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')
    fw.write('\nSum total cs charging time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')

    fw.write('\nSum total cs waiting time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')




    fw.write('\nSum total driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')



    fw.write('\nSum total cs chargingcost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingcost)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')


    fw.write('\nSum total home chargingcost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingcost)
        fw.write(key + '\t' + str(sum(r1_list)) + '\n')


    fw.write('\ncs\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cs.id)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('cs req soc\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.req_SOC)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')




    fw.write('\nTotal travel cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totalcost)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nTotal travel expense_cost_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_cost_part)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal travel expense_time_part\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.expense_time_part)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')




    fw.write('\ncs driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.csdistance)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.csdrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs charging energy\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cscharingenergy)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs charging cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingcost)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs waiting time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\ncs charging time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')







    fw.write('\nhome driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homedrivingdistance)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homedrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome charging energy\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingenergy)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome charging cost\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingcost)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nhome charging time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homechargingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')




    fw.write('\nTotal Driving distance\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingdistance)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal Driving time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.totaldrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nTotal travel time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingwaitingtime + ev.cschargingtime + ev.totaldrivingtime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('\nSource\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.source)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\ncharging price\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingprice)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('\ninit_SOC\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.init_SOC)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('at cs soc\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cssoc)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')



    fw.write('at home soc\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.homesoc)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')

    fw.write('\nt_start\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.t_start)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')


    fw.write('\nCharging start time\n')
    for key, EVlist in results.items():
        r1_list = []
        for ev in EVlist:
            r1_list.append(ev.cschargingstarttime)
        fw.write(key + '\t')
        for value in r1_list:
            fw.write(str(value) + '\t')
        fw.write('\n')











    fw.write('\nPath\n')
    for key, EVlist in results.items():
        fw.write(key+'\n')
        for ev in EVlist:
            for value in ev.path:
                fw.write(str(value) + '\t')
            fw.write('\n')
        fw.write('\n')


    fw.close()

