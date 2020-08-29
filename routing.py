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


ECRate = 0.16
UNITtimecost = 8


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
                # priority = new_cost + heuristic_astar(graph.nodes_xy(goal), graph.nodes_xy(next))
                frontier.put(next, priority)
                # print('frontier.put()', next, priority)
                came_from[next] = current

    return came_from, cost_so_far

def gen_envir_jeju(traffic_data_path, num_evs):
    np.random.seed(10)
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

def gen_envir_simple(num_evs):
    np.random.seed(10)
    graph = Graph_simple()

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

# def gen_evcs_random(cs):
#     cs.price = np.random.normal(cs.alpha, 0.15 * cs.alpha)
#     while cs.price < 0:
#         cs.price = np.random.normal(cs.alpha, 0.15 * cs.alpha)
#
#     cs.waittime = np.random.normal(-1200 * (cs.price - 0.07), 20)/60
#     while cs.waittime < 0:
#         cs.waittime = 0

def update_envir_timeweight(CS_list, graph, sim_time):

    time_idx = int(sim_time/5)
    # print(sim_time, time_idx)
    count = 0
    avg_price = 0.0

    for cs in CS_list:
        # gen_evcs_random(cs)
        avg_price += cs.price[time_idx]
    avg_price = avg_price / len(CS_list)

    for l_id in graph.link_data.keys():
        velo = graph.traffic_info[l_id][time_idx]
        # print(graph.link_data[l_id]['LENGTH'], velo)
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


def update_ev(pev, simple_graph, fnode, tnode, sim_time):
    time_idx = int(sim_time / 5)
    dist = simple_graph.distance(fnode, tnode)
    velo = simple_graph.velocity(fnode, tnode, time_idx)

    time_diff = (dist/velo)
    pev.traveltime = pev.traveltime + time_diff
    pev.totaldrivingtime += time_diff
    pev.totaldrivingdistance += dist
    soc_before = pev.curr_SOC
    pev.curr_SOC = pev.curr_SOC - (dist * pev.ECRate) / pev.maxBCAPA
    pev.totalenergyconsumption = pev.totalenergyconsumption + dist * pev.ECRate

    sim_time = sim_time + time_diff*60
    time_idx = int(sim_time / 5)
    if time_idx >= 288:
        print(sim_time, time_idx)
        print('time idx error')
        # input()
        return 0, 0

    # print("fnode {} tnode {} dist {} velo {} soc_b {} soc_a {}".format(fnode, tnode, dist, velo, soc_before, pev.SOC))
    if pev.charged != 1:
        pev.fdist += dist

    return sim_time, time_diff

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






def sim_main_first_time_check(sim_time, pev, CS_list, graph, ncandi):

    evcango = pev.curr_SOC * pev.maxBCAPA / pev.ECRate
    start = pev.curr_location
    end = pev.destination
    info = []
    avg_price = update_envir_timeweight(CS_list, graph, sim_time)
    # cslist = pruning_evcs(pev, CS_list, graph, ncandi)

    cslist = CS_list
    for cs in cslist:
        # cs, _ = cs
        evcs_id = cs.id
        came_from, cost_so_far = a_star_search(graph, start, evcs_id)
        front_path = reconstruct_path(came_from, start, evcs_id)
        front_path_distance = graph.get_path_distance(front_path)
        came_from, cost_so_far = a_star_search(graph, evcs_id, end)
        rear_path = reconstruct_path(came_from, evcs_id, end)
        rear_path_distance = graph.get_path_distance(rear_path)
        final_path = front_path + rear_path[1:]
        dist = graph.get_path_distance(final_path)
        # total_d_time = graph.get_path_drivingtime(front_path, int(sim_time / 5))

        front_d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5))
        rear_d_time = graph.get_path_drivingtime(final_path, int(sim_time / 5))
        rear_consump_energy = front_path_distance * pev.ECRate

        remainenergy = pev.maxBCAPA*pev.init_SOC - front_path_distance * pev.ECRate
        charging_energy = pev.maxBCAPA*pev.req_SOC - remainenergy
        charging_time = (charging_energy/(cs.chargingpower*pev.charging_effi))
        # w = total_d_time + cs.waittime[int(sim_time / 5)] + c_time
        waiting_time = cs.waittime[int(sim_time / 5)]
        # totaltraveltime = total_d_time + waiting_time + charging_time
        info.append((cs, pev.curr_SOC, front_path, rear_path, front_path_distance, rear_path_distance, front_d_time, rear_d_time, rear_consump_energy, waiting_time, charging_time))

    return info



