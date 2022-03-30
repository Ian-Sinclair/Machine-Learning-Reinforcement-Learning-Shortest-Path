# -*- coding: utf-8 -*-
"""
Created on Sun May 30 21:55:04 2021

@author: IanSi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import random
import time

"""
Initialize Graph Data set
"""
G = nx.gnm_random_graph(26,80,seed = 2021,directed = False)
destination = 25 #End of path

for i in G.nodes():
    G.nodes[i]['State'] = False
    G.nodes[i]['start'] = False
    G.nodes[i]['end'] = False

G.nodes[0]['start'] = True
G.nodes[24]['end'] = True


color_map = nx.get_node_attributes(G, name = "State")
for key in color_map:
        color_map[key] = "green"
color_map[0] = "red"
color_map[destination] = "red"


plt.figure(1)
colors = [color_map.get(node) for node in G.nodes()]
nx.draw_networkx(G,node_color = colors, node_size = 300, pos = nx.circular_layout(G),arrows = True, with_labels = True)


"""
Find Shortest Path using dijkstra's algorithm'
"""
Control_Path = nx.shortest_path(G, source=0, target=destination, weight=None, method='dijkstra')
for i in Control_Path:
    print(i)


"""
Initialize state and action space for Markov Decision Process (MDP)
"""
state_space = np.arange(nx.number_of_nodes(G))
#Fix action space
action_space = []

#Build Q-Table, is hard because action space does not have consistent rows...
maxActions = 0
for node in G:
    size = 0
    for i in nx.all_neighbors(G,node):
        size+=1
    if size > maxActions:
        maxActions = size

qtable = np.zeros((state_space.size, maxActions))

for node in G:
    size = 0
    for i in nx.all_neighbors(G,node):
        size+=1
    for j in range(size,maxActions,1):
        qtable[node][j] = -1

"""
Hyperparameters
"""
Total_episodes = 100
learning_rate = 0.8
max_steps = 50
gamma = 0.95
#Exploration Parameters
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.01

"""
Q-learning Algorithm
"""
rewards = []

for episode in range(Total_episodes):
    Path = []
    state_prev = 0
    state = 0
    G.nodes[state]['State'] = True
    step = 0
    done = False
    total_rewards = 0
    
    for step in range(max_steps):
        G.nodes[state]['State'] = True
        color_map = nx.get_node_attributes(G, name = "State")
        for key in color_map:
            if color_map[key] == True:
                color_map[key] = "green"
            else:
                color_map[key] = "blue"
        #For current action (s) choose an action (a) by selecting a neighbor
        
        #Epsilon Splitting
        exp_exp_tradeoff = random.uniform(0, 1)
        
        if exp_exp_tradeoff > epsilon: #Exploitation
            QargMax = qtable[state][0]
            size = 0
            for i in nx.all_neighbors(G,state):
                if qtable[state][size] > QargMax:
                    QargMax = qtable[state][size]
                    new_state = i
                size+=1
            action = np.argmax(qtable[state,:])
        if exp_exp_tradeoff <= epsilon: #Exploration
            size = 0
            for i in nx.all_neighbors(G,state):
                size +=1
            edgeIndex = int(random.uniform(0, size))
            size = 0
            for i in nx.all_neighbors(G,state):
                if size == edgeIndex:
                    new_state = i
                    action = edgeIndex
                size += 1
        
        #Perform the action, record the reward and new state.
        #new_state, reward, done, info = env.step(action)
        
        reward = random.uniform(0, 100)
        
        if new_state == destination:
            done = True
            
        qtable[state, action] = qtable[state, action] + learning_rate * (reward + gamma * np.max(qtable[new_state, :]) - qtable[state, action])
        
        #Update Q-Function
        total_rewards += reward
        #Update state
        color_map[state] = "blue"
        Path.append(state)
        state = new_state
        #Check if destination has been reached
        if done == True: 
            break
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode) 
        rewards.append(total_rewards)