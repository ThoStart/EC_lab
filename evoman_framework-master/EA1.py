import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
import numpy as np

import time
import glob, os

experiment_name = 'algorithm_one'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes simulation in individual evolution mode, for single static enemy.
env = Environment(experiment_name=experiment_name,
                  enemies=[1],
                  playermode="ai",
                  player_controller=player_controller(),
                  enemymode="static",
                  level=2,
                  speed='fastest')

# default environment fitness is assumed for experiment

env.state_to_log() # checks environment state


####   Optimization for controller solution (best genotype-weights for phenotype-network): Ganetic Algorihm    ###

ini = time.time()  # sets time marker


# genetic algorithm params

run_mode = 'train' # train or test

n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5 # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
npop = 10
gens = 30
mutation = 0.2
last_best = 0

np.random.seed(112)

#Play game for 10 population size, random init
pop = np.random.uniform(1, -1, (npop,n_vars))

def simulate(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

fit_pop = np.array(list(map(lambda y: simulate(env, y), pop)))

print(fit_pop)