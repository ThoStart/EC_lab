import sys
sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller

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
                  speed="fastest")
