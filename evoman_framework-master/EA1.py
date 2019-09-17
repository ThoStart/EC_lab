import sys
import time
import os
import random
import numpy as np

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools


########################### Init Environment ############################

experiment_name = 'log_EA1'
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

env.state_to_log()  # checks environment state

# Optimization for controller solution (best genotype-weights for phenotype-network): Genetic Algorithm

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5  # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
npop = 2
gens = 20
mutation = 0.2
cxpb = 0.2
last_best = 0
indpb = 0.2
tournsize = 5
mu = 0
sigma = 1

np.random.seed(112)


########################### Init Tools and Functions ############################

def evaluate(individual):
    f, p, e, t = env.play(pcont=np.array(individual))
    return f,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, dom_l, dom_u)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=mu, sigma=sigma, indpb=indpb)
toolbox.register("select", tools.selTournament, tournsize=tournsize)


########################### Init Population (or read old file) ############################

if not os.path.exists(experiment_name+'/evoman_solstate'):
    print( '\nNEW EVOLUTION\n')

    pop = toolbox.population(n=npop)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    start_gen = 0
    solutions = [pop, fitnesses]
    env.update_solutions(solutions)

else:

    print('\nCONTINUING EVOLUTION\n')

    env.load_state()
    pop = env.solutions[0]
    fit_pop = env.solutions[1]


################################## Evolution  ###################################

for g in range(gens):
    print("Generation: ", g)
    offspring = toolbox.select(pop, len(pop)) # we can vary tournament size here
    offspring = list(map(toolbox.clone, offspring))

    print("Recombination")
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    print("Mutate")
    for mutant in offspring:
        if random.random() < mutation:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    print("Evaluate")

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in pop]

    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x * x for x in fits)
    std = abs(sum2 / length - mean ** 2) ** 0.5

    print("  Min %s" % min(fits))
    print("  Max %s" % max(fits))
    print("  Avg %s" % mean)
    print("  Std %s" % std)

    ########################### Safe Evoman Log / Performance Log ############################
    file_aux = open(experiment_name+'')
    solutions = [pop, fits]
    env.update_solutions(solutions)
    env.save_state()