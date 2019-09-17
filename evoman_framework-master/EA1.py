import sys
import time
import os
import random
import numpy as np

sys.path.insert(0, 'evoman')
from environment import Environment
from demo_controller import player_controller
from deap import base, creator, tools


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

env.state_to_log()  # checks environment state

# Optimization for controller solution (best genotype-weights for phenotype-network): Genetic Algorithm

ini = time.time()  # sets time marker

# genetic algorithm params

run_mode = 'train'  # train or test

n_hidden = 10
n_vars = (env.get_num_sensors()+1)*n_hidden + (n_hidden+1)*5  # multilayer with 10 hidden neurons
dom_u = 1
dom_l = -1
npop = 10
gens = 30
mutation = 0.2
last_best = 0

np.random.seed(112)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(individual):
    f, p, e, t = env.play(pcont=np.array(individual))
    return f,

# ----------
# Operator registration
# ----------

toolbox.register("evaluate", evaluate)

# register the crossover operator
toolbox.register("mate", tools.cxTwoPoint)

# register a mutation operator with a probability to
# flip each attribute/gene of 0.05
toolbox.register("mutate", tools.mutPolynomialBounded(toolbox.individual, 2, -1, 1, indpb), indpb=0.05)

# operator for selecting individuals for breeding the next
# generation: each individual of the current generation
# is replaced by the 'fittest' (best) of three individuals
# drawn randomly from the current generation.
toolbox.register("select", tools.selTournament, tournsize=3)


pop = toolbox.population(n=npop)

# CXPB  is the probability with which two individuals are crossed
# MUTPB is the probability for mutating an individual
NGEN = 50
CXPB = 0.5
MUTPB = 0.2


fitnesses = list(map(toolbox.evaluate, pop))

for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

# Extracting all the fitnesses of population
fits = [ind.fitness.values[0] for ind in pop]

for g in range(NGEN):
    print("--- Generation %i ---" % g)

    # Select the next generation individuals
    offspring = toolbox.select(pop, len(pop))
    # Clone the selected individuals
    offspring = list(map(toolbox.clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):

        if random.random() < CXPB:
            toolbox.mate(child1, child2)

            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:

        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    print("  Evaluated %i individuals" % len(invalid_ind))

    pop[:] = offspring

    fits = [ind.fitness.values[0] for ind in pop]





print('second individual fitness value: ', pop[1].fitness)