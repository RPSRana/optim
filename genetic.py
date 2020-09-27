from deap import base
from deap import creator
from deap import tools
import random
import numpy as np
import pandas as pd
import os
from Farm_Eval import getAEP, getTurbLoc, loadPowerCurve, binWindResourceData, checkConstraints, preProcessing, \
    Calculate_Constraints
from elitism import eaSimpleWithElitism
import matplotlib.pyplot as plt
import seaborn as sns

data_path = 'Shell_Hackathon Dataset/'
turb_specs = {
    'Name': 'Anon Name',
    'Vendor': 'Anon Vendor',
    'Type': 'Anon Type',
    'Dia (m)': 100,
    'Rotor Area (m2)': 7853,
    'Hub Height (m)': 100,
    'Cut-in Wind Speed (m/s)': 3.5,
    'Cut-out Wind Speed (m/s)': 25,
    'Rated Wind Speed (m/s)': 15,
    'Rated Power (MW)': 3
}
turb_diam = turb_specs['Dia (m)']
turb_rad = turb_diam / 2


# Constants
BOUND_LOW = 50
BOUND_UP = 3950
NUM_OF_PARAMS = 2

# Genetic Algorithm constants:
POPULATION_SIZE = 50
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.3   # probability for mutating an individual
MAX_GENERATIONS = 1
HALL_OF_FAME_SIZE = 5
CROWDING_FACTOR = 15.0  # crowding factor for crossover and mutation
LAMBDA = 0.01

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def AEP(individual):
    x = individual[::2]
    y = individual[1::2]
    turb_coords = np.column_stack([x, y])
    turb_coords = turb_coords.astype(np.float32)
    # Turbine x,y coordinates
    # turb_coords = getTurbLoc(os.path.join(data_path, 'turbine_loc_test.csv'))

    # Load the power curve
    power_curve = loadPowerCurve(os.path.join(data_path, 'power_curve.csv'))

    # Pass wind data csv file location to function binWindResourceData.
    # Retrieve probabilities of wind instance occurence.
    wind_inst_freq = binWindResourceData(os.path.join(data_path, 'Wind Data/wind_data_2007.csv'))

    # Doing preprocessing to avoid the same repeating calculations. Record
    # the required data for calculations. Do that once. Data are set up (shaped)
    # to assist vectorization. Used later in function totalAEP.
    n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t = preProcessing(power_curve)

    # check if there is any constraint is violated before we do anything. Comment
    # out the function call to checkConstraints below if you desire. Note that
    # this is just a check and the function does not quantifies the amount by
    # which the constraints are violated if any.
    # checkConstraints(turb_coords, turb_diam)

    print('Calculating AEP......')
    AEP = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq,
                 n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
    print('Total power produced by the wind farm is: ', "%.12f" % (AEP), 'GWh')
    violations = Calculate_Constraints(turb_coords, turb_diam)
    print('Total violations: ', "%.12f" % (violations))
    return (LAMBDA * violations + AEP),

toolbox = base.Toolbox()

# Single Objective Function
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox.register("attr_flt", random.uniform, BOUND_LOW, BOUND_UP)
toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
toolbox.register("individual", tools.initCycle, creator.Individual,
             (toolbox.attr_flt, toolbox.attr_float),
             n=50)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individual)


toolbox.register("evaluate", AEP)

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate",
                 tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW,
                 up=BOUND_UP,
                 eta=CROWDING_FACTOR)

toolbox.register("mutate",
                 tools.mutPolynomialBounded,
                 low=BOUND_LOW,
                 up=BOUND_UP,
                 eta=CROWDING_FACTOR,
                 indpb=1.0 / NUM_OF_PARAMS)


# Genetic Algorithm flow:
def main():

    # create initial population (generation 0):
    population = toolbox.populationCreator(n=POPULATION_SIZE)

    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    stats.register("avg", np.mean)

    # define the hall-of-fame object:
    hof = tools.HallOfFame(HALL_OF_FAME_SIZE)

    # perform the Genetic Algorithm flow with hof feature added:
    population, logbook = eaSimpleWithElitism(population,
                                                      toolbox,
                                                      cxpb=P_CROSSOVER,
                                                      mutpb=P_MUTATION,
                                                      ngen=MAX_GENERATIONS,
                                                      stats=stats,
                                                      halloffame=hof,
                                                      verbose=True)

    # print best solution found:
    print("- Best solution is: ")
    # print("params = ", test.formatParams(hof.items[0]))
    print("Accuracy = %1.5f" % hof.items[0].fitness.values[0])

    # extract statistics:
    maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

    # plot statistics:
    sns.set_style("whitegrid")
    plt.plot(maxFitnessValues, color='red')
    plt.plot(meanFitnessValues, color='green')
    plt.xlabel('Generation')
    plt.ylabel('Max / Average Fitness')
    plt.title('Max and Average fitness over Generations')
    plt.show()


if __name__ == "__main__":
    main()

