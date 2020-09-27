from deap import base
from deap import creator
from deap import tools
import random
import numpy as np
import pandas as pd
import os
from Farm_Eval import getAEP, getTurbLoc, loadPowerCurve, binWindResourceData, checkConstraints, preProcessing

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


import matplotlib.pyplot as plt
# import seaborn as sns
# import elitism

#problem constants:

DIMENSIONS = 2
BOUND_LOW = 50
BOUND_UP = 3950

#genetic algorithm constants


#set random seeds
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# Single Objective Function
creator.create("FitnessMax", base.Fitness, weights=(1.0,))

creator.create("Individual", list, fitness=creator.FitnessMax)

# Function to generate random numbers for x and y coordinates

def randomCoordinates(low, up):
    return [(random.uniform(l, u), random.uniform(l,u)) for l, u in zip([low] * DIMENSIONS, [up] * DIMENSIONS)]

toolbox.register("attrCoordinates", randomCoordinates, BOUND_LOW, BOUND_UP)

toolbox.register("individualcreator", tools.initIterate, creator.Individual, toolbox.attrCoordinates)

toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualcreator)



"""
toolbox = base.Toolbox()
toolbox.register("attr_flt", random.uniform, BOUND_LOW, BOUND_UP)
toolbox.register("attr_float", random.uniform, BOUND_LOW, BOUND_UP)
toolbox.register("individual", tools.initRepeat, creator.Individual,
             (toolbox.attr_flt, toolbox.attr_float),
             n=50)
toolbox.register("evaluate", getAEP)
toolbox.decorate("evaluate", tools.DeltaPenalty(checkConstraints, 7.0, Calculate_Constraints))
"""


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
    checkConstraints(turb_coords, turb_diam)

    print('Calculating AEP......')
    AEP = getAEP(turb_rad, turb_coords, power_curve, wind_inst_freq,
                 n_wind_instances, cos_dir, sin_dir, wind_sped_stacked, C_t)
    print('Total power produced by the wind farm is: ', "%.12f" % (AEP), 'GWh')

    













