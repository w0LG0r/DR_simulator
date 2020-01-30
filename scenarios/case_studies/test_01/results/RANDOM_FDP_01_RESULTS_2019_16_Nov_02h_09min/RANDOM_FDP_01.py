#%%
import logging
import os
import sys
import openpyxl
import shutil
from random import seed
from datetime import datetime, timedelta
from pandas import set_option, ExcelWriter, date_range
from model import Scenario
from utils import prepare_result_directory, get_profile_data_from_excel
from strategies import QLearning, MaxSelection
from profile_generators import (
    create_buffer_profile,
    create_solar_generator_profile,
    create_fdp_inputs,
    create_grid_constraints,
    create_beta_distributed_inputs,
)
from plot import create_scenario_plots
import logger

# from pycallgraph import PyCallGraph
# from pycallgraph.output import GraphvizOutput


print(
    "\n____________________________________________________________________________________________________________________"
)
print(
    "|                                                                                                                  |"
)
print(
    "|                                          SCENARIO INPUT CONFIG                                                   |"
)
print(
    "|__________________________________________________________________________________________________________________|\n"
)

""" ________________________________  INPUT FILE """
scenario_name = os.path.basename(__file__).split(".")[0]
print("Scenario name:\t\t\t{}\n".format(scenario_name))

# scenario_name += "_100_EP_GREEDY_INCREASE_PROBS_Q_WHEN_MAX"

pathname = os.path.dirname(__file__)
print("Pathname:\t\t\t\t{}".format(pathname))

excel_file_name = os.path.basename(__file__).split(".")[0] + ".xlsx"
print("Excel file name:\t\t{}".format(excel_file_name))

excel_file_location = pathname + "/" + excel_file_name
print("Excel file location:\t{}".format(excel_file_location))

# scenario_path=pathname.rsplit("simulator\\")[-1]
# print('scenario_path: ', scenario_path)

# graphviz = GraphvizOutput(output_file='convert_2_generator.png')
show_preview = False

scenario_description = ""
""" ________________________________  RESULTS FOLDER """

# Add a directory for scenario results with a timestamp
result_directory = prepare_result_directory(
    scenario_name=scenario_name, path_name=pathname
)

""" ________________________________  RUNTIME """

# Number of daily episodes
episodes = 100

# Slice a daily episode
episode_duration = timedelta(hours=24)
# assert episode_duration > timedelta(hours=1), ""

# Choose either 15 minutes or 60 minutes
# resolution = timedelta(minutes=15)
resolution = timedelta(minutes=60)

simulation_start = datetime(year=2018, month=6, day=1, hour=0)

simulation_end = simulation_start + timedelta(days=episodes)

""" ________________________________  UNITS """

units = ["kW", "kWh"]

""" ________________________________  SEED """

# Set a seed value for all random generators
seed_value = 23

""" ________________________________  AGGREGATOR """

aggregator_name = "Aggregator"
prediction_delta_Aggregator = resolution * 3

""" ________________________________  FLEXOFFER STRATEGY """

flexoffer_strategy = QLearning(
    simulation_start=simulation_start,
    simulation_end=simulation_end,
    episode_duration=episode_duration,
    resolution=resolution,
    last_episode=episodes,
    # NOTE: Choose either "Epsilon_Greedy" or "UCB1"
    exploration="Epsilon_Greedy",
    prediction_delta=prediction_delta_Aggregator,
    alpha=0.025,
    alpha_decay=0.2,
    gamma=0.95,
    # NOTE: float 0.0 = RANDOM, float 1.0 = GREEDY
    epsilon=0.0,
    # NOTE: If epsilon=0.0 and greater than max episode = always random
    start_epsilon_decay=101,
    end_epsilon_decay=102,
    seed_value=seed_value,
    action_probabilites=[1, 1],
    action_probabilites_decay=0.005,
    max_reward_boost=10,
)

# flexoffer_strategy = MaxSelection(
#         simulation_start=simulation_start,
#         simulation_end=simulation_end,
#         episode_duration=episode_duration,
#         prediction_delta=prediction_delta_Aggregator,
#         resolution=resolution,
#         episodes=episodes,
# )

""" ________________________________  FDP """

FDP_name = "FDP"
prediction_delta_FDP = resolution * 3

print("\nFDP input data generation")

# Create inputs for FDP from excel file
create_fdp_inputs(
    result_directory=result_directory,
    excel_file_location=excel_file_location,
    excel_file_name=excel_file_name,
    seed_value=seed_value,
    start=simulation_start,
    end=simulation_end,
    resolution=resolution,
    upward_capacity_max=8,
    downward_capacity_max=8,
    autocorrelation=1,
    std=5,
    market_prices_noise_parameter=(1, 0.25),
    shape_weight=0.1,
    show_preview=show_preview,
    repeat_daily=True,
    pickled_profile="C:/Users/W/Desktop/ComOpt/comopt/simulator/scenarios/case_studies/master_thesis/FDP_PROFILES/PROFILE_FDP_01.p",
)

# create_beta_distributed_inputs(
#     start=simulation_start,
#     end=simulation_end,
#     resolution=resolution,
#     episodes=episodes,
#     result_directory=result_directory,
#     excel_file_location=excel_file_location,
#     excel_file_name=excel_file_name,
#     seed_value = seed_value,
#     imbalances=True,
#     market_prices=True,
#     scenario_name=scenario_name
# )

""" ________________________________ 1 EMS """

# Set number of active energy management systems
number_of_EMS = 1

# Set tariffs
power_purchase_tariff_per_EMS = [6]
power_feedin_tariff_per_EMS = [3]
flexibility_tariff_per_EMS = [9]

storage_charging_capacities = [12]
peak_power_solar = [26]

# ''' ________________________________ 2 EMS '''

# # Set number of active energy management systems
# number_of_EMS = 2

# # Set tariffs
# power_purchase_tariff_per_EMS = [5,6]
# power_feedin_tariff_per_EMS = [2,3]
# flexibility_tariff_per_EMS = [8,9]

# storage_charging_capacities = [16,20]
# peak_power_solar = [20,26]

for EMS in range(number_of_EMS):

    print("\n{} input data generation".format("EMS_" + str(EMS + 1)))

    # Delete sheets in excel
    create_grid_constraints(
        result_directory=result_directory,
        excel_file_location=excel_file_location,
        excel_file_name=excel_file_name,
        EMS="EMS_" + str(EMS + 1),
        start=simulation_start,
        end=simulation_end,
        resolution=resolution,
        power_capacity_max=100,
        feedin_capacity_max=100,
        show_preview=show_preview,
        # pickled_profile=pathname+"/profiles/Grid.p",
    )

    create_buffer_profile(
        seed_value=seed_value,
        result_directory=result_directory,
        excel_file_location=excel_file_location,
        excel_file_name=excel_file_name,
        EMS="EMS_" + str(EMS + 1),
        start=simulation_start,
        end=simulation_end,
        resolution=resolution,
        prediction_delta=prediction_delta_Aggregator,
        window_size_between=(4, 5),
        sample_rate=0.6,
        charging_capacity=storage_charging_capacities[EMS],
        show_preview=show_preview,
        # pickled_profile=pathname+"/profiles/Storage.p",
    )

    # create_solar_generator_profile(
    #     seed_value=seed_value,
    #     result_directory=result_directory,
    #     excel_file_location=excel_file_location,
    #     excel_file_name=excel_file_name,
    #     copy_from_excel=True,
    #     EMS= "EMS_"+str(EMS+1),
    #     start=simulation_start,
    #     end=simulation_end,
    #     resolution=resolution,
    #     histogram_parameters=(1,0.01),
    #     peak_power=peak_power_solar[EMS],
    #     dispatch_factor=1,
    #     show_preview=False,
    #     repeat_daily=True
    #     # pickled_profile=pathname+"/profiles/Generator.p",
    # )

# Copy excel file from files folder to scenario result folder
shutil.copy(excel_file_location, result_directory)
shutil.copy(os.path.abspath(__file__), result_directory)
shutil.copy(pathname.rsplit("scenarios")[0] + "strategies.py", result_directory)

""" ________________________________  CREATE SCENARIO """

scenario = Scenario(
    # GENERAL
    name=scenario_name,
    description=scenario_description,
    result_directory=result_directory,
    excel_file_name=excel_file_name,
    units=units,
    # TIME
    simulation_start=simulation_start,
    simulation_end=simulation_end,
    prediction_delta=prediction_delta_Aggregator,
    resolution=resolution,
    episodes=episodes,
    episode_duration=episode_duration,
    number_of_EMS=number_of_EMS,
)
scenario.add_Aggregator(
    aggregator_name=aggregator_name,
    prediction_delta_Aggregator=prediction_delta_Aggregator,
    flexoffer_strategy=flexoffer_strategy,
)

scenario.add_FDP(
    FDP_name=FDP_name, prediction_delta_FDP=prediction_delta_FDP,
)

scenario.add_EMS(
    power_purchase_tariff_per_EMS=power_purchase_tariff_per_EMS,
    power_feedin_tariff_per_EMS=power_feedin_tariff_per_EMS,
    flexibility_tariff_per_EMS=flexibility_tariff_per_EMS,
)

if __name__ == "__main__":

    # Console output configuration
    set_option("display.max_columns", 25)
    set_option("display.max_rows", 1000)
    set_option("display.width", 1000)
    set_option("display.colheader_justify", "left")
    set_option("max_info_columns", 1000)
    set_option("display.precision", 3)
    set_option("display.float_format", "{:.2f}".format)

    # with PyCallGraph(output=graphviz):
    scenario.run

    create_scenario_plots(
        scenario_path=scenario.result_directory.rsplit("simulator\\")[-1],
        # optimization=True,
        rewards_overview=True,
        orders=True,
        # reward_per_daytime_and_episode=True,
        # reward_per_daytime_and_episode_cumulated=True,
        # q_tables=True,
        # performed_action=True,
        # performed_action_cum=True,
    )


# %%
