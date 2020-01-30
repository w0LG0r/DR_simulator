import os
from functools import wraps

from typing import List, Optional, Union, Tuple, NamedTuple, Type
from datetime import date, datetime, timedelta
from numpy import sign, ndarray, nan
from pandas import *
from pandas import IndexSlice as IDX
from pandas.tseries.frequencies import to_offset

# from functools import wraps
import time
import pickle
from logger import log as logger
import gc

""" _____________________________________________________________ I/O """


def get_profile_data_from_excel(
    # horizon: DatetimeIndex,
    entity: str,
    excel_file_location: str,
    simulation_time: DatetimeIndex,
) -> dict:

    """ 
        Only assign cells that have actual values and leave other cells empty.
        (do not use 0 or nan)
        EMS: One excel file per EMS with sheetnames as devices:
            ["Load", "Generator", "Storage", "Grid"]
            Each sheet contains the following columns:
            ["equals", "max", "min", "derivative equals", "derivative max",
            "derivative min", "shifting", "shifting_start", "shifting_end", 
            "interdelay"]
            Shifting inputs are binary values to mark timeslots
        
        FDP: One excel file with columns: 
            ["imbalances", "market_prices", "deviation_price_up", 
            "deviation_down_function", "deviation_up_function", 
            "breakpoint_up","breakpoint_down"]
            Deviation_price_function can be "Linear", "Quadratic", "Log", "Exp"
            Breakpoints are either: 
                        1) None, then function increases monotonically 
                        2) Multiplicative factor that returns breakpoint relative
                           scale to flex request value
            
            Imbalances come as ENERGY values per step, announces at t0, and valid for t0+simulation_step
    """

    if "FDP" in entity:
        sheet_names = ["FDP"]

    elif "EMS" in entity:

        sheet_names = [
            entity + "_Load",
            entity + "_Generator",
            entity + "_Storage",
            entity + "_Grid",
        ]

        device_names = ["Load", "Generator", "Storage", "Grid"]

        grid_constraint = None

        # Output variable
        device_constraints = dict.fromkeys(device_names, "")

        # Output variable
        active_devices = []

    # ix = 0
    for sheet_name in sheet_names:

        parsed_df = read_excel(
            io=excel_file_location,
            sheet_name=sheet_name,
            header=0,
            squeeze=True,  
        )
        parsed_df.set_index("datetime", inplace=True)

        # Exit after first sheet, if FDP
        if "FDP" in entity:
            assert (
                parsed_df["deviation_price_down"] <= 0
            ).all(), "Downward deviation prices needs to be negative numbers!"

            assert (
                parsed_df["deviation_price_up"] >= 0
            ).all(), "Upward deviation prices needs to be positive numbers!"

            return parsed_df

        try:

            # only use sheets that have any valid values
            if parsed_df.isnull().all().all():
                logger.warning("{}: INACTIVE".format(sheet_name))
                if "Grid" in sheet_name:
                    raise ValueError(
                        logger.warning(
                            "ERROR ---x {}: No grid connection bounds!".format(
                                input_file
                            )
                        )
                    )

            else:
                # Check if grid values are valid
                if "Grid" in sheet_name:
                    assert (
                        parsed_df["derivative min"] <= 0
                    ).all(), (
                        "Grid feed in constraint must be less than or equal to zero"
                    )
                    assert (
                        parsed_df["derivative max"] >= 0
                    ).all(), "Grid power supply constraint must be greater than or equal to zero"
                    grid_constraint = parsed_df

                # Check if load values are valid
                elif "Load" in sheet_name:
                    assert (
                        parsed_df["derivative min"].all() >= 0
                    ), "Load can't produce power, change derivative min values to greater than or equal to 0"
                    device_constraints["Load"] = parsed_df
                    active_devices.append("Load")

                # Check if load values are valid
                elif "Generator" in sheet_name:
                    assert (
                        parsed_df["derivative max"].all() <= 0
                    ), "Generator can't consume power, change derivative max values to less than or equal to 0"
                    device_constraints["Generator"] = parsed_df
                    active_devices.append("Generator")

                elif "Storage" in sheet_name:
                    device_constraints["Storage"] = parsed_df
                    active_devices.append("Storage")

                logger.warning("   {}".format(sheet_name))
        except:
            pass

    # Check wheter any devices exist
    assert len(active_devices) > 0, "No active devices available, change data input!"

    # Output tuple
    ems_data = (
        active_devices,
        {
            device: profile
            for device, profile in device_constraints.items()
            if not len(profile) == 0
        },
        grid_constraint,
    )
    return ems_data


def get_profile_data_from_pickle(
    # horizon: DatetimeIndex,
    entity: str,
    result_directory: str,
    simulation_time: DatetimeIndex,
) -> dict:

    """ 
        Only assign cells that have actual values and leave other cells empty.
        (do not use 0 or nan)
        EMS: One excel file per EMS with sheetnames as devices:
            ["Load", "Generator", "Storage", "Grid"]
            Each sheet contains the following columns:
            ["equals", "max", "min", "derivative equals", "derivative max",
            "derivative min", "shifting", "shifting_start", "shifting_end", 
            "interdelay"]
            Shifting inputs are binary values to mark timeslots
        
        FDP: One excel file with columns: 
            ["imbalances", "market_prices", "deviation_price_up", 
            "deviation_down_function", "deviation_up_function", 
            "breakpoint_up","breakpoint_down"]
            Deviation_price_function can be "Linear", "Quadratic", "Log", "Exp"
            Breakpoints are either: 
                        1) None, then function increases monotonically 
                        2) Multiplicative factor that returns breakpoint relative
                           scale to flex request value
            
            Imbalances come as ENERGY values per step, announces at t0, and valid for t0+simulation_step
    """

    if "FDP" in entity:

        parsed_df = pickle.load(open(result_directory + "/PROFILES/FDP_Inputs.p", "rb"))
        return parsed_df

    elif "EMS" in entity:

        pickle_files = [
            entity + "_Load.p",
            entity + "_Generator.p",
            entity + "_Storage.p",
            entity + "_Grid.p",
        ]

        device_names = ["Load", "Generator", "Storage", "Grid"]

        grid_constraint = None

        # Output variable
        device_constraints = dict.fromkeys(device_names, "")

        # Output variable
        active_devices = []
        # # Loop over files and sheets

        for file in pickle_files:

            try:
                parsed_df = pickle.load(
                    open(result_directory + "/PROFILES/" + file, "rb")
                )

                # only use files that have any valid values
                if parsed_df.isnull().all().all():
                    logger.warning("\t\t\t\t\t\t     {}: INACTIVE".format(pickle_files))
                    if "Grid" in file:
                        raise ValueError(
                            logger.warning("ERROR ---x : No grid connection bounds!")
                        )

                else:
                    # Check if grid values are valid
                    if "Grid" in file:
                        assert (
                            parsed_df["derivative min"] <= 0
                        ).all(), (
                            "Grid feed in constraint must be less than or equal to zero"
                        )
                        assert (
                            parsed_df["derivative max"] >= 0
                        ).all(), "Grid power supply constraint must be greater than or equal to zero"
                        grid_constraint = parsed_df

                    # Check if load values are valid
                    elif "Load" in file:
                        assert (
                            parsed_df["derivative min"].all() >= 0
                        ), "Load can't produce power, change derivative min values to greater than or equal to 0"
                        device_constraints["Load"] = parsed_df
                        active_devices.append("Load")

                    # Check if load values are valid
                    elif "Generator" in file:
                        assert (
                            parsed_df["derivative max"].all() <= 0
                        ), "Generator can't consume power, change derivative max values to less than or equal to 0"
                        device_constraints["Generator"] = parsed_df
                        active_devices.append("Generator")

                    elif "Storage" in file:
                        device_constraints["Storage"] = parsed_df
                        active_devices.append("Storage")

                    # print('device_constraints[device_names[ix]]: ', device_constraints[device_names[ix]])
                    logger.warning("\t{}".format(file))
            except:
                pass

        # Check wheter any devices exist
        assert (
            len(active_devices) > 0
        ), "No active devices available, change data input!"

        # Output tuple
        ems_data = (
            active_devices,
            {
                device: profile
                for device, profile in device_constraints.items()
                if not len(profile) == 0
            },
            grid_constraint,
        )
        return ems_data


def prepare_result_directory(scenario_name: str, path_name: str):

    """ create a time stamped directory within the result folder """
    # timestamp for result directory
    datetime_now = datetime.now().strftime("%Y_%d_%b_%Hh_%Mmin")

    # create result directory if not existent
    result_directory = os.path.join(
        path_name + "/{}_RESULTS_{}".format(scenario_name, datetime_now)
    )

    if not os.path.exists(result_directory):
        os.makedirs(result_directory)

    # Create directory for output summary text files
    os.mkdir(result_directory + "/DATA")

    # Create directory for output profile pickle files
    os.mkdir(result_directory + "/PROFILES")

    # Create directory for output profile pickle files
    os.mkdir(result_directory + "/PLOTS/")

    # Create directory for output profile pickle files
    os.mkdir(result_directory + "/STRATEGY")

    return result_directory


def load_scenario(file):
    scenario = pickle.load(open(file, "rb"))
    return scenario


def remove_scenario_file_handler():
    logger.handlers.pop()


""" _____________________________________________________________ DATASTRUCTURES """


def create_commitment_stack(
    entity, simulation_time: DatetimeIndex, prediction_delta: Timedelta,
):

    # 2nd column index for commitments at EMS
    commitments_attribute_index = [
        "Profile",
        "Deviation",
        "Commitment",
        "Price_down",
        "Price_up",
    ]

    datetime_indices = []
    horizon_steps = []
    commitment_names = []

    # Time variables
    resolution = to_timedelta(simulation_time.freq)
    horizon_in_steps = int(prediction_delta / resolution)
    # ix = simulation_time.get_loc(horizon[0])
    ix = 1
    index = simulation_time[0]

    commitment_dict = dict.fromkeys(list(range(0, int(timedelta(days=1) / resolution))))
    index_last_commitment_of_day = simulation_time[0] + timedelta(days=1)

    if simulation_time[-1] < index_last_commitment_of_day:
        index_last_commitment_one_day = simulation_time[-1]

    # Create indices
    while index < index_last_commitment_of_day:

        # Create row multiindex from tuples
        arrays = [[index], list(range(1, horizon_in_steps + 1, 1))]
        # tuples = list(zip(*arrays))

        row_multiindex = MultiIndex.from_product(arrays, names=["Time", "Period"])

        # Add Energy contract at first
        if ix == 1:

            c_name = entity.name + str("_") + "EC"

            commitment_column_index_data = [[c_name], commitments_attribute_index]
            commitment_column_index = MultiIndex.from_product(
                commitment_column_index_data
            )

            commitment_dict[0] = DataFrame(
                index=row_multiindex, columns=commitment_column_index, dtype="float"
            )

            c_name = entity.name + str("_") + "COM_1"

            commitment_column_index_data = [[c_name], commitments_attribute_index]
            commitment_column_index = MultiIndex.from_product(
                commitment_column_index_data
            )

            commitment_dict[ix] = DataFrame(
                index=row_multiindex, columns=commitment_column_index, dtype="float"
            )

        # Allows up to 99.999 commitments ~ 1040 days with 15 min resolution
        else:
            c_name = entity.name + str("_") + "COM_" + str(ix)

            commitment_column_index_data = [[c_name], commitments_attribute_index]
            commitment_column_index = MultiIndex.from_product(
                commitment_column_index_data
            )

            commitment_dict[ix] = DataFrame(
                index=row_multiindex, columns=commitment_column_index, dtype="float"
            )

        index += resolution
        ix += 1

    # print('commitment_dict: ', commitment_dict)
    return commitment_dict


def create_database(
    entity, simulation_time: DatetimeIndex, prediction_delta: Timedelta,
) -> DataFrame:

    # Create lists for row and column indices
    datetime_indices = []
    horizon_steps = []

    # Loop variable
    index = simulation_time[0]
    ix = 0

    # Time variables
    resolution = to_timedelta(simulation_time.freq)
    horizon_in_steps = int(prediction_delta / resolution)
    applicable_simulation_time = simulation_time - prediction_delta + resolution

    # Create indices
    while index <= simulation_time[-1]:

        # 1st and 2nd row indices
        for x in range(1, horizon_in_steps + 1, 1):
            horizon_steps.append(x)
            datetime_indices.append(index)

        index += resolution
        ix += 1

    # 2nd column index for all entities summary column
    sum_values_attribute_index = [
        "Request",
        "Offer",
        "Order",
        "Deviation",
        "Baseline",
        "Commitment",
        "Realised",
    ]

    if "EMS" in entity.name:

        # 1st column index
        entity_index = [entity.name]

        # 2nd column index
        sum_values_attribute_index.extend(
            [
                # "COSTS_Base_dev",
                "COSTS_Deviation",
                "COSTS_Base_EC",
                "COSTS_Offer_EC",
                "COSTS_Offer",
                "COSTS_Order",
            ]
        )

        # Add an column for the devices power profile at baseline and at flexrequest
        devices_baseline = [device + "_BL" for device in entity.devices]
        devices_flexoffer = [device + "_FLEX" for device in entity.devices]

        # Insert them at certain column index
        for b, f in zip(devices_baseline, devices_flexoffer):

            sum_values_attribute_index.insert(6, b)
            sum_values_attribute_index.insert(7, f)

        sum_values_attribute_index.insert(5, "Flexprofile")

    elif "Aggregator" in entity.name:

        # 1st column index
        entity_index = [ems for ems in entity.EMS_names]
        entity_index.insert(0, entity.name)

        # 2nd column index
        sum_values_attribute_index.extend(
            [
                "Price_down",
                "Price_up",
                "COSTS_Offer",
                "COSTS_Order",
                "COSTS_Deviation",
                "REVENUES",
                "Price",
            ]
        )

    elif "FDP" in entity.name:

        # 1st column index
        entity_index = ["FDP"]

        # 2nd column index
        sum_values_attribute_index.extend(
            [
                "Price_down",
                "Price_up",
                "Market_prices",
                "COSTS_Deviation",
                "COSTS_Offer",
                "COSTS_Order",
            ]
        )

    # Create column multiindex from product
    sum_values_column_index_data = [entity_index, sum_values_attribute_index]

    # Create row multiindex from tuples
    arrays = [datetime_indices, horizon_steps]
    tuples = list(zip(*arrays))

    row_multiindex = MultiIndex.from_tuples(tuples, names=["Time", "Period"])
    sum_values_column_index = MultiIndex.from_product(sum_values_column_index_data)

    # Sum over commitments per datetime
    flexobjects_sum_values_dfx = DataFrame(
        index=row_multiindex, columns=sum_values_column_index, dtype="float32"
    )

    dfx = flexobjects_sum_values_dfx

    return dfx


def display_constraints(
    EMS_name: str = None,
    device_constraints: dict = None,
    grid_constraint: DataFrame = None,
    horizon: Series = None,
):
    """
        Pretty prints constraints by removing nan columns
    """
    if device_constraints is not None:

        device_column_index = [device for device, profile in device_constraints.items()]
        device_column_index.insert(0, "Grid")
        constraint_column_index = [
            list(profile.columns) for device, profile in device_constraints.items()
        ]

        column_index_data = [device_column_index, constraint_column_index[0]]

        column_index = MultiIndex.from_product(column_index_data)
        horizon.name = "Time"

        df = DataFrame(index=horizon, columns=column_index,)

        for device, profile in device_constraints.items():

            for column in profile.columns:

                df.loc[:, IDX[device, column]] = profile.loc[
                    horizon[0] : horizon[-1], column
                ]

                # Add grid constraints
                df.loc[:, IDX["Grid", column]] = grid_constraint.loc[
                    horizon[0] : horizon[-1], column
                ]

        # Remove columns with only nans and ensure result is a DF for slicing
        df = DataFrame(data=df[df.columns[~df.isnull().any()]])

        # Print relevant profile data
        logger.info("\n{} constraints:\n{} \n".format(EMS_name, df))

    return


""" _____________________________________________________________ HELPERS """


def timer(execution_start_time: datetime):
    return datetime.now() - execution_start_time


def convert_flow(
    values: float, energy_to_power: bool, power_to_energy: bool,
):

    factor = timedelta(minutes=60) / resolution
    if energy_to_power:
        output = values / factor

    if power_to_energy:
        output = values * factor

    return output


def get_difference_with_sign(first_value, second_value):

    if isna(first_value):
        # print("First value nan. Return nan")
        return nan

    if isna(second_value):
        # print("Second value nan. Return nan")
        return nan

    if sign(first_value) < 0 and sign(second_value) < 0:

        if first_value <= second_value:

            # Node A: Both signs negative 1-1

            # print('\nNode A: First value less/equal than second, both negative       -> POSITIVE SIGN: ')
            diff = -abs(-first_value - (-second_value))

        elif second_value < first_value:

            # Node A: Both signs negative 1-2

            # print('\nNode A: Second value less than first, both negative       -> NEGATIVE SIGN: ')
            "NOTE: Changed at 06.10.2019"
            diff = abs(-first_value - (-second_value))

    elif sign(first_value) > 0 and sign(second_value) > 0:

        if first_value < second_value:

            # Node C: Both signs positive 1-1

            # print('\nNode C: First value less than second, both positive -> NEGATIVE SIGN: ')
            "NOTE: Changed at 05.10.2019"
            diff = -abs(first_value - (second_value))

        elif second_value <= first_value:

            # Node C: Both signs positive 1-2
            # print('\nNode C: Second value less/equal than first, both positive -> POSITIVE SIGN: ')
            diff = abs(first_value - (second_value))

    elif first_value == 0 and second_value == 0:

        # Node E

        # print("Node E: Zero difference")
        diff = 0

    elif first_value == 0 or second_value == 0:

        if first_value < second_value:

            # Node D: One is zero, first one less 1-1 [Second value == 0]

            # print('\nNode D: First value less than second, one zero -> NEGATIVE SIGN: ')
            diff = -abs(first_value - (second_value))

        elif second_value < first_value:

            # Node D: One is zero, second one less 1-1 [First value == 0]

            # print('\nNode D: Second value less/equal  than first, one or two or both zero -> POSITIVE SIGN: ')
            diff = abs(first_value - (second_value))

    elif sign(first_value) != sign(second_value):

        if first_value < second_value:

            # Node B: Signs different 1-1

            # print('\nNode B: First value less than second, only first negative ->  NEGATIVE SIGN: ')
            diff = -abs(-first_value - (second_value))

        elif second_value < first_value:

            # Node B: Signs different 1-2

            # print('Node B: Second value less than first, only first positive -> POSITIVE SIGN: \n')
            diff = abs(first_value - (-second_value))
    else:
        print("\n Some error occured inside 'get_difference_with_sign' ")

    return diff


""" _____________________________________________________________ DECORATORS """

# Decorator
def remove_unused_levels(func):

    """Removes unused levels from multiindex dataframe"""

    def wrapper(*args, **kwargs):

        profile = func(*args, **kwargs)
        try:
            profile.columns = profile.columns.remove_unused_levels()
            profile.index = profile.index.remove_unused_levels()

        except:
            pass

        return profile

    return wrapper


# Decorator
def print_with_nan_columns(func):

    """Removes unused levels from multiindex dataframe"""

    def wrapper(*args, **kwargs):

        df = func(*args, **kwargs)

        # Remove columns with only nans and ensure result is a DF for slicing
        df = DataFrame(data=df[df.columns[~df.isnull().any()]])

        # TODO: Pass logger as argument
        logger.info("df: ", df)

        return df

    return wrapper


# Decorator
def call_counter(func):

    """Counts function calls in loops, utilizable as decorater"""

    @wraps(func)
    def helper(*args, **kwargs):

        helper.calls += 1
        return func(*args, **kwargs)

    helper.calls = 0
    helper.__name__ = func.__name__

    return helper
