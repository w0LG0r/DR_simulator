import os
from typing import List, Optional, Tuple, Union, Type, NamedTuple, TypeVar, Generic
import pickle
from copy import deepcopy
from datetime import datetime, timedelta
from pandas import *
from pandas import IndexSlice as IDX

from numpy import *

from utils import (
    timer,
    get_profile_data_from_excel,
    get_profile_data_from_pickle,
    create_commitment_stack,
    create_database,
    remove_scenario_file_handler,
    get_difference_with_sign,
)

from agents.Aggregator import Aggregator, Disaggregation
from agents.EMS import EnergyManagementSystem
from agents.FDP import FlexibilityDemandingParty

from logger import log as logger
from logger import create_file_handler


# TypeVars
_EMS = TypeVar("EMS")
_Aggregator = TypeVar("Aggregator")
_FDP = TypeVar("FDP")
_Commitment = TypeVar("Commitment")


class Scenario:
    def __init__(
        self,
        # GENERAL
        name: str,
        description: str,
        result_directory: str,
        excel_file_name: str,
        units: List,
        # TIME
        simulation_start: DatetimeIndex,
        simulation_end: DatetimeIndex,
        prediction_delta: timedelta,
        resolution: Timedelta,
        episodes: int,
        episode_duration: timedelta,
        number_of_EMS: int,
    ):

        logger.warning(
            "____________________________________________________________________________________________________________________"
        )
        logger.warning(
            "|                                                                                                                  |"
        )
        logger.warning(
            "|                                               SCENARIO INIT                                                      |"
        )
        logger.warning(
            "|__________________________________________________________________________________________________________________|"
        )

        # Variable for execution time measurement
        self.execution_start_time = datetime.now()

        """ _____________________________________________________________  ASSERTATIONS """

        assert episode_duration <= timedelta(
            hours=24
        ), "Epsiode duration needs to be between 2 and 24 hours"

        """ _____________________________________________________________  GENERAL """

        self.name = name
        self.description = description
        self.result_directory = result_directory
        self.units = units

        self.number_of_EMS = number_of_EMS

        # Stores EMS instances as a list
        self.EMS = list()

        # Create names for the energy management systems and store them as another list
        self.EMS_names = ["EMS_" + str(x) for x in range(1, number_of_EMS + 1)]

        """ _____________________________________________________________  LOGGER """

        logger.handlers = []
        create_file_handler(scenario=self)

        """ _____________________________________________________________  TIME """

        self.start = simulation_start
        self.end = simulation_end

        self.simulation_time = date_range(
            start=simulation_start,
            end=simulation_end,
            freq=resolution,
            closed="left",
            name="Time",
        )
        self.simulation_time
        self.resolution = resolution

        # Episode parameter
        self.episodes = episodes
        self.episode_duration = episode_duration

        # TODO: Check if simulation_step_index is still used?
        self.simulation_step_index = list(range(0, len(self.simulation_time)))
        self.total_simulation_steps = len(self.simulation_time)
        self.prediction_horizon_cut_off_steps = int(prediction_delta / resolution) - 1
        self.prediction_delta = prediction_delta

        if episode_duration >= timedelta(hours=24):
            self.episode_cut_off = prediction_delta + resolution
        else:
            self.episode_cut_off = resolution

        logger.warning(
            '\nName:\t\t\t\t\t{}\
            \nProcessed at:\t\t\t{}\
            \nDescription:\t\t\t"{}"\
            \nInput file:\t\t\t\t{}\
            \nResult directory:\t\t/{}\n\
            \nStart - End:\t\t\t{} - {}\
            \nSimulation time:\t\t{}\
            \nEpisode time:\t\t\t{}\
            \nTotal steps:\t\t\t{}\
            \nEpisode steps:\t\t\t{}\
            \nResolution:\t\t\t\t{}\
            \nPrediction delta:\t\t{}\
            \nNumber of EMS:\t\t\t{}\
            \nUnits:\t\t\t\t\t{}\n\
            '.format(
                name.capitalize(),
                datetime.now().strftime("%d %b %Y, %H:%M"),
                description,
                excel_file_name,
                result_directory,
                self.simulation_time[0].strftime("%d.%m.%Y %H:%M"),
                self.simulation_time[-1].strftime("%d.%m.%Y %H:%M"),
                to_timedelta(self.end - self.start),
                to_timedelta((self.start + self.episode_duration) - self.start),
                self.total_simulation_steps,
                ((self.start + self.episode_duration) - self.start) / self.resolution,
                resolution,
                prediction_delta,
                number_of_EMS,
                units,
            )
        )
        return

    def add_Aggregator(
        self,
        aggregator_name: str,
        prediction_delta_Aggregator: Timedelta,
        flexoffer_strategy: callable,
        baseline_fee: float,
    ):
        """ _____________________________________________________________  AGGREGATOR (AGG) """

        # Create an Aggregator agent
        self.Aggregator = Aggregator(
            name=aggregator_name,
            EMS_names=self.EMS_names,
            prediction_delta=prediction_delta_Aggregator,
            baseline_fee=baseline_fee,
        )

        # Add flexoffer strategy
        self.Aggregator.flexoffer_strategy = flexoffer_strategy

        self.Aggregator.data_storage = dict.fromkeys(range(self.episodes))

        logger.warning(
            ">>> {}:\t\t\tACTIVE\
            \n\tStrategy:\t\t\t{}".format(
                "Aggregator", type(self.Aggregator.flexoffer_strategy).__name__,
            )
        )

        """ _____________________________________________________________ FLEXOFFER STRATEGY (AGG) """

        # Assign a disaggregation object to the aggregator agent
        # TODO: Configure in scenario file and pass as argument
        self.Aggregator.disaggregation = Disaggregation(
            # NOTE: Changed from simulation time to applicable_simulation_time
            simulation_time=self.simulation_time,
            max_attempts=1,
            EMS_names=self.EMS_names,
        )

        # Select a disaggregation method
        self.Aggregator.disaggregation.method = Disaggregation.equal_split

        # Introduce the Aggregator agent
        logger.warning(
            "\tDisaggregation:\t\t{}\
            \n\n>>> {}:\t\t\t\tACTIVE".format(
                self.Aggregator.disaggregation.method.__name__, "FDP",
            )
        )

    def add_FDP(
        self,
        FDP_name: str,
        prediction_delta_FDP: Timedelta,
        randomize_flexrequest_prices: bool,
        randomize_flexrequest_values: bool,
    ):
        """ _____________________________________________________________ FLEXIBILITY DEMANDING PARTY (FDP) """

        # Create an FDP agent
        self.FDP = FlexibilityDemandingParty(
            name=FDP_name,
            simulation_time=self.simulation_time,
            prediction_delta=prediction_delta_FDP,
            randomize_flexrequest_prices=randomize_flexrequest_prices,
            randomize_flexrequest_values=randomize_flexrequest_values,
        )

        # Get FDP's imbalance data
        self.FDP.imbalances_data_storage = get_profile_data_from_pickle(
            simulation_time=self.simulation_time,
            result_directory=self.result_directory,
            entity="FDP",
        )

        self.FDP.imbalances_data_storage

        self.FDP.data_storage = dict.fromkeys(range(self.episodes))

    def add_EMS(
        self,
        power_purchase_tariff_per_EMS: List,
        power_feedin_tariff_per_EMS: List,
        flexibility_tariff_per_EMS: List,
    ):

        """ _____________________________________________________________ ENERGY MANAGEMENT SYSTEMS (EMS)"""

        # Setup the EMS instances
        for name, purchase_tariff, feedin_tariff, flexibility_tariff in zip(
            self.EMS_names,
            power_purchase_tariff_per_EMS,
            power_feedin_tariff_per_EMS,
            flexibility_tariff_per_EMS,
        ):

            # Initiate EMS
            EMS = EnergyManagementSystem(
                name=name,
                simulation_time=self.simulation_time,
                purchase_tariff=purchase_tariff,
                feedin_tariff=feedin_tariff,
                flexibility_tariff=flexibility_tariff,
            )

            logger.warning("\n>>> {}:\t\t\t\tACTIVE".format(EMS.name))
            logger.warning(
                "\tPurchase tariff:\t{}".format(power_purchase_tariff_per_EMS)
            )
            logger.warning(
                "\tFeed in tariff:\t\t{}".format(power_feedin_tariff_per_EMS)
            )
            logger.warning(
                "\tFlexibility tariff:\t{}\n".format(flexibility_tariff_per_EMS)
            )

            # Get profiles and setup devices
            (
                EMS.devices,
                EMS.device_constraints_storage,
                EMS.grid_constraint_storage,
            ) = get_profile_data_from_pickle(
                simulation_time=self.simulation_time,
                result_directory=self.result_directory,
                entity=name,
            )

            # EMS.device_constraints_storage
            EMS.data_storage = dict.fromkeys(range(self.episodes))

            # Backup initial storage constraints for plotting
            if "Storage" in EMS.devices:
                EMS.initial_storage_constraints = EMS.device_constraints_storage[
                    "Storage"
                ].copy()

            # Add EMS to the scenarios attributes
            self.EMS.append(EMS)

            # Add EMS to Aggregator attributes
            self.Aggregator.EMS.append(EMS)

            # Empty line
            logger.warning("")

        return

    @property
    def run(self):

        # Reassign agents to avoid self in the loop
        Aggregator = self.Aggregator
        FDP = self.FDP
        EMS = self.EMS

        # Create log file handlers
        logger.handlers = []
        create_file_handler(scenario=self)

        # print('\n----------- SCENARIO START: {} -----------'.format(timer()))
        for episode in range(0, self.episodes):

            # episode_time = date_range(
            #     start=self.start + timedelta(days=episode),
            #     end=self.start + timedelta(days=episode) + self.episode_duration - self.episode_cut_off,
            #     freq=self.resolution
            #     )

            episode_time = date_range(
                start=self.start + timedelta(days=episode),
                # @ flexibility test
                # end=self.start + timedelta(days=episode) + timedelta(hours=21),
                # @ q learning test
                end=self.start + timedelta(days=episode) + timedelta(hours=12),
                freq=self.resolution,
            )

            episode_time_steps = list(range(1, len(episode_time) + 1))

            full_day_time = date_range(
                start=self.start + timedelta(days=episode),
                end=self.start
                + timedelta(days=episode)
                + timedelta(days=1)
                - self.resolution,
                freq=self.resolution,
            )

            # Increase looping index
            episode += 1

            # Let the aggregator know the current episode
            Aggregator.flexoffer_strategy.current_episode = episode

            # Apply episode time as current simulation time
            Aggregator.simulation_time = episode_time

            # Create data base for Aggregator
            Aggregator.data = create_database(
                entity=self.Aggregator,
                prediction_delta=self.Aggregator.prediction_delta,
                simulation_time=full_day_time,
            )

            FDP.simulation_time = episode_time

            # Create FDP's data base
            FDP.data = create_database(
                entity=FDP,
                prediction_delta=FDP.prediction_delta,
                simulation_time=full_day_time,
            )

            FDP.imbalances_data = deepcopy(
                FDP.imbalances_data_storage.loc[full_day_time[0] : full_day_time[-1], :]
            )

            # FDP slices imbalances into requests
            self.FDP.initializes_flex_requests

            for ems in EMS:

                ems.simulation_time = episode_time

                # Create data base for whole simulation time
                ems.data = create_database(
                    entity=ems,
                    prediction_delta=self.Aggregator.prediction_delta,
                    simulation_time=full_day_time,
                )

                # Create commitment stack for a whole day
                ems.commitment_stack = create_commitment_stack(
                    entity=ems,
                    prediction_delta=self.prediction_delta,
                    simulation_time=full_day_time,
                )

                ems.device_constraints = dict.fromkeys(
                    ems.device_constraints_storage.keys()
                )

                for key, df in ems.device_constraints_storage.items():
                    ems.device_constraints[key] = deepcopy(
                        df.loc[full_day_time[0] : full_day_time[-1], :]
                    )

                ems.grid_constraint = deepcopy(
                    ems.grid_constraint_storage.loc[
                        full_day_time[0] : full_day_time[-1], :
                    ]
                )

                # Set data for energy contract
                ems.sets_energy_contract_data

            logger.warning(
                "____________________________________________________________________________________________________________________"
            )
            logger.warning(
                "|                                                                                                                  |"
            )
            logger.warning(
                "                                 SCENARIO: {}                                                                      ".format(
                    self.name
                )
            )
            logger.warning(
                "                                 EPISODE:  {}                                                                      ".format(
                    episode
                )
            )
            logger.warning(
                "|__________________________________________________________________________________________________________________|"
            )

            for index, ix in zip(episode_time, episode_time_steps):

                display_info = (
                    Aggregator.flexoffer_strategy.current_episode,  # CURRENT EPISODE
                    ix,  # CURRENT STEP
                    episode_time_steps[-1],  # TOTAL STEPS
                    index,  # SIMULATION INDEX
                    timer(self.execution_start_time),  # TIMER
                )
                logger.info(
                    "____________________________________________________________________________________________________________________"
                )
                logger.info(
                    "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                )
                logger.info(
                    "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                )
                logger.info(
                    "|                                                START STEP {}                                                      |".format(
                        ix
                    )
                )
                logger.info(
                    "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                )
                logger.info(
                    "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                )
                logger.warning(
                    "EPISODE {}				STEP {} OF {}				  INDEX: {}				TIME: {}".format(
                        *display_info
                    )
                )

                logger.info("\nAggregator requests baseline profiles from EMS")
                logger.info("\nEMS starts unit dispatch\n")

                # Setup the current prediction windows
                Aggregator.prediction_horizon = date_range(
                    start=index,
                    end=index + Aggregator.prediction_delta,
                    freq=self.resolution,
                    closed="left",
                    name="Prediction Aggregator",
                )

                FDP.prediction_horizon = date_range(
                    start=index,
                    end=index + FDP.prediction_delta,
                    freq=self.resolution,
                    closed="left",
                    name="Prediction FDP",
                )

                # FDP pays fee for baseline
                Aggregator.receives(baseline_request=FDP.sends(baseline_request=True))

                # EMS provides baselines for aggregators prediction horizon
                for ems in EMS:
                    logger.info(
                        "===================================================================================================================="
                    )
                    logger.info(
                        "EPISODE {}				STEP {} OF {}				  INDEX: {}				TIME: {}".format(
                            *display_info
                        )
                    )
                    logger.info(
                        "UC BASELINE {}                                                                                  ".format(
                            ems.name
                        )
                    )
                    logger.info(
                        "===================================================================================================================="
                    )

                    ems.prediction_horizon = Aggregator.prediction_horizon

                    # EMS sorts out the applicable commitments from the last rounds
                    ems.selects_applicable_commitments

                    # EMS calls device scheduler to perfom a unit commitment
                    ems.solves_unit_commitment(baseline=True)

                    # EMS sends baseline to Aggregator, who also updates his baseline data
                    Aggregator.receives(baseline=ems.sends(baseline=True))

                logger.info(
                    "===================================================================================================================="
                )
                logger.info(
                    "EPISODE {}				STEP {} OF {}				  INDEX: {}				TIME: {}".format(
                        *display_info
                    )
                )
                logger.info(
                    "FLEX REQUEST                                                                                                      "
                )
                logger.info(
                    "===================================================================================================================="
                )

                # Aggregator sends aggregated baseline profiles from all EMS to the FDP in return for fee
                FDP.receives(baseline=Aggregator.sends(aggregated_baseline=True))

                # FDP selects a FlexRequest and sends it to the Aggregator, who updates his data storage
                Aggregator.receives(flexrequest=FDP.sends(flexrequest=True))

                while not "OPTIMAL" in Aggregator.disaggregation.status:

                    # TODO: First split, than loop over splits, and let EMS do one UC each, then send to aggregator, then select best attempt
                    for ems in EMS:
                        logger.info(
                            "======================================================================================================================"
                        )
                        logger.info(
                            "EPISODE {}				STEP {} OF {}				  INDEX: {}				TIME: {}".format(
                                *display_info
                            )
                        )
                        logger.info(
                            "UC FLEXOFFER {} - ATTEMPT {}                                  ".format(
                                ems.name, Aggregator.disaggregation.attempt
                            )
                        )
                        logger.info(
                            "======================================================================================================================"
                        )

                        ems.receives(
                            flexrequest=Aggregator.sends(disaggregated_flexrequest=True)
                        )

                        ems.selects_applicable_commitments

                        ems.solves_unit_commitment(flexoffer=True)

                        logger.info(
                            "======================================================================================================================"
                        )
                        logger.info(
                            "EPISODE {}				STEP {} OF {}				  INDEX: {}				TIME: {}".format(
                                *display_info
                            )
                        )
                        logger.info(
                            "FLEXOFFER EVALUATION {} - ATTEMPT {}                          ".format(
                                ems.name, Aggregator.disaggregation.attempt
                            )
                        )
                        logger.info(
                            "======================================================================================================================"
                        )

                        # EMS sends flexoffer to Aggregator
                        Aggregator.receives(flexoffer=ems.sends(flexoffer=True))

                    Aggregator.selects_best_flexoffer

                logger.info(
                    "======================================================================================================================"
                )
                logger.info(
                    "EPISODE {}				STEP {} OF {}				  INDEX: {}				TIME: {}".format(
                        *display_info
                    )
                )
                logger.info("FLEXOFFER STRATEGY")
                logger.info(
                    "======================================================================================================================"
                )

                # Aggregator selects offer values based on offer selection strategy and sends it to the FDP
                FDP.receives(flexoffer=Aggregator.sends(aggregated_flexoffer=True))

                # Aggregator gets flexorder with flexbility attached
                Aggregator.receives(flexorder=FDP.sends(flexorder=True))

                for ems in EMS:
                    ems.receives(flexorder=Aggregator.sends(flexorder=True))
                    Aggregator.receives(deviation_costs=ems.sends(deviation_costs=True))

                FDP.receives(commitment=Aggregator.sends(commitment=True))

                if ix == episode_time_steps[-1]:

                    # Delete rewards and performed action table entries
                    Aggregator.flexoffer_strategy.reset

                    Aggregator.data_storage[episode] = Aggregator.data

                    FDP.data_storage[episode] = FDP.data

                    for ems in EMS:
                        ems.data_storage[episode] = ems.data

                    logger.warning(
                        "____________________________________________________________________________________________________________________"
                    )
                    logger.info(
                        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                    )
                    logger.info(
                        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                    )
                    logger.info(
                        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                    )
                    logger.info(
                        "|                                              END OF EPISODE {}                                                     |".format(
                            episode
                        )
                    )
                    logger.info(
                        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                    )
                    logger.info(
                        "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
                    )

            if episode == self.episodes:
                # if episode == 1:
                # TODO: Catch error when ix == 1

                self.last_simulated_step = ix
                self.last_simulated_index = index
                self.postprocesses_data()
                self.logs_data()
                self.dumps()
                self.execution_duration = timer(self.execution_start_time)

                logger.warning(
                    "____________________________________________________________________________________________________________________"
                )
                logger.warning(
                    "|                                                                                                                  |"
                )
                logger.warning(
                    "|                               SIMULATION FINISHED AFTER: {}                                         |".format(
                        self.execution_duration
                    )
                )
                logger.warning(
                    "|__________________________________________________________________________________________________________________|"
                )

                return

    def postprocesses_data(self):

        """
            NOTE: Request, Offer, Order, Deviation are power deltas
                Baseline, Flexprofile, Commitment and Realised are power profiles

            Aggregator: 
                        Stores Request, Offer, Order, Baseline and all Costs on the fly. 
                        Commitment, Realised, Deviation gets computed afterwards
                        Costs_Offer are the sum of order costs of the attached EMS.
                        Costs_Offer_neg need to be inserted after negotiation. 

            FDP: 
                        Stores Request, Offer, Order, Baseline, Commitment on the fly.
                        Realised and Deviation gets computed afterwards (copied from Aggregator)
                        Costs_Offer are the imbalance market prices times the committed flexibility.
                        Costs_Offer_neg need to be inserted after negotiation. 

            EMS: 
                        Stores Request, Offer, Order, Baseline, Flexprofile on the fly at summary.
                        Profile and Deviation for each commitment gets computed on the fly. 
                        Commitments and deviation at summary gets computed afterwards. 
                        Commitments uses the Flexprofile values.
                        Devation uses get_difference_with_sign and compares commiment and realised values
                        Costs_Offer are the imbalance market prices times the committed flexibility.
                        Costs_Offer_neg need to be inserted after negotiation. 

        """

        Aggregator = self.Aggregator
        FDP = self.FDP

        realised_values = Series(index=self.simulation_time, data=0)
        deviation_values = Series(index=self.simulation_time, data=0)
        commitment_values = Series(index=self.simulation_time, data=0)

        # a = self.Aggregator.data[self.Aggregator.name]
        # print('a: ', a.head())
        for episode in range(1, self.episodes + 1):

            episode_time = date_range(
                start=self.start + timedelta(days=episode),
                end=self.start
                + timedelta(days=episode)
                + self.episode_duration
                - self.episode_cut_off,
                freq=self.resolution,
            )

            Aggregator.data = Aggregator.data_storage[episode]
            FDP.data = FDP.data_storage[episode]

            for ems in self.EMS:

                ems.data = ems.data_storage[episode]

                Aggregator.data[ems.name, "Order"] = ems.data[ems.name, "Order"]
                Aggregator.data[ems.name, "Realised"] = ems.data[ems.name, "Realised"]
                Aggregator.data[ems.name, "Commitment"] = ems.data[
                    ems.name, "Commitment"
                ]
                Aggregator.data[ems.name, "Deviation"] = ems.data[ems.name, "Deviation"]
                # Aggregator.data[ems.name, "Deviation"] = ems.data[ems.name, "COSTS_Offer"]
                Aggregator.data[ems.name, "COSTS_Order"] = ems.data[
                    ems.name, "COSTS_Order"
                ]
                # print('e: ', e)

                ems.data_storage[episode] = ems.data

            Aggregator.data[Aggregator.name, "Commitment"].fillna(0, inplace=True)
            # Aggregator.data[Aggregator.name, "Commitment"].fillna(0)

            Aggregator.data[Aggregator.name, "Commitment"] = Aggregator.data.loc[
                IDX[:, :], IDX[[ems for ems in self.EMS_names], "Commitment"]
            ].sum(axis=1, min_count=1)

            """ _____________________________________________________________  AGG SUMMARY REALISED """

            """ _____________________________________________________________  AGG SUMMARY DEVIATION """
            # Aggregator.data[Aggregator.name, "COSTS_Order"].fillna(0, inplace=True)

            Aggregator.data[Aggregator.name, "COSTS_Order"] = Aggregator.data.loc[
                IDX[:, :], IDX[[ems for ems in self.EMS_names], "COSTS_Order"]
            ].sum(axis=1, min_count=1)

            Aggregator.data[Aggregator.name, "Deviation"].fillna(0, inplace=True)

            Aggregator.data[Aggregator.name, "Deviation"] = Aggregator.data.loc[
                IDX[:, :], IDX[[ems for ems in self.EMS_names], "Deviation"]
            ].sum(axis=1, min_count=1)

            Aggregator.data[Aggregator.name, "Realised"].fillna(0, inplace=True)

            Aggregator.data[Aggregator.name, "Realised"] = Aggregator.data.loc[
                IDX[:, :], IDX[[ems for ems in self.EMS_names], "Realised"]
            ].sum(axis=1, min_count=1)

            """ _____________________________________________________________  FDP SUMMARY REALISATION """

            FDP.data[FDP.name]["Realised"] = Aggregator.data[Aggregator.name][
                "Realised"
            ]
            FDP.data[FDP.name]["Deviation"] = Aggregator.data[Aggregator.name][
                "Deviation"
            ]
            # self.FDP.data[self.FDP.name]["Commitment"] = Aggregator.data[Aggregator.name]["Commitment"]

            # Write postprocessed data back to storage dict
            Aggregator.data_storage[episode] = Aggregator.data
            FDP.data_storage[episode] = FDP.data

        # Log postprocessing completed
        logger.warning("\n\t\t\t\t\t\t{} postprocessed.".format(self.name.capitalize()))

        return

    def logs_data(self, horizon: DatetimeIndex = None):

        """ Could be used with pickled self"""

        if horizon is None:
            horizon = self.simulation_time

        Aggregator = self.Aggregator
        FDP = self.FDP

        for episode in range(1, self.episodes + 1):

            Aggregator.data = Aggregator.data_storage[episode]
            FDP.data = FDP.data_storage[episode]

            summary_filename = bytes(
                self.result_directory
                + "/DATA/"
                + self.name
                + "_"
                + str(episode)
                + ".txt",
                encoding="utf-8",
            )

            # ems_commitments_filename=bytes(self.result_directory + "/"  + self.name + "_COMMITMENTS_" + ".txt", encoding='utf-8')

            with open(summary_filename, "w") as text_file:
                # text_file.write("Purchase Amount: {0}".format(TotalAmount))

                # Aggregator summary
                text_file.write(
                    "\nSummary Aggregator: \n{}".format(
                        Aggregator.data[Aggregator.name]
                    )
                )
                # text_file.write('\nAggregator.data EMS 1: \n{}'.format(self.Aggregator.data["EMS_1"]))
                # FDP summary
                text_file.write("\nSummary FDP: \n{}".format(FDP.data))

                for ems in self.EMS:

                    ems.data = ems.data_storage[episode]

                    # EMS summary
                    text_file.write(
                        "\nSummary {}: \n{}".format(
                            ems.name, ems.data.loc[IDX[:, :], IDX[ems.name, :]]
                        )
                    )

        aggregator_full_data = list(Aggregator.data_storage.values())
        Aggregator.data = concat(aggregator_full_data)

        fdp_full_data = list(FDP.data_storage.values())
        FDP.data = concat(fdp_full_data)

        FDP.imbalances_data = FDP.imbalances_data_storage

        for ems in self.EMS:

            ems_full_data = list(ems.data_storage.values())
            ems.data = concat(ems_full_data)

            ems_commitments_filename = bytes(
                self.result_directory + "/DATA/" + ems.name + ".txt", encoding="utf-8"
            )

            with open(ems_commitments_filename, "w") as text_file:

                for ems in self.EMS:

                    # EMS commitments
                    for (
                        commitment_name,
                        commitment_data,
                    ) in ems.commitment_stack.items():
                        # zip(
                        # ems.data.columns.levels[0][1:-1],
                        # horizon[:-self.last_simulated_step]
                        # ):
                        # text_file.write('\n{} {}'.format(ems.name, commitment))

                        text_file.write(
                            "\n{}: \n{}".format(
                                commitment_name,
                                commitment_data.loc[IDX[:, :], IDX[:, :]],
                            )
                        )

            # Print logging data successfully completed
        logger.warning("\t\t\t\t\t\t{} data logged.".format(self.name.capitalize()))

    # @property
    def dumps(self):

        # Print dumping self successfully completed
        logger.warning("\t\t\t\t\t\t{} dumped.".format(self.name.capitalize()))

        with open(
            self.result_directory + "/" + self.name + "_SCENARIO.p", "wb"
        ) as file:
            pickle.dump(self, file)

        with open(
            self.result_directory + "/STRATEGY/" + self.name + "_STRATEGY.p", "wb"
        ) as file:
            pickle.dump(self.Aggregator.flexoffer_strategy, file)

        try:
            with open(
                self.result_directory + "/STRATEGY/" + self.name + "_LAST_Q_TABLE.p",
                "wb",
            ) as file:
                pickle.dump(self.Aggregator.flexoffer_strategy.q_values, file)
        except:
            pass

    # def set_episode_time(self, start, duration: timedelta):

    #     self.episode_time = date_range(start=start, end=start+duration, freq=self.resolution)
    #     self.episode_time_steps = list(range(1,len(self.episode_time)+1))

    #     return
