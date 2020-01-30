from typing import List, Optional, Tuple, Union, Type, NamedTuple, TypeVar
from datetime import datetime
from collections import namedtuple
from copy import deepcopy
from numpy import *
from pandas import *
from pandas.core.common import flatten
from pandas import IndexSlice as IDX

from utils import remove_unused_levels, call_counter

import random

from logger import log as logger

Policy = TypeVar("Policy")

FlexOfferTuple = namedtuple(
    "FlexOfferTuple", "flexoffer_stack_index activation_periods"
)


class Aggregator:

    """
        A "Request" sent by the FDP is a delta relative to the baseline value.
        The aggregator splits this difference in parts (disaggregation) and sends them to the EMS.
        The EMS send back an offer (delta) back to Aggregator, who aggregates the offers from all EMS.
        The final offer send to the Aggregator is also a delta, not a profile. Same for the commitment.
        The fulfilled values in contrary relates to the power profile the Aggregator/EMS has been followed finally. 
    
    """

    def __init__(
        self,
        name: str,
        EMS_names: List[str],
        prediction_delta: Timedelta,
        baseline_fee: float,
    ):
        """
            prediction_delta: resolution * steps e.g. 4 * 00:15 = 1h = 60 min
        """
        self.name = name
        self.prediction_delta = prediction_delta
        self.deviation_costs = []
        self.baseline_revenues = 0
        self.baseline_fee = baseline_fee

        # EMS instances
        self.EMS = list()

        # EMS names
        self.EMS_names = EMS_names

    def __repr__(self):
        return str("AGGREGATOR ") + self.name

    @remove_unused_levels
    def sends(
        self,
        aggregated_baseline: bool = False,
        disaggregated_flexrequest: bool = False,
        aggregated_flexoffer: bool = False,
        flexorder: bool = False,
        commitment: bool = False,
    ):

        # ''' _____________________________________________________________  BASELINE '''
        if aggregated_baseline:

            profile = self.aggregates(baselines=True)

        # ''' _____________________________________________________________  REQUEST '''
        elif disaggregated_flexrequest:

            profile = self.disaggregates_flexrequest

        # ''' _____________________________________________________________  OFFER '''
        elif aggregated_flexoffer:

            # Aggregate EMS flexoffer values
            profile = self.aggregates(flexoffer=True)

            # Select flexibility for horizon steps based on adaptive strategy
            profile = self.flexoffer_strategy.select_action(profile=profile)

        # ''' _____________________________________________________________  ORDER '''
        elif flexorder:

            # NOTE: index_with_min_costs corresponds to the flexoffer stack of the EMS
            # Send activation periods from order to EMS
            return FlexOfferTuple(
                flexoffer_stack_index=self.disaggregation.index_with_min_costs[1],
                activation_periods=self.flexoffer_strategy.activation_periods,
            )

        # ''' _____________________________________________________________  COMMITMENT '''
        elif commitment:

            # Returns flexorder
            profile = self.data[self.name, "Order"].loc[self.prediction_horizon[0], :]

        return profile

    def receives(
        self,
        baseline_request: bool = None,
        baseline: DataFrame = None,
        flexrequest: DataFrame = None,
        flexoffer: DataFrame = None,
        flexorder: DataFrame = None,
        deviation_costs: DataFrame = None,
    ):

        if baseline_request:
            """ _____________________________________________________________  BASELINE REQUEST """

            self.baseline_revenues += self.baseline_fee

        elif baseline is not None:
            """ _____________________________________________________________  BASELINE """
            return self.stores(baseline=baseline)

        elif flexrequest is not None:
            """ _____________________________________________________________  REQUEST """
            return self.stores(flexrequest=flexrequest)

        elif flexoffer is not None:
            """ _____________________________________________________________  OFFER """
            return self.evaluates_flexoffer(flexoffer=flexoffer)

        elif flexorder is not None:
            """ _____________________________________________________________  ORDER """
            return self.stores(flexorder=flexorder)

        elif deviation_costs is not None:
            """ _____________________________________________________________  ORDER """
            return self.stores(deviation_costs=deviation_costs)

    def stores(
        self,
        baseline: DataFrame = None,
        flexrequest: DataFrame = None,
        flexoffer: DataFrame = None,
        best_ems_offers: DataFrame = None,
        flexorder: DataFrame = None,
        deviation_costs: float = None,
    ):

        # Get current datetimeindex
        datetime_index = self.prediction_horizon[0]

        # ''' _____________________________________________________________  BASELINE '''
        if baseline is not None:

            # Check which EMS sent the baseline
            ems = baseline.columns.get_level_values(0).unique().format()[0]

            # Store baseline data
            self.data.at[IDX[datetime_index, :], IDX[ems, "Baseline"]] = baseline.copy()

            logger.info("\nAggregator received {}".format(set(baseline)))

        # ''' _____________________________________________________________  REQUEST '''
        elif flexrequest is not None:

            # Select commitment data columns
            entries = ["Request", "Price_down", "Price_up", "Price"]

            # Store incoming flexrequest data at Aggregators summary data
            self.data.at[
                IDX[datetime_index, :], IDX[self.name, entries]
            ] = flexrequest.values

            # self.data

            # Erase flag
            self.disaggregation.status = ""

        # ''' _____________________________________________________________  OFFER '''
        elif flexoffer is not None:

            # Store offer values
            self.data.at[IDX[datetime_index, :], IDX[self.name, ["Offer"]]] = flexoffer[
                0
            ]

            # Store offer costs
            self.data.at[
                IDX[datetime_index, :], IDX[self.name, ["COSTS_Offer"]]
            ] = flexoffer[1]

            # Store offer costs
            self.data.at[
                IDX[datetime_index, :], IDX[self.name, ["COSTS_Deviation"]]
            ] = flexoffer[2]

        elif best_ems_offers is not None:

            for ems, best_offer in zip(self.EMS, best_ems_offers):

                self.data.at[
                    IDX[datetime_index, :],
                    IDX[ems.name, ["Offer", "COSTS_Offer", "COSTS_Deviation"]],
                ] = best_offer

        # ''' _____________________________________________________________  COMMITMENT '''
        elif flexorder is not None:

            # Get current datetimeindex
            datetime_index = self.prediction_horizon[0]

            self.data.at[
                IDX[datetime_index, :], IDX[self.name, ["Order", "REVENUES"]]
            ] = flexorder.loc[
                IDX[datetime_index, :], IDX[:, ["Offer", "COSTS_Order"]]
            ].values

        # ''' _____________________________________________________________  DEVIATION COSTS '''
        elif deviation_costs is not None:

            # Store received deviation costs from all EMS
            self.deviation_costs.append(deviation_costs)

            sum_deviation_costs = sum(self.deviation_costs)

            if isnan(sum_deviation_costs):
                sum_deviation_costs = 0

            # After the last EMS has sent its deviation costs, start updating
            if len(self.deviation_costs) == len(self.EMS):

                self.data.at[
                    IDX[datetime_index, 1], IDX[self.name, "COSTS_Deviation"]
                ] = sum_deviation_costs

                reward = self.data.loc[
                    IDX[datetime_index, :], IDX[self.name, "REVENUES"]
                ].sum(axis=0)

                if isna(reward):
                    reward = 0

                logger.info(" _________________________________________________")
                logger.info("|												  |")
                logger.info("|		   ---> AGGREGATOR UPDATES <---           |")
                logger.info("|_________________________________________________|\n")

                logger.info(
                    "Aggregator received order bid from FDP and stores it as reward: {}\n".format(
                        reward
                    )
                )

                self.flexoffer_strategy.update(
                    current_datetime=datetime_index, reward=round(reward, 2)
                )

                self.deviation_costs = []

        return

    def aggregates(
        self,
        baselines: bool = False,
        flexoffer: bool = False,
        commitments: bool = False,
    ):

        # Get current datetimeindex
        datetime_index = self.prediction_horizon[0]

        """ _____________________________________________________________  BASELINE """
        if baselines:

            # Create aggregated baseline profile for FDP
            aggregated_profile = (
                self.data.loc[
                    IDX[datetime_index, :],
                    IDX[[ems.name for ems in self.EMS], ["Baseline"]],
                ]
                .copy()
                .sum(level=1, axis=1)
            )

            # Update own database
            self.data.at[
                IDX[datetime_index, :], IDX[self.name, ["Baseline"]]
            ] = aggregated_profile.values

            logger.info("\nAggregator sends:\n{}".format(aggregated_profile))

        """ _____________________________________________________________  OFFER """
        if flexoffer:

            for ems in self.EMS:

                # Get index of disaggregation attempt with the lowest costs
                index_with_min_costs = self.disaggregation.index_with_min_costs

                # Update best attempt for each EMS in own database
                self.data.loc[
                    IDX[datetime_index, :],
                    IDX[ems.name, ["Offer", "COSTS_Offer", "COSTS_Deviation"]],
                ] = self.disaggregation.data.loc[index_with_min_costs, ems.name]

                logger.info(
                    "\n\t\t\t---> Best offers: \n\n{}".format(
                        self.disaggregation.data.loc[index_with_min_costs, ems.name]
                    )
                )

            applicable_offer_values_matrix = [
                # Each row vector consists of one value per Aggregator(first value) and one per EMS (could be multiple)
                where(
                    # Get only row vectors where at least one EMS value is not NaN (Aggregator is always NaN, yet)
                    self.data.loc[IDX[datetime_index, :], IDX[:, x]].isnull() != True,
                    # Sum together the row elements if, else set a nan value
                    self.data.loc[IDX[datetime_index, :], IDX[:, x]].sum(
                        level=1, axis=0
                    ),
                    nan,
                )
                # Apply the where() condition for the "Offers" and "COSTS_offer" columns of the flexoffer
                for x in ["Offer", "COSTS_Offer", "COSTS_Deviation"]
            ]

            # Sum over all entities per horizon step(= sum over row vectors)
            # and reshape to [2 x m] matrix, with one row vector for costs one for values,
            # where each element belongs to one of the horizon steps m
            applicable_offer_values_matrix = [
                # Make two lists, one for costs one for values
                list(
                    flatten(
                        # Take the sum with nums
                        [
                            nansum(vector[:], axis=0, keepdims=True)
                            # Only if not all row elements are NaN
                            if not isnan(vector[:]).all() else nan
                            for vector in matrix
                        ]
                    )
                )
                for matrix in applicable_offer_values_matrix
            ]

            # Store flexoffer values
            self.stores(flexoffer=applicable_offer_values_matrix)

            # Get aggregated profile with offer values
            aggregated_profile = self.data.loc[
                IDX[datetime_index, :], IDX[self.name, ["Offer", "COSTS_Deviation"]]
            ].copy()

            logger.info(
                "\nAggregator created aggregated: \n{}\n".format(aggregated_profile)
            )

        return aggregated_profile

    @property
    def disaggregates_flexrequest(self,):

        """
            Gets called per attempt and EMS 
        """
        # Get current datetimeindex
        datetime_index = self.prediction_horizon[0]

        # Select commitment data columns
        entries = ["Request", "Price_down", "Price_up"]

        # Get the current flexrequest from data
        flexrequest = self.data.loc[
            IDX[datetime_index, :], IDX[self.name, entries]
        ].copy()

        flexrequest_share = self.disaggregation.method(
            self=self.disaggregation,
            datetime_index=datetime_index,
            flexrequest=flexrequest,
            splits=len(self.EMS),
        )

        # Get current ems by evaluating the disaggregation method calls
        current_ems = self.EMS_names[self.disaggregation.method.calls - 1]

        # Store disaggregated flexrequest data at Aggregators EMS data
        self.data.at[
            IDX[datetime_index, :], IDX[current_ems, entries]
        ] = flexrequest_share.values

        return flexrequest_share

    def evaluates_flexoffer(self, flexoffer: DataFrame):

        """
            Gets called at received(). 
            Evaluates per EMS and attempt
        """
        # Get current datetimeindex
        datetime_index = self.prediction_horizon[0]

        # Get EMS name from flexoffer data
        ems_name = flexoffer.columns.get_level_values(0).unique().format()[0]

        # Get the current disaggregation attempt
        attempt = self.disaggregation.attempt

        # Store incoming flexoffer from EMS at disaggregation object
        self.disaggregation.data.at[
            IDX[datetime_index, attempt], ems_name
        ] = flexoffer.copy()

        # Sum of costs over flexoffer horizon
        sum_flexoffer_costs_over_horizon = flexoffer.loc[
            IDX[datetime_index, :], IDX[ems_name, "COSTS_Offer"]
        ].sum(axis=0)

        sum_flexoffer_costs_over_horizon += flexoffer.loc[
            IDX[datetime_index, :], IDX[ems_name, "COSTS_Deviation"]
        ].sum(axis=0)

        logger.info(
            "\n\t\t\t**** Aggregator evaluates {} flexoffer at attempt {}: Costs {}".format(
                ems_name, attempt, sum_flexoffer_costs_over_horizon
            )
        )

        # Store summed costs at current disaggregation attempt index
        self.disaggregation.costs.at[
            IDX[datetime_index, attempt], ems_name
        ] = sum_flexoffer_costs_over_horizon

        # print('self.disaggregation.data: ', self.disaggregation.data)
        logger.info(
            "\n\t\t\t**** Aggregator stores costs of disaggregation attempt {} and {}:\n\n{}\n".format(
                attempt,
                ems_name,
                self.disaggregation.costs.loc[IDX[datetime_index, :], :],
            )
        )

    @property
    def selects_best_flexoffer(self):

        # Increase attempt loop counter
        self.disaggregation.attempt += 1

        # Added +1 for scripting purpose (first attempt = 1, not 0)
        if self.disaggregation.attempt == self.disaggregation.max_attempts + 1:

            # Get current datetimeindex
            datetime_index = self.prediction_horizon[0]

            # Get min over the sum of costs per attempt
            costs_per_attempts = self.disaggregation.costs.loc[
                IDX[datetime_index, :], :
            ].sum(axis=1)

            # Get index of disaggregation attempt with the lowest costs
            self.disaggregation.index_with_min_costs = costs_per_attempts.loc[
                IDX[datetime_index, :], :
            ].idxmin(axis=1)

            # Get the EMS offers with the lowest costs
            best_ems_offers = self.disaggregation.data.loc[
                IDX[self.disaggregation.index_with_min_costs], :
            ]

            # Store best offer data of the EMS at Aggreagators summary data
            self.stores(best_ems_offers=best_ems_offers)

            # Set status
            self.disaggregation.status = "OPTIMAL"

            # TODO: Use method call counter?
            # Reset attempt counter variable
            self.disaggregation.attempt = 1

            # Reset method call counter
            self.disaggregation.method.calls = 0

            logger.info(
                "\n\t\t\t**** Aggregator selects attempt Nr. {} with the lowest costs.".format(
                    self.disaggregation.index_with_min_costs[1]
                )
            )

            logger.info(
                "\n\t\t\t---> Total costs over all EMS:\n\n{}\n".format(
                    # attempt,
                    costs_per_attempts
                )
            )


class Disaggregation:
    def __init__(
        self, simulation_time: DatetimeIndex, max_attempts: int, EMS_names: List,
    ):
        self.attempt = 1
        self.max_attempts = 1
        self.status = ""
        self.best_attempt_data = None
        self.index_with_min_costs = None
        self.simulation_time = simulation_time

        # Stores flexoffer objects per attempt and EMS
        self.data = DataFrame(
            index=MultiIndex.from_product(
                iterables=[
                    [time for time in simulation_time],
                    [attempt for attempt in list(range(1, self.max_attempts + 1))],
                ],
                names=["Time", "Attempt"],
            ),
            columns=EMS_names,
        )
        # Stores flexoffer costs per attempt and EMS
        self.costs = DataFrame(
            index=MultiIndex.from_product(
                iterables=[
                    [time for time in simulation_time],
                    [attempt for attempt in list(range(1, self.max_attempts + 1))],
                ],
                names=["Time", "Attempt"],
            ),
            columns=EMS_names,
        )

    @call_counter
    def equal_split(self, datetime_index, flexrequest, splits):
        flexrequest_label = flexrequest.columns.get_level_values(0).unique()[0]

        flexrequest_disaggregated = flexrequest.copy()

        share = flexrequest_disaggregated[flexrequest_label]["Request"] / splits

        # Store the shares of the flexrequest for each ems per attempt
        shares = [share for ems in self.data.columns]

        flexrequest_disaggregated.at[
            IDX[flexrequest.index], IDX[flexrequest_label, "Request"]
        ] = share

        logger.info(
            "\nAggregator sends disaggregated flexrequest: \n\n{}".format(
                flexrequest_disaggregated
            )
        )
        return flexrequest_disaggregated

    @staticmethod
    def random_split(datetime_index, flexrequest, splits):

        flexrequest_label = flexrequest.columns.get_level_values(0).unique()[0]

        flexrequest_disaggregated = flexrequest.copy()

        splits = np.random.random(splits)

        splits /= splits.sum()

        share = flexrequest_disaggregated[flexrequest_label]["Request"] / around(
            splits, 1
        )

        flexrequest_disaggregated.at[
            IDX[flexrequest.index], IDX[flexrequest_label, "Request"]
        ] = share

        logger.info(
            "\t\t\t\t**** Aggregator sends disaggregated flexrequest: \n{}".format(
                flexrequest_disaggregated
            )
        )
        return flexrequest_disaggregated
