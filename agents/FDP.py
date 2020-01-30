from typing import List, Optional, Tuple, Union, Type, NamedTuple, TypeVar
from datetime import datetime
from numpy import *
from pandas import *
from pandas import IndexSlice as IDX

import random
from utils import remove_unused_levels

from logger import log as logger


class FlexibilityDemandingParty:
    def __init__(
        self,
        name: str,
        prediction_delta: Timedelta,
        simulation_time: DatetimeIndex,
        randomize_flexrequest_prices: bool,
        randomize_flexrequest_values: bool,
    ):

        self.name = name
        self.prediction_delta = prediction_delta
        self.simulation_time = simulation_time
        self.randomize_flexrequest_prices = randomize_flexrequest_prices
        self.randomize_flexrequest_values = randomize_flexrequest_values
        return

    def __repr__(self):
        return str("FDP") + self.name

    @remove_unused_levels
    def sends(
        self,
        baseline_request: bool = False,
        flexrequest: bool = False,
        flexorder: bool = False,
    ):

        datetime_index = self.prediction_horizon[0]
        d = self.simulation_time.get_loc(datetime_index)

        # ''' _____________________________________________________________  REQUEST '''

        if baseline_request:
            return True

        # ''' _____________________________________________________________  REQUEST '''
        if flexrequest:

            entries = ["Request", "Price_down", "Price_up"]

            profile = self.data.loc[
                IDX[datetime_index, :], IDX[self.name, entries]
            ].copy()

            profile.at[IDX[datetime_index, :], IDX[self.name, "Price"]] = self.data.loc[
                IDX[datetime_index, :], IDX[self.name, "Market_prices"]
            ]

            if self.randomize_flexrequest_values:

                if d == 6:
                    # NOTE Randomizing prices
                    profile.at[IDX[datetime_index, 1], IDX[self.name, "Request"]] = -8

                    profile.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_down"]
                    ] = -100

                    profile.at[IDX[datetime_index, 1], IDX[self.name, "Price_up"]] = 100

                    self.data.at[IDX[datetime_index, 1], IDX[self.name, "Request"]] = -8

                    self.data.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_down"]
                    ] = -100

                    self.data.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_up"]
                    ] = 100

                if d == 10:
                    # NOTE Randomizing prices
                    profile.at[IDX[datetime_index, 1], IDX[self.name, "Request"]] = -8

                    profile.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_down"]
                    ] = -100

                    profile.at[IDX[datetime_index, 1], IDX[self.name, "Price_up"]] = 100

                    self.data.at[IDX[datetime_index, 1], IDX[self.name, "Request"]] = -8

                    self.data.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_down"]
                    ] = -100

                    self.data.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_up"]
                    ] = 100


                if d == 14:
                    # NOTE Randomizing prices
                    profile.at[IDX[datetime_index, 1], IDX[self.name, "Request"]] = -2

                    profile.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_down"]
                    ] = -100

                    profile.at[IDX[datetime_index, 1], IDX[self.name, "Price_up"]] = 100

                    self.data.at[IDX[datetime_index, 1], IDX[self.name, "Request"]] = -2

                    self.data.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_down"]
                    ] = -100

                    self.data.at[
                        IDX[datetime_index, 1], IDX[self.name, "Price_up"]
                    ] = 100


            if self.randomize_flexrequest_prices:
                # NOTE Randomizing prices
                profile[self.name]["Price_down"] *= random.uniform(0, 10)
                profile[self.name]["Price_up"] *= random.uniform(0, 15)

        # ''' _____________________________________________________________  ORDER '''
        if flexorder:

            # Compute offer costs at market prices
            p = self.data.loc[IDX[datetime_index, 1], IDX[self.name, "Market_prices"]]

            order_costs = (
                self.data.loc[IDX[datetime_index, :], IDX[self.name, "Offer"]]
                .fillna(0)
                .abs()
                * self.data.loc[IDX[datetime_index, 1], IDX[self.name, "Market_prices"]]
            )

            self.data.at[
                IDX[datetime_index, :], IDX[self.name, "COSTS_Order"]
            ] = order_costs

            # Send offer as order
            profile = self.data.loc[
                IDX[datetime_index, :], IDX[self.name, ["Offer", "COSTS_Order"]]
            ]

            # Store profile as order
            self.data.at[IDX[datetime_index, :], IDX[self.name, "Order"]] = profile[
                self.name
            ]["Offer"].values

        logger.info("\nFDP sends:\n {}\n".format(profile))

        return profile

    def receives(
        self,
        baseline: DataFrame = None,
        flexoffer: DataFrame = None,
        commitment: DataFrame = None,
    ):

        datetime_index = self.prediction_horizon[0]

        """ _____________________________________________________________  BASELINE """
        if baseline is not None:

            # Store baseline data, which comes as series, use values
            self.data.at[
                IDX[datetime_index, :], IDX[self.name, "Baseline"]
            ] = baseline.values

            logger.info("\nFDP received: \n{}".format(baseline))

            """ _____________________________________________________________  OFFER """

        elif flexoffer is not None:

            # Store flexoffer data
            self.data.at[
                IDX[datetime_index, :], IDX[self.name, ["Offer", "COSTS_Deviation"]]
            ] = flexoffer.values

            logger.info("\nFDP received: \n{}".format(flexoffer))

            """ _____________________________________________________________  COMMITMENT """
        elif commitment is not None:

            logger.info(" _________________________________________________")
            logger.info("|												  |")
            logger.info("|		      ---> FDP UPDATES <---               |")
            logger.info("|_________________________________________________|\n")

            # Store commitment from Aggregator
            self.data.at[IDX[datetime_index, :], IDX[self.name, "Commitment"]] = (
                self.data.loc[IDX[datetime_index, :], IDX[self.name, "Baseline"]].values
                + commitment.fillna(0).values
            )

            # Update request for next round
            current_request = self.data.loc[
                IDX[datetime_index, :], IDX[self.name, "Request"]
            ].copy()

            # Subtract the commited valus from the requested ones in order to not request them again in the next horizon
            updated_request = around(
                current_request.values - commitment.fillna(0).values, 2
            )

            # Remove current horizon step
            updated_request = delete(updated_request, [0])

            # Compute simulation time resolution
            resolution = self.simulation_time.freq

            # Update request (horizon step index j+1)
            self.data.loc[
                IDX[datetime_index + resolution, 1 : len(updated_request)],
                IDX[self.name, "Request"],
            ] = updated_request

            logger.info(
                "\nFDP stores commitment and updates request to: \n{}\n".format(
                    self.data[self.name]["Request"].loc[datetime_index + resolution, :]
                )
            )

        return

    @property
    def initializes_flex_requests(self,):

        """
        NOTE: Adds self.flex_request_objects
        NOTE: Imbalances comes as flexibility in watt from the excel profile, 
        """
        # Loop variables
        index = self.simulation_time[0]
        ix = 0
        resolution = self.simulation_time.freq

        # # Cutoff last prediction horizon from simulation time
        applicable_simulation_time = (
            self.simulation_time - self.prediction_delta
        )  # + resolution

        # Loop over applicable index range and slice imbalance values from data
        while index <= applicable_simulation_time[-1]:

            # Start and end of added flex request object
            starts_at = index
            ends_at = index + self.prediction_delta - resolution

            # Create label to index flex requests
            label = "FDP"

            data = self.imbalances_data.loc[
                starts_at:ends_at,
                [
                    "imbalances",
                    "deviation_price_down",
                    "deviation_price_up",
                    # "market_prices"
                ],
            ]

            self.data.at[
                IDX[index, :], IDX[label, ["Request", "Price_down", "Price_up"]]
            ] = data.values

            # # Power values of added flex request object
            market_prices = self.imbalances_data.loc[starts_at, "market_prices"]

            self.data.at[IDX[index, :], IDX[label, "Market_prices"]] = market_prices

            index += resolution
            ix += 1

        return

    def displays_flexrequests(
        self,
        datetime_indices=None,
        flexrequest_indices: Union[List, DatetimeIndex] = None,
        flexrequest_data=None,
        all_request: bool = False,
    ):

        """
        Arguments examples:
            
            'Requests 3-6 at time 3-6, two data columns
            datetime_indices=simulation_time[3:6],
            flexrequest_indices=list(range(3,6)),
            flexrequest_data=["Requested", "Price_down"],
        
            'Requests at time 3: Slice until next step(=4) in list(range(slice_1, slice_2))
            
            datetime_indices=simulation_time[3],
            flexrequest_indices=list(range(3,4)),
            flexrequest_data=["Requested", "Price_down"],
        """

        # Check if last index is greater than last applicable time index
        try:
            applicable_simulation_time = self.simulation_time - self.prediction_delta

            if len(datetime_indices) > 1:
                assert (
                    datetime_indices[-1] <= applicable_simulation_time
                ), AttributeError(
                    "Last index {} greater than end of applicable simulation time {}".format(
                        datetime_indices[-1], applicable_simulation_time
                    )
                )
        except:
            assert datetime_indices <= applicable_simulation_time, AttributeError(
                "Index {} greater than end of applicable simulation time {}".format(
                    datetime_indices, applicable_simulation_time
                )
            )

        if flexrequest_indices:
            flexrequest_labels = []

            for index in flexrequest_indices:
                label = "FDP_FR_" + str(index)
                flexrequest_labels.append(label)

        # 1) All inputs available
        if all(
            v is not None
            for v in [datetime_indices, flexrequest_indices, flexrequest_data]
        ):
            logger.info("1) All inputs available")

            request = self.commitment_data.loc[
                IDX[datetime_indices, :], IDX[flexrequest_labels, flexrequest_data]
            ]
            print("request: ", request)
            return

        # 2) No inputs at all, shows complete flexrequest data
        elif all(
            v is None for v in [datetime_indices, flexrequest_indices, flexrequest_data]
        ):
            print("2) No inputs at all, shows complete flexrequest data")
            request = self.commitment_data.loc[IDX[:, :], IDX[:, :]]
            print("request: ", request)
            return

        # 3) datetime_indices input only
        elif all(v is None for v in [flexrequest_indices, flexrequest_data]):
            print("3) datetime_indices input only")
            request = self.commitment_data.loc[IDX[datetime_indices, :], IDX[:, :]]
            print("request: ", request)
            return

        # 4) flexrequest_indices input only
        elif all(v is None for v in [datetime_indices, flexrequest_data]):
            print("4) flexrequest_indices input only")

            request = self.commitment_data.loc[IDX[:, :], IDX[flexrequest_labels, :]]

            print("request: ", request)
            return

        # 5) flexrequest_data input only
        elif all(v is None for v in [datetime_indices, flexrequest_indices]):
            print("5) flexrequest_data input only")

            request = self.commitment_data.loc[IDX[:, :], IDX[:, flexrequest_data]]

            print("request: ", request)
            return

        # 6) flexrequest_indices and flexrequest_data
        elif all(v is None for v in [datetime_indices]):
            print("6) flexrequest_indices and flexrequest_data input only")

            request = self.commitment_data.loc[
                IDX[:, :], IDX[flexrequest_labels, flexrequest_data]
            ]

            print("request: ", request)
            return

        # 7) datetime_indices and flexrequest_indices
        elif all(v is None for v in [flexrequest_indices]):
            print("7) datetime_indices and flexrequest_indices input only")

            request = self.commitment_data.loc[
                IDX[datetime_indices, :], IDX[:, flexrequest_data]
            ]

            print("request: ", request)
            return

        # 8) datetime_indices and flexrequest_data
        elif all(v is None for v in [flexrequest_data]):
            print("8) datetime_indices and flexrequest_data input only")

            request = self.commitment_data.loc[
                IDX[datetime_indices, :], IDX[flexrequest_labels, :]
            ]

            print("request: ", request)
            return

        return
