from typing import List, Optional, Tuple, Union, Type, NamedTuple, TypeVar
import datetime as dt
from datetime import timedelta
from copy import deepcopy
from numpy import *
from numpy import sign
from pandas import *
from pandas import IndexSlice as IDX

from utils import remove_unused_levels, get_difference_with_sign

from solver import device_scheduler

from logger import log as logger

import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=RuntimeWarning)


class EnergyManagementSystem:
    def __init__(
        self,
        name: str,
        simulation_time: DatetimeIndex,
        purchase_tariff: float,
        feedin_tariff: float,
        flexibility_tariff: float,
    ):

        self.name = name
        self.simulation_time = simulation_time
        self.purchase_tariff = purchase_tariff
        self.feedin_tariff = feedin_tariff
        self.flexibility_tariff = flexibility_tariff

    def __repr__(self):
        return self.name

    def solves_unit_commitment(self, baseline: bool = False, flexoffer: bool = False):

        if baseline:

            baseline_flag = True

        else:

            baseline_flag = False

        device_scheduler(
            EMS_name=self.name,
            optimization_horizon=self.prediction_horizon,
            device_constraints=self.device_constraints,
            grid_constraint=self.grid_constraint,
            applicable_commitments=self.applicable_commitments,
            EMS_data=self.data,
            baseline=baseline_flag,
        )

        return

    @property
    def selects_applicable_commitments(self,):

        horizon = self.prediction_horizon

        first_horizon_step = self.simulation_time.get_loc(horizon[0])

        try:
            # LHS: Get horizon steps end as integer position of total simulation runtime
            last_horizon_step = self.simulation_time.get_loc(horizon[-1]) + 1
        except:
            extended_simulation_time = (
                self.simulation_time + (horizon[-1] - horizon[0]) + horizon.freq
            )
            last_horizon_step = first_horizon_step + len(horizon)

        # FACIP: Compute position index of first horizon step in dataindex of total simulation
        first_applicable_commitment_index_position = (
            first_horizon_step - last_horizon_step
        )

        # FACSI: Add FHS to FACIP
        first_applicable_commitment_step_index = (
            first_applicable_commitment_index_position + first_horizon_step
        )

        # Node A
        if first_applicable_commitment_step_index < 0:

            first_applicable_commitment_step_index = 0

            FACSI_added_indices = 1

        # Node B
        else:

            FACSI_added_indices = 2


        self.indices_applicable_commitments = list(
            range(
                first_applicable_commitment_step_index + FACSI_added_indices,
                first_horizon_step + 2,
            )
        )

        self.applicable_commitments = [
            self.commitment_stack[x] for x in self.indices_applicable_commitments
        ]

        # Insert the EC commitment df
        self.applicable_commitments.insert(0, self.commitment_stack[0])

        return

    # @call_counter
    def appends_flexoffer_to_stack(self, flexoffer):

        # if stack exists append flexoffer, else create one
        if "flexoffer_stack" in locals():
            self.flexoffer_stack.append(flexoffer)

        else:
            self.flexoffer_stack = []
            self.flexoffer_stack.append(flexoffer)

        return

    def receives(
        self, flexrequest: DataFrame = None, flexorder: DataFrame = None,
    ):

        if flexrequest is not None:

            self.stores(flexrequest=flexrequest)

        if flexorder is not None:

            self.stores(flexorder=flexorder)

        return

    @remove_unused_levels
    def sends(
        self,
        baseline: bool = False,
        flexrequest: DataFrame = None,
        flexoffer: bool = False,
        deviation_costs: bool = False,
    ):

        # # Get current datetime index

        # ''' _____________________________________________________________  BASELINE '''
        if baseline:

            profile = self.prepares(baseline=True)

        # ''' _____________________________________________________________  OFFER '''
        if flexoffer:

            profile = self.prepares(flexoffer=True)

        # ''' _____________________________________________________________  OFFER '''
        if deviation_costs:

            profile = self.prepares(deviation_costs=True)

        return profile

    def stores(
        self,
        baseline: DataFrame = None,
        flexrequest: DataFrame = None,
        flexorder: DataFrame = None,
    ):

        # Get current datetime index
        datetime_index = self.prediction_horizon[0]

        # Get current simulation step
        current_simulation_step = (
            self.simulation_time.get_loc(self.prediction_horizon[0]) + 1
        )

        # Create label name e.g. EMS_1_COM_00001 for second commitment
        label = self.name + str("_COM_") + str(current_simulation_step)

        #''' _____________________________________________________________  REQUEST '''
        if flexrequest is not None:

            # Data columns
            entries = ["Profile", "Price_down", "Price_up"]

            # Copy and rename colum index with ems name
            flexrequest = flexrequest.copy()

            flexrequest.columns.set_levels([self.name], level=0, inplace=True)

            # Store requested values at ems summary
            self.data.at[
                IDX[datetime_index, :], IDX[self.name, "Request"]
            ] = flexrequest

            # Add/subract share from baseline - converting from difference to profile
            flexrequest[self.name, "Request"] += self.data.loc[
                IDX[datetime_index, :], IDX[self.name, "Baseline"]
            ]

            # Store requested values at ems commitment data
            self.commitment_stack[current_simulation_step].at[
                IDX[datetime_index, :], IDX[label, entries]
            ] = flexrequest.values

        # ''' _____________________________________________________________  ORDER '''
        if flexorder is not None:

            logger.info(" _________________________________________________")
            logger.info("|												  |")
            logger.info("|		     ---> {} UPDATES <---              |".format(self.name))
            logger.info("|_________________________________________________|")

            # NOTE:
            logger.info(
                "\n{} picks offer from flexoffer stack and stores it as order: \n\n{}".format(
                    self.name, self.flexoffer_stack[0]
                )
            )


            # Get stacked offer
            stacked_flexoffer = self.flexoffer_stack[
                flexorder.flexoffer_stack_index - 1
            ].copy()

            # Get offer status for period 1
            offer_value_period_1 = stacked_flexoffer[self.name, "Offer"].loc[
                datetime_index, 1
            ]

            current_commitment = self.commitment_stack[current_simulation_step]
            current_commitment_name = current_commitment.columns[0][0]

            # Update summary values OFFER & COST OFFER
            self.data[self.name, "Offer"].loc[datetime_index, :] = (
                stacked_flexoffer[self.name, "Offer"]
                .loc[datetime_index, :]
                .fillna(nan)
                .values
            )

            self.data[self.name, "COSTS_Offer"].loc[datetime_index, :] = (
                stacked_flexoffer[self.name, "COSTS_Offer"]
                .loc[datetime_index, :]
                .fillna(nan)
                .values
            )

            # Sort out non activated values from stacked offer and from current commitment
            for idx, row in flexorder.activation_periods.iteritems():

                if row == False:

                    stacked_flexoffer.at[
                        IDX[datetime_index, idx],
                        IDX[:, ["Offer", "COSTS_Offer", "Deviation"]],
                    ] = nan

                    current_commitment.at[IDX[datetime_index, idx], IDX[label, :]] = nan

            # Update PROFILE values of current commitment
            # TODO: Check if this works with negative profile values
            current_commitment.at[IDX[datetime_index, :], IDX[label, "Profile"]] = (
                current_commitment.loc[IDX[datetime_index, :], IDX[label, "Profile"]]
                + current_commitment.loc[
                    IDX[datetime_index, :], IDX[label, "Deviation"]
                ]
            )

            # Copy profile values to commitment column
            current_commitment.at[
                IDX[datetime_index, :], IDX[label, "Commitment"]
            ] = current_commitment.loc[IDX[datetime_index, :], IDX[label, "Profile"]]

            # Check the deviation costs for all commitments:
            deviation_values = []
            deviation_costs = []

            # 1) Loop over commitments
            for c in self.applicable_commitments[1:]:

                m = c.loc[IDX[:, :], IDX[:, "Deviation"]].values

                for period in range(1, len(self.prediction_horizon) + 1):

                    d = c.loc[IDX[:, period], IDX[:, "Deviation"]].values

                    d = ravel(d)[0]

                    v = c.loc[IDX[:, period], IDX[:, "Commitment"]].values

                    v = ravel(v)[0]

                    g = c.loc[IDX[:, period], IDX[:, "COSTS_Deviation"]].values

                    g = ravel(g)[0]

                    if not isnan(v):
                        if g != 0:

                            deviation_values.append(d)

                            if d >= 0:
                                costs = (
                                    d * c.loc[IDX[:, period], IDX[:, "Price_up"]].values
                                )

                            else:
                                costs = (
                                    d
                                    * c.loc[IDX[:, period], IDX[:, "Price_down"]].values
                                )

                            deviation_costs.append(costs)


                    # Nan to rows where commitment not available
                    c.loc[isnan(c[c.columns[0][0], "Commitment"]) == True, :] = nan

            # Copy "Offer" to "Order" @ SUMMARY
            if not isnan(self.data[self.name, "Commitment"].loc[datetime_index, 1]):

                if deviation_values:

                    # Replace nan with 0
                    self.data[self.name, "Deviation"].loc[datetime_index, 1] = 0

                    # Set deviation values
                    self.data[self.name, "Deviation"].loc[datetime_index, 1] = sum(
                        deviation_values
                    )

                if deviation_costs:

                    # Replace nan with 0
                    self.data[self.name, "COSTS_Deviation"].loc[datetime_index, 1] = 0

                    # Set deviation costs
                    self.data[self.name, "COSTS_Deviation"].loc[
                        datetime_index, 1
                    ] = sum(deviation_costs)

            # Copy "Offer" to "Order" @ SUMMARY
            self.data[self.name, "Order"].loc[datetime_index, :] = (
                stacked_flexoffer[self.name, "Offer"]
                .loc[datetime_index, :]
                .fillna(nan)
                .values
            )

            # Copy "COSTS_Offer" to "COSTS_Order" @ SUMMARY
            self.data[self.name, "COSTS_Order"].loc[datetime_index, :] = (
                stacked_flexoffer[self.name, "COSTS_Offer"]
                .loc[datetime_index, :]
                .fillna(nan)
                .values
            )

            # Iter over commitment rows of current commitment and write the values to corresponding future commitments
            for index, row in current_commitment[
                current_commitment_name, "Commitment"
            ].items():

                datetime_ahead = self.prediction_horizon[index[1] - 1]

                if not isnan(row):
                    self.data[self.name, "Commitment"].loc[datetime_ahead, 1] = row

            # Write current "Flexprofile" value as the realised value
            self.data[self.name, "Realised"].loc[datetime_index, 1] = self.data[
                self.name, "Flexprofile"
            ].loc[datetime_index, 1]

            # Shifting commitments
            # logger.info('\n{} stored flexorder as commitment:\n\n{}\n'.format(self.name,_commitment))
            next_time_step_index = self.prediction_horizon[0] + to_timedelta(
                self.simulation_time.freq
            )
            arrays = [
                [next_time_step_index],
                list(range(1, len(self.prediction_horizon) + 1)),
            ]
            row_multiindex = MultiIndex.from_product(arrays, names=["Time", "Period"])
            number_of_prediction_steps = len(self.prediction_horizon)

            for i in self.indices_applicable_commitments:

                self.commitment_stack[i] = self.commitment_stack[i].shift(-1)

                self.commitment_stack[i].index = row_multiindex

            if "Storage" in self.device_constraints:

                realised_power = self.data.loc[
                    IDX[datetime_index, 1], IDX[self.name, "Storage_FLEX"]
                ]  # .droplevel(level=0)


                last_timeslot_of_current_day = self.simulation_time[-1]

                if last_timeslot_of_current_day.hour > self.simulation_time[-1].hour:
                    last_timeslot_of_current_day = self.simulation_time[-1]

                self.device_constraints["Storage"].loc[
                    (
                        datetime_index + self.simulation_time.freq
                    ) : last_timeslot_of_current_day,
                    "max",
                ] -= realised_power

                self.device_constraints["Storage"].loc[
                    (
                        datetime_index + self.simulation_time.freq
                    ) : last_timeslot_of_current_day,
                    "min",
                ] -= realised_power

                self.device_constraints["Storage"]["min"] = where(
                    self.device_constraints["Storage"]["min"] < 0,
                    0,
                    self.device_constraints["Storage"]["min"],
                )

                logger.info(
                    "\n{} updated buffer storage constraints\n".format(
                        self.name, last_timeslot_of_current_day
                    )
                )

            # Empty the flexoffer stack
            self.flexoffer_stack = []

            logger.info(
                "\n{} resets flexoffer stack \n{}\n".format(
                    self.name, self.flexoffer_stack
                )
            )

        return

    def prepares(
        self,
        baseline: bool = False,
        flexoffer: bool = False,
        deviation_costs: bool = False,
    ):

        # Get current datetime index
        datetime_index = self.prediction_horizon[0]

        # ''' _____________________________________________________________  BASELINE '''

        if baseline:

            profile = self.data.loc[
                IDX[datetime_index, :], IDX[self.name, ["Baseline"]]
            ]

            logger.info(
                "\n{} sends baseline to Aggregator:\n{}".format(self.name, profile)
            )

            return profile

        # ''' _____________________________________________________________  OFFER '''

        if flexoffer:

            # Get current flexoffer with all columns, they'll be sliced afterwards
            flexoffer = self.data.loc[IDX[datetime_index, :], IDX[self.name, :]].copy()

            # Make a copy for the stack
            flexoffer_temp = flexoffer.copy()

            # 1.1) Get difference between baseline values and flexprofile and store them as "Offer"
            flexoffer[self.name, "Offer"] = [
                get_difference_with_sign(
                    first_value=flexprofile_power_value,
                    second_value=basline_power_value,
                )
                for flexprofile_power_value, basline_power_value, in zip(
                    flexoffer[self.name, "Flexprofile"],
                    flexoffer[self.name, "Baseline"],
                )
            ]

            # Check if offer is valid

            # 1) Set NaN at horizon steps with equal values for baseline and flexprofile
            # TODO: Place switch here
            flexoffer[self.name, "Offer"] = where(
                # If baseline is not equal to flexprofile..
                to_numeric(flexoffer[self.name, "Baseline"])
                != to_numeric(flexoffer[self.name, "Flexprofile"]),
                # .. use scheduled value from flexprofile, else use nan
                flexoffer[self.name, "Offer"],
                nan,
            )

            # 2) If offer and request have different signs, write NaN
            flexoffer[self.name, "Offer"] = where(
                # If baseline is not equal to flexprofile..
                sign(flexoffer[self.name, "Offer"])
                != sign(flexoffer[self.name, "Request"]),
                # .. use scheduled value from flexprofile, else use nan
                nan,
                flexoffer[self.name, "Offer"],
            )

            flexoffer[self.name, "Offer"][flexoffer[self.name, "Offer"] > 0] = where(
                flexoffer[self.name, "Request"] > flexoffer[self.name, "Offer"],
                flexoffer[self.name, "Offer"],
                flexoffer[self.name, "Request"],
            )

            flexoffer[self.name, "Offer"][flexoffer[self.name, "Offer"] < 0] = where(
                flexoffer[self.name, "Request"] < flexoffer[self.name, "Offer"],
                flexoffer[self.name, "Offer"],
                flexoffer[self.name, "Request"],
            )

            # Compute FLEXCOSTS using EMS flexibility price
            flexoffer[self.name, "COSTS_Offer"] = (
                abs(flexoffer[self.name, "Offer"]) * self.flexibility_tariff
            )

            flexoffer[self.name, "COSTS_Deviation"] = (
                abs(flexoffer[self.name, "Offer"]) * 0
            )
            flexoffer_temp[self.name, "COSTS_Deviation"] = (
                abs(flexoffer[self.name, "Offer"]) * 0
            )

            # Iter over applicable commitments and extract devation costs per period
            for commitment in self.applicable_commitments[1:]:

                for period in range(1, len(self.prediction_horizon) + 1):

                    flexoffer[self.name, "COSTS_Deviation"].loc[
                        datetime_index, period
                    ] += commitment.loc[
                        IDX[datetime_index, period], IDX[:, "COSTS_Deviation"]
                    ].values

            # Set NaN for cost values if no offer values have been selected
            flexoffer[self.name, "COSTS_Offer"] = where(
                flexoffer[self.name, "Offer"].isnull() != True,
                flexoffer[self.name, "COSTS_Offer"],
                nan,
            )

            # Slice flexoffer before sending to the Aggregator
            profile = flexoffer.loc[
                IDX[datetime_index, :],
                IDX[self.name, ["Request", "Offer", "COSTS_Offer", "COSTS_Deviation"]],
            ]

            # Update flexoffer and put it on temporary stack (necessary if several disaggregation attempts)
            flexoffer_temp.loc[
                IDX[datetime_index, :],
                IDX[self.name, ["Request", "Offer", "COSTS_Offer", "COSTS_Deviation"]],
            ] = profile.values

            self.appends_flexoffer_to_stack(flexoffer=profile)

            logger.info("\n{} sends flexoffer: \n\n{}".format(self.name, profile))

        # ''' _____________________________________________________________  DEVIATION COSTS '''

        if deviation_costs:
            profile = self.data[self.name, "COSTS_Deviation"].loc[datetime_index, 1]

        return profile

    @property
    def sets_energy_contract_data(self):

        # Populate Energy Contract data for the EMS
        self.commitment_stack[0].at[
            IDX[:, :], IDX[self.name + str("_") + "EC", "Profile"]
        ] = 0

        self.commitment_stack[0].at[
            IDX[:, :], IDX[self.name + str("_") + "EC", "Price_down"]
        ] = self.feedin_tariff

        self.commitment_stack[0].at[
            IDX[:, :], IDX[self.name + str("_") + "EC", "Price_up"]
        ] = self.purchase_tariff

        return
