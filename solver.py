# import cplex
import os

from typing import List, Tuple, Union, NamedTuple, Type
from datetime import datetime

from pyomo.gdp import Disjunct, Disjunction
from pyomo.environ import UnknownSolver, Suffix
from pyomo.core.expr.numvalue import value
from pyomo.opt import SolverFactory
from pyomo.opt import SolverStatus, TerminationCondition

from pandas import (
    Index,
    IndexSlice,
    DataFrame,
    MultiIndex,
    Series,
    to_timedelta,
    DatetimeIndex,
)
from pandas import IndexSlice as IDX

# from pandas.core.common import flatten

from numpy import isnan, nanmin, nanmax, ravel

from pyomo.core import (
    ConcreteModel,
    Var,
    Set,
    RangeSet,
    Param,
    Reals,
    Binary,
    Constraint,
    Objective,
    minimize,
    TransformationFactory,
    BuildAction,
)

from utils import display_constraints

from logger import log as logger

infinity = float("inf")
decimal_points = 4


def convert_commitment_data_to_solver_input_2(
    # applicable_commitments: dict,
    applicable_commitments: DataFrame,
    optimization_horizon: DatetimeIndex,
):

    # commitment_labels = list(commitment_data.columns.levels[0].format())
    commitment_labels = [
        commitment.columns.levels[0].format()[0]
        for commitment in applicable_commitments
    ]

    commitment_quantities = [
        Series(
            data=commitment.loc[
                IDX[commitment.index.get_level_values(0).unique()[-1], :],
                IDX[commitment.columns.levels[0].format()[0], "Profile"],
            ]
            .values.ravel()
            .tolist(),
            index=optimization_horizon,
        )
        if isnan(
            commitment.loc[
                IDX[commitment.index.get_level_values(0).unique()[-1], :],
                IDX[commitment.columns.levels[0].format()[0], "Commitment"],
            ]
            .values.ravel()
            .tolist()
        ).all()
        else Series(
            data=commitment.loc[
                IDX[commitment.index.get_level_values(0).unique()[-1], :],
                IDX[commitment.columns.levels[0].format()[0], "Commitment"],
            ]
            .values.ravel()
            .tolist(),
            index=optimization_horizon,
        )
        for commitment in applicable_commitments
    ]

    commitment_downwards_deviation_price = [
        Series(
            data=list(
                commitment.loc[
                    IDX[commitment.index.get_level_values(0).unique()[-1], :],
                    IDX[commitment.columns.levels[0].format()[0], "Price_down"],
                ]
                .values.ravel()
                .tolist()
            ),
            index=optimization_horizon,
        ).fillna(0)
        for commitment in applicable_commitments
    ]
    commitment_upwards_deviation_price = [
        Series(
            data=list(
                commitment.loc[
                    IDX[commitment.index.get_level_values(0).unique()[-1], :],
                    IDX[commitment.columns.levels[0].format()[0], "Price_up"],
                ]
                .values.ravel()
                .tolist()
            ),
            index=optimization_horizon,
        ).fillna(0)
        for commitment in applicable_commitments
    ]

    return (
        commitment_labels,
        commitment_quantities,
        commitment_downwards_deviation_price,
        commitment_upwards_deviation_price,
    )


def convert_commitment_data_to_solver_input(
    # applicable_commitments: dict,
    applicable_commitments: DataFrame,
    optimization_horizon: DatetimeIndex,
):

    # commitment_labels = list(commitment_data.columns.levels[0].format())
    commitment_labels = []
    commitment_quantities = []
    commitment_downwards_deviation_price = []
    commitment_upwards_deviation_price = []

    for commitment in applicable_commitments:

        label = commitment.columns.levels[0].format()[0]
        commitment_labels.append(label)

        last_index = commitment.index.get_level_values(0).unique()[-1]

        quantity = (
            commitment.loc[IDX[last_index, :], IDX[label, "Commitment"]]
            .values.ravel()
            .tolist()
        )


        if isnan(quantity).all():

            quantity = (
                commitment.loc[IDX[last_index, :], IDX[label, "Profile"]]
                .values.ravel()
                .tolist()
            )

        commitment_quantities.append(Series(data=quantity, index=optimization_horizon))

        downwards_deviation_price = list(
            commitment.loc[IDX[last_index, :], IDX[label, "Price_down"]]
            .values.ravel()
            .tolist()
        )

        commitment_downwards_deviation_price.append(
            Series(data=downwards_deviation_price, index=optimization_horizon).fillna(0)
        )

        upwards_deviation_price = list(
            commitment.loc[IDX[last_index, :], IDX[label, "Price_up"]]
            .values.ravel()
            .tolist()
        )

        commitment_upwards_deviation_price.append(
            Series(data=upwards_deviation_price, index=optimization_horizon).fillna(0)
        )

    return (
        commitment_labels,
        commitment_quantities,
        commitment_downwards_deviation_price,
        commitment_upwards_deviation_price,
    )


def device_scheduler(
    EMS_name: str,
    EMS_data: DataFrame,
    applicable_commitments: DataFrame,
    optimization_horizon: Series,
    device_constraints: dict,
    grid_constraint: DataFrame,
    baseline: bool = False,
) -> NamedTuple:

    """Schedule devices given constraints on a device and EMS level, and given a list of commitments by the EMS.
    The commitments are assumed to be with regards to the flow of energy to the device (positive for consumption,
    negative for production). The solver minimises the costs of deviating from the commitments, and returns the costs
    per commitment.
    Device constraints are on a device level. Handled constraints (listed by column name):
        max: maximum stock assuming an initial stock of zero (e.g. in MWh or boxes)
        min: minimum stock assuming an initial stock of zero
        derivative max: maximum flow (e.g. in MW or boxes/h)
        derivative min: minimum flow
        derivative equals: exact amount of flow
    EMS constraints are on an EMS level. Handled constraints (listed by column name):
        derivative max: maximum flow
        derivative min: minimum flow
    Commitments are on an EMS level. Parameter explanations:
        commitment_quantities: amounts of flow specified in commitments (both previously ordered and newly requested)
            - e.g. in MW or boxes/h
        commitment_downwards_deviation_price: penalty for downwards deviations of the flow
            - e.g. in EUR/MW or EUR/(boxes/h)
            - either a single value (same value for each flow value) or a Series (different value for each flow value)
        commitment_upwards_deviation_price: penalty for upwards deviations of the flow
    All Series and DataFrames should have the same resolution.
    For now we pass in the various constraints and prices as separate variables, from which we make a MultiIndex
    DataFrame. Later we could pass in a MultiIndex DataFrame directly.
    """

    """ ________ PREPROCESS INPUT DATA ________"""

    # Pretty print device constraints
    display_constraints(
        EMS_name=EMS_name,
        device_constraints=device_constraints,
        grid_constraint=grid_constraint,
        horizon=optimization_horizon,
    )

    # Convert device names and constraints
    device_names = [device for device, constraint in device_constraints.items()]
    device_constraints = [
        constraint.loc[optimization_horizon[0] : optimization_horizon[-1]]
        for constraint in device_constraints.values()
    ]

    if baseline:
        del applicable_commitments[-1]

    (
        commitment_names,
        commitment_quantities,
        commitment_downwards_deviation_price,
        commitment_upwards_deviation_price,
    ) = convert_commitment_data_to_solver_input_2(
        applicable_commitments=applicable_commitments,
        optimization_horizon=optimization_horizon,
    )

    """ ________ CATCH INPUT ERRORS ________"""

    # If the EMS has no devices, don't bother
    if len(device_constraints) == 0:
        return [], [] * len(commitment_quantities)

    # Check if commitments have the same time window and resolution as the constraints
    start = device_constraints[0].index[0]
    resolution = to_timedelta(device_constraints[0].index.freq)
    end = device_constraints[0].index[-1] + resolution

    if len(commitment_quantities) != 0:
        start_c = commitment_quantities[0].index[0]
        resolution_c = to_timedelta(commitment_quantities[0].index.freq)
        end_c = commitment_quantities[0].index[-1] + resolution
        if not (start_c == start and end_c == end):
            raise Exception(
                "Not implemented for different time windows.\n(%s,%s)\n(%s,%s)"
                % (start, end, start_c, end_c)
            )
        if resolution_c != resolution:
            raise Exception(
                "Not implemented for different resolutions.\n%s\n%s"
                % (resolution, resolution_c)
            )

    """ ________ MAX MIN PRICE BOUNDS ________"""

    # Determine appropriate overall bounds for power and price
    min_down_price = min(min(p) for p in commitment_downwards_deviation_price)
    max_down_price = max(max(p) for p in commitment_downwards_deviation_price)
    min_up_price = min(min(p) for p in commitment_upwards_deviation_price)
    max_up_price = max(max(p) for p in commitment_upwards_deviation_price)
    overall_min_price = min(min_down_price, min_up_price)
    overall_max_price = max(max_down_price, max_up_price)
    overall_min_power = min(grid_constraint["derivative min"])
    overall_max_power = max(grid_constraint["derivative max"])

    """ ________ MODEL INSTANCE ________"""

    model = ConcreteModel()

    """ ________ SETS  ________"""

    # Add indices for devices (d), datetimes (j) and commitments (c)
    model.d = RangeSet(0, len(device_constraints) - 1, doc="Set of devices")

    model.j = RangeSet(
        0, len(device_constraints[0].index.values) - 1, doc="Set of datetimes"
    )
    model.c = RangeSet(0, len(commitment_quantities) - 1, doc="Set of commitments")

    """ ________ PARAMETER SELECTION  ________"""

    # Add parameters
    def commitment_quantity_select(m, c, j):
        # NOTE: The energy contract commitment comes with zeros as quantities,
        #   	so that the price does not get changed here
        v = commitment_quantities[c].iloc[j]
        if isnan(v):  # Discount this nan commitment by setting the prices to 0
            commitment_downwards_deviation_price[c].iloc[j] = 0
            commitment_upwards_deviation_price[c].iloc[j] = 0
            return 0
        else:
            return v

    def price_down_select(m, c, j):
        return commitment_downwards_deviation_price[c].iloc[j]

    def price_up_select(m, c, j):
        return commitment_upwards_deviation_price[c].iloc[j]

    def device_max_select(m, d, j):
        v = device_constraints[d]["max"].iloc[j]
        if isnan(v):
            return infinity
        else:
            return v

    def device_min_select(m, d, j):
        v = device_constraints[d]["min"].iloc[j]
        if isnan(v):
            return -infinity
        else:
            return v

    def device_derivative_max_select(m, d, j):
        max_v = device_constraints[d]["derivative max"].iloc[j]
        equal_v = device_constraints[d]["derivative equals"].iloc[j]
        if isnan(max_v) and isnan(equal_v):
            return infinity
        else:
            return nanmin([max_v])

    def device_derivative_min_select(m, d, j):
        min_v = device_constraints[d]["derivative min"].iloc[j]
        equal_v = device_constraints[d]["derivative equals"].iloc[j]
        if isnan(min_v) and isnan(equal_v):
            return -infinity
        else:
            return nanmax([min_v])

    def device_derivative_equal_select(m, d, j):
        min_v = device_constraints[d]["derivative min"].iloc[j]
        equal_v = device_constraints[d]["derivative equals"].iloc[j]

        if isnan(equal_v):
            return 0
        else:
            return nanmax([equal_v])

    def ems_derivative_max_select(m, j):
        v = grid_constraint["derivative max"].iloc[j]
        if isnan(v):
            return infinity
        else:
            return v

    def ems_derivative_min_select(m, j):
        v = grid_constraint["derivative min"].iloc[j]
        if isnan(v):
            return -infinity
        else:
            return v

    """ ________ PARAMETER  ________"""

    model.commitment_quantity = Param(
        model.c, model.j, initialize=commitment_quantity_select
    )

    model.up_price = Param(model.c, model.j, initialize=price_up_select)
    model.down_price = Param(model.c, model.j, initialize=price_down_select)
    model.device_max = Param(model.d, model.j, initialize=device_max_select)
    model.device_min = Param(model.d, model.j, initialize=device_min_select)

    model.device_derivative_max = Param(
        model.d, model.j, initialize=device_derivative_max_select
    )

    model.device_derivative_min = Param(
        model.d, model.j, initialize=device_derivative_min_select
    )

    model.device_derivative_equal = Param(
        model.d, model.j, initialize=device_derivative_equal_select
    )

    model.ems_derivative_max = Param(model.j, initialize=ems_derivative_max_select)
    model.ems_derivative_min = Param(model.j, initialize=ems_derivative_min_select)

    """ ________ VARIABLES  ________"""

    # Add variables
    model.power = Var(
        model.d,
        model.j,
        domain=Reals,
        initialize=0,
        bounds=(overall_min_power, overall_max_power),
    )

    # Add logical disjunction for deviations
    model.price = Var(
        model.c, model.j, initialize=0, bounds=(overall_min_price, overall_max_price)
    )

    """ ________ CONSTRAINT RULES  ________"""

    # Add constraints as a tuple of (lower bound, value, upper bound)
    def device_bounds(m, d, j):
        return (
            m.device_min[d, j],
            sum(m.power[d, k] for k in range(0, j + 1)),
            m.device_max[d, j],
        )

    def device_derivative_bounds(m, d, j):
        return (
            m.device_derivative_min[d, j],
            m.power[d, j] - m.device_derivative_equal[d, j],
            m.device_derivative_max[d, j],
        )

    def ems_derivative_bounds(m, j):
        return m.ems_derivative_min[j], sum(m.power[:, j]), m.ems_derivative_max[j]

    """ ________ CONSTRAINTS  ________"""

    # Constraints on devices energy
    model.device_energy_bounds = Constraint(model.d, model.j, rule=device_bounds)

    # Constraints on devices power
    model.device_power_bounds = Constraint(
        model.d, model.j, rule=device_derivative_bounds
    )

    # Constraints on EMS power
    model.ems_power_bounds = Constraint(model.j, rule=ems_derivative_bounds)

    """ ________ DISJUNCTS SELECTION  ________"""

    def up_linker(b, c, d, j):
        m = b.model()
        ems_power_in_j = sum(m.power[d, j] for d in m.d)
        ems_power_deviation = ems_power_in_j - m.commitment_quantity[c, j]
        b.linker = Constraint(expr=m.price[c, j] == m.up_price[c, j])
        b.constr = Constraint(expr=ems_power_deviation >= 0)
        b.BigM = Suffix(direction=Suffix.LOCAL)
        b.BigM[b.linker] = 10e5
        return

    def down_linker(b, c, d, j):
        m = b.model()
        ems_power_in_j = sum(m.power[d, j] for d in m.d)
        ems_power_deviation = ems_power_in_j - m.commitment_quantity[c, j]
        b.linker = Constraint(expr=m.price[c, j] == m.down_price[c, j])
        b.constr = Constraint(expr=ems_power_deviation <= 0)
        b.BigM = Suffix(direction=Suffix.LOCAL)
        b.BigM[b.linker] = 10e5
        return

    """ ________ DISJUNCTS  ________"""

    model.up_deviation = Disjunct(model.c, model.d, model.j, rule=up_linker)
    model.down_deviation = Disjunct(model.c, model.d, model.j, rule=down_linker)
    # model.zero_deviation = Disjunct(model.c, model.d, model.j, rule=zero_linker)

    def bind_prices(m, c, d, j):
        return [
            model.up_deviation[c, d, j],
            model.down_deviation[c, d, j],
            # model.zero_deviation[c, d, j]
        ]

    """ ________ DISJUNCTIONS  ________"""

    model.up_or_down_deviation = Disjunction(
        model.c, model.d, model.j, rule=bind_prices  # xor=True
    )

    """ ________ OBJECTIVE FUNCTION  ________"""

    # Add objective
    def cost_function(m):
        costs = 0
        for j in m.j:
            for c in m.c:
                ems_power_in_j = sum(m.power[d, j] for d in m.d)
                ems_power_deviation = ems_power_in_j - m.commitment_quantity[c, j]
                costs += ems_power_deviation * m.price[c, j]
        return costs

    model.costs = Objective(rule=cost_function, sense=minimize)

    """ ________ SOLVER CONFIGURATION  ________"""

    # Transform and solve
    xfrm = TransformationFactory("gdp.bigm")
    xfrm.apply_to(model)

    executable = os.getcwd()+'/ipopt/bin/ipopt.exe'
    solver = SolverFactory("ipopt", executable=executable,solver_io='nl')

    # solver = SolverFactory(
    #     "cplex", executable="D:/CPLEX/Studio/cplex/bin/x64_win64/cplex"
    # )
    # solver.options["qpmethod"] = 1
    # solver.options["optimalitytarget"] = 3

    """ ________ SOLVING  ________"""

    results = solver.solve(model, tee=False, keepfiles=False)

    """ ________ SOLVER STATUS OUTPUT ________"""

    if (results.solver.status == SolverStatus.ok) and (
        results.solver.termination_condition == TerminationCondition.optimal
    ):
        pass
        logger.info(">>>> {} UC: STATUS {}".format(EMS_name, "OPTIMAL"))
        # print("---> Status: OPTIMAL \n")

        # Do something when the solution in optimal and feasible
    elif results.solver.termination_condition == TerminationCondition.infeasible:
        # Do something when model in infeasible
        logger.warning(">>>> {} UC: Infeasible Solution <<<<".format(EMS_name))
    else:
        # Something else is wrong
        logger.warning(">>>> {} UC: Infeasible Solution <<<<".format(EMS_name))

    """ ________ DATA POSTPROCESS ________"""

    # Replace nan with zero, then add costs below
    if baseline:

        profile_type = "Baseline"
        _type = "_BL"
        # deviation_costs = "COSTS_Base_dev"
        energy_contract_costs = "COSTS_Base_EC"

    else:

        profile_type = "Flexprofile"
        _type = "_FLEX"
        # deviation_costs = "COSTS_Offer_dev"
        energy_contract_costs = "COSTS_Offer_EC"

    # Planned POWER VALUES per device and horizon step
    for d, name in zip(model.d, device_names):

        for j, index in zip(model.j, optimization_horizon):

            # Get power values per device and datetime
            EMS_data.loc[
                IDX[optimization_horizon[0], j + 1], IDX[EMS_name, name + _type]
            ] = round(value(model.power[d, j]), decimal_points)

    EMS_data.loc[IDX[optimization_horizon[0], :], IDX[EMS_name, profile_type]] = 0
    EMS_data.loc[
        IDX[optimization_horizon[0], :], IDX[EMS_name, energy_contract_costs]
    ] = 0

    # DEVIATION, COSTS, PRICES per commitment
    for c, c_name in zip(model.c, commitment_names):

        for j, index in zip(model.j, optimization_horizon):

            # Total power
            ems_power_in_j = sum(model.power[d, j] for d in model.d)

            # Deviations
            ems_power_deviation = ems_power_in_j - model.commitment_quantity[c, j]

            # Commitment quantities per commitment over horizon steps
            applicable_commitments[c].loc[
                IDX[optimization_horizon[0], j + 1], IDX[c_name, "Profile"]
            ] = round(value(model.commitment_quantity[c, j]), decimal_points)

            commitment = applicable_commitments[c].loc[
                IDX[optimization_horizon[0], j + 1], IDX[c_name, "Commitment"]
            ]

            # if not isnan(commitment):
            # Deviations per horizon step and current simulation index
            applicable_commitments[c].loc[
                IDX[optimization_horizon[0], j + 1], IDX[c_name, "Deviation"]
            ] = round(value(ems_power_deviation), decimal_points)

            # Sum of requested quantities over all commitments
            applicable_commitments[c].loc[
                IDX[optimization_horizon[0], j + 1], IDX[c_name, "Price_up"]
            ] = round(value(model.up_price[c, j]), decimal_points)

            applicable_commitments[c].loc[
                IDX[optimization_horizon[0], j + 1], IDX[c_name, "Price_down"]
            ] = round(value(model.down_price[c, j]), decimal_points)

            # Input prices per horizon step and current simulation index
            if value(ems_power_deviation) >= 0:

                if not isnan(commitment):
                    # Costs per horizon step and current simulation index
                    applicable_commitments[c].loc[
                        IDX[optimization_horizon[0], j + 1],
                        IDX[c_name, "COSTS_Deviation"],
                    ] = round(
                        value(ems_power_deviation * model.up_price[c, j]),
                        decimal_points,
                    )
                else:
                    applicable_commitments[c].loc[
                        IDX[optimization_horizon[0], j + 1],
                        IDX[c_name, "COSTS_Deviation"],
                    ] = 0

                # Dont add energy contract costs
                if "EC" in c_name:
                    EMS_data.loc[
                        IDX[optimization_horizon[0], j + 1],
                        IDX[EMS_name, energy_contract_costs],
                    ] += round(
                        value(ems_power_deviation * model.up_price[c, j]),
                        decimal_points,
                    )
            else:

                if not isnan(commitment):

                    # Costs per horizon step and current simulation index
                    applicable_commitments[c].loc[
                        IDX[optimization_horizon[0], j + 1],
                        IDX[c_name, "COSTS_Deviation"],
                    ] = round(
                        value(ems_power_deviation * model.down_price[c, j]),
                        decimal_points,
                    )
                else:
                    applicable_commitments[c].loc[
                        IDX[optimization_horizon[0], j + 1],
                        IDX[c_name, "COSTS_Deviation"],
                    ] = 0

                # Dont add energy contract costs
                if "EC" in c_name:
                    EMS_data.loc[
                        IDX[optimization_horizon[0], j + 1],
                        IDX[EMS_name, energy_contract_costs],
                    ] += round(
                        value(ems_power_deviation * model.down_price[c, j]),
                        decimal_points,
                    )

        for j, index in zip(model.j, optimization_horizon):
            # Store total EMS Power per horizon step and current simulation index
            EMS_data.loc[
                IDX[optimization_horizon[0], j + 1], IDX[EMS_name, profile_type]
            ] = round(value(sum(model.power[d, j] for d in model.d)), decimal_points)

    energy_contract_costs_total = EMS_data.loc[
        IDX[optimization_horizon[0], :], IDX[EMS_name, energy_contract_costs]
    ].sum(axis=0)

    logger.info(
        "\n---> Total costs: {}".format(round(value(model.costs), decimal_points))
    )
    logger.info('---> Deviation costs: {}'.format(deviation_costs_total))
    logger.info("---> Energy contract costs: {}".format(energy_contract_costs_total))

    # logger.info('\n{} UC Summary: \n\n{}\n'.format(
    #     EMS_name, EMS_data[EMS_data.columns[~EMS_data.isnull().any()]].sum()))

    logger.info("\n{} results per commitment:\n".format(EMS_name))
    for commitment in applicable_commitments:
        logger.info(
            "{}\n".format(commitment.loc[IDX[optimization_horizon[0], :], IDX[:, :]])
        )

    return

    # WRITE CPLEX TXT FILE
    # import os
    # path_to_script = os.path.dirname(os.path.abspath(__file__))
    # my_filename = os.path.join(path_to_script, "my_file.csv")
    #
    # # with open(my_filename, "w") as cplexlog:
    # # #print("Hello world!", file=handle)
    # #     cpx = cplex.Cplex()
    # #     # cplexlog = "cplex.log"
    # #     cpx.set_results_stream(cplexlog.write())
    # #     cpx.set_warning_stream(cplexlog.write())
    # #     cpx.set_error_stream(cplexlog.write())
    # #     cpx.set_log_stream(cplexlog.write())
    #
    # cpx = cplex.Cplex(model)
    # # cplexlog = "cplex.log"
    # # cpx.set_results_stream(None)
    # print(cpx.get_problem_type())
    # print(cpx.problem_type[cpx.get_problem_type()])
    # # cpx.set_warning_stream(cplexlog)
    # # cpx.set_error_stream(cplexlog)
    # # cpx.set_log_stream(cplexlog)

    # SOLUTION RESULTS
    # help(results.pprint)
    # model.pprint()
    # results.write(num=1)

    # Loading solution into results object
    # model.load(results)
    # model.display()
    # results.pprint()
    # model.down_deviation.pprint()
    # model.up_deviation.pprint()
