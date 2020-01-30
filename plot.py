#%%
import warnings
import logging
import sys
import os
from copy import deepcopy
from random import uniform
import matplotlib.pyplot as plt
import plotly.offline
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from datetime import timedelta, datetime, date

from numpy import *
from numpy import sign
from pandas import *
from pandas import IndexSlice as IDX

from utils import load_scenario, get_difference_with_sign

# Console output configuration
set_option("display.max_columns", 22)
set_option("display.max_rows", 1000)
set_option("display.width", 1000)
set_option("display.colheader_justify", "left")
set_option("display.colheader_justify", "left")
set_option("max_info_columns", 1000)
set_option("display.precision", 6)

warnings.simplefilter(action="ignore", category=errors.PerformanceWarning)

def plot_FDP_input_preview(
    start: DatetimeIndex,
    end: DatetimeIndex,
    episodes: int,
    one_day_index: DatetimeIndex,
    result_directory: str,
    scenario_name: str,
    imbalances_profile: DataFrame = None,
    market_price_profile: DataFrame = None,
    imbalances_stats: DataFrame = None,
    market_price_stats: DataFrame = None,
):
    """ _____________________________________________________________ FONTS """

    xaxis_tick_font = dict(family="Roboto", size=18, color="black")

    yaxis_tick_font = dict(family="Roboto", size=18, color="black")

    yaxis_title_font = dict(family="Roboto", size=24, color="black")

    revenues_row = 1
    imbalances_rolling_row = 2
    imbalances_row = 3
    market_prices_row = 4

    fig = make_subplots(rows=4, cols=1, column_widths=[1], vertical_spacing=0.07)

    index_strf = one_day_index.strftime("%H:%M")

    if market_price_profile is not None:

        for column in market_price_profile.columns:

            fig.append_trace(
                go.Box(
                    name=str(one_day_index[column[0]].strftime("%H:%M")),
                    y=market_price_profile[column],
                    boxmean=True,
                    marker_color="lightseagreen",
                    showlegend=False,
                    xaxis="x" + str(market_prices_row),
                    yaxis="y" + str(market_prices_row),
                ),
                row=market_prices_row,
                col=1,
            )

    if imbalances_profile is not None:

        for column in imbalances_profile.columns:
            fig.append_trace(
                go.Box(
                    name=str(one_day_index[column[0]].strftime("%H:%M")),
                    y=imbalances_profile[column],
                    boxmean=True,
                    marker_color="royalblue",
                    showlegend=False,
                    xaxis="x" + str(imbalances_row),
                    yaxis="y" + str(imbalances_row),
                ),
                row=imbalances_row,
                col=1,
            )

    if imbalances_profile is not None and market_price_profile is not None:

        ib_means_rolling = imbalances_stats.loc["Mean", :].rolling(3).mean().shift(-2)

        fig.append_trace(
            go.Scatter(
                x=index_strf,
                y=ib_means_rolling.values,
                showlegend=False,
                mode="lines+markers",
                line=dict(color="green"),
                marker=dict(
                    symbol="circle",
                    line=dict(color="black", width=1),
                    size=4,
                    color="lightgreen",
                    showscale=False,
                ),
                xaxis="x" + str(imbalances_rolling_row),
                yaxis="y" + str(imbalances_rolling_row),
            ),
            row=imbalances_rolling_row,
            col=1,
        )

    if imbalances_profile is not None and market_price_profile is not None:

        estimated_revenues = market_price_stats.loc["Mean", :] * ib_means_rolling

        fig.append_trace(
            go.Scatter(
                x=index_strf,
                y=estimated_revenues.values,
                mode="lines+markers",
                marker=dict(
                    symbol="circle",
                    line=dict(color="black", width=1),
                    size=4,
                    color="lightblue", 
                    showscale=False,
                ),
                showlegend=False,
                line=dict(color="blue"),
                xaxis="x" + str(revenues_row),
                yaxis="y" + str(revenues_row),
            ),
            row=revenues_row,
            col=1,
        )

    """ _____________________________________________________________  SCENARIO TABLE """

    # Scenario data
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 1], y=[0.88, 1]),
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            columnwidth=[
                0.18,  # 1
                0.32,  # 2
                0.08,  # 3
                0.08,  # 4
                0.08,  # 5
                0.04,  # 6
                0.04,  # 7
                0.04,  # 8
                0.04,  # 9
                0.04,  # 10
                0.06,  # 11
            ],
            header=dict(
                values=[
                    "Scenario",
                    "Description",
                    "Plotted",
                    "Start",
                    "End",
                    "Episodes",
                ],
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    scenario_name,
                    "",
                    "{}".format(datetime.now().strftime("%d.%m.%Y %H:%M")),
                    "{}".format(start.strftime("%d.%m.%Y %H:%M")),
                    "{}".format(end.strftime("%d.%m.%Y %H:%M")),
                    episodes,
                ],
                line_color="lightgrey",
                font=dict(color="black", size=11),
                height=25,
            ),
        )
    )

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        tickangle=-90,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightPink",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        tickfont=xaxis_tick_font,
    )

    fig.update_yaxes(
        domain=[0, 0.86],
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        tickfont=yaxis_tick_font,
        title="Cents/unit",
        zeroline=False,
        showgrid=True,
        gridwidth=1,
        gridcolor="LightPink",
    )

    fig["layout"]["margin"] = {"b": 10, "t": 10}
    fig["layout"]["yaxis1"].update(domain=[0, 0.18])
    fig["layout"]["yaxis2"].update(domain=[0.25, 0.43])
    fig["layout"]["yaxis2"].update(title="Power")
    fig["layout"]["yaxis3"].update(domain=[0.50, 0.68])
    fig["layout"]["yaxis4"].update(domain=[0.75, 0.93])

    fig.update_layout(
        showlegend=False,
        height=2400,
        annotations=[
            go.layout.Annotation(
                x=0.5,
                y=0.18,
                xref="paper",
                yref="paper",
                text="Revenues",
                showarrow=False,
            ),
            go.layout.Annotation(
                x=0.5,
                y=0.44,
                xref="paper",
                yref="paper",
                text="Imbalances rolling",
                showarrow=False,
            ),
            go.layout.Annotation(
                x=0.5,
                y=0.7,
                xref="paper",
                yref="paper",
                text="Distribution of request values",
                showarrow=False,
            ),
            go.layout.Annotation(
                x=0.5,
                y=0.95,
                xref="paper",
                yref="paper",
                text="Distribution of request prices",
                showarrow=False,
            ),
        ],
    )

    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=18, color="black", family="Roboto")

    # Store plot
    plotly.offline.plot(
        fig, filename=result_directory + "/PLOTS/" + scenario_name + "_FDP_Inputs.html"
    )


def plot_optimization_data(scenario, horizon=None, stats=True):

    """ _____________________________________________________________ FONTS """

    xaxis_tick_font = dict(family="Times New Roman", size=22, color="black")

    yaxis_tick_font = dict(family="Times New Roman", size=22, color="black")

    yaxis_title_font = dict(family="Times New Roman", size=26, color="black")
    """ _____________________________________________________________ UNITS """

    yaxis_price = "cents/{}".format(scenario.units[0])
    yaxis_costs = "cents"
    yaxis_power = scenario.units[0]
    yaxis_energy = scenario.units[1]

    """ _____________________________________________________________ MARKER & LINES WIDTH """

    line_width = 2
    marker_line_width = 2

    marker_size_1 = 4
    marker_size_2 = 4

    """ _____________________________________________________________ COLORS """

    ems_colors_1 = ["blue", "green"]
    ems_colors_075 = [
        "rgba(108,166,255,{})".format(0.75),  # SkyBlue3
        "rgba(46,139,87,{})".format(0.75),  # Seagreen
    ]
    ems_colors_025 = [
        "rgba(108,166,255,{})".format(0.25),  # SkyBlue3
        "rgba(46,139,87,{})".format(0.25),  # Seagreen
    ]

    device_colors = dict()
    device_colors["Load"] = "blue"
    device_colors["Generator"] = "darkgoldenrod"
    device_colors["Storage"] = "darkgreen"

    # SkyBlue3
    upshift_color = "rgba(240,128,128,{})".format(1)
    upshift_border_color = "rgba(139,0,0,{})".format(1)
    flex_up_color = "crimson"
    flex_up_color_border_color = "darkred"

    downshift_color = "rgba(64,224,208,{})".format(1)
    downshift_border_color = "darkgreen"
    flex_down_color = "seagreen"
    flex_up_color_border_color = "darkseagreen"

    """ _____________________________________________________________ SUBPLOT TITLES """

    subplot_titles = [
        "",
        "FDP: Imbalance market prices",
        "FDP/EMS: Upward prices per datetime",
        "FDP/EMS: Downward prices per datetime",
        "FDP: Horizon requests",
        "FDP: Horizon orders",
        "AGG: Requests and offer per EMS",
        "FDP: Imbalances",
        "AGG: Orders",
        "AGG: Deviations",
        "AGG: Realised ({})".format(scenario.units[0]),
        "AGG: Baseline and realised",
    ]
    """ _____________________________________________________________ HORIZON """

    # horizon = scenario.simulation_time
    if horizon == None:
        horizon = date_range(
            start=scenario.start,
            end=scenario.last_simulated_index,
            freq=scenario.resolution,
        )
    else:
        horizon[-1] = scenario.last_simulated_index

    """ _____________________________________________________________ ENTITIES """

    Aggregator = scenario.Aggregator
    aggregator_prediction_periods = int(
        Aggregator.prediction_delta / scenario.resolution
    )

    FDP = scenario.FDP
    fdp_prediction_periods = int(FDP.prediction_delta / scenario.resolution)

    EMS = dict.fromkeys(Aggregator.EMS_names)
    EMS_realised = dict.fromkeys(Aggregator.EMS_names)
    EMS_devices = dict()
    EMS_device_names = dict()
    EMS_devices_shifts = dict()
    EMS_devices_flexibility = dict()
    EMS_baselines_total = dict()
    EMS_flexibility = dict()

    for enum, name in enumerate(scenario.EMS_names):
        EMS[name] = dict()
        # EMS_flexprofiles[name] = dict()
        EMS_realised[name] = dict()
        EMS_device_names[name] = dict()
        EMS_devices[name] = dict()
        EMS_devices_shifts[name] = dict()
        EMS_devices_flexibility[name] = dict()
        EMS_baselines_total[name] = 0
        EMS_flexibility[name] = Series(data=0, index=horizon)

        for device in scenario.EMS[enum].devices:
            EMS_device_names[name][device] = device
            EMS_devices_shifts[name][device] = device
            EMS_devices_flexibility[name][device] = device
            EMS_devices_flexibility[name][device] = Series(data=0, index=horizon)

    """ _____________________________________________________________ INDICES """

    repeating_periods_index = (
        Aggregator.data[Aggregator.name]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()["Period"]
    )
    repeating_datetimes_index = (
        Aggregator.data[Aggregator.name]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()["Time"]
    )

    rolling_datetimes_index = []
    rolling_periods_index = []

    for enum, datetime_index in enumerate(repeating_datetimes_index):
        # Get index with the predication delta ahead
        r_ix = (
            datetime_index
            - horizon.freq
            + repeating_periods_index.loc[enum] * to_timedelta(horizon.freq)
        )

        # Get r_ix index position in total simualation time
        rolling_periods_index.append(scenario.simulation_time.get_loc(r_ix))

        rolling_datetimes_index.append(r_ix)

    rolling_periods_index = [x + 1 for x in rolling_periods_index]

    """ _____________________________________________________________ ROWS """

    # summary_table_row = 1
    fdp_imbalances_market_prices_row = 2
    up_prices_row = 3
    down_prices_row = 4
    fdp_requests_row = 5
    fdp_orders_row = 6
    fdp_imbalances_row = 7
    aggregator_order_per_EMS_over_total_request_row = 8
    aggregator_deviation_per_EMS_over_total_order_row = 9
    aggregator_flexibility_row = 10
    aggregator_realised_and_baseline_row = 11

    ems_realised_and_baseline_rows = dict()
    ems_storage_SOC_rows = dict()
    ems_device_power_profile_rows = dict()
    ems_shifts_per_device_rows = dict()
    ems_flexibility_rows = dict()
    ems_energy_contract_costs_rows = dict()

    trace = aggregator_realised_and_baseline_row

    """ _____________________________________________________________  TRACES """

    for enum, name in enumerate(scenario.EMS_names):

        # print('name: ', name)

        trace += 1
        ems_realised_and_baseline_rows[name] = trace
        subplot_titles.append("{}: Realised power values and baseline profile".format(name))

    for enum, name in enumerate(scenario.EMS_names):

        # Total shifts trace per EMS
        trace += 1
        ems_flexibility_rows[name] = trace
        subplot_titles.append("{}: Flexibility in {}".format(name, scenario.units[0]))

        ems_device_power_profile_rows[name] = dict()
        ems_shifts_per_device_rows[name] = dict()

        # Add a trace per EMS
    for device in EMS_device_names[name]:

        for enum, name in enumerate(scenario.EMS_names):

            # if "Storage" in device:
            trace += 1
            ems_storage_SOC_rows[name] = trace
            subplot_titles.append("{}: {} storage SOC profile".format(name, device))


        for enum, name in enumerate(scenario.EMS_names):

            # if "Storage" in device:
            #     pass
            # else:
            trace += 1
            ems_device_power_profile_rows[name][device] = trace

        for enum, name in enumerate(scenario.EMS_names):

            trace += 1
            ems_shifts_per_device_rows[name][device] = trace
            subplot_titles.append("{}: {} shifts".format(name, device))

    for enum, name in enumerate(scenario.EMS_names):

        trace = trace + 1
        ems_energy_contract_costs_rows[name] = trace
    ems_dev_costs_row = trace + 1
    ems_offer_costs_row = ems_dev_costs_row + 1
    subplot_titles.append("Offer costs per EMS".format(name))

    aggregator_offer_and_order_costs_row = ems_offer_costs_row + 1
    subplot_titles.append("AGG: Offer and order costs".format(name))

    columns = 1
    rows = aggregator_offer_and_order_costs_row

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                     DATA                                        """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________ DATA AGG REALISED """

    aggregator_realised = (
        Aggregator.data[Aggregator.name]["Realised"]
        .loc[: scenario.last_simulated_index, 1]
        .reset_index()
    )
    aggregator_realised.set_index(
        "Time", inplace=True
    )  # .sort_index(ascending=True, inplace=True)
    """ _____________________________________________________________ DATA AGG ORDER """

    aggregator_orders = (
        Aggregator.data[Aggregator.name]["Order"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    aggregator_orders = aggregator_orders.loc[:, ["Order"]]

    aggregator_orders["Rolling_datetime_index"] = rolling_datetimes_index  #
    aggregator_orders.sort_values("Rolling_datetime_index", inplace=True)

    aggregator_orders = aggregator_orders.groupby("Rolling_datetime_index")[
        ["Order"]
    ].sum()
    aggregator_orders = aggregator_orders.iloc[: -aggregator_prediction_periods + 1]

    """ _____________________________________________________________  DATA AGG OFFER COSTS DATETIME """

    aggregator_offer_costs = (
        Aggregator.data[Aggregator.name]["COSTS_Offer"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    aggregator_offer_costs = aggregator_offer_costs.loc[:, ["COSTS_Offer"]]

    aggregator_offer_costs["Rolling_datetime_index"] = rolling_datetimes_index  #
    aggregator_offer_costs.sort_values("Rolling_datetime_index", inplace=True)

    aggregator_offer_costs = aggregator_offer_costs.groupby("Rolling_datetime_index")[
        ["COSTS_Offer"]
    ].sum()
    aggregator_offer_costs = aggregator_offer_costs.iloc[
        : -aggregator_prediction_periods + 1
    ]

    """ _____________________________________________________________  DATA AGG OFFER COSTS DATETIME """

    aggregator_order_costs = (
        Aggregator.data[Aggregator.name]["COSTS_Order"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    aggregator_order_costs = aggregator_order_costs.loc[:, ["COSTS_Order"]]

    aggregator_order_costs["Rolling_datetime_index"] = rolling_datetimes_index  #
    aggregator_order_costs.sort_values("Rolling_datetime_index", inplace=True)

    aggregator_order_costs = aggregator_order_costs.groupby("Rolling_datetime_index")[
        ["COSTS_Order"]
    ].sum()
    aggregator_order_costs = aggregator_order_costs.iloc[
        : -aggregator_prediction_periods + 1
    ]

    """ _____________________________________________________________ DATA FDP IMBALANCES DATA """

    FDP_imbalances = FDP.imbalances_data["imbalances"][
        : -aggregator_prediction_periods + 1
    ]
    FDP_imbalances_market_prices = FDP.imbalances_data["market_prices"]
    FDP_deviation_price_down = FDP.imbalances_data["deviation_price_down"] * -1
    FDP_deviation_price_up = FDP.imbalances_data["deviation_price_up"]
    """ _____________________________________________________________ DATA FDP REQUEST PER PERIOD """

    fdp_requests = (
        FDP.data[FDP.name]["Request"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    fdp_requests.set_index("Time", inplace=True)

    """ _____________________________________________________________ DATA FDP ORDER PER PERIOD """

    fdp_orders = (
        FDP.data[FDP.name]["Order"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    fdp_orders.set_index("Time", inplace=True)

    """ _____________________________________________________________ DATA FDP OFFER PER DATETIME """

    fdp_requests_per_datetime = (
        FDP.data[FDP.name]["Request"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    fdp_requests_per_datetime = fdp_requests_per_datetime.loc[:, ["Request"]]

    fdp_requests_per_datetime["Rolling_datetime_index"] = rolling_datetimes_index  #
    fdp_requests_per_datetime.sort_values("Rolling_datetime_index", inplace=True)

    fdp_requests_per_datetime = fdp_requests_per_datetime.groupby(
        "Rolling_datetime_index"
    )[["Request"]].sum()
    fdp_requests_per_datetime = fdp_requests_per_datetime.iloc[
        : -fdp_prediction_periods + 1
    ]

    """ _____________________________________________________________ DATA FDP OFFER PER DATETIME """

    fdp_offers_per_datetime = (
        FDP.data[FDP.name]["Offer"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    fdp_offers_per_datetime = fdp_offers_per_datetime.loc[:, ["Offer"]]

    fdp_offers_per_datetime["Rolling_datetime_index"] = rolling_datetimes_index  #
    fdp_offers_per_datetime.sort_values("Rolling_datetime_index", inplace=True)

    fdp_offers_per_datetime = fdp_offers_per_datetime.groupby("Rolling_datetime_index")[
        ["Offer"]
    ].sum()
    fdp_offers_per_datetime = fdp_offers_per_datetime.iloc[
        : -fdp_prediction_periods + 1
    ]

    """ _____________________________________________________________ DATA FDP ORDER PER DATETIME """

    fdp_orders_per_datetime = (
        FDP.data[FDP.name]["Order"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    fdp_orders_per_datetime = fdp_orders_per_datetime.loc[:, ["Order"]]

    fdp_orders_per_datetime["Rolling_datetime_index"] = rolling_datetimes_index  #
    fdp_orders_per_datetime.sort_values("Rolling_datetime_index", inplace=True)

    fdp_orders_per_datetime = fdp_orders_per_datetime.groupby("Rolling_datetime_index")[
        ["Order"]
    ].sum()
    fdp_orders_per_datetime = fdp_orders_per_datetime.iloc[
        : -fdp_prediction_periods + 1
    ]

    """ _____________________________________________________________ DATA FDP DEVIATION PER DATETIME """

    fdp_deviation_per_datetime = (
        FDP.data[FDP.name]["Deviation"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    fdp_deviation_per_datetime = fdp_deviation_per_datetime.loc[:, ["Deviation"]]

    fdp_deviation_per_datetime["Rolling_datetime_index"] = rolling_datetimes_index  #
    fdp_deviation_per_datetime.sort_values("Rolling_datetime_index", inplace=True)

    fdp_deviation_per_datetime = fdp_deviation_per_datetime.groupby(
        "Rolling_datetime_index"
    )[["Deviation"]].sum()
    fdp_deviation_per_datetime = fdp_deviation_per_datetime.iloc[
        : -fdp_prediction_periods + 1
    ]

    """ _____________________________________________________________ STATS VARIABLES"""

    # Collect data for stats
    sum_derivative_min = 0
    sum_derivative_max = 0
    sum_generation_real = 0
    sum_generation_max = 0
    sum_load_real = 0
    sum_load_max = 0

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                     EMS DATA                                    """
    """_________________________________________________________________________________"""
    """                                                                                 """

    for enum, name in enumerate(scenario.EMS_names):

        """ _____________________________________________________________ EMS PRICES """

        EMS[name]["Purchase_price"] = scenario.EMS[enum].purchase_tariff

        EMS[name]["Feedin_price"] = scenario.EMS[enum].feedin_tariff

        EMS[name]["Feedin_price"] = scenario.EMS[enum].flexibility_tariff

        """ _____________________________________________________________ EMS REQUEST """

        EMS[name]["Request"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "Request"]
        ]

        """ _____________________________________________________________ EMS OFFER """

        EMS[name]["Offer"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "Offer"]
        ]

        """ _____________________________________________________________ EMS ORDER """

        EMS[name]["Order"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "Order"]
        ]

        """ _____________________________________________________________ EMS DEVIATION """

        EMS[name]["Deviation"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "Deviation"]
        ]

        """ _____________________________________________________________ EMS COSTS """

        EMS[name]["COSTS_Base_EC"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "COSTS_Base_EC"]
        ]

        EMS[name]["COSTS_Offer_EC"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "COSTS_Offer_EC"]
        ]

        EMS[name]["COSTS_Deviation"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, 1], IDX[name, "COSTS_Deviation"]
        ]

        EMS[name]["COSTS_Offer"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "COSTS_Offer"]
        ]

        EMS[name]["Commitment"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "Commitment"]
        ]

        EMS[name]["Realised"] = scenario.EMS[enum].data.loc[
            IDX[: scenario.last_simulated_index, :], IDX[name, "Realised"]
        ]

        # Recast EMS dict to DataFrame
        EMS[name] = DataFrame.from_dict(EMS[name])

        # Assign rolling datetime index
        EMS[name]["Rolling_datetime_index"] = rolling_datetimes_index  #
        EMS[name].set_index(["Rolling_datetime_index"], inplace=True)

        """ _____________________________________________________________ EMS GENERATOR """

        if "Generator" in EMS_device_names[name]:

            EMS_devices[name]["Generator_BL"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, :], IDX[name, "Generator_BL"]
            ]

            EMS_devices[name]["Generator_BL"] = EMS_devices[name][
                "Generator_BL"
            ].reset_index()
            EMS_devices[name]["Generator_BL"].index = rolling_datetimes_index
            EMS_devices[name]["Generator_BL"].columns = EMS_devices[name][
                "Generator_BL"
            ].columns.droplevel(0)
            EMS_devices[name]["Generator_BL"] = EMS_devices[name]["Generator_BL"][
                "Generator_BL"
            ]

            device_baseline = where(
                EMS[name]["COSTS_Offer"].isnull() == False,
                EMS_devices[name]["Generator_BL"],
                nan,
            )

            EMS_devices[name]["Generator_BL"] = Series(
                index=rolling_datetimes_index, data=device_baseline
            )
            EMS_devices[name]["Generator_BL"] = (
                EMS_devices[name]["Generator_BL"]
                .groupby(rolling_datetimes_index)
                .last()
            )

            EMS_devices[name]["Generator_BL_REAL"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, 1], IDX[name, "Generator_BL"]
            ]

            EMS_devices[name]["Generator_FLEX_REAL"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, 1], IDX[name, "Generator_FLEX"]
            ]

            """_____________________________________________________________ GENERATOR CONSTRAINTS"""

            EMS_devices[name]["Generator_derivative_min"] = (
                scenario.EMS[enum]
                .device_constraints_storage["Generator"]["derivative min"]
                .loc[: scenario.last_simulated_index]
            )

            EMS_devices[name]["Generator_derivative_max"] = (
                scenario.EMS[enum]
                .device_constraints_storage["Generator"]["derivative max"]
                .loc[: scenario.last_simulated_index]
            )

            EMS_devices_shifts[name]["Generator"] = (
                EMS_devices[name]["Generator_FLEX_REAL"]
                - EMS_devices[name]["Generator_BL_REAL"]
            )
            EMS_devices_shifts[name]["Generator"] = EMS_devices_shifts[name][
                "Generator"
            ].droplevel(1)
            EMS_baselines_total[name] += EMS_devices[name]["Generator_derivative_min"]

            """ _____________________________________________________________ EMS GENERATOR SHIFTS """

            EMS_devices_shifts[name]["Generator"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, :],
                IDX[name, ["Generator_BL", "Generator_FLEX", "Order"]],
            ]

            EMS_devices_shifts[name]["Generator"].columns = EMS_devices_shifts[name][
                "Generator"
            ].columns.droplevel(0)
            EMS_devices_shifts[name]["Generator"] = EMS_devices_shifts[name][
                "Generator"
            ].reset_index()
            EMS_devices_shifts[name]["Generator"].set_index(
                "Time", inplace=True
            )  
            EMS_devices_shifts[name]["Generator"]["Flexibility"] = 0

            datetimes_with_orders = EMS_devices_shifts[name]["Generator"][
                isna(EMS_devices_shifts[name]["Generator"]["Order"]) == False
            ].copy()
            datetimes_with_orders.set_index("Period", append=True, inplace=True)

            if len(datetimes_with_orders) != 0:
                datetimes_with_orders[
                    "Generator_BL_FLEX_diff_at_order"
                ] = datetimes_with_orders.apply(
                    lambda row: get_difference_with_sign(
                        first_value=row["Generator_FLEX"],
                        second_value=row["Generator_BL"],
                    ),
                    axis=1,
                    result_type="expand",
                )

                for _enum, date in enumerate(
                    datetimes_with_orders.index.get_level_values(level=0).unique()
                ):

                    for period_index in datetimes_with_orders.loc[date].index:

                        try:

                            values_at_order_period = EMS_devices_shifts[name][
                                "Generator"
                            ]["Generator_FLEX"].loc[date]
                            values_at_fulfillment_period = EMS_devices_shifts[name][
                                "Generator"
                            ]["Generator_FLEX"].loc[
                                date + scenario.resolution * (period_index - 1)
                            ]

                            realised_diff_from_order = get_difference_with_sign(
                                first_value=values_at_fulfillment_period.iloc[0],
                                second_value=values_at_order_period.iloc[
                                    period_index - 1
                                ],
                            )

                            generator_BL_FLEX_diff_at_order = datetimes_with_orders.loc[
                                # IDX[date, period_index],
                                IDX[date, period_index],
                                IDX["Generator_BL_FLEX_diff_at_order"]
                                # date + scenario.resolution * (period_index-1)
                            ]
                            EMS_devices_shifts[name]["Generator"]["Flexibility"].loc[
                                values_at_fulfillment_period.index[0]
                            ] = (
                                generator_BL_FLEX_diff_at_order
                                + realised_diff_from_order
                            )

                            EMS_flexibility[name].loc[
                                values_at_fulfillment_period.index[0]
                            ] += (
                                generator_BL_FLEX_diff_at_order
                                + realised_diff_from_order
                            )

                        except:
                            pass

            """_____________________________________________________________ GENERATOR STATS"""

            sum_derivative_min += EMS_devices[name]["Generator_derivative_min"].min()
            sum_generation_max += EMS_devices[name]["Generator_derivative_min"].sum()
            sum_generation_real += EMS_devices[name]["Generator_FLEX_REAL"].sum()

        """ _____________________________________________________________ EMS STORAGE """

        if "Storage" in EMS_device_names[name]:

            """ _____________________________________________________________ EMS STORAGE BASELINE """

            EMS_devices[name]["Storage_BL"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, :], IDX[name, "Storage_BL"]
            ]

            EMS_devices[name]["Storage_BL"] = EMS_devices[name][
                "Storage_BL"
            ].reset_index()
            EMS_devices[name]["Storage_BL"].index = rolling_datetimes_index
            EMS_devices[name]["Storage_BL"].columns = EMS_devices[name][
                "Storage_BL"
            ].columns.droplevel(0)
            EMS_devices[name]["Storage_BL"] = EMS_devices[name]["Storage_BL"][
                "Storage_BL"
            ]

            device_baseline = where(
                EMS[name]["COSTS_Offer"].isnull() == False,
                EMS_devices[name]["Storage_BL"],
                nan,
            )

            device_baseline = EMS_devices[name]["Storage_BL"]

            EMS_devices[name]["Storage_BL"] = DataFrame(
                index=rolling_datetimes_index,
                columns=["device_baseline"],
                data=device_baseline,
            )
            EMS_devices[name]["Storage_BL"] = Series(
                index=rolling_datetimes_index, data=device_baseline
            )
            EMS_devices[name]["Storage_BL"] = (
                EMS_devices[name]["Storage_BL"].groupby(rolling_datetimes_index).first()
            )

            EMS_devices[name]["Storage_BL_REAL"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, 1], IDX[name, "Storage_BL"]
            ]

            storage_baseline_values = DataFrame(
                index=EMS_devices[name]["Storage_BL_REAL"].index.get_level_values(0),
                data=EMS_devices[name]["Storage_BL_REAL"].values,
            )
            storage_baseline_values = EMS_devices[name]["Storage_BL_REAL"].droplevel(1)

            EMS_baselines_total[name] += storage_baseline_values

            """ _____________________________________________________________ EMS STORAGE FLEXIBLEPROFILE """

            EMS_devices[name]["Storage_FLEX_REAL"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, 1], IDX[name, "Storage_FLEX"]
            ]

            """ _____________________________________________________________ EMS STORAGE SHIFTS """

            EMS_devices_shifts[name]["Storage"] = scenario.EMS[enum].data.loc[
                IDX[: scenario.last_simulated_index, :],
                IDX[name, ["Storage_BL", "Storage_FLEX", "Order"]],
            ]

            EMS_devices_shifts[name]["Storage"].columns = EMS_devices_shifts[name][
                "Storage"
            ].columns.droplevel(0)
            EMS_devices_shifts[name]["Storage"] = EMS_devices_shifts[name][
                "Storage"
            ].reset_index()
            EMS_devices_shifts[name]["Storage"].set_index(
                "Time", inplace=True
            ) 
            EMS_devices_shifts[name]["Storage"]["Flexibility"] = 0

            datetimes_with_orders = EMS_devices_shifts[name]["Storage"][
                isna(EMS_devices_shifts[name]["Storage"]["Order"]) == False
            ].copy()
            datetimes_with_orders.set_index("Period", append=True, inplace=True)


            datetimes_with_orders[
                "Storage_BL_FLEX_diff_at_order"
            ] = datetimes_with_orders.apply(
                lambda row: get_difference_with_sign(
                    first_value=row["Storage_FLEX"], second_value=row["Storage_BL"]
                ),
                axis=1,
                result_type="expand",
            )

            for _enum, date in enumerate(
                datetimes_with_orders.index.get_level_values(level=0).unique()
            ):

                for period_index in datetimes_with_orders.loc[date].index:
                    try:

                        values_at_order_period = EMS_devices_shifts[name]["Storage"][
                            "Storage_FLEX"
                        ].loc[date]
                        values_at_fulfillment_period = EMS_devices_shifts[name][
                            "Storage"
                        ]["Storage_FLEX"].loc[
                            date + scenario.resolution * (period_index - 1)
                        ]
                        realised_diff_from_order = get_difference_with_sign(
                            first_value=values_at_fulfillment_period.iloc[0],
                            second_value=values_at_order_period.iloc[period_index - 1],
                        )

                        generator_BL_FLEX_diff_at_order = datetimes_with_orders.loc[
                            IDX[date, period_index],
                            IDX["Storage_BL_FLEX_diff_at_order"]
                        ]
                        EMS_devices_shifts[name]["Storage"]["Flexibility"].loc[
                            values_at_fulfillment_period.index[0]
                        ] = (generator_BL_FLEX_diff_at_order + realised_diff_from_order)

                        EMS_flexibility[name].loc[
                            values_at_fulfillment_period.index[0]
                        ] += (
                            generator_BL_FLEX_diff_at_order + realised_diff_from_order
                        )

                    except:
                        pass

            """_____________________________________________________________ STORAGE CONSTRAINTS"""

            EMS_devices[name]["Storage_derivative_min"] = (
                scenario.EMS[enum]
                .device_constraints["Storage"]["derivative min"]
                .loc[: scenario.last_simulated_index]
            )

            EMS_devices[name]["Storage_derivative_max"] = (
                scenario.EMS[enum]
                .device_constraints["Storage"]["derivative max"]
                .loc[: scenario.last_simulated_index]
            )

            """_____________________________________________________________ STORAGE STATS"""

            # Collect data for stats
            sum_derivative_min += EMS_devices[name]["Storage_derivative_max"].min()
            sum_derivative_max += EMS_devices[name]["Storage_derivative_max"].max()

            EMS_devices[name]["Storage_INIT_CONSTRAINTS"] = scenario.EMS[
                enum
            ].initial_storage_constraints

        EMS_realised[name] = (
            scenario.EMS[enum]
            .data[name]["Realised"]
            .loc[: scenario.last_simulated_index, 1]
            .reset_index()
        )
        EMS_realised[name].set_index(
            "Time", inplace=True
        )  

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                  FIGURE AND LAYOUT                              """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """_____________________________________________________________ FIGURE """

    vertical_spacing_factor = 0
    layout_height_factor = 1

    for ems in scenario.EMS_names:
        layout_height_factor += 0
        vertical_spacing_factor += 0.05

    fig = make_subplots(
        rows=rows,
        cols=columns,
        column_widths=[1 for x in range(columns)],
        row_heights=[0.5 if x == 0 else 1 for x in range(0, rows)],
        subplot_titles=subplot_titles,
        vertical_spacing=0.015 * (1 - vertical_spacing_factor),
    )

    """ _____________________________________________________________ LAYOUT """

    layout_width = 1450
    layout_height = 8850
    layout = go.Layout(
        width=layout_width,
        height=layout_height * layout_height_factor,
        bargap=0,
        barmode="relative",
        title=go.layout.Title(
            text="{}".format(scenario.name.capitalize(),),
            font=dict(size=24, family="Roboto"),
            xref="paper",
            x=0.5,
            y=1,
        ),
        hoverlabel=dict(
            bgcolor="black", font=dict(color="white") 
        ),
        legend=dict(
            x=0.5,
            y=1,
            traceorder="grouped",
            yanchor="top",
            xanchor="center",
            orientation="h",
        ),
        margin=go.layout.Margin(
            t=30,
        ),
    )

    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=28, color="black", family="Roboto")

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                     TABLE                                       """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________  SUMMARY TABLE """

    summary_column = [
        "Scenario name",
        "Description",
        "Plot start time",
        "Simulation start time",
        "Simulation horizon start",
        "Simulation horizon end",
        "Simulation resolution",
        "Episodes",
        "Episode duration",
        "Horizon FDP",
        "Horizon Aggregator",
    ]

    input_data_table = go.Table(
        domain=dict(x=[0, 1], y=[0.98, 1]),
        columnorder=[1, 2, 3, 4],
        columnwidth=[
            layout_width * 0.2,
            layout_width * 0.3,
            layout_width * 0.3,
            layout_width * 0.2,
        ],
        header=dict(
            values=["<b></b>", "<b>Scenario inputs</b>", "<b>Values</b>", "<b></b>",],
            line_color="darkslategray",
            # fill_color=headerColor,
            height=30,
            align=["center"],
            font=dict(color="black", size=14, family="Roboto"),
        ),
        cells=dict(
            values=[
                # Empty left column
                [],
                summary_column,
                [
                    scenario.name,
                    scenario.description,
                    datetime.now().strftime("%d.%m.%Y | %H:%M"),
                    scenario.execution_start_time.strftime("%d.%m.%Y | %H:%M"),
                    horizon[0].strftime("%d.%m.%Y | %H:%M"),
                    horizon[-1].strftime("%d.%m.%Y | %H:%M"),
                    str(scenario.resolution)[:-3],
                    scenario.episodes,
                    str(scenario.episode_duration)[:-3],
                    str(fdp_prediction_periods * scenario.resolution)[:-3],
                    str(aggregator_prediction_periods * scenario.resolution)[:-3],
                ],
                # Empty right column
                [],
            ],
            font=dict(size=16, family="Roboto"),
            align="center",
            height=35,
        ),
    )

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                     TRACES                                      """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________  EMS PRICES UP """

    for enum, name in enumerate(scenario.EMS_names):

        fig.append_trace(
            go.Scatter(
                x=horizon,
                y=EMS[name]["Purchase_price"],
                mode="lines+markers",
                name=name,
                showlegend=False,
                line=dict(color=ems_colors_1[enum], dash="solid", width=line_width),
                xaxis="x" + str(up_prices_row),
                yaxis="y" + str(up_prices_row),
            ),
            row=up_prices_row,
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                x=horizon,
                y=EMS[name]["Feedin_price"],
                mode="lines+markers",
                name=name,
                showlegend=False,
                line=dict(color=ems_colors_1[enum], dash="solid", width=line_width),
                xaxis="x" + str(down_prices_row),
                yaxis="y" + str(down_prices_row),
            ),
            row=down_prices_row,
            col=1,
        )

    """ _____________________________________________________________  FDP IMBALANCE MP """

    fig.append_trace(
        go.Scatter(
            x=horizon,
            y=FDP_imbalances_market_prices,
            line=dict(color="purple", dash="solid", width=line_width),
            name="Imbalance MP",
            mode="lines+markers",
            showlegend=False,
            xaxis="x" + str(fdp_imbalances_market_prices_row),
            yaxis="y" + str(fdp_imbalances_market_prices_row),
            marker=dict(
                symbol="diamond",
                size=marker_size_1,
                line=dict(color="black", width=marker_line_width),
                color="red", 
                showscale=False,
            ),
        ),
        row=fdp_imbalances_market_prices_row,
        col=1,
    )

    """ _____________________________________________________________  FDP/EMS PRICES UP """

    fig.append_trace(
        go.Scatter(
            x=horizon,
            y=FDP_deviation_price_up,
            mode="lines+markers",
            name="FDP",
            showlegend=False,
            line=dict(color="magenta", dash="solid", width=line_width),
            xaxis="x" + str(up_prices_row),
            yaxis="y" + str(up_prices_row),
            marker=dict(
                symbol="diamond",
                size=marker_size_1,
                line=dict(color="black", width=marker_line_width),
                color="red",
                showscale=False,
            ),
        ),
        row=up_prices_row,
        col=1,
    )

    """ _____________________________________________________________  FDP/EMS PRICES DOWN """

    fig.append_trace(
        go.Scatter(
            x=horizon,
            y=FDP_deviation_price_down,
            mode="lines+markers",
            name="FDP",
            showlegend=False,
            line=dict(color="olive", dash="solid", width=line_width),
            xaxis="x" + str(down_prices_row),
            yaxis="y" + str(down_prices_row),
        ),
        row=down_prices_row,
        col=1,
    )

    """ _____________________________________________________________  FDP IMBALANCES"""

    fig.append_trace(
        go.Bar(
            x=FDP_imbalances[FDP_imbalances > 0].index,
            y=FDP_imbalances[FDP_imbalances > 0],
            showlegend=False,
            name="Imbalance",
            xaxis="x" + str(fdp_imbalances_row),
            yaxis="y" + str(fdp_imbalances_row),
            marker=dict(
                color="grey",
                opacity=1,
                line=dict(color="black", width=marker_line_width),
            ),
        ),
        row=fdp_imbalances_row,
        col=1,
    )

    fig.append_trace(
        go.Bar(
            x=FDP_imbalances[FDP_imbalances < 0].index,
            y=FDP_imbalances[FDP_imbalances < 0],
            showlegend=False,
            name="Imbalance",
            xaxis="x" + str(fdp_imbalances_row),
            yaxis="y" + str(fdp_imbalances_row),
            marker=dict(
                color="lightgrey",
                opacity=1,
                line=dict(color="black", width=marker_line_width),
            ),
        ),
        row=fdp_imbalances_row,
        col=1,
    )

    """ _____________________________________________________________  FDP REQUEST """
    fig.append_trace(
        go.Bar(
            x=fdp_requests.index,
            y=fdp_requests["Request"],
            name="Request",
            showlegend=False,
            # offset=0.5,
            xaxis="x" + str(fdp_requests_row),
            yaxis="y" + str(fdp_requests_row),
            marker=dict(
                color=repeating_periods_index,
                colorscale="jet", 
                line=dict(color="black", width=marker_line_width),
            ),
        ),
        row=fdp_requests_row,
        col=1,
    )

    """ _____________________________________________________________  FDP ORDER """

    fig.append_trace(
        go.Bar(
            x=fdp_orders.index,
            y=fdp_orders["Order"],
            name="Order",
            showlegend=False,
            xaxis="x" + str(fdp_orders_row),
            yaxis="y" + str(fdp_orders_row),
            marker=dict(
                color=repeating_periods_index,
                colorscale="jet",
                line=dict(color="black", width=marker_line_width),
            ),
        ),
        row=fdp_orders_row,
        col=1,
    )

    """ _____________________________________________________________  AGG REQUEST/ORDER PROFILE """
    for enum, name in enumerate(scenario.EMS_names):
        fig.append_trace(
            go.Bar(
                x=horizon,
                y=EMS[name].groupby("Rolling_datetime_index")["Order"].sum(),
                name=name,
                showlegend=False,
                marker=dict(
                    color=ems_colors_1[enum],
                    line=dict(color="black", width=marker_line_width),
                ),
                xaxis="x" + str(aggregator_order_per_EMS_over_total_request_row),
                yaxis="y" + str(aggregator_order_per_EMS_over_total_request_row),
            ),
            row=aggregator_order_per_EMS_over_total_request_row,
            col=1,
        )

    """ _____________________________________________________________  AGG EMS DEVIATION PROFILE """

    for enum, name in enumerate(scenario.EMS_names):

        fig.append_trace(
            go.Bar(
                x=horizon,
                y=EMS[name].groupby("Rolling_datetime_index")["Deviation"].sum(),
                name=name,
                showlegend=False,
                xaxis="x" + str(aggregator_deviation_per_EMS_over_total_order_row),
                yaxis="y" + str(aggregator_deviation_per_EMS_over_total_order_row),
                marker=dict(
                    color=ems_colors_1[enum],
                    line=dict(color="black", width=marker_line_width),
                ),
            ),
            row=aggregator_deviation_per_EMS_over_total_order_row,
            col=1,
        )

    """ _____________________________________________________________  EMS DEVICES POWER PROFILE"""

    for enum, name in enumerate(scenario.EMS_names):

        for device in EMS_device_names[name]:

            if "Storage" in device:

                cumulated_storage_loads = dict()
                storage_realised = EMS_devices[name]["Storage_FLEX_REAL"].droplevel(1)

                """ _____________________________________________________________ SOC """
                fig.append_trace(
                    go.Scatter(
                        x=horizon,
                        y=EMS_devices[name]["Storage_INIT_CONSTRAINTS"]["max"],
                        mode="lines",
                        showlegend=False,
                        name="max",
                        line=dict(color="red", width=marker_line_width + 1),
                        xaxis="x" + str(ems_storage_SOC_rows[name]),
                        yaxis="y" + str(ems_storage_SOC_rows[name]),
                    ),
                    row=ems_storage_SOC_rows[name],
                    col=1,
                )

                fig.append_trace(
                    go.Scatter(
                        x=horizon,
                        y=EMS_devices[name]["Storage_INIT_CONSTRAINTS"]["min"],
                        mode="lines",
                        showlegend=False,
                        name="min",
                        line=dict(color="green", width=marker_line_width + 1),
                        xaxis="x" + str(ems_storage_SOC_rows[name]),
                        yaxis="y" + str(ems_storage_SOC_rows[name]),
                    ),
                    row=ems_storage_SOC_rows[name],
                    col=1,
                )

                for daily_load in storage_realised.groupby(storage_realised.index.date):

                    fig.append_trace(
                        go.Scatter(
                            x=daily_load[1].index,
                            y=daily_load[1].cumsum(),
                            mode="lines+markers",
                            showlegend=False,
                            name="SOC",
                            line=dict(
                                color="black", width=marker_line_width, dash="solid"
                            ),
                            xaxis="x" + str(ems_storage_SOC_rows[name]),
                            yaxis="y" + str(ems_storage_SOC_rows[name]),
                            marker=dict(
                                symbol="circle",
                                size=marker_size_1,
                                color="cyan", 
                                line=dict(color="black", width=marker_line_width),
                                colorscale="jet",
                                showscale=False,
                            ),
                        ),
                        row=ems_storage_SOC_rows[name],
                        col=1,
                    )

                fig["layout"]["yaxis" + str(ems_storage_SOC_rows[name])].update(
                    title=yaxis_energy
                )
            else:

                try:
                    fig.append_trace(
                        go.Scatter(
                            x=horizon,
                            y=EMS_devices[name][device + "_derivative_min"],
                            mode="lines",
                            name="Constraint min",
                            showlegend=False,
                            line=dict(
                                color="black", width=marker_line_width, dash="dot"
                            ),
                            xaxis="x"
                            + str(ems_device_power_profile_rows[name][device]),
                            yaxis="y"
                            + str(ems_device_power_profile_rows[name][device]),
                        ),
                        row=ems_device_power_profile_rows[name][device],
                        col=1,
                    )

                    fig.append_trace(
                        go.Scatter(
                            x=horizon,
                            y=EMS_devices[name][device + "_derivative_max"],
                            mode="lines",
                            showlegend=False,
                            name="Constraint min",
                            line=dict(color="black", width=0, dash="dot"),
                            xaxis="x"
                            + str(ems_device_power_profile_rows[name][device]),
                            yaxis="y"
                            + str(ems_device_power_profile_rows[name][device]),
                        ),
                        row=ems_device_power_profile_rows[name][device],
                        col=1,
                    )

                    fig["layout"][
                        "yaxis" + str(ems_device_power_profile_rows[name][device])
                    ].update(title=yaxis_power)

                except:
                    pass

                fig.append_trace(
                    go.Scatter(
                        x=horizon,
                        y=EMS_devices[name][device + "_FLEX_REAL"],
                        mode="lines+markers",
                        showlegend=False,
                        name="Realised",
                        line=dict(color="black", width=marker_line_width),
                        xaxis="x" + str(ems_device_power_profile_rows[name][device]),
                        yaxis="y" + str(ems_device_power_profile_rows[name][device]),
                        marker=dict(
                            symbol="circle",
                            size=marker_size_1,
                            color="black",  
                            line=dict(color="black", width=marker_line_width),
                            colorscale="jet", 
                            showscale=False,
                        ),
                    ),
                    row=ems_device_power_profile_rows[name][device],
                    col=1,
                )

    """ _____________________________________________________________  DEVICES SHIFTS """
    for enum, name in enumerate(scenario.EMS_names):

        for device in EMS_device_names[name]:

            try:
                shifts = (
                    EMS_devices_shifts[name][device]
                    .groupby(EMS_devices_shifts[name][device].index)["Flexibility"]
                    .mean()
                )  
                upshifts = shifts[shifts > 0]
                downshifts = shifts[shifts < 0]
                fig.append_trace(
                    go.Bar(
                        x=upshifts.index,
                        y=upshifts,
                        offset=0.5,
                        name="Upshift",
                        showlegend=False,
                        xaxis="x" + str(ems_shifts_per_device_rows[name][device]),
                        yaxis="y" + str(ems_shifts_per_device_rows[name][device]),
                        marker=dict(
                            color=upshift_color,
                            line=dict(
                                color=upshift_border_color, width=marker_line_width
                            ),
                        ),
                    ),
                    row=ems_shifts_per_device_rows[name][device],
                    col=1,
                )

                fig.append_trace(
                    go.Bar(
                        x=downshifts.index,
                        y=downshifts,
                        offset=0.5,
                        name="Downshift",
                        showlegend=False,
                        xaxis="x" + str(ems_shifts_per_device_rows[name][device]),
                        yaxis="y" + str(ems_shifts_per_device_rows[name][device]),
                        marker=dict(
                            color=downshift_color,
                            line=dict(
                                color=downshift_border_color, width=marker_line_width
                            ),
                        ),
                    ),
                    row=ems_shifts_per_device_rows[name][device],
                    col=1,
                )

                fig["layout"][
                    "yaxis" + str(ems_shifts_per_device_rows[name][device])
                ].update(title=yaxis_power)
            except:
                pass

    """ _____________________________________________________________  EMS FLEX """

    for enum, name in enumerate(scenario.EMS_names):

        flexibility = EMS_flexibility[name]
        flexibility_up = flexibility[flexibility > 0]
        flexibility_down = flexibility[flexibility < 0]

        fig.append_trace(
            go.Bar(
                x=flexibility_up.index,
                y=flexibility_up,
                name=name,
                showlegend=False,
                xaxis="x" + str(ems_flexibility_rows[name]),
                yaxis="y" + str(ems_flexibility_rows[name]),
                marker=dict(
                    color=ems_colors_075[enum],
                    line=dict(color="black", width=marker_line_width),
                ),
            ),
            row=ems_flexibility_rows[name],
            col=1,
        )

        fig.append_trace(
            go.Bar(
                x=flexibility_down.index,
                y=flexibility_down,
                name=name,
                showlegend=False,
                xaxis="x" + str(ems_flexibility_rows[name]),
                yaxis="y" + str(ems_flexibility_rows[name]),
                marker=dict(
                    color=ems_colors_075[enum],
                    line=dict(color="black", width=marker_line_width),
                ),
            ),
            row=ems_flexibility_rows[name],
            col=1,
        )

        fig["layout"]["yaxis" + str(ems_flexibility_rows[name])].update(
            title=yaxis_power
        )

        """ _____________________________________________________________  AGG FLEX """

        fig.append_trace(
            go.Bar(
                x=horizon,
                y=EMS_flexibility[name],
                name=name,
                showlegend=False,
                marker=dict(
                    color=ems_colors_1[enum],
                    line=dict(color="black", width=marker_line_width),
                ),
                xaxis="x" + str(aggregator_flexibility_row),
                yaxis="y" + str(aggregator_flexibility_row),
            ),
            row=aggregator_flexibility_row,
            col=1,
        )

    fig["layout"]["yaxis" + str(aggregator_flexibility_row)].update(title=yaxis_power)

    """ _____________________________________________________________  EMS REAL + BASELINE """
    flexibility_sum = 0
    for enum, name in enumerate(scenario.EMS_names):

        flexibility_sum += EMS_flexibility[name]

        fig.append_trace(
            go.Scatter(
                x=EMS_realised[name].index,
                y=EMS_realised[name]["Realised"],
                mode="lines+markers",
                name="Realised",
                showlegend=False,
                line=dict(color="black", width=marker_line_width),
                xaxis="x" + str(ems_realised_and_baseline_rows[name]),
                yaxis="y" + str(ems_realised_and_baseline_rows[name]),
            ),
            row=ems_realised_and_baseline_rows[name],
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                x=EMS_realised[name].index,
                y=EMS_realised[name]["Realised"]
                - EMS_flexibility[name], 
                mode="lines+markers",
                name="Baseline",
                showlegend=False,
                line=dict(color="red", width=marker_line_width, dash="dash"),
                xaxis="x" + str(ems_realised_and_baseline_rows[name]),
                yaxis="y" + str(ems_realised_and_baseline_rows[name]),
                # fill="tonext",
                marker=dict(
                    size=marker_size_1,
                    symbol="circle",
                    color="red",  
                    line=dict(color="black", width=marker_line_width),
                    showscale=False,
                ),
            ),
            row=ems_realised_and_baseline_rows[name],
            col=1,
        )

    """ _____________________________________________________________  AGG REAL + BASELINE """

    fig.append_trace(
        go.Scatter(
            x=aggregator_realised.index,
            y=aggregator_realised["Realised"],
            mode="lines+markers",
            name="Realised",
            showlegend=False,
            line=dict(color="black", width=marker_line_width),
            xaxis="x" + str(aggregator_realised_and_baseline_row),
            yaxis="y" + str(aggregator_realised_and_baseline_row),
        ),
        row=aggregator_realised_and_baseline_row,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=aggregator_realised.index,
            y=aggregator_realised["Realised"] - flexibility_sum,
            mode="lines+markers",
            name="Baseline",
            showlegend=False,
            line=dict(color="chocolate", width=marker_line_width, dash="dash"),
            xaxis="x" + str(aggregator_realised_and_baseline_row),
            yaxis="y" + str(aggregator_realised_and_baseline_row),
        ),
        row=aggregator_realised_and_baseline_row,
        col=1,
    )

    fig["layout"]["yaxis" + str(aggregator_realised_and_baseline_row)].update(
        title=yaxis_power
    )

    """ _____________________________________________________________  EMS EC COSTS """

    for enum, name in enumerate(scenario.EMS_names):

        fig.append_trace(
            go.Scatter(
                x=horizon,
                y=EMS[name].groupby("Rolling_datetime_index")["COSTS_Base_EC"].sum(),
                mode="lines",
                name="EC Base",
                showlegend=False,
                marker=dict(
                    symbol="circle",
                    color="black",
                    size=marker_size_1,
                    line=dict(color="black", width=marker_line_width),
                ),
                line=dict(color="black", width=marker_line_width, dash="dot"),
                xaxis="x" + str(ems_energy_contract_costs_rows[name]),
                yaxis="y" + str(ems_energy_contract_costs_rows[name]),
            ),
            row=ems_energy_contract_costs_rows[name],
            col=1,
        )

        fig.append_trace(
            go.Scatter(
                x=horizon,
                y=EMS[name].groupby("Rolling_datetime_index")["COSTS_Offer_EC"].sum(),
                mode="lines+markers",
                name="EC Offer",
                showlegend=False,
                fill="tonexty",
                fillcolor="lightcoral",
                marker=dict(
                    symbol="circle",
                    color="black",
                    size=marker_size_1,
                    line=dict(color="black", width=marker_line_width),
                ),
                line=dict(color="black", width=line_width + 1),
                xaxis="x" + str(ems_energy_contract_costs_rows[name]),
                yaxis="y" + str(ems_energy_contract_costs_rows[name]),
            ),
            row=ems_energy_contract_costs_rows[name],
            col=1,
        )

        fig["layout"]["yaxis" + str(ems_energy_contract_costs_rows[name])].update(
            title=yaxis_costs
        )

    """ _____________________________________________________________  EMS DEV COSTS """

    stacked = 0
    for enum, name in enumerate(scenario.EMS_names):

        devation_ems = EMS[name]["Deviation"]

        downward_deviation_costs = (
            devation_ems[devation_ems < 0] * FDP_deviation_price_down.loc[: horizon[-1]]
        )
        upward_deviation_costs = (
            devation_ems[devation_ems > 0] * FDP_deviation_price_up.loc[: horizon[-1]]
        )

        deviation_costs = downward_deviation_costs + upward_deviation_costs

        fig.append_trace(
            go.Scatter(
                x=horizon,
                y=deviation_costs,
                mode="lines+markers",
                name=name,
                showlegend=False,
                marker=dict(
                    symbol="square",
                    color="black",
                    size=marker_size_1,
                    line=dict(color="black", width=marker_line_width),
                ),
                line=dict(color=ems_colors_1[enum], width=line_width + 1),
                xaxis="x" + str(ems_dev_costs_row),
                yaxis="y" + str(ems_dev_costs_row),
            ),
            row=ems_dev_costs_row,
            col=1,
        )

        stacked += deviation_costs

    fig["layout"]["yaxis" + str(ems_dev_costs_row)].update(title=yaxis_costs)

    """ _____________________________________________________________  EMS OFFER COSTS """

    stacked = 0
    for enum, name in enumerate(scenario.EMS_names):

        fig.append_trace(
            go.Scatter(
                x=horizon,
                y=stacked
                + EMS[name].groupby("Rolling_datetime_index")["COSTS_Offer"].sum(),
                mode="lines",
                name=name,
                showlegend=False,
                marker=dict(
                    symbol="circle",
                    color="black",
                    size=marker_size_1,
                    line=dict(color="black", width=marker_line_width),
                ),
                line=dict(color=ems_colors_1[enum], width=line_width),
                xaxis="x" + str(ems_offer_costs_row),
                yaxis="y" + str(ems_offer_costs_row),
            ),
            row=ems_offer_costs_row,
            col=1,
        )

        stacked += EMS[name].groupby("Rolling_datetime_index")["COSTS_Offer"].sum()

    fig["layout"]["yaxis" + str(ems_offer_costs_row)].update(title=yaxis_costs)

    """ _____________________________________________________________  AGG OFFER COSTS """

    fig.append_trace(
        go.Scatter(
            x=horizon,
            y=aggregator_offer_costs.groupby("Rolling_datetime_index")[
                "COSTS_Offer"
            ].sum(),
            mode="lines+markers",
            name="Offer",
            showlegend=False,
            marker=dict(
                symbol="circle",
                size=marker_size_1,
                color="black",
                line=dict(color="black", width=marker_line_width),
            ),
            line=dict(color="black", width=line_width, dash="dot"),
            xaxis="x" + str(aggregator_offer_and_order_costs_row),
            yaxis="y" + str(aggregator_offer_and_order_costs_row),
        ),
        row=aggregator_offer_and_order_costs_row,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=horizon,
            y=aggregator_order_costs.groupby("Rolling_datetime_index")[
                "COSTS_Order"
            ].sum(),
            mode="lines+markers",
            showlegend=False,
            marker=dict(
                symbol="circle",
                size=marker_size_1,
                color="black",
                line=dict(color="black", width=marker_line_width),
            ),
            line=dict(color="black", width=line_width + 1),
            name="Order",
            xaxis="x" + str(aggregator_offer_and_order_costs_row),
            yaxis="y" + str(aggregator_offer_and_order_costs_row),
        ),
        row=aggregator_offer_and_order_costs_row,
        col=1,
    )

    fig["layout"]["yaxis" + str(aggregator_offer_and_order_costs_row)].update(
        title=yaxis_costs
    )

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                     SHAPES                                      """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________  COLORBARS """

    """COLORBAR TRACE REQUESTS"""

    plot_height = (
        fig["layout"]["yaxis" + str(fdp_requests_row)].domain[1]
        - fig["layout"]["yaxis" + str(fdp_requests_row)].domain[0]
    )

    fig.append_trace(
        go.Scatter(
            x=[None],
            y=[None],
            showlegend=False,
            xaxis="x" + str(fdp_requests_row),
            yaxis="y" + str(fdp_requests_row),
            mode="markers",
            marker=dict(
                colorscale="jet",
                showscale=True,
                cmin=0,
                cmax=fdp_prediction_periods - 1,
                colorbar=dict(
                    thickness=20,
                    yanchor="top",
                    lenmode="fraction",
                    len=plot_height,
                    nticks=fdp_prediction_periods,
                    y=fig["layout"]["yaxis" + str(fdp_requests_row)].domain[1],
                    ypad=0,
                ),
            ),
            hoverinfo="none",
        ),
        row=fdp_requests_row,
        col=1,
    )

    """ COLORBAR TRACE ORDERS"""

    plot_height = (
        fig["layout"]["yaxis" + str(fdp_orders_row)].domain[1]
        - fig["layout"]["yaxis" + str(fdp_orders_row)].domain[0]
    )
    plot_height

    fig.append_trace(
        go.Scatter(
            x=[None],
            y=[None],
            showlegend=False,
            xaxis="x" + str(fdp_orders_row),
            yaxis="y" + str(fdp_orders_row),
            mode="markers",
            marker=dict(
                colorscale="jet",
                showscale=True,
                cmin=0,
                cmax=fdp_prediction_periods - 1,
                colorbar=dict(
                    thickness=20,
                    yanchor="top",
                    lenmode="fraction",
                    len=plot_height,
                    nticks=fdp_prediction_periods,
                    y=fig["layout"]["yaxis" + str(fdp_orders_row)].domain[1],
                    ypad=0,
                ),
            ),
            hoverinfo="none",
        ),
        row=fdp_orders_row,
        col=1,
    )

    """ SHAPES """

    shapes = []

    for episode in range(0, scenario.episodes):

        episode_time = date_range(
            start=scenario.start + timedelta(days=episode),
            end=scenario.start
            + timedelta(days=episode)
            + scenario.episode_duration
            - scenario.episode_cut_off,
            freq=scenario.resolution,
        )

        episode_time_steps = list(range(1, len(episode_time) + 1))

        for date_time_index in episode_time:

            # Order
            shape = {
                "type": "rect",
                "x0": date_time_index - scenario.resolution / 2,
                "y0": 0,
                "x1": date_time_index + scenario.resolution / 2,
                "y1": FDP_imbalances.loc[date_time_index],
                "xref": "x" + str(aggregator_order_per_EMS_over_total_request_row),
                "yref": "y" + str(aggregator_order_per_EMS_over_total_request_row),
                "line": dict(
                    color="black",
                    width=marker_line_width,
                ),
            }
            shapes.append(shape)
    fig["layout"].update(shapes=shapes)

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                               LAYOUT UPDATE                                     """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________  YAXIS TITLES """

    fig["layout"]["yaxis" + str(fdp_imbalances_market_prices_row)].update(
        title=yaxis_price
    )
    fig["layout"]["yaxis" + str(up_prices_row)].update(title=yaxis_price)
    fig["layout"]["yaxis" + str(down_prices_row)].update(title=yaxis_price)
    fig["layout"]["yaxis" + str(fdp_imbalances_row)].update(title=yaxis_power)
    fig["layout"]["yaxis" + str(fdp_requests_row)].update(title=yaxis_power)
    fig["layout"]["yaxis" + str(fdp_orders_row)].update(title=yaxis_power)
    fig["layout"][
        "yaxis" + str(aggregator_order_per_EMS_over_total_request_row)
    ].update(title=yaxis_power)
    fig["layout"][
        "yaxis" + str(aggregator_deviation_per_EMS_over_total_order_row)
    ].update(title=yaxis_power)

    """ _____________________________________________________________  ADD TABLE """

    fig.add_traces(input_data_table)

    """ ________ UPDATE LAYOUT ________"""

    total_days = (horizon[-1] - horizon[0]).days
    if total_days <= 1:
        dtick_multiplicator = 4

    elif total_days > 1 and total_days <= 3:
        dtick_multiplicator = 4

    elif total_days > 3 and total_days <= 7:
        dtick_multiplicator = 8

    elif total_days > 7 and total_days <= 14:
        dtick_multiplicator = 12

    elif total_days > 14 and total_days <= 21:
        dtick_multiplicator = 16

    elif total_days > 21:
        dtick_multiplicator = 24

    elif total_days > 50:
        dtick_multiplicator = 12

    elif total_days > 99:
        dtick_multiplicator = 6

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        type="date",
        tickangle=-90,
        range=[horizon[0] - scenario.resolution, horizon[-1] + scenario.resolution],
        tickformat="%H:%M",
        automargin=True,
        dtick=900000 * dtick_multiplicator,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        tickfont=xaxis_tick_font,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        tickfont=yaxis_tick_font,
        title_font=yaxis_title_font,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        showgrid=True,
    )

    fig.update_layout(layout)

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                               STATS                                             """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________  STATS """

    if stats:

        for enum, name in enumerate(scenario.EMS_names):
            EMS[name].sort_values("Rolling_datetime_index", inplace=True)
            EMS[name] = EMS[name].loc[: scenario.last_simulated_index]

        ems_agents = scenario.EMS_names.copy()
        aggregation_index = []
        index_names = ["Entity", ""]
        column_names = ["Value", "Unit"]

        df = DataFrame(
            index=MultiIndex.from_product(
                iterables=[ems_agents, aggregation_index], names=index_names,
            ),
            columns=column_names,
        )

        imbalances = FDP_imbalances.loc[:]

        df.loc[IDX["DUMMY", ""], :] = ""

        df["Stats"] = "DUMMY"
        df.set_index("Stats", append=True, inplace=True)
        df = df.reorder_levels(["Stats", "Entity", ""])
        df = df.sort_index(level="Entity", ascending=False)

        """_____________________________________________________________ MODEL STATS"""
        df.loc[IDX["{}".format(" " * 1), "", ""], :] = "", ""
        df.loc[IDX["----------------- MODEL -----------------", "", ""], :] = "", ""
        df.loc[IDX["{}".format(" " * 2), "", ""], :] = "", ""

        df.loc[IDX["Simulation name", "Model", ""], :] = scenario.name, ""

        df.loc[IDX["Execution start time", "Model", "Time"], :] = (
            scenario.execution_start_time.strftime("%Y/%m/%d, %H:%M"),
            "",
        )
        df.loc[IDX["Simulation horizon start", "Model", "Time"], :] = (
            horizon[0].strftime("%Y/%m/%d, %H:%M"),
            "",
        )
        df.loc[IDX["Simulation horizon end", "Model", "Time"], :] = (
            horizon[-1].strftime("%Y/%m/%d, %H:%M"),
            "",
        )
        df.loc[IDX["Simulation resolution", "Model", "Time"], :] = (
            str(scenario.resolution)[:-3],
            "h",
        )
        df.loc[IDX["Simulation periods", "Model", "Periods"], :] = len(horizon), ""
        df.loc[IDX["Horizon FDP", "Model", "Time"], :] = (
            str(fdp_prediction_periods * scenario.resolution)[:-3],
            "h",
        )
        df.loc[IDX["Horizon Aggregator", "Model", "Time"], :] = (
            str(aggregator_prediction_periods * scenario.resolution)[:-3],
            "h",
        )

        df.loc[IDX["{}".format(" " * 3), "", ""], :] = "", ""
        df.loc[IDX["----------------- DEVICES -----------------", "", ""], :] = "", ""
        df.loc[IDX["{}".format(" " * 4), "", ""], :] = "", ""

        """_____________________________________________________________ GENERATION STATS """

        df.loc[IDX["Generation", "AGG", "Max power"], :] = (
            round(sum_generation_max, 2) * -1,
            scenario.units[0],
        )

        try:
            for name in scenario.EMS_names:
                df.loc[IDX["Generation", name, "Max power"], :] = (
                    round(EMS_devices[name]["Generator_derivative_min"].min() * -1, 2),
                    scenario.units[0],
                )

            df.loc[IDX["Generation", " ", ""], :] = "", ""
        except:
            pass

        df.loc[IDX["Generation", "AGG", "Max energy"], :] = (
            round(
                sum_generation_max * -1 * (scenario.resolution / timedelta(minutes=60)),
                2,
            ),
            scenario.units[1],
        )

        try:
            for name in scenario.EMS_names:
                df.loc[IDX["Generation", name, "Energy realised"], :] = (
                    round(
                        EMS_devices[name]["Generator_FLEX_REAL"].sum()
                        * -1
                        * (scenario.resolution / timedelta(minutes=60)),
                        2,
                    ),
                    scenario.units[1],
                )
            df.loc[IDX["Generation", " ", ""], :] = "", ""
        except:
            pass

        df.loc[IDX["Generation", "  ", ""], :] = "", ""

        """_____________________________________________________________ LOAD STATS """

        df.loc[IDX["Load", "AGG", "Max power"], :] = (
            round(sum_load_max, 2),
            scenario.units[0],
        )

        try:
            for name in scenario.EMS_names:
                df.loc[IDX["Load", name, "Energy realised"], :] = (
                    round(
                        EMS_devices[name]["Load_FLEX_REAL"].sum()
                        * -1
                        * (scenario.resolution / timedelta(minutes=60)),
                        2,
                    ),
                    scenario.units[1],
                )

            df.loc[IDX["Load", " ", ""], :] = "", ""

        except:
            pass

        df.loc[IDX["Load", "AGG", "Max energy"], :] = (
            round(sum_load_max * -1 * (scenario.resolution / timedelta(minutes=60)), 2),
            scenario.units[1],
        )

        try:
            for name in scenario.EMS_names:
                df.loc[IDX["Load", name, "Max energy"], :] = (
                    round(
                        EMS_devices[name]["Load_derivative_max"].sum()
                        * -1
                        * (scenario.resolution / timedelta(minutes=60)),
                        2,
                    ),
                    scenario.units[1],
                )
            df.loc[IDX["Load", " ", ""], :] = "", ""
        except:
            pass

        df.loc[IDX["Load", "AGG", "Energy realised"], :] = (
            round(
                sum_load_real * -1 * (scenario.resolution / timedelta(minutes=60)), 2
            ),
            scenario.units[1],
        )

        df.loc[IDX["Load", " ", ""], :] = "", ""

        """_____________________________________________________________ STORAGE STATS """

        try:
            for name in scenario.EMS_names:
                df.loc[IDX["Storage SOC constraint", name, "Sum"], :] = (
                    round(
                        EMS_devices[name]["Storage_INIT_CONSTRAINTS"]["min"].loc[
                            horizon[-1]
                        ],
                        2,
                    ),
                    scenario.units[1],
                )

                df.loc[IDX["Storage SOC final", name, "Sum"], :] = (
                    round(EMS_devices[name]["Storage_FLEX_REAL"].sum(), 2),
                    scenario.units[1],
                )

                df.loc[IDX["Storage max charging", name, "Sum"], :] = (
                    round(EMS_devices[name]["Storage_derivative_max"].max(), 2),
                    scenario.units[0],
                )

                df.loc[IDX["Storage max discharging", name, "Sum"], :] = (
                    round(EMS_devices[name]["Storage_derivative_min"].min(), 2),
                    scenario.units[0],
                )

                df.loc[IDX["Storage max discharging", " ", ""], :] = "", ""

        except:
            pass

        df.loc[IDX["Storage max discharging", " ", " "], :] = "", ""

        df.loc[IDX["Upward capacity", "AGG", "Max"], :] = (
            sum_derivative_max,
            scenario.units[0],
        )

        """_____________________________________________________________ FLEX UP STATS EMS"""

        try:
            for name in scenario.EMS_names:
                df.loc[IDX["Upward capacity", name, "Max"], :] = (
                    EMS_devices[name]["Generator_derivative_min"].min() * -1,
                    scenario.units[0],
                )
        except:
            pass

        df.loc[IDX["Upward capacity", " ", ""], :] = "", ""

        """_____________________________________________________________ FLEX DOWN STATS EMS"""

        df.loc[IDX["Downward capacity", "AGG", "Max"], :] = (
            sum_derivative_min * -1,
            scenario.units[0],
        )

        try:
            for name in scenario.EMS_names:
                df.loc[IDX["Downward capacity", name, "Max"], :] = (
                    EMS_devices[name]["Load_derivative_max"].max(),
                    scenario.units[0],
                )
        except:
            pass
        """_____________________________________________________________ IMBALANCE STATS"""
        df.loc[IDX["{}".format(" " * 5), "", ""], :] = "", ""
        df.loc[IDX["----------------- IMBALANCES --------------", "", ""], :] = "", ""
        df.loc[IDX["{}".format(" " * 6), "", ""], :] = "", ""

        df.loc[IDX["Imbalances", "FDP", "Sum"], :] = (
            abs(imbalances).sum(),
            scenario.units[0],
        )
        df.loc[IDX["Imbalances upwards", "FDP", "Sum"], :] = (
            imbalances[imbalances > 0].sum(),
            scenario.units[0],
        )
        df.loc[IDX["Imbalances downwards", "FDP", "Sum"], :] = (
            imbalances[imbalances < 0].sum(),
            scenario.units[0],
        )

        """_____________________________________________________________ REMAINING IMBALANCES STATS """

        _fdp_orders_per_datetime = fdp_orders_per_datetime.reset_index()
        _fdp_orders_per_datetime = _fdp_orders_per_datetime["Order"]

        _fdp_deviation_per_datetime = fdp_deviation_per_datetime.reset_index()
        _fdp_deviation_per_datetime = _fdp_deviation_per_datetime["Deviation"]

        remaining_imbalances = (
            abs(imbalances).sum()
            - abs(_fdp_orders_per_datetime).sum()
            + abs(_fdp_deviation_per_datetime).sum()
        )

        fulfilled_requests = (
            abs(_fdp_orders_per_datetime).sum() - abs(_fdp_deviation_per_datetime).sum()
        )

        df.loc[IDX["Remaining imbalances", "FDP", "Sum"], :] = (
            remaining_imbalances,
            scenario.units[0],
        )
        df.loc[IDX["Fulfilled requests", "FDP", "Sum"], :] = (
            fulfilled_requests,
            scenario.units[0],
        )


        """_____________________________________________________________ REQUESTED FLEX STATS """

        df.loc[
            IDX[
                "___________________________________________", "{}".format(" " * 1), ""
            ],
            :,
        ] = ("", "")
        df.loc[IDX["{}".format(" " * 10), "", ""], :] = "", ""

        df.loc[IDX["Requested flexibility", "AGG", "Sum"], :] = (
            around(abs(fdp_requests_per_datetime.values).sum(), 3),
            scenario.units[0],
        )

        df.loc[IDX["Requested flexibility", "AGG", "Sum up"], :] = (
            around(
                fdp_requests_per_datetime[fdp_requests_per_datetime > 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )

        df.loc[IDX["Requested flexibility", "AGG", "Sum down"], :] = (
            around(
                fdp_requests_per_datetime[fdp_requests_per_datetime < 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )

        for enum, name in enumerate(scenario.EMS_names):

            requested_flexibility = EMS[name].groupby(EMS[name].index)["Request"].sum()

            df.loc[IDX["Requested flexibility", name, "Sum"], :] = (
                around(abs(requested_flexibility).sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Requested flexibility", name, "Sum up"], :] = (
                around(requested_flexibility[requested_flexibility > 0].sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Requested flexibility", name, "Sum down"], :] = (
                around(requested_flexibility[requested_flexibility < 0].sum(), 2),
                scenario.units[0],
            )


        """_____________________________________________________________ RE-REQUESTED FLEX STATS """

        df.loc[
            IDX[
                "___________________________________________", "{}".format(" " * 2), ""
            ],
            :,
        ] = ("", "")
        df.loc[IDX["{}".format(" " * 12), "", ""], :] = "", ""

        re_requested = abs(imbalances).sum() - around(
            abs(fdp_requests_per_datetime.values).sum(), 2
        )

        df.loc[IDX["Re-Requested flexibility", "AGG", "Sum"], :] = (
            abs(re_requested),
            scenario.units[0],
        )


        """_____________________________________________________________ OFFERED FLEX STATS """

        df.loc[
            IDX[
                "___________________________________________", "{}".format(" " * 3), ""
            ],
            :,
        ] = ("", "")
        df.loc[IDX["{}".format(" " * 14), "", ""], :] = "", ""

        df.loc[IDX["Offered flexibility", "AGG", "Sum"], :] = (
            around(abs(fdp_offers_per_datetime.values).sum(), 2),
            scenario.units[0],
        )

        df.loc[IDX["Offered flexibility", "AGG", "Sum up"], :] = (
            around(
                fdp_offers_per_datetime[fdp_offers_per_datetime > 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )

        df.loc[IDX["Offered flexibility", "AGG", "Sum down"], :] = (
            around(
                fdp_offers_per_datetime[fdp_offers_per_datetime < 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )


        for enum, name in enumerate(scenario.EMS_names):

            offered_flexibility = EMS[name].groupby(EMS[name].index)["Offer"].sum()

            df.loc[IDX["Offered flexibility", name, "Sum"], :] = (
                around(abs(offered_flexibility).sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Offered flexibility", name, "Sum up"], :] = (
                around(offered_flexibility[offered_flexibility > 0].sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Offered flexibility", name, "Sum down"], :] = (
                around(offered_flexibility[offered_flexibility < 0].sum(), 2),
                scenario.units[0],
            )


        """_____________________________________________________________ ORDERED FLEX STATS """

        df.loc[
            IDX[
                "___________________________________________", "{}".format(" " * 4), ""
            ],
            :,
        ] = ("", "")
        df.loc[IDX["{}".format(" " * 16), "", ""], :] = "", ""

        df.loc[IDX["Ordered flexibility", "AGG", "Sum"], :] = (
            around(abs(fdp_orders_per_datetime.values).sum(), 2),
            scenario.units[0],
        )

        df.loc[IDX["Ordered flexibility", "AGG", "Sum up"], :] = (
            around(
                fdp_orders_per_datetime[fdp_orders_per_datetime > 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )

        df.loc[IDX["Ordered flexibility", "AGG", "Sum down"], :] = (
            around(
                fdp_orders_per_datetime[fdp_orders_per_datetime < 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )


        for enum, name in enumerate(scenario.EMS_names):

            ordered_flexibility = EMS[name].groupby(EMS[name].index)["Offer"].sum()

            df.loc[IDX["Ordered flexibility", name, "Sum"], :] = (
                around(abs(ordered_flexibility).sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Ordered flexibility", name, "Sum up"], :] = (
                around(ordered_flexibility[ordered_flexibility > 0].sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Ordered flexibility", name, "Sum down"], :] = (
                around(ordered_flexibility[ordered_flexibility < 0].sum(), 2),
                scenario.units[0],
            )


        """_____________________________________________________________ DEVIATED FLEX STATS """

        df.loc[
            IDX[
                "___________________________________________", "{}".format(" " * 5), ""
            ],
            :,
        ] = ("", "")
        df.loc[IDX["{}".format(" " * 18), "", ""], :] = "", ""

        df.loc[IDX["Deviated flexibility", "AGG", "Sum"], :] = (
            around(abs(fdp_deviation_per_datetime.values).sum(), 2),
            scenario.units[0],
        )

        df.loc[IDX["Deviated flexibility", "AGG", "Sum up"], :] = (
            around(
                fdp_deviation_per_datetime[fdp_deviation_per_datetime > 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )

        df.loc[IDX["Deviated flexibility", "AGG", "Sum down"], :] = (
            around(
                fdp_deviation_per_datetime[fdp_deviation_per_datetime < 0]
                .sum(skipna=True)
                .values,
                2,
            )[0],
            scenario.units[0],
        )


        for enum, name in enumerate(scenario.EMS_names):

            deviated_flexibility = EMS[name].groupby(EMS[name].index)["Deviation"].sum()

            df.loc[IDX["Deviated flexibility", name, "Sum"], :] = (
                around(abs(deviated_flexibility).sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Deviated flexibility", name, "Sum up"], :] = (
                around(deviated_flexibility[deviated_flexibility > 0].sum(), 2),
                scenario.units[0],
            )

            df.loc[IDX["Deviated flexibility", name, "Sum down"], :] = (
                around(deviated_flexibility[deviated_flexibility < 0].sum(), 2),
                scenario.units[0],
            )


        """_____________________________________________________________ FLEX STATS RATIOS """

        df.loc[IDX["{}".format(" " * 19), "", ""], :] = "", ""
        df.loc[IDX["----------------- RATIOS ------------------", "", ""], :] = "", ""
        df.loc[IDX["{}".format(" " * 20), "", ""], :] = "", ""

        """_____________________________________________________________ RE-REQUEST """

        df.loc[IDX["Re-Requested flexibility", "AGG", "Ratio"], :] = (
            abs(re_requested) / abs(imbalances).sum() * 100,
            "%",
        )

        """_____________________________________________________________ OFFER/REQUEST """

        df.loc[IDX["Offered over requested", "AGG", "Ratio"], :] = (
            around(abs(fdp_offers_per_datetime.values).sum() / abs(imbalances).sum())
            * 100,
            "%",
        )

        """_____________________________________________________________ OFFER/ORDER """

        df.loc[IDX["Offered over ordered", "AGG", "Ratio"], :] = (
            around(
                abs(fdp_offers_per_datetime.values).sum()
                / abs(fdp_orders_per_datetime.values).sum(),
                2,
            )
            * 100,
            "%",
        )


        """_____________________________________________________________ DEVIATION/ORDER """

        if around(abs(fdp_deviation_per_datetime.values).sum(), 2) == 0:
            df.loc[IDX["Deviated over ordered", "AGG", "Ratio"], :] = 0, "%"

        else:
            df.loc[IDX["Deviated over ordered", "AGG", "Ratio"], :] = (
                around(
                    abs(fdp_deviation_per_datetime.values).sum()
                    / abs(fdp_orders_per_datetime.values).sum(),
                    2,
                ),
                "%",
            )


        """_____________________________________________________________ COSTS STATS """

        df.loc[IDX["{}".format(" " * 21), "", ""], :] = "", ""
        df.loc[IDX["----------------- COSTS -------------------", "", ""], :] = "", ""
        df.loc[IDX["{}".format(" " * 22), "", ""], :] = "", ""

        """_____________________________________________________________ COSTS REMAINING """

        remaining_imbalances_costs = (
            remaining_imbalances * FDP_imbalances_market_prices.loc[: horizon[-1]]
        )

        df.loc[IDX["Costs remaining imbalances", "FDP", "Sum"], :] = (
            remaining_imbalances_costs.sum(),
            "Euro",
        )


        """_____________________________________________________________ COSTS FULFILLED """

        fulfilled_requests_at_market_prices = (
            fulfilled_requests * FDP_imbalances_market_prices.loc[: horizon[-1]]
        )

        df.loc[IDX["Costs DR-services (market)", "FDP", "Sum"], :] = (
            fulfilled_requests_at_market_prices.sum(),
            "Euro",
        )


        df.loc[IDX["Costs DR-services (negotation)", "FDP", "Sum"], :] = (
            fulfilled_requests_at_market_prices.sum(),
            "Euro",
        )

        df.loc[IDX["Costs DR-services (negotation)", "", ""], :] = "", ""

        """_____________________________________________________________ COSTS DEVIATION """

        downward_deviation_costs = (
            _fdp_deviation_per_datetime[_fdp_deviation_per_datetime < 0]
            * FDP_deviation_price_down.loc[: horizon[-1]]
        )

        upward_deviation_costs = (
            _fdp_deviation_per_datetime[_fdp_deviation_per_datetime < 0]
            * FDP_deviation_price_up.loc[: horizon[-1]]
        )

        df.loc[IDX["Costs deviations", "AGG", "Sum"], :] = (
            downward_deviation_costs.sum() + upward_deviation_costs.sum(),
            "Euro",
        )


        for enum, name in enumerate(scenario.EMS_names):

            devation_ems = EMS[name]["Deviation"]

            downward_deviation_costs = (
                devation_ems[devation_ems < 0]
                * FDP_deviation_price_down.loc[: horizon[-1]]
            )
            upward_deviation_costs = (
                devation_ems[devation_ems > 0]
                * FDP_deviation_price_up.loc[: horizon[-1]]
            )

            deviation_costs = downward_deviation_costs + upward_deviation_costs

            df.loc[IDX["Costs deviations", name, "Sum"], :] = (
                downward_deviation_costs.sum() + upward_deviation_costs.sum(),
                "Euro",
            )


        df.loc[IDX["Costs deviations", " ", ""], :] = "", ""

        """_____________________________________________________________ COSTS OFFER """

        df.loc[IDX["Costs offer", "AGG", "Sum"], :] = (
            aggregator_offer_costs.values.sum(),
            "Euro",
        )


        for enum, name in enumerate(scenario.EMS_names):

            costs_offer_ems = EMS[name].groupby(EMS[name].index)["COSTS_Offer"].sum()

            df.loc[IDX["Costs offer", name, "Sum"], :] = costs_offer_ems.sum(), "Euro"


        df.loc[IDX["Costs offer", " ", ""], :] = "", ""

        """_____________________________________________________________ COSTS ORDER """

        df.loc[IDX["Costs order", "AGG", "Sum"], :] = (
            aggregator_offer_costs.values.sum(),
            "Euro",
        )


        df.loc[IDX["Costs order", " ", ""], :] = "", ""
        """_____________________________________________________________ PROFITS STATS """

        df.loc[IDX["----------------- PROFITS ------------------", "", ""], :] = "", ""
        df.loc[IDX["{}".format(" " * 24), "", ""], :] = "", ""

        df.loc[IDX["Profit FDP", "Model", "Sum"], :] = 1000, "Euro"
        df.loc[IDX["Profit Aggregator", "Model", "Sum"], :] = 999, "Euro"

        for name in scenario.EMS_names:
            df.loc[IDX["Profit {}".format(name), "Model", "Sum"], :] = 888, "Euro"

        df.drop("DUMMY", inplace=True)

        log = logging.getLogger(__name__)

        # Terminal output handler
        log.setLevel(logging.DEBUG)
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.WARNING)
        stdout_formatter = logging.Formatter("%(message)s")
        stdout_handler.setFormatter(stdout_formatter)

        log.addHandler(stdout_handler)

        # File handler for SUMMARY file and STDOUT
        filename = (
            scenario.result_directory
            + "/"
            + scenario.name
            + "_STATS_"
            + datetime.now().strftime("%Y-%m-%d-%Hh-%Mmin")
            + ".txt"
        )
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(stdout_formatter)

        log.addHandler(file_handler)

        log.warning(
            "\n____________________________________________________________________________________________________________________"
        )
        log.warning(
            "|                                                                                                                  |"
        )
        log.warning(
            "|                                                   STATS OUTPUT                                                   |"
        )
        log.warning(
            "|                                                                                                                  |"
        )
        log.warning(
            "|__________________________________________________________________________________________________________________|\n"
        )

        log.warning(df)

    return plotly.offline.plot(
        fig,
        filename=scenario.result_directory
        + "/PLOTS/"
        + scenario.name
        + "_OPTIMIZATION.html",
    )




def plot_FDP_bid_per_daytime_and_episode(scenario, horizon=None):

    """ _____________________________________________________________  FIG """

    fig = {"data": [], "layout": {}, "frames": []}
    """ _____________________________________________________________  TIME """

    start = scenario.start
    end = start + timedelta(days=1) - scenario.resolution
    one_day_datetime_index = date_range(start=start, end=end, freq=scenario.resolution)

    # Transition duration
    duration = 300
    """ _____________________________________________________________  INPUTS """

    episodes_index = list(str(range(1, scenario.episodes + 1)))
    flexoffer_strategy = scenario.Aggregator.flexoffer_strategy
    fdp_bids = scenario.FDP.imbalances_data["market_prices"].loc[
        : scenario.simulation_time[-1]
    ]

    fdp_bids = DataFrame(data=fdp_bids)
    unique_dates = fdp_bids.index.normalize().unique()

    fdp_bids["Episode"] = 0

    for enum, day in enumerate(unique_dates):
        # Get episode number
        fdp_bids.loc[day : day + timedelta(days=1), "Episode"] = enum + 1

    fdp_bids.set_index("Episode", append=True, inplace=True)
    fdp_bids = fdp_bids.swaplevel(axis=0)


    reward_values = flexoffer_strategy.reward_values_data
    rewards = reward_values.copy()

    r = around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1).cumsum())
    tuples = r.index

    row_multiindex = MultiIndex.from_tuples(tuples, names=["Episode", "Datetime"])

    rewards = DataFrame(
        data=around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1)),
        index=row_multiindex,
    )

    total_reward = rewards.groupby(level=0).sum().sum()

    """ _____________________________________________________________  FRAMES """
    for episode in fdp_bids.groupby(level=0):

        # Get episode number
        nr = episode[0]

        # Create episode frame
        frame = {"data": [], "name": str(nr)}

        # Get episode data and reindex
        episode = episode[1]
        episode = episode.droplevel(0)
        episode.index = one_day_datetime_index

        data_dict = {
            "x": episode.index,
            "y": ravel(episode.values),
            "name": str(nr),
            "line": dict(color="red"),
        }

        frame["name"] = str(nr)
        frame["data"].append(data_dict)
        fig["frames"].append(frame)

    """ _____________________________________________________________  SLIDER """

    fig["layout"]["sliders"] = {
        "args": ["transition", {"duration": duration, "easing": "cubic-in-out"}],
        "initialValue": "1",
        "plotlycommand": "restyle",
        "values": episodes_index,
        "visible": True,
    }

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": dict(size=14, family="Roboto"),
            "prefix": "Episode:",
            "visible": True,
            "xanchor": "center",
            "offset": 10,
        },
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 0, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    for episode in fdp_bids.groupby(level=0):

        nr = episode[0]

        slider_step = {
            "args": [
                [episode[0]],
                {
                    "frame": {"duration": duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": duration},
                },
            ],
            "label": str(episode[0]),
            "method": "animate",
        }

        sliders_dict["steps"].append(slider_step)

    fig["layout"]["sliders"] = [sliders_dict]

    """ _____________________________________________________________  BUTTONS """

    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {
                                "duration": duration,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.09,
            "xanchor": "right",
            "y": 0.02,
            "yanchor": "top",
        }
    ]

    """ _____________________________________________________________  XAXIS """

    fig["layout"]["xaxis"] = {
        "tickformat": "%H:%M",
        "type": "date",
        "tickangle": -90,
        "dtick": 900000 * 2,
        "showgrid": True,
        "gridwidth": 1,
        "gridcolor": "LightPink",
        "domain": [0, 1],
        "ticklen": 5,
        "mirror": True,
        "showline": True,
        "linewidth": 2,
        "linecolor": "black",
    }

    fig["layout"]["yaxis"] = {
        "title": "Euro",
        "domain": [0, 0.75],
        "mirror": True,
        "showline": True,
        "linewidth": 2,
        "linecolor": "black",
    }

    """ _____________________________________________________________  TITLE """

    fig["layout"]["title"] = go.layout.Title(
        text="FDP bids per daytime and episode",
        font=dict(size=20, family="Roboto"),
        xref="paper",
        x=0.5,
        y=0.84,
    )
    """ _____________________________________________________________  LAYOUT """

    fig["layout"]["paper_bgcolor"] = "white"
    fig["layout"]["plot_bgcolor"] = "white"
    fig["layout"]["hovermode"] = "closest"
    fig["layout"]["height"] = 700
    fig["layout"]["margin"] = {"b": 100, "t": 20, "pad": 10}

    fig = go.Figure(fig)

    """ _____________________________________________________________  ANNOTATIONS """

    """ _____________________________________________________________  TABLE ACTIONS """
    df = deepcopy(flexoffer_strategy.action_values)

    df.sort_index(ascending=False, inplace=True)
    prediction_steps = [
        "t_" + str(x)
        for x in range(int(scenario.prediction_delta / scenario.resolution) - 1)
    ]
    df.columns = prediction_steps
    colors = deepcopy(df)  
    colors[colors == True] = "lightgreen"
    colors[colors == False] = "aliceblue"

    df.loc[:, :] = ""

    # Prediction Horizon timesteps
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 0], y=[0, 0.85]),
            header=dict(
                values=prediction_steps,
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[df[x] for x in prediction_steps],
                line_color="lightgrey",
                fill_color=[colors[x] for x in prediction_steps],
                align=["center", "center"],
                font=dict(color="black", size=11),
                height=52,
            ),
        )
    )
    """ _____________________________________________________________  SCENARIO TABLE """

    # Scenario data
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 1], y=[0.86, 1]),
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columnwidth=[
                0.18,  # 1
                0.28,  # 2
                0.08,  # 3
                0.08,  # 4
                0.08,  # 5
                0.04,  # 6
                0.04,  # 7
                0.04,  # 8
                0.04,  # 9
                0.04,  # 10
                0.04,  # 11
                0.06,  # 12
            ],
            header=dict(
                values=[
                    "Scenario",
                    "Description",
                    "Plotted",
                    "Start",
                    "End",
                    "Episodes",
                    "Alpha",
                    "Gamma",
                    "Epsilon",
                    "SED",
                    "EED",
                    "Reward",
                ],
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    scenario.name,
                    scenario.description,
                    "{}".format(datetime.now().strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.start.strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.end.strftime("%d.%m.%Y %H:%M")),
                    scenario.episodes,
                    flexoffer_strategy.alpha,
                    flexoffer_strategy.gamma,
                    round(flexoffer_strategy.epsilon, 2),
                    flexoffer_strategy.start_epsilon_decay,
                    flexoffer_strategy.end_epsilon_decay,
                    total_reward,
                ],
                line_color="lightgrey",
                font=dict(color="black", size=11),
                height=25,
            ),
        )
    )
    return plotly.offline.plot(
        fig,
        filename=scenario.result_directory
        + "/PLOTS/"
        + scenario.name
        + "_Q_VALUES.html",
    )


def plot_orders_per_daytime_and_episode(
    scenario, cumulated: bool = False, horizon=None
):

    """ _____________________________________________________________  FIG """

    fig = {"data": [], "layout": {}, "frames": []}
    """ _____________________________________________________________  TIME """

    start = scenario.start
    end = start + timedelta(days=1) - scenario.resolution
    one_day_datetime_index = date_range(start=start, end=end, freq=scenario.resolution)

    # Transition duration
    duration = 300
    """ _____________________________________________________________  INPUTS """

    """ _____________________________________________________________ INDICES """

    Aggregator = scenario.Aggregator

    FDP = scenario.FDP
    fdp_prediction_periods = int(FDP.prediction_delta / scenario.resolution)
    """ _____________________________________________________________ DATA FDP ORDER PER PERIOD """

    fdp_orders = (
        FDP.data[FDP.name]["Order"]
        .loc[: scenario.last_simulated_index, :]
        .reset_index()
    )
    fdp_orders.set_index("Time", inplace=True)
    unique_dates = fdp_orders.index.normalize().unique()

    fdp_orders["Episode"] = 0
    for enum, day in enumerate(unique_dates):
        # Get episode number
        fdp_orders.loc[day : day + timedelta(days=1), "Episode"] = enum + 1

    fdp_orders.set_index("Episode", append=True, inplace=True)
    fdp_orders = fdp_orders.swaplevel(axis=0)

    episodes_index = list(str(range(1, scenario.episodes + 1)))
    flexoffer_strategy = scenario.Aggregator.flexoffer_strategy

    reward_values = flexoffer_strategy.reward_values_data
    rewards = reward_values.copy()

    r = around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1).cumsum())
    tuples = r.index

    row_multiindex = MultiIndex.from_tuples(tuples, names=["Episode", "Datetime"])

    rewards = DataFrame(
        data=around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1)),
        index=row_multiindex,
    )

    total_reward = rewards.groupby(level=0).sum().sum()

    index = fdp_orders.index.get_level_values(level=1)
    index = index[: len(one_day_datetime_index) * fdp_prediction_periods]

    """ _____________________________________________________________  FRAMES """

    for episode in fdp_orders.groupby(level=0):
        # Get episode number
        nr = episode[0]

        # Create episode frame
        frame = {"data": [], "name": str(nr)}

        # Get episode data and reindex
        episode = episode[1]
        episode = episode.droplevel(0)

        try:
            episode.index = index

        except:
            episode.index = index[: len(episode)]

        if nr == 1:
            trace = go.Bar(
                x=episode.index,
                y=episode["Order"],
                name=str(nr),
                showlegend=False,
                xaxis="x1",
                yaxis="y1",
                marker=dict(
                    color=episode["Period"],
                    colorscale="jet", 
                    line=dict(color="black", width=1),
                ),
            )

            fig["data"].append(trace)

        else:
            trace = go.Bar(
                y=episode["Order"],
                name=str(nr),
                showlegend=False,
                xaxis="x1",
                yaxis="y1",
                marker=dict(
                    color=[x for x in episode["Period"]],
                    colorscale="jet",  
                    line=dict(color="black", width=1),
                ),
            )

        frame["name"] = str(nr)
        frame["data"].append(trace)
        fig["frames"].append(frame)

    fig["data"].append(
        go.Scatter(
            x=[None],
            y=[None],
            showlegend=False,
            xaxis="x1",
            yaxis="y1",
            mode="markers",
            marker=dict(
                colorscale="jet",
                showscale=True,
                cmin=0,
                cmax=fdp_prediction_periods - 1,
                colorbar=dict(
                    thickness=20,
                    yanchor="top",
                    lenmode="fraction",
                    len=0.8,
                    nticks=fdp_prediction_periods,
                    y=0.77,
                    x=0.99,
                    ypad=0,
                ),
            ),
            hoverinfo="none",
        )
    )

    """ _____________________________________________________________  SLIDER """

    fig["layout"]["sliders"] = {
        "args": ["transition", {"duration": duration, "easing": "cubic-in-out"}],
        "initialValue": "1",
        "plotlycommand": "restyle",
        "values": episodes_index,
        "visible": True,
    }

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": dict(size=14, family="Roboto"),
            "prefix": "Episode:",
            "visible": True,
            "xanchor": "center",
            "offset": 10,
        },
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 0, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    for episode in fdp_orders.groupby(level=0):

        nr = episode[0]

        slider_step = {
            "args": [
                [episode[0]],
                {
                    "frame": {"duration": duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": duration},
                },
            ],
            "label": str(episode[0]),
            "method": "animate",
        }

        sliders_dict["steps"].append(slider_step)

    fig["layout"]["sliders"] = [sliders_dict]

    """ _____________________________________________________________  BUTTONS """

    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {
                                "duration": duration,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.09,
            "xanchor": "right",
            "y": 0.02,
            "yanchor": "top",
        }
    ]

    """ _____________________________________________________________  XAXIS """

    fig["layout"]["xaxis"] = {
        "tickformat": "%H:%M",
        "type": "date",
        "tickangle": -90,
        "dtick": 900000 * flexoffer_strategy.prediction_steps,
        "showgrid": True,
        "gridwidth": 1,
        "gridcolor": "LightPink",
        "domain": [0.02, 0.98],
        "ticklen": 5,
        "mirror": True,
        "showline": True,
        "linewidth": 2,
        "linecolor": "black",
    }

    fig["layout"]["yaxis"] = {
        "title": scenario.units[0],
        "showgrid": True,
        "gridwidth": 1,
        "gridcolor": "LightPink",
        "domain": [0, 0.75],
        "mirror": True,
        "showline": True,
        "linewidth": 2,
        "linecolor": "black",
    }

    """ _____________________________________________________________  TITLE """

    fig["layout"]["title"] = go.layout.Title(
        text="Orders per daytime and episode".format(scenario.name.capitalize(),),
        font=dict(size=20, family="Roboto"),
        xref="paper",
        x=0.5,
        y=0.84,
    )
    """ _____________________________________________________________  LAYOUT """

    fig["layout"]["paper_bgcolor"] = "white"
    fig["layout"]["plot_bgcolor"] = "white"
    fig["layout"]["hovermode"] = "closest"
    fig["layout"]["barmode"] = "relative"
    fig["layout"]["bargap"] = 0
    fig["layout"]["height"] = 700
    fig["layout"]["margin"] = {"b": 100, "t": 20, "pad": 10}
    fig = go.Figure(fig)

    """ _____________________________________________________________  ANNOTATIONS """

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        autorange=True,
        automargin=True,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
    )
    """ _____________________________________________________________  SCENARIO TABLE """

    # Scenario data
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 1], y=[0.86, 1]),
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columnwidth=[
                0.18,  # 1
                0.28,  # 2
                0.08,  # 3
                0.08,  # 4
                0.08,  # 5
                0.04,  # 6
                0.04,  # 7
                0.04,  # 8
                0.04,  # 9
                0.04,  # 10
                0.04,  # 11
                0.06,  # 12
            ],
            header=dict(
                values=[
                    "Scenario",
                    "Description",
                    "Plotted",
                    "Start",
                    "End",
                    "Episodes",
                    "Alpha",
                    "Gamma",
                    "Epsilon",
                    "SED",
                    "EED",
                    "Reward",
                ],
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    scenario.name,
                    scenario.description,
                    "{}".format(datetime.now().strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.start.strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.end.strftime("%d.%m.%Y %H:%M")),
                    scenario.episodes,
                    flexoffer_strategy.alpha,
                    flexoffer_strategy.gamma,
                    round(flexoffer_strategy.epsilon, 2),
                    flexoffer_strategy.start_epsilon_decay,
                    flexoffer_strategy.end_epsilon_decay,
                    total_reward,
                ],
                line_color="lightgrey",
                font=dict(color="black", size=11),
                height=25,
            ),
        )
    )
    return plotly.offline.plot(
        fig,
        filename=scenario.result_directory
        + "/PLOTS/"
        + scenario.name
        + "_Q_VALUES.html",
    )


def plot_reward_overview(scenario, horizon=None):

    """ _____________________________________________________________ FONTS """

    xaxis_tick_font = dict(family="Roboto", size=18, color="black")

    yaxis_tick_font = dict(family="Roboto", size=18, color="black")

    yaxis_title_font = dict(family="Roboto", size=24, color="black")
    """ _____________________________________________________________ UNITS """

    yaxis_costs = "Euro"

    """ _____________________________________________________________ MARKER & LINES WIDTH """

    line_width = 2
    marker_line_width = 1

    marker_size_1 = 4
    marker_size_2 = 4

    """ _____________________________________________________________ SUBPLOT TITLES """

    subplot_titles = [
        "",
        "Cumulated rewards over simulation time",
        "Rewards per episode",
        "Max/min/mean rewardsover simulation time",
        "Cumulated rewards per episode",
        "Epsilon",
        "FDP bids",
        "Rewards per datetime",
        "Exploration choice",
        "Reward boost",
        "Alpha",
    ]
    """ _____________________________________________________________ HORIZON """

    horizon = scenario.simulation_time

    """ _____________________________________________________________ DATA """
    flexoffer_strategy = scenario.Aggregator.flexoffer_strategy
    reward_values = flexoffer_strategy.reward_values_data
    rewards_per_datetime = flexoffer_strategy.reward_per_datetime.fillna(0)
    fdp_bids = scenario.FDP.imbalances_data["market_prices"].loc[
        : scenario.simulation_time[-1]
    ]

    """ _____________________________________________________________ ROWS """

    cum_rewards_over_simulation_time_row = 2
    rewards_per_episode_row = 3
    min_max_mean_row = 4
    cumulated_rewards_per_episode_row = 5
    epsilon_row = 6
    FDP_bid_row = 7
    rewards_per_datetime_row = 8
    exploration_choice_row = 9
    alpha_row = 11

    rows = alpha_row
    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                  FIGURE AND LAYOUT                              """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """_____________________________________________________________ FIGURE """

    fig = make_subplots(
        rows=rows,
        cols=1,
        column_widths=[1],
        row_heights=[0.02 if x == 0 else 1 for x in range(0, rows)],
        subplot_titles=subplot_titles,
        vertical_spacing=0.04,
    )

    """ _____________________________________________________________ LAYOUT """

    layout_width = 1550
    layout_height = 8550
    layout = go.Layout(
        width=layout_width,
        height=layout_height,
        bargap=0,
        barmode="relative",
        title=go.layout.Title(
            text="Rewards",
            font=dict(size=24, family="Roboto"),
            xref="paper",
            x=0.5,
            y=0.98,
        ),
        hoverlabel=dict(bgcolor="black", font=dict(color="white")),
        legend=dict(
            x=0.5,
            y=1,
            traceorder="grouped",
            yanchor="top",
            xanchor="center",
            orientation="h",
        ),
        margin=go.layout.Margin(t=10,),
    )

    for i in fig["layout"]["annotations"]:
        i["font"] = dict(size=18, color="black", family="Roboto")

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                     TRACES                                      """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________  PREPROCESS """

    episodes_index = list(range(1, scenario.episodes + 1))
    rewards = reward_values.copy()
    r = around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1).cumsum())
    tuples = r.index

    row_multiindex = MultiIndex.from_tuples(tuples, names=["Episode", "Datetime"])

    rewards = DataFrame(
        data=around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1)),
        index=row_multiindex,
    )

    # rewards
    min_max_mean_data = DataFrame(
        index=episodes_index, columns=["min", "max", "mean"], data=0
    )

    rewards_grouped_by_episode = rewards.groupby(level=[0]).sum()

    min_ = 0
    max_ = 0

    for index, row in rewards_grouped_by_episode.iterrows():

        if index == 1:
            min_ = row[0]

        if row[0] > max_:
            max_ = row[0]
            min_max_mean_data["max"].iloc[index - 1] = row[0]

        if row[0] < min_:
            min_ = row[0]
            min_max_mean_data["min"].iloc[index - 1] = row[0]

        min_max_mean_data["max"].iloc[index - 1] = max_
        min_max_mean_data["min"].iloc[index - 1] = min_
        min_max_mean_data["mean"].iloc[index - 1] = round(
            rewards_grouped_by_episode.loc[:index].values.mean(), 2
        )

    """ _____________________________________________________________  REWARDS PER DATETIME """


    fig.append_trace(
        go.Scatter(
            x=rewards_per_datetime.index,
            y=rewards_per_datetime.values,
            showlegend=False,
            line=dict(color="blue"),
            fillcolor="blue",
            xaxis="x" + str(rewards_per_datetime_row),
            yaxis="y" + str(rewards_per_datetime_row),
        ),
        row=rewards_per_datetime_row,
        col=1,
    )

    """ _____________________________________________________________  EPSILON PER DATETIME """
    fig.append_trace(
        go.Scatter(
            x=rewards_per_datetime.index,
            y=flexoffer_strategy.epsilon_data.values,
            showlegend=False,
            line=dict(color="blue"),
            fillcolor="blue",
            xaxis="x" + str(epsilon_row),
            yaxis="y" + str(epsilon_row),
        ),
        row=epsilon_row,
        col=1,
    )
    """ _____________________________________________________________  FDP BIDS """

    fig.append_trace(
        go.Scatter(
            x=fdp_bids.index,
            y=fdp_bids.values,
            showlegend=False,
            line=dict(color="red"),
            xaxis="x" + str(FDP_bid_row),
            yaxis="y" + str(FDP_bid_row),
        ),
        row=FDP_bid_row,
        col=1,
    )

    """ _____________________________________________________________  CUM REWARDS OVER SIMULATION TIME """

    cum_rewards_over_simulation_time = rewards_per_datetime.cumsum()

    fig.append_trace(
        go.Scatter(
            x=cum_rewards_over_simulation_time.index,
            y=ravel(cum_rewards_over_simulation_time.values),
            showlegend=False,
            xaxis="x" + str(cum_rewards_over_simulation_time_row),
            yaxis="y" + str(cum_rewards_over_simulation_time_row),
        ),
        row=cum_rewards_over_simulation_time_row,
        col=1,
    )

    """ _____________________________________________________________  REWARDS PER EPISODE """
    rewards_per_episode = rewards.copy().groupby(level=0).sum(axis=1)

    fig.append_trace(
        go.Bar(
            x=episodes_index,
            y=ravel(rewards_per_episode.values),
            showlegend=False,
            xaxis="x" + str(rewards_per_episode_row),
            yaxis="y" + str(rewards_per_episode_row),
        ),
        row=rewards_per_episode_row,
        col=1,
    )

    """ _____________________________________________________________  CUM REWARDS PER EPISODE OVER SIM TIME """

    cumulated_rewards_per_episode = rewards_per_episode.cumsum()

    fig.append_trace(
        go.Bar(
            x=cumulated_rewards_per_episode.index,
            y=ravel(cumulated_rewards_per_episode.values),
            showlegend=False,
            xaxis="x" + str(cumulated_rewards_per_episode_row),
            yaxis="y" + str(cumulated_rewards_per_episode_row),
        ),
        row=cumulated_rewards_per_episode_row,
        col=1,
    )

    """ _____________________________________________________________  MIN MAX MEAN """

    fig.append_trace(
        go.Scatter(
            x=min_max_mean_data.index,
            y=min_max_mean_data["min"],
            name="MIN",
            showlegend=False,
            xaxis="x" + str(min_max_mean_row),
            yaxis="y" + str(min_max_mean_row),
        ),
        row=min_max_mean_row,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=min_max_mean_data.index,
            y=min_max_mean_data["max"],
            name="MAX",
            showlegend=False,
            xaxis="x" + str(min_max_mean_row),
            yaxis="y" + str(min_max_mean_row),
        ),
        row=min_max_mean_row,
        col=1,
    )

    fig.append_trace(
        go.Scatter(
            x=min_max_mean_data.index,
            y=min_max_mean_data["mean"],
            name="MEAN",
            showlegend=False,
            xaxis="x" + str(min_max_mean_row),
            yaxis="y" + str(min_max_mean_row),
        ),
        row=min_max_mean_row,
        col=1,
    )

    """ _____________________________________________________________  REWARD BOOST """

    """ _____________________________________________________________  ALPHA """
    try:
        fig.append_trace(
            go.Scatter(
                x=flexoffer_strategy.alpha_data.index,
                y=flexoffer_strategy.alpha_data,
                showlegend=False,
                xaxis="x" + str(alpha_row),
                yaxis="y" + str(alpha_row),
            ),
            row=alpha_row,
            col=1,
        )
    except:
        pass
    """_________________________________________________________________________________"""
    """                                                                                 """
    """                                     TABLE                                       """
    """_________________________________________________________________________________"""
    """                                                                                 """

    # Scenario data
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 1], y=[0.86, 1]),
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columnwidth=[
                0.18,  # 1
                0.28,  # 2
                0.08,  # 3
                0.08,  # 4
                0.08,  # 5
                0.04,  # 6
                0.04,  # 7
                0.04,  # 8
                0.04,  # 9
                0.04,  # 10
                0.04,  # 11
                0.06,  # 12
            ],
            header=dict(
                values=[
                    "Scenario",
                    "Description",
                    "Plotted",
                    "Start",
                    "End",
                    "Episodes",
                    "Alpha",
                    "Gamma",
                    "Epsilon",
                    "SED",
                    "EED",
                    "Reward",
                ],
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    scenario.name,
                    scenario.description,
                    "{}".format(datetime.now().strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.start.strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.end.strftime("%d.%m.%Y %H:%M")),
                    scenario.episodes,
                    flexoffer_strategy.alpha,
                    flexoffer_strategy.gamma,
                    round(flexoffer_strategy.epsilon, 2),
                    flexoffer_strategy.start_epsilon_decay,
                    flexoffer_strategy.end_epsilon_decay,
                    rewards.sum(),
                ],
                line_color="lightgrey",
                font=dict(color="black", size=11),
                height=25,
            ),
        )
    )

    """_________________________________________________________________________________"""
    """                                                                                 """
    """                               LAYOUT UPDATE                                     """
    """_________________________________________________________________________________"""
    """                                                                                 """

    """ _____________________________________________________________  YAXIS TITLES """
    fig["layout"]["yaxis" + str(FDP_bid_row)].update(title=yaxis_costs)
    fig["layout"]["yaxis" + str(rewards_per_datetime_row)].update(title=yaxis_costs)
    fig["layout"]["yaxis" + str(cum_rewards_over_simulation_time_row)].update(
        title=yaxis_costs
    )
    fig["layout"]["yaxis" + str(rewards_per_episode_row)].update(title=yaxis_costs)
    fig["layout"]["yaxis" + str(cumulated_rewards_per_episode_row)].update(
        title=yaxis_costs
    )


    """ ________ UPDATE LAYOUT ________"""

    total_days = (scenario.end - scenario.start).days

    if total_days <= 1:
        dtick_multiplicator = 2

    elif total_days > 1 and total_days <= 3:
        dtick_multiplicator = 4

    elif total_days > 3 and total_days <= 7:
        dtick_multiplicator = 8

    elif total_days > 7 and total_days <= 14:
        dtick_multiplicator = 12

    elif total_days > 14 and total_days <= 21:
        dtick_multiplicator = 16

    elif total_days > 21:
        dtick_multiplicator = 24

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        type="date",
        tickangle=-90,
        tickformat="%d.%m.%Y | %H:%M",
        tickmode="auto",
        automargin=True,
        dtick=900000 * dtick_multiplicator,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        tickfont=xaxis_tick_font,
    )

    for x in [
        str(rewards_per_episode_row),
        str(cumulated_rewards_per_episode_row),
        str(exploration_choice_row),
        str(min_max_mean_row),
    ]:
        fig["layout"]["xaxis" + x].update(
            showline=True,
            linewidth=2,
            linecolor="black",
            mirror=True,
            type="category",
            tickangle=0,
            tickmode="auto",
            automargin=True,
            dtick=900000 * dtick_multiplicator,
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=1,
            tickfont=xaxis_tick_font,
        )

    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        tickfont=yaxis_tick_font,
        title_font=yaxis_title_font,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
    )

    fig.update_layout(layout)

    return plotly.offline.plot(
        fig,
        filename=scenario.result_directory
        + "/PLOTS/"
        + scenario.name
        + "_REWARDS.html",
    )

def plot_reward_per_daytime_and_episode(
    scenario, cumulated: bool = False, horizon=None
):

    """ _____________________________________________________________  FIG """

    fig = {"data": [], "layout": {}, "frames": []}
    """ _____________________________________________________________  TIME """

    start = scenario.start
    end = start + timedelta(days=1) - scenario.resolution

    # Transition duration
    duration = 300
    """ _____________________________________________________________  INPUTS """

    episodes_index = list(str(range(1, scenario.episodes + 1)))
    flexoffer_strategy = scenario.Aggregator.flexoffer_strategy
    reward_values = flexoffer_strategy.reward_values_data
    rewards = reward_values.copy()

    total_reward = rewards.groupby(level=0).sum().sum()

    r = around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1).cumsum())
    tuples = r.index

    row_multiindex = MultiIndex.from_tuples(tuples, names=["Episode", "Datetime"])

    rewards = DataFrame(
        data=around(rewards.groupby(rewards.index).sum(axis=0).sum(axis=1)),
        index=row_multiindex,
    )

    one_day_datetime_index = flexoffer_strategy.state_date_index

    """ _____________________________________________________________  FRAMES """

    for episode in rewards.groupby(level=0):

        # Get episode number
        nr = episode[0]

        # Create episode frame
        frame = {"data": [], "name": str(nr)}

        # Get episode data and reindex
        episode = episode[1]
        episode = episode.droplevel(0)
        episode.index = one_day_datetime_index

        if cumulated:
            episode = episode.cumsum()
        data_dict = {
            "x": episode.index,
            "y": ravel(episode.values),
            "name": str(nr),
        }


        frame["name"] = str(nr)
        frame["data"].append(data_dict)
        fig["frames"].append(frame)

    """ _____________________________________________________________  SLIDER """

    fig["layout"]["sliders"] = {
        "args": ["transition", {"duration": duration, "easing": "cubic-in-out"}],
        "initialValue": "1",
        "plotlycommand": "restyle",
        "values": episodes_index,
        "visible": True,
    }

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": dict(size=14, family="Roboto"),
            "prefix": "Episode:",
            "visible": True,
            "xanchor": "center",
            "offset": 10,
        },
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 0, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    for episode in reward_values.groupby(level=0):

        nr = episode[0]

        slider_step = {
            "args": [
                [episode[0]],
                {
                    "frame": {"duration": duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": duration},
                },
            ],
            "label": str(episode[0]),
            "method": "animate",
        }

        sliders_dict["steps"].append(slider_step)

    fig["layout"]["sliders"] = [sliders_dict]

    """ _____________________________________________________________  BUTTONS """

    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {
                                "duration": duration,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.09,
            "xanchor": "right",
            "y": 0.02,
            "yanchor": "top",
        }
    ]

    """ _____________________________________________________________  XAXIS """

    fig["layout"]["xaxis"] = {
        "tickformat": "%H:%M",
        "tickangle": -90,
        "gridwidth": 1,
        "gridcolor": "LightPink",
        "domain": [0, 1],
        "ticklen": 5,
        "mirror": True,
        "showline": True,
        "linewidth": 2,
        "linecolor": "black",
    }

    fig["layout"]["yaxis"] = {
        "title": "Euro",
        "domain": [0.1, 0.75],
        "mirror": True,
    }

    """ _____________________________________________________________  TITLE """

    fig["layout"]["title"] = go.layout.Title(
        text="Rewards per daytime and episode".format(scenario.name.capitalize(),),
        font=dict(size=20, family="Roboto"),
        xref="paper",
        x=0.5,
        y=0.84,
    )
    """ _____________________________________________________________  LAYOUT """

    fig["layout"]["paper_bgcolor"] = "white"
    fig["layout"]["plot_bgcolor"] = "white"
    fig["layout"]["hovermode"] = "closest"
    fig["layout"]["height"] = 700
    fig["layout"]["margin"] = {"b": 100, "t": 20, "pad": 10}

    fig = go.Figure(fig)

    """ _____________________________________________________________  ANNOTATIONS """

    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
    )

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        automargin=True,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
    )

    """ _____________________________________________________________  SCENARIO TABLE """
    # Scenario data
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 1], y=[0.86, 1]),
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columnwidth=[
                0.18,  # 1
                0.28,  # 2
                0.08,  # 3
                0.08,  # 4
                0.08,  # 5
                0.04,  # 6
                0.04,  # 7
                0.04,  # 8
                0.04,  # 9
                0.04,  # 10
                0.04,  # 11
                0.06,  # 12
            ],
            header=dict(
                values=[
                    "Scenario",
                    "Description",
                    "Plotted",
                    "Start",
                    "End",
                    "Episodes",
                    "Alpha",
                    "Gamma",
                    "Epsilon",
                    "SED",
                    "EED",
                    "Reward",
                ],
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    scenario.name,
                    scenario.description,
                    "{}".format(datetime.now().strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.start.strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.end.strftime("%d.%m.%Y %H:%M")),
                    scenario.episodes,
                    flexoffer_strategy.alpha,
                    flexoffer_strategy.gamma,
                    round(flexoffer_strategy.epsilon, 2),
                    flexoffer_strategy.start_epsilon_decay,
                    flexoffer_strategy.end_epsilon_decay,
                    total_reward,
                ],
                line_color="lightgrey",
                font=dict(color="black", size=11),
                height=25,
            ),
        )
    )
    return plotly.offline.plot(
        fig,
        filename=scenario.result_directory
        + "/PLOTS/"
        + scenario.name
        + "_Q_VALUES.html",
    )



#%%


def plot_q_tables(scenario):

    """ _____________________________________________________________  FIG """

    fig = {"data": [], "layout": {}, "frames": []}
    """ _____________________________________________________________  TIME """

    start = scenario.start
    end = start + timedelta(days=1) - scenario.resolution
    """ _____________________________________________________________  INPUTS """

    episodes_index = list(str(range(1, scenario.episodes + 1)))
    flexoffer_strategy = scenario.Aggregator.flexoffer_strategy
    q_values = flexoffer_strategy.q_values_data

    q_max = q_values.max().max()
    q_min = q_values.min().min()
    q_values = (q_values - q_min) / (q_max - q_min)


    reward_values = flexoffer_strategy.reward_values_data
    total_reward = reward_values.groupby(level=0).sum().sum().sum()

    one_day_datetime_index = flexoffer_strategy.daily_date_range

    """ _____________________________________________________________  FRAMES """

    for episode in q_values.groupby(level=0):

        # Get episode number
        nr = episode[0]

        # Create episode frame
        frame = {"data": [], "name": str(nr)}

        # Get episode data and reindex
        episode = episode[1]
        episode = episode.droplevel(0)
        episode.index = one_day_datetime_index

        # Get actions as list of strings
        actions = episode.columns

        # Heatmap values
        z = around(episode.T.values.astype(np.double), decimals=2)

        if nr == 1:
            # episode = episode.round(2)
            data_dict = {
                "x": episode.index,
                "y": [a.capitalize() for a in actions],
                "z": z,
                "type": "heatmap",
                "name": str(nr),
                "xgap": 0.1,
                "ygap": 0.1,
                "xaxis": "x1",
                "colorbar": dict(
                    thickness=20,
                    yanchor="top",
                    lenmode="fraction",
                    len=0.71,
                    y=0.81,
                    ypad=0,
                    xpad=0,
                ),
                "zmin": 0,
                "zmax": 1
            }

            fig["data"].append(data_dict)
        else:
            # episode = episode.round(2)
            data_dict = {
                "z": z,
                "type": "heatmap",
                "name": str(nr),
            }

        frame["name"] = str(nr)
        frame["data"].append(data_dict)
        fig["frames"].append(frame)

    """ _____________________________________________________________  SLIDER """

    fig["layout"]["sliders"] = {
        "args": ["transition", {"duration": duration, "easing": "cubic-in-out"}],
        "initialValue": "1",
        "plotlycommand": "restyle",
        "values": episodes_index,
        "visible": True,
    }

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": dict(size=14, family="Roboto"),
            "prefix": "Episode:",
            "visible": True,
            "xanchor": "center",
            "offset": 10,
        },
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 0, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    for episode in q_values.groupby(level=0):
        nr = episode[0]

        slider_step = {
            "args": [
                [episode[0]],
                {
                    "frame": {"duration": duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": duration},
                },
            ],
            "label": str(episode[0]),
            "method": "animate",
        }

        sliders_dict["steps"].append(slider_step)

    fig["layout"]["sliders"] = [sliders_dict]

    """ _____________________________________________________________  BUTTONS """

    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {
                                "duration": duration,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.09,
            "xanchor": "right",
            "y": 0.02,
            "yanchor": "top",
        }
    ]

    """ _____________________________________________________________  XAXIS """

    fig["layout"]["xaxis"] = {
        "tickformat": "%H:%M",
        "tickangle": -90,
        "gridwidth": 1,
        "gridcolor": "LightPink",
        "domain": [0, 1],
        "ticklen": 5
    }
    fig["layout"]["yaxis"] = {"domain": [0.1, 0.81]}

    """ _____________________________________________________________  TITLE """

    fig["layout"]["title"] = go.layout.Title(
        text="Q-value tables per episode".format(scenario.name.capitalize(),),
        font=dict(size=20, family="Roboto"),
        xref="paper",
        x=0.54,
        y=0.85,
    )
    """ _____________________________________________________________  LAYOUT """

    fig["layout"]["paper_bgcolor"] = "white"
    fig["layout"]["plot_bgcolor"] = "white"
    fig["layout"]["hovermode"] = "closest"
    fig["layout"]["height"] = 700
    fig["layout"]["margin"] = {"b": 100, "t": 30, "pad": 10}

    fig = go.Figure(fig)

    """ _____________________________________________________________  SCENARIO TABLE """

    # Scenario data
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 1], y=[0.86, 1]),
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columnwidth=[
                0.18,  # 1
                0.28,  # 2
                0.08,  # 3
                0.08,  # 4
                0.08,  # 5
                0.04,  # 6
                0.04,  # 7
                0.04,  # 8
                0.04,  # 9
                0.04,  # 10
                0.04,  # 11
                0.06,  # 12
            ],
            header=dict(
                values=[
                    "Scenario",
                    "Description",
                    "Plotted",
                    "Start",
                    "End",
                    "Episodes",
                    "Alpha",
                    "Gamma",
                    "Epsilon",
                    "SED",
                    "EED",
                    "Reward",
                ],
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    scenario.name,
                    scenario.description,
                    "{}".format(datetime.now().strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.start.strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.end.strftime("%d.%m.%Y %H:%M")),
                    scenario.episodes,
                    flexoffer_strategy.alpha,
                    flexoffer_strategy.gamma,
                    round(flexoffer_strategy.epsilon, 2),
                    flexoffer_strategy.start_epsilon_decay,
                    flexoffer_strategy.end_epsilon_decay,
                    total_reward,
                ],
                line_color="lightgrey",
                font=dict(color="black", size=11),
                height=25,
            ),
        )
    )
    return plotly.offline.plot(
        fig,
        filename=scenario.result_directory
        + "/PLOTS/"
        + scenario.name
        + "_Q_VALUES.html",
    )


# plot_q_tables(scenario=s)

#%%
def plot_performed_action_tables(scenario, cumulated: bool = False):

    """ _____________________________________________________________  FIG """

    fig = {"data": [], "layout": {}, "frames": []}
    """ _____________________________________________________________  TIME """

    start = scenario.start
    end = start + timedelta(days=1) - scenario.resolution

    # Transition duration
    duration = 300
    """ _____________________________________________________________  INPUTS """

    episodes_index = list(str(range(1, scenario.episodes + 1)))
    flexoffer_strategy = scenario.Aggregator.flexoffer_strategy
    performed_actions = flexoffer_strategy.performed_actions_data
    one_day_datetime_index = flexoffer_strategy.daily_date_range

    last_episode = None

    if cumulated:
        plot_title = "Cumulative performed actions per datetime and episode"
        filename = (
            scenario.result_directory
            + "/PLOTS/"
            + scenario.name
            + "_ACTION_VALUES_CUM.html"
        )
    else:
        plot_title = "Performed actions per datetime and episode"
        filename = (
            scenario.result_directory
            + "/PLOTS/"
            + scenario.name
            + "_ACTION_VALUES.html"
        )

    reward_values = flexoffer_strategy.reward_values_data
    total_reward = reward_values.groupby(level=0).sum().sum().sum()

    """ _____________________________________________________________  FRAMES """

    last_episode = []

    for episode in performed_actions.groupby(level=0):

        # Get episode number
        nr = episode[0]

        # Create episode frame
        frame = {"data": [], "name": str(nr)}

        # Get episode data and reindex
        episode = episode[1]
        episode = episode.droplevel(0)
        episode.index = one_day_datetime_index

        # Get actions as list of strings
        actions = episode.columns

        # Heatmap values
        z = around(episode.T.values.astype(np.double), decimals=2)

        if nr == 1:
            # episode = episode.round(2)
            data_dict = {
                "x": episode.index,
                "y": [a.capitalize() for a in actions],
                "z": z,
                "type": "heatmap",
                "name": str(nr),
                "xgap": 0.1,
                "ygap": 0.1,
                "xaxis": "x1",
                "colorbar": dict(
                    thickness=20,
                    yanchor="top",
                    lenmode="fraction",
                    len=0.71,
                    y=0.81,
                    ypad=0,
                    xpad=0,
                ),
                "colorscale": "portland",
            }

            last_episode.append(deepcopy(z))
            fig["data"].append(data_dict)

        else:

            if cumulated:
                z = deepcopy(z) + last_episode[-1]
                last_episode.append(z)

            data_dict = {
                "z": z,
                "type": "heatmap",
                "name": str(nr),
            }

        frame["name"] = str(nr)
        frame["data"].append(data_dict)
        fig["frames"].append(frame)

    """ _____________________________________________________________  SLIDER """

    fig["layout"]["sliders"] = {
        "args": ["transition", {"duration": duration, "easing": "cubic-in-out"}],
        "initialValue": "1",
        "plotlycommand": "restyle",
        "values": episodes_index,
        "visible": True,
    }

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": dict(size=14, family="Roboto"),
            "prefix": "Episode:",
            "visible": True,
            "xanchor": "center",
            "offset": 10,
        },
        "transition": {"duration": duration, "easing": "cubic-in-out"},
        "pad": {"b": 0, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [],
    }

    for episode in performed_actions.groupby(level=0):
        nr = episode[0]

        slider_step = {
            "args": [
                [episode[0]],
                {
                    "frame": {"duration": duration, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": duration},
                },
            ],
            "label": str(episode[0]),
            "method": "animate",
        }

        sliders_dict["steps"].append(slider_step)

    fig["layout"]["sliders"] = [sliders_dict]

    """ _____________________________________________________________  BUTTONS """

    fig["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [
                        None,
                        {
                            "frame": {"duration": duration, "redraw": True},
                            "fromcurrent": True,
                            "transition": {
                                "duration": duration,
                                "easing": "quadratic-in-out",
                            },
                        },
                    ],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    "label": "Pause",
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.09,
            "xanchor": "right",
            "y": 0.02,
            "yanchor": "top",
        }
    ]

    """ _____________________________________________________________  XAXIS """

    fig["layout"]["xaxis"] = {
        "tickformat": "%H:%M",
        "tickangle": -90,
        "gridwidth": 1,
        "gridcolor": "LightPink",
        "domain": [0, 1],
        "ticklen": 5
    }
    fig["layout"]["yaxis"] = {"domain": [0.1, 0.81]}

    """ _____________________________________________________________  TITLE """

    fig["layout"]["title"] = go.layout.Title(
        text=plot_title,
        font=dict(size=20, family="Roboto"),
        xref="paper",
        x=0.55,
        y=0.85,
    )
    """ _____________________________________________________________  LAYOUT """

    fig["layout"]["paper_bgcolor"] = "white"
    fig["layout"]["plot_bgcolor"] = "white"
    fig["layout"]["hovermode"] = "closest"
    fig["layout"]["height"] = 700
    fig["layout"]["margin"] = {"b": 100, "t": 30, "pad": 10}

    fig = go.Figure(fig)
    """ _____________________________________________________________  SCENARIO TABLE """
    # Scenario data
    fig.add_trace(
        go.Table(
            domain=dict(x=[0, 1], y=[0.86, 1]),
            columnorder=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            columnwidth=[
                0.18,  # 1
                0.28,  # 2
                0.08,  # 3
                0.08,  # 4
                0.08,  # 5
                0.04,  # 6
                0.04,  # 7
                0.04,  # 8
                0.04,  # 9
                0.04,  # 10
                0.04,  # 11
                0.06,  # 12
            ],
            header=dict(
                values=[
                    "Scenario",
                    "Description",
                    "Plotted",
                    "Start",
                    "End",
                    "Episodes",
                    "Alpha",
                    "Gamma",
                    "Epsilon",
                    "SED",
                    "EED",
                    "Reward",
                ],
                line_color="lightgrey",
                fill_color="white",
                align="center",
                font=dict(color="black", size=12),
            ),
            cells=dict(
                values=[
                    scenario.name,
                    scenario.description,
                    "{}".format(datetime.now().strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.start.strftime("%d.%m.%Y %H:%M")),
                    "{}".format(scenario.end.strftime("%d.%m.%Y %H:%M")),
                    scenario.episodes,
                    flexoffer_strategy.alpha,
                    flexoffer_strategy.gamma,
                    round(flexoffer_strategy.epsilon, 2),
                    flexoffer_strategy.start_epsilon_decay,
                    flexoffer_strategy.end_epsilon_decay,
                    total_reward,
                ],
                line_color="lightgrey",
                font=dict(color="black", size=11),
                height=25,
            ),
        )
    )
    return plotly.offline.plot(fig, filename=filename)




def plot_scenario_reward_comparison(case_study_directory: str):

    # List of scenario objects
    reward_data = dict()

    case_study_directory = os.path.dirname(__file__) + "/" + case_study_directory

    case_study = case_study_directory.rsplit("/", 1)[-1]

    scenario_directories = list(os.walk(case_study_directory))[0][1]


    for sub_dir in scenario_directories:

        absolute_path_sub_dir = case_study_directory + "/" + sub_dir

        for file in os.listdir(absolute_path_sub_dir):

            if file.endswith(".p") and "SCENARIO" in file:

                scenario_file_path = os.path.join(absolute_path_sub_dir, file)
                scenario = load_scenario(file=scenario_file_path)

                reward_data[
                    sub_dir
                ] = scenario.Aggregator.flexoffer_strategy.reward_values_data.droplevel(
                    0
                )

    """ _____________________________________________________________ FONTS """

    xaxis_tick_font = dict(family="Roboto", size=14, color="black")

    yaxis_tick_font = dict(family="Roboto", size=14, color="black")

    yaxis_title_font = dict(family="Roboto", size=16, color="black")
    """ _____________________________________________________________ UNITS """

    yaxis_costs = "Euro"

    """ _____________________________________________________________ MARKER & LINES WIDTH """

    line_width = 2
    marker_line_width = 1

    marker_size_1 = 4
    marker_size_2 = 4

    """_____________________________________________________________ FIGURE """

    fig = go.Figure()

    """ _____________________________________________________________  REWARDS PER DATETIME """

    for scenario_name, scenario_rewards in reward_data.items():

        scenario_rewards = scenario_rewards.sum(axis=1).cumsum()
        scenario_rewards = scenario_rewards.round(2)

        fig.add_trace(
            go.Scatter(
                x=scenario_rewards.index,
                y=scenario_rewards,
                showlegend=True,
                name=scenario_name
            )
        )

    layout = go.Layout(
        title=go.layout.Title(
            text="Rewards comparison: {}".format(case_study),
            font=dict(size=24, family="Roboto"),
            xref="paper",
            x=0.5,
            y=0.96,
        ),
        legend=dict(
            x=0,
            y=1,
            traceorder="normal",
            font=dict(family="sans-serif", size=12, color="#000"),
        ),
    )

    fig["layout"] = layout

    """ ________ UPDATE LAYOUT ________"""

    total_days = (scenario.end - scenario.start).days

    if total_days <= 1:
        dtick_multiplicator = 2

    elif total_days > 1 and total_days <= 3:
        dtick_multiplicator = 4

    elif total_days > 3 and total_days <= 7:
        dtick_multiplicator = 8

    elif total_days > 7 and total_days <= 14:
        dtick_multiplicator = 12

    elif total_days > 14 and total_days <= 21:
        dtick_multiplicator = 16

    elif total_days > 21:
        dtick_multiplicator = 24

    fig.update_xaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        type="date",
        tickangle=-90,
        tickformat="%d.%m.%Y | %H:%M",
        tickmode="auto",
        automargin=True,
        dtick=900000 * dtick_multiplicator,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
        tickfont=xaxis_tick_font,
    )

    fig.update_yaxes(
        showline=True,
        linewidth=2,
        linecolor="black",
        mirror=True,
        tickfont=yaxis_tick_font,
        title_font=yaxis_title_font,
        zeroline=True,
        zerolinecolor="black",
        zerolinewidth=1,
    )

    return plotly.offline.plot(
        fig,
        filename=case_study_directory + "/" + case_study + "_REWARD_comparison.html",
    )

def create_scenario_plots(
    scenario_path: str,
    optimization: bool = False,
    orders: bool = False,
    rewards_overview: bool = False,
    reward_per_daytime_and_episode: bool = False,
    reward_per_daytime_and_episode_cumulated: bool = False,
    q_tables: bool = False,
    performed_action: bool = False,
    performed_action_cum: bool = False,
):


    # Add system path
    scenario_path = os.path.dirname(__file__) + "/" + scenario_path

    # List of scenario objects
    scenarios = []

    # Get scenarios from folder
    for file in os.listdir(scenario_path):

        if file.endswith(".p"):

            scenario_file = os.path.join(scenario_path, file)

            assert "SCENARIO" in scenario_file, "Scenario file not found."

            scenarios.append(load_scenario(file=scenario_file))

    if optimization:

        for scenario in scenarios:
            plot_optimization_data(scenario=scenario)

    if rewards_overview:

        for scenario in scenarios:
            plot_reward_overview(scenario=scenario)

    if reward_per_daytime_and_episode:

        for scenario in scenarios:
            plot_reward_per_daytime_and_episode(scenario=scenario)

    if reward_per_daytime_and_episode_cumulated:

        for scenario in scenarios:
            plot_reward_per_daytime_and_episode(scenario=scenario, cumulated=True)

    if orders:

        for scenario in scenarios:
            plot_orders_per_daytime_and_episode(scenario=scenario)

    if q_tables:

        for scenario in scenarios:
            plot_q_tables(scenario=scenario)

    if performed_action:

        for scenario in scenarios:
            plot_performed_action_tables(scenario=scenario)

    if performed_action_cum:

        for scenario in scenarios:
            plot_performed_action_tables(scenario=scenario, cumulated=True)
