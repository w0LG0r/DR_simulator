from pandas import set_option
from plot import create_scenario_plots

# Console output configuration
set_option("display.max_columns", 25)
set_option("display.max_rows", 1000)
set_option("display.width", 1000)
set_option("display.colheader_justify", "left")
set_option("display.colheader_justify", "left")
set_option("max_info_columns", 1000)
set_option("display.precision", 3)

print("       _______ _____ _______ _     _        _______ _______  _____   ______")
print("       |______   |   |  |  | |     | |      |_____|    |    |     | |_____/")
print("       ______| __|__ |  |  | |_____| |_____ |     |    |    |_____| |    \_")

# Import the scenarios to simulate
from scenarios.case_studies.test_01.NAIVE_01 import scenario as NAIVE_01
# from scenarios.case_studies.test_01.Q_LEARNING_01 import scenario as Q_LEARNING_01
# from scenarios.case_studies.test_01.RANDOM_01 import scenario as RANDOM_01

if __name__ == "__main__":

    # Run scenarios
    for scenario in [
        NAIVE_01,
        # Q_LEARNING_FDP_01,
        # RANDOM_FDP_01,
    ]:

        scenario.run

        create_scenario_plots(
            scenario_path=scenario.result_directory.rsplit("simulator\\")[-1],
            optimization=True,
            # rewards_overview=True,
            # orders=True,
            # reward_per_daytime_and_episode=True,
            # reward_per_daytime_and_episode_cumulated=True,
            # q_tables=True,
            # performed_action=True,
            # performed_action_cum=True,
        )

    print("\nFINISHED TEST RUN")
