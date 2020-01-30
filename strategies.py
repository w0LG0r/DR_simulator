#%%
from pandas import *
from pandas import IndexSlice as IDX
from pandas.core.common import flatten

import numpy as np
from numpy.random import seed as numpy_seed
from datetime import date, datetime, timedelta
from random import seed

from copy import deepcopy
import random
import math
import itertools

from logger import log as logger

class QLearning:
    def __init__(
        self,
        simulation_start: datetime,
        simulation_end: datetime,
        episode_duration: timedelta,
        resolution: Timedelta,
        last_episode: int,
        prediction_delta: Timedelta,
        exploration: str,
        alpha_0: float,
        alpha_1: float,
        gamma: float,
        epsilon: float,
        start_epsilon_decay: int,
        end_epsilon_decay: int,
        seed_value: int,
        action_probabilites: list,
        action_probabilites_decay: float,
        max_reward_boost: float,
    ):

        # Check AttributeErrors
        for attribute in [alpha_1, alpha_0, gamma, epsilon]:

            if isinstance(attribute, float) is False:
                raise AttributeError(
                    "Flexoffer strategy input error: Alpha, gamma, epsilon must be float values"
                )

            if attribute < 0:
                raise AttributeError(
                    "Atttibute with negative value {} must be a float between 0 and 1".format(
                        attribute
                    )
                )

        # SIMULATION TIME
        self.start = simulation_start
        self.end = simulation_end
        self.resolution = resolution

        # EPISODES
        self.last_episode = last_episode

        # LEARNING RATE
        self.alpha_0 = alpha_0
        self.alpha_1 = alpha_1

        # FUTRURE REWARD DISCOUNT RATE
        self.gamma = gamma

        # EXPLORATION TYPE
        self.exploration = exploration

        # GREEDY ACTION SELECTION PARAMETER
        self.epsilon = epsilon
        self.start_epsilon_decay = start_epsilon_decay
        self.end_epsilon_decay = end_epsilon_decay
        self.epsilon_decay_value = (1 - epsilon) / (
            end_epsilon_decay - start_epsilon_decay
        )

        self.epsilon_data = Series(
            index=date_range(
                start=simulation_start,
                end=simulation_end,
                freq=resolution,
                closed="left",
                name="Time",
            )
        )

        self.alpha_data = Series(
            index=date_range(
                start=simulation_start,
                end=simulation_end,
                freq=resolution,
                closed="left",
                name="Time",
            )
        )

        self.reward_per_datetime = Series(
            index=date_range(
                start=simulation_start,
                end=simulation_end,
                freq=resolution,
                closed="left",
                name="Time",
            )
        )

        self.exploration_choice = Series(
            index=date_range(
                start=simulation_start,
                end=simulation_end,
                freq=resolution,
                closed="left",
                name="Time",
            )
        )

        self.max_reward = 0
        self.max_reward_boost = max_reward_boost

        # Get prediction steps and
        self.prediction_steps = int(prediction_delta / self.resolution) - 1
        self.possible_flexibility_activations = 2

        d = date_range(
            start="00:00", end="23:59", freq=resolution, closed="left", name="Time"
        ).strftime("%H:%M:%S")

        dates = []

        for date in d:

            if date != d[0]:

                a0 = date + "_a0"
                dates.append(a0)

                a1 = date + "_a1"
                dates.append(a1)

            else:
                dates.append(date)

        self.daily_date_range = Index(dates)

        # INIT STATE
        self.current_state = self.daily_date_range[0]

        steps_full_day = timedelta(hours=1) / resolution * 24

        episode_steps = ((self.start + episode_duration) - self.start) / self.resolution
        self.current_episode = 1

        if episode_steps < steps_full_day - 1:
            self.termination_time = self.daily_date_range[episode_steps]

        else:
            self.termination_time = self.daily_date_range[steps_full_day - 1]

        logger.info(
            "\n\t\t\t\t\t\tEpisode termination at: {}".format(self.termination_time)
        )

        self.action_values = DataFrame(
            index=["a" + str(x) for x in range(self.possible_flexibility_activations)],
            columns=[x for x in range(1, self.prediction_steps + 2)],
        )

        self.action_values.loc["a0", :] = False
        self.action_values.loc["a1", :] = True

        # Create dataframe that counts the number of times an action got called
        (
            self.possible_actions,
            self.performed_actions,
            self.performed_actions_cumulated,
            self.action_probabilties_per_state,
            self.q_values,
            self.reward_values,
        ) = [
            DataFrame(
                index=self.daily_date_range,
                columns=[
                    "a" + str(x)
                    for x in range(0, self.possible_flexibility_activations)
                ],
                data=0,
            )
            for x in range(6)
        ]

        self.action_probabilites_decay = action_probabilites_decay
        self.action_probabilties_per_state["a0"] = action_probabilites[0]
        self.action_probabilties_per_state["a1"] = action_probabilites[1]

        episodes_datetimes_lists = [
            date_range(
                start=simulation_start + timedelta(days=x),
                periods=steps_full_day,
                freq=self.resolution,
                closed="left",
                name="Time",
            ).tolist()
            for x in range(self.last_episode)
        ]

        episodes_datetimes_lists = [
            self.daily_date_range for x in range(self.last_episode)
        ]

        episode_numbers_list = list(range(1, self.last_episode + 1))
        tuples = []

        for episode in episode_numbers_list:
            for _datetime in episodes_datetimes_lists[episode - 1]:
                tuples.append((episode, _datetime))

        self.row_multiindex = MultiIndex.from_tuples(tuples)

        (
            self.performed_actions_data,
            self.q_values_data,
            self.reward_values_data,
            self.action_probabilties_per_state_data,
        ) = [
            DataFrame(
                index=self.row_multiindex,
                columns=[
                    "a" + str(x) for x in range(self.possible_flexibility_activations)
                ],
            )
            for x in range(4)
        ]

    def select_action(self, profile):

        logger.info(
            "\nAggregator selects flexibility from aggregated offer based on EPSILON-Greedy-Policy\n"
        )

        # Store copy for return
        _profile = profile.copy()

        # Remove unused levels
        profile.columns = profile.columns.droplevel(level=0)
        profile.index = profile.index.droplevel(level=0)

        # Get indices of profile periods where no flexibility is available
        # NOTE: Periods with no flexibility are marked as "False"
        mask = profile["Offer"].notnull()
        logger.info("Flexibility available at:\n{}".format(mask))

        # Copy all possible actions array
        possible_actions = self.action_values.copy()
        possible_actions.loc["a1", :] = mask.values

        logger.info(" _________________________________________________")
        logger.info("|												  |")
        logger.info("|	   ---> ---> FULL ACTION SPACE <--- <---      |")
        logger.info("|_________________________________________________|\n")

        logger.info(
            "Possible flexoffer combinations (rows) over prediction horizon periods (cols):\n{}\n".format(
                possible_actions
            )
        )

        # Cut action space at horizon steps where no profile flexibility exists
        for idx, mask_row in mask.iteritems():

            logger.info(
                ">>>> Flexibility available at period {} ---> {}\n".format(
                    idx, mask_row
                )
            )

            if mask_row == False:
                possible_actions = possible_actions[possible_actions[idx] == mask_row]

                logger.info(
                    "Aggregator sliced action space to:\n{}\n".format(possible_actions)
                )

        if possible_actions.loc["a0", :].equals(possible_actions.loc["a1", :]) == True:
            possible_actions.drop("a1", axis=0, inplace=True)

        self.possible_actions.loc[
            self.current_state, possible_actions.index
        ] = possible_actions

        logger.info(" _________________________________________________")
        logger.info("|												  |")
        logger.info("|		  ---> AVAILABLE ACTION SPACE <---        |")
        logger.info("|_________________________________________________|\n")

        logger.info("Possible actions to choose from:\n{}\n".format(possible_actions))

        weights = self.action_probabilties_per_state.loc[
            self.current_state, possible_actions.index
        ]
        if weights.sum() == 0:
            weights[:] = 1 / len(weights)

        # RANDOM ACTION ==> ALWAYS IF EPSILON IS 0
        if "Epsilon_Greedy" in self.exploration:

            random_value = random.uniform(0, 1)

            # random_value = -1
            print("self.epsilon: ", self.epsilon)
            if random_value >= self.epsilon:

                # Get random action values from possible actions
                action_sample = possible_actions.sample(
                    1, weights=weights
                )
                action_name = action_sample.index
                action_values = list(flatten(action_sample.T.values))

                self.activation_periods = Series(index=mask.index, data=action_values)
                self.current_action = action_name
                self.exploration_choice[_profile.index[0][0]] = "RANDOM"

                logger.info(" _________________________________________________")
                logger.info("|												  |")
                logger.info("|		        RANDOM SELECTION   		          |")
                logger.info("|		            ACTION: {}		              |".format(
                        self.current_action[0]
                    )
                )
                logger.info("|_________________________________________________|\n")

            # MAX Q ACTION -> IF GREATER THAN EPSILON ==> ALWAYS IF EPSILON IS 1
            elif random_value < self.epsilon:

                # Get action key with highest Q-value
                self.current_action = Index(
                    [
                        self.q_values.loc[
                            self.current_state, possible_actions.index
                        ].idxmax()
                    ]
                )
                action_values = list(
                    flatten(possible_actions.loc[self.current_action].T.values)
                )
                self.activation_periods = Series(index=mask.index, data=action_values)

                self.exploration_choice[_profile.index[0][0]] = "GREEDY"

                logger.info("_________________________________________________")
                logger.info("|												  |")
                logger.info("|		  GREEDY SELECTION BASED ON MAX Q 		  |")
                logger.info(
                    "|		            ACTION: {}		              |".format(
                        self.current_action[0]
                    )
                )
                logger.info("|_________________________________________________|\n")

            logger.info(
                "Flexibility selected at prediction periods (rows):\n{}\n".format(
                    self.activation_periods
                )
            )

            q_values = deepcopy(
                self.q_values.loc[self.current_state, possible_actions.index]
            )

        elif "UCB1" in self.exploration:

            q_values = deepcopy(
                self.q_values.loc[self.current_state, possible_actions.index]
            )

            performed_actions_cumulated = self.performed_actions_cumulated.loc[
                self.current_state, possible_actions.index
            ].copy()

            action_probabilties_per_state = self.action_probabilties_per_state.loc[
                self.current_state, possible_actions.index
            ]

            if self.epsilon >= 0.09:

                performed_actions_cumulated[performed_actions_cumulated == 0] = 0.1

                df = performed_actions_cumulated.apply(
                    lambda x: math.sqrt((2 * math.log(self.current_episode)) / x)
                )

                q_values = q_values + df

                # Get action key with highest Q-value
                self.current_action = Index([q_values.idxmax()])
                action_values = list(
                    flatten(possible_actions.loc[self.current_action].T.values)
                )
                self.activation_periods = Series(index=mask.index, data=action_values)

                logger.info("_________________________________________________")
                logger.info("|												  |")
                logger.info("|		         UCB1 SELECTION            		  |")
                logger.info(
                    "|		            ACTION: {}		              |".format(
                        self.current_action[0]
                    )
                )
                logger.info("|_________________________________________________|\n")

                logger.warning("UCB1 chosen\n")

            else:

                # Get random action values from possible actions
                action_sample = possible_actions.sample(
                    1, weights=weights
                )  
                action_name = action_sample.index
                action_values = list(flatten(action_sample.T.values))

                self.activation_periods = Series(index=mask.index, data=action_values)
                self.current_action = action_name

                logger.info(" _________________________________________________")
                logger.info("|												  |")
                logger.info("|		        RANDOM SELECTION   		          |")
                logger.info(
                    "|		            ACTION: {}		              |".format(
                        self.current_action[0]
                    )
                )
                logger.info("|_________________________________________________|\n")

                logger.warning("RANDOM chosen\n")


            logger.info(
                "Flexibility selected at prediction periods (rows):\n{}\n".format(
                    self.activation_periods
                )
            )

        # ------------------------------------------------------------------------------------------#
        print("Q values: \n{}\n".format(self.q_values.head(24)))
        # ------------------------------------------------------------------------------------------#

        if "a0" in self.current_action:
            _profile.at[IDX[:, 2:], IDX[:, :]] = np.nan

            p_0 = _profile.loc[IDX[:, 1], IDX[:, :]].values[0][0]

            if isna(p_0) == False:
                self.activation_periods.iloc[0] = True

        else:
            self.activation_periods[:] = mask.values

        logger.info(
            "Flexoffer profile after EPSILON-Greedy-selection:\n\n{}\n".format(_profile)
        )
        logger.info("Aggregator sends updated profile to FDP")

        return _profile

    def update(self, reward: float, current_datetime: datetime):

        """
            Parameter:

                reward: float 
                    The amount of cash the FDP is willing to pay for the selected flexoffer profile

        """
        logger.info(
            "Aggregator starts to update Q-learning parameter in EPISODE {}\n".format(
                self.current_episode
            )
        )

        self.current_datetime = current_datetime

        """ _____________________________________________________________  ACTION COUNTER UPDATE """

        # Update performed action table
        self.performed_actions.loc[self.current_state, self.current_action] = 1

        # Update cumulated performed action table
        self.performed_actions_cumulated.loc[
            self.current_state, self.current_action
        ] += 1

        logger.info(
            ":::: EPISODE {} - Applied action in current state: {}".format(
                self.current_episode, self.current_action[0]
            )
        )

        logger.info(
            ":::: EPISODE {} - Aggregator updates action count table:\n\n{}\n".format(
                self.current_episode,
                self.performed_actions.loc[self.current_state, :].to_frame().T,
            )
        )

        """ _____________________________________________________________  ACTION PROBABILITY UPDATE """

        performed_actions = self.performed_actions_cumulated.loc[self.current_state, :]
        count_current_action_at_current_state = self.performed_actions_cumulated.loc[
            self.current_state, self.current_action
        ].values

        possible_actions = self.possible_actions.loc[self.current_state, :]

        """ ________________________________  PARAMETER TUNING """

        """ ________________________________  ALPHA """

        # If action a1..
        if int(str(self.current_action[0]).split("a")[1]) > 0:
            self.alpha = self.alpha_1

        # If action a0..
        else:

            self.alpha = self.alpha_0

            self.alpha_data[self.current_datetime] = self.alpha

        """ ________________________________  ACTION PROBABILITY """

        p = (
            math.log(count_current_action_at_current_state)
            * self.action_probabilites_decay
        )

        if (
            self.action_probabilties_per_state.loc[
                self.current_state, self.current_action
            ].values
            > 1
        ):

            self.action_probabilties_per_state.loc[
                self.current_state, self.current_action
            ] -= p

        else:

            self.action_probabilties_per_state.loc[
                self.current_state, self.current_action
            ] = 1

        if (
            self.action_probabilties_per_state.loc[
                self.current_state, self.current_action
            ].values
            < 1
        ):

            self.action_probabilties_per_state.loc[
                self.current_state, self.current_action
            ] = 1

        """ _____________________________________________________________  TARGET CONSTRAINT UPDATE """

        # Compute next state
        next_state = (
            (current_datetime + self.resolution).strftime("%H:%M:%S")
            + "_"
            + self.current_action[0]
        )

        possible_next_q_values = self.q_values.loc[next_state, :]

        # Get max reward of next state
        max_possible_next_Q_value = np.max(possible_next_q_values)

        logger.info(
            ":::: EPISODE {} - Observed reward {}".format(self.current_episode, reward,)
        )

        """ _____________________________________________________________  REWARD STORING """

        # Update reward table
        self.reward_values.loc[self.current_state, self.current_action] = reward

        logger.info(
            ":::: EPISODE {} - Updated reward table:\n\n{}\n".format(
                self.current_episode,
                self.reward_values.loc[self.current_state, :].to_frame().T,
            )
        )

        """ _____________________________________________________________  Q VALUE UPDATE """

        if current_datetime.strftime("%H:%M:%S") not in self.termination_time:

            # Current Q value (for current state and performed action)
            current_q_value = self.q_values.loc[
                self.current_state, self.current_action
            ].values[0]

            logger.info(
                ":::: EPISODE {} - Current q value {} for action: {}".format(
                    self.current_episode, current_q_value, self.current_action[0]
                )
            )

            # Update value for current state's Q(S,A) pair
            self.q_values.loc[self.current_state, self.current_action] = (
                current_q_value
                + self.alpha
                * (reward + self.gamma * max_possible_next_Q_value - current_q_value)
            )

        else:
            self.q_values.loc[self.current_state, self.current_action] = 0
            # self.store_data

        logger.info(
            ":::: EPISODE {} - Updated q value {} for action: {}".format(
                self.current_episode,
                self.q_values.loc[self.current_state, self.current_action].values,
                self.current_action[0],
            )
        )

        logger.info(
            ":::: EPISODE {} - Updated Q-table:\n\n{}\n".format(
                self.current_episode,
                self.q_values.loc[self.current_state, :].to_frame().T,
            )
        )

        """ _____________________________________________________________  CURRENT STATE UPDATE """

        # Update current state
        self.current_state = next_state

        logger.info(
            ":::: EPISODE {} - Update current state from {} ---> {}".format(
                self.current_episode,
                current_datetime.strftime("%H:%M:%S"),
                self.current_state,
            )
        )

        """ _____________________________________________________________  DATA STORE """

        self.epsilon_data.loc[current_datetime] = self.epsilon
        self.reward_per_datetime.loc[current_datetime] = reward

        return

    @property
    def reset(self):

        # Store adaptive strategy data per episode
        self.store_data

        episode_reward = self.reward_values.sum().sum()

        """ ________________________________  PARAMETER TUNING """

        """ ________________________________  REWARD BOOST """

        if episode_reward > self.max_reward:  #
            self.max_reward = episode_reward

        """ _____________________________________________________________  EPSILON DECAY UPDATE """

        # Increase epsilon if episode lays within decaying range
        if self.end_epsilon_decay >= self.current_episode > self.start_epsilon_decay:
            self.epsilon += self.epsilon_decay_value

        """ _____________________________________________________________  ALPHA DECAY UPDATE """

        # Create fresh dataframes for rewards and performed action at each new episode
        self.reward_values, self.performed_actions = [
            DataFrame(
                index=self.daily_date_range,
                columns=[
                    "a" + str(x) for x in range(self.possible_flexibility_activations)
                ],
                data=0,
            )
            for x in range(2)
        ]

        self.current_state = self.daily_date_range[0]

    @property
    def store_data(self):

        for episode_table, df in zip(
            [
                self.q_values,
                self.performed_actions,
                self.reward_values,
                self.action_probabilties_per_state,
            ],
            [
                self.q_values_data,
                self.performed_actions_data,
                self.reward_values_data,
                self.action_probabilties_per_state_data,
            ],
        ):

            df.loc[self.current_episode] = episode_table.values

        logger.info(
            ":::: EPISODE {} - Stored episode data".format(self.current_episode)
        )


class MaxSelection:
    def __init__(
        self,
        simulation_start: datetime,
        simulation_end: datetime,
        episode_duration: timedelta,
        resolution: Timedelta,
        episodes: int,
        prediction_delta: Timedelta,
    ):

        # SIMULATION TIME
        self.start = simulation_start
        self.end = simulation_end
        self.resolution = resolution

        # EPISODES
        self.last_episode = episodes

        # self.current_episode = 1
        self.n_steps = 0

        # INIT STATE
        self.current_state = simulation_start.strftime("%H:%M:%S")

        # LEARNING RATE
        self.alpha = 0

        # FUTRURE REWARD DISCOUNT RATE
        self.gamma = 0

        # GREEDY ACTION SELECTION PARAMETER
        self.epsilon = 0
        self.start_epsilon_decay = 0
        self.end_epsilon_decay = 0

        # Get the horizon span as steps (== degrees of freedom)
        self.prediction_steps = int(prediction_delta / self.resolution) - 1

        # Flexibility at current simulation time always gets sold, thus subtract one prediction step
        self.possible_flexibility_activations = 2 ** self.prediction_steps

        self.daily_date_range = date_range(
            start="00:00", end="23:59", freq=resolution, closed="left", name="Time"
        ).strftime("%H:%M:%S")

        steps_full_day = timedelta(hours=1) / resolution * 24

        episode_steps = ((self.start + episode_duration) - self.start) / self.resolution


        if episode_steps < steps_full_day - 1:
            self.termination_time = self.daily_date_range[episode_steps]

        else:
            self.termination_time = self.daily_date_range[steps_full_day - 1]

        logger.info(
            "\n\t\t\t\t\t\tEpisode termination at: {}".format(self.termination_time)
        )

        self.action_values = DataFrame(
            index=[
                "a" + str(x)
                for x in range(1, self.possible_flexibility_activations + 1)
            ],
            columns=[x for x in range(2, self.prediction_steps + 2)],
            data=[
                list(i)
                for i in itertools.product([False, True], repeat=self.prediction_steps)
            ],
        )

        # Create dataframe that counts the number of times an action got called
        self.performed_actions, self.q_values, self.reward_values = [
            DataFrame(
                index=self.daily_date_range,
                columns=[
                    "a" + str(x)
                    for x in range(1, self.possible_flexibility_activations + 1)
                ],
                data=0,
            )
            for x in range(3)
        ]

        episodes_datetimes_lists = [
            date_range(
                start=simulation_start + timedelta(days=x),
                periods=steps_full_day,
                freq=self.resolution,
                closed="left",
                name="Time",
            ).tolist()
            for x in range(self.last_episode)
        ]

        episode_numbers_list = list(range(1, self.last_episode + 1))
        tuples = []
        for episode in episode_numbers_list:
            for _datetime in episodes_datetimes_lists[episode - 1]:
                tuples.append((episode, _datetime))

        row_multiindex = MultiIndex.from_tuples(tuples)

        self.performed_actions_data, self.q_values_data, self.reward_values_data = [
            DataFrame(
                index=row_multiindex,
                columns=[
                    "a" + str(x)
                    for x in range(1, self.possible_flexibility_activations + 1)
                ],
            )
            for x in range(3)
        ]

        self.epsilon_data = Series(
            data=0,
            index=date_range(
                start=simulation_start,
                end=simulation_end,
                freq=resolution,
                closed="left",
                name="Time",
            ),
        )

        self.reward_per_datetime = Series(
            index=date_range(
                start=simulation_start,
                end=simulation_end,
                freq=resolution,
                closed="left",
                name="Time",
            )
        )

    def select_action(self, profile):

        logger.info(
            "\nAggregator selects flexibility from aggregated offer based on EPSILON-Greedy-Policy\n"
        )

        # Store copy for return
        _profile = profile.copy()

        # Remove unused levels
        profile.columns = profile.columns.droplevel(level=0)
        profile.index = profile.index.droplevel(level=0)

        # Get indices of profile periods where no flexibility is available
        # NOTE: Periods with no flexibility are marked as "False"
        mask = profile["Offer"].notnull()
        logger.info("Flexibility available at:\n{}".format(mask))

        # Copy all possible actions array
        possible_actions = self.action_values.copy()

        logger.info(" _________________________________________________")
        logger.info("|												  |")
        logger.info("|	   ---> ---> FULL ACTION SPACE <--- <---      |")
        logger.info("|_________________________________________________|\n")

        logger.info(
            "Possible flexoffer combinations (rows) over prediction horizon periods (cols):\n{}\n".format(
                possible_actions
            )
        )

        # Cut action space at horizon steps where no profile flexibility exists
        for idx, mask_row in mask.iloc[1:].iteritems():

            logger.info(
                ">>>> Flexibility available at period {} ---> {}\n".format(
                    idx, mask_row
                )
            )

            if mask_row == False:
                possible_actions = possible_actions[possible_actions[idx] == mask_row]

                logger.info(
                    "Aggregator sliced action space to:\n{}\n".format(possible_actions)
                )

        logger.info(" _________________________________________________")
        logger.info("|												  |")
        logger.info("|		  ---> AVAILABLE ACTION SPACE <---        |")
        logger.info("|_________________________________________________|\n")

        logger.info("Possible actions to choose from:\n{}\n".format(possible_actions))

        possible_actions = possible_actions.T
        possible_actions = possible_actions[possible_actions.isin(mask[1:])]
        possible_actions.dropna(axis=1, inplace=True)
        action_sample = possible_actions.T

        action_name = action_sample.index
        action_values = list(flatten(action_sample.T.values))

        self.activation_periods = Series(index=mask.iloc[1:].index, data=action_values)
        self.current_action = action_name

        logger.info(" _________________________________________________")
        logger.info("|												  |")
        logger.info("|		        OFFER SELECTION   		          |")
        logger.info(
            "|		            ACTION: {}		              |".format(self.current_action[0])
        )
        logger.info("|_________________________________________________|\n")

        logger.info(
            "Flexibility selected at prediction periods (rows):\n{}\n".format(
                self.activation_periods
            )
        )

        # Set offer values to nan if not selected in activation periods
        for idx, row in self.activation_periods.iteritems():

            if row == False:
                _profile.at[IDX[:, idx], IDX[:, :]] = np.nan

        logger.info("Flexoffer profile after MaxSelection:\n\n{}\n".format(_profile))
        logger.info("Aggregator sends updated profile to FDP")

        return _profile

    def update(self, reward: float, current_datetime: datetime):

        """
            Parameter:

                reward: float 
                    The amount of cash the FDP is willing to pay for the selected flexoffer profile

        """
        logger.info(
            "Aggregator starts to update Q-learning parameter in EPISODE {}\n".format(
                self.current_episode
            )
        )

        # Update performed action table
        self.performed_actions.loc[self.current_state, self.current_action] = 1

        logger.info(
            ":::: EPISODE {} - Applied action in current state: {}".format(
                self.current_episode, self.current_action[0]
            )
        )

        logger.info(
            ":::: EPISODE {} - Aggregator updates action count table:\n\n{}\n".format(
                self.current_episode,
                self.performed_actions.loc[self.current_state, :].to_frame().T,
            )
        )

        logger.warning(
            ":::: EPISODE {} - reward {}".format(self.current_episode, reward,)
        )

        # Update reward table
        self.reward_values.loc[self.current_state, self.current_action] = reward

        logger.info(
            ":::: EPISODE {} - Updated reward table:\n\n{}\n".format(
                self.current_episode,
                self.reward_values.loc[self.current_state, :].to_frame().T,
            )
        )

        if current_datetime.strftime("%H:%M:%S") not in self.termination_time:

            pass

        else:
            # self.q_values.loc[self.current_state, self.current_action] = 0
            self.store_data

        # Update current state
        self.current_state = (current_datetime + self.resolution).strftime("%H:%M:%S")

        self.reward_per_datetime[current_datetime] = reward

        logger.info(
            ":::: EPISODE {} - Update current state from {} ---> {}".format(
                self.current_episode,
                current_datetime.strftime("%H:%M:%S"),
                self.current_state,
            )
        )

        # Decaying is being done every episode if episode number is within decaying range
        if self.end_epsilon_decay >= self.current_episode >= self.start_epsilon_decay:
            self.epsilon += self.epsilon_decay_value

        return

    @property
    def reset(self):
        # Store adaptive strategy data per episode
        self.store_data

        # Create fresh dataframes for rewards and performed action at each new episode
        self.reward_values, self.performed_actions = [
            DataFrame(
                index=self.daily_date_range,
                columns=[
                    "a" + str(x)
                    for x in range(1, self.possible_flexibility_activations + 1)
                ],
                data=0,
            )
            for x in range(2)
        ]

        self.current_state = self.start.strftime("%H:%M:%S")

    @property
    def store_data(self):

        for episode_table, df in zip(
            [self.performed_actions, self.reward_values],
            [self.performed_actions_data, self.reward_values_data],
        ):

            df.loc[self.current_episode] = episode_table.values

        logger.info(
            ":::: EPISODE {} - Stored episode data".format(self.current_episode)
        )
