import logging
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.freqai.prediction_models.ReinforcementLearner import ReinforcementLearner
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions, Base5ActionRLEnv, Positions
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms

logger = logging.getLogger(__name__)

class MyCoolRLModel(ReinforcementLearner):
    """
    User created Reinforcement Learning Model prediction model.
    """

    class MyRLEnv(Base5ActionRLEnv):
        """
        User can override any function in BaseRLEnv and gym.Env. Here the user
        sets a custom reward based on profit and trade duration.
        """

        def calculate_reward(self, action: int) -> float:
            """
            An example reward function. This is the one function that users will likely
            wish to inject their own creativity into.
            :param action: int = The action made by the agent for the current candle.
            :return:
            float = the reward to give to the agent for current step (used for optimization
                of weights in NN)
            """
            # first, penalize if the action is not valid
            if not self._is_valid(action):
                self.tensorboard_log("invalid", category="actions")
                return -2

            pnl = self.get_unrealized_profit()
            factor = 100.

            # reward agent for entering trades
            if (action == Actions.Long_enter.value
                    and self._position == Positions.Neutral):
                return 25
            if (action == Actions.Short_enter.value
                    and self._position == Positions.Neutral):
                return 25
            # discourage agent from not entering trades
            if action == Actions.Neutral.value and self._position == Positions.Neutral:
                return -1

            max_trade_duration = self.rl_config.get('max_trade_duration_candles', 300)
            trade_duration = self._current_tick - self._last_trade_tick  # type: ignore

            if trade_duration <= max_trade_duration:
                factor *= 1.5
            elif trade_duration > max_trade_duration:
                factor *= 0.5

            # discourage sitting in position
            if (self._position in (Positions.Short, Positions.Long) and
                    action == Actions.Neutral.value):
                return -1 * trade_duration / max_trade_duration

            # close long
            if action == Actions.Long_exit.value and self._position == Positions.Long:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)

            # close short
            if action == Actions.Short_exit.value and self._position == Positions.Short:
                if pnl > self.profit_aim * self.rr:
                    factor *= self.rl_config['model_reward_parameters'].get('win_reward_factor', 2)
                return float(pnl * factor)

            return 0.

        def render(self, mode="human"):

            def transform_y_offset(ax, offset):
                    return mtransforms.offset_copy(
                        ax.transData, fig=fig,
                        x=0, y=offset, units="inches")

            def plot_markers(ax, ticks: DataFrame, marker, color, size, offset):
                ax.plot(
                    ticks,
                    marker=marker, color=color,
                    markersize=size, fillstyle="full",
                    transform=transform_y_offset(ax, offset),
                    linestyle="none")

            plt.style.use("dark_background")
            fig, axs = plt.subplots(
                nrows=5, ncols=1,
                figsize=(21, 9),
                height_ratios=[6, 1, 1, 1, 1],
                sharex=True
            )

            _history = DataFrame.from_dict(self.history)
            _trade_history = DataFrame.from_dict(self.trade_history)
            try:
                _rollout_history = _history.merge(_trade_history, left_on="tick",right_on="index", how="left")
            except KeyError:
                return fig  # return empty plot if no trades
            _price_history = self.prices.iloc[_rollout_history.tick].copy()

            history = None
            try:
                history = pd.merge(
                    _rollout_history, _price_history.reset_index(),
                    left_index=True, right_index=True
                )
            except Exception as e:
                return  fig

            axs[0].plot(history["open"], linewidth=1, color="#c28ce3")
            plot_markers(axs[0], history.loc[history["type"] == "enter_short"]
                        ["price"], "v", "#f53580", 7, 0.1)
            plot_markers(axs[0], history.loc[history["type"] == "enter_long"]
                        ["price"], "^", "#4ae747", 7, -0.1)
            plot_markers(axs[0], history.loc[history["type"] == "exit_long"]
                        ["price"], "<", "#4ae747", 5, 0)
            plot_markers(axs[0], history.loc[history["type"] == "exit_short"]
                        ["price"], ">", "#f53580", 5, 0)

            axs[1].plot(history["position"], linewidth=1, color="#a29db9")
            axs[1].set_ylabel("position")

            axs[2].plot(history["trade_duration"], linewidth=1, color="#a29db9")
            axs[2].set_ylabel("duration")

            axs[3].axhline(y=self.profit_aim * self.rr, color='#4ae747', label='take_profit', alpha=0.33)
            axs[3].plot(history["current_profit_pct"], linewidth=1, color="#a29db9")
            axs[3].axhline(y=-1 * self.profit_aim, color='#f53580', label='stop_loss', alpha=0.33)
            axs[3].set_ylabel("pnl")

            axs[4].plot(history["total_profit"], linewidth=1, color="#a29db9")
            axs[4].axhline(y=self.max_drawdown, color='#f53580', label='max_drawdown', alpha=0.33)
            axs[4].set_ylabel("total_profit")
            axs[4].set_xlabel("tick")

            for _ax in axs:
                for _border in ["top", "right", "bottom", "left"]:
                    _ax.spines[_border].set_color("#5b5e4b")

            fig.suptitle(
                "Total Reward: %.6f" % self.total_reward + " ~ " +
                "Total Profit: %.6f" % self._total_profit
            )
            fig.tight_layout()

            return fig