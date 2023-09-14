import logging
import gc
from enum import Enum
from gymnasium import spaces
from pandas import DataFrame
import numpy as np
import torch as th
import pandas as pd

import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt

from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment, Positions
from freqtrade.strategy import timeframe_to_minutes

logger = logging.getLogger(__name__)


class Actions(Enum):
    DOUBLE_SELL = 0
    SELL = 1
    HOLD = 2
    BUY = 3
    DOUBLE_BUY = 4


class CoolActionRLEnv(BaseEnvironment):
    """
    Base class for a 5 action environment
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Enable force trade exits by take profit, stop loss and timeout
        self.force_exits: bool = self.rl_config.get('force_exits', False)
        self.take_profit = round(self.profit_aim * self.rr, 3)
        self.stop_loss = round(self.profit_aim * -1, 3)
        self.timeout_candles: int = self.rl_config.get('max_trade_duration_candles', 100)
        self.timeout_minutes: int = (self.timeout_candles *
                                     timeframe_to_minutes(self.config['timeframe']))
        if self.force_exits:
            logger.info(
                f"Force exits enabled, take_profit: {self.take_profit}, "
                f"stop_loss: {self.stop_loss}, "
                f"timeout: {self.timeout_candles} candles ({self.timeout_minutes} min)"
            )

    def set_action_space(self):
        self.action_space = spaces.Discrete(len(Actions))

    def reset(self) -> DataFrame:
        obs, history = super().reset()
        self._last_closed_trade_tick: int = None
        self.max_trades_count: int = (self._end_tick - self._start_tick) / len(Positions)
        self.max_possible_profit = self.get_max_possible_profit(self.can_short)
        self.tensorboard_log("max_possible_profit", self.max_possible_profit)
        self.tensorboard_log("max_trades_count", int(self.max_trades_count))
        return obs, history

    def step(self, action: int):
        """
        Logic for a single step by the agent
        """
        self._current_tick += 1
        self._update_unrealized_total_profit()
        self.tensorboard_log(self.actions._member_names_[action], category="actions")

        reward = self.calculate_reward(action)
        info = self.get_info()

        executed_trade_type = self.execute_trade(action)

        if self.force_exits and executed_trade_type is None:
            executed_trade_type = self.execute_force_exit_trade()

        self.total_reward += reward
        self.update_trade_history(executed_trade_type)
        self._position_history.append(self._position)
        self._update_history(info)
        truncated = False

        return self._get_observation(), reward, self.is_done(), truncated, info


    def execute_trade(self, action) -> str:
        if not self.is_tradesignal(action):
            return None

        executed_trade_type = None
        if action == Actions.BUY.value:
            self._position = Positions.Long
            executed_trade_type = "open_long_postion"
            self._last_trade_tick = self._current_tick
        elif action == Actions.SELL.value:
            self._position = Positions.Short
            executed_trade_type = "open_short_postion"
            self._last_trade_tick = self._current_tick
        elif action == Actions.DOUBLE_SELL.value:
            if self._position == Positions.Long:
                self._update_total_profit()
            self._position = Positions.Short
            executed_trade_type = "close_long_postion_and_open_short_postion"
            self._last_closed_trade_tick = self._last_trade_tick
            self._last_trade_tick = self._current_tick
        elif action == Actions.DOUBLE_BUY.value:
            if self._position == Positions.Short:
                self._update_total_profit()
            self._position = Positions.Long
            executed_trade_type = "close_short_postion_and_open_long_postion"
            self._last_closed_trade_tick = self._last_trade_tick
            self._last_trade_tick = self._current_tick
        elif action == Actions.HOLD.value:
            self._update_total_profit()
            self._position = Positions.Neutral
            self._last_closed_trade_tick = self._last_trade_tick
            self._last_trade_tick = None

        return executed_trade_type


    def execute_force_exit_trade(self) -> str:
        if self._position == Positions.Neutral:
            return None

        executed_trade_type = None
        pnl = self.get_unrealized_profit()
        trade_duration = self.get_trade_duration()

        if pnl >= self.take_profit:
            executed_trade_type = "take_profit"
        elif pnl <= self.stop_loss:
            executed_trade_type = "stop_loss"
        elif trade_duration >= self.timeout_candles:
            executed_trade_type = "timeout_profit" if pnl > 0 else "timeout_loss"

        if executed_trade_type:
            self._update_total_profit()
            self._position = Positions.Neutral
            self._last_closed_trade_tick = self._last_trade_tick
            self._last_trade_tick = None
            self.tensorboard_log(executed_trade_type, category="force_exit")
            return executed_trade_type

    def update_trade_history(self, executed_trade_type) -> None:
        if executed_trade_type is not None:
            self.trade_history.append({
                'tick': self._current_tick,
                'price': self.current_price(),
                'type': executed_trade_type,
                'profit': self.get_unrealized_profit()
            })

    def get_info(self) -> dict:
        return dict(
            tick=self._current_tick,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
            trade_duration=self.get_trade_duration(),
            current_profit_pct=self.get_unrealized_profit(),
            trades_count=len(self.trade_history)
        )

    def is_done(self) -> bool:
        return True if (
            self._current_tick == self._end_tick or
            self._total_profit <= self.max_drawdown or
            self._total_unrealized_profit <= self.max_drawdown or
            len(self.trade_history) >= self.max_trades_count or
            self._total_profit >= self.max_possible_profit
        ) else False

    def is_tradesignal(self, action: int) -> bool:
        """
        Determine if the action is entry or exit
        """
        if action == Actions.SELL:

            if self._position == Positions.Long:
                return  False

            if self._position == Positions.Neutral:
                return True

        if action == Actions.BUY:

            if self._position == Positions.Short:
                return  False

            if self._position == Positions.Neutral:
                return  True

        if action == Actions.DOUBLE_SELL and (self._position == Positions.Long or self._position == Positions.Neutral):
            return True

        if action == Actions.DOUBLE_BUY and (self._position == Positions.Short or self._position == Positions.Neutral):
            return True

        if action == Actions.HOLD and self._position in (Positions.Long or Positions.Short):
            return True

        return False

    def get_feature_value(self, name: str,
                          period: int = None, shift: int = None, timeframe: str = None,
                          normalized=False) -> float:
        """
        Get the raw or normalized value of the feature on the current tick
        """
        period = f"-period_{period}" if period else ""
        shift = f"_shift-{shift}" if shift else ""
        pair = self.pair.replace(":", "")
        timeframe = self.config["timeframe"] if not timeframe else timeframe

        feature_col = f"%-{name}{period}{shift}_{pair}_{timeframe}"

        if normalized:
            return self.signal_features[feature_col].iloc[self._current_tick]
        return self.raw_features[feature_col].iloc[self._current_tick]

    def get_idle_duration(self) -> int:
        """
        Get idle ticks since the last closed trade tick
        """
        if self._last_closed_trade_tick is None:
            return self._current_tick - self._start_tick
        return self._current_tick - self._last_closed_trade_tick

    def get_most_recent_return(self) -> float:
        """
        Calculate the tick to tick return if in a trade.
        Return is generated from rising prices in Long
        and falling prices in Short positions.
        The actions Sell/Buy or Hold during a Long position trigger the sell/buy-fee.
        """
        if self._position == Positions.Long:
            previous_price = self.previous_price()
            if (self._position_history[self._current_tick - 1] == Positions.Short
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_entry_fee(previous_price)
            return np.log(self.current_price()) - np.log(previous_price)
        elif self._position == Positions.Short:
            previous_price = self.previous_price()
            if (self._position_history[self._current_tick - 1] == Positions.Long
                    or self._position_history[self._current_tick - 1] == Positions.Neutral):
                previous_price = self.add_exit_fee(previous_price)
            return np.log(previous_price) - np.log(self.current_price())
        return 0.

    def get_most_recent_profit(self):
        """
        Calculate the tick to tick unrealized profit if in a trade
        """
        if self._position == Positions.Long:
            current_price = self.add_exit_fee(self.current_price())
            previous_price = self.add_entry_fee(self.previous_price())
            return (current_price - previous_price) / previous_price
        elif self._position == Positions.Short:
            current_price = self.add_entry_fee(self.current_price())
            previous_price = self.add_exit_fee(self.previous_price())
            return (previous_price - current_price) / previous_price
        return 0

    def get_max_possible_profit(self, can_short: bool = True) -> float:
        """
        The maximum possible profit that an RL agent can obtain
        Modified from https://github.com/AminHP/gym-anytrading
        """
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        max_possible_profit = 1.

        while current_tick <= self._end_tick:
            position = None

            if self.prices.iloc[current_tick].open < self.prices.iloc[current_tick - 1].open:
                while (current_tick <= self._end_tick and
                        self.prices.iloc[current_tick].open < self.prices.iloc[current_tick - 1].open):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                        self.prices.iloc[current_tick].open >= self.prices.iloc[current_tick - 1].open):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Short and can_short:
                current_price = self.add_entry_fee(self.prices.iloc[current_tick - 1].open)
                last_trade_price = self.add_exit_fee(self.prices.iloc[last_trade_tick].open)
                pnl = (last_trade_price - current_price) / last_trade_price
                max_possible_profit += pnl

            elif position == Positions.Long:
                current_price = self.add_exit_fee(self.prices.iloc[current_tick - 1].open)
                last_trade_price = self.add_entry_fee(self.prices.iloc[last_trade_tick].open)
                pnl = (current_price - last_trade_price) / last_trade_price
                max_possible_profit += pnl

            last_trade_tick = current_tick - 1

        return max_possible_profit

    def previous_price(self) -> float:
        return self.prices.iloc[self._current_tick - 1].open

    def get_rollout_history(self) -> DataFrame:
        """
        Get environment data from the first to the last trade
        """
        _history_df = pd.DataFrame.from_dict(self.history)
        _trade_history_df = pd.DataFrame.from_dict(self.trade_history)
        _rollout_history = _history_df.merge(_trade_history_df, on="tick", how="left")
        _price_history = self.prices.iloc[_rollout_history.tick].copy().reset_index()

        history = pd.merge(
            _rollout_history,
            _price_history,
            left_index=True, right_index=True
        )
        return history

    def get_rollout_plot(self):
        """
        Plot trades and environment data
        """
        def transform_y_offset(ax, offset):
            return mtransforms.offset_copy(ax.transData, fig=fig, x=0, y=offset, units="inches")

        def plot_markers(ax, ticks, marker, color, size, offset):
            ax.plot(ticks, marker=marker, color=color, markersize=size, fillstyle="full",
                    transform=transform_y_offset(ax, offset), linestyle="none")

        plt.style.use("dark_background")
        fig, axs = plt.subplots(
            nrows=5, ncols=1,
            figsize=(16, 9),
            height_ratios=[6, 1, 1, 1, 1],
            sharex=True
        )

        # Return empty fig if no trades
        if len(self.trade_history) == 0:
            return fig

        history = self.get_rollout_history()
        enter_long_prices = history.loc[history["type"] == "enter_long"]["price"]
        enter_short_prices = history.loc[history["type"] == "enter_short"]["price"]
        exit_long_prices = history.loc[history["type"] == "exit_long"]["price"]
        exit_short_prices = history.loc[history["type"] == "exit_short"]["price"]

        axs[0].plot(history["open"], linewidth=1, color="#c28ce3")
        plot_markers(axs[0], enter_long_prices, "^", "#4ae747", 5, -0.05)
        plot_markers(axs[0], enter_short_prices, "v", "#f53580", 5, 0.05)
        plot_markers(axs[0], exit_long_prices, "o", "#73b3e4", 3, 0)
        plot_markers(axs[0], exit_short_prices, "o", "#73b3e4", 3, 0)

        axs[1].set_ylabel("pnl")
        axs[1].plot(history["current_profit_pct"], linewidth=1, color="#a29db9")
        axs[1].axhline(y=0, label='0', alpha=0.33)
        axs[2].set_ylabel("duration")
        axs[2].plot(history["trade_duration"], linewidth=1, color="#a29db9")
        axs[3].set_ylabel("total_reward")
        axs[3].plot(history["total_reward"], linewidth=1, color="#a29db9")
        axs[3].axhline(y=0, label='0', alpha=0.33)
        axs[4].set_ylabel("total_profit")
        axs[4].set_xlabel("tick")
        axs[4].plot(history["total_profit"], linewidth=1, color="#a29db9")
        axs[4].axhline(y=1, label='1', alpha=0.33)

        for _ax in axs:
            for _border in ["top", "right", "bottom", "left"]:
                _ax.spines[_border].set_color("#5b5e4b")

        fig.suptitle(
            "Total Reward: %.6f" % self.total_reward + " ~ " +
            "Total Profit: %.6f" % self._total_profit
        )
        fig.tight_layout()

        return fig

    def close(self) -> None:
        gc.collect()
        th.cuda.empty_cache()


def transform(position: Positions, action: int):
    '''
    Overview:
        used by env.tep().
        This func is used to transform the env's position from
        the input (position, action) pair according to the status machine.
    Arguments:
        - position(Positions) : Long, Short or Flat
        - action(int) : Doulbe_Sell, Sell, Hold, Buy, Double_Buy
    Returns:
        - next_position(Positions) : the position after transformation.
    '''
    if action == Actions.SELL:

        if position == Positions.Long:
            return Positions.Neutral, False, ""

        if position == Positions.Neutral:
            return Positions.Short, True, "enter_short"

    if action == Actions.BUY:

        if position == Positions.Short:
            return Positions.Neutral, False, ""

        if position == Positions.Neutral:
            return Positions.Long, True, "enter_long"

    if action == Actions.DOUBLE_SELL and (position == Positions.Long or position == Positions.Neutral):
        return Positions.Short, True, "enter_double_short"

    if action == Actions.DOUBLE_BUY and (position == Positions.Short or position == Positions.Neutral):
        return Positions.Long, True, "enter_double_long"

    return position, False