import logging
from typing import List
import numpy as np
import pandas as pd
from pandas import DataFrame
from freqtrade.freqai.RL.Base4ActionRLEnv import Actions, Base4ActionRLEnv, Positions
from ReforceXBaseModel import ReforceXBaseModel
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms


logger = logging.getLogger(__name__)


class ReforceXModel4ac(ReforceXBaseModel):
    """
    ReforceX Pack - 4ac env
    """

    class MyRLEnv(Base4ActionRLEnv):

        metadata = {"render.modes": ["human"]}

        def action_masks(self) -> List[bool]:
            return [self._is_valid(action) for action in np.arange(self.action_space.n)]

        def calculate_reward(self, action):

            if not self._is_valid(action):
                self.tensorboard_log("action_invalid")
                return 0.

            elif self._position == Positions.Neutral:
                # Idle
                if action == Actions.Neutral.value:
                    self.tensorboard_log("action_idle")
                    return 0.

                # Entry
                elif action in (Actions.Long_enter.value, Actions.Short_enter.value):
                    return 0.

            elif self._position != Positions.Neutral:
                # Hold
                if action == Actions.Neutral.value:
                    self.tensorboard_log("action_hold")
                    return 0.

                # Exit
                elif action == Actions.Exit.value:
                    return self.get_unrealized_profit()
            return 0.

        def step(self, action: int):
            self._current_tick += 1
            self.tensorboard_log(self.actions._member_names_[action])
            self._update_unrealized_total_profit()

            step_reward = self.calculate_reward(action)
            self.total_reward += step_reward

            trade_type = self.transform_state(action)
            self.trade_history_append(trade_type)
            self.tensorboard_log("trade_count", len(self.trade_history), False)

            info = self.gather_info(action)
            self._update_history(info)
            self._position_history.append(self._position)
            return self._get_observation(), step_reward, self.is_done(), info

        def gather_info(self, action) -> dict:
            return dict(
                tick=self._current_tick,
                action=action,
                total_reward=self.total_reward,
                total_profit=self._total_profit,
                position=self._position.value,
                trade_duration=self.get_trade_duration(),
                current_profit_pct=self.get_unrealized_profit(),
                is_success=self.is_success(),
            )

        def transform_state(self, action) -> str:
            trade_type: str = None
            if self.is_tradesignal(action):
                if action == Actions.Neutral.value:
                    self._position = Positions.Neutral
                    self._last_trade_tick = None
                    trade_type = "neutral"
                elif action == Actions.Long_enter.value:
                    self._position = Positions.Long
                    self._last_trade_tick = self._current_tick
                    trade_type = "long"
                elif action == Actions.Short_enter.value:
                    self._position = Positions.Short
                    self._last_trade_tick = self._current_tick
                    trade_type = "short"
                elif action == Actions.Exit.value:
                    self._update_total_profit()
                    self._position = Positions.Neutral
                    self._last_trade_tick = None
                    trade_type = "neutral"
                else:
                    logger.warning("Tradesignal not defined!")

            elif self.rl_config.get("use_env_signals", False):
                if self.is_tradesignal_from_env():
                    if self._position != Positions.Neutral:
                        self._update_total_profit()
                        self._position = Positions.Neutral
                        self._last_trade_tick = None
                        trade_type = "neutral"
            return trade_type

        def is_tradesignal_from_env(self) -> bool:
            if (self.get_unrealized_profit() >= self.profit_aim * self.rr):
                self.tensorboard_log("take_profit")
                return True
            elif (self.get_unrealized_profit() <= -self.profit_aim):
                self.tensorboard_log("stop_loss")
                return True
            elif (self.get_trade_duration() >= self.rl_config.get("max_trade_duration_candles", 300)):
                self.tensorboard_log("timeout")
                return True
            return False

        def is_done(self) -> bool:
            if self._current_tick == self._end_tick:
                return True
            if self._total_profit >= self.rl_config.get("max_total_profit", 10):
                return True
            if (self._total_profit <= self.max_drawdown
                or self._total_unrealized_profit <= self.max_drawdown):
                return True
            return False

        def is_success(self) -> bool:
            if self._total_profit >= self.rl_config.get("max_total_profit", 3):
                return True
            return False

        def trade_history_append(self, trade_type: str):
            if trade_type is not None:
                self.trade_history.append(
                    {
                        "tick": self._current_tick,
                        "trade_type": trade_type,
                        "trade_price": self.current_price(),
                    }
                )

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
                _rollout_history = _history.merge(_trade_history, on="tick", how="left")
            except KeyError:
                return fig  # return empty plot if no trades
            _price_history = self.prices.iloc[_rollout_history.tick].copy()

            history = pd.merge(
                _rollout_history, _price_history.reset_index(),
                left_index=True, right_index=True
            )

            axs[0].plot(history["open"], linewidth=1, color="#c28ce3")
            plot_markers(axs[0], history.loc[history["trade_type"] == "short"]
                        ["trade_price"], "v", "#f53580", 7, 0.1)
            plot_markers(axs[0], history.loc[history["trade_type"] == "long"]
                        ["trade_price"], "^", "#4ae747", 7, -0.1)
            plot_markers(axs[0], history.loc[history["trade_type"] == "neutral"]
                        ["trade_price"], "o", "#73b3e4", 5, 0)

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
