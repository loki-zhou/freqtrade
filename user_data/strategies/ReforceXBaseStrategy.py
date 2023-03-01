
import logging
import numpy as np
import talib.abstract as ta
import pandas_ta as pta

from abc import abstractmethod
from datetime import datetime
from pandas import DataFrame

from freqtrade.strategy import IStrategy, timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.freqai.RL.Base4ActionRLEnv import Actions


logger = logging.getLogger(__name__)


class ReforceXBaseStrategy(IStrategy):
    process_only_new_candles = True
    startup_candle_count: int = 100
    can_short = True

    @property
    def plot_config(self):
        return {
            "main_plot": {},
            "subplots": {
                "ev": {
                    "explained_var": {"color": "#c28ce3"}
                },
                "agent": {
                    "&-action": {"color": "purple"},
                    "enter_long": {"color": "green", "type": "bar"},
                    "enter_short": {"color": "red", "type": "bar"},
                    "exit_long": {"color": "cyan", "type": "bar"},
                    "exit_short": {"color": "magenta", "type": "bar"},
                },
                "outliers": {
                    "do_predict": {"color": "red", "plotly": {"opacity": 0.5}},
                    "DI_values": {"color": "orange", "type": "bar"},
                },
            },
        }

    def feature_engineering_expand_all(self, dataframe, period, **kwargs):
        return dataframe

    def feature_engineering_expand_basic(self, dataframe, **kwargs):
        return dataframe

    def feature_engineering_standard(self, dataframe, **kwargs):
        dataframe["%-day_of_week"] = (dataframe["date"].dt.dayofweek + 1) / 7
        dataframe["%-hour_of_day"] = (dataframe["date"].dt.hour + 1) / 25
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_low"] = dataframe["low"]
        dataframe["%-raw_high"] = dataframe["high"]
        return dataframe

    def set_freqai_targets(self, dataframe, **kwargs):
        dataframe["&-action"] = Actions.Neutral.value
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    @abstractmethod
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    @abstractmethod
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):

        trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        max_trade_duration_candles = self.freqai_info['rl_config']['max_trade_duration_candles']
        profit_aim = self.freqai_info['rl_config']['model_reward_parameters']['profit_aim']
        rr = self.freqai_info['rl_config']['model_reward_parameters']['rr']

        max_trade_duration_minutes = (
            max_trade_duration_candles * timeframe_to_minutes(self.timeframe)
        )
        if current_profit >= profit_aim * rr:
            return "take_profit" 

        if trade_duration_minutes >= max_trade_duration_minutes:
            return "timeout"

        if current_profit <= -profit_aim:
            return "stop_loss"

    def get_ticker_indicator(self):
        return int(self.config["timeframe"][:-1])
