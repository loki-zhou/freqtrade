
# freqtrade download-data --exchange binance --pairs BTC/USDT ETH/USDT 1INCH/USDT ALGO/USDT --trading-mode futures --timerange 20220101- -t 3m 15m 1h
# freqtrade backtesting --strategy FreqaiExampleRLStrategy --strategy-path freqtrade/templates --config config_examples/config_freqai.RL.example.json --freqaimodel ReinforcementLearner --timerange 20220601-20221128 --breakdown day week --logfile uniqe-id-RL-000.log |& tee uniqe-id-RL-000.terminal
# test1
# test2

import logging
from datetime import datetime
from functools import reduce

import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.persistence import Trade
from freqtrade.strategy import (CategoricalParameter, DecimalParameter, IntParameter, IStrategy,
                                merge_informative_pair)


logger = logging.getLogger(__name__)

class FreqaiExampleRLStrategy(IStrategy):
    # 定义ROI
    minimal_roi = {"0": 0.1, "240": -1}
    # 定义绘图配置: 附图指标, 收盘价预测值蓝色, 异常值棕色
    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"prediction": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }
    # 只在新K线出现时计算
    process_only_new_candles = True
    # 止损
    stoploss = -0.05
    # 启用退出信号: custom_exit
    use_exit_signal = True
    # 起始K线数量
    startup_candle_count: int = 40
    # 是否可做空
    can_short = True
    # 自定义参数空间
    std_dev_multiplier_buy = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], default=1.25, space="buy", optimize=True)
    std_dev_multiplier_sell = CategoricalParameter(
        [0.75, 1, 1.25, 1.5, 1.75], space="sell", default=1.25, optimize=True)

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                        **kwargs):

        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame,  **kwargs):

        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        dataframe["%-raw_close"] = dataframe["close"]
        dataframe["%-raw_open"] = dataframe["open"]
        dataframe["%-raw_high"] = dataframe["high"]
        dataframe["%-raw_low"] = dataframe["low"]

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, **kwargs):

        dataframe["&-action"] = 0

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # 添加入场做多信号
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > df[f"target_roi_{self.std_dev_multiplier_buy.value}"],
            ]

        if enter_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
            ] = (1, "long")
        # 添加入场做空信号
        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < df[f"sell_roi_{self.std_dev_multiplier_sell.value}"],
            ]

        if enter_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
            ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # 添加做多离场信号
        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] < df[f"sell_roi_{self.std_dev_multiplier_sell.value}"] * 0.25,
            ]
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1
        # 添加做空离场信号
        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > df[f"target_roi_{self.std_dev_multiplier_buy.value}"] * 0.25,
            ]
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time,
        entry_tag,
        side: str,
        **kwargs,
    ) -> bool:
        # 自定义确认交易结果

        # 获取最新的K线数据
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1].squeeze()
        # 如果当前交易方向为做多, 且委托价格大于最新收盘价的1.0025倍则不交易
        if side == "long":
            if rate > (last_candle["close"] * (1 + 0.0025)):
                return False
        # 如果当前交易方向为做空, 且委托价格小于最新收盘价的0.9975倍则不交易
        else:
            if rate < (last_candle["close"] * (1 - 0.0025)):
                return False

        return True
