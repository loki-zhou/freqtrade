import logging
from pandas import DataFrame
from freqtrade.freqai.RL.Base5ActionRLEnv import Actions
from ReforceXBaseStrategy import ReforceXBaseStrategy


logger = logging.getLogger(__name__)


class ReforceXStrategy5ac(ReforceXBaseStrategy):
   
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["&-action"] == Actions.Long_enter.value)
                & (dataframe["do_predict"] == 1)
                & (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"],
        ] = (1, "agent_long")

        dataframe.loc[
            (
                (dataframe["&-action"] == Actions.Short_enter.value)
                & (dataframe["do_predict"] == 1)
                & (dataframe["volume"] > 0)
            ),
            ["enter_short", "enter_tag"],
        ] = (1, "agent_short")

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe.loc[
            (
                (dataframe["&-action"] == Actions.Long_exit.value)
                & (dataframe["do_predict"] == 1)
                & (dataframe["volume"] > 0)
            ),
            ["exit_long", "exit_tag"],
        ] = (1, "agent_exit_long")

        dataframe.loc[
            (
                (dataframe["&-action"] == Actions.Short_exit.value)
                & (dataframe["do_predict"] == 1)
                & (dataframe["volume"] > 0)
            ),
            ["exit_short", "exit_tag"],
        ] = (1, "agent_exit_short")

        return dataframe
