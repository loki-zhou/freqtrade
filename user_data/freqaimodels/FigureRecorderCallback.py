from enum import Enum
from freqtrade.freqai.RL.TensorboardCallback import TensorboardCallback
from typing import Any, Dict, Type, Union
from freqtrade.freqai.RL.BaseEnvironment import BaseActions, BaseEnvironment
from stable_baselines3.common.logger import Figure

class FigureRecorderCallback(TensorboardCallback):
    def __init__(self, verbose=1, actions: Type[Enum] = BaseActions):
        super(FigureRecorderCallback, self).__init__(verbose, actions)

    # def _on_rollout_end(self) -> None:
    #     figure = self.training_env.render()
    #     self.logger.record(
    #         "rollout/positions",
    #         Figure(figure, close=True),
    #         exclude=("stdout", "log", "json", "csv")
    #     )
    #     return True

    def _on_rollout_end(self) -> None:
        figure = self.training_env.envs[0].render()
        self.logger.record(
            "rollout/positions",
            Figure(figure, close=True),
            exclude=("stdout", "log", "json", "csv")
        )
        return