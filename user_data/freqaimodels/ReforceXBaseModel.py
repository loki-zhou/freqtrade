from abc import abstractmethod
import logging
from pathlib import Path
from typing import Any, Dict

import torch as th
import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel
from freqtrade.freqai.RL.TensorboardCallback import TensorboardCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import explained_variance

from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.utils import get_action_masks
from freqtrade.freqai.RL.BaseReinforcementLearningModel import make_env, Positions


logger = logging.getLogger(__name__)


class ReforceXBaseModel(BaseReinforcementLearningModel):
    """
    ReforceX Pack - base prediction model

    Available custom settings:
    "rl_config": {
        "net_arch": [0],            // automatic network size calculation (features*positions) 
        "max_total_profit": int     // episode "done" condition
        "use_env_signals" : bool,   // takeprofit stoploss env integration
        "progress_bar": bool,       // required: !pip install tqdm rich 
        "use_multiproc": bool,      // except MaskablePPO, may conflict with render()
    }

    The user should modify the TensorboardCallback
    to use env render() function, for example:

    def _on_rollout_end(self) -> None:
        figure = self.training_env.render() 
        self.logger.record(
            "rollout/positions",
            Figure(figure, close=True),
            exclude=("stdout", "log", "json", "csv")
        )
        return True

    The following model types are supported:

        "model_type": "PPO",
        "policy_type": "MlpPolicy"

        "model_type": "MaskablePPO",
        "policy_type": "MlpPolicy"

        "model_type": "RecurrentPPO",
        "policy_type": "MlpLstmPolicy"

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.get_model_type()
        self.lstm_states = None
        self.episode_starts = None
        self.predict_params = {}

    def get_model_type(self):
        self.use_masking = self.model_type == "MaskablePPO"
        self.use_recurrent = self.model_type == "RecurrentPPO"
        self.use_multiproc = self.rl_config.get('use_multiproc', False)
        if (self.use_masking and self.use_multiproc):
            logger.warning(
                "User tried to use multiproc with MaskablePPO. Deactivating multiproc...")
            self.use_multiproc = False

    def set_model_specific_params(self, data_dictionary, dk):
        if self.use_masking:
            self.eval_callback = MaskableEvalCallback(
                self.eval_env, deterministic=True,
                render=False, eval_freq=len(data_dictionary["train_features"]),
                best_model_save_path=str(dk.data_path)
            )
            self.predict_params = {
                "action_masks": get_action_masks(self.eval_env)
            }
        elif self.use_recurrent:
            self.num_envs = self.max_threads if self.use_multiproc else 1
            self.episode_starts = np.ones((self.num_envs,), dtype=bool)
            self.predict_params = {
                "state": self.lstm_states,
                "episode_start": self.episode_starts,
            }

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, DataFrame],
                                        prices_train: DataFrame, prices_test: DataFrame,
                                        dk: FreqaiDataKitchen):
        if not self.use_multiproc:
            super().set_train_and_eval_environments(
                data_dictionary, prices_train, prices_test, dk
            )
        else:
            logger.info(f"Multiproc enabled. Creating {self.max_threads} environments...")
            train_df = data_dictionary["train_features"]
            test_df = data_dictionary["test_features"]
            env_info = self.pack_env_dict()
            env_id = "train_env"

            self.train_env = SubprocVecEnv(
                [make_env(
                    self.MyRLEnv, env_id, i, 1,
                    train_df, prices_train,
                    monitor=True,
                    env_info=env_info
                ) for i in range(self.max_threads)]
            )
            eval_env_id = 'eval_env'
            self.eval_env = SubprocVecEnv(
                [make_env(
                    self.MyRLEnv, eval_env_id, i, 1,
                    test_df, prices_test,
                    monitor=True,
                    env_info=env_info
                ) for i in range(self.max_threads)]
            )
            self.eval_callback = EvalCallback(
                self.eval_env, deterministic=True,
                render=False, eval_freq=len(train_df),
                best_model_save_path=str(dk.data_path)
            )
            actions = self.train_env.env_method("get_actions")[0]
            self.tensorboard_callback = TensorboardCallback(verbose=1, actions=actions)

        self.set_model_specific_params(data_dictionary, dk)

    def rl_model_predict(self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any) -> DataFrame:
        output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)

        def _predict(window):
            observations = dataframe.iloc[window.index]
            if self.live and self.rl_config.get('add_state_info', False):
                market_side, current_profit, trade_duration = self.get_state_info(dk.pair)
                observations['current_profit_pct'] = current_profit
                observations['position'] = market_side
                observations['trade_duration'] = trade_duration
            
            predicted_action, self.lstm_states = model.predict(
                observations, deterministic=True,
                **self.predict_params
            )
            if self.use_recurrent:
                self.episode_starts = self.eval_callback.locals["dones"]
            return predicted_action

        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

        return output

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):

        train_df = data_dictionary["train_features"]
        total_timesteps = self.freqai_info["rl_config"]["train_cycles"] * len(train_df)

        if self.net_arch == [0]:
            layer_size = len(dk.training_features_list) * len(Positions)
            self.net_arch = [layer_size, layer_size]
            logger.info(f"Calculated net_arch: {self.net_arch}")

        policy_kwargs = dict(
            activation_fn=th.nn.SELU,
            optimizer_class=th.optim.AdamW,
            net_arch=self.net_arch,
        )
        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type, self.train_env, policy_kwargs=policy_kwargs,
                tensorboard_log=Path(dk.full_path / "tensorboard" / dk.pair.split('/')[0]),
                **self.freqai_info.get('model_training_parameters', {})
            )
        else:
            logger.info('Continual training activated - starting training from previously '
                        'trained agent.')
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)

        model.learn(
            total_timesteps=int(total_timesteps),
            callback=[self.eval_callback, self.tensorboard_callback],
            progress_bar=self.rl_config.get('progress_bar', False)
        )

        explained_var = explained_variance(
            model.rollout_buffer.values.flatten(),
            model.rollout_buffer.returns.flatten()
        )
        dk.data['extra_returns_per_train']['explained_var'] = explained_var

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info('Callback found a best model.')
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info('Couldnt find best model, using final model instead.')

        return model


