import logging
import gc
import copy
import numpy as np
import torch as th
import pandas as pd
import matplotlib.transforms as mtransforms
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Any, Dict, Type, Callable, List
from pandas import DataFrame
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam, Figure
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from freqtrade.strategy import timeframe_to_minutes
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen

from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment
from freqtrade.freqai.RL.BaseReinforcementLearningModel import BaseReinforcementLearningModel, make_env
from freqtrade.freqai.tensorboard.TensorboardCallback import TensorboardCallback

from user_data.freqaimodels.BaseCoolActionRLEnv import Actions, CoolActionRLEnv, Positions

logger = logging.getLogger(__name__)


class MyCoolRLmodel(BaseReinforcementLearningModel):
    """
    Reinforcement learning prediction model.

    "rl_config": {
        // ReforceX
        "force_exits": false,
        "lr_schedule": false,
        "tensorboard_plot": false,
        "frame_staking": false,
        "frame_staking_n_stack": 2,
        "multiproc": false
    }
    """

    def __init__(self, **kwargs) -> None:
        """
        Model specific config
        """
        super().__init__(**kwargs)

        # Enable action masking for MaskablePPO
        self.use_masking: bool = self.model_type == "MaskablePPO"

        # Enable learning rate linear schedule
        self.lr_schedule: bool = self.rl_config.get('lr_schedule', False)

        # Enable tensorboard logging
        self.activate_tensorboard: bool = self.rl_config.get('activate_tensorboard', True)
        # TENSORBOARD CALLBACK DOES NOT RECOMMENDED TO USE WITH MULTIPLE ENVS,
        # IT WILL RETURN FALSE INFORMATIONS, NEVERTHLESS NOT THREAD SAFE WITH SB3!!!

        # Enable tensorboard rollout plot
        self.tensorboard_plot: bool = self.rl_config.get('tensorboard_plot', False)

        # [Experimental]
        # Demonstration of how to build vectorized environments:
        # Enable frame stacking
        self.frame_staking: bool = self.rl_config.get('frame_staking', False)
        self.frame_staking_n_stack: int = self.rl_config.get('frame_staking_n_stack', 2)
        # OR
        # Enable multiproc
        self.multiproc: bool = self.rl_config.get('multiproc', False)

        self.unset_unsupported()

    def unset_unsupported(self):
        if self.use_masking and self.multiproc:
            logger.warning(
                "User tried to use MaskablePPO with multiproc. "
                "Deactivating multiproc")
            self.multiproc = False

        if self.frame_staking and self.multiproc:
            logger.warning(
                "User tried to use frame_staking with multiproc. "
                "Deactivating multiproc")
            self.multiproc = False

        if self.continual_learning and self.frame_staking:
            logger.warning(
                "User tried to use continual_learning with frame_staking. "
                "Deactivating continual_learning")
            self.continual_learning = False

    def _set_envs(self, train_df: DataFrame, prices_train: DataFrame,
                  test_df: DataFrame, prices_test: DataFrame,
                  env_dict: Dict):
        """
        Set single training and evaluation environments
        """
        self.train_env = self.MyRLEnv(df=train_df, prices=prices_train, **env_dict)
        self.eval_env = Monitor(self.MyRLEnv(df=test_df, prices=prices_test, **env_dict))

    def _set_envs_frame_staking(self, train_df: DataFrame, prices_train: DataFrame,
                                test_df: DataFrame, prices_test: DataFrame,
                                env_dict: Dict, n_stack: int):
        """
        Set dummy vectorized frame stacked training and evaluation environments
        """
        logger.info(f"Frame staking enabled, rank: {self.max_threads}, n_stack: {n_stack}")
        self.train_env = DummyVecEnv([
            make_env(
                self.MyRLEnv, "train_env", i, 42,
                train_df, prices_train,
                env_info=env_dict
            ) for i in range(self.max_threads)
        ])
        self.eval_env = DummyVecEnv([
            make_env(
                self.MyRLEnv, "eval_env", i, 42,
                test_df, prices_test,
                env_info=env_dict
            ) for i in range(self.max_threads)
        ])
        self.train_env = VecMonitor(VecFrameStack(self.train_env, n_stack=n_stack))
        self.eval_env = VecMonitor(VecFrameStack(self.eval_env, n_stack=n_stack))

    def _set_envs_multiproc(self, train_df: DataFrame, prices_train: DataFrame,
                            test_df: DataFrame, prices_test: DataFrame,
                            env_dict: Dict):
        """
        Set vectorized subproc training and evaluation environments
        """
        logger.info(f"Multiproc enabled, rank: {self.max_threads}")
        self.train_env = SubprocVecEnv([
                make_env(
                    self.MyRLEnv, "train_env", i, 42,
                    train_df, prices_train,
                    env_info=env_dict
                ) for i in range(self.max_threads)
        ])
        self.eval_env = SubprocVecEnv([
                make_env(
                    self.MyRLEnv, 'eval_env', i, 42,
                    test_df, prices_test,
                    env_info=env_dict
                ) for i in range(self.max_threads)
        ])
        self.train_env = VecMonitor(self.train_env)
        self.eval_env = VecMonitor(self.eval_env)

    def set_train_and_eval_environments(self, data_dictionary: Dict[str, DataFrame],
                                        prices_train: DataFrame, prices_test: DataFrame,
                                        dk: FreqaiDataKitchen):
        """
        Set training and evaluation environments
        """
        self.close_envs()
        logger.info("Creating environments...")
        env_data = {
            "train_df": data_dictionary["train_features"],
            "prices_train": prices_train,
            "test_df": data_dictionary["test_features"],
            "prices_test": prices_test,
            "env_dict": self.pack_env_dict(dk.pair)
        }
        if self.frame_staking:
            self._set_envs_frame_staking(n_stack=self.frame_staking_n_stack, **env_data)
        elif self.multiproc:
            self._set_envs_multiproc(**env_data)
        else:
            self._set_envs(**env_data)

    def get_model_params(self):
        """
        Get the model specific parameters
        """
        model_params = copy.deepcopy(self.freqai_info["model_training_parameters"])

        if self.lr_schedule:
            _lr = model_params.get('learning_rate', 0.0003)
            model_params["learning_rate"] = linear_schedule(_lr)
            logger.info(f"Learning rate linear schedule enabled, initial value: {_lr}")

        model_params["policy_kwargs"] = dict(
            net_arch=dict(vf=self.net_arch, pi=self.net_arch),
            activation_fn=th.nn.ReLU,
            optimizer_class=th.optim.Adam
        )
        return model_params

    def get_callbacks(self, eval_freq, data_path) -> list:
        """
        Get the model specific callbacks
        """
        callbacks = []
        callbacks.append(MaskableEvalCallback(
            self.eval_env, deterministic=True,
            render=False, eval_freq=eval_freq,
            best_model_save_path=data_path,
            use_masking=self.use_masking
        ))
        if self.activate_tensorboard:
            callbacks.append(CustomTensorboardCallback())
        if self.tensorboard_plot:
            callbacks.append(FigureRecorderCallback())
        return callbacks

    def fit(self, data_dictionary: Dict[str, Any], dk: FreqaiDataKitchen, **kwargs):
        """
        User customizable fit method
        :param data_dictionary: dict = common data dictionary containing all train/test
            features/labels/weights.
        :param dk: FreqaiDatakitchen = data kitchen for current pair.
        :return:
        model Any = trained model to be used for inference in dry/live/backtesting
        """
        train_df = data_dictionary["train_features"]
        total_timesteps = int(self.freqai_info["rl_config"]["train_cycles"] * len(train_df))
        tensorboard_log_path = Path(dk.full_path / "tensorboard" / dk.pair.split('/')[0])
        model_params = self.get_model_params()

        logger.info(f"Action masking enabled") if self.use_masking else None
        logger.info(f"Total timesteps: {total_timesteps}")
        logger.info(f"Params: {model_params}")

        if dk.pair not in self.dd.model_dictionary or not self.continual_learning:
            model = self.MODELCLASS(
                self.policy_type,
                self.train_env,
                tensorboard_log=tensorboard_log_path if self.activate_tensorboard else None,
                **model_params
            )
        else:
            logger.info('Continual training activated - starting training from previously '
                        'trained agent.')
            model = self.dd.model_dictionary[dk.pair]
            model.set_env(self.train_env)

        model.learn(
            total_timesteps=total_timesteps,
            callback=self.get_callbacks(len(train_df), str(dk.data_path)),
            progress_bar=self.rl_config.get('progress_bar', False)
        )

        if Path(dk.data_path / "best_model.zip").is_file():
            logger.info('Callback found a best model.')
            best_model = self.MODELCLASS.load(dk.data_path / "best_model")
            return best_model

        logger.info('Couldnt find best model, using final model instead.')

        return model

    def _action_masks_predict(self, market_side: float) -> List[bool]:
        if self.frame_staking or self.multiproc:
            _is_valid_func = self.train_env.envs[0]._is_valid
        else:
            _is_valid_func = self.train_env._is_valid
        return [_is_valid_func(action.value, market_side) for action in Actions]

    def rl_model_predict(self, dataframe: DataFrame, dk: FreqaiDataKitchen, model: Any) -> DataFrame:
        """
        A helper function to make predictions in the Reinforcement learning module.
        :param dataframe: DataFrame = the dataframe of features to make the predictions on
        :param dk: FreqaiDatakitchen = data kitchen for the current pair
        :param model: Any = the trained model used to inference the features.
        """
        output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)

        def _predict(window):
            observations = dataframe.iloc[window.index]
            action_masks_param = {}

            if self.live and self.rl_config.get('add_state_info', False):
                market_side, current_profit, trade_duration = self.get_state_info(dk.pair)
                observations['current_profit_pct'] = current_profit
                observations['position'] = market_side
                observations['trade_duration'] = trade_duration

                if self.use_masking:
                    action_masks_param["action_masks"] = self._action_masks_predict(market_side)

            if self.frame_staking:
                observations = observations.to_numpy()
                observations = np.repeat(observations, repeats=self.frame_staking_n_stack, axis=1)

            action, _states = model.predict(observations, deterministic=True, **action_masks_param)
            return action

        output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

        return output

    def close_envs(self):
        if self.train_env:
            self.train_env.close()
        if self.eval_env:
            self.eval_env.close()

    MyRLEnv: Type[BaseEnvironment]

    class MyRLEnv(CoolActionRLEnv):  # type: ignore[no-redef]
        """
        User can override any function in BaseRLEnv and gym.Env.
        """

        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def calculate_reward(self, action):
            return 0.



def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


class CustomTensorboardCallback(TensorboardCallback):
    """
    Tensorboard callback
    """

    def _on_training_start(self) -> None:
        _lr = self.model.learning_rate
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": _lr if _lr is float else "lr_schedule",
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
        }
        metric_dict = {
            "eval/mean_reward": 0,
            "rollout/ep_rew_mean": 0,
            "rollout/ep_len_mean": 0,
            "info/total_profit": 1,
            "info/trades_count": 0,
            "info/trade_duration": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:

        local_info = self.locals["infos"][0]
        if self.training_env is None:
            return True
        tensorboard_metrics = self.training_env.get_attr("tensorboard_metrics")[0]

        for metric in local_info:
            if metric not in ["episode", "terminal_observation", "TimeLimit.truncated"]:
                self.logger.record(f"info/{metric}", local_info[metric])

        for category in tensorboard_metrics:
            for metric in tensorboard_metrics[category]:
                self.logger.record(f"{category}/{metric}", tensorboard_metrics[category][metric])

        return True


class FigureRecorderCallback(BaseCallback):
    """
    Tensorboard figures callback
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self):
        try:
            figures = [env.get_rollout_plot() for env in self.training_env.envs]
        except AttributeError:
            figures = self.training_env.env_method("get_rollout_plot")

        for i, fig in enumerate(figures):
            self.logger.record(
                f"rollout/env_{i}",
                Figure(fig, close=True),
                exclude=("stdout", "log", "json", "csv")
            )
            plt.close(fig)
        return True
