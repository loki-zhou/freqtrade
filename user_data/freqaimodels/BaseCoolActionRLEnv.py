import logging
from enum import Enum

from gymnasium import spaces

from freqtrade.freqai.RL.BaseEnvironment import BaseEnvironment, Positions


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
        self.actions = Actions

    def set_action_space(self):
        self.action_space = spaces.Discrete(len(Actions))

    def step(self, action: int):
        """
        Logic for a single step (incrementing one candle in time)
        by the agent
        :param: action: int = the action type that the agent plans
            to take for the current step.
        :returns:
            observation = current state of environment
            step_reward = the reward from `calculate_reward()`
            _done = if the agent "died" or if the candles finished
            info = dict passed back to openai gym lib
        """
        self._done = False
        self._current_tick += 1

        if self._current_tick == self._end_tick:
            self._done = True

        self._update_unrealized_total_profit()
        step_reward = self.calculate_reward(action)
        self.total_reward += step_reward
        self.tensorboard_log(self.actions._member_names_[action], category="actions")

        trade_type = None

        self._position, trade, tradetag = transform(self._position, action)
        if trade:
            self._last_trade_tick = self._current_tick
            self.trade_history.append(
                {'price': self.current_price(), 'index': self._current_tick,
                 'type': tradetag, 'profit': self.get_unrealized_profit()})

        if (self._total_profit < self.max_drawdown or
                self._total_unrealized_profit < self.max_drawdown):
            self._done = True

        self._position_history.append(self._position)

        info = dict(
            tick=self._current_tick,
            action=action,
            total_reward=self.total_reward,
            total_profit=self._total_profit,
            position=self._position.value,
            trade_duration=self.get_trade_duration(),
            current_profit_pct=self.get_unrealized_profit()
        )

        observation = self._get_observation()
        # user can play with time if they want
        truncated = False

        self._update_history(info)

        return observation, step_reward, self._done, truncated, info



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