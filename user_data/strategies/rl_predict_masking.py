def rl_model_predict(self, dataframe: DataFrame,
                         dk: FreqaiDataKitchen, model: Any) -> DataFrame:
    output = pd.DataFrame(np.zeros(len(dataframe)), columns=dk.label_list)
    use_masking = True if self.rl_config['model_type'] == 'MaskablePPO' else False
    # Simulate market side for backtesting
    # TODO: This is a hack, find a better way to do this
    bt_market_side = Positions.Neutral.value
    # In case 'continual_learning' is enabled, use previous action to determine market side
    if dk.full_df.empty == False:
      action_to_postion = {
        Actions.Exit.value: Positions.Neutral.value,
        Actions.Hold.value: Positions.Long.value,
        Actions.Long_enter.value: Positions.Long.value,
        Actions.Neutral.value: Positions.Neutral.value,
      }
      last_row = dk.full_df.iloc[-1].squeeze()
      last_action = last_row['&-action'].astype(int)
      logger.info(f'Previous prediction Action: [{Actions(last_action).name}]')
      bt_market_side = action_to_postion[last_action]

    def _predict_action_mask(position):
      pam = {
        0: [False, True, False, True], # Short (not used)
        1: [False, True, False, True], # Long
        0.5: [True, False, True, False] # Neutral
      }
      return pam[position]

    # In case 'add_state_info' is not enabled
    def _current_market_side():
      open_trades = Trade.get_trades_proxy(is_open=True)
      for trade in open_trades:
        if trade.pair == dk.pair:
          return Positions.Short.value if trade.is_short else Positions.Long.value
      return Positions.Neutral.value

    def _predict(window):
        observations = dataframe.iloc[window.index]
        nonlocal bt_market_side
        if self.live:
          if self.rl_config.get('add_state_info', False):
            market_side, current_profit, trade_duration = self.get_state_info(dk.pair)
            observations['current_profit_pct'] = current_profit
            observations['position'] = market_side
            observations['trade_duration'] = trade_duration
          elif use_masking is True:
            # we still need the current market_side when masking is enabled
            market_side = _current_market_side()
        else:
          # fake it until you make it
          market_side = bt_market_side

        predict_args = {
          "deterministic":True
        }
        if use_masking is True:
          # Make it less painful when trying other model types
          predict_args['action_masks'] = _predict_action_mask(market_side)

        res, _ = model.predict(observations, **predict_args)

        if use_masking is True:
          logger.info(f'Position: [{Positions(market_side).name}] Action: [{Actions(res).name}]')
          if not self.live:
            # Keep track of the *assumed* new position :fingers_crossed:
            if res == Actions.Long_enter.value:
              bt_market_side = Positions.Long.value
            elif res == Actions.Exit.value:
              bt_market_side = Positions.Neutral.value

        return res

    output = output.rolling(window=self.CONV_WIDTH).apply(_predict)

    return output
