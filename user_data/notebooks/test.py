import os
from pathlib import Path

# Change directory
# Modify this cell to insure that the output shows the correct path.
# Define all paths relative to the project root shown in the cell output
project_root = "somedir/freqtrade"
i=0
try:
    os.chdirdir(project_root)
    assert Path('LICENSE').is_file()
except:
    while i<4 and (not Path('LICENSE').is_file()):
        os.chdir(Path(Path.cwd(), '../'))
        i+=1
    project_root = Path.cwd()
print(Path.cwd())

from freqtrade.configuration import Configuration
from pprint import pprint
# Customize these according to your needs.

# Initialize empty configuration object
config = Configuration.from_files(["config_examples/config_freqai.RL.example2.json"])
pprint(config)
# Optionally (recommended), use existing configuration file
# config = Configuration.from_files(["user_data/config.json"])
# Define some constants
#config["timeframe"] = "1h"
# Name of the strategy class
config["strategy"] = "ReforceXStrategy4ac"
# Location of the data
data_location = config["datadir"]
# Pair to analyze - Only use one pair here
pair = "BTC/USDT:USDT"

# Load data using values set above
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

candles = load_pair_history(datadir=data_location,
                            timeframe=config["timeframe"],
                            pair=pair,
                            data_format = "json",  # Make sure to update this to your data
                            candle_type=CandleType.FUTURES,
                            )

# Confirm success
print(f"Loaded {len(candles)} rows of data for {pair} from {data_location}")

# Load strategy using values set above
from freqtrade.resolvers import StrategyResolver
from freqtrade.data.dataprovider import DataProvider
strategy = StrategyResolver.load_strategy(config)
strategy.dp = DataProvider(config, None, None)
strategy.ft_bot_start()

# Generate buy/sell signals using strategy
df = strategy.analyze_ticker(candles, {'pair': pair})
df.tail()
