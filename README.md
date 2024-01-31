# Nimble Miners

This repo provides a set of miners for anyone to mine Nimble native tokens. It enables single
machine model training and distributed training to increase your chances of successful token mining.

# Development

### Code Downloading

```bash
# clone the code repo to local
git clone https://github.com/nimble-technology/nimble-miners.git
cd nimble-miners
```

### Virtual Env Setup

```bash
# install python3.9 or higher if not yet.
brew install python3.9

# create virtualenv
python3 -m venv nimble-env

# activate virtualenv
source nimble-env/bin/activate

# install nimble miners dependencies
python3 -m pip install -r requirements.txt

# deactivate virtualenv
deactivate

# clean virtualenv
rm -rf nimble-env
```

# Run Miners
Nimble contributors provide various miners any developer can run. Developers can add more miners as well.

### Full Usage
```
usage: miner.py [-h] [--axon.port AXON.PORT] [--nbnetwork.network NBNETWORK.NETWORK] [--nbnetwork.chain_endpoint NBNETWORK.CHAIN_ENDPOINT] [--netuid NETUID] [--miner.root MINER.ROOT] [--miner.name MINER.NAME]
                [--miner.blocks_per_epoch MINER.BLOCKS_PER_EPOCH] [--miner.blacklist.blacklist [MINER.BLACKLIST.BLACKLIST ...]] [--miner.blacklist.whitelist [MINER.BLACKLIST.WHITELIST ...]]
                [--miner.blacklist.force_validator_permit] [--miner.blacklist.allow_non_registered] [--miner.blacklist.minimum_stake_requirement MINER.BLACKLIST.MINIMUM_STAKE_REQUIREMENT]
                [--miner.blacklist.prompt_cache_block_span MINER.BLACKLIST.PROMPT_CACHE_BLOCK_SPAN] [--miner.blacklist.use_prompt_cache] [--miner.blacklist.min_request_period MINER.BLACKLIST.MIN_REQUEST_PERIOD]
                [--miner.priority.default MINER.PRIORITY.DEFAULT] [--miner.priority.time_stake_multiplicate MINER.PRIORITY.TIME_STAKE_MULTIPLICATE]
                [--miner.priority.len_request_timestamps MINER.PRIORITY.LEN_REQUEST_TIMESTAMPS] [--miner.no_set_weights] [--miner.no_serve] [--miner.no_start_axon] [--miner.mock_nbnetwork] [--wandb.on]
                [--wandb.project_name WANDB.PROJECT_NAME] [--wandb.entity WANDB.ENTITY] [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--wallet.name WALLET.NAME]
                [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH] [--config CONFIG] [--strict] [--no_version_checking] [--no_prompt]

options:
  -h, --help            show this help message and exit
  --axon.port AXON.PORT
                        Port to run the axon on.
  --nbnetwork.network NBNETWORK.NETWORK
                        Nimble network to connect to.
  --nbnetwork.chain_endpoint NBNETWORK.CHAIN_ENDPOINT
                        Chain endpoint to connect to.
  --netuid NETUID       The chain subnet uid.
  --miner.root MINER.ROOT
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --miner.name MINER.NAME
                        Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name
  --miner.blocks_per_epoch MINER.BLOCKS_PER_EPOCH
                        Blocks until the miner sets weights on chain
  --miner.blacklist.blacklist [MINER.BLACKLIST.BLACKLIST ...]
                        Blacklist certain hotkeys
  --miner.blacklist.whitelist [MINER.BLACKLIST.WHITELIST ...]
                        Whitelist certain hotkeys
  --miner.blacklist.force_validator_permit
                        Only allow requests from validators
  --miner.blacklist.allow_non_registered
                        If True, the miner will allow non-registered hotkeys to mine.
  --miner.blacklist.minimum_stake_requirement MINER.BLACKLIST.MINIMUM_STAKE_REQUIREMENT
                        Minimum stake requirement
  --miner.blacklist.prompt_cache_block_span MINER.BLACKLIST.PROMPT_CACHE_BLOCK_SPAN
                        Amount of blocks to keep a prompt in cache
  --miner.blacklist.use_prompt_cache
                        If True, the miner will use the prompt cache to store recent request prompts.
  --miner.blacklist.min_request_period MINER.BLACKLIST.MIN_REQUEST_PERIOD
                        Time period (in minute) to serve a maximum of 50 requests for each hotkey
  --miner.priority.default MINER.PRIORITY.DEFAULT
                        Default priority of non-registered requests
  --miner.priority.time_stake_multiplicate MINER.PRIORITY.TIME_STAKE_MULTIPLICATE
                        Time (in minute) it takes to make the stake twice more important in the priority queue
  --miner.priority.len_request_timestamps MINER.PRIORITY.LEN_REQUEST_TIMESTAMPS
                        Number of historic request timestamps to record
  --miner.no_set_weights
                        If True, the miner does not set weights.
  --miner.no_serve      If True, the miner doesnt serve the axon.
  --miner.no_start_axon
                        If True, the miner doesnt start the axon.
  --miner.mock_nbnetwork
                        If True, the miner will allow non-registered hotkeys to mine.
  --wandb.on            Turn on wandb.
  --wandb.project_name WANDB.PROJECT_NAME
                        The name of the project where youre sending the new run.
  --wandb.entity WANDB.ENTITY
                        An entity is a username or team name where youre sending runs.
  --logging.debug       Turn on nimble debugging information
  --logging.trace       Turn on nimble trace level information
  --logging.record_log  Turns on logging to file.
  --logging.logging_dir LOGGING.LOGGING_DIR
                        Logging default root directory.
  --wallet.name WALLET.NAME
                        The name of the wallet to unlock for running nimble (name mock is reserved for mocking this wallet)
  --wallet.hotkey WALLET.HOTKEY
                        The name of the wallet's hotkey.
  --wallet.path WALLET.PATH
                        The path to your nimble wallets
  --config CONFIG       If set, defaults are overridden by passed file.
  --strict              If flagged, config will check that only exact arguments have been set.
  --no_version_checking
                        Set true to stop cli version checking.
  --no_prompt           Set true to stop cli from prompting the user.
```

### Language Model Miner

Nimble language model (LM) is a small (3B) language model. It is still powerful for model inferences. Please
create and manage virtualenv as detailed [above](https://github.com/nimble-technology/nimble-miners?tab=readme-ov-file#virtual-env-setup).

```bash
# install nimbleLM dependencies
cd miners/nimbleLM/
python3 -m pip install -r requirements.txt

# Example usage of the nimbleLM miner
python -m miners/nimbleLM/miner.py
    --netuid 1
    --wallet.name <your miner wallet>
    --wallet.hotkey <your miner hotkey>
    --miner.blacklist.whitelist <hotkeys of the validators to be connected>
    --logging.debug
```
