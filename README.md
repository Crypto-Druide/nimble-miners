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
# create virtualenv
python3 -m venv nbenv

# activate virtualenv
source nbenv/bin/activate

# install python3.9 or higher if not yet.
brew install python@3.9

# upgrade pip
python3 -m pip install --upgrade pip

# install nimble miners dependencies
python3 -m pip install -r requirements.txt
python3 -m pip install -e .

# test the repo setup
cd miners/nblm
python3 miner.py -h

# deactivate virtualenv
deactivate

# clean virtualenv
rm -rf nbenv
```

# Run as Miners
Nimble contributors provide various miners any developer can run. Developers can add more miners as well.

### Nimble CLI: Wallet and Staking Management
`nimcli` is the command line tool to interact with nimble network such token transfers, wallet management, staking and other operations.

After creating wallets, please share your public keys to [Nimble Discord node-runners channel](https://discord.com/invite/nimble).

```bash
# usage: nimcli <command> <command args>
nimcli --help

# coldkey generation
nimcli wallet new_coldkey --wallet.name miner

# hotkey generation
nimcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# check wallet balance
nimcli wallet overview --wallet.name miner

# unstake
nimcli stake remove --wallet.name miner
```

### Miner Wallets
Miners create wallets by themselves and register to start mining by contact Discord channel 
et up the miner's wallets:

```bash

```

Set up the validator's wallets:

```bash
nimcli wallet new_coldkey --wallet.name validator
```
```bash
nimcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

### Language Model Miner

#### Miner Requirements
The minimum machine requirements are:
```bash
CPU - Intel Core i7 12700 or equivalent
GPU - v100 32gb
Memory - 64gb
Disk - 1TB
```

#### Example Usage
Nimble language model (LM) is a small (3B) language model. It is still powerful for model inferences. Please
create and manage virtualenv as detailed [above](https://github.com/nimble-technology/nimble-miners?tab=readme-ov-file#virtual-env-setup).

```bash
# install nblm dependencies
cd miners/nblm/
python3 -m pip install -r requirements.txt

# Example usage of the nblm miner
python -m miners/nblm/miner.py
    --netuid 1
    --wallet.name <your miner wallet>
    --wallet.hotkey <your miner hotkey>
    --miner.blacklist.whitelist <hotkeys of the validators to be connected>
    --logging.debug
```

#### Full Usage
```
usage: miner.py [-h] [--fermion.port FERMION.PORT] [--nbnetwork.network NBNETWORK.NETWORK] [--nbnetwork.chain_endpoint NBNETWORK.CHAIN_ENDPOINT] [--netuid NETUID] [--miner.root MINER.ROOT] [--miner.name MINER.NAME]
                [--miner.blocks_per_epoch MINER.BLOCKS_PER_EPOCH] [--miner.blacklist.blacklist [MINER.BLACKLIST.BLACKLIST ...]] [--miner.blacklist.whitelist [MINER.BLACKLIST.WHITELIST ...]]
                [--miner.blacklist.force_validator_permit] [--miner.blacklist.allow_non_registered] [--miner.blacklist.minimum_stake_requirement MINER.BLACKLIST.MINIMUM_STAKE_REQUIREMENT]
                [--miner.blacklist.predict_cache_block_span MINER.BLACKLIST.PREDICT_CACHE_BLOCK_SPAN] [--miner.blacklist.use_predict_cache] [--miner.blacklist.min_request_period MINER.BLACKLIST.MIN_REQUEST_PERIOD]
                [--miner.priority.default MINER.PRIORITY.DEFAULT] [--miner.priority.time_stake_multiplicate MINER.PRIORITY.TIME_STAKE_MULTIPLICATE]
                [--miner.priority.len_request_timestamps MINER.PRIORITY.LEN_REQUEST_TIMESTAMPS] [--miner.no_set_weights] [--miner.no_serve] [--miner.no_start_fermion] [--miner.mock_nbnetwork] [--wandb.on]
                [--wandb.project_name WANDB.PROJECT_NAME] [--wandb.entity WANDB.ENTITY] [--logging.debug] [--logging.trace] [--logging.record_log] [--logging.logging_dir LOGGING.LOGGING_DIR] [--wallet.name WALLET.NAME]
                [--wallet.hotkey WALLET.HOTKEY] [--wallet.path WALLET.PATH] [--config CONFIG] [--strict] [--no_version_checking] [--no_predict]

options:
  -h, --help            show this help message and exit
  --fermion.port FERMION.PORT
                        Port to run the fermion on.
  --nbnetwork.network NBNETWORK.NETWORK
                        Nimble network to connect to.
  --nbnetwork.chain_endpoint NBNETWORK.CHAIN_ENDPOINT
                        Chain endpoint to connect to.
  --netuid NETUID       The chain cosmos uid.
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
  --miner.blacklist.predict_cache_block_span MINER.BLACKLIST.PREDICT_CACHE_BLOCK_SPAN
                        Amount of blocks to keep a predict in cache
  --miner.blacklist.use_predict_cache
                        If True, the miner will use the predict cache to store recent requests.
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
  --miner.no_serve      If True, the miner doesnt serve the fermion.
  --miner.no_start_fermion
                        If True, the miner doesnt start the fermion.
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
  --no_predict           Set true to stop cli from the user inferences.
```

