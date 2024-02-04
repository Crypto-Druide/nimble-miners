# The MIT License (MIT)
# Copyright © 2023 Nimble Labs LTD

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os
import argparse
import nimble as nb


def check_config(cls, config: "nb.Config"):
    """
    Validates the given configuration for the Miner by ensuring all necessary settings
    and directories are correctly set up. It checks the config for fermion, wallet, logging,
    and nbnetwork. Additionally, it ensures that the logging directory exists or creates one.

    Args:
        cls: The class reference, typically referring to the Miner class.
        config (nb.Config): The configuration object holding various settings for the miner.

    Raises:
        Various exceptions can be raised by the check_config methods of fermion, wallet, logging,
        and nbnetwork if the configurations are not valid.
    """
    nb.fermion.check_config(config)
    nb.logging.check_config(config)
    full_path = os.path.expanduser(
        "{}/{}/{}/{}".format(
            config.logging.logging_dir,
            config.wallet.get("name", nb.defaults.wallet.name),
            config.wallet.get("hotkey", nb.defaults.wallet.hotkey),
            config.miner.name,
        )
    )
    config.miner.full_path = os.path.expanduser(full_path)
    if not os.path.exists(config.miner.full_path):
        os.makedirs(config.miner.full_path)


def get_config() -> "nb.Config":
    """
    Initializes and retrieves a configuration object for the Miner. This function sets up
    and reads the command-line arguments to customize various miner settings. The function
    also sets up the logging directory for the miner.

    Returns:
        nb.Config: A configuration object populated with settings from command-line arguments
                   and defaults where necessary.

    Note:
        Running this function with the `--help` argument will print a help message detailing
        all the available command-line arguments for customization.
    """
    # This function initializes the necessary command-line arguments.
    # Using command-line arguments allows users to customize various miner settings.
    parser = argparse.ArgumentParser()

    # Subtensor network to connect to
    parser.add_argument(
        "--nbnetwork.network",
        default="nimble-test",
        help="Nimble network to connect to.",
    )
    # Chain endpoint to connect to
    parser.add_argument(
        "--nbnetwork.chain_endpoint",
        default="wss://testnet.nimble.technology",
        help="Chain endpoint to connect to.",
    )
    # Adds override arguments for network and netuid.
    parser.add_argument("--netuid", type=int, default=1, help="The chain cosmos uid.")

    parser.add_argument(
        "--miner.root",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="~/.nimble/miners/",
    )
    parser.add_argument(
        "--miner.name",
        type=str,
        help="Trials for this miner go in miner.root / (wallet_cold - wallet_hot) / miner.name ",
        default="Nimble Miner",
    )

    # Run config.
    parser.add_argument(
        "--miner.blocks_per_epoch",
        type=str,
        help="Blocks until the miner sets weights on chain",
        default=100,
    )

    # Blacklist.
    parser.add_argument(
        "--miner.blacklist.blacklist",
        type=str,
        required=False,
        nargs="*",
        help="Blacklist certain hotkeys",
        default=[],
    )
    parser.add_argument(
        "--miner.blacklist.whitelist",
        type=str,
        required=False,
        nargs="*",
        help="Whitelist certain hotkeys",
        default=[],
    )
    parser.add_argument(
        "--miner.blacklist.force_validator_permit",
        action="store_true",
        help="Only allow requests from validators",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.allow_non_registered",
        action="store_true",
        help="If True, the miner will allow non-registered hotkeys to mine.",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.minimum_stake_requirement",
        type=float,
        help="Minimum stake requirement",
        default=0.0,
    )
    parser.add_argument(
        "--miner.blacklist.request_cache_block_span",
        type=int,
        help="Amount of blocks to keep a request in cache",
        default=7200,
    )
    parser.add_argument(
        "--miner.blacklist.use_request_cache",
        action="store_true",
        help="If True, the miner will use the request cache to store recent requests.",
        default=False,
    )
    parser.add_argument(
        "--miner.blacklist.min_request_period",
        type=int,
        help="Time period (in minute) to serve a maximum of 50 requests for each hotkey",
        default=5,
    )

    # Priority.
    parser.add_argument(
        "--miner.priority.default",
        type=float,
        help="Default priority of non-registered requests",
        default=0.0,
    )
    parser.add_argument(
        "--miner.priority.time_stake_multiplicate",
        type=int,
        help="Time (in minute) it takes to make the stake twice more important in the priority queue",
        default=10,
    )
    parser.add_argument(
        "--miner.priority.len_request_timestamps",
        type=int,
        help="Number of historic request timestamps to record",
        default=50,
    )
    # Switches.
    parser.add_argument(
        "--miner.no_set_weights",
        action="store_true",
        help="If True, the miner does not set weights.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_serve",
        action="store_true",
        help="If True, the miner doesnt serve the fermion.",
        default=False,
    )
    parser.add_argument(
        "--miner.no_start_fermion",
        action="store_true",
        help="If True, the miner doesnt start the fermion.",
        default=False,
    )

    # Mocks.
    parser.add_argument(
        "--miner.mock_nbnetwork",
        action="store_true",
        help="If True, the miner will allow non-registered hotkeys to mine.",
        default=False,
    )

    # Wandb
    parser.add_argument(
        "--wandb.on", action="store_true", help="Turn on wandb.", default=False
    )
    parser.add_argument(
        "--wandb.project_name",
        type=str,
        help="The name of the project where youre sending the new run.",
        default=None,
    )
    parser.add_argument(
        "--wandb.entity",
        type=str,
        help="An entity is a username or team name where youre sending runs.",
        default=None,
    )

    # Adds nbnetwork specific arguments i.e. --nbnetwork.chain_endpoint ... --nbnetwork.network ...
    nb.nbnetwork.add_args(parser)

    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    nb.logging.add_args(parser)

    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    nb.wallet.add_args(parser)

    # Adds fermion specific arguments i.e. --fermion.port ...
    nb.fermion.add_args(parser)

    # Activating the parser to read any command-line inputs.
    # To print help message, run python3 template/miner.py --help
    config = nb.config(parser)

    # Logging captures events for diagnosis or understanding miner's behavior.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config
