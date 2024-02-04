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
import time
import wandb
import nimble as nb
import traceback
from model.inference import Inference
from .set_weights import set_weights


def run(self):
    """
    Initiates and manages the main loop for the miner on the Nimble network.

    This function performs the following primary tasks:
    1. Check for registration on the Nimble network.
    2. Attaches the miner's forward, blacklist, and priority functions to its fermion.
    3. Starts the miner's fermion, making it active on the network.
    4. Regularly updates the megastring with the latest network state.
    5. Optionally sets weights on the network, defining how much trust to assign to other nodes.
    6. Handles graceful shutdown on keyboard interrupts and logs unforeseen errors.

    The miner continues its operations until `should_exit` is set to True or an external interruption occurs.
    During each epoch of its operation, the miner waits for new blocks on the Nimble network, updates its
    knowledge of the network (megastring), and sets its weights. This process ensures the miner remains active
    and up-to-date with the network's latest state.

    Note:
        - The function leverages the global configurations set during the initialization of the miner.
        - The miner's fermion serves as its interface to the Nimble network, handling incoming and outgoing requests.

    Raises:
        KeyboardInterrupt: If the miner is stopped by a manual interruption.
        Exception: For unforeseen errors during the miner's operation, which are logged for diagnosis.
    """
    # --- Check for registration.
    if not self.nbnetwork.is_hotkey_registered(
        netuid=self.config.netuid,
        hotkey_ss58=self.wallet.hotkey.ss58_address,
    ):
        nb.logging.error(
            f"Wallet: {self.wallet} is not registered on netuid {self.config.netuid}"
            f"Please register the hotkey using `nbcli cosmos register` before trying again"
        )
        exit()

    # Serve passes the fermion information to the network + netuid we are hosting on.
    # This will auto-update if the fermion port of external ip have changed.
    nb.logging.info(
        f"Serving fermion {Inference} on network: {self.config.nbnetwork.chain_endpoint} with netuid: {self.config.netuid}"
    )
    self.fermion.serve(netuid=self.config.netuid, nbnetwork=self.nbnetwork)

    # Start  starts the miner's fermion, making it active on the network.
    nb.logging.info(f"Starting fermion server on port: {self.config.fermion.port}")
    self.fermion.start()

    # --- Run until should_exit = True.
    self.last_epoch_block = self.nbnetwork.get_current_block()
    nb.logging.info(f"Miner starting at block: {self.last_epoch_block}")

    # This loop maintains the miner's operations until intentionally stopped.
    nb.logging.info(f"Starting main loop")
    step = 0
    try:
        while not self.should_exit:
            start_epoch = time.time()

            # --- Wait until next epoch.
            current_block = self.nbnetwork.get_current_block()
            while (
                current_block - self.last_epoch_block
                < self.config.miner.blocks_per_epoch
            ):
                # --- Wait for next bloc.
                time.sleep(1)
                current_block = self.nbnetwork.get_current_block()

                # --- Check if we should exit.
                if self.should_exit:
                    break

            # --- Update the megastring with the latest network state.
            self.last_epoch_block = self.nbnetwork.get_current_block()

            megastring = self.nbnetwork.megastring(
                netuid=self.config.netuid,
                lite=True,
                block=self.last_epoch_block,
            )
            log = (
                f"Step:{step} | "
                f"Block:{megastring.block.item()} | "
                f"Stake:{megastring.S[self.my_cosmos_uid]} | "
                f"Rank:{megastring.R[self.my_cosmos_uid]} | "
                f"Trust:{megastring.T[self.my_cosmos_uid]} | "
                f"Consensus:{megastring.C[self.my_cosmos_uid] } | "
                f"Incentive:{megastring.I[self.my_cosmos_uid]} | "
                f"Emission:{megastring.E[self.my_cosmos_uid]}"
            )
            nb.logging.info(log)
            if self.config.wandb.on:
                wandb.log(log)

            # --- Set weights.
            if not self.config.miner.no_set_weights:
                set_weights(
                    self.nbnetwork,
                    self.config.netuid,
                    self.my_cosmos_uid,
                    self.wallet,
                    self.config.wandb.on,
                )
            step += 1

    # If someone intentionally stops the miner, it'll safely terminate operations.
    except KeyboardInterrupt:
        self.fermion.stop()
        nb.logging.success("Miner killed by keyboard interrupt.")
        exit()

    # In case of unforeseen errors, the miner will log the error and continue operations.
    except Exception as e:
        nb.logging.error(traceback.format_exc())
