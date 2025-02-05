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

# Nimble Miner Template:# Step 1: Import necessary libraries and modules
"""
Nimble Miner Template

This is a basic template for creating a Nimble miner. This miner listens to incoming requests 
from the Nimble network and responds with a static message: "I am a chatbot".

Developers can extend this template by introducing more sophisticated response generation 
mechanisms, custom configurations, and additional functionalities.
"""

import argparse
from prompting.baseminer.miner import Miner
from prompting.protocol import Prompting
import nimble as nb


class TemplateMiner(Miner):
    def config(self) -> "nb.Config":
        """
        Returns the configuration object specific to this miner.

        Implement and extend this method to provide custom configurations for the miner.
        Currently, it sets up a basic configuration parser.

        Returns:
            nb.Config: A configuration object with the miner's operational parameters.
        """
        parser = argparse.ArgumentParser(description="Template Miner Configs")
        self.add_args(parser)
        return nb.config(parser)

    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds custom arguments to the command line parser.

        Developers can introduce additional command-line arguments specific to the miner's
        functionality in this method. These arguments can then be used to configure the miner's operation.

        Args:
            parser (argparse.ArgumentParser):
                The command line argument parser to which custom arguments should be added.
        """
        pass

    def prompt(self, synapse: Prompting) -> Prompting:
        """
        Handles incoming requests and provides a static response.

        This method processes the incoming request encapsulated in the `synapse` object and
        returns a static response: "I am a chatbot". Developers can extend this method to
        provide dynamic or more sophisticated responses. See nimble's implentation of Synapse
        for more information.

        Args:
            synapse (Prompting): The incoming request object.

        Returns:
            Prompting: The response object with the static message.
        """
        nb.logging.debug("In prompt!")
        synapse.completion = "I am a chatbot"
        return synapse


# This is the main function, which runs the miner.
if __name__ == "__main__":
    """
    Main execution point for the TemplateMiner.

    This script initializes and runs the TemplateMiner, connecting it to the Nimble network.
    The miner listens for incoming requests and responds with a static message.

    Developers can start the miner by simply executing this script. To add more functionalities
    or customize the miner's behavior, consider extending the TemplateMiner class above.
    """
    TemplateMiner().run()
