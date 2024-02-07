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

"""
Language model miner example. Miners are permissionless. Anyone can optimize and customize the models for
best mining yields.
"""

import argparse
import deepspeed
import nimble as nb
import os
import time
import torch
from typing import List

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)

from lib.miner import Miner
from inference import Inference


class StopOnTokens(StoppingCriteria):
    """
    Custom stopping criteria for the NBLM model.

    This class defines a stopping criterion based on specific tokens. The model stops generating
    once it encounters one of the specified stop tokens.
    """

    def __init__(self, stop_token_ids: List[int]):
        self.stop_token_ids = stop_token_ids

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        for stop_id in self.stop_token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


class NBLMMiner(Miner):
    """
    Nimble miner implementation using the NBLM model.

    This miner processes incoming requests from the Nimble network and uses the
    NBLM model to generate appropriate responses based on the provided context.
    """

    def config(self) -> "nb.Config":
        """
        Returns the configuration object specific to this miner. Creates an argument parser
        and then adds the args to it that can be defined in `add_args()`

        Developers can extend this method to provide custom configurations for the miner.
        """
        parser = argparse.ArgumentParser(description="Nimble-LM Miner Config")
        self.add_args(parser)
        return nb.config(parser)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser):
        """
        Adds NBLM-specific arguments to the command line parser.

        This method introduces command-line arguments that pertain specifically to the
        NBLM model's generation settings, such as device, max length, and sampling method.
        """
        parser.add_argument(
            "--nblm.device",
            type=str,
            help="Device to load model",
            default="cuda",
        )
        parser.add_argument(
            "--nblm.max_length",
            type=int,
            default=100,
            help="The maximum length (in tokens) of the generated text.",
        )
        parser.add_argument(
            "--nblm.do_sample",
            action="store_true",
            default=False,
            help="Whether to use sampling or not (if not, uses greedy decoding).",
        )
        parser.add_argument(
            "--nblm.no_repeat_ngram_size",
            type=int,
            default=2,
            help="The size of the n-grams to avoid repeating in the generated text.",
        )
        parser.add_argument(
            "--nblm.do_request_injection",
            action="store_true",
            default=False,
            help='Whether to use a custom "system" request instead of the one sent by nimble.',
        )
        parser.add_argument(
            "--nblm.system_request",
            type=str,
            help="What request to replace the system request with",
            default="A chat between a curious user and an artificial intelligence assistant.\nThe assistant gives helpful, detailed, and polite answers to the user's questions. ",
        )
        parser.add_argument(
            "--nblm.use_deepspeed",
            action="store_true",
            default=False,
            help="Whether to use deepspeed or not (if not, uses vanilla huggingface).",
        )
        parser.add_argument(
            "--nblm.temperature",
            type=float,
            default=0.7,
            help="Sampling temperature.",
        )
        parser.add_argument(
            "--nblm.model",
            type=str,
            default="cerebras/btlm-3b-8k-base",
            help="model use for mine",
        )

    def __init__(self, *args, **kwargs):
        """
        Initializes the miner and loads the NBLM model.

        This method loads the NBLM model and tokenizer from the HuggingFace model hub and
        sets up the model pipeline for generation. It also sets up the stopping criteria
        for the model's generation.
        """
        super(NBLMMiner, self).__init__(*args, **kwargs)

        nb.logging.info("Loading {} model...".format(self.config.nblm.model))
        model = AutoModelForCausalLM.from_pretrained(
            self.config.nblm.model,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            torch_dtype="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.nblm.model,
            trust_remote_code=True,
        )
        self.stop_token_ids = tokenizer.convert_tokens_to_ids(["<|endoftext|>"])
        self.stop = StopOnTokens(self.stop_token_ids)

        # Determine correct device id (int) from device string.
        if self.config.nblm.device == "cuda":
            self.config.nblm.device = 0
        elif len(self.config.nblm.device.split(":")) == 2:
            try:
                self.config.nblm.device = int(
                    self.config.nblm.device.split(":")[1]
                )
            except:
                raise ValueError(
                    "Invalid device string: {}".format(self.config.nblm.device)
                )
        # Setup the pipeline for generating tokens
        self.pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.config.nblm.device,
            do_sample=self.config.nblm.do_sample,
            max_new_tokens=self.config.nblm.max_length,
            no_repeat_ngram_size=self.config.nblm.no_repeat_ngram_size,
        )
        # Optionally initialize deepspeed for inference speedup
        if self.config.nblm.use_deepspeed:
            self.pipe.model = deepspeed.init_inference(
                self.pipe.model,
                mp_size=int(os.getenv("WORLD_SIZE", "1")),
                dtype=torch.float,
                replace_with_kernel_inject=False,
            )

    def _process_history(self, roles: List[str], messages: List[str]) -> str:
        """
        Processes the conversation history for model input.

        This method takes the roles and messages from the incoming request and constructs
        a conversation history suitable for model input. It also injects a system request
        if the configuration specifies to do so.
        """
        processed_history = ""
        if self.config.nblm.do_request_injection:
            processed_history += self.config.nblm.system_request
        for role, message in zip(roles, messages):
            if role == "system":
                if (
                    not self.config.nblm.do_request_injection
                    or message != messages[0]
                ):
                    processed_history += "system: " + message + "\n"
            if role == "assistant":
                processed_history += "assistant: " + message + "\n"
            if role == "user":
                processed_history += "user: " + message + "\n"
        return processed_history

    def predict(self, nucleon: Inference) -> Inference:
        """
        Processes incoming requests using the NBLM model.

        This is a required method to implement and must take a `Inference` nucleon as input

        This method constructs a conversation history from the incoming request and uses
        the NBLM model to generate a response based on the provided context.
        """
        history = self._process_history(
            roles=nucleon.roles, messages=nucleon.messages
        )
        history += "assistant: "
        nb.logging.debug("History: {}".format(history))
        completion = (
            self.pipe(
                history,
                temperature=self.config.nblm.temperature,
                max_new_tokens=self.config.nblm.max_length,
                no_repeat_ngram_size=self.config.nblm.no_repeat_ngram_size,
                do_sample=self.config.nblm.do_sample,
                eos_token_id=self.pipe.tokenizer.eos_token_id,
                pad_token_id=self.pipe.tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([self.stop]),
            )[0]["generated_text"]
            .split(":")[-1]
            .replace(str(history), "")
        )
        nb.logging.debug("Completion: {}".format(completion))
        nucleon.completion = completion
        return nucleon


if __name__ == "__main__":
    """
    Main execution point for the NBLMMiner.

    This script initializes and runs the NBLMMiner, connecting it to the Nimble network.
    The miner listens for incoming requests and responds using the NBLM model.

    Developers can start the miner by executing this script. It uses the context manager to ensure
    proper cleanup of resources after the miner is stopped.
    """
    nb.debug()
    miner = NBLMMiner()
    with miner:
        while True:
            time.sleep(1)
