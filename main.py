from typing import List, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from storyteller import CLIPTeller

if __name__ == "__main__":

    # Iterate over models.

    # model_name = "distilgpt2"
    model_name = "gpt2-medium"
    # model_name = "gpt2"

    test_baseline = False

    story = CLIPTeller(model_name=model_name,
                       batch_size=8).eval_set(test_baseline=test_baseline)
    print(story)

    # Example output:
#  �Eating from a home in Huntingdon in August was a major part of the eco-trotting programme.
#   In September 2011, Eating from a home in Huntingdon, WA was the single most popular meal in Australia.
#   The popular seafood meal was being served to the public from watery drive-ins on beaches and beaches.
#   In August 2012, the Australian government introduced a pet food rule that allowed dogs (or bulls) to be found on beaches and beaches.
#   In May 2013, a petition by an online
