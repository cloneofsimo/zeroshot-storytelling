from typing import List, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from storyteller import CLIPTeller

if __name__ == "__main__":
    model_name = "distilgpt2"
    # model_name = "gpt2"

    # model_name = "gpt2-medium"

    stories = CLIPTeller(model_name=model_name).generate_vist_story_v2(
        story_id=40470, init_str=" ", temperature=0.8)

    story = CLIPTeller(
        model_name=model_name).continue_single_image_caption()[0]

    print(story)
#  �Eating from a home in Huntingdon in August was a major part of the eco-trotting programme.
#   In September 2011, Eating from a home in Huntingdon, WA was the single most popular meal in Australia.
#   The popular seafood meal was being served to the public from watery drive-ins on beaches and beaches.
#   In August 2012, the Australian government introduced a pet food rule that allowed dogs (or bulls) to be found on beaches and beaches.
#   In May 2013, a petition by an online
