from typing import List, Tuple, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import pandas as pd
from guider import CLIPMaximizer

torch.no_grad()


def load_model(name: str = "distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tokenizer


class CLIPTeller:
    def __init__(self, model_name: str = "distilgpt2", device: str = "cuda:0"):

        # Load guider
        self.device = device
        self.guider = CLIPMaximizer(device=device)

        print("Done Loading CLIP")

        # Load generator
        self.model, self.tokenizer = load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Done Loading Model")

    @staticmethod
    def get_story_sentence_img_pairs(story_id: int = 40470):
        val_path = "./VIST/sis/val.csv"
        annot = pd.read_csv(val_path)
        l = annot.loc[annot["story_id"] == story_id]
        texts = l["text"].values.tolist()
        imgs = l["url"].values.tolist()
        return texts, imgs

    @torch.no_grad()
    def random_continue(
        self,
        init_str: Union[str, List[str]] = "Description: This is a story of",
        n: int = 100,
        max_length: int = 3,
        temperature: float = 0.7,
    ):
        """
        Use gpt to create random story
        """

        if isinstance(init_str, str):
            story = [init_str] * n
        else:
            n = len(init_str)
            story = init_str

        if isinstance(init_str, str):
            encoded_input = self.tokenizer(story, return_tensors="pt").to(self.device)

        else:
            encoded_input_ids = []
            for st in story:
                this_enc = self.tokenizer(st, return_tensors="pt")["input_ids"]
                # print(this_enc)
                encoded_input_ids.append(this_enc[0])

            lens = [len(enc) for enc in encoded_input_ids]

            min_length = min(lens)
            truncated = [enc[:min_length] for i, enc in enumerate(encoded_input_ids)]

            # print("TRUCCATED at", min_length)

            encoded_input = {
                "input_ids": torch.stack(truncated).to(self.device),
                "attention_mask": torch.stack(
                    [torch.ones_like(enc) for enc in truncated]
                ).to(self.device),
            }

        # print(encoded_input['input_ids'].shape)

        # print(encoded_input)

        for i in range(max_length):

            encoded_output = self.model(
                **encoded_input,
            )
            # print(encoded_output)
            encoded_output = encoded_output["logits"].detach()
            # random sample from softmax of logits

            softmaxed = (encoded_output[:, -1, :] / temperature).softmax(dim=-1)

            sample = torch.multinomial(softmaxed, 1)
            # append sample to encoded input

            encoded_input["input_ids"] = torch.cat(
                [encoded_input["input_ids"], sample], dim=-1
            )
            encoded_input["attention_mask"] = torch.tensor(
                [1] * len(encoded_input["input_ids"])
            ).to(self.device)

        for i in range(n):
            story[i] = self.tokenizer.decode(encoded_input["input_ids"][i])

        return story

    @torch.no_grad()
    def generate_single_img_story(
        self,
        img_path="./VIST/sis/val/693397887_7a3eee6eeb_o.jpg",
        n_pool: int = 1200,
        n_candidate: int = 400,
        total_length: str = 200,
        batch_size: int = 16,
        temperature: float = 1.0,
    ):
        """
        Use gpt to create random story
        """

        # Load image
        self.guider.set_img(img_path)

        # Get story
        init_batch = self.random_continue(n=n_pool, max_length=5)
        for idx in range(100):

            scores = []
            print(
                f"{idx}'s batch : {len(init_batch)}, Current Length : {len(init_batch[0])}"
            )
            jdx = 0
            while jdx < len(init_batch):
                this_score = (
                    self.guider.score(init_batch[jdx : jdx + batch_size])
                    .squeeze()
                    .tolist()
                )
                scores.extend(this_score)
                jdx += batch_size

            scores = torch.tensor(scores)
            val, inds = scores.topk(n_candidate)
            print("Max Score : ", scores[inds].max())
            cands = [init_batch[i] for i in inds]

            if len(cands[0]) > total_length:
                break

            cands = cands * (n_pool // n_candidate)

            print(cands[0])

            jdx = 0
            init_batch = []
            while jdx < len(cands):
                init_batch.extend(
                    self.random_continue(
                        cands[jdx : jdx + batch_size], n=n_pool, max_length=3
                    )
                )
                jdx += batch_size

        return cands

    @torch.no_grad()
    def generate_story(
        self,
        story_id: int = 40470,
        init_str: str = "This is story of",
        n_pool: int = 1600,
        n_candidate: int = 800,
        total_length: str = 500,
        gradual_length: int = 100,
        batch_size: int = 16,
        temperature: float = 0.7,
        base_path: str = "./VIST/sis/val/",
        search_length: int = 2,
    ):

        texts, img_urls = self.get_story_sentence_img_pairs(story_id)

        # Get story
        init_batch = self.random_continue(
            init_str, n=n_pool, max_length=search_length, temperature=temperature
        )
        n_imgs = len(img_urls)
        stages = list(range(1, n_imgs + 1))
        sidx = 0
        self.guider.set_img(base_path + img_urls[sidx])

        for idx in range(100):

            scores = []
            print(
                f"{idx}'s batch : {len(init_batch)}, Current Length : {len(init_batch[0])}"
            )
            jdx = 0
            while jdx < len(init_batch):
                start_len = max(sidx * gradual_length - 20, 0)
                trunc_set = [
                    se[start_len:] for se in init_batch[jdx : jdx + batch_size]
                ]
                this_score = self.guider.score(trunc_set).squeeze().tolist()
                scores.extend(this_score)
                jdx += batch_size

            scores = torch.tensor(scores)
            val, inds = scores.topk(n_candidate)
            print("Max Score : ", scores[inds].max())
            cands = [init_batch[i] for i in inds]

            if len(cands[0]) > total_length:
                break

            if len(cands[0]) > gradual_length * stages[sidx]:
                sidx += 1
                self.guider.set_img(base_path + img_urls[sidx])
                print("Setted Image")

            cands = cands * (n_pool // n_candidate)

            print(cands[0])

            jdx = 0
            init_batch = []
            while jdx < len(cands):
                init_batch.extend(
                    self.random_continue(
                        cands[jdx : jdx + batch_size],
                        n=n_pool,
                        max_length=search_length,
                        temperature=temperature,
                    )
                )
                jdx += batch_size

        return cands


if __name__ == "__main__":
    model_name = "distilgpt2"
    # model_name = "gpt2"
    # model_name = "gpt2-medium"

    stories = CLIPTeller(model_name=model_name).generate_story(
        story_id=45529, init_str=" ", temperature=0.8
    )

    print(stories)

#  �Eating from a home in Huntingdon in August was a major part of the eco-trotting programme.
#   In September 2011, Eating from a home in Huntingdon, WA was the single most popular meal in Australia.
#   The popular seafood meal was being served to the public from watery drive-ins on beaches and beaches.
#   In August 2012, the Australian government introduced a pet food rule that allowed dogs (or bulls) to be found on beaches and beaches.
#   In May 2013, a petition by an online
