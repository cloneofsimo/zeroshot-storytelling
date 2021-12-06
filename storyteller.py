from typing import List, Tuple, Union
from PIL.Image import Image

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from guider import CLIPGuide


def load_model(name: str = "distilgpt2"):
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    return model, tokenizer


class CLIPTeller:
    def __init__(self,
                 model_name: str = "distilgpt2",
                 device: str = "cuda:0",
                 batch_size: int = 16) -> None:

        # Load guider
        self.device = device
        self.guider = CLIPGuide(device=device)
        self.batch_size = batch_size

        print("Done Loading CLIP")

        # Load generator
        self.model, self.tokenizer = load_model(model_name)
        self.model.to(self.device)
        self.model.eval()

        print("Done Loading Model")

    @staticmethod
    def get_story_sentence_img_pairs(
            story_id: int = 40470) -> Tuple[List[str], List[str]]:
        val_path = "./VIST/sis/val.csv"
        annot = pd.read_csv(val_path)
        l = annot.loc[annot["story_id"] == story_id]
        texts = l["text"].values.tolist()
        imgs = l["url"].values.tolist()
        return texts, imgs

    @torch.no_grad()
    def lm_beam(
        self,
        init_str: Union[str, List[str]] = "Description: This is a story of",
        n: int = 100,
        beam_length: int = 3,
        temperature: float = 0.7,
    ) -> List[str]:
        """
        Use gpt to create random story
        """

        if isinstance(init_str, str):
            story = [init_str] * n
        else:
            n = len(init_str)
            story = init_str

        if isinstance(init_str, str):
            encoded_input = self.tokenizer(story,
                                           return_tensors="pt").to(self.device)

        else:
            encoded_input_ids = []
            for st in story:
                this_enc = self.tokenizer(st, return_tensors="pt")["input_ids"]
                # print(this_enc)
                encoded_input_ids.append(this_enc[0])

            lens = [len(enc) for enc in encoded_input_ids]

            min_length = min(lens)
            truncated = [
                enc[:min_length] for i, enc in enumerate(encoded_input_ids)
            ]

            # print("TRUCCATED at", min_length)

            encoded_input = {
                "input_ids":
                torch.stack(truncated).to(self.device),
                "attention_mask":
                torch.stack([torch.ones_like(enc)
                             for enc in truncated]).to(self.device),
            }

        # print(encoded_input['input_ids'].shape)

        # print(encoded_input)

        for i in range(beam_length):

            encoded_output = self.model(**encoded_input, )
            # print(encoded_output)
            encoded_output = encoded_output["logits"].detach()
            # random sample from softmax of logits

            softmaxed = (encoded_output[:, -1, :] /
                         temperature).softmax(dim=-1)

            sample = torch.multinomial(softmaxed, 1)
            # append sample to encoded input

            encoded_input["input_ids"] = torch.cat(
                [encoded_input["input_ids"], sample], dim=-1)
            encoded_input["attention_mask"] = torch.tensor(
                [1] * len(encoded_input["input_ids"])).to(self.device)

        for i in range(n):
            story[i] = self.tokenizer.decode(encoded_input["input_ids"][i])

        return story

    @torch.no_grad()
    def _batch_lm_beam(
        self,
        init_str: List[str],
        n: int = 100,
        beam_length: int = 3,
        temperature: float = 0.7,
    ) -> List[str]:
        jdx = 0
        init_batch = []
        while jdx < len(init_str):
            init_batch.extend(
                self.lm_beam(
                    init_str[jdx:jdx + self.batch_size],
                    n=n,
                    beam_length=beam_length,
                    temperature=temperature,
                ))
            jdx += self.batch_size

        return init_batch

    @torch.no_grad()
    def continue_single_image_caption(
        self,
        init_str: Union[str, List[str]] = "This is image of",
        img_path: str = "./VIST/sis/val/693397887_7a3eee6eeb_o.jpg",
        n_pool: int = 1200,
        n_candidate: int = 400,
        extension_length: str = 30,
        temperature: float = 1.0,
        beam_length: int = 3,
        verbose: bool = False,
    ) -> List[str]:
        """
        Continue Single Image Caption.
        """

        # Load image
        self.guider.set_img(img_path)

        if isinstance(init_str, str):
            init_length = len(init_str)
        else:
            init_length = len(init_str[0])

        init_batch = self._batch_lm_beam(init_str,
                                         n=n_pool,
                                         beam_length=beam_length,
                                         temperature=temperature)

        for idx in range(100):

            scores = self.guider.batch_score(init_batch, self.batch_size)
            scores = torch.tensor(scores)
            val, inds = scores.topk(n_candidate)
            cands = [init_batch[i] for i in inds]

            if verbose:
                print(
                    f"{idx}'s batch : {len(init_batch)}, Best Candidate's Current Length : {len(init_batch[0])}"
                )
                print("Current Best Candidate", cands[0])
                print("Max Score : ", scores[inds].max())

            if len(cands[0]) >= init_length + extension_length:
                break

            cands = cands * (n_pool // n_candidate)

            cands = self._batch_lm_beam(
                init_str=cands,
                n=n_pool,
                beam_length=beam_length,
                temperature=temperature,
            )

            init_batch = cands

        return cands

    @torch.no_grad()
    def generate_vist_story_v2(
        self,
        story_id: int = 40470,
        img_base_path: str = "./VIST/sis/val/",
        init_str: str = "This is story of",
        n_pool: int = 400,
        n_candidate: int = 200,
        gradual_length: int = 100,
        temperature: float = 0.7,
        beam_length: int = 2,
        verbose: bool = True,
        early_stop_idx: int = None,
    ):
        texts, img_urls = self.get_story_sentence_img_pairs(story_id)

        # Get story
        init_batch = self._batch_lm_beam(init_str,
                                         n=n_pool,
                                         beam_length=beam_length,
                                         temperature=temperature)
        n_imgs = len(img_urls)

        for idx in range(n_imgs):

            init_batch = self.continue_single_image_caption(
                init_str=init_batch,
                img_path=img_base_path + img_urls[idx],
                n_pool=n_pool,
                n_candidate=n_candidate,
                extension_length=gradual_length,
                temperature=temperature,
                beam_length=beam_length,
                verbose=verbose,
            )

            if idx is early_stop_idx:
                break

        return init_batch

    @torch.no_grad()
    def _generate_vist_story(  # Legacy. 
        self,
        story_id: int = 40470,
        img_base_path: str = "./VIST/sis/val/",
        init_str: str = "This is story of",
        n_pool: int = 1600,
        n_candidate: int = 800,
        total_length: str = 500,
        gradual_length: int = 100,
        temperature: float = 0.7,
        beam_length: int = 2,
    ) -> List[str]:

        texts, img_urls = self.get_story_sentence_img_pairs(story_id)

        # Get story
        init_batch = self.lm_beam(init_str,
                                  n=n_pool,
                                  beam_length=beam_length,
                                  temperature=temperature)
        n_imgs = len(img_urls)
        sidx = 0

        self.guider.set_img(img_base_path + img_urls[sidx])

        for idx in range(100):

            scores = []
            print(
                f"{idx}'s Iteration : {len(init_batch)}, Current Length : {len(init_batch[0])}"
            )

            start_len = max(sidx * gradual_length - 20, 0)
            trunc_set = [se[start_len:] for se in init_batch]

            scores = self.guider.batch_score(trunc_set,
                                             batch_size=self.batch_size)
            scores = torch.tensor(scores)

            val, inds = scores.topk(n_candidate)
            print("Max Score : ", scores[inds].max())
            cands = [init_batch[i] for i in inds]

            if len(cands[0]) > total_length:
                break

            elif len(cands[0]) > gradual_length * (sidx + 1):
                sidx += 1
                self.guider.set_img(img_base_path + img_urls[sidx])
                print(f"\n\n Setted Image To {sidx}th Image")

            cands = cands * (n_pool // n_candidate)

            print(f"Current Best Candidate : {cands[0]}")

            init_batch = self._batch_lm_beam(
                init_str=cands,
                n=n_pool,
                beam_length=beam_length,
                temperature=temperature,
            )

        return cands

    def acc_evaluate(self,
                     img_path: str,
                     sentence: str,
                     prv_sentences: List[str],
                     n_pool: int = 800,
                     n_candidate: int = 400,
                     extension_for_eval: int = 8,
                     beam_length: int = 2,
                     temperature: float = 0.5,
                     verbose: bool = False,
                     test_baseline: bool = False) -> float:
        """
        Evaluate the score of a sentence with a given image
        """
        self.guider.set_img(img_path)
        prv_sentences = " ".join(prv_sentences)

        cnt = 0

        for end_ptr in range(len(sentence)):
            init_str = prv_sentences + " " + sentence[:end_ptr]
            gt_char = sentence[end_ptr]

            if test_baseline:
                init_str = [init_str] * 2
                candidates = self.lm_beam(init_str,
                                          n=2,
                                          beam_length=extension_for_eval,
                                          temperature=temperature)
            else:
                candidates = self.continue_single_image_caption(
                    init_str=init_str,
                    img_path=img_path,
                    n_pool=n_pool,
                    n_candidate=n_candidate,
                    extension_length=extension_for_eval,
                    temperature=temperature,
                    beam_length=beam_length,
                    verbose=False,
                )

            #print("DOING", end_ptr, "th eval", candidates[0], init_str)

            char_hat = candidates[0][len(init_str)]
            if char_hat == gt_char:
                cnt += 1

                #print("\n\n GOT RIGHT \n\n")

        return cnt / len(sentence)

    def eval_vist_story(
        self,
        story_id: int = 40470,
        img_base_path: str = "./VIST/sis/val/",
        verbose: bool = True,
        n_set: int = 2,
        test_baseline: bool = False,
    ) -> float:
        texts, img_urls = self.get_story_sentence_img_pairs(story_id)

        tot_acc = 0

        for idx in range(n_set):
            acc = self.acc_evaluate(img_base_path + img_urls[idx],
                                    texts[idx],
                                    texts[:idx],
                                    verbose=verbose,
                                    test_baseline=test_baseline)
            tot_acc += acc

        return tot_acc / n_set

    def eval_set(self,
                 n_set: int = 2,
                 n_testset: int = 5,
                 test_baseline: bool = False) -> float:
        testset = [
            40470, 40471, 40472, 40473, 40474, 40475, 40476, 40477, 40478,
            40479
        ]

        total_acc = 0

        for idx in testset[:n_testset]:
            acc = self.eval_vist_story(story_id=idx,
                                       n_set=n_set,
                                       verbose=False,
                                       test_baseline=test_baseline)
            total_acc += acc
            print(f"Accuracy of Story {idx} : {acc}")

        return total_acc / n_testset


if __name__ == "__main__":

    init_str = "This is a picture of"

    model_name = "distilgpt2"
    #model_name = "gpt2-medium"
    #model_name = "gpt2"

    story = CLIPTeller(model_name=model_name).generate_vist_story_v2(
        story_id=40481,
        init_str=init_str,
        temperature=1.0,
        n_pool=800,
        n_candidate=200,
        early_stop_idx=2,
        gradual_length=100)[0]

    print("DONE GEN")
    print(story)

    # This is a pictur from the YMCA's website, which shows a group of kids playing with a toy called the "smoke ball."

    # This is a picturational example of a boy in a group of kids playing a game with a girl.
