import os
import requests
import json
from multiprocessing import Pool
import pandas as pd


def download_file(url):
    url, download_path = url
    filename = url.split("/")[-1]
    r = requests.get(url, stream=True)
    with open(os.path.join(download_path, filename), "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
                f.flush()
    print("Downloaded {}".format(filename))


def fast_download_ruls(url_lists, download_path):
    """
    Use multiprocessing to fast download imgs
    """

    url_download_joined = list(zip(url_lists, [download_path] * len(url_lists)))

    pool = Pool(processes=20)
    pool.map(download_file, url_download_joined)


def download_files(url_lists, download_path):
    """
    Download files from a list of URLs.
    :param url_lists: list of URLs
    :param download_path: path to save the files
    """
    for url in url_lists:
        if url is None:
            continue
        filename = url.split("/")[-1]
        r = requests.get(url, stream=True)
        with open(os.path.join(download_path, filename), "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()
        print("Downloaded {}".format(filename))


def load_info_from_json(json_path):
    info = json.load(open(json_path))

    def printd(di):
        print(len(di))
        print(di[0])

    print(info.keys())
    printd(info["images"])
    printd(info["info"])
    printd(info["albums"])
    printd(info["type"])
    printd(info["annotations"])


def download_img_from_json(json_path):
    info = json.load(open(json_path))
    img_links = [ob.get("url_o", None) for ob in info["images"]]
    fast_download_ruls(img_links, "./val")


def arrange_and_save_csv(json_path, save_path):
    info = json.load(open(json_path))
    img_ids_titles_url = [
        (
            ob.get("id", None),
            ob.get("title", None),
            ob.get("url_o", None).split("/")[-1],
        )
        for ob in info["images"]
        if ob.get("url_o", None) is not None
    ]

    idx_url_dict = {a: c for a, b, c in img_ids_titles_url}

    annots = info["annotations"]

    stories = []
    for annot in annots:
        annot = annot[0]

        idx = annot["photo_flickr_id"]
        url = idx_url_dict.get(idx, None)
        if url is None:
            continue

        stories.append(
            (
                annot["story_id"],
                annot["text"],
                url,
                annot["worker_arranged_photo_order"],
            )
        )

    df = pd.DataFrame(stories, columns=["story_id", "text", "url", "order"])
    df.to_csv("val.csv", index=False)

    print(stories)


if __name__ == "__main__":
    types = "val"
    arrange_and_save_csv(
        f"/home/simo/dl/global_dataset/VIST/sis/{types}.story-in-sequence.json",
        f"./{types}.csv",
    )
