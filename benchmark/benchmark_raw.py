import json
import time
import argparse
import requests
import numpy as np

from tqdm import tqdm


def _bern_raw(text, url="https://bern.korea.ac.kr", use_bern2=False):
    if use_bern2:
        url = "http://163.152.20.133:8866"
    
    body_data ={"param":
        json.dumps({"text":text})
    }
    return requests.post(url, data=body_data).json()

def _bern_time(texts, use_bern2=False):
    to_times = []
    for text in tqdm(texts):
        start_time = time.time()
        result = _bern_raw(text, use_bern2=use_bern2)

        elapsed_time = time.time() - start_time
        to_times.append(elapsed_time)

    # stat
    print (f"bern = {np.mean(to_times)}")

def _hunflair_time(texts):
    import flair
    from flair.tokenization import SciSpacySentenceSplitter, SciSpacyTokenizer

    hunflair = flair.models.MultiTagger.load("hunflair")
    # neg_spec = flair.models.SequenceTagger.load("negation-speculation")

    to_times = []
    tokenizer = SciSpacyTokenizer()
    for text in tqdm(texts):
        start_time = time.time()
        sentence = flair.data.Sentence(text, use_tokenizer=tokenizer)
        hunflair.predict(sentence)
        # print (neg_spec.predict(sentence))

        elapsed_time = time.time() - start_time
        to_times.append(elapsed_time)

    # stat
    print(f"hunflair = {np.mean(to_times)}")

# load data
def _load_data(file_path):
    texts = []
    with open(file_path) as f:
        for line in f:
            d = json.loads(line)
            title = d['title']
            abstract = d['abstract']

            text = title + " " + abstract
            texts.append(text)
    return texts
    

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--tool', type=str, default='bern2')
    args = argparser.parse_args()

    # path to benchmark data
    file_path = './benchmark_1k.jsonl'

    # load benchmark data
    texts = _load_data(file_path)
    # check query size
    print(f"len(texts)={len(texts)}")

    if args.tool in ['bern', 'bern2']:
        if args.tool == 'bern':
            _bern_time(texts, use_bern2=False)
        elif args.tool == 'bern2':
            _bern_time(texts, use_bern2=True)

    elif args.tool == 'hunflair':
        _hunflair_time(texts)


if __name__ == "__main__":
    main()
