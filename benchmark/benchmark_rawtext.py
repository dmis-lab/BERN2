
import requests
import json
import numpy as np
from tqdm import tqdm
import time

file_path = './benchmark_1k.jsonl'
def query_raw(text, url="http://0.0.0.0:8888"):
# def query_raw(text, url="https://bern.korea.ac.kr"):
    body_data ={"param":
        json.dumps({"text":text})
    }
    return requests.post(url, data=body_data).json()

# load data
texts = []
with open(file_path) as f:
    for line in f:
        d = json.loads(line)
        title = d['title']
        abstract = d['abstract']

        text = title + " " + abstract
        texts.append(text)

# query all
print(f"len(texts)={len(texts)}")

g_times = []
t_times = []
b_times = []
ne_times = []
rn_times = []
nn_times = []
n_times = []

# debug
# texts = texts[:5]

# warmup
for text in tqdm(texts[:5], desc="warmup"):
    result = query_raw(text)

query_times = []

for text in tqdm(texts):
    start_time = time.time()
    result = query_raw(text)
    elapse_time = result['elapse_time']
    gnormplus_elapse_time = elapse_time['gnormplus_elapse_time'] if 'gnormplus_elapse_time' in elapse_time else 0.0
    tmvar2_elapse_time = elapse_time['tmvar2_elapse_time']  if 'tmvar2_elapse_time' in elapse_time else 0.0
    biobert_elapse_time = elapse_time['biobert_elapse_time']
    ner_elapse_time = elapse_time['ner_elapse_time'] if 'ner_elapse_time' in elapse_time else 0.0
    norm_elapse_time = elapse_time['norm_elapse_time']
    r_norm_elapse_time = elapse_time['r_norm_elapse_time'] if 'r_norm_elapse_time' in elapse_time else 0.0
    n_norm_elapse_time = elapse_time['n_norm_elapse_time'] if 'n_norm_elapse_time' in elapse_time else 0.0
    norm_elapse_time = elapse_time['norm_elapse_time']

    g_times.append(gnormplus_elapse_time)
    t_times.append(tmvar2_elapse_time)
    b_times.append(biobert_elapse_time)
    ne_times.append(ner_elapse_time)
    rn_times.append(r_norm_elapse_time)
    nn_times.append(n_norm_elapse_time)
    n_times.append(norm_elapse_time)

    query_times.append(time.time()-start_time)


# stat time
print(f"ner = {round(np.mean(ne_times),3)}")
print(f"..gnormplus = {round(np.mean(g_times),3)}")
print(f"..tmvar = {round(np.mean(t_times),3)}")
print(f"..biobert = {round(np.mean(b_times),3)}")
print(f"normalize = {round(np.mean(n_times),3)}")
print(f"..rule norm = {round(np.mean(rn_times),3)}")
print(f"..neural norm = {round(np.mean(nn_times),3)}")
print(f"total time: mean={round(np.mean(query_times),3)} std={round(np.std(query_times),3)}")

