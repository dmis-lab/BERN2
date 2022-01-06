import json
import random

random.seed(0)

file_path = '/hdd1/mujeen/PLMs/corpus/pubmed190102_biobert/pubmed190102_bb63.json'
output_path = '/home/mujeen/works/bern/benchmark/benchmark_10.jsonl'

with open(file_path) as f:
    data = []
    for line in f:
        data.append(json.loads(line))

benchmark_set = random.sample(data, k=10)

with open(output_path, 'w') as f:
    for k in benchmark_set:
        k = json.dumps(k, ensure_ascii=False)
        f.write(k + "\n")
