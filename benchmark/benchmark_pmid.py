import time
import json
import argparse
import requests
import numpy as np

from tqdm import tqdm

def _bern_pmid(pmid, url="https://bern.korea.ac.kr/pubmed", use_bern2=False, output_format='json', verbose=False):
    if use_bern2:
        # url = "http://163.152.20.133:8866/pubmed"
        url = "http://0.0.0.0:8888/pubmed"

        res = None
        if type(pmid) is str or type(pmid) is int:
            res = requests.get('{}?pmid={}'.format(url, pmid))
        elif type(pmid) is list:
            if len(pmid) == 0:
                print ('No pmid')
                return res
            
            pmid = [str(p) for p in pmid if type(p) is not str]
            res = requests.get('{}/{}/{}'.format(url, ','.join(pmid), output_format))

        if verbose:
            print ('pmid:', pmid, '\tresult:', res.text)

        if output_format == 'pubtator':
            return res.text

        return res.json()

    else:
        res = None
        if type(pmid) is str or type(pmid) is int:
            res = requests.get('{}/{}/{}'.format(url, pmid, output_format))
        elif type(pmid) is list:
            if len(pmid) == 0:
                print ('No pmid')
                return res
            
            pmid = [str(p) for p in pmid if type(p) is not str]
            res = requests.get('{}/{}/{}'.format(url, ','.join(pmid), output_format))

        if verbose:
            print ('pmid:', pmid, '\tresult:', res.text)

        if output_format == 'pubtator':
            return res.text

        return res.json()

def _bern_time(pmids, use_bern2=False):
    t_times = []
    ne_times = []
    n_times = []
    to_times = []

    # start_time = time.time()
    # result = _bern_pmid(','.join(pmids), use_bern2=use_bern2)
    # to_times.append(time.time()- start_time)

    for pmid in tqdm(pmids):
        if use_bern2:
            start_time = time.time()
            result = _bern_pmid(pmid, use_bern2=use_bern2)
            to_times.append(time.time()- start_time)
        else:
            result = _bern_pmid(pmid, use_bern2=use_bern2)
            elapse_time = result[0]['elapse_time']
            tmtool_elapse_time = elapse_time['tmtool']
            ner_elapse_time = elapse_time['ner']
            norm_elapse_time = elapse_time['normalization']
            total_elapse_time = elapse_time['total']

            t_times.append(tmtool_elapse_time)
            ne_times.append(ner_elapse_time)
            n_times.append(norm_elapse_time)
            to_times.append(total_elapse_time)

    if use_bern2:
        print(f"total = {np.mean(to_times)}")
    else:
        # stat time
        print(f"tmtool = {np.mean(t_times)}")
        print(f"ner = {np.mean(ne_times)}")
        print(f"normalize = {np.mean(n_times)}")
        print(f"total = {np.mean(to_times)}")

def _ptc_pmid(pmid, url="https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export", output_format='biocjson', verbose=False):
    '''
    output_format: pubtator, biocxml, biocjson
    '''

    res = None
    if type(pmid) is str or type(pmid) is int:
        # process all six biomedical concepts
        res = requests.get('{}/{}?{}={}'.format(url, output_format, "pmids", pmid))
    elif type(pmid) is list:
        if len(pmid) == 0:
            print ("No pmid")
            return res

        pmid = [str(p) for p in pmid if type(p) is not str]
        res = requests.get('{}/{}?{}={}'.format(url, output_format, "pmcids", pmid))

    if verbose:
        print ('pmid:', pmid, '\tresult:', res.text)

    if output_format == 'pubtator':
        return res.text

    return res.json()


def _ptc_time(pmids):
    to_times = []

    for pmid in tqdm(pmids):
        start_time = time.time()
        result = _ptc_pmid(pmid)
        elapse_time = time.time() - start_time
        to_times.append(elapse_time)

    print(f"total = {np.mean(to_times)}")

# load data
def _load_data(file_path):
    pmids = []
    with open(file_path) as f:
        for line in f:
            d = json.loads(line)
            pmid = d['pmid']
            pmids.append(pmid)

    return pmids

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--tool', type=str, default='bern2')
    args = argparser.parse_args()

    # path to benchmark data
    file_path = './benchmark_1k.jsonl'

    # load benchmark data
    pmids = _load_data(file_path)
    
    # check query size
    print(f"len(pmids)={len(pmids)}")

    if args.tool in ['bern', 'bern2']:
        if args.tool == 'bern':
            _bern_time(pmids, use_bern2=False)
        elif args.tool == 'bern2':
            _bern_time(pmids, use_bern2=True)
    elif args.tool == 'ptc':
        _ptc_time(pmids)
    

if __name__ == "__main__":
    main()