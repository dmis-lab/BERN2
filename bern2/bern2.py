import random
import requests
import os
import string
import numpy as np
import hashlib
import time
import shutil
import asyncio
import socket
import struct
import json
import sys
from datetime import datetime
from collections import OrderedDict
import traceback
import bioregistry

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from convert import pubtator2dict_list, get_pub_annotation
from normalizer import Normalizer

import pymongo
from pymongo import MongoClient



class BERN2():
    def __init__(self, 
        gnormplus_home,
        gnormplus_port,
        tmvar2_home,
        tmvar2_port,
        mtner_home,
        mtner_port,
        gene_norm_port,
        disease_norm_port,
        cache_port,
        gnormplus_host='localhost',
        tmvar2_host='localhost',
        mtner_host='localhost',
        cache_host='localhost',
        time_format='[%d/%b/%Y %H:%M:%S.%f]',
        max_word_len=50, 
        seed=2019,
        use_neural_normalizer=True,
        keep_files=False,
        no_cuda=False):

        self.time_format = time_format

        print(datetime.now().strftime(self.time_format), 'BERN2 LOADING..')
        random.seed(seed)
        np.random.seed(seed)

        if not os.path.exists('./output'):
            os.mkdir('output')

        # delete prev. version outputs
        if not keep_files:
            delete_files('./output')
            delete_files(os.path.join(gnormplus_home, 'input'))
            delete_files(os.path.join(tmvar2_home, 'input'))
            delete_files(os.path.join('./multi_ner', 'input'))
            delete_files(os.path.join('./multi_ner', 'tmp'))
            delete_files(os.path.join('./multi_ner', 'output'))

        # FOR NER
        self.gnormplus_home =  gnormplus_home
        self.gnormplus_host = gnormplus_host
        self.gnormplus_port = gnormplus_port

        self.tmvar2_home =  tmvar2_home
        self.tmvar2_host = tmvar2_host
        self.tmvar2_port = tmvar2_port

        self.mtner_home =  mtner_home
        self.mtner_host = mtner_host
        self.mtner_port = mtner_port

        self.max_word_len = max_word_len

        # FOR NEN
        self.normalizer = Normalizer(
            gene_port = gene_norm_port,
            disease_port = disease_norm_port,
            use_neural_normalizer = use_neural_normalizer,
            no_cuda = no_cuda
        )

        # (Optional) For caching, use mongodb
        try:
            client = MongoClient(cache_host, cache_port, serverSelectionTimeoutMS = 2000)
            client.server_info()
            self.caching_db = client.bern2_v1_1.pmid
        except Exception as e:
            self.caching_db = None

        print(datetime.now().strftime(self.time_format), 'BERN2 LOADED..')
    
    def annotate_text(self, text, pmid=None):
        try:
            text = text.strip()
            base_name = self.generate_base_name(text) # for the name of temporary files
            text = self.preprocess_input(text, base_name)
            output = self.tag_entities(text, base_name)
            output['error_code'], output['error_message'] = 0, ""
            output = self.post_process_output(output)
        except Exception as e:
            errStr = traceback.format_exc()
            print(errStr)

            output = {"error_code": 1, "error_message": "Something went wrong. Try again."}

        return output

    def annotate_pmid(self, pmid):
        pmid = pmid.strip()
        
        # validate pmid
        if not pmid.isdecimal():
            print(f"warn! pmid is not valid: {pmid}")
            output = {
                'pmid': pmid,
                'text': ""
            }
            return output

        # if pmid is cached in db, return it
        output = None
        if self.caching_db:
            output = self.caching_db.find_one({"_id": pmid})
            if output:
                # hotfix
                if 'pmid' not in output:
                    self.caching_db.delete_one({'_id': pmid})
                    output = None
                elif 'error_code' in output and output['error_code'] != 0:
                    self.caching_db.delete_one({'_id': pmid})
                    output = None
        
        # otherwise, get pubmed article from web and annotate the text
        if output is None:    
            text, status_code = self.get_text_data_from_pubmed(pmid)

            if status_code == 200:
                output = OrderedDict()
                output['pmid'] = pmid 
                json_result = self.annotate_text(text, pmid)
                
                output.update(json_result)

                # if db is running, cache the annotation into db
                if self.caching_db:
                    output['_id'] = pmid
                    self.caching_db.insert_one(output)

            # error from pubmed (Not Found)
            else:
                output = {
                    'pmid': pmid,
                    'text': ""
                }

        return self.post_process_output(output)

    def post_process_output(self, output):
        # hotfix
        if 'annotations' not in output:
            return output
        
        # split_cuis (e.g., "OMIM:608627,MESH:C563895" => ["OMIM:608627","MESH:C563895"])
        output = self.split_cuis(output)

        # standardize prefixes (e.g., EntrezGene:10533 => NCBIGene:10533)
        output = self.standardize_prefixes(output)

        return output

    def split_cuis(self, output):
        # "OMIM:608627,MESH:C563895" or "OMIM:608627|MESH:C563895" 
        # => ["OMIM:608627","MESH:C563895"]

        for anno in output['annotations']:
            cuis = anno['id']
            new_cuis = []
            for cui in cuis:
                # hotfix in case cui is ['cui-less']
                if isinstance(cui, list):
                    cui = cui[0]

                new_cuis += cui.replace("|", ",").split(",")
            anno['id'] = new_cuis                 
        return output

    def standardize_prefixes(self, output):
        # EntrezGene:10533 => NCBIGene:10533
        for anno in output['annotations']:
            cuis = anno['id']
            obj = anno['obj']
            if obj not in ['disease', 'gene', 'drug', 'species', 'cell_line', 'cell_type']:
                continue

            new_cuis = []
            for cui in cuis:
                if "NCBI:txid" in cui: # NCBI:txid10095
                    prefix, numbers = cui.split("NCBI:txid")
                    prefix = "ncbitaxon"
                elif "_" in cui: # CVCL_J260
                    prefix, numbers = cui.split("_")
                elif ":" in cui: # MESH:C563895
                    prefix, numbers = cui.split(":")
                else:
                    new_cuis.append(cui)
                    continue
                    
                normalized_prefix = bioregistry.normalize_prefix(prefix)
                if normalized_prefix is not None:
                    prefix = normalized_prefix
                
                preferred_prefix = bioregistry.get_preferred_prefix(prefix)
                if preferred_prefix is not None:
                    prefix = preferred_prefix
                
                # to convert CVCL_J260 to cellosaurus:CVCL_J260
                if prefix == 'cellosaurus':
                    numbers = "CVCL_" + numbers

                new_cuis.append(":".join([prefix,numbers]))
            
            anno['id'] = new_cuis

        return output

    def get_text_data_from_pubmed(self, pmid):
        title = ""
        abstract = ""

        URL = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/?format=pubmed"
        response = requests.get(URL) 
        status_code = response.status_code
        
        if status_code == 200:
            is_abs = False
            is_title = False
            for line in response.text.split("\n"):
                if line.startswith("TI  -"):
                    is_title = True
                    title = line.replace("TI  -", "").strip()
                elif is_title and line.startswith("      "):
                    title += " " + line.strip()
                elif line.startswith("AB  -"):
                    is_title = False
                    is_abs = True
                    abstract += line.replace("AB  -", "").strip()
                elif is_abs and line.startswith("      "):
                    abstract += " " + line.strip()
                else:
                    is_abs = False
            text = title + " " + abstract
        else:
            print(f"warn! response.status_code={response.status_code}")
            text = ""
            
        return text, status_code

    def preprocess_input(self, text, base_name):
        if '\r\n' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a CRLF -> replace it w/ a space')
            text = text.replace('\r\n', ' ')

        if '\n' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a line break -> replace it w/ a space')
            text = text.replace('\n', ' ')

        if '\t' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a tab -> replace w/ a space')
            text = text.replace('\t', ' ')

        if '\xa0' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a \\xa0 -> replace w/ a space')
            text = text.replace('\xa0', ' ')

        if '\x0b' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a \\x0b -> replace w/ a space')
            text = text.replace('\x0b', ' ')
            
        if '\x0c' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a \\x0c -> replace w/ a space')
            text = text.replace('\x0c', ' ')
        
        # remove non-ascii
        text = text.encode("ascii", "ignore").decode()

        found_too_long_words = 0
        tokens = text.split(' ')
        for idx, tk in enumerate(tokens):
            if len(tk) > self.max_word_len:
                tokens[idx] = tk[:self.max_word_len]
                found_too_long_words += 1
        if found_too_long_words > 0:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a too long word -> cut the suffix of the word')
            text = ' '.join(tokens)

        return text

    def tag_entities(self, text, base_name):
        n_ascii_letters = 0
        for l in text:
            if l not in string.ascii_letters:
                continue
            n_ascii_letters += 1

        if n_ascii_letters == 0:
            text = 'No ascii letters. Please enter your text in English.'

        base_name = self.generate_base_name(text)
        print(datetime.now().strftime(self.time_format),
              f'id: {base_name}')

        pubtator_file = f'{base_name}.PubTator'
        input_gnormplus = os.path.join(self.gnormplus_home, 'input', pubtator_file)
        output_gnormplus = os.path.join(self.gnormplus_home, 'output', pubtator_file)

        input_dir_tmvar2 = os.path.join(self.tmvar2_home, 'input')
        input_tmvar_ner = os.path.join(input_dir_tmvar2, pubtator_file)
        output_tmvar_ner = os.path.join(self.tmvar2_home, 'output',
                                     f'{pubtator_file}.PubTator')
        input_tmvar_gene = os.path.join(self.tmvar2_home, 'input',
                                     f'{pubtator_file}.PubTator.Gene')
        input_tmvar_norm = os.path.join(self.tmvar2_home, 'input',
                                     f'{pubtator_file}.PubTator')
        output_tmvar_norm = os.path.join(self.tmvar2_home, 'output',
                                     f'{pubtator_file}.PubTator.PubTator')
        input_mtner = os.path.join(self.mtner_home, 'input',
                                     f'{pubtator_file}.PubTator')
        output_mtner = os.path.join(self.mtner_home, 'output',
                                     f'{pubtator_file}.json')

        if not os.path.exists(self.mtner_home + '/input'):
            os.mkdir(self.mtner_home + '/input')
        if not os.path.exists(self.mtner_home + '/output'):
            os.mkdir(self.mtner_home + '/output')

        # Write input str to a .PubTator format file
        with open(input_gnormplus, 'w', encoding='utf-8') as f:
            # only abstract
            f.write(f'{base_name}|t|\n')
            f.write(f'{base_name}|a|{text}\n\n')

        shutil.copy(input_gnormplus, input_tmvar_ner)
        shutil.copy(input_gnormplus, input_mtner)
        ner_start_time = time.time()
        
        # async call for gnormplus and tmvar
        arguments_for_coroutines = []
        loop = asyncio.new_event_loop()
        for ner_type in ['tmvar', 'gnormplus', 'mtner']:
            arguments_for_coroutines.append([ner_type, pubtator_file, output_mtner, base_name, loop])
        async_result = loop.run_until_complete(self.async_ner(arguments_for_coroutines))
        loop.close()
        gnormplus_elapse_time = async_result['gnormplus_elapse_time']
        tmvar2_elapse_time = async_result['tmvar2_elapse_time']
        mtner_elapse_time = async_result['mtner_elapse_time']

        # get output result to merge
        tagged_docs = async_result['tagged_docs']
        num_entities = async_result['num_entities']
        
        # mutation normalization using the outputs of gnormplus and tmvar
        # TODO! need to check beforehand the output of gnormplus and tmvar
        tm_norm_start_time = time.time()
        shutil.move(output_gnormplus, input_tmvar_gene)
        shutil.move(output_tmvar_ner, input_tmvar_norm)
        sync_tell_inputfile(self.tmvar2_host,
                       self.tmvar2_port,
                       os.path.basename(input_tmvar_norm) + "|" + os.path.basename(input_tmvar_gene))
        tmvar2_elapse_time += time.time() - tm_norm_start_time # add normalization time 

        tmvar_docs = pubtator2dict_list(output_tmvar_norm)
        
        ner_elapse_time = time.time() - ner_start_time
        print(datetime.now().strftime(self.time_format),
              f'[{base_name}] ALL NER {ner_elapse_time} sec')

        # Rule-based Normalization models
        r_norm_start_time = time.time()
        if num_entities > 0:
            tagged_docs = self.normalizer.normalize(base_name, tagged_docs)
        r_norm_elapse_time = time.time() - r_norm_start_time

        # Neural-based normalization models
        n_norm_start_time = time.time()
        if self.normalizer.use_neural_normalizer and num_entities > 0:
            tagged_docs = self.normalizer.neural_normalize(
                ent_type='disease', 
                tagged_docs=tagged_docs
            )
            tagged_docs = self.normalizer.neural_normalize(
                ent_type='drug', 
                tagged_docs=tagged_docs
            )
            tagged_docs = self.normalizer.neural_normalize(
                ent_type='gene', 
                tagged_docs=tagged_docs
            )

        tagged_docs = self.resolve_overlap(tagged_docs, tmvar_docs)
        n_norm_elapse_time = time.time() - n_norm_start_time

        print(datetime.now().strftime(self.time_format),
            f'[{base_name}] Neural Normalization {n_norm_elapse_time} sec')

        # Convert to PubAnnotation JSON
        tagged_docs[0] = get_pub_annotation(tagged_docs[0])

        norm_elapse_time = r_norm_elapse_time + n_norm_elapse_time
        print(datetime.now().strftime(self.time_format),
              f'[{base_name}] ALL NORM {norm_elapse_time} sec')

        # time record
        tagged_docs[0]['elapse_time'] = {
            'gnormplus_elapse_time': gnormplus_elapse_time,
            'tmvar2_elapse_time':tmvar2_elapse_time,
            'mtner_elapse_time':mtner_elapse_time,
            'ner_elapse_time': ner_elapse_time,
            'r_norm_elapse_time':r_norm_elapse_time,
            'n_norm_elapse_time':n_norm_elapse_time,
            'norm_elapse_time':norm_elapse_time,
        } 

        # Delete temp files
        os.remove(input_gnormplus)
        os.remove(input_tmvar_ner)
        os.remove(input_tmvar_gene)
        os.remove(input_tmvar_norm)
        os.remove(output_tmvar_norm)
        os.remove(input_mtner)
        os.remove(output_mtner)
        
        return tagged_docs[0]

    def resolve_overlap(self, tagged_docs, tmvar_docs):
        """
        Step 1: check CUI and logit probability for same mention
        Step 2: check overlap with mutation and tags with the highest probability
        """

        # [Step 1] compare CUI and probability for same mention
        span2mentions = {}
        for entity_type, entity_dict in tagged_docs[0]['entities'].items():
            # check CUI and probability
            for mention_idx, mention_dict in enumerate(entity_dict):
                start = mention_dict['start']
                end = mention_dict['end']
                if "%d-%d" % (start, end) not in span2mentions:
                    span2mentions["%d-%d" % (start, end)] = []
                
                span2mentions["%d-%d"%(start, end)].append({"type":entity_type,
                                                            "CUI": mention_dict['id'],
                                                            "check_CUI": 1 if mention_dict['id'] != 'CUI-less' else 0,
                                                            "prob": tagged_docs[0]['prob'][entity_type][mention_idx][1],
                                                            "is_neural_normalized":mention_dict['is_neural_normalized']})
        
        for span in span2mentions.keys():
            # sort elements with CUI
            span2mentions[span] = sorted(span2mentions[span], key=lambda x:(x['check_CUI'], x['prob']), reverse=True)

        for entity_type, entity_dict in tagged_docs[0]['entities'].items():
            update_list = []
            for mention_idx, mention_dict in enumerate(entity_dict):
                start = mention_dict['start']
                end = mention_dict['end']
                
                if span2mentions["%d-%d"%(start, end)][0]['CUI'] == mention_dict['id'] and span2mentions["%d-%d"%(start, end)][0]['type'] == entity_type:
                    update_list.append(mention_dict)

            tagged_docs[0]['entities'].update({entity_type:update_list})

        # [Step 2] add mutation annotation
        tagged_docs[0]['entities']['mutation'] = tmvar_docs[0]['entities']['mutation']
                       
        return tagged_docs

    # generate id for temporary files
    def generate_base_name(self, text):
        # add time.time() to avoid collision
        base_name = hashlib.sha224((text+str(time.time())).encode('utf-8')).hexdigest()
        return base_name

    async def async_ner(self, arguments):
        coroutines = [self._ner_wrap(*arg) for arg in arguments]
        result = await asyncio.gather(*coroutines)
        result = {k:v for e in result for k,v in e.items()} # merge
        return result

    async def _ner_wrap(self, ner_type, pubtator_file, output_mtner, base_name, loop):
        if ner_type == 'gnormplus':
            # Run GNormPlus
            gnormplus_start_time = time.time()
            gnormplus_resp = await async_tell_inputfile(self.gnormplus_host,
                                            self.gnormplus_port,
                                            pubtator_file,
                                            loop)
            # Print time for GNormPlus
            gnormplus_elapse_time = time.time() - gnormplus_start_time
            print(datetime.now().strftime(self.time_format),
                f'[{base_name}] GNormPlus {gnormplus_elapse_time} sec')

            return {"gnormplus_elapse_time": gnormplus_elapse_time,
                    "gnormplus_resp": gnormplus_resp}

        elif ner_type == 'tmvar':
            # Run tmVar 2.0
            tmvar2_start_time = time.time()
            tmvar2_resp = await async_tell_inputfile(self.tmvar2_host,
                                         self.tmvar2_port,
                                         pubtator_file,
                                         loop)
            tmvar2_elapse_time = time.time() - tmvar2_start_time
            print(datetime.now().strftime(self.time_format),
                f'[{base_name}] tmVar 2.0 {tmvar2_elapse_time} sec')

            return {"tmvar2_elapse_time": tmvar2_elapse_time,
                    "tmvar2_resp": tmvar2_resp}

        elif ner_type == 'mtner':            
            # Run neural model
            start_time = time.time()
            mtner_resp = await async_tell_inputfile(self.mtner_host,
                                         self.mtner_port,
                                         pubtator_file,
                                         loop)
            
            with open(output_mtner, 'r', encoding='utf-8') as f:
                tagged_docs = [json.load(f)]

            num_entities = tagged_docs[0]['num_entities']
            if tagged_docs is None:
                return None

            assert len(tagged_docs) == 1
            mtner_elapse_time = time.time() - start_time
            print(datetime.now().strftime(self.time_format),
                f'[{base_name}] Multi-task NER {mtner_elapse_time} sec, #entities: {num_entities}')

            return {"mtner_elapse_time": mtner_elapse_time,
                    "tagged_docs": tagged_docs,
                    "num_entities": num_entities}

async def async_tell_inputfile(host, port, inputfile, loop):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
        input_str = inputfile
        input_stream = struct.pack('>H', len(input_str)) + input_str.encode(
            'utf-8')
        sock.send(input_stream)
        # output_stream = sock.recv(512)
        output_stream = await loop.run_in_executor(None, sock.recv, 512) # for async
        resp = output_stream.decode('utf-8')[2:]

        sock.close()
        return resp
    except ConnectionRefusedError as e:
        print(e)
        return None
    except TimeoutError as e:
        print(e)
        return None
    except ConnectionResetError as e:
        print(e)
        return None

def sync_tell_inputfile(host, port, inputfile):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
        input_str = inputfile
        input_stream = struct.pack('>H', len(input_str)) + input_str.encode(
            'utf-8')
        sock.send(input_stream)
        output_stream = sock.recv(512) # for sync
        # output_stream = await loop.run_in_executor(None, sock.recv, 512)
        resp = output_stream.decode('utf-8')[2:]

        sock.close()
        return resp
    except ConnectionRefusedError as e:
        print(e)
        return None
    except TimeoutError as e:
        print(e)
        return None
    except ConnectionResetError as e:
        print(e)
        return None

def delete_files(dirname):
    if not os.path.exists(dirname):
        return

    n_deleted = 0
    for f in os.listdir(dirname):
        f_path = os.path.join(dirname, f)
        if not os.path.isfile(f_path):
            continue
        # print('Delete', f_path)
        os.remove(f_path)
        n_deleted += 1
    print(dirname, n_deleted)

if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--max_word_len', type=int, help='word max chars',
                           default=50)
    argparser.add_argument('--seed', type=int, help='seed value', default=2019)
    argparser.add_argument('--gnormplus_home',
                           help='GNormPlus home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'bern', 'GNormPlusJava'))
    argparser.add_argument('--gnormplus_host',
                           help='GNormPlus host', default='localhost')
    argparser.add_argument('--gnormplus_port', type=int,
                           help='GNormPlus port', default=18895)
    argparser.add_argument('--tmvar2_home',
                           help='tmVar 2.0 home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'bern', 'tmVarJava'))
    argparser.add_argument('--tmvar2_host',
                           help='tmVar 2.0 host', default='localhost')
    argparser.add_argument('--tmvar2_port', type=int,
                           help='tmVar 2.0 port', default=18896)
    argparser.add_argument('--mtner_home',
                           help='biomedical language model home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'bern', 'mtnerHome'))
    argparser.add_argument('--mtner_host',
                           help='biomedical language model host', default='localhost')
    argparser.add_argument('--mtner_port', type=int, 
                           help='biomedical language model port', default=18894)
    argparser.add_argument('--cache_host',
                           help='annotation cached db host', default='localhost')
    argparser.add_argument('--cache_port', type=int, 
                           help='annotation cached db port', default=27017)
    argparser.add_argument('--gene_norm_port', type=int,
                           help='GNormPlus port', default=18888)
    argparser.add_argument('--disease_norm_port', type=int,
                           help='Sieve port', default=18892)
    argparser.add_argument('--time_format',
                           help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')
    argparser.add_argument("--use_neural_normalizer", action="store_true")
    argparser.add_argument("--keep_files", action="store_true")
    argparser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    args = argparser.parse_args()

    bern2 = BERN2(
        max_word_len=args.max_word_len,
        seed=args.seed,
        gnormplus_home=args.gnormplus_home,
        gnormplus_host=args.gnormplus_host,
        gnormplus_port=args.gnormplus_port,
        tmvar2_home=args.tmvar2_home,
        tmvar2_host=args.tmvar2_host,
        tmvar2_port=args.tmvar2_port,
        gene_norm_port=args.gene_norm_port,
        disease_norm_port=args.disease_norm_port,
        mtner_home=args.mtner_home,
        mtner_host=args.mtner_host,
        mtner_port=args.mtner_port,
        cache_host=args.cache_host,
        cache_port=args.cache_port,
        time_format=args.time_format,
        use_neural_normalizer=args.use_neural_normalizer,
        keep_files=args.keep_files,
        no_cuda=args.no_cuda,
    )

    # result = bern2.annotate_text("cancer is a disease")
    # print(result)
    result = bern2.annotate_pmid("30429607")
    print(result)
