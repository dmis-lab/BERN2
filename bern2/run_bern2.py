import argparse
import hashlib
import json
import os
import random
import string
import time
from datetime import datetime

import bioregistry
import numpy as np
import pandas as pd

from bern2.bern2.convert import get_pub_annotation
# from bern2.normalizer import Normalizer
from bern2.multi_ner.main import MTNER
from bern2.multi_ner.ner_server import mtner_recognize

argparser = argparse.ArgumentParser()
argparser.add_argument('--max_word_len', type=int, help='word max chars', default=50)
argparser.add_argument('--seed', type=int, help='seed value', default=2019)
argparser.add_argument('--mtner_home', help='biomedical language model home',
                       default=os.path.join(os.path.expanduser('~'), 'bern', 'mtnerHome'))
argparser.add_argument('--time_format', help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')
argparser.add_argument("--use_neural_normalizer", action="store_true")
argparser.add_argument("--keep_files", action="store_true")

args = argparser.parse_args()


def append_text_to_pubtator(input_mtner: str, pmid: str, text: str):
    # Write input str to a .PubTator format file
    with open(input_mtner, 'a', encoding='utf-8') as f:
        # only abstract
        f.write(f'{pmid}|t|\n')
        f.write(f'{pmid}|a|{text}\n\n')


class LocalBERN2():
    def __init__(self,
                 mtner_home,
                 time_format='[%d/%b/%Y %H:%M:%S.%f]',
                 max_word_len=50,
                 seed=2019,
                 use_neural_normalizer=True,
                 keep_files=False):

        self.time_format = time_format

        print(datetime.now().strftime(self.time_format), 'BERN2 LOADING..')
        random.seed(seed)
        np.random.seed(seed)

        if not os.path.exists('./output'):
            os.mkdir('output')

        # delete prev. version outputs
        if not keep_files:
            delete_files('./output')
            delete_files(os.path.join('multi_ner', 'input'))
            delete_files(os.path.join('multi_ner', 'tmp'))
            delete_files(os.path.join('multi_ner', 'output'))

        # FOR NER
        self.mtner_home = mtner_home
        if not os.path.exists(self.mtner_home):
            os.mkdir(self.mtner_home)

        self.max_word_len = max_word_len

        print(datetime.now().strftime(self.time_format), 'BERN2 LOADED..')

    def annotate_text(self, list_of_texts: list):
        # TODO: Make this a pd.Series instead of a list
        list_of_texts = [text.strip() for text in list_of_texts]
        base_name = self.generate_base_name(list_of_texts[0])  # for the name of temporary files
        list_of_texts = [self.preprocess_input(text, base_name) for text in list_of_texts]
        output = self.tag_entities(list_of_texts, base_name)
        output = self.post_process_output(output)
        return output

    def post_process_output(self, output):
        # split_cuis (e.g., "OMIM:608627,MESH:C563895" => ["OMIM:608627","MESH:C563895"])
        output = [self.split_cuis(doc) for doc in output]

        # standardize prefixes (e.g., EntrezGene:10533 => NCBIGene:10533)
        output = [self.standardize_prefixes(doc) for doc in output]

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
                if "NCBI:txid" in cui:  # NCBI:txid10095
                    prefix, numbers = cui.split("NCBI:txid")
                    prefix = "ncbitaxon"
                elif "_" in cui:  # CVCL_J260
                    prefix, numbers = cui.split("_")
                elif ":" in cui:  # MESH:C563895
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

                new_cuis.append(":".join([prefix, numbers]))

            anno['id'] = new_cuis

        return output

    def preprocess_input(self, text, base_name):
        if '\r\n' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a CRLF -> replace it w/ a space')
            text = text.replace('\r\n', '\u200c')

        if '\n' in text:
            print(datetime.now().strftime(self.time_format),
                  f'[{base_name}] Found a line break -> replace it w/ a space')
            text = text.replace('\n', '\u200c')

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
        # text = text.encode("ascii", "ignore").decode()

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

    def tag_entities(self, list_of_texts: list, base_name):
        base_name = self.generate_base_name(list_of_texts[0])
        print(datetime.now().strftime(self.time_format),
              f'id: {base_name}')

        pubtator_file = f'{base_name}.PubTator'

        input_mtner = os.path.join(self.mtner_home, 'input',
                                   f'{pubtator_file}.PubTator')
        output_mtner = os.path.join(self.mtner_home, 'output',
                                    f'{pubtator_file}.json')

        if not os.path.exists(self.mtner_home + '/input'):
            os.mkdir(self.mtner_home + '/input')
        if not os.path.exists(self.mtner_home + '/output'):
            os.mkdir(self.mtner_home + '/output')

        for text in list_of_texts:
            pmid = self.generate_base_name(text)
            append_text_to_pubtator(input_mtner, pmid, text)

        ner_start_time = time.time()
        ner_result = self.ner(pubtator_file, output_mtner, base_name)

        mtner_elapse_time = ner_result['mtner_elapse_time']

        # get output result to merge
        tagged_docs = ner_result['tagged_docs']
        num_entities = ner_result['num_entities']

        ner_elapse_time = time.time() - ner_start_time
        print(datetime.now().strftime(self.time_format),
              f'[{base_name}] ALL NER {ner_elapse_time} sec')

        # Rule-based Normalization models
        # r_norm_start_time = time.time()
        # if num_entities > 0:
        #     tagged_docs = self.normalizer.normalize(base_name, tagged_docs)
        # r_norm_elapse_time = time.time() - r_norm_start_time
        #
        # # Neural-based normalization models
        # n_norm_start_time = time.time()
        # if self.normalizer.use_neural_normalizer and num_entities > 0:
        #     tagged_docs = self.normalizer.neural_normalize(
        #         ent_type='disease',
        #         tagged_docs=tagged_docs
        #     )
        #     tagged_docs = self.normalizer.neural_normalize(
        #         ent_type='drug',
        #         tagged_docs=tagged_docs
        #     )
        #     tagged_docs = self.normalizer.neural_normalize(
        #         ent_type='gene',
        #         tagged_docs=tagged_docs
        #     )

        # tagged_docs = self.resolve_overlap(tagged_docs, tmvar_docs)
        # n_norm_elapse_time = time.time() - n_norm_start_time

        # print(datetime.now().strftime(self.time_format),
        #       f'[{base_name}] Neural Normalization {n_norm_elapse_time} sec')

        # Convert to PubAnnotation JSON
        tagged_docs = [get_pub_annotation(tagged_doc) for tagged_doc in tagged_docs]

        # norm_elapse_time = r_norm_elapse_time + n_norm_elapse_time
        # print(datetime.now().strftime(self.time_format),
        #       f'[{base_name}] ALL NORM {norm_elapse_time} sec')

        # time record
        [d.update({'elapse_time': {'mtner_elapse_time': mtner_elapse_time, 'ner_elapse_time': ner_elapse_time}}) for d
         in tagged_docs]

        # Delete temp files
        os.remove(input_mtner)
        os.remove(output_mtner)

        return tagged_docs

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

                span2mentions["%d-%d" % (start, end)].append({"type": entity_type,
                                                              "CUI": mention_dict['id'],
                                                              "check_CUI": 1 if mention_dict['id'] != 'CUI-less' else 0,
                                                              "prob": tagged_docs[0]['prob'][entity_type][mention_idx][
                                                                  1],
                                                              "is_neural_normalized": mention_dict[
                                                                  'is_neural_normalized']})

        for span in span2mentions.keys():
            # sort elements with CUI
            span2mentions[span] = sorted(span2mentions[span], key=lambda x: (x['check_CUI'], x['prob']), reverse=True)

        for entity_type, entity_dict in tagged_docs[0]['entities'].items():
            update_list = []
            for mention_idx, mention_dict in enumerate(entity_dict):
                start = mention_dict['start']
                end = mention_dict['end']

                if span2mentions["%d-%d" % (start, end)][0]['CUI'] == mention_dict['id'] and \
                        span2mentions["%d-%d" % (start, end)][0]['type'] == entity_type:
                    update_list.append(mention_dict)

            tagged_docs[0]['entities'].update({entity_type: update_list})

        # [Step 2] add mutation annotation
        tagged_docs[0]['entities']['mutation'] = tmvar_docs[0]['entities']['mutation']

        return tagged_docs

    # generate id for temporary files
    def generate_base_name(self, text):
        # add time.time() to avoid collision
        base_name = hashlib.sha224((text + str(time.time())).encode('utf-8')).hexdigest()
        return base_name

    def ner(self, pubtator_file, output_mtner, base_name) -> dict:
        # Run neural model
        start_time = time.time()
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--seed', type=int, help='random seed for initialization', default=1)
        argparser.add_argument('--model_name_or_path', default='dmis-lab/bern2-ner')
        argparser.add_argument('--max_seq_length', type=int,
                               help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.',
                               default=128)
        argparser.add_argument('--mtner_home', help='biomedical language model home')
        argparser.add_argument('--disease_only', help='use disease only NER', type=bool, default=True)
        argparser.add_argument('--time_format', help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')
        mt_ner_params = argparser.parse_args()

        mt_ner_model = MTNER(mt_ner_params)
        base_name = pubtator_file.split('.')[0]
        # hotfix
        base_name = base_name.replace("\x00A", "")

        mtner_recognize(mt_ner_model, pubtator_file, base_name, self.mtner_home)

        with open(output_mtner, 'r', encoding='utf-8') as f:
            tagged_docs = json.load(f)

        num_entities = tagged_docs[0]['num_entities']
        if tagged_docs is None:
            return None

        mtner_elapse_time = time.time() - start_time
        print(datetime.now().strftime(self.time_format),
              f'[{base_name}] Multi-task NER {mtner_elapse_time} sec, #entities: {num_entities}')

        return {"mtner_elapse_time": mtner_elapse_time,
                "tagged_docs": tagged_docs,
                "num_entities": num_entities}


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


def run_bern2_on_batch(df: pd.DataFrame, text_col: str = 'content'):
    bern2 = get_initialized_bern()
    results = []
    for cur_text in df[text_col]:
        temp = bern2.annotate_text(cur_text)
        results.append(temp)
    res_df = pd.DataFrame(results, index=df.index)
    res_df = pd.merge(df, res_df, left_index=True, right_index=True, how='left')
    return res_df


def run_bern2_annotation(list_of_texts: list) -> list:
    if initialize_bern2_annotator.annotator is None:
        raise Exception('BERN2 annotator is not initialized!')
    results = initialize_bern2_annotator.annotator.annotate_text(list_of_texts)
    annotations = [result['annotations'] for result in results]
    return annotations


def initialize_bern2_annotator(max_word_len: int = 50,
                               mtner_home: str = os.path.join(os.path.expanduser('~'), 'bern', 'mtnerHome'),
                               use_neural_normalizer: bool = False, keep_files: bool = False):
    if initialize_bern2_annotator.annotator is None:
        initialize_bern2_annotator.annotator = LocalBERN2(max_word_len=max_word_len,
                                                          seed=args.seed,
                                                          mtner_home=mtner_home,
                                                          time_format=args.time_format,
                                                          use_neural_normalizer=use_neural_normalizer,
                                                          keep_files=keep_files)


initialize_bern2_annotator.annotator = None


def get_initialized_bern():
    bern2 = LocalBERN2(
        max_word_len=args.max_word_len,
        seed=args.seed,
        mtner_home=args.mtner_home,
        time_format=args.time_format,
        use_neural_normalizer=args.use_neural_normalizer,
        keep_files=args.keep_files)

    return bern2
