import re
import copy
import json
import time

import numpy as np
import xml.etree.ElementTree as ElTree

from datetime import datetime, timezone
from operator import itemgetter


tokenize_regex = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')

def json_to_sent(data):
    '''data: list of json file [{pmid,abstract,title}, ...] '''
    out = dict()
    for paper in data:
        sentences = list()
        
        if len(CoNLL_tokenizer(paper['title'])) < 50:
            title = [paper['title']]
        else:
            title = sentence_split(paper['title'])
        if len(title) != 1 or len(title[0].strip()) > 0:
            sentences.extend(title)

        if len(paper['abstract']) > 0:
            abst = sentence_split(paper['abstract'])
            if len(abst) != 1 or len(abst[0].strip()) > 0:
                sentences.extend(abst)
        out[paper['pmid']] = dict()
        out[paper['pmid']]['sentence'] = sentences
    return out

def input_form(sent_data):
    '''sent_data: dict of sentence, key=pmid {pmid:[sent,sent, ...], pmid: ...}'''
    for pmid in sent_data:
        sent_data[pmid]['words'] = list()
        sent_data[pmid]['wordPos'] = list()
        doc_piv = 0
        for sent in sent_data[pmid]['sentence']:
            wids = list()
            wpos = list()
            sent_piv = 0
            tok = CoNLL_tokenizer(sent)

            for w in tok:
                if len(w) > 20:
                    wids.append(w[:10])
                else:
                    wids.append(w)

                start = doc_piv + sent_piv + sent[sent_piv:].find(w)
                end = start + len(w) - 1
                sent_piv = end - doc_piv + 1
                wpos.append((start, end))
            doc_piv += len(sent)
            sent_data[pmid]['words'].append(wids)
            sent_data[pmid]['wordPos'].append(wpos)

    return sent_data

def softmax(logits):
    out = list()
    for logit in logits:
        temp = np.subtract(logit, np.max(logit))
        p = np.exp(temp) / np.sum(np.exp(temp))
        out.append(np.max(p))
    return out

def CoNLL_tokenizer(text):
    rawTok = [t for t in tokenize_regex.split(text) if t]
    assert ''.join(rawTok) == text
    tok = [t for t in rawTok if t != ' ']
    return tok

def sentence_split(text):
    sentences = list()
    sent = ''
    piv = 0
    for idx, char in enumerate(text):
        if char in "?!":
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            else:
                sent = text[piv:idx + 1]
                piv = idx + 1

        elif char == '.':
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            elif (text[idx + 1] == ' ') and (
                    text[idx + 2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-"' + "'"):
                sent = text[piv:idx + 1]
                piv = idx + 1

        if sent != '':
            toks = CoNLL_tokenizer(sent)
            if len(toks) > 100:
                while True:
                    rawTok = [t for t in tokenize_regex.split(sent) if t]
                    cut = ''.join(rawTok[:200])
                    sent = ''.join(rawTok[200:])
                    sentences.append(cut)

                    if len(CoNLL_tokenizer(sent)) < 100:
                        if sent.strip() == '':
                            sent = ''
                            break
                        else:
                            sentences.append(sent)
                            sent = ''
                            break
            else:
                sentences.append(sent)
                sent = ''

            if piv == -1:
                break

    if piv != -1:
        sent = text[piv:]
        toks = CoNLL_tokenizer(sent)
        if len(toks) > 100:
            while True:
                rawTok = [t for t in tokenize_regex.split(sent) if t]
                cut = ''.join(rawTok[:200])
                sent = ''.join(rawTok[200:])
                sentences.append(cut)

                if len(CoNLL_tokenizer(sent)) < 100:
                    if sent.strip() == '':
                        sent = ''
                        break
                    else:
                        sentences.append(sent)
                        sent = ''
                        break
        else:
            sentences.append(sent)
            sent = ''

    return sentences

def get_prob(data, sent_data, predicDict, logitsDict, entity_types=None):
    for idx, paper in enumerate(data):
        pmid = paper['pmid']
        
        if len(paper['abstract']) > 0:
            content = paper['title'] + ' ' + paper['abstract']
        else:
            content = paper['title']

        for ent_type in entity_types:
            paper['entities'][ent_type] = []
        paper['prob'] = dict()

        for dtype in entity_types:
            for sentidx, tags in enumerate(predicDict[dtype][pmid]):
                B_flag = False
                # get position of entity corresponding to types
                for widx, tag in enumerate(tags):
                    if tag == 'O':
                        if B_flag:
                            tmpSE["end"] = \
                            sent_data[pmid]['wordPos'][sentidx][widx - 1][1]
                            paper['entities'][dtype].append(tmpSE)
                        B_flag = False
                        continue
                    elif tag == 'B':
                        if B_flag:
                            tmpSE["end"] = \
                            sent_data[pmid]['wordPos'][sentidx][widx - 1][1]
                            paper['entities'][dtype].append(tmpSE)
                        tmpSE = {
                            "start": sent_data[pmid]['wordPos'][sentidx][widx][
                                0]}
                        B_flag = True
                    elif tag == "I":
                        continue
                if B_flag:
                    tmpSE["end"] = sent_data[pmid]['wordPos'][sentidx][-1][1]
                    paper['entities'][dtype].append(tmpSE)

            # get prob. of entity logits corresponding to types
            logs = list()
            for t_sent in logitsDict[dtype][pmid]:
                logs.extend(t_sent)
            paper['prob'][dtype] = list()
            for pos in paper['entities'][dtype]:
                if pos['start'] == pos['end']:
                    soft = softmax(logs[len(
                        CoNLL_tokenizer(content[:pos['start']])):len(
                        CoNLL_tokenizer(content[:pos['end']])) + 1])
                    paper['prob'][dtype].append(
                        (pos, float(np.average(soft))))
                else:
                    soft = softmax(logs[len(
                        CoNLL_tokenizer(content[:pos['start']])):len(
                        CoNLL_tokenizer(content[:pos['end']]))])
                    paper['prob'][dtype].append(
                        (pos, float(np.average(soft))))

    return data

def detokenize(tokens, predicts, logits):
    pred = dict({
        'toks': tokens[:],
        'labels': predicts[:],
        'logit': logits[:]
    })  # dictionary for predicted tokens and labels.

    bert_toks = list()
    bert_labels = list()
    bert_logits = list()
    tmp_p = list()
    tmp_l = list()
    tmp_s = list()
    for t, l, s in zip(pred['toks'], pred['labels'], pred['logit']):
        if t == '[CLS]' or t == '<s>':  # non-text tokens will not be evaluated.
            continue
        elif t == '[SEP]' or t == '</s>':  # newline
            bert_toks.append(tmp_p)
            bert_labels.append(tmp_l)
            bert_logits.append(tmp_s)
            tmp_p = list()
            tmp_l = list()
            tmp_s = list()
            continue
        elif t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
            try:
                tmp_p[-1] = tmp_p[-1] + t[2:]  # append pieces
            except:
                # for sliding window start point has ## - add start point to end tokens of previous results, update 23.11.13
                bert_toks[-1][-1] += t[2:]
        elif t.startswith('Ġ'): # roberta tokenizer
            t = t.replace('Ġ', ' ')
            tmp_p[-1] = tmp_p[-1] + t
        else:
            tmp_p.append(t)
            tmp_l.append(l)
            tmp_s.append(s)
    return bert_toks, bert_labels, bert_logits

# https://stackoverflow.com/a/3620972
PROF_DATA = {}

class Profile(object):
    def __init__(self, prefix):
        self.prefix = prefix

    def __call__(self, fn):
        def with_profiling(*args, **kwargs):
            global PROF_DATA
            start_time = time.time()
            ret = fn(*args, **kwargs)

            elapsed_time = time.time() - start_time
            key = '[' + self.prefix + '].' + fn.__name__

            if key not in PROF_DATA:
                PROF_DATA[key] = [0, list()]
            PROF_DATA[key][0] += 1
            PROF_DATA[key][1].append(elapsed_time)

            return ret

        return with_profiling

def show_prof_data():
    for fname, data in sorted(PROF_DATA.items()):
        max_time = max(data[1])
        avg_time = sum(data[1]) / len(data[1])
        total_time = sum(data[1])
        print("\n{} -> called {} times".format(fname, data[0]))
        print("Time total: {:.3f}, max: {:.3f}, avg: {:.3f}".format(
            total_time, max_time, avg_time))

def clear_prof_data():
    global PROF_DATA
    PROF_DATA = {}

# Ref. dict of SR4GN
species_human_excl_homo_sapiens = \
    'person|infant|Child|people|participants|woman|' \
    'Girls|Man|Peoples|Men|Participant|Patients|' \
    'humans|Persons|mans|participant|Infants|Boys|' \
    'Human|Humans|Women|children|Mans|child|Participants|Girl|' \
    'Infant|girl|patient|patients|boys|men|infants|' \
    'man|girls|Children|Boy|women|persons|human|Woman|' \
    'peoples|Patient|People|boy|Person'.split('|')
    
def filter_entities(ner_results):
    num_filtered_species_per_doc = list()

    for idx, paper in enumerate(ner_results):

        if len(paper['abstract']) > 0:
            content = paper['title'] + ' ' + paper['abstract']
        else:
            content = paper['title']

        valid_species = list()
        species = paper['entities']['species']
        for spcs in species:
            entity_mention = content[spcs['start']:spcs['end']+1]
            if entity_mention in species_human_excl_homo_sapiens:
                spcs['end'] += 1
                continue
            valid_species.append(spcs)

        num_filtered_species = len(species) - len(valid_species)
        if num_filtered_species > 0:
            paper['entities']['species'] = valid_species

        num_filtered_species_per_doc.append((paper['pmid'],
                                             num_filtered_species))

    return num_filtered_species_per_doc

# from convert.py
def pubtator2dict_list(pubtator_file_path):
    dict_list = list()

    title_pmid = ''
    # abstract_pmid = ''
    title = ''
    abstract_text = ''
    doc_line_num = 0

    with open(pubtator_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0:
                               
                doc_dict = {
                    'pmid': title_pmid,
                    'entities': {},
                }
                doc_dict['title'] = title
                doc_dict['abstract'] = abstract_text

                dict_list.append(doc_dict)

                doc_line_num = 0
                continue

            if doc_line_num == 0:
                title_cols = line.split('|t|')

                if len(title_cols) != 2:
                    return '{"error": "wrong #title_cols {}"}'\
                        .format(len(title_cols))

                title_pmid = title_cols[0]

                if '- No text -' == title_cols[1]:
                    # make tmvar2 results empty
                    title = ''
                else:
                    title = title_cols[1]
            elif doc_line_num == 1:
                abstract_cols = line.split('|a|')

                if len(abstract_cols) != 2:
                    if len(abstract_cols) > 2:
                        abstract_text = "|a|".join(abstract_cols[1:])
                    else:
                        return '{"error": "wrong #abstract_cols {}"}'.format(len(abstract_cols))
                else:
                    if '- No text -' == abstract_cols[1]:
                        # make tmvar2 results empty
                        abstract_text = ''
                    else:
                        abstract_text = abstract_cols[1]

            doc_line_num += 1
    return dict_list

def preprocess(text):
    text = text.replace('\r ', ' ')

    text = text.replace('\u2028', ' ')
    text = text.replace('\u2029', ' ')

    # HAIR SPACE
    # https://www.fileformat.info/info/unicode/char/200a/index.htm
    text = text.replace('\u200A', ' ')

    # THIN SPACE
    # https://www.fileformat.info/info/unicode/char/2009/index.htm
    text = text.replace('\u2009', ' ')
    text = text.replace('\u2008', ' ')

    # FOUR-PER-EM SPACE
    # https://www.fileformat.info/info/unicode/char/2005/index.htm
    text = text.replace('\u2005', ' ')
    text = text.replace('\u2004', ' ')
    text = text.replace('\u2003', ' ')

    # EN SPACE
    # https://www.fileformat.info/info/unicode/char/2002/index.htm
    text = text.replace('\u2002', ' ')

    # NO-BREAK SPACE
    # https://www.fileformat.info/info/unicode/char/00a0/index.htm
    text = text.replace('\u00A0', ' ')

    # https://www.fileformat.info/info/unicode/char/f8ff/index.htm
    text = text.replace('\uF8FF', ' ')

    # https://www.fileformat.info/info/unicode/char/202f/index.htm
    text = text.replace('\u202F', ' ')

    text = text.replace('\uFEFF', ' ')
    text = text.replace('\uF044', ' ')
    text = text.replace('\uF02D', ' ')
    text = text.replace('\uF0BB', ' ')

    text = text.replace('\uF048', 'Η')
    text = text.replace('\uF0B0', '°')

    # MIDLINE HORIZONTAL ELLIPSIS: ⋯
    # https://www.fileformat.info/info/unicode/char/22ef/index.htm
    # text = text.replace('\u22EF', '...')

    return text
