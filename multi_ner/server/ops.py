import re
import time

import numpy as np

tokenize_regex = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')


def json_to_sent(data, is_raw_text=False):
    '''data: list of json file [{pmid,abstract,title}, ...] '''
    out = dict()
    for paper in data:
        sentences = list()
        if is_raw_text:
            # assure that paper['abstract'] is not empty
            abst = sentence_split(paper['abstract'])
            if len(abst) != 1 or len(abst[0].strip()) > 0:
                sentences.extend(abst)
        else:
            # assure that paper['title'] is not empty
            if len(CoNLL_tokenizer(paper['title'])) < 50:
                title = [paper['title']]
            else:
                title = sentence_split(paper['title'])
            if len(title) != 1 or len(title[0].strip()) > 0:
                sentences.extend(title)

            if len(paper['abstract']) > 0:
                abst = sentence_split(' ' + paper['abstract'])
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


# def isInt(string):
#     try:
#         int(string)
#         return True
#     except ValueError:
#         return False


# def isFloat(string):
#     try:
#         float(string)
#         return True
#     except ValueError:
#         return False


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


def _overlap(paper, content, entity_types, log_list):
    for first_idx, first_dtype in enumerate(entity_types):
        for sec_idx, sec_dtype in enumerate(entity_types):
            # combination of searching overlap entity types
            if sec_idx > first_idx:
                first_ent = paper['entities'][first_dtype][:]
                sec_ent = paper['entities'][sec_dtype][:]
                for first_e in first_ent:
                    removed_first_e = False

                    for sec_e in sec_ent:
                        if first_e['end'] == sec_e['end'] and first_e['start'] == sec_e['start']:
                            if first_e['end'] == first_e['start']:
                                first_soft = softmax(
                                    log_list[first_dtype][len(CoNLL_tokenizer(
                                        content[:first_e['start']])) : len(CoNLL_tokenizer(
                                        content[:first_e['end']])) + 1]
                                )
                                sec_soft = softmax(
                                    log_list[sec_dtype][len(CoNLL_tokenizer(
                                        content[:sec_e['start']])) : len(CoNLL_tokenizer(
                                        content[:sec_e['end']])) + 1]
                                )
                            else:
                                first_soft = softmax(
                                    log_list[first_dtype][len(CoNLL_tokenizer(
                                        content[:first_e['start']])) : len(CoNLL_tokenizer(
                                        content[:first_e['end']]))]
                                )
                                sec_soft = softmax(
                                    log_list[sec_dtype][len(CoNLL_tokenizer(
                                        content[:sec_e['start']])) : len(CoNLL_tokenizer(
                                        content[:sec_e['end']]))]
                                )
                            if np.average(first_soft) < np.average(sec_soft):
                                paper['entities'][first_dtype].remove(first_e)
                                removed_first_e = True
                                break
                            elif np.average(first_soft) > np.average(sec_soft):
                                paper['entities'][sec_dtype].remove(sec_e)
                                break
                            
                        elif first_e['end'] < sec_e['start']:
                            break


def merge_results(data, sent_data, predicDict, logitsDict, rep_ent,
                  is_raw_text=False, entity_types=None):
    for idx, paper in enumerate(data):
        pmid = paper['pmid']
        if is_raw_text:
            content = paper['abstract']
        else:
            if len(paper['abstract']) > 0:
                content = paper['title'] + ' ' + paper['abstract']
            else:
                content = paper['title']

        for ent_type in entity_types:
            paper['entities'][ent_type] = []
        paper['logit'] = dict()

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

            # get logit prob. of entity corresponding to types
            logs = list()
            for t_sent in logitsDict[dtype][pmid]:
                logs.extend(t_sent)
            paper['logit'][dtype] = list()
            for pos in paper['entities'][dtype]:
                if pos['start'] == pos['end']:
                    soft = softmax(logs[len(
                        CoNLL_tokenizer(content[:pos['start']])):len(
                        CoNLL_tokenizer(content[:pos['end']])) + 1])
                    paper['logit'][dtype].append(
                        (pos, float(np.average(soft))))
                else:
                    soft = softmax(logs[len(
                        CoNLL_tokenizer(content[:pos['start']])):len(
                        CoNLL_tokenizer(content[:pos['end']]))])
                    paper['logit'][dtype].append(
                        (pos, float(np.average(soft))))

    if rep_ent:
        return data
    else:
        # check overlap entities and resolve with highest logit value
        # import time
        # start_time = time.time()
        for idx, paper in enumerate(data):
            pmid = paper['pmid']
            if is_raw_text:
                content = paper['abstract']
            else:
                if len(paper['abstract']) > 0:
                    content = paper['title'] + ' ' + paper['abstract']
                else:
                    content = paper['title']

            log_list = {k:list() for k in entity_types}
            for dtype in entity_types:
                for t_sent in logitsDict[dtype][pmid]:
                    log_list[dtype].extend(t_sent)

            _overlap(paper, content, entity_types, log_list)
            
        # print ("overlap time:  {:.3f} sec.".format(time.time()-start_time))
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
            tmp_p[-1] = tmp_p[-1] + t[2:]  # append pieces
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
    
def filter_entities(ner_results, is_raw_text):
    num_filtered_species_per_doc = list()

    for idx, paper in enumerate(ner_results):

        if is_raw_text:
            content = paper['abstract']
        else:
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
