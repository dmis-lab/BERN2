# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import logging
import os
import time
import json
import torch
import argparse
import numpy as np

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, NewType, NamedTuple, Union, Tuple
from tqdm import tqdm
from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler

from transformers import (
    AutoConfig,
    AutoTokenizer,
    set_seed,
    PreTrainedTokenizer,
)

from ops import (
    json_to_sent, 
    input_form, 
    get_prob,
    detokenize, 
    preprocess, 
    Profile, 
)
from modeling import RoBERTaMultiNER2

logger = logging.getLogger(__name__)

InputDataClass = NewType("InputDataClass", Any)
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, torch.Tensor]])

@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]
    entity_labels: Optional[List[int]]

@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    entity_type_ids: Optional[List[int]] = None

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, data, pmids):
        """Reads a BIO data."""
        lines = []
        words = []
        labels = []
        entity_labels = []
        for pmid in pmids:
            for sent in data[pmid]['words']:
                words = sent[:]
                labels = ['O'] * len(words)
                entity_labels = [str(0)] * len(words)
                
                if len(words) >= 30:
                    while len(words) >= 30:
                        tmplabel = labels[:30]
                        l = ' '.join([label for label
                                      in labels[:len(tmplabel)]
                                      if len(label) > 0])
                        w = ' '.join([word for word
                                      in words[:len(tmplabel)]
                                      if len(word) > 0])
                        e = ' '.join([el for el
                                      in entity_labels[:len(tmplabel)]
                                      if len(el) > 0])              
                        lines.append([l, w, e])
                        words = words[len(tmplabel):]
                        labels = labels[len(tmplabel):]
                        entity_labels = entity_labels[len(tmplabel):]
                if len(words) == 0:
                    continue

                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                e = ' '.join([el for el in entity_labels if len(entity_labels) > 0])
                lines.append([l, w, e])
                words = []
                labels = []
                entity_labels = []
                continue

        return lines

class NerDataset(Dataset):
    """
        This will be superseded by a framework-agnostic approach soon.
    """
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    def __init__(
        self,
        predict_examples,
        labels: List[str],
        tokenizer: PreTrainedTokenizer,
        config,
        params,
        base_name
    ):
        logger.info(f"Creating features from dataset file")
        self.labels = labels
        self.predict_examples = predict_examples
        self.tokenizer = tokenizer
        self.config = config
        self.params = params

        self.features = convert_examples_to_features(
            self.predict_examples,
            self.labels,
            self.params.max_seq_length,
            self.tokenizer,
            cls_token_at_end=bool(self.config.model_type in ["xlnet"]),
            cls_token=self.tokenizer.cls_token,
            cls_token_segment_id=2 if self.config.model_type in ["xlnet"] else 0,
            sep_token=self.tokenizer.sep_token,
            sep_token_extra=False,
            pad_on_left=bool(self.tokenizer.padding_side=="left"),
            pad_token=self.tokenizer.pad_token_id,
            pad_token_segment_id=self.tokenizer.pad_token_type_id,
            pad_token_label_id=self.pad_token_label_id,
            base_name=base_name,
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]

def default_data_collator(features: List[InputDataClass]) -> Dict[str, torch.Tensor]:
    """
    Very simple data collator that:
    - simply collates batches of dict-like objects
    - Performs special handling for potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    - does not do any additional preprocessing

    i.e., Property names of the input object will be used as corresponding inputs to the model.
    See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.
    if not isinstance(features[0], dict):
        features = [vars(f) for f in features]

    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        dtype = torch.long if type(first["label"]) is int else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features], dtype=torch.long)

    return batch


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: PreTrainedTokenizer,
    cls_token_at_end=False,
    cls_token="[CLS]",
    cls_token_segment_id=1,
    sep_token="[SEP]",
    sep_token_extra=False,
    pad_on_left=False,
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
    base_name="",
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    # TODO clean up all this to leverage built-in features of tokenizers

    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    
    for (ex_index, example) in tqdm(enumerate(examples)):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens, label_ids, = [], []
        det_tokens = []

        for word_idx, (word, label) in enumerate(zip(example.words.split(), example.labels.split())):
            word_tokens = tokenizer.tokenize(word)
            
            # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

                if len(word_tokens) == 1:
                    det_tokens.extend(word_tokens)
                elif len(word_tokens) > 1:
                    for det_idx, det_word in enumerate(word_tokens):
                        if det_idx > 0:
                            det_word = '##' + det_word
                            det_tokens.append(det_word)
                        else:
                            det_tokens.append(det_word)

        # calculate temperature with length : temp = 1 - 0.02 * length
        # temperature = [1 - sharpening * i if i > 1 else i for _, i in enumerate(entity_length)]

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        special_tokens_count = tokenizer.num_special_tokens_to_add()
        ## truncating tokens with max_seq_length
        # if len(tokens) > max_seq_length - special_tokens_count:
        #     tokens = tokens[: (max_seq_length - special_tokens_count)]
        #     label_ids = label_ids[: (max_seq_length - special_tokens_count)]
        #     det_tokens = det_tokens[: (max_seq_length - special_tokens_count)]

        # for sliding window tokens - update 23.11.13
        for i in range(0, (len(tokens) // max_seq_length) + 1):
            if i == 0:
                window_tokens = tokens[i*max_seq_length:(i+1)*max_seq_length-special_tokens_count]
                window_label_ids = label_ids[i*max_seq_length:(i+1)*max_seq_length-special_tokens_count]
                window_det_tokens = det_tokens[i*max_seq_length:(i+1)*max_seq_length-special_tokens_count]
            elif i >= 1:
                window_tokens = tokens[i*max_seq_length-special_tokens_count:(i+1)*max_seq_length-special_tokens_count]
                window_label_ids = label_ids[i*max_seq_length-special_tokens_count:(i+1)*max_seq_length-special_tokens_count]
                window_det_tokens = det_tokens[i*max_seq_length-special_tokens_count:(i+1)*max_seq_length-special_tokens_count]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            window_tokens += [sep_token]
            window_label_ids += [pad_token_label_id]
            window_det_tokens += [sep_token]

            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                window_tokens += [sep_token]
                window_label_ids += [pad_token_label_id]
                window_det_tokens += [sep_token]

            # make entity type label index for multiner
            entity_type_ids = [int(example.entity_labels[0])] * len(window_tokens)
            segment_ids = [sequence_a_segment_id] * len(window_tokens)
            if cls_token_at_end:
                window_tokens += [cls_token]
                window_label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
                entity_type_ids += [int(example.entity_labels[0])]
                window_det_tokens += [cls_token]
            else:
                window_tokens = [cls_token] + window_tokens
                window_label_ids = [pad_token_label_id] + window_label_ids
                segment_ids = [cls_token_segment_id] + segment_ids
                entity_type_ids = [int(example.entity_labels[0])] + entity_type_ids
                window_det_tokens = [cls_token] + window_det_tokens

            input_ids = tokenizer.convert_tokens_to_ids(window_tokens)
            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                window_label_ids = ([pad_token_label_id] * padding_length) + window_label_ids
                entity_type_ids = ([int(example.entity_labels[0])] * padding_length) + entity_type_ids
                window_tokens = (["**NULL**"] * padding_length) + window_tokens
                window_det_tokens = (["**NULL**"] * padding_length) + window_det_tokens
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                window_label_ids += [pad_token_label_id] * padding_length
                entity_type_ids += [int(example.entity_labels[0])] * padding_length
                window_tokens += ["**NULL**"] * padding_length
                window_det_tokens += ["**NULL**"] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(window_label_ids) == max_seq_length
            assert len(entity_type_ids) == max_seq_length
            assert len(window_tokens) == max_seq_length

            if ex_index < 1:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in window_tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in window_label_ids]))
                logger.info("entity_type_ids: %s", " ".join([str(x) for x in entity_type_ids]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None
            
            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, \
                    label_ids=window_label_ids, entity_type_ids=entity_type_ids, \
                )
            )
            write_tokens(window_tokens, window_det_tokens, 'test', base_name)

    return features

def write_tokens(tokens, det_tokens, mode, base_name):
    if mode == "test":
        tmp_path = os.path.join('multi_ner', 'tmp')
        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        path = os.path.join("multi_ner", "tmp",
                            "token_{}_{}.txt".format(mode, base_name))
        with open(path, 'a') as wf:
            for token in tokens:
                if token != "**NULL**":
                    wf.write(token + '\n')

        det_path = os.path.join("multi_ner", "tmp",
                            "det_token_{}_{}.txt".format(mode, base_name))
        with open(det_path, 'a') as wf:
            for token in det_tokens:
                if token != "**NULL**":
                    wf.write(token + '\n')

class NerProcessor(DataProcessor):
    def get_test_examples(self, data_dir):
        data = list()
        pmids = list()
        with open(data_dir, 'r') as in_:
            for line in in_:
                line = line.strip()
                tmp = json.loads(line)
                tmp['title'] = preprocess(tmp['title'])
                tmp['abstract'] = preprocess(tmp['abstract'])
                data.append(tmp)
                pmids.append(tmp["pmid"])

        json_file = input_form(json_to_sent(data))

        return \
            self._create_example(self._read_data(json_file, pmids), "test"), \
            json_file, data

    def get_test_dict_list(self, dict_list):
        pmids = list()
        for d in dict_list:
            pmids.append(d["pmid"])
            
        json_file = input_form(json_to_sent(dict_list))

        return \
            self._create_example(self._read_data(json_file, pmids), "test"), \
            json_file

    def get_labels(self):
        return ["B", "I", "O"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i,line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = line[1]
            label = line[0]
            entity_labels = line[2]
            examples.append(InputExample(guid=guid, words=text, labels=label, entity_labels=entity_labels))

        return examples


class MTNER:
    def __init__(self, params):
        # See all possible arguments in src/transformers/training_args.py
        # or by passing the --help flag to this script.
        # We now keep distinct sets of args, for a cleaner separation of concerns.

        init_start_t = time.time()

        # Set ner processor
        self.processor = NerProcessor()
        
        # Setup parsing
        self.params = params
        self.prediction_loss_only = False

        # Set seed
        set_seed(self.params.seed)
        
        # Prepare Labels
        self.labels = self.processor.get_labels()
        self.id2label: Dict[int, str] = {i: label for i, label in enumerate(self.labels)}
        self.label2id = {label:i for i, label in enumerate(self.labels)}
        self.num_labels = len(self.labels)

        self.config = AutoConfig.from_pretrained(
            self.params.model_name_or_path,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.params.model_name_or_path,
        )
        self.model = RoBERTaMultiNER2.from_pretrained(
            self.params.model_name_or_path,
            num_labels=self.num_labels,
            config=self.config,
        )
        if not self.params.no_cuda:
            self.model = self.model.cuda()
        self.entity_types = ['disease', 'drug', 'gene', 'species', 'cell_line', 'DNA', 'RNA', 'cell_type']
        self.estimator_dict = {}
        for etype in self.entity_types:
            self.estimator_dict[etype] = {}
            self.estimator_dict[etype]['prediction'] = []
            self.estimator_dict[etype]['log_probs'] = []

        self.counter = 0
        self.pad_token_label_id:int = nn.CrossEntropyLoss().ignore_index
        init_end_t = time.time()
        print('MTNER init_t {:.3f} sec.'.format(init_end_t - init_start_t))

    @Profile(__name__)
    def recognize(self, input_dl, base_name, indent=None):
        if type(input_dl) is str:
            predict_examples, self.json_dict, self.data_list = \
                self.processor.get_test_examples(input_dl)
        elif type(input_dl) is list:
            predict_examples, self.json_dict = \
                self.processor.get_test_dict_list(input_dl)
            self.data_list = input_dl
        else:
            raise ValueError('Wrong type')

        token_path = os.path.join("multi_ner", "tmp",        
                                  "token_test_{}.txt".format(base_name))
        det_token_path = os.path.join("multi_ner", "tmp",
                                  "det_token_test_{}.txt".format(base_name))

        if os.path.exists(token_path):
            os.remove(token_path)
        if os.path.exists(det_token_path):
            os.remove(det_token_path)

        predict_example_list = (NerDataset(predict_examples, self.labels,\
                                self.tokenizer, self.config, self.params, base_name))
        
        tokens, tot_tokens = list(), list()

        """
        Aggregate label results with detokenized tokens

        words: <s> Auto phagy main tain s tumour growth ... </s>
        label:  O   O     O     O    O  O    B      O   ...   O

        detok_words: <s> Authophagy maintains tumour growth ... </s>
        detok_label:  O       O         O        B      O   ... </s>
        """
        
        with open(det_token_path, 'r') as reader:
            for line_idx, line in enumerate(reader):
                tok = line.strip()
                tot_tokens.append(tok)
                
                if tok == '[CLS]' or tok == '<s>':
                    tmp_toks = [tok]
                elif tok == '[SEP]' or tok == '</s>':
                    tmp_toks.append(tok)
                    tokens.append(tmp_toks)
                else:
                    tmp_toks.append(tok)

        self.predict_dict, self.prob_dict = dict(), dict()
        threads, self.out_tag_dict = list(), dict()

        all_type = self._predict(predict_example_list)
        # disease, drug, gene, spec, cell_line, dna, rna, cell_type
        for etype_idx, etype in enumerate(self.entity_types):
            
            predictions, label_ids = all_type[etype_idx] # batch, seq, labels
            preds_array = self.align_predictions(predictions) # batch, seq

            self.out_tag_dict[etype] = (False, None)
            self.recognize_etype(etype, tokens, tot_tokens, predictions, preds_array)

        for etype in self.entity_types:
            if self.out_tag_dict[etype][0]:
                if type(input_dl) is str:
                    print(os.path.split(input_dl)[1],
                          'Found an error:', self.out_tag_dict[etype][1])
                else:
                    print('Found an error:', self.out_tag_dict[etype][1])
                if os.path.exists(token_path):
                    os.remove(token_path)
                return None

        # get probability of all mentions
        data_list = get_prob(self.data_list, self.json_dict, self.predict_dict,
                                  self.prob_dict, entity_types=self.entity_types)

        if type(input_dl) is str:
            output_path = os.path.join('result/', os.path.splitext(
                os.path.basename(input_dl))[0] + '_NER_{}.json'.format(base_name))
            print('pred', output_path)

            with open(output_path, 'w') as resultf:
                for paper in data_list:
                    paper['ner_model'] = "MULTI-TASK NER v.20210707"
                    resultf.write(
                        json.dumps(paper, sort_keys=True, indent=indent) + '\n'
                    )
        # delete temp files
        if os.path.exists(token_path):
            os.remove(token_path)
        if os.path.exists(det_token_path):
            os.remove(det_token_path)

        return data_list

    @Profile(__name__)
    def recognize_etype(self, etype, tokens, tot_tokens, predictions, preds_array):
        result = []
        
        for one_batch in range(predictions.shape[0]):
            result.append({'prediction':preds_array[one_batch],
                           'log_probs':predictions[one_batch]})

        predicts = list()
        logits = list()
        
        for pidx, prediction in enumerate(result):
            slen = len(tokens[pidx])
            for p in prediction['prediction'][:slen]:
                predicts.append(self.id2label[p])
            for l in prediction['log_probs'][:slen]:
                logits.append(l)

        de_toks, de_labels, de_logits = detokenize(tot_tokens, predicts, logits)

        self.predict_dict[etype] = dict()
        self.prob_dict[etype] = dict()
        piv = 0
        for data in self.data_list:
            pmid = data['pmid']
            self.predict_dict[etype][pmid] = list()
            self.prob_dict[etype][pmid] = list()

            sent_lens = list()
            for sent in self.json_dict[pmid]['words']:
                sent_lens.append(len(sent))
            sent_idx = 0
            de_i = 0
            overlen = False
            while True:
                if overlen:
                    
                    try:
                        self.predict_dict[etype][pmid][-1].extend(
                            de_labels[piv + de_i])
                    except Exception as e:
                        self.out_tag_dict[etype] = (True, e)
                        break
                    self.prob_dict[etype][pmid][-1].extend(de_logits[piv + de_i])
                    de_i += 1
                    if len(self.predict_dict[etype][pmid][-1]) == len(
                            self.json_dict[pmid]['words'][
                                len(self.predict_dict[etype][pmid]) - 1]):
                        sent_idx += 1
                        overlen = False

                else:
                    self.predict_dict[etype][pmid].append(de_labels[piv + de_i])
                    self.prob_dict[etype][pmid].append(de_logits[piv + de_i])
                    de_i += 1
                    if len(self.predict_dict[etype][pmid][-1]) == len(
                            self.json_dict[pmid]['words'][
                                len(self.predict_dict[etype][pmid]) - 1]):
                        sent_idx += 1
                        overlen = False
                    else:
                        overlen = True

                if sent_idx == len(self.json_dict[pmid]['words']):
                    piv += de_i
                    break

            if self.out_tag_dict[etype][0]:
                break

    def _predict(self, test_dataset:Dataset):
        sampler = SequentialSampler(test_dataset)
        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=32, # you can adjust evaluation batch size, we prefer using 32
            collate_fn=default_data_collator,
            drop_last=False,
        )
        return self._prediction_loop(data_loader, description="Prediction")

    def _prediction_loop(
        self, dataloader: DataLoader, description: str, prediction_loss_only: Optional[bool] = None
    ) -> PredictionOutput:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.

        Works both with or without labels.
        """
        
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else self.prediction_loss_only

        model = self.model
        
        eval_losses: List[float] = []
        dise_preds: torch.Tensor = None
        chem_preds: torch.Tensor = None
        gene_preds: torch.Tensor = None
        spec_preds: torch.Tensor = None
        cl_preds: torch.Tensor = None
        dna_preds: torch.Tensor = None
        rna_preds: torch.Tensor = None
        ct_preds: torch.Tensor = None
        label_ids: torch.Tensor = None
        model.eval()

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(inputs.get(k) is not None for k in ["labels", "lm_labels", "masked_lm_labels"])

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.model.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            if not prediction_loss_only:
                (dise_logits, chem_logits, gene_logits, spec_logits, cl_logits, dna_logits, rna_logits, _, ct_logits) = logits
                if dise_preds is None and chem_preds is None and gene_preds is None and spec_preds is None and cl_preds is None and dna_preds is None and rna_preds is None and ct_preds is None:
                    dise_preds = dise_logits.detach()
                    chem_preds = chem_logits.detach()
                    gene_preds = gene_logits.detach()
                    spec_preds = spec_logits.detach()
                    cl_preds = cl_logits.detach()
                    dna_preds = dna_logits.detach()
                    rna_preds = rna_logits.detach()
                    ct_preds = ct_logits.detach()
                else:
                    dise_preds = torch.cat((dise_preds, dise_logits.detach()), dim=0)
                    chem_preds = torch.cat((chem_preds, chem_logits.detach()), dim=0)
                    gene_preds = torch.cat((gene_preds, gene_logits.detach()), dim=0)
                    spec_preds = torch.cat((spec_preds, spec_logits.detach()), dim=0)
                    cl_preds = torch.cat((cl_preds, cl_logits.detach()), dim=0)
                    dna_preds = torch.cat((dna_preds, dna_logits.detach()), dim=0)
                    rna_preds = torch.cat((rna_preds, rna_logits.detach()), dim=0)
                    ct_preds = torch.cat((ct_preds, ct_logits.detach()), dim=0)
                if inputs.get("labels") is not None:
                    if label_ids is None:
                        label_ids = inputs["labels"].detach()
                    else:
                        label_ids = torch.cat((label_ids, inputs["labels"].detach()), dim=0)

        # Finally, turn the aggregated tensors into numpy arrays.
        if dise_preds is not None and chem_preds is not None and gene_preds is not None and spec_preds is not None:
            dise_preds = dise_preds.cpu().numpy()
            chem_preds = chem_preds.cpu().numpy()
            gene_preds = gene_preds.cpu().numpy()
            spec_preds = spec_preds.cpu().numpy()
            cl_preds = cl_preds.cpu().numpy()
            dna_preds = dna_preds.cpu().numpy()
            rna_preds = rna_preds.cpu().numpy()
            ct_preds = ct_preds.cpu().numpy()
        if label_ids is not None:
            label_ids = label_ids.cpu().numpy()

        return_output = (PredictionOutput(predictions=dise_preds, label_ids=label_ids), \
                        PredictionOutput(predictions=chem_preds, label_ids=label_ids), \
                        PredictionOutput(predictions=gene_preds, label_ids=label_ids), \
                        PredictionOutput(predictions=spec_preds, label_ids=label_ids), \
                        PredictionOutput(predictions=cl_preds, label_ids=label_ids), \
                        PredictionOutput(predictions=dna_preds, label_ids=label_ids), \
                        PredictionOutput(predictions=rna_preds, label_ids=label_ids), \
                        PredictionOutput(predictions=ct_preds, label_ids=label_ids))
                        
        return return_output

    def align_predictions(self, predictions: np.ndarray) -> List[int]:
        preds = np.argmax(predictions, axis=2)
        batch_size, seq_len = preds.shape

        preds_list = [[] for _ in range(batch_size)]
        
        for i in range(batch_size):
            for j in range(seq_len):
                preds_list[i].append(preds[i][j])

        return np.array(preds_list)

def main():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model_name_or_path', default='dmis-lab/bern2-ner')
    argparser.add_argument('--max_seq_length', type=int, help='The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.',
                            default=128)
    argparser.add_argument('--seed', type=int, help='random seed for initialization',
                            default=1)
    args = argparser.parse_args()

    mtner = MTNER(args)

if __name__ == "__main__":
    main()
