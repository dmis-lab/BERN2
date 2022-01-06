import spacy
import scispacy
import torch
import torch.nn.functional as F

import numpy as np

from transformers import (
    AutoConfig,
    logging
)

from spacy_transformers import TransformerModel
from spacy_transformers.span_getters import get_doc_spans
from spacy_transformers.util import *
from spacy_transformers.tests.test_model_wrapper import *

from spacy.util import filter_spans
from spacy.tokens import Doc, Span
from spacy.language import Language

from scispacy.linking import EntityLinker
from multi_ner.modeling import RoBERTaMultiNER2

logging.set_verbosity_warning()
nlp = spacy.blank("en")

class spacy_pipeline():
    def __init__(self,):
        # load config and transformer model with bern2 parameters
        name='dmis-lab/bern2-ner'
        global model
        model = TransformerModel(name,
                                get_spans=get_doc_spans,
                                tokenizer_config={'use_fast': True})
        model.initialize()

        config = AutoConfig.from_pretrained(name)
        
        global ner_model
        ner_model = RoBERTaMultiNER2.from_pretrained(name,
            from_tf=False,
            num_labels=3,
            config=config
        )

    @Language.component("bern2_transformer")
    def bern2_transformer(doc):
        pred = model.predict([doc])
        return pred

    @Language.component("bern2_ner")
    def bern2_ner(pred):
        def _make_entity_list(bio_logit, label_map):
            a_list = torch.argmax(bio_logit.view(-1, 3), axis=-1).tolist()
            _entity = [label_map[i] for i in a_list]
            return _entity

        def _make_label_text(convert_dict, output_dict):
            for key, val in convert_dict.items():
                if key in ['tokens', 'subwords']:
                    continue

                entity_str = ""
                entity_str_list = []
                start_idx = -100
                idx_cnt = 0

                for entity_idx, entity_label in enumerate(val):
                    if output_dict['subwords'][entity_idx].startswith('Ġ'):
                        output_dict['subwords'][entity_idx] = output_dict['subwords'][entity_idx].replace('Ġ', ' ')

                    if entity_label == 'B':
                        if entity_idx == start_idx + 1:
                            entity_str_list.append((entity_str.strip(), start_idx))
                            start_idx = entity_idx
                            entity_str = ""
                        else:
                            start_idx = entity_idx
                            idx_cnt += 1
                        entity_str = output_dict['subwords'][entity_idx]
                    elif entity_label == 'I' and entity_idx == start_idx + 1:
                        start_idx = entity_idx                    
                        entity_str += output_dict['subwords'][entity_idx]
                        idx_cnt += 1
                    elif entity_label == 'O' and entity_idx == start_idx + 1:
                        entity_str_list.append((entity_str.strip(), start_idx-idx_cnt+1))
                        start_idx = -100
                        idx_cnt = 0
                        entity_str = ""

                output_dict[key] = entity_str_list

            return output_dict

        def _make_output_as_doc(final_output):
            assert len(final_output) == 1
            out_doc = Doc(nlp.vocab, words=final_output[0]['subwords'][1:-1], spaces=[True] * len(final_output[0]['subwords'][1:-1]))
            ents = []

            for key, val in final_output[0].items():
                if key in ['tokens', 'subwords']:
                    continue

                if val:
                    for val_inst in val:
                        ents.append(Span(out_doc, val_inst[1]-1, val_inst[1] + len(val_inst[0].split())-1, key))

            # resolve overlap entities
            filtered = filter_spans(ents)
            out_doc.ents = list(out_doc.ents) + filtered
            ents = []
            return out_doc

        label_map = {0:"B", 1:"I", 2: "O"}

        tensors = pred.tensors
        dise_logit = ner_model.dise_classifier(F.relu(ner_model.dise_classifier_2(tensors[0])))
        chem_logit = ner_model.chem_classifier(F.relu(ner_model.chem_classifier_2(tensors[0])))
        gene_logit = ner_model.gene_classifier(F.relu(ner_model.gene_classifier_2(tensors[0])))
        spec_logit = ner_model.spec_classifier(F.relu(ner_model.spec_classifier_2(tensors[0])))
        cellline_logit = ner_model.cellline_classifier(F.relu(ner_model.cellline_classifier_2(tensors[0])))
        dna_logit = ner_model.dna_classifier(F.relu(ner_model.dna_classifier_2(tensors[0])))
        rna_logit = ner_model.rna_classifier(F.relu(ner_model.rna_classifier_2(tensors[0])))
        celltype_logit = ner_model.celltype_classifier(F.relu(ner_model.celltype_classifier_2(tensors[0])))
        
        final_output = []
        for inst_idx, subwords in enumerate(pred.wordpieces.strings):
            merged_subword = ""
            for sub_idx, subword in enumerate(subwords):
                if subword.startswith('Ġ'):
                    subword = subword.replace('Ġ', ' ')
                    merged_subword += subword
                else:
                    if sub_idx == len(subwords)-2:
                        merged_subword += ' ' + subword
                    else:
                        merged_subword += '' + subword
                    
            merged_subword = merged_subword.strip()[3:-4]
        
            dise_entity = _make_entity_list(dise_logit, label_map)
            chem_entity = _make_entity_list(chem_logit, label_map)
            gene_entity = _make_entity_list(gene_logit, label_map)
            spec_entity = _make_entity_list(spec_logit, label_map)
            cellline_entity = _make_entity_list(cellline_logit, label_map)
            dna_entity = _make_entity_list(dna_logit, label_map)
            rna_entity = _make_entity_list(rna_logit, label_map)
            celltype_entity = _make_entity_list(celltype_logit, label_map)

            convert_dict = {'Diseases': dise_entity,
                        'Drugs/Chemicals': chem_entity,
                        'Genes/Proteins': gene_entity,
                        'Species': spec_entity,
                        'DNA': dna_entity,
                        'RNA': rna_entity,
                        'Cell Line': cellline_entity,
                        'Cell Type': celltype_entity,
                        'tokens':merged_subword.split(),
                        'subwords': pred.wordpieces.strings[inst_idx]}

            output_dict = {'tokens':merged_subword.split(),
                        'subwords':pred.wordpieces.strings[inst_idx]}

            output_dict = _make_label_text(convert_dict, output_dict)
            final_output.append(output_dict)
        
        out_doc = _make_output_as_doc(final_output)
        return out_doc


def main():
    # customizing bern2 pipeline: add bern2 components on spacy pipeline
    # nlp.add_pipe("bern2_transformer")
    # nlp.add_pipe("bern2_ner")
    # nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
    
    spacy_pipeline()
    nlp = spacy.load("/home/minbyul/bern2_spacy")
    doc = nlp("Autophagy maintains tumour growth through circulating arginine. Deletion of essential autophagy genes impairs the metabolism.")
    print (doc.ents)
    
if __name__ == "__main__":
    main()

