# coding=utf-8

import os
import pdb
import copy
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn import CrossEntropyLoss
from transformers import (
        BertConfig,
        BertModel,
        RobertaModel,
        BertForTokenClassification,
        BertTokenizer,
        RobertaConfig,
        RobertaForTokenClassification,
        RobertaTokenizer
)

class BERTMultiNER2(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(BERTMultiNER2, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.dise_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # disease
        self.chem_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # chemical
        self.gene_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # gene/protein
        self.spec_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # species
        self.cell_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # cell line

        self.dise_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.chem_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.gene_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.spec_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.cell_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, entity_type_ids=None):
        sequence_output = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            dise_sequence_output = F.relu(self.dise_classifier_2(sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(sequence_output)) # species logit value
            cell_sequence_output = F.relu(self.spec_classifier_2(sequence_output)) # cell line logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            cell_logits = self.cell_classifier(cell_sequence_output) # cell line logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cell_sequence_output
            logits = (dise_logits, chem_logits, gene_logits, spec_logits, cell_logits)
        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            dise_idx = copy.deepcopy(entity_type_ids)
            chem_idx = copy.deepcopy(entity_type_ids)
            gene_idx = copy.deepcopy(entity_type_ids)
            spec_idx = copy.deepcopy(entity_type_ids)
            cell_idx = copy.deepcopy(entity_type_ids)

            dise_idx[dise_idx != 1] = 0
            chem_idx[chem_idx != 2] = 0
            gene_idx[gene_idx != 3] = 0
            spec_idx[spec_idx != 4] = 0
            cell_idx[cell_idx != 5] = 0

            dise_sequence_output = dise_idx.unsqueeze(-1) * sequence_output        
            chem_sequence_output = chem_idx.unsqueeze(-1) * sequence_output
            gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            spec_sequence_output = spec_idx.unsqueeze(-1) * sequence_output
            cell_sequence_output = cell_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu
            dise_sequence_output = F.relu(self.dise_classifier_2(dise_sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(chem_sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(gene_sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(spec_sequence_output)) # species logit value
            cell_sequence_output = F.relu(self.cell_classifier_2(cell_sequence_output)) # cell line logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            cell_logits = self.cell_classifier(cell_sequence_output) # cell line logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cell_sequence_output
            logits = dise_logits + chem_logits + gene_logits + spec_logits + cell_logits

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits

class RoBERTaMultiNER2(RobertaForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(RoBERTaMultiNER2, self).__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.dise_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # disease
        self.chem_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # chemical
        self.gene_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # gene/protein
        self.spec_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # species
        self.cellline_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # cell line
        self.dna_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # dna
        self.rna_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # rna
        self.protein_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # protein
        self.celltype_classifier = torch.nn.Linear(config.hidden_size, self.num_labels) # cell type

        self.dise_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.chem_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.gene_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.spec_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.cellline_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.dna_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.rna_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.protein_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.celltype_classifier_2 = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, entity_type_ids=None):
        sequence_output = self.roberta(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        if entity_type_ids[0][0].item() == 0:
            '''
            Raw text data with trained parameters
            '''
            dise_sequence_output = F.relu(self.dise_classifier_2(sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(sequence_output)) # species logit value
            cellline_sequence_output = F.relu(self.cellline_classifier_2(sequence_output)) # cell line logit value
            dna_sequence_output = F.relu(self.dna_classifier_2(sequence_output)) # dna logit value
            rna_sequence_output = F.relu(self.rna_classifier_2(sequence_output)) # rna logit value
            protein_sequence_output = F.relu(self.protein_classifier_2(sequence_output)) # protein logit value
            celltype_sequence_output = F.relu(self.celltype_classifier_2(sequence_output)) # cell type logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            cellline_logits = self.cellline_classifier(cellline_sequence_output) # cell line logit value
            dna_logits = self.dna_classifier(dna_sequence_output) # dna logit value
            rna_logits = self.rna_classifier(rna_sequence_output) # rna logit value
            protein_logits = self.protein_classifier(protein_sequence_output) # protein logit value
            celltype_logits = self.celltype_classifier(celltype_sequence_output) # cell type logit value


            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cellline_sequence_output + dna_sequence_output + rna_sequence_output + protein_sequence_output + celltype_sequence_output
            logits = (dise_logits, chem_logits, gene_logits, spec_logits, cellline_logits, dna_logits, rna_logits, protein_logits, celltype_logits)
        else:
            ''' 
            Train, Eval, Test with pre-defined entity type tags
            '''
            # make 1*1 conv to adopt entity type
            dise_idx = copy.deepcopy(entity_type_ids)
            chem_idx = copy.deepcopy(entity_type_ids)
            gene_idx = copy.deepcopy(entity_type_ids)
            spec_idx = copy.deepcopy(entity_type_ids)
            cellline_idx = copy.deepcopy(entity_type_ids)
            dna_idx = copy.deepcopy(entity_type_ids)
            rna_idx = copy.deepcopy(entity_type_ids)
            protein_idx = copy.deepcopy(entity_type_ids)
            celltype_idx = copy.deepcopy(entity_type_ids)

            dise_idx[dise_idx != 1] = 0
            chem_idx[chem_idx != 2] = 0
            gene_idx[gene_idx != 3] = 0
            spec_idx[spec_idx != 4] = 0
            cellline_idx[cellline_idx != 5] = 0
            dna_idx[dna_idx != 6] = 0
            rna_idx[rna_idx != 7] = 0
            protein_idx[protein_idx != 8] = 0
            celltype_idx[celltype_idx != 9] = 0

            dise_sequence_output = dise_idx.unsqueeze(-1) * sequence_output        
            chem_sequence_output = chem_idx.unsqueeze(-1) * sequence_output
            gene_sequence_output = gene_idx.unsqueeze(-1) * sequence_output
            spec_sequence_output = spec_idx.unsqueeze(-1) * sequence_output
            cellline_sequence_output = cellline_idx.unsqueeze(-1) * sequence_output
            dna_sequence_output = dna_idx.unsqueeze(-1) * sequence_output
            rna_sequence_output = rna_idx.unsqueeze(-1) * sequence_output
            protein_sequence_output = protein_idx.unsqueeze(-1) * sequence_output
            celltype_sequence_output = celltype_idx.unsqueeze(-1) * sequence_output

            # F.tanh or F.relu
            dise_sequence_output = F.relu(self.dise_classifier_2(dise_sequence_output)) # disease logit value
            chem_sequence_output = F.relu(self.chem_classifier_2(chem_sequence_output)) # chemical logit value
            gene_sequence_output = F.relu(self.gene_classifier_2(gene_sequence_output)) # gene/protein logit value
            spec_sequence_output = F.relu(self.spec_classifier_2(spec_sequence_output)) # species logit value
            cellline_sequence_output = F.relu(self.cellline_classifier_2(cellline_sequence_output)) # cell line logit value
            dna_sequence_output = F.relu(self.dna_classifier_2(dna_sequence_output)) # dna logit value
            rna_sequence_output = F.relu(self.rna_classifier_2(rna_sequence_output)) # rna logit value
            protein_sequence_output = F.relu(self.protein_classifier_2(protein_sequence_output)) # protein logit value
            celltype_sequence_output = F.relu(self.celltype_classifier_2(celltype_sequence_output)) # cell type logit value

            dise_logits = self.dise_classifier(dise_sequence_output) # disease logit value
            chem_logits = self.chem_classifier(chem_sequence_output) # chemical logit value
            gene_logits = self.gene_classifier(gene_sequence_output) # gene/protein logit value
            spec_logits = self.spec_classifier(spec_sequence_output) # species logit value
            cellline_logits = self.cellline_classifier(cellline_sequence_output) # cell line logit value
            dna_logits = self.dna_classifier(dna_sequence_output) # dna logit value
            rna_logits = self.rna_classifier(rna_sequence_output) # rna logit value
            protein_logits = self.protein_classifier(protein_sequence_output) # protein logit value
            celltype_logits = self.celltype_classifier(celltype_sequence_output) # cell type logit value

            # update logit and sequence_output
            sequence_output = dise_sequence_output + chem_sequence_output + gene_sequence_output + spec_sequence_output + cellline_sequence_output + dna_sequence_output + rna_sequence_output + protein_sequence_output + celltype_sequence_output
            logits = dise_logits + chem_logits + gene_logits + spec_logits + cellline_logits + dna_logits + rna_logits + protein_logits + celltype_logits

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                if entity_type_ids[0][0].item() == 0:
                    active_loss = attention_mask.view(-1) == 1
                    dise_logits, chem_logits, gene_logits, spec_logits, cellline_logits, dna_logits, rna_logits, protein_logits, celltype_logits = logits

                    active_dise_logits = dise_logits.view(-1, self.num_labels)
                    active_chem_logits = chem_logits.view(-1, self.num_labels)
                    active_gene_logits = gene_logits.view(-1, self.num_labels)
                    active_spec_logits = spec_logits.view(-1, self.num_labels)
                    active_cellline_logits = cellline_logits.view(-1, self.num_labels)
                    active_dna_logits = dna_logits.view(-1, self.num_labels)
                    active_rna_logits = rna_logits.view(-1, self.num_labels)
                    active_protein_logits = protein_logits.view(-1, self.num_labels)
                    active_celltype_logits = celltype_logits.view(-1, self.num_labels)
                    
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    dise_loss = loss_fct(active_dise_logits, active_labels)
                    chem_loss = loss_fct(active_chem_logits, active_labels)
                    gene_loss = loss_fct(active_gene_logits, active_labels)
                    spec_loss = loss_fct(active_spec_logits, active_labels)
                    cellline_loss = loss_fct(active_cellline_logits, active_labels)
                    dna_loss = loss_fct(active_dna_logits, active_labels)
                    rna_loss = loss_fct(active_rna_logits, active_labels)
                    protein_loss = loss_fct(active_protein_logits, active_labels)
                    celltype_loss = loss_fct(active_celltype_logits, active_labels)
                    loss = dise_loss + chem_loss + gene_loss + spec_loss + cellline_loss + dna_loss + rna_loss + protein_loss + celltype_loss

                    return ((loss,) + outputs)
                else:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = logits.view(-1, self.num_labels)
                    active_labels = torch.where(
                        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                    )
                    loss = loss_fct(active_logits, active_labels)
                    return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits


class NER(BertForTokenClassification):
    def __init__(self, config, num_labels=3):
        super(NER, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask, head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape
        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        outputs = (logits, sequence_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
                return ((loss,) + outputs)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                return loss
        else:
            return logits

