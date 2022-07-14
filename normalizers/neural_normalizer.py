import torch
import numpy as np
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    default_data_collator
)
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import re
from string import punctuation
import argparse
import pickle
import os
import time
import faiss

class NamesDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, no_cuda=False):
        self.encodings = encodings
        self.no_cuda=no_cuda

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]).cuda() for key, val in self.encodings.items()} if not self.no_cuda else \
                {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)

# TODO! inherit Normalizer class
class NeuralNormalizer(object):
    # def __init__(self, model_name_or_path, dictionary_path):
    def __init__(self, model_name_or_path, cache_path=None, no_cuda=False):
        self.max_length = 25
        self.batch_size = 1024
        self.k = 1 # top 1

        self.no_cuda = no_cuda
        self.device = "cuda:0" if torch.cuda.is_available() and not self.no_cuda else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path).to(self.device)
        # self.model = AutoModel.from_pretrained(model_name_or_path).cuda()
        
        # for basic normalization
        self.rmv_puncts_regex = re.compile(r'[\s{}]+'.format(re.escape(punctuation)))

        if cache_path:
            self.load_cache(cache_path)

    def load_dictionary(self, dictionary_path=''):
        # load dictionary
        self.dictionary = []
        with open(dictionary_path) as f:
            for line in f:
                line = line.strip()
                cui, names = line.split("||")
                for name in names.split("|"):
                    # important! load only within max_length
                    if len(self.tokenizer.tokenize(self._basic_normalize(name))) <= self.max_length:
                        self.dictionary.append((cui,name))

        # generate dictionary embeddings
        self.dict_embeds =  self._embed_dictionary()

    def normalize(self, names):
        if len(names) == 0:
            return names
        start_time = time.time()
        names = [self._basic_normalize(i) for i in names]

        # encode names
        self.model.eval()
        name_embeds = []

        # tokenize inputs
        name_encodings = self.tokenizer(names, padding="max_length", max_length=self.max_length, truncation=True)
        name_dataset = NamesDataset(name_encodings, no_cuda=self.no_cuda)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=self.batch_size)
        
        start_time = time.time()

        with torch.no_grad():
            for batch in name_dataloader:
                batch_name_embeds = self.model(**batch)
                batch_name_embeds = batch_name_embeds[0][:,0].cpu().detach().numpy() # [CLS] representations
                name_embeds.append(batch_name_embeds)
        
        name_embeds = np.concatenate(name_embeds, axis=0)

        start_time = time.time()

        distances, indices = self.dict_embeds.search(name_embeds, self.k)
        start_time = time.time()
        top1s = indices[:,0]

        assert (len(top1s) == len(names))

        outputs = [self.dictionary[top1] if top1 != -1 else (-1, -1) for top1 in top1s]
        return outputs
    
    def _basic_normalize(self, input):
        output = input
        output = output.lower()
        output = re.sub(self.rmv_puncts_regex, ' ', output)
        output = ' '.join(output.split())
        return output

    def _embed_dictionary(self, show_progress=True):
        """
        Embedding dictionary into entity representations

        Parameters
        ----------
        Returns
        -------
        dense_embeds : list
            A list of dense embeddings
        """

        self.model.eval()
        
        dict_embeds = []

        # extract names from dictionary
        cuis = [d[0] for d in self.dictionary]
        names = [self._basic_normalize(d[1]) for d in self.dictionary]

        # tokenize names
        name_encodings = self.tokenizer(names, padding="max_length", max_length=self.max_length, truncation=True)
        name_dataset = NamesDataset(name_encodings, no_cuda=self.no_cuda)
        name_dataloader = DataLoader(name_dataset, shuffle=False, collate_fn=default_data_collator, batch_size=self.batch_size)

        with torch.no_grad():
            for batch in tqdm(name_dataloader):
                outputs = self.model(**batch)
                batch_dict_embeds = outputs[0][:,0].cpu().detach().numpy() # [CLS] representations
                dict_embeds.append(batch_dict_embeds)
        dict_embeds = np.concatenate(dict_embeds, axis=0)
        
        return dict_embeds

    # save dictionary embeddings into faiss IVFFlat
    def save_cache(self, cache_path):
        nlist = 2048 # number of clusters
        dimension=self.dict_embeds.shape[1]
        assert dimension == 768
        quantiser = faiss.IndexFlatIP(dimension)
        index = faiss.IndexIVFFlat(quantiser, dimension, nlist, faiss.METRIC_INNER_PRODUCT)
        assert not index.is_trained
        print("index training")
        index.train(self.dict_embeds)  # train on the database vectors
        print("index trained")
        index.add(self.dict_embeds)   # add the vectors and update the index
        print("index added")
        index.nprobe = 25
        assert index.is_trained
        faiss.write_index(index, cache_path + ".index")
        with open(cache_path, 'wb') as f:
            pickle.dump(self.dictionary,f)

    def load_cache(self, cache_path):
        self.dict_embeds = faiss.read_index(cache_path + ".index")
        with open(cache_path, 'rb') as f:
            self.dictionary = pickle.load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['indexing', 'predict'], default='indexing')
    parser.add_argument("--model_name_or_path", default='cambridgeltl/SapBERT-from-PubMedBERT-fulltext')
    parser.add_argument("--dictionary_path", default='../normalization/resources/dictionary/best_dict_Disease_20210630_tmp.txt')
    parser.add_argument("--cache_dir", default='../normalization/biosyn_cache')
    parser.add_argument('--no_cuda', action="store_true", help="avoid using CUDA when available")

    args = parser.parse_args()

    biosyn = NeuralNormalizer(
        model_name_or_path=args.model_name_or_path,
    )
    print("model load")
    if args.mode == 'indexing':
        biosyn.load_dictionary(dictionary_path=args.dictionary_path)
        print("dictionary load")
        # make sure the cache directiory exists
        os.makedirs(args.cache_dir, exist_ok=True) 

        # save cache
        cache_path = os.path.join(args.cache_dir, args.dictionary_path.split("/")[-1] + ".pk")
        biosyn.save_cache(cache_path=cache_path)
        print("dictionary cache")

    # elif args.mode == 'predict':
    #     # load cache
    #     cache_path = os.path.join(args.cache_dir, args.dictionary_path.split("/")[-1] + ".pk")
    #     biosyn.load_cache(cache_path)
    #     print("dictionary loaded")
        
    #     while True:
    #         query = input()
    #         result = biosyn.normalize(querys=query.split("|"))
    
    #         print(result)

if __name__ == "__main__":
    main()
