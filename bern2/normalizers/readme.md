# Neural normalizer

## How to index dictionary embeddings
```
# index disease
CUDA_VISIBLE_DEVICES=0 python neural_normalizer.py \
    --model_name_or_path dmis-lab/biosyn-sapbert-bc5cdr-disease \
    --dictionary_path ../resources/normalization/dictionary/dict_Disease_20210630.txt \
    --cache_dir ../resources/normalization/normalizers/neural_norm_caches

# index chemical
CUDA_VISIBLE_DEVICES=0 python neural_normalizer.py \
    --model_name_or_path dmis-lab/biosyn-sapbert-bc5cdr-chemical \
    --dictionary_path ../resources/normalization/dictionary/dict_ChemicalCompound_20210630.txt \
    --cache_dir ../resources/normalization/normalizers/neural_norm_caches

# index gene
CUDA_VISIBLE_DEVICES=0 python neural_normalizer.py \
    --model_name_or_path dmis-lab/biosyn-sapbert-bc2gn  \
    --dictionary_path ../resources/normalization/dictionary/dict_Gene.txt \
    --cache_dir ../resources/normalization/normalizers/neural_norm_caches
```
