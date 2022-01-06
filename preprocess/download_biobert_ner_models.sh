#!/bin/bash

cd ../biobert_ner

if [ -f "biobert_ner_models.zip" ]; then
    echo "Found biobert_ner_models.zip"
else
    echo "Not found biobert_ner_models.zip. Downloading.."
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1sSVEqvMBVLj1RJmlQDhRKyt_oe-wc5LK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1sSVEqvMBVLj1RJmlQDhRKyt_oe-wc5LK" -O biobert_ner_models.zip && rm -rf /tmp/cookies.txt
fi

unzip biobert_ner_models.zip
mkdir result
mkdir tmp
# rm biobert_ner_models.zip

echo "BioBERT NER models are downloaded"