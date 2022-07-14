#!/bin/bash

cd ..
mkdir logs

####################################
#####          NER             #####
####################################

# run neural NER
start python multi_ner/ner_server.py \
    --mtner_home multi_ner \
    --mtner_port 18894 \
    --no_cuda >> logs/nohup_multi_ner.out 2>&1 &

cd resources

# run gnormplus
cd GNormPlusJava
start java -Xmx16G -Xms16G -jar GNormPlusServer.main.jar 18895 >> ../../logs/nohup_gnormplus.out 2>&1 &
cd ..

# run tmVar
cd tmVarJava
start java -Xmx8G -Xms8G -jar tmVar2Server.main.jar 18896 >> ../../logs/nohup_tmvar.out 2>&1 &
cd ..

####################################
#####     Normalization        #####
####################################

# Disease (working dir: normalization/)
cd normalization
start java -Xmx16G -jar normalizers/disease/disease_normalizer_21.jar \
    "inputs/disease" \
    "outputs/disease" \
    "dictionary/dict_Disease_20210630.txt" \
    "normalizers/disease/resources" \
    9 \
    18892 \
    >> ../../logs/nohup_disease_normalize.out 2>&1 &

# Gene (working dir: normalization/normalizers/gene/, port:18888)
cd normalizers/gene
start java -Xmx20G -jar gnormplus-normalization_21.jar \
    18888 \
    >> ../../../../logs/nohup_gene_normalize.out 2>&1 &
cd ../../../..

####################################
#####       Run BERN2          #####
####################################
env "PATH=$PATH" start python -u server.py \
    --mtner_home ./multi_ner \
    --mtner_port 18894 \
    --gnormplus_home ./resources/GNormPlusJava \
    --gnormplus_port 18895 \
    --tmvar2_home ./resources/tmVarJava \
    --tmvar2_port 18896 \
    --gene_norm_port 18888 \
    --disease_norm_port 18892 \
    --no_cuda \
    --use_neural_normalizer \
    --port 8888 \
    >> logs/nohup_bern2.out 2>&1 &

tail -f logs/nohup_bern2.out 
