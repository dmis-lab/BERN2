#!/bin/bash

cd ..
mkdir logs

####################################
#####          NER             #####
####################################

# run neural NER
nohup python multi_ner/ner_server.py \
    --mtner_home multi_ner \
    --mtner_port 18894 \
    --no_cuda >> logs/nohup_multi_ner.out 2>&1 &

cd resources

# run gnormplus
cd GNormPlusJava
nohup java -Xmx16G -Xms16G -jar GNormPlusServer.main.jar 18895 >> ../../logs/nohup_gnormplus.out 2>&1 &
cd ..

# run tmVar
cd tmVarJava
nohup java -Xmx8G -Xms8G -jar tmVar2Server.main.jar 18896 >> ../../logs/nohup_tmvar.out 2>&1 &
cd ..

####################################
#####     Normalization        #####
####################################

# Disease (working dir: normalization/)
cd normalization
nohup java -Xmx16G -jar normalizers/disease/disease_normalizer_21.jar \
    "inputs/disease" \
    "outputs/disease" \
    "dictionary/dict_Disease_20210630.txt" \
    "normalizers/disease/resources" \
    9 \
    18892 \
    >> ../../logs/nohup_disease_normalize.out 2>&1 &

# Gene (working dir: normalization/normalizers/gene/, port:18888)
cd normalizers/gene
nohup java -Xmx20G -jar gnormplus-normalization_21.jar \
    18888 \
    >> ../../../../logs/nohup_gene_normalize.out 2>&1 &
cd ../../../..

####################################
#####       Run BERN2          #####
####################################
env "PATH=$PATH" nohup python -u server.py \
    --mtner_home ./multi_ner \
    --mtner_port 18894 \
    --gnormplus_home ./resources/GNormPlusJava \
    --gnormplus_port 18895 \
    --tmvar2_home ./resources/tmVarJava \
    --tmvar2_port 18896 \
    --no_cuda \
    --gene_norm_port 18888 \
    --disease_norm_port 18892 \
    --use_neural_normalizer \
    --port 8888 \
    >> logs/nohup_bern2.out 2>&1 &

tail -f logs/nohup_bern2.out
