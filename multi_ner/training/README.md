# BERN2

## Train and Evaluate of BERN2 NER model

### 1. Datasets
You first need to download the in-domain evaluation datasets.
* [Named Entity Recognition](http://nlp.dmis.korea.edu/projects/bern2-sung-et-al-2022/NERdata.tar.gz)

```bash
tar -zxvf NERdata.tar.gz
```


### 2. Fine-tuning an NER model
```bash
wget https://dl.fbaipublicfiles.com/biolm/RoBERTa-large-PM-M3-Voc-hf.tar.gz
tar -zxvf RoBERTa-large-PM-M3-Voc-hf.tar.gz

export DATA_LIST=NCBI-disease+BC4CHEMD+BC2GM+linnaeus+JNLPBA-dna+JNLPBA-rna+JNLPBA-ct+JNLPBA-cl
export EVAL_DATA_LIST=NCBI-disease+BC4CHEMD+BC2GM+linnaeus+JNLPBA-dna+JNLPBA-rna+JNLPBA-ct+JNLPBA-cl
export OUTPUT_DIR=./finetuned_model

export ENTITY=NCBI-disease # BC4CHEMD, BC2GM etc.
export NUM_EPOCHS=50
export BATCH_SIZE=32
export SAVE_STEPS=-1
export LOG_STEPS=1000
export SEED=1
export LR=3e-5
export WARMUP=0

python run_ner.py
    --model_name_or_path RoBERTa-large-PM-M3-Voc-hf
    --data_dir NERdata/
    --labels NERdata/$ENTITY/labels.txt
    --output_dir $OUTPUT_DIR
    --data_list $DATA_LIST
    --eval_data_list $EVAL_DATA_LIST
    --num_train_epochs $NUM_EPOCHS
    --max_seq_length 128
    --warmup_steps $WARMUP
    --learning_rate $LR
    --per_device_train_batch_size $BATCH_SIZE
    --per_device_eval_batch_size $BATCH_SIZE
    --seed $SEED
    --logging_steps $LOG_STEPS
    --save_steps $SAVE_STEPS
    --do_train
    --do_eval
    --do_predict
    --evaluate_during_training
    --overwrite_output_dir
```

### 3. Evaluating the fine-tuned NER model (or BERN2 NER model)
```bash
export MODEL_NAME=./finetuned_model # or 'dmis-lab/bern2-ner'
export OUTPUT_DIR=./output # Save an output file for evaluation results
export ENTITY=NCBI-disease
export BATCH_SIZE=32
export SEED=1

python run_eval.py 
    --model_name_or_path $MODEL_NAME
    --data_dir NERdata/
    --labels NERdata/$ENTITY/labels.txt
    --output_dir $OUTPUT_DIR
    --eval_data_type $ENTITY
    --eval_data_list $ENTITY
    --max_seq_length 128
    --per_device_eval_batch_size $BATCH_SIZE
    --seed $SEED
    --do_eval
    --do_predict
```

