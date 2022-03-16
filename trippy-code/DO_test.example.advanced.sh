#!/bin/bash

# Parameters ------------------------------------------------------

#TASK="sim-m"
#DATA_DIR="data/simulated-dialogue/sim-M"
#TASK="sim-r"
#DATA_DIR="data/simulated-dialogue/sim-R"
# TASK="woz2"
# DATA_DIR="data/woz2"
# TASK="multiwoz21"
# DATA_DIR="data/MULTIWOZ2.1"

TASK="multiwoz21"
DATA_DIR="data/MULTIWOZ2.1"

# Project paths etc. ----------------------------------------------
export CUDA_VISIBLE_DEVICES=0

# OUT_DIR=results_advanced_mw_mlm
# bert_model_path="./pre_training/multiwoz_mlm_bert/"
# OUT_DIR=results_advanced_mlm_all
# bert_model_path="./pre_training/mlm_all_bert_5/"

# OUT_DIR=results_advanced_2_0_all_mw_mlm
# bert_model_path="./pre_training/pretrain_mw_task_mlm_bert/"
# OUT_DIR=results_advanced_2_0_span_pretrain_all_mw_mlm

OUT_DIR=results_advanced_paraphrase_multi_50_1
bert_model_path="bert-base-uncased"
# bert_model_path="./pre_training/span_mw_mlm"
# bert_model_path="./pre_training/span_pretrain_all_mw_mlm"

mkdir -p ${OUT_DIR}


# Main ------------------------------------------------------------

for step in test; do
    args_add=""
    if [ "$step" = "train" ]; then
	args_add="--do_train --predict_type=dummy"
    elif [ "$step" = "dev" ] || [ "$step" = "test" ]; then
	args_add="--do_eval --predict_type=${step}"
    fi

    python3 run_dst.py \
	    --task_name=${TASK} \
	    --data_dir=${DATA_DIR} \
	    --dataset_config=dataset_config/${TASK}.json \
	    --model_type="bert" \
	    --model_name_or_path=${bert_model_path} \
		--tokenizer_name="bert-base-uncased" \
	    --do_lower_case \
	    --learning_rate=1e-4 \
	    --num_train_epochs=10 \
	    --max_seq_length=180 \
	    --per_gpu_train_batch_size=24 \
	    --per_gpu_eval_batch_size=1 \
	    --output_dir=${OUT_DIR} \
		--cached_file="./cached_${step}_features_advanced" \
	    --save_epochs=2 \
	    --logging_steps=10 \
	    --warmup_proportion=0.1 \
	    --adam_epsilon=1e-6 \
	    --label_value_repetitions \
        --swap_utterances \
	    --append_history \
	    --use_history_labels \
	    --delexicalize_sys_utts \
	    --class_aux_feats_inform \
	    --class_aux_feats_ds \
	    ${args_add} \
	    2>&1 | tee ${OUT_DIR}/${step}.log
    
    if [ "$step" = "dev" ] || [ "$step" = "test" ]; then
    	python3 metric_bert_dst.py \
    		${TASK} \
		dataset_config/${TASK}.json \
    		"${OUT_DIR}/pred_res.${step}*json" \
    		2>&1 | tee ${OUT_DIR}/eval_pred_${step}.log
    fi
done


### --- Use this tag for evaluating all the checkpoints	 --eval_all_checkpoints \

# OUT_DIR=results_advanced_mw_mlm
# bert_model_path="./pre_training/multiwoz_mlm_bert/"

# OUT_DIR=results_advanced_mlm_all
# bert_model_path="./pre_training/mlm_all_bert_5/"

# OUT_DIR=results_advanced_all_mw_seq_mlm
# bert_model_path="./pre_training/pretrain_mw_task_mlm_seq_txt_bert/"

# OUT_DIR=results_advanced_mw_seq_mlm
# bert_model_path="./pre_training/mw_mlm_seq_txt_bert/"

# OUT_DIR=results_advanced_span_mw_mlm
# bert_model_path="./pre_training/span_mw_mlm/"

# OUT_DIR=results_advanced_span_mlm_all
# bert_model_path="./pre_training/span_mlm_all/"

# OUT_DIR=results_advanced_span_mw_seq_mlm
# bert_model_path="./pre_training/span_mw_seq_mlm/"


# OUT_DIR=results_advanced_span_pretrain_all_mw_seq_mlm
# bert_model_path="./pre_training/span_pretrain_all_mw_seq_mlm/"

# OUT_DIR=results_advanced_span_pretrain_all_mw_mlm
# bert_model_path="./pre_training/span_pretrain_all_mw_mlm/"

# OUT_DIR=results_advanced_2_0_mlm_all
# bert_model_path="./pre_training/mlm_all_bert_5"

# OUT_DIR=results_advanced_2_0_mw_mlm
# bert_model_path="./pre_training/multiwoz_mlm_bert"


# OUT_DIR=results_advanced_2_0_all_mw_mlm
# bert_model_path="./pre_training/pretrain_mw_task_mlm_bert"


# OUT_DIR=results_advanced_2_0_span_mlm_all
# bert_model_path="./pre_training/span_mlm_all"

# OUT_DIR=results_advanced_2_0_span_mw_mlm
# bert_model_path="./pre_training/span_mw_mlm"

# OUT_DIR=results_advanced_2_0_span_pretrain_all_mw_mlm
# bert_model_path="./pre_training/span_pretrain_all_mw_mlm"
