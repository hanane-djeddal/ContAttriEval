models=("google/t5_xxl_true_nli_mixture")
export HF_HOME=$WORK/.cache/huggingface
export WANDB_MODE=offline

export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


for model in "${models[@]}"; do
    # 32 1e-5
    # ***************** Set parameters here *****************
    dataset_version=attributionBench_hardpos_augmentedQwen30B_allerrors_contrastive #attributionBench_augmentedQwen30B_allerrors_contrastive_extendalltrain #attributionBench_augmentedQwen30B_allerrors_contrastive_extendalltrainshuffled #   
    template=base_c_e
    lr=5e-5 #3e-5  #1e-5 #5e-5 #
    cont_weight=1.0 #0.5 #0.1  #1e-5  0.5 #
    classif_weight=1.0
    num_train_epoches=4
    tau=1.0
    start_gpu_index=0
    master_port=11111
    per_device_train_batch_size=1
    gas=4
    nodes=8
    num_neg=5
    num_pos=5
    cche_dir=$WORK/.cache/huggingface
    # ***************** The followings are auto-calculated parameters *****************
    cuda_devices=$(seq -s ',' $start_gpu_index $(($start_gpu_index + $nodes - 1)))
    export CUDA_VISIBLE_DEVICES=$cuda_devices
    nodes=8
    bs=$((gas * nodes))
    eval_bs=1 #$((per_device_train_batch_size * 2)) #template-${template}
    setting=bs${bs}-lr${lr}-gas${gas}-contW${cont_weight}-classifW${classif_weight}-Ep${num_train_epoches}-nneg${num_neg}-npos${num_pos}-filtered-tau${tau}
    current_time=$(date +"%Y-%m-%d-%H:%M:%S")

    echo ${CUDA_VISIBLE_DEVICES}
    # make sure you want to do the deletion  rm -rf $OUTPUT_DIR
    # ************************************************************************************
    export OUTPUT_DIR=../models/attribution_models/T5-use_InbatchNegatives-${dataset_version}-${setting}
    #rm -rf $OUTPUT_DIR
    # ************************************************************************************

    export WANDB_NAME=${model}_${setting}_dataset_${dataset_version}_${current_time}

    # # train         --evaluation_strategy "no" \ --cache_dir ${cche_dir}  --generation_max_length 128 \ --generation_num_beams 1 \
    torchrun --nproc_per_node ${nodes} --master-port ${master_port} ../src/train/autoais_train_contv2.py \
        --model_name_or_path $model \
        --data_path AttributionBench \
        --template ${template} \
        --template_path ../src/prompts.json \
        --dataset_version ${dataset_version} \
        --output_dir $OUTPUT_DIR \
        --num_train_epochs $num_train_epoches \
        --per_device_train_batch_size $per_device_train_batch_size \
        --per_device_eval_batch_size ${eval_bs} \
        --gradient_accumulation_steps ${gas} \
        --contrastive_weight ${cont_weight} \
        --classification_weight ${classif_weight} \
        --num_positives ${num_pos}\
        --num_negatives ${num_neg}\
        --filter_error_types True\
        --contrastive_temperature ${tau}\
        --use_decoder_embedding True\
        --save_total_limit 1 \
        --eval_strategy "no"\
        --save_strategy "no" \
        --save_only_model True \
        --logging_steps 10 \
        --learning_rate $lr \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --predict_with_generate True \
        --lr_scheduler_type "cosine" \
        --bf16 True \
        --tf32 True \
        --report_to wandb \
        --fsdp 'full_shard auto_wrap' \
        --fsdp_transformer_layer_cls_to_wrap 'T5Block'\

    # inference
    python ../src/inference/run_inference.py \
        --method autoais \
        --data_path AttributionBench \
        --dataset_version ${dataset_version} \
        --template_path ../src/prompts.json \
        --model_name ${OUTPUT_DIR} \
        --bs 1 \
        --split test test_ood\
        --output_dir ../inference_results/${dataset_version} \
        --max_length 2048  \
        --max_new_tokens 6 \
        --template ${template}

    # zero-shot
    # python ../src/inference/run_inference.py \
    #     --method autoais \
    #     --data_path AttributionBench \
    #     --dataset_version ${dataset_version} \
    #     --template_path ../src/prompts.json \
    #     --model_name $model \
    #     --bs 4 \
    #     --split test \
    #     --output_dir ../inference_results/${dataset_version} \
    #     --max_length 2048  \
    #     --max_new_tokens 6 \
    #     --template ${template}
done