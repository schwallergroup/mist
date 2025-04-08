HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file configs/deepspeed_zero3.yaml \
    --main_process_port 29501 \
    --num_processes 1 \
    src/open_r1/run_r1_grpo.py \
    --config recipes/smi_permute.test.yaml \
    --model_name_or_path Qwen/Qwen2.5-0.5B \
    --output_dir output/grpo \
    --dataset_id_or_path /home/vu/Documents/open-r1/data/CRLLM-PubChem-compounds1M-simple.csv