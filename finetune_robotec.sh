#!/bin/bash

torchrun --standalone --nnodes 1 --nproc-per-node 2 vla-scripts/finetune.py \
  --vla_path "openvla/openvla-7b" \
  --data_root_dir "/home/mkotynia/tensorflow_datasets" \
  --dataset_name robotec_o3de_panda_dataset_4_cameras \
  --run_root_dir "/robo-srv-004-storage-001/home/mkotynia/openvla/checkpoints" \
  --adapter_tmp_dir "/robo-srv-004-storage-001/home/mkotynia/openvla/adapter_weights" \
  --lora_rank 128 \
  --batch_size 1 \
  --grad_accumulation_steps 16 \
  --learning_rate 5e-4 \
  --image_aug False \
  --wandb_project openvla_test \
  --wandb_entity robotecai-ml \
  --save_steps 50 \
  --num_warmup_steps 1000 \
