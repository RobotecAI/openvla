from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
import torch
import os

VLA_PATH = "openvla/openvla-7b"

BASE_PATH = "/robo-srv-004-storage-001/home/mkotynia/openvla"
EXP = "openvla-7b+robotec_o3de_panda_dataset_4_cameras+b16+lr-0.0005+lr-decay+lora-r128+dropout-0.0"
STEP = "step-1200"
ADAPTER_DIR =  os.path.join(BASE_PATH, "adapter_weights", EXP, STEP), 
STEP_RUN_DIR = os.path.join(BASE_PATH, "checkpoints", EXP, STEP)

base_vla = AutoModelForVision2Seq.from_pretrained(
    VLA_PATH, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
)
merged_vla = PeftModel.from_pretrained(base_vla, ADAPTER_DIR[0])
merged_vla = merged_vla.merge_and_unload()
merged_vla.save_pretrained(STEP_RUN_DIR)

