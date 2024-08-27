import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel
import wandb
from tqdm import tqdm
from natsort import natsorted
import gc

# Import necessary modules from your training script
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSDataset, RLDSBatchTransform
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder

def load_checkpoint(base_model_path, adapter_path, device):
    base_model = AutoModelForVision2Seq.from_pretrained(
        base_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    model = PeftModel.from_pretrained(base_model, adapter_path)
    return model

def evaluate(model, dataloader, action_tokenizer, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    total_l1_loss = 0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if i >= len(dataloader):
                break
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                labels=batch["labels"].to(device)
            )

            loss = outputs.loss
            total_loss += loss.item() * batch["input_ids"].size(0)

            action_logits = outputs.logits[:, model.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > action_tokenizer.action_token_begin_idx

            correct_preds = (action_preds == action_gt) & mask
            accuracy = correct_preds.sum().float() / mask.sum().float()
            total_accuracy += accuracy.item() * mask.sum().item()

            continuous_actions_pred = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_preds[mask].cpu().numpy())
            )
            continuous_actions_gt = torch.tensor(
                action_tokenizer.decode_token_ids_to_actions(action_gt[mask].cpu().numpy())
            )
            l1_loss = torch.nn.functional.l1_loss(continuous_actions_pred, continuous_actions_gt)
            total_l1_loss += l1_loss.item() * mask.sum().item()

            total_samples += mask.sum().item()

    avg_loss = total_loss / total_samples
    avg_accuracy = total_accuracy / total_samples
    avg_l1_loss = total_l1_loss / total_samples

    return avg_loss, avg_accuracy, avg_l1_loss

def main():
    # Configuration
    VLA_PATH = "openvla/openvla-7b"
    BASE_PATH = "/robo-srv-004-storage-001/home/mkotynia/openvla"
    EXP = "openvla-7b+robotec_o3de_panda_dataset_200_train_episodes+b16+lr-0.0005+lr-decay+lora-r128+dropout-0.0"
    ADAPTERS_DIR =  os.path.join(BASE_PATH, "adapter_weights", EXP), 
    STEPS_RUN_DIR = os.path.join(BASE_PATH, "checkpoints", EXP)
    val_dataset_name = "robotec_o3de_panda_dataset_200_train_episodes"
    val_data_root_dir = "/home/mkotynia/tensorflow_datasets"
    batch_size = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize wandb
    wandb.init(project="openvla-evaluation", name=f"val+{EXP}")

    # Load processor and create action tokenizer
    processor = AutoProcessor.from_pretrained(VLA_PATH, trust_remote_code=True)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # Prepare validation dataset and dataloader
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder if "v01" not in BASE_PATH else VicunaV15ChatPromptBuilder,
    )
    val_dataset = RLDSDataset(
        val_data_root_dir,
        val_dataset_name,
        batch_transform,
        resize_resolution=(224, 224),  # Update this with the correct image size
        shuffle_buffer_size=10000,
        image_aug=False,
        train=False
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=0,
    )

    # Evaluate checkpoints
    for checkpoint_dir in natsorted(Path(ADAPTERS_DIR[0]).glob("step-*")):
        step = int(checkpoint_dir.name.split("-")[-1])
        print(f"Evaluating checkpoint at step {step}")

        model = load_checkpoint(VLA_PATH, checkpoint_dir, device)
        avg_loss, avg_accuracy, avg_l1_loss = evaluate(model, val_dataloader, action_tokenizer, device)

        # Log metrics to wandb
        wandb.log({
            "step": step,
            "val_loss": avg_loss,
            "val_accuracy": avg_accuracy,
            "val_l1_loss": avg_l1_loss,
        })

        print(f"Step {step}: Loss = {avg_loss:.4f}, Accuracy = {avg_accuracy:.4f}, L1 Loss = {avg_l1_loss:.4f}")
            # Free up GPU memory
        model.to('cpu')  # Move model to CPU
        del model  # Delete the model object
        torch.cuda.empty_cache()  # Clear CUDA cache
        gc.collect()  # Run garbage collector
    wandb.finish()

if __name__ == "__main__":
    main()