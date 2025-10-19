#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to download Qwen3-VL-30B-A3B-Thinking model and run inference using vLLM with tensor parallelism.
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

# Limit visible CUDA devices to 2 and 3
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
# Enable FlashInfer CUTLASS MoE optimization
os.environ['VLLM_USE_FLASHINFER_MOE_FP16'] = '1'


def download_model(model_name: str, local_dir: str) -> str:
    """
    Download model from Hugging Face Hub to local directory.
    
    Args:
        model_name: Name of the model on Hugging Face (e.g., "Qwen/Qwen3-VL-30B-A3B-Thinking")
        local_dir: Local directory to save the model
        
    Returns:
        Path to the downloaded model
    """
    local_dir = os.path.join(".models", model_name.replace("/", "-"))
    Path(local_dir).mkdir(parents=True, exist_ok=True)
    print(f"Downloading model {model_name}...")
    print(f"Saving to {local_dir}")
    model_path = snapshot_download(
        repo_id=model_name,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"Model downloaded successfully to {model_path}")
    return model_path


def prepare_inputs_for_vllm(messages, processor):
    """
    Prepare inputs for vLLM from messages using the processor.
    
    Args:
        messages: List of message dictionaries with role and content
        processor: AutoProcessor for the model
        
    Returns:
        Dictionary with prompt, multi_modal_data, and mm_processor_kwargs
    """
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ required
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    print(f"video_kwargs: {video_kwargs}")

    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }


def run_inference(model_path: str, tensor_parallel_size: int):
    """
    Run inference using vLLM with tensor parallelism.
    
    Args:
        model_path: Path to the model
        tensor_parallel_size: Number of GPUs to use for tensor parallelism
    """
    print(f"\nLoading processor from {model_path}...")
    processor = AutoProcessor.from_pretrained(model_path)
    
    # Example 1: Video input
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-VL/space_woaudio.mp4",
    #             },
    #             {"type": "text", "text": "这段视频有多长"},
    #         ],
    #     }
    # ]

    # Example 2: Image input
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://ofasys-multimodal-wlcb-3-toshanghai.oss-accelerate.aliyuncs.com/wpf272043/keepme/image/receipt.png",
                },
                {"type": "text", "text": "Read all the text in the image."},
            ],
        }
    ]

    # Example 3: Text-only input
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "text", "text": "Explain the theory of relativity in simple terms."},
    #         ],
    #     }
    # ]

    # Prepare inputs for vLLM
    inputs = [prepare_inputs_for_vllm(message, processor) for message in [messages]]

    print(f"\nInitializing vLLM with tensor_parallel_size={tensor_parallel_size}...")
    llm = LLM(
        model=model_path,
        mm_encoder_tp_mode="data",
        enable_expert_parallel=True,
        tensor_parallel_size=tensor_parallel_size,
        seed=0
    )

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=4096,
        presence_penalty=1.5,
    )

    # Print input prompts
    for i, input_ in enumerate(inputs):
        print()
        print('=' * 40)
        print(f"Inputs[{i}]: {input_['prompt']=!r}")
    print('\n' + '>' * 40)

    # Run inference
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    
    # Print results
    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        print()
        print('=' * 40)
        print(f"Generated text: {generated_text!r}")
    
    print("\nInference completed successfully!")


def main():
    """Main function to download model and run inference."""
    # Configuration
    # MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking"
    MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"
    # MODEL_NAME = "QuantTrio/Qwen3-VL-30B-A3B-Instruct-AWQ"
    MODEL_DIR = ".models"
    TENSOR_PARALLEL_SIZE = torch.cuda.device_count()
    
    # Get the full path to the model directory
    model_dir = os.path.join(os.getcwd(), MODEL_DIR, MODEL_NAME.replace("/", "-"))
    
    # Check if model already exists
    if os.path.exists(model_dir) and os.listdir(model_dir):
        print(f"Model directory {model_dir} already exists and is not empty.")
        print("Skipping download...")
        model_path = model_dir
    else:
        # Download the model
        model_path = download_model(MODEL_NAME, model_dir)
    
    # Run inference with tensor parallelism
    run_inference(model_path, tensor_parallel_size=TENSOR_PARALLEL_SIZE)


if __name__ == "__main__":
    main()

