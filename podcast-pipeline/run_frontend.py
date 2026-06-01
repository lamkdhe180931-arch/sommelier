# Sommelier
# Copyright (c) 2026-present NAVER Cloud Corp.
# MIT

import os
import sys
import glob
import subprocess
import multiprocessing
import json
import time
import argparse
import math
from datetime import datetime
import torch
import wandb
from tqdm import tqdm

# ====================================================
# [Configuration] Modify according to your environment
# ====================================================

# 1. Path to the backend Python script to execute
BACKEND_SCRIPT = "main_original_ASR_MoE.py"

# 2. Input data root folder (parent folder containing opus files)
INPUT_ROOT = "/your/audio/folder"

# 3. Log and progress tracking file
PROGRESS_LOG_FILE = "processed_folders_log.txt"

# 4. WandB configuration
WANDB_PROJECT = ""
WANDB_ENTITY = ""  # Change to your own entity (optional)
WANDB_RUN_NAME = f"pipeline-run-{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# 5. Fixed parameters to pass to the backend script (reflecting bash script settings)
# Define the required flags as a list.
BACKEND_ARGS = [
    "--vad",
    "--dia3",
    "--no-initprompt", # Reflects initprompt_flags=(--no-initprompt) from the bash script
    "--ASRMoE",
    "--demucs",
    "--whisperx_word_timestamps",
    "--no-qwen3omni",
    "--sepreformer",
    "--sortformer-param",
    "--sortformer-pad-onset", "-0.05",
    "--sortformer-pad-offset", "0.15",
    "--sortformer-soft-label-thres", "0.15",
    "--LLM", "case_0",
    "--seg_th", "0.11",
    "--min_cluster_size", "11",
    "--clust_th", "0.5",
    "--merge_gap", "2",
    "--overlap_threshold", "0.2",
    "--min_sepreformer_overlap", "1.0",
    "--min_sepreformer_segment", "1.0",
    "--asr_quality_guard",
    "--asr_micro_segment_seconds", "0.5",
    "--asr_short_segment_seconds", "1.0",
    "--asr_vi_agreement_threshold", "0.75",
    "--speaker-link-threshold", "0.6",
    "--opus_decode_workers", "20", # Adjust CPU threads allocated per worker
    "--ffmpeg_threads_per_decode", "1"
]

# ====================================================

def get_gpu_count():
    """Returns the number of available GPUs."""
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def find_subfolders_with_opus(root_dir):
    """
    Recursively scans all subdirectories under the root folder and returns
    a list of folders containing .opus files.
    Folders containing '_opus_cache' in their name are excluded from the search.
    """
    target_folders = set()
    print(f"📂 [Search] Scanning subdirectories in {root_dir}...")
    
    # Recursively traverse using os.walk
    for dirpath, dirs, filenames in os.walk(root_dir):
        # [Important] Prune the dirs list in-place to skip cache directories
        # Folders containing '_opus_cache' are removed from the traversal list
        dirs[:] = [d for d in dirs if "_opus_cache" not in d]

        for f in filenames:
            if f.endswith('.opus') or f.endswith('.ogg'):
                # Only add if the current folder (dirpath) is not a cache folder (double safety check)
                if "_opus_cache" not in dirpath:
                    target_folders.add(dirpath)
                break # This folder is confirmed, move to the next one

    # Sort and return
    return sorted(list(target_folders))

def load_processed_list():
    """Loads the list of already processed folders."""
    if not os.path.exists(PROGRESS_LOG_FILE):
        return set()
    with open(PROGRESS_LOG_FILE, 'r') as f:
        return set(line.strip() for line in f)

def append_to_processed_list(folder_path):
    """Records a completed folder to the log."""
    with open(PROGRESS_LOG_FILE, 'a') as f:
        f.write(f"{folder_path}\n")

def get_audio_duration_from_json(folder_path):
    """
    Finds the JSON file generated after backend processing and extracts the audio duration.
    Uses glob to locate the JSON since it depends on the backend's save path logic.
    """
    # Find the expected result JSON (folder_name.json or json within the folder)
    # According to the backend logic, it is created under the _final folder,
    # but since the exact location is hard to determine, it's better to find
    # the most recently created json or parse the info printed to stdout by the backend script.
    # For now, simply returns 0.0; can be enhanced later with backend log parsing.
    # *Note: For accurate duration logging, parsing the subprocess output is more precise
    # since the backend script is designed to print the duration.*
    return 0.0

def worker_process(gpu_id, session_id, folder_queue, result_queue, error_queue):
    """
    Individual worker process.
    """
    # Assign GPU to the current process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Store all caches under /mnt/ddn/kyudan/
    cache_base = "./.cache"
    os.environ["TORCH_HOME"] = f"{cache_base}/torch"
    os.environ["HF_HOME"] = f"{cache_base}/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_base}/huggingface/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{cache_base}/huggingface/datasets"
    os.environ["XDG_CACHE_HOME"] = cache_base
    os.environ["TRITON_CACHE_DIR"] = f"{cache_base}/triton"
    
    worker_name = f"GPU{gpu_id}-Worker{session_id}"
    print(f"🚀 [{worker_name}] Started.")

    while True:
        try:
            folder_path = folder_queue.get_nowait()
        except Exception:
            # Exit if the queue is empty
            break

        try:
            # Build the command
            cmd = [sys.executable, BACKEND_SCRIPT, "--input_folder_path", folder_path] + BACKEND_ARGS
            
            start_time = time.time()
            
            # Execute subprocess (capture stdout for duration parsing)
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                env=os.environ.copy() # Pass environment variables (CUDA_VISIBLE_DEVICES)
            )

            if process.returncode != 0:
                print(f"❌ [{worker_name}] Error processing {folder_path}")
                print(f"Return code: {process.returncode}")
                print(f"Stderr:\n{process.stderr}")
                print(f"Stdout:\n{process.stdout[-1000:]}")
                error_queue.put(folder_path)
                continue

            # Parse duration (refer to the backend code's print statements)
            # Look for the "Audio duration: 123.45 seconds" pattern
            duration = 0.0
            for line in process.stdout.split('\n'):
                if "Audio duration:" in line and "seconds" in line:
                    try:
                        # e.g.: Audio duration: 123.45 seconds ...
                        parts = line.split("Audio duration:")[1].split("seconds")[0].strip()
                        duration = float(parts)
                    except:
                        pass
            
            # Send to the result queue (folder_path, audio_duration)
            result_queue.put((folder_path, duration))
            
        except Exception as e:
            print(f"💥 [{worker_name}] Critical Exception: {e}")
            error_queue.put(folder_path)

    print(f"💤 [{worker_name}] Finished.")

def preload_models():
    """
    Pre-downloads shared models before all workers start.
    """
    print("⏳ Pre-downloading shared models...")

    # Set environment variables (same as those set in main)
    cache_base = "./.cache"
    os.environ["TORCH_HOME"] = f"{cache_base}/torch"
    os.environ["HF_HOME"] = f"{cache_base}/huggingface"
    os.environ["TRANSFORMERS_CACHE"] = f"{cache_base}/huggingface/transformers"
    os.environ["HF_DATASETS_CACHE"] = f"{cache_base}/huggingface/datasets"
    os.environ["XDG_CACHE_HOME"] = cache_base
    os.environ["TRITON_CACHE_DIR"] = f"{cache_base}/triton"

    try:
        # Download Silero VAD model
        print("  - Downloading Silero VAD model...")
        torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        print("  ✅ Silero VAD model downloaded")
    except Exception as e:
        print(f"  ⚠️ Warning: Could not pre-download Silero VAD: {e}")

    print("✅ Model pre-loading complete")

def main():
    # 0. Create cache directories
    cache_base = "./.cache"
    os.makedirs(f"{cache_base}/torch/hub", exist_ok=True)
    os.makedirs(f"{cache_base}/huggingface/transformers", exist_ok=True)
    os.makedirs(f"{cache_base}/huggingface/datasets", exist_ok=True)
    os.makedirs(f"{cache_base}/triton", exist_ok=True)
    print(f"📁 Cache directories created at {cache_base}")

    # 0.5. Pre-download shared models
    preload_models()

    # 1. Check GPUs
    num_gpus = get_gpu_count()
    if num_gpus == 0:
        print("❌ No GPUs found. Exiting.")
        return

    sessions_per_gpu = 3
    total_workers = num_gpus * sessions_per_gpu
    print(f"⚙️  Configuration: {num_gpus} GPUs x {sessions_per_gpu} Sessions = {total_workers} Workers")

    # 2. Scan and sort data
    all_folders = find_subfolders_with_opus(INPUT_ROOT)
    print(f"Found {len(all_folders)} directories containing audio.")

    # 3. Resume filtering
    processed_folders = load_processed_list()
    todo_folders = [f for f in all_folders if f not in processed_folders]
    print(f"Skipping {len(processed_folders)} already processed. Remaining: {len(todo_folders)}")

    if not todo_folders:
        print("✅ All tasks completed.")
        return

    # 4. Initialize WandB (main process only)
    wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY if WANDB_ENTITY else None, name=WANDB_RUN_NAME)
    
    # 5. Prepare multiprocessing
    manager = multiprocessing.Manager()
    folder_queue = manager.Queue()
    result_queue = manager.Queue()
    error_queue = manager.Queue()

    # Load data into the queue
    for folder in todo_folders:
        folder_queue.put(folder)

    # 6. Create and start workers
    processes = []
    for gpu_id in range(num_gpus):
        for session_id in range(sessions_per_gpu):
            p = multiprocessing.Process(
                target=worker_process,
                args=(gpu_id, session_id, folder_queue, result_queue, error_queue)
            )
            p.start()
            processes.append(p)
            # Stagger process starts to prevent initial load spikes
            time.sleep(2) 

    # 7. Monitoring loop (main thread)
    total_tasks = len(todo_folders)
    pbar = tqdm(total=total_tasks, desc="Processing Audio")
    
    completed_count = 0
    total_audio_duration = 0.0
    
    while completed_count < total_tasks:
        # Check for errors
        if not error_queue.empty():
            err_folder = error_queue.get()
            print(f"\n⚠️ Process failed for: {err_folder}")
            # Skip the error and increment the count (retry logic can be added later)
            completed_count += 1
            pbar.update(1)
            continue

        # Process results
        if not result_queue.empty():
            folder, duration = result_queue.get()
            
            # Record to log
            append_to_processed_list(folder)
            
            # Update statistics
            completed_count += 1
            total_audio_duration += duration
            
            # WandB logging
            wandb.log({
                "progress_percent": (completed_count / total_tasks) * 100,
                "processed_folders": completed_count,
                "cumulative_audio_hours": total_audio_duration / 3600,
                "current_audio_seconds": duration
            })
            
            pbar.update(1)
        
        # Check if all processes have died (guard against abnormal termination)
        if not any(p.is_alive() for p in processes) and result_queue.empty() and error_queue.empty():
            print("\nAll workers stopped unexpectedly.")
            break
        
        time.sleep(0.1)

    # 8. Shutdown
    for p in processes:
        p.join()

    wandb.finish()
    print(f"\n🎉 Pipeline Finished. Total Audio Processed: {total_audio_duration/3600:.2f} hours.")

if __name__ == "__main__":
    # Usage: python run_frontend.py
    main()
