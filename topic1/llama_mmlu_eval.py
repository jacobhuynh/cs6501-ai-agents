"""
Llama 3.2-1B MMLU Evaluation Script (Laptop Optimized with Quantization)

This script evaluates Llama 3.2-1B on the MMLU benchmark.
Optimized for laptops with 4-bit or 8-bit quantization to reduce memory usage.

Quantization options:
- 4-bit: ~1.5 GB VRAM/RAM (default for laptop)
- 8-bit: ~2.5 GB VRAM/RAM
- No quantization: ~5 GB VRAM/RAM

Usage:
1. Install: pip install transformers torch datasets accelerate tqdm bitsandbytes
2. Login: huggingface-cli login
3. Run: python llama_mmlu_eval_quantized.py

Set QUANTIZATION_BITS below to choose quantization level.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
import json
from tqdm.auto import tqdm
import os
from datetime import datetime
import sys
import platform
import time
import resource

# ============================================================================
# CONFIGURATION - Modify these settings
# ============================================================================

# List of models to evaluate
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

# GPU settings
# If True, will attempt to use the best available GPU (CUDA for NVIDIA, MPS for Apple Silicon)
# If False, will always use CPU regardless of available hardware
USE_GPU = False  # Set to False to force CPU-only execution

MAX_NEW_TOKENS = 1

# Verbose output - print each question, model answer, and correctness
VERBOSE_OUTPUT = True  # Set to True to see detailed question-by-question output

# Quantization settings
# Options: 4, 8, or None (default is None for full precision)
#
# To enable quantization, change QUANTIZATION_BITS to one of the following:
#   QUANTIZATION_BITS = 4   # 4-bit quantization: ~1.5 GB memory (most memory efficient)
#   QUANTIZATION_BITS = 8   # 8-bit quantization: ~2.5 GB memory (balanced quality/memory)
#   QUANTIZATION_BITS = None  # No quantization: ~5 GB memory (full precision, best quality)
#
# Notes:
# - Quantization requires the 'bitsandbytes' package: pip install bitsandbytes
# - Quantization only works with CUDA (NVIDIA GPUs), not with Apple Metal (MPS)
# - If using Apple Silicon, quantization will be automatically disabled

QUANTIZATION_BITS = 4  # Change to 4 or 8 to enable quantization

# 10 diverse MMLU subjects for evaluation (uncomment others as needed)
MMLU_SUBJECTS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    # "business_ethics",
    # "clinical_knowledge",
    # "college_biology",
    # "college_chemistry",
    "college_computer_science",
    # "college_mathematics",
    # "college_medicine",
    "college_physics",
    # "computer_security",
    # "conceptual_physics",
    "econometrics",
    # "electrical_engineering",
    # "elementary_mathematics",
    # "formal_logic",
    # "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    # "high_school_computer_science",
    # "high_school_european_history",
    # "high_school_geography",
    # "high_school_government_and_politics",
    # "high_school_macroeconomics",
    # "high_school_mathematics",
    # "high_school_microeconomics",
    # "high_school_physics",
    # "high_school_psychology",
    # "high_school_statistics",
    # "high_school_us_history",
    # "high_school_world_history",
    # "human_aging",
    # "human_sexuality",
    # "international_law",
    # "jurisprudence",
    # "logical_fallacies",
    "machine_learning",
    # "management",
    # "marketing",
    # "medical_genetics",
    # "miscellaneous",
    # "moral_disputes",
    # "moral_scenarios",
    # "nutrition",
    # "philosophy",
    # "prehistory",
    # "professional_accounting",
    # "professional_law",
    # "professional_medicine",
    # "professional_psychology",
    # "public_relations",
    # "security_studies",
    # "sociology",
    # "us_foreign_policy",
    # "virology",
    "world_religions",
]


class TimingStats:
    """Track real time, CPU time, and GPU time for model evaluation."""

    def __init__(self):
        self.real_time = 0.0
        self.cpu_time = 0.0
        self.gpu_time = 0.0
        self._start_real = None
        self._start_cpu = None
        self._start_gpu_event = None
        self._end_gpu_event = None

    def start(self, device):
        """Start timing."""
        self._start_real = time.perf_counter()
        self._start_cpu = resource.getrusage(resource.RUSAGE_SELF)

        # GPU timing (CUDA only)
        if device == "cuda" and torch.cuda.is_available():
            self._start_gpu_event = torch.cuda.Event(enable_timing=True)
            self._end_gpu_event = torch.cuda.Event(enable_timing=True)
            self._start_gpu_event.record()

    def stop(self, device):
        """Stop timing and accumulate results."""
        # Real time
        end_real = time.perf_counter()
        self.real_time += (end_real - self._start_real)

        # CPU time (user + system)
        end_cpu = resource.getrusage(resource.RUSAGE_SELF)
        cpu_user = end_cpu.ru_utime - self._start_cpu.ru_utime
        cpu_sys = end_cpu.ru_stime - self._start_cpu.ru_stime
        self.cpu_time += (cpu_user + cpu_sys)

        # GPU time (CUDA only)
        if device == "cuda" and self._end_gpu_event is not None:
            self._end_gpu_event.record()
            torch.cuda.synchronize()
            self.gpu_time += self._start_gpu_event.elapsed_time(self._end_gpu_event) / 1000.0  # Convert ms to seconds

    def get_summary(self):
        """Return timing summary as dict."""
        return {
            "real_time_seconds": round(self.real_time, 2),
            "cpu_time_seconds": round(self.cpu_time, 2),
            "gpu_time_seconds": round(self.gpu_time, 2),
        }

    def print_summary(self, model_name=""):
        """Print timing summary."""
        prefix = f"[{model_name}] " if model_name else ""
        print(f"{prefix}Real time: {self.real_time:.2f}s")
        print(f"{prefix}CPU time:  {self.cpu_time:.2f}s")
        print(f"{prefix}GPU time:  {self.gpu_time:.2f}s")


def detect_device():
    """Detect the best available device (CUDA, MPS, or CPU)"""

    # If GPU is disabled, always use CPU
    if not USE_GPU:
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        return "cuda"

    # Check for Apple Silicon with Metal
    if torch.backends.mps.is_available():
        # Check if we're actually on Apple ARM
        is_apple_arm = platform.system() == "Darwin" and platform.processor() == "arm"

        if is_apple_arm:
            # Metal is available but incompatible with quantization
            if QUANTIZATION_BITS is not None:
                print("\n" + "="*70)
                print("ERROR: Metal and Quantization Conflict")
                print("="*70)
                print("Metal Performance Shaders (MPS) is incompatible with quantization.")
                print(f"You have USE_GPU = True and QUANTIZATION_BITS = {QUANTIZATION_BITS}")
                print("")
                print("Please choose one of the following options:")
                print("  1. Set USE_GPU = False to use CPU with quantization")
                print("  2. Set QUANTIZATION_BITS = None to use Metal without quantization")
                print("="*70 + "\n")
                sys.exit(1)
            return "mps"

    # Default to CPU
    return "cpu"




def check_environment():
    global QUANTIZATION_BITS
    """Check environment and dependencies"""
    print("="*70)
    print("Environment Check")
    print("="*70)

    # Check if in Colab
    try:
        import google.colab
        print("✓ Running in Google Colab")
        in_colab = True
    except:
        print("✓ Running locally (not in Colab)")
        in_colab = False

    # Check system info
    print(f"✓ Platform: {platform.system()} ({platform.machine()})")
    if platform.system() == "Darwin":
        print(f"✓ Processor: {platform.processor()}")

    # Detect and set device
    device = detect_device()

    # Check device
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU Available: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.2f} GB")
    elif device == "mps":
        print("✓ Apple Metal (MPS) Available")
        print("✓ Using Metal Performance Shaders for GPU acceleration")
    else:
        print("⚠️  No GPU detected - running on CPU")
       
    # Check quantization support

    if QUANTIZATION_BITS is not None:
        try:
            import bitsandbytes
            print(f"✓ bitsandbytes installed - {QUANTIZATION_BITS}-bit quantization available")
        except ImportError:
            print(f"❌ bitsandbytes NOT installed - cannot use quantization")
            sys.exit(1)
        if device == 'mps':
            print(f"❌ Apple METAL is incompatible with quantization")
            print("✓ Quantization disabled - loading full precision model")
            QUANTIZATION_BITS = None
            sys.exit(1)
    else:
        print("✓ Quantization disabled - loading full precision model")
    
    # Check HF authentication
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ Hugging Face authenticated")
        else:
            print("⚠️  No Hugging Face token found")
            print("Run: huggingface-cli login")
    except:
        print("⚠️  Could not check Hugging Face authentication")
    
    # Print configuration
    print("\n" + "="*70)
    print("Configuration")
    print("="*70)
    print(f"Models to evaluate: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"  {i}. {model}")
    print(f"Device: {device}")
    if QUANTIZATION_BITS is not None:
        print(f"Quantization: {QUANTIZATION_BITS}-bit")
        if QUANTIZATION_BITS == 4:
            print(f"Expected memory: ~1.5 GB")
        elif QUANTIZATION_BITS == 8:
            print(f"Expected memory: ~2.5 GB")
    else:
        print(f"Quantization: None (full precision)")
        if device == "cuda":
            print(f"Expected memory: ~2.5 GB (FP16)")
        elif device == "mps":
            print(f"Expected memory: ~2.5 GB (FP16)")
        else:
            print(f"Expected memory: ~5 GB (FP32)")
    print(f"Number of subjects: {len(MMLU_SUBJECTS)}")
    print(f"Verbose output: {VERBOSE_OUTPUT}")

    print("="*70 + "\n")
    return in_colab, device


def get_quantization_config():
    """Create quantization config based on settings"""
    if QUANTIZATION_BITS is None:
        return None
    
    if QUANTIZATION_BITS == 4:
        # 4-bit quantization (most memory efficient)
        config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Double quantization for extra compression
            bnb_4bit_quant_type="nf4"  # NormalFloat4 - better for LLMs
        )
        print("Using 4-bit quantization (NF4 + double quant)")
        print("Memory usage: ~1.5 GB")
    elif QUANTIZATION_BITS == 8:
        # 8-bit quantization (balanced)
        config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False
        )
        print("Using 8-bit quantization")
        print("Memory usage: ~2.5 GB")
    else:
        raise ValueError(f"Invalid QUANTIZATION_BITS: {QUANTIZATION_BITS}. Use 4, 8, or None")
    
    return config


def load_model_and_tokenizer(model_name, device):
    """Load model with optional quantization"""
    print(f"\nLoading model {model_name}...")
    print(f"Device: {device}")

    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded")

        # Get quantization config
        quant_config = get_quantization_config()

        # Load model
        print("Loading model (this may take 2-3 minutes)...")

        if quant_config is not None:
            # Quantized model loading (only works with CUDA)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        else:
            # Non-quantized model loading
            if device == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            elif device == "mps":
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)
            else:  # CPU
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                model = model.to(device)

        model.eval()

        # Print model info
        print("✓ Model loaded successfully!")
        print(f"  Model device: {next(model.parameters()).device}")
        print(f"  Model dtype: {next(model.parameters()).dtype}")

        # Check memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1e9
            memory_reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"  GPU Memory: {memory_allocated:.2f} GB allocated, {memory_reserved:.2f} GB reserved")

            # Check if using quantization
            if quant_config is not None:
                print(f"  Quantization: {QUANTIZATION_BITS}-bit active")
        elif device == "mps":
            print(f"  Running on Apple Metal (MPS)")

        return model, tokenizer

    except Exception as e:
        print(f"\n❌ Error loading model: {e}")
        print("\nPossible causes:")
        print("1. No Hugging Face token - Run: huggingface-cli login")
        print("2. Llama license not accepted - Visit: https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct")
        print("3. bitsandbytes not installed - Run: pip install bitsandbytes")
        print("4. Out of memory - Try 4-bit quantization or smaller model")
        raise


def format_mmlu_prompt(question, choices):
    """Format MMLU question as multiple choice"""
    choice_labels = ["A", "B", "C", "D"]
    prompt = f"{question}\n\n"
    for label, choice in zip(choice_labels, choices):
        prompt += f"{label}. {choice}\n"
    prompt += "\nAnswer:"
    return prompt


def get_model_prediction(model, tokenizer, prompt):
    """Get model's prediction for multiple-choice question"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            temperature=1.0
        )
    
    generated_text = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:], 
        skip_special_tokens=True
    )
    
    answer = generated_text.strip()[:1].upper()
    
    if answer not in ["A", "B", "C", "D"]:
        for char in generated_text.upper():
            if char in ["A", "B", "C", "D"]:
                answer = char
                break
        else:
            answer = "A"
    
    return answer


def evaluate_subject(model, tokenizer, subject, device, timing_stats):
    """Evaluate model on a specific MMLU subject"""
    print(f"\n{'='*70}")
    print(f"Evaluating subject: {subject}")
    print(f"{'='*70}")

    try:
        dataset = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"❌ Error loading subject {subject}: {e}")
        return None

    correct = 0
    total = 0

    for example in tqdm(dataset, desc=f"Testing {subject}", leave=True):
        question = example["question"]
        choices = example["choices"]
        correct_answer_idx = example["answer"]
        correct_answer = ["A", "B", "C", "D"][correct_answer_idx]

        prompt = format_mmlu_prompt(question, choices)

        # Time the model prediction
        timing_stats.start(device)
        predicted_answer = get_model_prediction(model, tokenizer, prompt)
        timing_stats.stop(device)

        is_correct = predicted_answer == correct_answer
        if is_correct:
            correct += 1
        total += 1

        # Verbose output
        if VERBOSE_OUTPUT:
            status = "CORRECT" if is_correct else "WRONG"
            print(f"\n--- Question {total} ---")
            print(f"Q: {question[:200]}..." if len(question) > 200 else f"Q: {question}")
            print(f"Choices: A) {choices[0][:50]}  B) {choices[1][:50]}  C) {choices[2][:50]}  D) {choices[3][:50]}")
            print(f"Model answer: {predicted_answer} | Correct answer: {correct_answer} | {status}")

    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"✓ Result: {correct}/{total} correct = {accuracy:.2f}%")

    return {
        "subject": subject,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


def evaluate_model(model_name, device):
    """Evaluate a single model on all MMLU subjects."""
    print("\n" + "="*70)
    print(f"EVALUATING MODEL: {model_name}")
    print("="*70)

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Create timing stats for this model
    timing_stats = TimingStats()

    # Evaluate
    results = []
    total_correct = 0
    total_questions = 0

    print(f"\n{'='*70}")
    print(f"Starting evaluation on {len(MMLU_SUBJECTS)} subjects")
    print(f"{'='*70}\n")

    for i, subject in enumerate(MMLU_SUBJECTS, 1):
        print(f"\nProgress: {i}/{len(MMLU_SUBJECTS)} subjects")
        result = evaluate_subject(model, tokenizer, subject, device, timing_stats)
        if result:
            results.append(result)
            total_correct += result["correct"]
            total_questions += result["total"]

    # Calculate overall accuracy
    overall_accuracy = (total_correct / total_questions * 100) if total_questions > 0 else 0

    # Clean up model to free memory before loading next model
    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": model_name,
        "results": results,
        "total_correct": total_correct,
        "total_questions": total_questions,
        "overall_accuracy": overall_accuracy,
        "timing": timing_stats.get_summary(),
        "timing_stats": timing_stats,
    }


def main():
    """Main evaluation function"""
    print("\n" + "="*70)
    print("Multi-Model MMLU Evaluation")
    print("="*70 + "\n")

    # Check environment
    in_colab, device = check_environment()

    # Store results for all models
    all_model_results = []

    start_time = datetime.now()

    # Evaluate each model
    for model_idx, model_name in enumerate(MODELS, 1):
        print(f"\n{'#'*70}")
        print(f"# Model {model_idx}/{len(MODELS)}: {model_name}")
        print(f"{'#'*70}")

        model_result = evaluate_model(model_name, device)
        all_model_results.append(model_result)

    end_time = datetime.now()
    total_duration = (end_time - start_time).total_seconds()

    # Print comprehensive summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY - ALL MODELS")
    print("="*70)
    print(f"Total models evaluated: {len(MODELS)}")
    print(f"Subjects per model: {len(MMLU_SUBJECTS)}")
    print(f"Quantization: {QUANTIZATION_BITS}-bit" if QUANTIZATION_BITS else "Quantization: None (full precision)")
    print(f"Device: {device}")
    print(f"Total duration: {total_duration/60:.1f} minutes")
    print("="*70)

    # Print per-model results with timing
    print("\n" + "-"*70)
    print("PER-MODEL RESULTS")
    print("-"*70)

    for result in all_model_results:
        model_short_name = result["model"].split("/")[-1]
        timing = result["timing"]
        print(f"\n{model_short_name}:")
        print(f"  Accuracy: {result['overall_accuracy']:.2f}% ({result['total_correct']}/{result['total_questions']})")
        print(f"  Timing:")
        print(f"    Real time: {timing['real_time_seconds']:.2f}s")
        print(f"    CPU time:  {timing['cpu_time_seconds']:.2f}s")
        print(f"    GPU time:  {timing['gpu_time_seconds']:.2f}s")

    # Print comparison table
    print("\n" + "-"*70)
    print("MODEL COMPARISON TABLE")
    print("-"*70)
    print(f"{'Model':<40} {'Accuracy':>10} {'Real(s)':>10} {'CPU(s)':>10} {'GPU(s)':>10}")
    print("-"*70)
    for result in all_model_results:
        model_short_name = result["model"].split("/")[-1][:38]
        timing = result["timing"]
        print(f"{model_short_name:<40} {result['overall_accuracy']:>9.2f}% {timing['real_time_seconds']:>10.2f} {timing['cpu_time_seconds']:>10.2f} {timing['gpu_time_seconds']:>10.2f}")
    print("-"*70)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    quant_suffix = f"_{QUANTIZATION_BITS}bit" if QUANTIZATION_BITS else "_full"
    output_file = f"mmlu_multi_model_results{quant_suffix}_{timestamp}.json"

    output_data = {
        "models": MODELS,
        "quantization_bits": QUANTIZATION_BITS,
        "timestamp": timestamp,
        "device": str(device),
        "total_duration_seconds": total_duration,
        "subjects": MMLU_SUBJECTS,
        "model_results": [
            {
                "model": r["model"],
                "overall_accuracy": r["overall_accuracy"],
                "total_correct": r["total_correct"],
                "total_questions": r["total_questions"],
                "timing": r["timing"],
                "subject_results": r["results"],
            }
            for r in all_model_results
        ]
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to: {output_file}")

    # Colab-specific instructions
    if in_colab:
        print("\n" + "="*70)
        print("To download results in Colab:")
        print("="*70)
        print(f"from google.colab import files")
        print(f"files.download('{output_file}')")

    print("\n✅ Evaluation complete!")
    return output_file


if __name__ == "__main__":
    try:
        output_file = main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
