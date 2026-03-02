# Topic 1: LLM Benchmarking with MMLU

AI Assistance: Used ClaudeCode to help complete assignment and generate README. Assignment Answers were AI assisted, but not completely AI generated.

## Files

### Scripts & Notebooks

- **[chat_agent.py](chat_agent.py)** — Interactive chatbot using Llama 3.2-1B-Instruct with 4-bit quantization. Implements a sliding window context manager that keeps the system prompt and the last 10 messages, dropping older ones to stay within memory limits. Uses `TextStreamer` for real-time streaming output.

- **[llama_mmlu_eval.py](llama_mmlu_eval.py)** — Evaluation script that benchmarks multiple small models (Llama-3.2-1B-Instruct, Qwen2.5-0.5B-Instruct, TinyLlama-1.1B) on 10 MMLU subjects. Supports CPU, GPU (CUDA), and Apple Silicon (MPS) with optional 4-bit or 8-bit quantization. Tracks real time, CPU time, and GPU time via a `TimingStats` class, and saves per-subject accuracy and timing results to JSON.

- **[topic1_ipnyb.ipynb](topic1_ipnyb.ipynb)** — Google Colab notebook covering Parts 6 & 7. Evaluates small models (Llama-3.2-1B, Qwen2.5-1.5B, Gemma-2-2b in FP16) and medium models (Llama-3.1-8B, Mistral-7B, Qwen2.5-7B with 4-bit quantization) across 10 MMLU subjects. Generates accuracy and timing bar charts, and analyzes the overlap in wrong answers between model pairs to assess whether errors are random or systematic.

### Results Files

- **[llama_3.2_1b_cpu_no_quantization.json](llama_3.2_1b_cpu_no_quantization.json)** — CPU run, no quantization. Overall accuracy: **37.45%** (88/235 questions), real time: ~55.6s.

- **[llama_3.2_1b_cpu_4_quantization.json](llama_3.2_1b_cpu_4_quantization.json)** — CPU run, 4-bit quantization. Overall accuracy: **35.32%** (83/235 questions), real time: ~132.1s — notably slower than no-quantization due to dequantization overhead on CPU.

- **[llama_3.2_1b_gpu_no_quantization.json](llama_3.2_1b_gpu_no_quantization.json)** — GPU (Apple MPS) run, no quantization. Overall accuracy: **37.45%** (88/235 questions), real time: ~18.0s — approximately 3× faster than the CPU baseline.

- **[colab_mmlu_results.json](colab_mmlu_results.json)** — Full results from the Colab multi-model evaluation (Parts 6 & 7), covering all six models across all 10 subjects.

### Graphs

- **[graph_accuracy.png](graph_accuracy.png)** — Bar chart comparing accuracy across all evaluated models and MMLU subjects.

- **[graph_timing.png](graph_timing.png)** — Bar chart comparing execution time across all evaluated models and configurations.

---

# Assignment Answers

## Part 4

Looking at the results of the 3 runs, we can see that GPU was fastest, followed by CPU with no quantization, and lastly CPU with 4 bit quantization, with the GPU being almost 3 times as fast as the CPU. Unexpectedly, quantization actually made the CPU slower. Looking up potential causes, this seemes to be because standard CPUs do not have hardware acceleration for 4-bit math, meaning the CPU has to dequantize the weights, resulting in added overhead.

## Part 6/7

The mistakes do not appear to be random. For example, almost all models struggle on more complex questions like math and physics based questions. In addition, for the majority of questions where stronger models (like Qwen 7B) fail, the weaker models also fail (like Llama 1B). In short, the models with less parameters tend to perform worse overall, while models with more parameters tend to do better.
