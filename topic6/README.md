# Topic 6: Vision-Language Models (VLM)

AI Assistance: Used ClaudeCode to help complete assignment and generate README. Assignment Answers were AI assisted, but not completely AI generated.

## Files

### Scripts & Notebooks

- **[vlm.ipynb](vlm.ipynb)** — Jupyter notebook implementing two VLM exercises using LLaVA (via Ollama) and LangGraph.

  **Exercise 1: Vision-Language Chat Agent** — A multi-turn interactive chat agent that answers questions about uploaded images. Builds a single-node LangGraph workflow where the LLaVA node maintains full conversation history across turns, attaching the image only to the first user message to avoid redundant processing. Provides an IPython widget UI with image upload (auto-resized to 512×512), a text input, and a scrollable chat display.

  **Exercise 2: Video Surveillance Agent** — A person detection system that analyzes video for entry/exit events. Uses OpenCV to extract frames at a configurable interval (default 2s), encodes each frame as base64, and queries LLaVA with a binary YES/NO prompt ("Is there a person visible in the scene?"). Tracks presence state across frames to detect ENTER/EXIT transitions and outputs a timestamped surveillance report (e.g., `➡️ Person ENTER at 00:05`, `⬅️ Person EXIT at 00:15`).
