import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer

# ==========================================
# CONFIGURATION
# ==========================================
MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
MAX_HISTORY = 10 # STRATEGY: Keep only the last 10 messages (5 turns)

print(f"Loading {MODEL_ID} with Sliding Window Strategy...")

# 1. Load Model (4-bit for Colab)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto"
    )
except Exception as e:
    print(f"❌ Error: {e}")
    raise e

# 2. Context Management Function (The Strategy)
def manage_context(history, limit):
    """
    Implements Sliding Window:
    - Always keeps the System Prompt (index 0)
    - Trims the middle if history exceeds limit
    """
    if len(history) > limit:
        # Keep [System Prompt] + [Last (limit-1) messages]
        return [history[0]] + history[-(limit-1):]
    return history

# 3. Chat Loop
messages = [
    {"role": "system", "content": "You are a helpful and concise AI assistant."}
]

print("\n" + "="*50)
print(f"🤖 AGENT ONLINE (Memory Limit: {MAX_HISTORY} msgs)")
print("Type 'quit' to stop.")
print("="*50)

while True:
    try:
        user_input = input("\n👤 You: ").strip()
        if user_input.lower() in ["quit", "exit"]: break
        if not user_input: continue
            
        messages.append({"role": "user", "content": user_input})
        
        # Strategy: Trim Memory
        messages = manage_context(messages, MAX_HISTORY)
        
        text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        # --- NEW: Create Streamer ---
        # This makes it print directly to the console as it generates
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        print("🤖 Agent: ", end="", flush=True) # Print header
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                streamer=streamer,       # <--- Attach streamer here
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # We still need to save the response to memory for the NEXT turn
        # We decode the generated IDs again just to store them string-wise
        response_ids = generated_ids[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        messages.append({"role": "assistant", "content": response})

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"❌ Error: {e}")