import json
import random
from dotenv import load_dotenv
import numpy as np
import tinker
import tinker.types as types
from sql_matches import sql_matches

load_dotenv()

# ── Step 1: Load and explore ──────────────────────────────────────────────────

with open("sql_create_context_v4.json") as f:
    data = json.load(f)

print(f"Total examples: {len(data)}")

ex = data[0]
print(f"\nSample example:")
print(f"  Question: {ex['question']}")
print(f"  Context:  {ex['context'][:120]}...")
print(f"  Answer:   {ex['answer']}")

print("\nA few more examples to see SQL complexity range:")
for i in [100, 1000, 5000]:
    e = data[i]
    print(f"  [{i}] Q: {e['question']}")
    print(f"       A: {e['answer']}\n")

NUM_TEST_EXAMPLES = 200
random.seed(42)
random.shuffle(data)
test_data = data[:NUM_TEST_EXAMPLES]
train_data = data[NUM_TEST_EXAMPLES:]

print(f"Training examples: {len(train_data)} (all except evaluation)")
print(f"Test examples:     {len(test_data)}")

# ── Step 2: Define the prompt format ─────────────────────────────────────────

def make_prompt(example: dict) -> str:
    """Format a dataset example as a model prompt (input only)."""
    return (
        f"Table schema:\n{example['context']}\n"
        f"Question: {example['question']}\n"
        f"SQL:"
    )


def make_prompt_with_completion(example: dict) -> dict:
    """Return prompt and completion strings for SFT."""
    return {
        "prompt": make_prompt(example),
        "completion": f" {example['answer']}",
    }


# Sanity-check the format
sample = make_prompt_with_completion(train_data[0])
print("\nPrompt format example:")
print(sample["prompt"])
print(f"Completion: {sample['completion']}")

# ── Step 3: Evaluate the base model ──────────────────────────────────────────

def sample_from_model(sampling_client, tokenizer, context: str, question: str) -> str:
    """Generate SQL from the model given schema and question."""
    prompt = (
        f"Table schema:\n{context}\n"
        f"Question: {question}\n"
        f"SQL: "
    )
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    model_input = tinker.ModelInput.from_ints(tokens=prompt_tokens)
    params = tinker.SamplingParams(
        max_tokens=150,
        temperature=0.0,
        stop=["\n\n", "Question:"],
    )
    result = sampling_client.sample(
        prompt=model_input, sampling_params=params, num_samples=1
    ).result()
    if result.sequences:
        return tokenizer.decode(result.sequences[0].tokens).strip()
    return ""


def eval_one(sampling_client, tokenizer, ex: dict) -> bool:
    """Evaluate one example: generate SQL and check if it matches expected."""
    sql = sample_from_model(sampling_client, tokenizer, ex["context"], ex["question"])
    return sql_matches(sql, ex["answer"], schema=ex["context"])


def evaluate_test_set(sampling_client, tokenizer, test_data: list, label: str) -> float:
    """Compute accuracy = fraction of test examples where generated SQL matches expected."""
    correct = 0
    for i, ex in enumerate(test_data):
        hit = eval_one(sampling_client, tokenizer, ex)
        correct += hit
        if (i + 1) % 20 == 0:
            print(f"  [{label}] {i + 1}/{len(test_data)} — running accuracy: {correct/(i+1):.2%}")
    return correct / len(test_data)


service_client = tinker.ServiceClient()
base_model = "meta-llama/Llama-3.2-1B"
training_client = service_client.create_lora_training_client(base_model=base_model)
tokenizer = training_client.get_tokenizer()

RUN_BASE_EVAL = False  # Set True to re-run base model evaluation (already done: 47%)

if RUN_BASE_EVAL:
    print("\n--- Evaluating Base Model on 200 Test Questions ---")
    base_sampling_client = training_client.save_weights_and_get_sampling_client(
        name="base-model"
    )
    base_accuracy = evaluate_test_set(base_sampling_client, tokenizer, test_data, "base")
    print(f"\nBase model accuracy: {base_accuracy:.2%} ({int(base_accuracy * len(test_data))}/{len(test_data)})")
else:
    print("\n[Skipping base eval — known result: 47.00% (94/200)]")

# ── Step 4: Prepare training data ────────────────────────────────────────────

def format_prompt(example: dict) -> tuple[str, str]:
    """Format example as prompt and completion for text-to-SQL."""
    prompt = (
        f"Table schema:\n{example['context']}\n"
        f"Question: {example['question']}\n"
        f"SQL: "
    )
    completion = example["answer"]
    return prompt, completion


def process_example(example: dict, tokenizer) -> types.Datum:
    """Convert a (question, context, answer) example into a Tinker Datum.

    Loss weights are 0 on the prompt tokens so the model only learns to
    predict the SQL completion, not to memorize the question/schema.
    """
    prompt, completion = format_prompt(example)

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    prompt_weights = [0.0] * len(prompt_tokens)

    # Space before completion; \n\n teaches the model when to stop
    completion_str = f" {completion}\n\n"
    completion_tokens = tokenizer.encode(completion_str, add_special_tokens=False)
    completion_weights = [1.0] * len(completion_tokens)

    tokens = prompt_tokens + completion_tokens
    weights = prompt_weights + completion_weights

    # Next-token prediction: input is tokens[:-1], target is tokens[1:]
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]

    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "target_tokens": np.array(target_tokens, dtype=np.int64),
            "weights": np.array(weights, dtype=np.float32),
        },
    )


print("\n--- Preparing Training Data ---")
processed_train = [process_example(ex, tokenizer) for ex in train_data]
random.shuffle(processed_train)
print(f"Processed {len(processed_train)} training examples")

# Sanity-check one datum
d = processed_train[0]
n_prompt = int(sum(1 for w in d.loss_fn_inputs["weights"].tolist() if w == 0.0))
n_completion = int(sum(1 for w in d.loss_fn_inputs["weights"].tolist() if w == 1.0))
print(f"  Sample datum — prompt tokens (weight=0): {n_prompt}, completion tokens (weight=1): {n_completion}")

# ── Step 5: Train ─────────────────────────────────────────────────────────────

NUM_EPOCHS = 1
BATCH_SIZE = 256
LEARNING_RATE = 5e-4  # Tinker-recommended for Llama-3.2-1B with LoRA

print("\n--- Training ---")
step = 0
for epoch in range(NUM_EPOCHS):
    random.shuffle(processed_train)
    for batch_idx in range(0, len(processed_train), BATCH_SIZE):
        batch = processed_train[batch_idx : batch_idx + BATCH_SIZE]
        if len(batch) == 0:
            break

        fwdbwd_future = training_client.forward_backward(batch, "cross_entropy")
        optim_future = training_client.optim_step(
            types.AdamParams(learning_rate=LEARNING_RATE)
        )

        fwdbwd_result = fwdbwd_future.result()
        optim_result = optim_future.result()

        to_arr = lambda x: x.to_numpy() if hasattr(x, "to_numpy") else np.array(x.tolist())
        logprobs = np.concatenate([to_arr(o["logprobs"]) for o in fwdbwd_result.loss_fn_outputs])
        weights = np.concatenate([to_arr(d.loss_fn_inputs["weights"]) for d in batch])
        loss = float(-np.dot(logprobs, weights) / (weights.sum() + 1e-8))

        step += 1
        if step % 100 == 0 or batch_idx + BATCH_SIZE >= len(processed_train):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, update {step}, loss: {loss:.4f}")

# ── Step 6: Evaluate the fine-tuned model ────────────────────────────────────

print("\n--- Evaluating Fine-Tuned Model on 200 Test Questions ---")
save_result = training_client.save_weights_for_sampler(name="finetuned-model").result()
finetuned_sampling_client = service_client.create_sampling_client(
    model_path=save_result.path
)
finetuned_accuracy = evaluate_test_set(finetuned_sampling_client, tokenizer, test_data, "finetuned")
print(f"\nBase model accuracy:      47.00% (94/200)")
print(f"Fine-tuned model accuracy: {finetuned_accuracy:.2%} ({int(finetuned_accuracy * len(test_data))}/{len(test_data)})")
print(f"Improvement: {(finetuned_accuracy - 0.47):.2%}")

# ── Step 7: Test on novel out-of-distribution schemas ────────────────────────

NOVEL_TESTS = [
    # Easy — single table, simple WHERE
    {
        "label": "Easy 1",
        "context": "CREATE TABLE employees (id INTEGER, name VARCHAR, salary REAL, department VARCHAR)",
        "question": "What are the names of employees in the engineering department?",
        "expected": "SELECT name FROM employees WHERE department = 'engineering'",
    },
    {
        "label": "Easy 2",
        "context": "CREATE TABLE products (id INTEGER, name VARCHAR, price REAL, category VARCHAR)",
        "question": "How many products cost more than 50 dollars?",
        "expected": "SELECT COUNT(*) FROM products WHERE price > 50",
    },
    # Medium — aggregation, ORDER BY
    {
        "label": "Medium 1",
        "context": "CREATE TABLE students (id INTEGER, name VARCHAR, score INTEGER, class VARCHAR)",
        "question": "What is the highest score in the science class?",
        "expected": "SELECT MAX(score) FROM students WHERE class = 'science'",
    },
    {
        "label": "Medium 2",
        "context": "CREATE TABLE orders (id INTEGER, customer VARCHAR, amount REAL, date VARCHAR)",
        "question": "List the top 3 customers by total order amount.",
        "expected": "SELECT customer FROM orders GROUP BY customer ORDER BY SUM(amount) DESC LIMIT 3",
    },
    # Hard — JOIN, GROUP BY
    {
        "label": "Hard 1",
        "context": (
            "CREATE TABLE courses (id INTEGER, name VARCHAR, department VARCHAR); "
            "CREATE TABLE enrollments (student_id INTEGER, course_id INTEGER, grade VARCHAR)"
        ),
        "question": "How many students are enrolled in each department?",
        "expected": "SELECT c.department, COUNT(*) FROM enrollments e JOIN courses c ON e.course_id = c.id GROUP BY c.department",
    },
]

print("\n--- Step 7: Novel Out-of-Distribution Schema Tests ---")
for test in NOVEL_TESTS:
    generated = sample_from_model(
        finetuned_sampling_client, tokenizer, test["context"], test["question"]
    )
    correct = sql_matches(generated, test["expected"], schema=test["context"])
    print(f"\n[{test['label']}] {test['question']}")
    print(f"  Generated: {generated}")
    print(f"  Expected:  {test['expected']}")
    print(f"  Match:     {'✓' if correct else '✗'}")
