
from datasets import load_dataset


# Stream NQ Q&A pairs 
dataset = load_dataset(
    "sentence-transformers/natural-questions",
    split="train",
    streaming=True,
)

print("Connected! Pulling first 5 examples...\n")

# Print the first 5 examples
for i, example in enumerate(dataset):
    print(f"--- Example {i + 1} ---")
    print("Question:", example["query"])
    print("Answer  :", example["answer"][:150], "...")
    print()
    if i >= 5:
        break

