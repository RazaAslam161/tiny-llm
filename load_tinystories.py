from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories")

print(ds)
print("\n--- Example story (train[0]) ---\n")
print(ds["train"][0]["text"])
