import json

def count_characters(dataset):
    return sum(len(json.dumps(message)) for example in dataset for message in example["messages"])

# Load the training set
with open('./legalbench_subsets_train.jsonl', 'r', encoding='utf-8') as f:
    training_dataset = [json.loads(line) for line in f]

# Training dataset stats
print("Number of examples in training set:", len(training_dataset))
print("Total number of characters in training set:", count_characters(training_dataset))
print("First example in training set:")
for message in training_dataset[0]["messages"]:
    print(message)

# Load the validation set
with open('./legalbench_subsets_val.jsonl', 'r', encoding='utf-8') as f:
    validation_dataset = [json.loads(line) for line in f]

# Validation dataset stats
print("\nNumber of examples in validation set:", len(validation_dataset))
print("Total number of characters in validation set:", count_characters(validation_dataset))
print("First example in validation set:")
for message in validation_dataset[0]["messages"]:
    print(message)