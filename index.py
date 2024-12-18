from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load the JSON dataset
with open('invoice_data.json', 'r') as f:
    data = json.load(f)

# Format the dataset for fine-tuning
train_data = []

for item in data:
    input_text = item['input']
    output_text = json.dumps(item['output'])  # Output will be a string representation of the dictionary
    train_data.append({'input': input_text, 'output': output_text})

# Create a dataset
dataset = Dataset.from_list(train_data)

# Load the AutoTokenizer and AutoModel for Seq2SeqLM
model_name = "meta-llama/Llama-3.2-1B"  # Replace with the correct model name if needed
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = examples['input']
    targets = examples['output']
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up Training Arguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory for model predictions and checkpoints
    num_train_epochs=3,              # number of training epochs
    per_device_train_batch_size=4,   # batch size for training
    gradient_accumulation_steps=2,   # number of gradient accumulation steps
    evaluation_strategy="epoch",     # evaluation strategy to use
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    save_steps=500,                  # Save checkpoint every 500 steps
    save_total_limit=2,              # Limit the total amount of checkpoints
    load_best_model_at_end=True,     # Load the best model when finished training
    remove_unused_columns=False      # Keep the output dictionary from the tokenizer
)

# Set up Data Collator
from transformers import DataCollatorForSeq2Seq
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create Trainer and Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Start training
trainer.train()

# Save the model after training
model.save_pretrained('./fine_tuned_llama')
tokenizer.save_pretrained('./fine_tuned_llama')
