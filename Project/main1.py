from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("sidhq/email-thread-summary")

def tokenize_function(examples):
    return tokenizer(examples["summary"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_test_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(400))

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

import numpy as np
from sklearn.metrics import accuracy_score, classification_report

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    # Compute accuracy
    acc = accuracy_score(labels, predictions)
    
    # Compute precision, recall, f1-score, and support
    class_report = classification_report(labels, predictions, output_dict=True)
    
    # You can add more metrics as needed
    # ...

    # Return computed metrics
    return {
        'accuracy': acc,
        'precision': class_report['macro avg']['precision'],
        'recall': class_report['macro avg']['recall'],
        'f1-score': class_report['macro avg']['f1-score'],
        # Add more metrics as needed
        # ...
    }


training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    output_dir="./results",
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_test_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()


def extract_action_items(email_text):
    inputs = tokenizer(email_text, return_tensors="pt")
    outputs = model(**inputs)
    # Process the model’s output to extract action items
    # This step depends on how your model is designed and trained
    # You might need to write additional code to interpret the model’s predictions
    return outputs

email = "Kimberly is asking Kate and Monica when the trades will be changed in Enpower to reflect the Dow Jones Index. Kate informs Kimberly that Bob Badeer is leading the project and offers to update her on the progress. Monica confirms that Bob is handling the negotiations and assures Kimberly that deals liquidating within the month have been updated. Kimberly sends a follow-up email to Kate and Monica, reiterating her question about when the trades will be changed in Enpower."
action_items = extract_action_items(email)
print(action_items)
