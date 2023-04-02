!pip install transformers==4.15.0

import torch
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

# Load pre-trained PEGASUS model and tokenizer
model_name = 'google/pegasus-large'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

# Define input text
input_text = "Climate change is a major threat to the planet, with rising temperatures and sea levels causing widespread damage to ecosystems and human societies. Scientists and policymakers around the world are working to address the issue through a variety of measures, including reducing greenhouse gas emissions and investing in renewable energy."

# Tokenize input text
inputs = tokenizer([input_text], max_length=1024, truncation=True, return_tensors='pt')

# Generate summary
summary_ids = model.generate(inputs['input_ids'], max_length=50, min_length=10, length_penalty=2.0)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Input text:", input_text)
print("Generated summary:", summary)
