from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
tokenizer = T5Tokenizer.from_pretrained('./t5-qa-tokenizer')
model = T5ForConditionalGeneration.from_pretrained('./t5-qa-model')

question = "What is dopamine?"
input_text = "question: " + question + " </s>"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate an answer
outputs = model.generate(inputs.input_ids, max_length=50, num_beams=5, early_stopping=True)
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Question:", question)
print("Answer:", answer)
