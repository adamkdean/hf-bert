from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
model = AutoModelForCausalLM.from_pretrained('mosaicml/mpt-1b-redpajama-200b', trust_remote_code=True)

# The text you want to complete
input_text = "Once upon a time, in a land far away,"

# Tokenize the input text
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate a response
output = model.generate(input_ids, max_length=100, temperature=0.7, pad_token_id=tokenizer.eos_token_id)

# Decode the output
completed_text = tokenizer.decode(output[0], skip_special_tokens=True)

print(completed_text)

