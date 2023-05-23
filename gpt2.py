from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_text = "Once upon a time, "
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# generate text
output = model.generate(input_ids, max_length=50, num_return_sequences=5, no_repeat_ngram_size=2, do_sample=True, temperature=0.7)

for i in range(5):  # print all 5 generated sequences
    print(tokenizer.decode(output[i], skip_special_tokens=True))
