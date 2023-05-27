# Copyright (C) 2023 Adam K Dean <adamkdean@googlemail.com>
# Use of this source code is governed by the GPL-3.0
# license that can be found in the LICENSE file.

import timeit

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

def print_message(message, padding=0, prefix='#', skip_message_prefix=False):
    padding_str = '\n'.join([prefix for _ in range(padding)])
    message_prefix = '' if skip_message_prefix else prefix
    if padding > 0:
        print(f'{padding_str}\n{message_prefix} {message}\n{padding_str}')
    else:
        print(f'{message_prefix} {message}')

# [mpt-7b-chat]
# hyperparameter : value
# ---------------+------
# n_parameters   : 6.7B
# n_layers       : 32
# n_heads        : 32
# d_model        : 4096
# vocab size     : 50432
# sequence length: 2048
model_name = 'mosaicml/mpt-7b-chat'
tokenizer_name = 'EleutherAI/gpt-neox-20b'

#
# Initialise
#
times = {}
output_max_length = 150
output_temperature = 0.25
start_time_init = timeit.default_timer()

#
# Load tokenizer
#
print_message('Loading tokenizer', padding=1)
start_time = timeit.default_timer()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
end_time = timeit.default_timer()
execution_time = end_time - start_time
times['tokenizer'] = execution_time
print_message(f'Tokenizer loaded in {execution_time} seconds\n', prefix='~')

#
# Load the model
#
print_message('Loading model', padding=1)
start_time = timeit.default_timer()
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
end_time = timeit.default_timer()
execution_time = end_time - start_time
times['loading_model'] = execution_time
print_message(f'Model loaded in {execution_time} seconds\n', prefix='~')

#
# Tokenize input
#
print_message('Tokenizing input', padding=1)
start_time = timeit.default_timer()
input = 'Describe the planet Earth for me.'
input_ids = tokenizer.encode(input, return_tensors='pt')
end_time = timeit.default_timer()
execution_time = end_time - start_time
times['tokenizing_input'] = execution_time
input_token_count = len(input_ids[0])
print_message(f'Tokenized input ({input_token_count} tokens) in {execution_time} seconds\n', prefix='~')

#
# Generate output
#
print_message('Generating output', padding=1)
start_time = timeit.default_timer()
output = model.generate(
  input_ids,
  max_length=output_max_length,
  temperature=output_temperature,
  pad_token_id=tokenizer.eos_token_id
)
end_time = timeit.default_timer()
execution_time = end_time - start_time
times['generating_output'] = execution_time
output_token_count = len(output[0])
output_tokens_per_second = output_token_count / execution_time
output_tokens_per_minute = output_tokens_per_second * 60
print_message(f'Output generated ({output_token_count} tokens) in {execution_time} seconds\n', prefix='~')

#
# Decode output
#
print_message('Decoding output', padding=1)
start_time = timeit.default_timer()
completed_text = tokenizer.decode(output[0], skip_special_tokens=True)
end_time = timeit.default_timer()
execution_time = end_time - start_time
times['decoding_output'] = execution_time
print_message(f'Output decoded in {execution_time} seconds\n', prefix='~')

#
# Determine total execution time
#
end_time_total = timeit.default_timer()
execution_time_total = end_time_total - start_time_init

#
# Print Results
#
print_message('Results', padding=1)
print_message(f'model_name: {model_name}', prefix='~')
print_message(f'tokenizer_name: {tokenizer_name}', prefix='~')

for key, value in times.items():
    print_message(f'{key}: {value} s', prefix='~')

print_message(f'execution_time_total: {execution_time_total}', prefix='~')
print_message(f'input_token_count: {input_token_count}', prefix='~')
print_message(f'output_token_count: {output_token_count}', prefix='~')
print_message(f'total_token_count: {input_token_count + output_token_count}', prefix='~')
print_message(f'output_max_length: {output_max_length}', prefix='~')
print_message(f'output_temperature: {output_temperature}', prefix='~')
print_message(f'output_tokens_per_second: {output_tokens_per_second}', prefix='~')
print_message(f'output_tokens_per_minute: {output_tokens_per_minute}', prefix='~')

print('\n')
print_message(f'Input ({input_token_count} tokens)', padding=1)
print_message(input, padding=1, prefix='---', skip_message_prefix=True)

print('\n')
print_message(f'Output ({output_token_count} tokens)', padding=1)
print_message(completed_text, padding=1, prefix='---', skip_message_prefix=True)