from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Install bitsandbytes, accelerate, and transformers to use this code
# Note that we have not been able to get usable scripts out of this model,
# but you may have better luck

# 4 bit quantization
nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the tokenizer and model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = AutoTokenizer.from_pretrained("cerebras/btlm-3b-8k-base")
model = AutoModelForCausalLM.from_pretrained("cerebras/btlm-3b-8k-base", trust_remote_code=True, device_map='auto', quantization_config=nf4_config)

# Set the prompt for generating text
prompt = '\n\n'.join(open(f).read() for f in [
    'script_prompt.txt',
    'documentation_summary.txt',
    'example_code.py',
])

# Tokenize the prompt and convert to PyTorch tensors
inputs = tokenizer(prompt[:10], return_tensors="pt").to(device)

print('Generating script...')

# Generate text using the model
outputs = model.generate(
    **inputs,
    #num_beams=5,
    max_new_tokens=50,
    early_stopping=True,
    no_repeat_ngram_size=2
)

# Convert the generated token IDs back to text
generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)