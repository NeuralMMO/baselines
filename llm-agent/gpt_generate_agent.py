from pdb import set_trace as T
import os

import openai


# Set these with export OPENAI_ORGANIZATION=your_key etc. in ~/.bashrc
openai.organization = os.environment['OPENAI_ORGANIZATION']
openai.api_key = os.environment9['OPENAI_API_KEY']

prompt = '\n\n'.join(open(f).read() for f in [
    'prompt_generate_agent.txt',
    'prompt_documentation_summary.txt',
    'prompt_example_code.py',
])

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0,
)

content = response["choices"][0]["message"]["content"]
print(content)

with open('generated_agent.py', 'w') as f:
    f.write(content)