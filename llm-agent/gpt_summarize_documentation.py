import openai
import os


# Set these with export OPENAI_ORGANIZATION=your_key etc. in ~/.bashrc
openai.organization = os.environment['OPENAI_ORGANIZATION']
openai.api_key = os.environment9['OPENAI_API_KEY']

# Prompt uses documentation minus the narrative to fit in 4097 context
prompt = '\n\n'.join(open(f).read() for f in [
    'prompt_summarize_documentation.txt',
    'prompt_documentation.txt',
])

response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=[
        {"role": "user", "content": prompt},
    ],
    temperature=0,
)

content = response["choices"][0]["message"]["content"]

with open('prompt_documentation_summary.txt', 'w') as f:
    f.write(content)

print(content)