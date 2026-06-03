from ollama import chat
from ollama import ChatResponse

response: ChatResponse = chat(model='huggingface.co/ggml-org/Qwen3-Omni-30B-A3B-Instruct-GGUF:latest', messages=[
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
])

print(response['message']['content'])