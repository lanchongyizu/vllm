from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key=openai_api_key,
    base_url=openai_api_base,
)

models = client.models.list()
model = models.data[0].id
history_openai_format = [{
    "role": "system",
    "content": "You are a great ai assistant."
}]
print(history_openai_format[0]["content"])

while True:
    print("Input:")
    message = input()
    history_openai_format.append({"role": "user", "content": message})

    chat_completion = client.chat.completions.create(
        messages=history_openai_format,
        model=model,
        stream=True
    )

    reply = ""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="")
            reply += chunk.choices[0].delta.content
    print()
    history_openai_format.append({"role": "assistant", "content": reply})
