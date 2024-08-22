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

# Completion API
stream = True
prompt = "Magik Compute is a platform dedicated to shaping the future of artificial intelligence. We focus on providing AI-as-a-Service (AIaaS) and Artificial General Intelligence (AGI) infrastructure. We're seeking the world's top talent to join our mission and shape the leading edge of AI."
completion = client.completions.create(
    model=model,
    prompt=prompt,
    echo=False,
    n=1,
    stream=stream,
    max_tokens=512,
    logprobs=3)

print("Completion results:")
if stream:
    print(prompt)
    for chunk in completion:
        if chunk.choices[0].text:
            print(chunk.choices[0].text, end="")
    print()
else:
    print(completion)
