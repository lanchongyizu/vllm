from openai import OpenAI
from vllm.utils import FlexibleArgumentParser

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
def main(args):
    stream = True
    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "Magik Compute is a platform dedicated to shaping the future of artificial intelligence. We focus on providing AI-as-a-Service (AIaaS) and Artificial General Intelligence (AGI) infrastructure. We're seeking the world's top talent to join our mission and shape the leading edge of AI."
    completion = client.completions.create(
        model=model,
        prompt=prompt,
        echo=False,
        n=1,
        stream=stream,
        max_tokens=512,
        )

    print("Completion results:")
    if stream:
        print(prompt)
        for chunk in completion:
            if chunk.choices[0].text:
                print(chunk.choices[0].text, end="")
        print()
    else:
        print(completion)

if __name__ == '__main__':
    parser = FlexibleArgumentParser(
        description='Demo on using LLMEngine directly')
    parser.add_argument("--prompt",
                        type=str,
                        help="prompt string")
    args = parser.parse_args()
    main(args)
