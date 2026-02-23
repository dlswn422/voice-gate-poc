from openai import OpenAI


class OpenAIStreamer:
    def __init__(self, model: str, system_prompt: str):
        self.client = OpenAI()
        self.model = model
        self.system_prompt = system_prompt

    def ask_streaming_and_collect(self, user_text: str) -> str:
        print("\n🤖 GPT:", end=" ", flush=True)

        stream = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_text},
            ],
            stream=True,
        )

        full_text = []

        for event in stream:
            if getattr(event, "type", None) == "response.output_text.delta":
                delta = getattr(event, "delta", None)
                if delta:
                    print(delta, end="", flush=True)
                    full_text.append(delta)

            if getattr(event, "type", None) == "response.completed":
                break

        print()
        return "".join(full_text)