import os
from dotenv import load_dotenv
from openai import AzureOpenAI

def main():
    load_dotenv()

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip().rstrip("/")
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "").strip()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "").strip()

    print("[DEBUG] endpoint   =", endpoint)
    print("[DEBUG] api_version=", api_version)
    print("[DEBUG] deployment =", deployment)
    print("[DEBUG] key_prefix =", api_key[:8], "len=", len(api_key))

    if not endpoint or not api_key or not api_version or not deployment:
        raise RuntimeError("ENV 누락: ENDPOINT/KEY/API_VERSION/DEPLOYMENT 확인")

    client = AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )

    resp = client.chat.completions.create(
        model=deployment,
        messages=[{"role": "user", "content": "핑 테스트. 한 문장으로 답해줘."}],
        max_tokens=80,
        temperature=0.2,
    )

    print("[OK]", resp.choices[0].message.content)

if __name__ == "__main__":
    main()
