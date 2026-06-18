from openai import OpenAI

base_url = input("Enter base_url (e.g. https://api.openai.com/v1): ").strip()
api_key = input("Enter api_key: ").strip()

client = OpenAI(api_key=api_key, base_url=base_url)
models = client.models.list()

print(f"\nAvailable models ({len(models.data)}):")
for m in models.data:
    print(f"  - {m.id}")
