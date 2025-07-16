import openai

openai.api_key = "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"  # Replace with your actual key

prompt = """
Generate 200 variants for “emergency light faulty” in lowercase, spelt into various phonetic variants, with permutations for usage in natural speech. 

The usage for this is for training ASR models powering hotel concierge bots, and automating guest service request tagging. Make sure the overall semantic meaning remains the same. Do not repeat any variants in their entirety and in substrings.
"""

response = openai.chat.completions.create(
    model="gpt-3.5-turbo",  # Use a model you have access to
    messages=[
        {"role": "system", "content": "You are a helpful assistant and a Natural Language Processing expert trying to create a list of fault variants for hotel guest service requests."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.6,  # Controls creativity; lower = more deterministic
    max_tokens=3000   # Adjust as needed to handle long output
)

print(response.choices[0].message.content)