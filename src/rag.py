import json
from search import search
#env
from dotenv import load_dotenv
#embeddings from OpenAI
from openai import OpenAI

# Call OpenAI embeddings API
load_dotenv()
client = OpenAI() # grabs api key from env

with open("data/specialties.json", "r") as f:
    specialties = json.load(f)
specialty_lookup = {s["name"]: s["description"] for s in specialties}

def explain(query: str, top_k: int = 3) -> str:
    # 1. Call search() to get top-k specialties
    top_k_specialties = search(query, top_k)
    #return str(len(top_k_specialties))
    # top_k_specialties looks like:
    # [{'name': 'Dermatology', 'score': 0.341224294001222}, {'name': 'Allergy and Immunology', 'score': 0.22140223810060888}, {'name': 'Hepatology', 'score': 0.1809635413619617}]

    # 2. Build a prompt that includes the query + the retrieved descriptions
    
    # Retrieve the descriptions for these top k specialties. Use the lookup from above.
    top_names = [r["name"] for r in top_k_specialties]
    top_descriptions = [specialty_lookup[name] for name in top_names]
    top_names_with_their_descriptions = [
        f"{name}: {desc}"
        for name, desc in zip(top_names, top_descriptions)
    ]
    specialty_blocks = "\n\n".join(top_names_with_their_descriptions)

    system_prompt = f"""
    You are a medical assistant. Based on the following specialty information, explain which specialty would be most appropriate for the user's query.
    If none of the listed specialties are clearly appropriate for the user's query, say so explicitly rather than guessing.
    """

    user_prompt = f"""
    Relevant specialties:
    {specialty_blocks}

    User's query: {query}
    """

    # 3. Call client.chat.completions.create(...) with that prompt
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 4. Return the generated text
    if response.choices[0].message.content is not None:
        return response.choices[0].message.content
    else:
        return "Error"

if __name__ == "__main__":
    query = input("Search for a type of specialty. Use specialty name or symptoms or any natural language query: ")
    print(explain(query, 3))
