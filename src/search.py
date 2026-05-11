#env
from dotenv import load_dotenv
#embeddings from OpenAI
from openai import OpenAI
#numpy
import numpy as np

# OpenAI embeddings API
load_dotenv()
client = OpenAI()

# Load names & vectors from the embeddings.npz file that was generated
data = np.load("embeddings.npz", allow_pickle=False)
names = data["names"]
vectors = data["vectors"]


def search(query: str, top_k: int = 3) -> list[dict]:
    """Return top_k specialties ranked by cosine similarity to the query."""
    # 1. Embed query
    query_response = client.embeddings.create(
        model = "text-embedding-3-small",
        input = query
    )
    query_vector = np.array(query_response.data[0].embedding)

    # 2. Compute scores
    # vectors has shape (40, 1536) and query_vector has shape (1536,)
    # So we use np's @ to multiply across all vectors and compute dot products.
    # This will give us scores. This is a dot product answer. OpenAI's embeddings are
    # already normalized to magnitude=1, so this is the equivalent of cosine similarity.
    scores = vectors @ query_vector
    #print ("Scores shape should be (40,). It is: ", scores.shape)

    # 3. Get top-k indices
    # argsort returns indices that would sort the array in ASCENDING order
    # sorted_indices = np.argsort(scores)

    # To get descending (highest first), reverse with [::-1]
    top_k_indices = np.argsort(scores)[::-1][:top_k]

    # 4. Return [{"name": ..., "score": ...}, ...]
    results = []
    for i in top_k_indices:
        #results.append({"name": names[i], "score": float(scores[i])})
        results.append({"name": str(names[i]), "score": float(scores[i])})
    return results


if __name__ == "__main__":
    # only runs when this file is executed directly
    # safe to import this module without triggering this block

    # Grab query from user input
    query = input("Search for a type of specialty. Use specialty name or symptoms or any natural language query: ")
    print(search(query))
