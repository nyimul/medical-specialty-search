#env
from dotenv import load_dotenv
#embeddings from OpenAI
from openai import OpenAI
#read json
import json
#numpy
import numpy as np

# Call OpenAI embeddings API
load_dotenv()
client = OpenAI() # grabs api key from env

with open("data/specialties.json", "r") as f:
    specialties = json.load(f)
#specialties is a list of 40 dicts, each with "name" and "description" keys
# e.g. {'name': 'Dermatology', 'description': 'Treats skin, hair, and nail conditions including acne, eczema, psoriasis, rashes, rosacea, and skin cancer. Handles concerns like itchy skin, suspicious moles, hair loss, and chronic breakouts. Performs procedures such as biopsies, mole removal, and cosmetic skin treatments.'}

# Save names & descriptions separately because the output of 
# your search has to be the specialty name, not the embedded blob
# texts is combination input to embeddings api
names = []
#descriptions = []
texts = []
for s in specialties:
    names.append(s["name"])
    #descriptions.append(s["description"])
    texts.append(s["name"] + ": " + s["description"])


response = client.embeddings.create(
    model="text-embedding-3-small",
    input=texts
)

# As many outputs as inputs. So if just one string, just [0]. But texts is size 40.
# This is batching. response.data[i].embedding corresponds to input[i].

names = np.array(names)
#vectors = np.zeros((40, 1536)) is a code smell, so instead we do this:
vectors = np.array([item.embedding for item in response.data])
for i in range(len(specialties)):
    vectors[i] = response.data[i].embedding

# 1. Confirm shape is (40, 1536)
print(vectors.shape)

# 2. Confirm names loaded correctly
print(names[:3])

# 3. Spot-check one vector — first 3 floats of Dermatology
print(vectors[0][:3])

# 4. Sanity check: no all-zero rows (would mean the loop didn't fill them)
print("Zero rows:", np.sum(np.all(vectors == 0, axis=1)))

# Save to file
np.savez("embeddings.npz", names=names, vectors=vectors)

# # Loading later:
# data = np.load("embeddings.npz", allow_pickle=False)
# names = data["names"]
# vectors = data["vectors"]