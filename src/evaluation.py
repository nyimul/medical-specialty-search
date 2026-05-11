import json
from search import search

with open("data/eval_set.json") as f:
    eval_set = json.load(f)

passed_top1 = 0
passed_top3 = 0

for item in eval_set:
    results = search(item["query"], top_k=3)
    top_names = [r["name"] for r in results]

    if item["expected"] == top_names[0]:
        passed_top1 += 1
    if item["expected"] in top_names:
        passed_top3 += 1
    else:
        print(f"FAIL: '{item['query']}'")
        print(f"  Expected: {item['expected']}")
        print(f"  Got:      {top_names}\n")

total = len(eval_set)
print(f"\nTop-1: {passed_top1}/{total} ({100*passed_top1/total:.0f}%)")
print(f"Top-3: {passed_top3}/{total} ({100*passed_top3/total:.0f}%)")
