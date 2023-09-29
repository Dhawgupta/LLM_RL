import tyro
from LLM_RL.utils import convert_path
from typing import Any, Dict, List
import json

def get_examples(n: int) -> List[Dict[str, Any]]:
    examples = []
    for i in range(2**n):
        curr = i
        bits = []
        for x in range(n):
            bits.append(curr % 2)
            curr = curr // 2
        bits.reverse()
        examples.append({
            "in_text": "", 
            "out_text": " "+' '.join(map(str, bits))+"\n", 
            "reward": float(sum(bits) > (n // 2))*10.0, 
        })
    return examples

def main(
    n: int, 
    output_path: str, 
):
    examples = get_examples(n)
    with open(convert_path(output_path), "w") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")

if __name__ == "__main__":
    tyro.cli(main)
