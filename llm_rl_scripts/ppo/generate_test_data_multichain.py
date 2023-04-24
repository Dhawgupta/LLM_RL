import tyro
from LLM_RL.utils import convert_path
from typing import Any, Dict, List
import json

def get_examples(n: int) -> List[Dict[str, Any]]:
    examples = []
    for i in range(2**n):
        curr1 = i
        bits = []
        for _ in range(n):
            bits.append(curr1 % 2)
            curr1 = curr1 // 2
        bits.reverse()

        for z in range(2**n):
            curr2 = z
            signs = []
            for _ in range(n):
                signs.append(2*(curr2 % 2) - 1)
                curr2 = curr2 // 2
            signs.reverse()

            cum_sum = 0
            for i, (bit, sign) in enumerate(zip(bits, signs)):
                cum_sum += bit * sign
                examples.append({
                    "in_text": f"total: {cum_sum}, sign: {sign}, turn: {i}\n", 
                    "out_text": f" {bit}\n", 
                    "reward": bit * sign, 
                })
            examples[-1]['reward'] = float(cum_sum < -(n // 2))*(n**2)
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
