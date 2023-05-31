import tyro
from LLM_RL.utils import convert_path
from typing import Any, Dict, List
import json

def get_examples(n: int) -> List[Dict[str, Any]]:
    examples = []
    for i in range(2**n):
        curr = i
        bits = []
        for _ in range(n):
            bits.append(curr % 2)
            curr = curr // 2
        bits.reverse()
        examples.append({
            "in_text": "", 
            "out_text": " "+' '.join(map(str, bits))+"\n", 
            "reward": float(sum(bits) > (n // 2))*10.0, 
        })
        for bool_response in [True, False]:
            for x in range(2**n):
                curr2 = x
                bits2 = []
                for _ in range(n):
                    bits2.append(curr2 % 2)
                    curr2 = curr2 // 2
                bits2.reverse()
                # do the same as the previous response if bool_response is True, otherwise do the opposite
                if (sum(bits) > (n // 2)) and (bool_response):
                    examples.append({
                        "in_text": " "+' '.join(map(str, bits))+"\n"+str(bool_response)+"\n", 
                        "out_text": " "+' '.join(map(str, bits2))+"\n", 
                        "reward": float(sum(bits2) > (n // 2))*10.0, 
                    })
                elif (sum(bits) > (n // 2)) and (not bool_response):
                    examples.append({
                        "in_text": " "+' '.join(map(str, bits))+"\n"+str(bool_response)+"\n", 
                        "out_text": " "+' '.join(map(str, bits2))+"\n", 
                        "reward": float(sum(bits) <= (n // 2))*10.0, 
                    })
                elif (sum(bits) <= (n // 2)) and (bool_response):
                    examples.append({
                        "in_text": " "+' '.join(map(str, bits))+"\n"+str(bool_response)+"\n", 
                        "out_text": " "+' '.join(map(str, bits2))+"\n", 
                        "reward": float(sum(bits) <= (n // 2))*10.0, 
                    })
                elif (sum(bits) <= (n // 2)) and (not bool_response):
                    examples.append({
                        "in_text": " "+' '.join(map(str, bits))+"\n"+str(bool_response)+"\n", 
                        "out_text": " "+' '.join(map(str, bits2))+"\n", 
                        "reward": float(sum(bits2) > (n // 2))*10.0, 
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
