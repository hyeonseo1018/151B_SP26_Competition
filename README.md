# CSE 151B Competition - Hyeonseo An


## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.py` | Main excutable file |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

## Environment


GPU type :  A30 (2~3 hour approximate generation time with vLLM)
Run starter_code_cse151b_comp.py to generate.  

Tested with:

```text
antlr4-python3-runtime==4.11.1
bitsandbytes==0.49.2
numpy==2.2.6
sympy==1.13.1
torch==2.6.0
torchaudio==2.6.0
torchvision==0.21.0
tqdm==4.67.3
transformers==4.51.3
vllm==0.8.5.post1
```
