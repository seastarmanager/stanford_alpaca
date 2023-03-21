# -*- coding: utf-8 -*-
import torch
import transformers
from generation import LLaMA

model_name_or_path = '/data/models/llama_hf'
model_max_length = 512

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

prompts = [
# For these prompts, the expected answer is the natural continuation of the prompt
"I believe the meaning of life is",
"Simply put, the theory of relativity states that ",
"Building a website can be done in 10 simple steps:\n",
# Few shot prompts: https://huggingface.co/blog/few-shot-learning-gpt-neo-and-inference-api
"""Translate English to French:
sea otter => loutre de mer
peppermint => menthe poivrée
plush girafe => girafe peluche
cheese =>""",
    ]

def main():
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    torch.set_default_tensor_type(torch.FloatTensor)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name_or_path, device_map="auto"
    )
    model.eval()
    #model.cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        model_max_length=model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if "llama" in model_name_or_path:
        tokenizer.add_special_tokens(
            {
                "eos_token": DEFAULT_EOS_TOKEN,
                "bos_token": DEFAULT_BOS_TOKEN,
                "unk_token": DEFAULT_UNK_TOKEN,
            }
        )

    generator = LLaMA(model, tokenizer)

    results = generator.generate(
        prompts, max_gen_len=256, temperature=0.8, top_p=0.95
    )

    for result in results:
        print(result)
        print("\n==================================\n")



if __name__ == "__main__":
    main()
