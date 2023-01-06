import argparse
import json
import os
import string

import torch
from transformers import (T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, T5ForConditionalGeneration)
from tqdm import tqdm, trange


def infer(args):
    with open(args.infer_file, "r") as f:
        data = json.load(f)
        print(f"Load inference file from: {args.infer_file}")

    tokenizer = T5Tokenizer.from_pretrained(args.model_path)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model.eval()
    model.to("gpu:0")
    # print(f"Load model from: {args.model_path}")

    query_generation(data,tokenizer, model, args.batch_size, args.beam_size)

    if not os.path.exists(os.path.dirname(args.save_file)):
        os.makedirs(os.path.dirname(args.save_file))
    with open(args.save_file, "w") as fout:
        json.dump(data, fout, indent=2)
        print(f"Save inference output to: {args.save_file}")
    return 

def query_generation(data, tokenizer, model, batch_size, beam_size):
    samples = []

    for dial in data:
        context = []
        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                src = "translate dialogue context to query : " + " | ".join(context)
            utt = preprocess_text(turn["utterance"])
            context.append(utt)

