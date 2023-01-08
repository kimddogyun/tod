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

    generated_queries = generate(samples, tokenizer, model, batch_size, beam_size)

    sample_idx = 0
    for dial in data:
        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                turn["generated_query"] = generated_queries[sample_idx]
                sample_idx += 1

    return

def preprocess_text(text):
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text=  " ".join(text.split())
    return text


def generate(samples, tokenizer, model, batch_size, beam_size):
    outputs = []
    for idx in trange(0, len(samples), batch_size, desc="Generation")
        batch = samples[idx:idx+batch_size]
        tokenized_batch = tokenizer(batch,max_length=1024, padding=True, truncation=True, return_tensors="pt")

        batch_out, _ = model.generate(
            tokenized_batch["input_ids"],
            decode_strategy="beam search",
            num_beams=beam_size,
            max_length=128,
            length_penalty=1,
            attention_mask = tokenized_batch.get("attention_mask")
        )

        # batch_pred = tokenizer.batch_decode(batch_out, skip_special_tokens=True)

        outputs.extend(batch_pred)

    return outputs

