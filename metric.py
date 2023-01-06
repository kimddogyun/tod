import json
import os.path
# from utils.eval_metrics import moses_multi_bleu
import glob as glob
import numpy as np
import jsonlines
from tabulate import tabulate
import re
from tqdm import tqdm
import string
import pprint
import argparse
from utils.eval_metric_bleu import moses_multi_bleu

parser = argparse.ArgumentParser()
args = parser.parse_args() 

# def setup_args():
#     """Setup arguments."""
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--dataset", type=str, choices=["SMD", "CamRest", "MultiWOZ"], required=True)
#     parser.add_argument("--pred_file", type=str, required=True)
#     parser.add_argument("--entity_file", type=str, required=True)
#     parser.add_argument("--save_file", type=str, required=True)
#     args = parser.parse_args()
#     return args


def hasNoNumbers(inputString):
  return not any(char.isdigit() for char in inputString)

def checker_global_ent(e,gold):
    nonumber = hasNoNumbers(e)
    sub_string = True
    for g in gold:
        if (e.lower() in g.lower()):
            sub_string = False
    return sub_string and nonumber

def substringSieve(string_list):
    string_list.sort(key=lambda s: len(s), reverse=True)
    out = []
    for s in string_list:
        if not any([s in o for o in out]):
            out.append(s)
    return out

def compute_prf(pred, gold, global_entity_list):
    TP, FP, FN = 0, 0, 0
    if len(gold) != 0:
        count = 1
        for g in gold:
            if g.lower() in pred.lower():
                TP += 1
            else:
                FN += 1
        list_FP = []

        for e in list(set(global_entity_list)):
            if e.lower() in pred.lower() and checker_global_ent(e,gold):
                if (e.lower() not in gold):
                    list_FP.append(e)
        FP = len(list(set(substringSieve(list_FP))))
        precision = TP / float(TP+FP) if (TP+FP) != 0 else 0
        recall = TP / float(TP+FN) if (TP+FN) != 0 else 0
        F1 = 2 * precision * recall / float(precision+recall) if (precision+recall)!=0 else 0

    else:
        precision, recall, F1, count = 0,0,0,0

    return F1, count



def get_splits(data,test_split,val_split):
    train = {}
    valid = {}
    test  = {}
    for k, v in data.items():
        if(k in test_split):
            test[k] = v
        elif(k in val_split):
            valid[k] = v
        else:
            train[k] = v
    return train, valid, test


def get_global_entity_MWOZ():
    with open('data/ontology.json') as f:
        global_entity = json.load(f)
        global_entity_list = []
        for key in global_entity.keys():
            if (key not in["restaurant-book-people","hotel-semi-stars","hotel-book-stay","hotel-book-people"]): # no train-book-people", q-tod
                global_entity_list += global_entity[key]

            global_entity_list = list(set(global_entity_list))

        return global_entity_list

with open('data/ontology.json') as f:
    global_entity = json.load(f)
    ontology = {"attraction":{},"hotel":{},"restaurant":{}}
    for key in global_entity.keys():
        domain, _, slot = key.split("-")
        if (domain in ontology):
            ontology[domain][slot] = global_entity[key]

# Evaluate - BM25

dialogue_mwoz = json.load(open("data/MultiWOZ_2.1/data.json"))
test_split = open("data/MultiWOZ_2.1/testListFile.txt","r").read()
val_split = open("data/MultiWOZ_2.1/valListFile.txt","r").read()
split_by_single_and_domain = json.load(open("data/dialogue_by_domain.json"))
_,_,gold_json = get_splits(dialogue_mwoz,test_split,val_split)
# gold_json = get_dialog_single(gold_json,split_by_single_and_domain)

test = json.load(open(f"data/test.json")).items()
entity_KB = get_global_entity_MWOZ()


def preprocess_text(text):
    """Preprocess utterance and table value."""
    text = text.strip().replace("\t", " ").lower()
    for p in string.punctuation:
        text = text.replace(p, f" {p} ")
    text = " ".join(text.split())
    return text

def score_MWOZ(mdoel,file_to_score):
    pred_json = json.load(open(file_to_score))
    print(f"Load prediction file from: {file_to_score}")
    
    preds = []
    refs = []

    for dial in pred_json:
        for turn in dial["dialogue"]:
            if turn["turn"] == "system":
                preds.append(preprocess_text(turn["generated_response"]))
            else:
                preds.append(turn["generated_response"])
            refs.append(turn["utterance"])
    assert len(preds) == len(refs), f"{len(preds)} != {len(refs)}"

    bleu_metric = moses_multi_bleu(np.array(preds), np.array(refs), lowercase=True)
    # print(bleu_metric)
    entity_metric = EntityMetric(args)
    
    bleu_res = bleu_metric
    entity_res = entity_metric.evaluate(preds, refs)
    results = {
        "BLEU": bleu_res,
        "Entity-F1": entity_res
    }

    print(json.dumps(results, indent=2))
    with open(args.save_file, "w") as fout:
        json.dump(results, fout, indent=2)
    return 

class EntityMetric(object):
    def __init__(self,args) -> None:
        self.entities = self.entity_KB
        
    def evaluate(self, preds, refs):
        extracted_preds_entities = []
        extracted_refs_entities =[]

        for pred, ref in zip(preds, refs):
            pred_entities = self._extract_entities(pred)
            ref_entities = self._extract_entities(ref)
            extracted_preds_entities.append(pred_entities)
            extracted_refs_entities.append(ref_entities)
        entity_f1 = compute_prf(extracted_preds_entities,extracted_refs_entities)
        return entity_f1

    def _extract_entities(self, response):

        def _is_sub_str(str_list, sub_str):
            for str_item in str_list:
                if sub_str in str_item:
                    return True
            return False

        response = f"{response}".lower()
        extracted_entities = []

        for entity in self.entities:
            if self.dataset == "MultiWOZ":
                success_tag = False
                if entity.startswith("choice-"):
                    entity = entity[7:]
                    if entity == "many":
                        if entity in re.sub(r"(many (other types|food types|cuisines)|how many)", " ", response):
                            success_tag = True
                    elif entity == "all":
                        if re.search(r"all (of the|expensive|moderate|cheap)", response):
                            success_tag = True
                    elif entity == "to":
                        success_tag = False
                    else:
                        if re.search(f"(there are|there is|found|have about|have)( only|) {entity}", response):
                            success_tag = True
                elif entity == "centre":
                    if entity in response.replace("cambridge towninfo centre", " "):
                        success_tag = True
                elif entity == "free":
                    if re.search(r"free (parking|internet|wifi)", response):
                        success_tag = True
                elif entity in response or entity.lower() in response.lower():
                    success_tag = True

                if success_tag:
                    extracted_entities.append(entity)
                    response = response.replace(entity, " ")

            else:
                if entity in response and not _is_sub_str(extracted_entities, entity):
                    extracted_entities.append(entity)

        return extracted_entities