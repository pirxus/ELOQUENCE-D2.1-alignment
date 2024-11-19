from typing import Dict, List

import numpy as np
import json
import torch
from jiwer import cer, compute_measures
from transformers import PreTrainedTokenizer, WhisperTokenizer
from transformers.trainer_utils import PredictionOutput
from utilities.english_normalizer import EnglishNormalizer

import os
import csv

os.environ["HF_EVALUATE_OFFLINE"] = "1"
#from datasets import load_metric
import evaluate
from nltk.translate.bleu_score import corpus_bleu

import wandb

SACREBLEU = True
if SACREBLEU:
    bleu = evaluate.load("sacrebleu")
else:
    bleu = evaluate.load("bleu")

def write_wandb_pred(pred_str: List[str], label_str: List[str], rows_to_log: int = 10):
    current_step = wandb.run.step
    columns = ["id", "label_str", "hyp_str"]
    wandb.log(
        {
            f"eval_predictions/step_{int(current_step)}": wandb.Table(
                columns=columns,
                data=[
                    [i, ref, hyp] for i, hyp, ref in zip(range(min(len(pred_str), rows_to_log)), pred_str, label_str)
                ],
            )
        },
        current_step,
    )


def get_metrics(labels: List[str], preds: List[str]):
    metrics = compute_measures(labels, preds)
    del metrics["ops"]
    del metrics["truth"]
    del metrics["hypothesis"]
    return {"cer": cer(labels, preds), **metrics}


def get_most_likely_tokens(logits: torch.Tensor, _: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def compute_metrics_ctc(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    pred.predictions[pred.predictions == -100] = tokenizer.pad_token_id
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return get_metrics(label_str, pred_str)


def compute_metrics(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    normalizer = EnglishNormalizer()
    pred_str = [ normalizer(s).strip() for s in pred_str ]
    label_str = [ normalizer(s).strip() for s in label_str ]

    return get_metrics(label_str, pred_str)

def compute_metrics_slurp(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10,
    use_slots=False, dump_pred=False
) -> Dict[str, float]:
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        if dump_pred:
            write_wandb_pred(pred_str, label_str, rows_to_log=len(label_str))
        else:
            write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    # first, parse out the transcript, the scenario and action from the labels
    label_dicts = []
    for label in label_str:
        try:
            label_dicts.append(json.loads('{"transcript": ' + label))
        except:
            print(label)
            raise ValueError
        
    #label_dicts = [ json.loads('{"transcript": ' + label) for label in label_str ]

    action_cnt = 0
    action_correct = 0
    scenario_cnt = 0
    scenario_correct = 0
    intent_cnt = 0
    intent_correct = 0
    json_errors = 0

    slot_cnt = 0
    slot_correct = 0

    transcript_preds = []
    transcript_labels = []

    for i in range(len(label_dicts)):
        label_dict = label_dicts[i]
        scenario_ok = False
        action_ok = False
        json_error = False

        # These should be here
        scenario_cnt += 1
        action_cnt += 1
        intent_cnt += 1

        try:
            pred_dict = json.loads('{"transcript": ' + pred_str[i])
        except:
            json_error = True
            print("Error parsing prediction")
            print(pred_str[i])

            label_slots = label_dict['slots'].items()
            label_slots_len = len(label_slots)
            slot_cnt += max(label_slots_len, 1)
            continue

        try:
            if label_dict['scenario'] == pred_dict['scenario']:
                scenario_correct += 1
            scenario_ok = True
        except:
            json_error = True
            print("Error: scenario key missing")
            print("Label: ", label_dict)
            print("Prediction: ", pred_dict)

        try:
            if label_dict['action'] == pred_dict['action']:
                action_correct += 1
            action_ok = True
        except:
            json_error = True
            print("Error: action key missing")
            print("Label: ", label_dict)
            print("Prediction: ", pred_dict)

        if scenario_ok and action_ok:
            if label_dict['action'] == pred_dict['action'] and label_dict['scenario'] == pred_dict['scenario']:
                intent_correct += 1

        # slots
        if use_slots:
            label_slots = label_dict['slots'].items()
            label_slots_len = len(label_slots)
            slot_cnt += max(label_slots_len, 1)

            if 'slots' not in pred_dict.keys():
                json_error = True
                print("Error: slots missing")
                print("Label: ", label_dict)
                print("Prediction: ", pred_dict)

            else:
                pred_slots = pred_dict['slots']
                # slot should be empty
                if label_slots_len == 0 and label_slots_len == len(pred_slots.items()):
                    slot_correct += 1

                else:
                    # determine the correctness of each slot
                    for slot_key, slot_val in label_slots:
                        if slot_key in pred_slots.keys() and slot_val == pred_slots[slot_key]:
                            slot_correct += 1

        try:
            transcript_preds.append(pred_dict['transcript'])
            transcript_labels.append(label_dict['transcript'])
        except:
            json_error = True
            print("Error: transcript key missing")
            print("Label: ", label_dict)
            print("Prediction: ", pred_dict)

        if json_error:
            json_errors += 1


    normalizer = EnglishNormalizer()
    transcript_preds = [ normalizer(s).strip() for s in transcript_preds ]
    transcript_labels = [ normalizer(s).strip() for s in transcript_labels ]

    metrics = get_metrics(transcript_labels, transcript_preds)

    metrics['action_acc'] = action_correct / max(1, action_cnt)
    metrics['scenario_acc'] = scenario_correct / max(1, scenario_cnt)
    metrics['intent_acc'] = intent_correct / max(1, intent_cnt)
    metrics['json_errors'] = json_errors
    if use_slots:
        metrics['slot_acc'] = slot_correct / max(1, slot_cnt)

    return metrics

def compute_metrics_fisher_turns(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10, remove_spk_tags: bool = True) -> Dict[str, float]:
    label_ids = pred.label_ids
    pred_ids = pred.predictions

    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    # remove the speaker tags
    if remove_spk_tags:
        label_str = [ label.replace('A: ', '').replace('B: ', '') for label in label_str ]
        pred_str = [ pred.replace('A: ', '').replace('B: ', '') for pred in pred_str ]

    normalizer = EnglishNormalizer()
    pred_str = [ normalizer(s).strip() for s in pred_str ]
    label_str = [ normalizer(s).strip() for s in label_str ]

    return get_metrics(label_str, pred_str)

def compute_metrics_translation(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    """Compute metrics for MT and ST"""
    pred_ids = pred.predictions

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id
    pred_ids[pred_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    metrics = get_metrics(label_str, pred_str)

    bleu_result = bleu.compute(predictions=pred_str, references=label_str)

    if SACREBLEU:
        metrics['bleu'] = bleu_result['score']
    else:
        metrics['bleu'] = bleu_result['bleu']

    print("Bleu on the eval set:", bleu_result)
    return metrics
