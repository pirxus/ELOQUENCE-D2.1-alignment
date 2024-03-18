from typing import Dict, List

import numpy as np
from jiwer import cer, compute_measures
from transformers import PreTrainedTokenizer
from transformers.trainer_utils import PredictionOutput

import os

os.environ["HF_EVALUATE_OFFLINE"] = "1"
#from datasets import load_metric
import evaluate
from nltk.translate.bleu_score import corpus_bleu

import wandb

SACREBLEU = False
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


def compute_metrics_ctc(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    pred.label_ids[pred.label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(pred.label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return get_metrics(label_str, pred_str)


def compute_metrics(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    pred_ids = pred.predictions

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = [label if label else "-" for label in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

    if wandb.run is not None:
        write_wandb_pred(pred_str, label_str, rows_to_log=wandb_pred_to_save)

    return get_metrics(label_str, pred_str)

def compute_metrics_translation(
    tokenizer: PreTrainedTokenizer, pred: PredictionOutput, wandb_pred_to_save: int = 10
) -> Dict[str, float]:
    pred_ids = pred.predictions

    label_ids = pred.label_ids
    label_ids[label_ids == -100] = tokenizer.pad_token_id

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
