from evaluate import load
from safe_gpu import safe_gpu
safe_gpu.claim_gpus()

MODEL = 'allenai/OLMo-1B-hf'

files = [
    #'/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_wsm_olmo1b_stte_lp0.5/predictions_how2_val_wer10.65_hyp.trn',
    #'/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_wsm_olmo1b_stte_lp0.5/predictions_how2_val_wer10.65_ref.trn',
    #'/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_wsm_olmo1b_stte_lp0.5/predictions_how2_dev5_wer10.50_hyp.trn',
    #'/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_wsm_olmo1b_stte_lp0.5/predictions_how2_dev5_wer10.50_ref.trn',
    #'/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_whisper_how2_en_normalized_beam1/predictions_val_wer9.69_hyp.trn',
    #'/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_whisper_how2_en_normalized_beam1/predictions_dev5_wer7.90_hyp.trn',
    #'/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_whisper_how2_en_normalized_beam2/predictions_val_wer10.73_hyp.trn',
    '/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_wsm_olmo1b_stte/predictions_how2_dev5_wer10.94_hyp.trn',
    '/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/eval_wsm_olmo1b_stte/predictions_how2_val_wer11.18_hyp.trn',
]

OUT_DIR = '/mnt/matylda6/xsedla1h/projects/huggingface_asr/exp/ppl_test'

HAS_HEADER = False

perplexity = load('perplexity', module_type='metric')

for f in files:
    with open(f, 'r') as ff:
        lines = [ line.rstrip().lstrip() for line in ff ]

    if HAS_HEADER:
        lines = lines[1:]

    lines = [ line.rpartition(' ')[0] for line in lines ]

    # clean empty lines
    lines = list(filter(lambda x: len(x) > 1, lines))

    results = perplexity.compute(model_id=MODEL, predictions=lines, batch_size=32, add_start_token=False)

    print("Perplexity for file:")
    print(f)
    print(results['mean_perplexity'])
