# Alignment recipes for How2

Here is a list of recipes, that are up to date with the experiments presented in the thesis and should work as good starting points for your own experiments:
- `train_ecd.sh` -- main general recipe for training the ECD alignment architecture
- `train_eced.sh` -- main general recipe for training the ECED alignment architecture
- `train_eced_low_resource.sh` -- used for fine-tuning of the ECED architecture on low resource How2 splits defined by text files containing lists of utterance IDs
- `train_marian_mt.sh`-- main recipe for training MarianMT model, also used for the English identity re-training 
- `train_ebr_asr.sh` -- main recipe for training the E-Branchformer asr model
- `train_t5_mt.sh` -- recipe for re-training of the T5 decoder for identity
- `train_eced_t5_asr_pretrain.sh` -- recipe for pre-training the ECED/T5 model for English ASR

