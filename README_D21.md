# ELOQUENCE D2.1

## Getting started
TODO

## Installation
TODO

## Writing your own recipe
TODO

## Pre-trained models

Some of the key pre-trained models described in the deliverable are available in the ELOQUENCE WP2 [sharepoint](https://telefonicacorp.sharepoint.com/:f:/r/sites/ELOQUENCE.TMEHI/Shared%20Documents/WP2/1.%20Deliverables/models_T2.1?csf=1&web=1&e=ycICTb).

There are three archives available in the sharepoint, one for each pre-traininig phase, as described in the deliverable. For phase one, there are two models: FT\_NC and FR\_NC. For phase two, there are three models: two of them come from the FT\_NC checkpoint and the WavLM encoder is either fine-tuned or frozen, and the third model is initialized from the FR\_NC model. For phase three, we provide the single best model trained for joint ASR, user intent recognition, and dialogue slot filling on SLURP.

The individual checkpoints all contain the respective configuration files, which can be used to load the model for inference or further fine-tuning. More detailed information about how the models can be loaded and used for downstream tasks/experiments will be included here in the future. For now, it's best to refer to the recipes in ```recipes/eloquence/```.
