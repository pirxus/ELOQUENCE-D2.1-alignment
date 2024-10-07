# CHIME TODO
- [ ] make the bos token available in the model directly

- [ ] finish the trainer
    - [x] add llama initialization
    - [x] bitsandbytes
    - [ ] clean the generation config stuff

- [ ] refactor the alignment module to support different subsampling types and transformations

## Training
- [ ] replicate the original paper's training setup
    - same constant lr schedule
    - no quantization
    - exact same prompt
    - stacked downsampling factor

- [ ] try with no prompt, just the embeddings and <bos>
- [ ] update the environment to work with OLMo


- [ ] determine the best architecture for the alignment
    - [ ] determine whisper-small + tinyllama memory requirements for training
        - 1024:16:2048, batch size 12: ~19G
        - 2048:16:4096, batch size 12: ~20G
        - 2048:32:4096, batch size 12: ~21G, 80m params
        - 1024:16:4096, batch size 12: ~19G, 31m params
    - [ ] determine whisper-small + llama2-chat memory requirements for training
        - 1024:16:4096, batch size  1: ~18G, 33m params
        - =====||=====, batch size  2: ~21G, 33m params
        - 1280:20:5120, batch size  2: ~22G, 50m params

    - [ ] test stte vs ste



- [ ] train whisper small + best connector + llama2 chat
- [ ] finetune the connector on notsofar data
- [ ] do inference on gss, get results

- [ ] repeat for whisper large v3 + llama chat
    - [ ] determine whisper-large-v3 + llama2-chat memory requirements for training
        - 1024:16:4096, batch size ??:  TODO
        - 1280:20:5120, batch size ??:  TODO
        - 2048:32:4096, batch size ??:  TODO


# Eval
stlin lc: 1444 empty line and dots

## Questions

## Problems
- it seems stupid that there are so many embeddigns (~300) in the connector output when most of that is actually silence... Perhaps the Q-Former really has some merit here?
