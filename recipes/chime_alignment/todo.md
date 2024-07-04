# chime todo

- [x] fix qformer arguments
    - [x] rename
    - [x] add prefix and suffix
- [x] fix speech aligned collator
    - [x] add support for prompt suffix
    - [x] rename for all other files

- [ ] make the bos token available in the model directly
- [x] fix the bos token for the prefixes and suffixes


- [ ] finish the trainer
    - [x] add llama initialization
    - [x] bitsandbytes
    - [ ] clean the generation config stuff


- build the recipe
- RUN!



## QUESTIONS
- what should the labels look like?
    - check older scripts...

- set labels prior to the generated text to -100?


## PROBLEMS
- labels and decoder input ids
- generation doesn't work due to the mismatched types. Possible solutions:
    - workaround: just don't use predict with generate
    - potential solution: get everything up-to-date
- it seems stupid that there are so many embeddigns (~300) in the connector output when most of that is actually silence... Perhaps the Q-Former really has some merit here?




xxxxxxxxxxabcdx
assistant: abcd
