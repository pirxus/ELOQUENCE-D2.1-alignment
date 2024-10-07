import argparse
from datasets import load_from_disk, Dataset, concatenate_datasets, Audio
from datasets.utils.logging import disable_progress_bar, enable_progress_bar, set_verbosity_error
from tqdm import tqdm
from random import uniform
from copy import deepcopy
import numpy as np
import multiprocessing as mp
import os
from preprocess_fisher import clean_fisher

def get_start_time(example):
    example['start_time'] = int(example['uttid'].rpartition('_')[2].partition('-')[0])
    return example

def process_recording(args):
    id, index, dataset, split, min_len, max_len, max_context = args
    disable_progress_bar()

    # select the relevant rows and sort according to the start time of each utterance
    rows = dataset[split].select(index).map(get_start_time).sort('start_time')

    length_limit = uniform(min_len, max_len)
    length = 0
    concat = []
    accum = []

    for row in rows:
        input_len = row['input_len']

        if length + input_len >= length_limit and len(concat) >= 1:
            accum.append(concat)
            length = 0
            concat = []
            length_limit = uniform(min_len, max_len)

        concat.append(row)
        length += input_len

    accum.append(concat)

    # now process the accumulated recordings
    # concatenate the audio, labels, uttids, lengths, ...
    combined = []
    for i, acc in enumerate(accum):
        feats = acc[0]
        feats['audio']['array'] = [ feats['audio']['array'] ] # for easier concat
        feats['labels'] = [ feats['labels'] ]
        feats['uttid'] = [ feats['uttid'] ]
        feats.pop('input_len')
        feats['recording'] = [ feats['recording'] ]
        feats['turn_index'] = [ feats['turn_index'] ]

        for a in acc[1:]:
            feats['audio']['array'].append(a['audio']['array'])
            feats['labels'].append(a['labels'])
            feats['uttid'].append(a['uttid'])
            feats['recording'].append(a['recording'])
            feats['turn_index'].append(a['turn_index'])

        # concatenate all the audio arrays and cast them to fp32, obviously...
        feats['audio']['array'] = np.concatenate(feats['audio']['array'], dtype='float32')
        feats['total_len'] = feats['audio']['array'].size / 16000
        feats['speakers'] = list(map(lambda x: x[-1], feats['recording']))
        combined.append(feats)

    # combine the contexts for all combined utterances up to the current one
    for i, utt in enumerate(combined):
        utt['context'] = []
        for j in range(max(0, i - max_context), i):
            context_copy = deepcopy(combined[j])
            context_copy.pop('audio')
            context_copy.pop('context')
            context_copy.pop('total_len')
            context_copy.pop('start_time')
            context_copy.pop('uttid')
            context_copy.pop('recording')
            utt['context'].append(context_copy)

        if len(utt['context']) == 0: utt['context'] = None

    combined = Dataset.from_list(combined).cast_column('audio', Audio(sampling_rate=16000))
    enable_progress_bar()
    return combined

# casts the datasets shard to Audio and saves the shard to disk
def save_shard(data, output_dir, split, shard_index, num_cores):
    shard = concatenate_datasets(data) 
    shard_path = os.path.join(output_dir, f"{split}_shard_{shard_index}")
    shard.save_to_disk(shard_path, num_proc=num_cores)
    return shard_path


def main(num_cores, dataset_path, output_dir, shard_size, min_len, max_len, max_context, clean):
    dataset = load_from_disk(dataset_path)
    splits = dataset.keys()
    os.makedirs(output_dir, exist_ok=True)

    # First, preprocess the fisher data so that the labels are clean
    if clean:
        dataset = clean_fisher(dataset, num_cores)

    # Now, let's build up the contextualized dataset
    for split in splits:
        set_verbosity_error()

        def get_recording_id(example):
            example['recording_id'] = example['recording'][:-2]
            return example

        df = dataset[split].remove_columns(['audio']).map(get_recording_id, num_proc=num_cores).to_pandas()

        print(f"Preparing indices for the '{split}' split")
        index_dict = df.groupby('recording_id').apply(lambda x: x.index.tolist()).to_dict()
        rec_ids = list(index_dict.keys()) 

        # Prepare arguments for multiprocessing
        rec_ids_sharded = [ rec_ids[i:i+shard_size] for i in range(0, len(rec_ids), shard_size) ]

        # Use multiprocessing to parallelize the processing
        shard_paths = []
        shard_index = 0

        for rec_ids_shard in tqdm(rec_ids_sharded):
            args_list = [(id, index_dict[id], dataset, split, min_len, max_len, max_context) for id in rec_ids_shard]

            current_shard = []

            print(f"Processing rec_id chunk {shard_index}..")
            with mp.Pool(processes=num_cores) as pool:
                for result in tqdm(pool.imap(process_recording, args_list), total=len(args_list)):
                    current_shard.append(result)

            shard_path = save_shard(current_shard, output_dir, split, shard_index, num_cores)
            del current_shard
            shard_paths.append(shard_path)
            shard_index += 1

        # Combine all shards for this split
        print(f"Combining shards for split: {split}")
        combined_dataset = concatenate_datasets([load_from_disk(path) for path in shard_paths])

        # Save the combined dataset for this split
        combined_path = os.path.join(output_dir, f"{split}")
        combined_dataset.save_to_disk(combined_path, num_proc=num_cores)
        print(f"Combined dataset for split {split} saved to {combined_path}")

        # Remove individual shards to save disk space
        for path in shard_paths:
            os.system(f"rm -rf {path}")

    print("All splits processed and combined.")
    
    # Add the dataset dict to the destination dataset directory
    with open(os.path.join(output_dir, 'dataset_dict.json'), 'w') as f:
        f.write('{"splits": ["train", "test", "dev"]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio data with specified number of CPU cores and save progressively.')
    parser.add_argument('--cores', type=int, default=mp.cpu_count(),
                        help='Number of CPU cores to use (default: all available cores)')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the fisher dataset to be processed.')
    parser.add_argument('--clean', type=bool, default=False,
                        help='Clean the dataset labels first before processing..')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the processed dataset shards and final combined dataset')
    parser.add_argument('--shard_size', type=int, default=1000,
                        help='Number of processed recordings per shard (default: 1000)')
    parser.add_argument('--min_len', type=float, default=5.0,
                        help='Concatenated recording length soft lower bound (default: 10.0).')
    parser.add_argument('--max_len', type=float, default=20.0,
                        help='Concatenated recording length upper bound (default: 20.0).')
    parser.add_argument('--max_context', type=int, default=10,
                        help='Maximum context length (default: 10).')
    args = parser.parse_args()

    print(f"Using {args.cores} CPU cores")
    print(f"Saving output to {args.output_dir}")
    print(f"Shard size: {args.shard_size}")
    main(args.cores, args.dataset_path, args.output_dir, args.shard_size, args.min_len, args.max_len, args.max_context, clean=args.clean)
