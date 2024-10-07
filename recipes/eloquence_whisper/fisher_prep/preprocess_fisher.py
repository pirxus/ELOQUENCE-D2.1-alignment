from datasets import load_from_disk
import argparse
import re

from transformers.utils.logging import get_logger

logger = get_logger('transformers')

def clean_train_split(example):
    example['labels'] = re.sub(' +', ' ', example['labels'].replace('[laughter]', '').replace('[noise]', ''))
    example['labels'] = example['labels'].replace('_', ' ')
    return example

def clean_dev_test_split(example):
    example['labels'] = re.sub(' +', ' ', example['labels'].replace('(%HESITATION)', ''))
    example['labels'] = example['labels'].translate(str.maketrans("", "", "()")).lower()
    return example

def clean_fisher(dataset, num_cores):
    logger.info("Processing the train split")
    dataset['train'] = dataset['train'].map(clean_train_split, num_proc=num_cores, desc="Removing train special tokens and underscores...")

    logger.info("Processing the dev split")
    dataset['dev'] = dataset['dev'].map(clean_dev_test_split, num_proc=num_cores, desc="Removing dev/test special tokens...")

    logger.info("Processing the test split")
    dataset['test'] = dataset['test'].map(clean_dev_test_split, num_proc=num_cores, desc="Removing dev/test special tokens...")

    dataset = dataset.filter(
        lambda x: x['labels'] != "",
        num_proc=num_cores,
        desc="Filtering out empty utterances"
    )

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process audio data with specified number of CPU cores and save progressively.')
    parser.add_argument('--cores', type=int, default=1,
                        help='Number of CPU cores to use (default: all available cores)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save the processed dataset shards and final combined dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the fisher dataset to be processed.')
    args = parser.parse_args()


    # Load dataset from disk
    dataset = load_from_disk(args.dataset_path)

    dataset = clean_fisher(dataset, args.cores)

    # Save the cleaned dataset to disk..
    dataset.save_to_disk(args.output_dir, num_proc=args.cores)
