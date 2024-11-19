from datasets import load_dataset

if __name__ == "__main__":

    dataset = load_dataset(
        '/mnt/matylda6/isedlacek/projects/huggingface_asr/src/dataset_builders/spokenwoz',
        data_dir='/mnt/matylda4/kesiraju/datasets/dialogue_datasets/SpokenWoz_2023',
        num_proc=16,
        trust_remote_code=True,
    )

    print(dataset)
    dataset.save_to_disk('/mnt/matylda6/isedlacek/data/spokenwoz', num_proc=16)
