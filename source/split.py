import os
import random
import argparse


def divide_and_store(full_metadata_path, test_ratio=0.1):
    metadata_directory = os.path.dirname(os.path.realpath(full_metadata_path))
    train_metadata_path = os.path.join(metadata_directory, 'train_metadata.txt')
    test_metadata_path = os.path.join(metadata_directory, 'test_metadata.txt')
    
    with open(full_metadata_path) as file:
        all_metadata = file.readlines()
        train_metadata = []
        test_metadata = []

    total_entries = len(all_metadata)
    test_entries_count = int(test_ratio * total_entries)
    test_indices = random.sample(range(total_entries), test_entries_count)

    for index, entry in enumerate(all_metadata):
        if index in test_indices:
            test_metadata.append(entry)
        else:
            train_metadata.append(entry)

    with open(train_metadata_path, 'w') as train_file:
        train_file.write(''.join(train_metadata))
        
    with open(test_metadata_path, 'w') as test_file:
        test_file.write(''.join(test_metadata))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', type = str)
    args = parser.parse_args()
    full_metadata_path = args.metadata
    test_ratio = 0.1 

    divide_and_store(full_metadata_path, test_ratio)
