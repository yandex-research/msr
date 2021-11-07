import os
import argparse
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='.')
    parser.add_argument('--source-path', type=str, required=True,
        help='Path to the source file txt')
    parser.add_argument('--target-path', type=str, required=True,
        help='Path to the taget file txt')
    parser.add_argument('--directory', type=str, required=True,
        help='Directory to save resampled files')
    parser.add_argument('--max-sentences', type=int, default=3, 
        help='Max sentences in the resampled files')
    parser.add_argument('--length-clip', type=int, default=125, 
        help="Maximum allowed length of a sentence. If a sentence is longer than length_clip, it's weight will be cliped to this value")
    parser.add_argument('--dataset-size-multiplier', type=int, default=10,
        help="How many times bigger the new dataset will be in lines"
    )
    parser.add_argument('--not-weighted', action='store_true',
        help="Wheather to use lengths as weights for resampling distribution"
    )
    parser.add_argument('--silent', action='store_true',
        help="Wheather to show tqdm progress bar"
    )
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()


    np.random.seed(args.seed)
    source_file = []
    target_file = []
    with open(args.source_path) as f:
        for line in f:
            source_file.append(line.strip())
    with open(args.target_path) as f:
        for line in f:
            target_file.append(line.strip())

    target_lengths =  np.array( list(map(lambda s: len(s.split(' ')), target_file)) )
    target_lengths[target_lengths>args.length_clip] = args.length_clip
    distribution_coeffs = target_lengths / target_lengths.sum()


    new_source = []
    new_target = []
    new_size = len(target_file) * args.dataset_size_multiplier
    num_sentences = np.random.choice(np.arange(1, args.max_sentences+1), size=new_size)
    range_of_indices = np.arange(0, len(target_file))
    if args.not_weighted:
        print("Not weighted")
        all_sentences_ids = np.random.choice(a=range_of_indices, size=(new_size, args.max_sentences))
    else:
        all_sentences_ids = np.random.choice(a=range_of_indices, p=distribution_coeffs, size=(new_size, args.max_sentences))

    if args.silent:
        size_range = range(new_size)
    else:
        size_range = tqdm(range(new_size))
    for i in size_range:
        cur_num = num_sentences[i]
        sentences_ids = all_sentences_ids[i, :cur_num]
        cur_source = ''
        cur_target = ''
        for idx in sentences_ids:
            if cur_source == '':
                cur_source += source_file[idx]
                cur_target += target_file[idx]
            else:
                cur_source += ' ' + source_file[idx]
                cur_target += ' ' + target_file[idx]
        new_source.append(cur_source)
        new_target.append(cur_target)

    with open(os.path.join(args.directory, args.source_path.split('/')[-1]),'w') as f:
        for line in new_source:
            f.write(line + '\n')
    with open(os.path.join(args.directory, args.target_path.split('/')[-1]),'w') as f:
        for line in new_target:
            f.write(line + '\n')

if __name__ == "__main__":
    main()
