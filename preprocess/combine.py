from pathlib import Path

import numpy as np

from preprocess.utils.data.indexed_dataset import MMapIndexedDatasetBuilder
from preprocess.utils.data.data_utils import load_indexed_dataset
from preprocess.intermediate import indexed_dataset_path


def combine_loose_json(input_path, output_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')

    input_name = input_path.stem

    output_path = Path(output_path)
    loose_json_file = output_path / input_path.with_suffix('.ljson').name

if __name__ == '__main__':

    DTYPE = np.uint16
    input_path = Path('/data/standard_datasets/binarized')
    output_path = Path('/data/standard_datasets/binarized_combined')
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = None

    for dataset_base_path in input_path.iterdir():
        if not dataset_base_path.is_dir():
            continue
        
        dataset_name = dataset_base_path.name
        if datasets is not None and dataset_name not in datasets:
            print(f'Skipping {dataset_name}')
            continue
        print(f'Processing {dataset_name}')

        bin_path = output_path / f'{dataset_name}.bin'

        combined_dataset_builder = MMapIndexedDatasetBuilder(str(bin_path), DTYPE)

        for dataset_path in dataset_base_path.glob('*.bin'):
            if not dataset_path.with_suffix('.idx').is_file():
                print(f'Skipping {dataset_path}, missing .idx file')
                continue
            dataset_file = dataset_path.with_suffix('')
            combined_dataset_builder.merge_file_(str(dataset_file))

        combined_dataset_builder.finalize(str(bin_path.with_suffix('.idx')))