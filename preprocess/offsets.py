import pickle
from pathlib import Path

def document_offsets(input_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')
    
    with input_path.open(encoding='utf8') as f:
        document_offsets = [0]
        pos = 0
        line = f.readline()
        while line:
            pos = f.tell()
            if line == '\n':
                document_offsets.append(pos)
            line = f.readline()
        document_offsets.append(f.tell())

    return document_offsets

def document_offsets_path(input_path, output_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')
    
    if output_path is None:
        return input_path.with_name(input_path.stem + '.offsets.pkl')
    else:
        return Path(output_path) / (input_path.stem + '.offsets.pkl')

def write_document_offsets(input_path, output_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'invalid input path {input_path}, must be a file')
    
    output_path = document_offsets_path(input_path, output_path)

    with output_path.open('wb') as f:
        pickle.dump(document_offsets(input_path), f)