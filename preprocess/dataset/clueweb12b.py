import unicodedata
from pathlib import Path

import newspaper
import dragnet

from ..vendor import warc
from ..segmentation import sentence_segment

def process(input_path, output_path, fallback=True):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'input path must be a file, {input_path} is not')

    output_path = Path(output_path)
    if not output_path.is_dir():
        raise ValueError(f'output path must be a directory, {output_path} is not')

    output_file = output_path / input_path.name.replace('.warc.gz', '.txt')

    with output_file.open('w', encoding='utf8') as f:
        for doc in warc_documents(input_path, fallback=fallback):
            lines = [l for l in doc.split('\n') if l.strip()]
            sentences = sentence_segment(lines)
            f.write('\n'.join(sentences) + '\n\n')

def warc_documents(input_path, fallback=True):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'input path must be a file, {input_path} is not')

    warc_file = warc.open(str(input_path))
    for record in warc_file:
        try:
            content = dragnet.extract_content(record.payload)
            content = unicodedata.normalize('NFKC', content)  # fix things like \xa0 (&nbsp;)
        except dragnet.blocks.BlockifyError:
            if fallback:
                a = newspaper.Article(url=record.url, fetch_images=False)
                a.set_html(record.payload)
                a.parse()
                content = a.text
            else:
                content = ''
        
        content = content.strip()
        if content:
            yield content

