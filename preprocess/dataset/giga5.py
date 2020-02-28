import gzip
import re
from pathlib import Path

import lxml.etree

from ..segmentation import sentence_segment

def process(input_path, output_path):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise ValueError(f'input path must be a file, {input_path} is not')

    output_path = Path(output_path)
    if not output_path.is_dir():
        raise ValueError(f'output path must be a directory, {output_path} is not')

    output_file = output_path / (input_path.stem + '.txt')

    with gzip.open(str(input_path), 'rb') as f:
        content = f.read()

    # parse the tree
    parser = lxml.etree.HTMLParser()  # use html parser because of stuff like &amp;
    tree = lxml.etree.fromstring(content, parser=parser)
    docs = tree.find('body')

    prefix = re.compile('^.*?///\s?')

    with output_file.open('w', encoding='utf8') as f:
        # iterate over all the docs in the tree
        for doc in docs.iterchildren('doc'):
            # throw out other and advis docs
            if doc.get('type') in {'other', 'advis', 'multi'}:
                continue

            # mash all the text together
            text_node = doc.find('text')
            text = lxml.etree.tostring(text_node, method='text', encoding='unicode')

            # keep only non-blank lines
            lines = [l for l in text.split('\n') if l.strip()]
            if lines:
                lines[0] = re.sub(prefix, '', lines[0])
                if not lines[0]:
                    lines.pop(0)
                sentences = sentence_segment(' '.join(lines))
                f.write('\n'.join(sentences) + '\n\n')
