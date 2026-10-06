#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Булев поиск. Построение индекса и генерация сабмишна.

    python solution.py --build_index --index_dir=index PATH-TO-DATA
    python solution.py --submission_file=submission.csv --index_dir=index PATH-TO-DATA
"""

import argparse
import csv
from array import array
from collections import defaultdict
from pathlib import Path
from timeit import default_timer as timer

from nltk import tokenize


def preprocess(text):
    """Токенизация и нормализация. НЕ МЕНЯЙТЕ эту функцию:
    ей же размечен эталон на Kaggle."""
    tokenizer = tokenize.RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    return [token.lower() for token in tokens]



def build_index(data_dir, index_dir):
    data_dir, index_dir = Path(data_dir), Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    index = defaultdict(lambda: array('I'))
    with (data_dir / 'vkmarco-docs.tsv').open(encoding='utf-8') as source, \
            (index_dir / 'documents.tsv').open('w', encoding='utf-8') as mapping:
        for number, line in enumerate(source):
            doc_id, url, title, body = line.rstrip('\r\n').split('\t', 3)
            mapping.write(f'{doc_id}\t{number}\n')
            for word in set(preprocess(title + ' ' + body)):
                index[word].append(number)

    with (index_dir / 'postings.bin').open('wb') as binary, \
            (index_dir / 'lexicon.tsv').open('w', encoding='utf-8') as lexicon:
        for word, numbers in index.items():
            lexicon.write(f'{word}\t{binary.tell()}\t{len(numbers)}\n')
            numbers.tofile(binary)


def make_submission(data_dir, index_dir, submission_file):
    data_dir, index_dir = Path(data_dir), Path(index_dir)
    queries = {}
    with (data_dir / 'vkmarco-doceval-queries.tsv').open(encoding='utf-8') as source:
        for line in source:
            query_id, text = line.rstrip('\r\n').split('\t', 1)
            queries[query_id] = set(preprocess(text))
    needed_words = set().union(*queries.values())
    locations = {}
    with (index_dir / 'lexicon.tsv').open(encoding='utf-8') as lexicon:
        for line in lexicon:
            word, offset, count = line.rstrip('\r\n').split('\t')
            if word in needed_words:
                locations[word] = (int(offset), int(count))

    document_numbers = {}
    with (index_dir / 'documents.tsv').open(encoding='utf-8') as mapping:
        for line in mapping:
            doc_id, number = line.rstrip('\r\n').split('\t')
            document_numbers[doc_id] = int(number)

    matches = {}
    with (index_dir / 'postings.bin').open('rb') as binary:
        for query_id, words in queries.items():
            if not words:
                matches[query_id] = range(len(document_numbers))
                continue
            if any(word not in locations for word in words):
                matches[query_id] = set()
                continue
            result = None
            for word in sorted(words, key=lambda w: locations[w][1]):
                offset, count = locations[word]
                binary.seek(offset)
                numbers = array('I')
                numbers.fromfile(binary, count)
                if result is None:
                    result = set(numbers)
                else:
                    result.intersection_update(numbers)
                if not result:
                    break
            matches[query_id] = result

    with (data_dir / 'objects.csv').open(encoding='utf-8', newline='') as source, \
            Path(submission_file).open('w', encoding='utf-8', newline='') as output:
        writer = csv.writer(output)
        writer.writerow(['ObjectId', 'Label'])
        for row in csv.DictReader(source):
            number = document_numbers[row['DocumentId']]
            label = int(number in matches[row['QueryId']])
            writer.writerow([row['ObjectId'], label])


def main():
    parser = argparse.ArgumentParser(description='Boolean retrieval homework solution')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('--build_index', action='store_true', help='режим построения индекса')
    parser.add_argument('--index_dir', required=True, help='папка с индексом')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()
    if args.build_index:
        build_index(args.data_dir, args.index_dir)
    else:
        if not args.submission_file:
            parser.error('в режиме генерации сабмишна нужен --submission_file')
        make_submission(args.data_dir, args.index_dir, args.submission_file)
    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
