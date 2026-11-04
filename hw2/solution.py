#!/usr/bin/env python3
"""Текстовое ранжирование с параметрами, подобранными на обучающих запросах."""

import argparse
import csv
import heapq
import math
import re
import time
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pymorphy3

WORD_RE = re.compile(r"\w+")
RUSSIAN_RE = re.compile(r"[а-яё]", re.IGNORECASE)
K1, B_TITLE, B_BODY, W_TITLE = 1.2, 0.3, 0.75, 0.5


def read_queries(path):
    """QueryId -> QueryText."""
    queries = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            query_id, text = line.rstrip('\n').split('\t', 1)
            queries[int(query_id)] = text
    return queries


def read_qrels(path):
    """QueryId -> {DocumentId -> Label}. Формат строки: QueryId 0 DocumentId Label."""
    qrels = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, label = line.split()
            qrels.setdefault(int(query_id), {})[doc_id] = int(label)
    return qrels


def read_docs(path, doc_ids=None):
    """DocumentId -> (URL, Title, Body). Если передан doc_ids, читает только эти документы."""
    docs = {}
    with open(path, encoding='utf-8') as f:
        for line in f:
            fields = line.rstrip('\n').split('\t')
            fields += [''] * (4 - len(fields))
            doc_id, url, title, body = fields[:4]
            if doc_ids is None or doc_id in doc_ids:
                docs[doc_id] = (url, title, body)
    return docs


class Normalizer:
    def __init__(self):
        self.morph = pymorphy3.MorphAnalyzer()
        # Bounded cache: long web pages contain many unique names and strings.
        self.lemma = lru_cache(maxsize=250_000)(self._lemma)

    def _lemma(self, word):
        if not RUSSIAN_RE.search(word):
            return word
        return self.morph.parse(word)[0].normal_form.replace("ё", "е")

    def tokens(self, text):
        text = text.lower()
        if "\u0301" in text or "\u0300" in text:
            text = unicodedata.normalize("NFD", text)
            text = text.replace("\u0301", "").replace("\u0300", "")
            text = unicodedata.normalize("NFC", text)
        text = text.replace("ё", "е")
        return [self.lemma(t) for t in WORD_RE.findall(text)]


def minimum_cover_span(position_lists):
    """Shortest window covering one occurrence of every distinct query term.

    Merge sorted position lists using a min-heap. At each step advance the
    current leftmost term. O(P log m), O(m) space for m query terms and P
    examined occurrences. Missing terms have no covering window.
    """
    if len(position_lists) < 2 or any(not p for p in position_lists):
        return None
    heap = [(p[0], term, 0) for term, p in enumerate(position_lists)]
    heapq.heapify(heap)
    right = max(p[0] for p in position_lists)
    best = math.inf
    while True:
        left, term, offset = heap[0]
        best = min(best, right - left + 1)
        if best == len(position_lists):
            break
        offset += 1
        if offset == len(position_lists[term]):
            break
        value = position_lists[term][offset]
        right = max(right, value)
        heapq.heapreplace(heap, (value, term, offset))
    return int(best)


def position_features(query_tokens, unique, positions):
    if len(query_tokens) < 2:
        return 0.0, 0.0
    span = minimum_cover_span([positions.get(t, []) for t in unique])
    proximity = len(unique) / span if span is not None else 0.0
    # Ordered exact phrase; repeated query terms keep their own offsets.
    if span is None and any(t not in positions for t in unique):
        return proximity, 0.0
    anchor = min(range(len(query_tokens)), key=lambda j: len(positions[query_tokens[j]]))
    candidates = {p - anchor for p in positions[query_tokens[anchor]] if p >= anchor}
    for offset, term in enumerate(query_tokens):
        if offset != anchor:
            candidates.intersection_update(p - offset for p in positions[term])
        if not candidates:
            break
    return proximity, float(bool(candidates))


def collect_statistics(path, tokens, terms, candidates, normalizer):
    """Вычислить статистику корпуса и признаки заданных пар."""
    query_terms = set().union(*terms.values())
    linked = defaultdict(set)
    for qid, pairs in candidates.items():
        for _, doc_id in pairs:
            linked[doc_id].add(qid)
    df = Counter()
    features, seen = {}, set()
    title_total = body_total = 0
    with path.open(encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t", 3)
            fields += [""] * (4 - len(fields))
            doc_id, _, title, body = fields
            if not doc_id or doc_id in seen:
                raise ValueError(f"Empty or duplicate DocumentId at line {number}: {doc_id}")
            seen.add(doc_id)
            title_tokens = normalizer.tokens(title)
            body_tokens = normalizer.tokens(body)
            title_total += len(title_tokens)
            body_total += len(body_tokens)
            df.update(query_terms.intersection(title_tokens) | query_terms.intersection(body_tokens))
            if doc_id in linked:
                wanted = set().union(*(terms[qid] for qid in linked[doc_id]))
                title_pos = word_positions(title_tokens, wanted)
                body_pos = word_positions(body_tokens, wanted)
                for qid in linked[doc_id]:
                    title_tf = [len(title_pos.get(t, ())) for t in terms[qid]]
                    body_tf = [len(body_pos.get(t, ())) for t in terms[qid]]
                    tp, tf = position_features(tokens[qid], terms[qid], title_pos)
                    bp, bf = position_features(tokens[qid], terms[qid], body_pos)
                    features[qid, doc_id] = (
                        len(title_tokens), len(body_tokens), title_tf, body_tf, (tp, bp, tf, bf)
                    )
            if number % 5000 == 0:
                print(f"Read {number:,} documents", flush=True)
    missing = linked.keys() - seen
    if missing:
        raise ValueError(f"Missing candidate documents: {sorted(missing)[:5]}")
    n_docs = len(seen)
    if not n_docs:
        raise ValueError("Empty document collection")
    return features, df, n_docs, title_total / n_docs or 1.0, body_total / n_docs or 1.0


def word_positions(tokens, wanted):
    positions = defaultdict(list)
    for pos, term in enumerate(tokens):
        if term in wanted:
            positions[term].append(pos)
    return positions


def score_query(rows, idf, avg_title, avg_body):
    """Смесь BM25 по полям и бонусов из отчёта подбора."""
    if not len(idf):
        return np.zeros(len(rows))
    title_len, body_len, title_tf, body_tf, bonuses = zip(*rows)
    title_tf, body_tf = np.array(title_tf, dtype=float), np.array(body_tf, dtype=float)
    title_norm = 1 - B_TITLE + B_TITLE * np.array(title_len)[:, None] / avg_title
    body_norm = 1 - B_BODY + B_BODY * np.array(body_len)[:, None] / avg_body
    weights = idf * (K1 + 1) * (
        W_TITLE * title_tf / (title_tf + K1 * title_norm)
        + body_tf / (body_tf + K1 * body_norm)
    )
    base = np.add.accumulate(weights, axis=1)[:, -1]
    width = float(base.max() - base.min())
    scores = (base - base.min()) / width if width > 1e-12 else np.zeros_like(base)
    title_window, body_window, title_phrase, body_phrase = np.asarray(bonuses).T
    coverage = ((title_tf + body_tf) > 0) @ idf / idf.sum()
    scores += 0.4 * title_window
    scores += body_window
    scores += title_phrase
    scores += 0.05 * body_phrase
    scores += 0.05 * coverage
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission_file", type=Path, required=True)
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()
    start = time.perf_counter()

    queries = read_queries(args.data_dir / "vkmarco-doceval-queries.tsv")
    objects, candidates = [], defaultdict(list)
    object_ids = set()
    with (args.data_dir / "objects.csv").open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            oid, qid, doc_id = int(row["ObjectId"]), int(row["QueryId"]), row["DocumentId"]
            if oid in object_ids or qid not in queries:
                raise ValueError(f"Duplicate ObjectId or unknown QueryId: {oid}, {qid}")
            object_ids.add(oid)
            objects.append(oid)
            candidates[qid].append((oid, doc_id))
    if not objects:
        raise ValueError("Empty objects.csv")

    normalizer = Normalizer()
    tokens = {qid: normalizer.tokens(queries[qid]) for qid in candidates}
    terms = {qid: sorted(set(words)) for qid, words in tokens.items()}
    features, df, n_docs, avg_title, avg_body = collect_statistics(
        args.data_dir / "vkmarco-docs.tsv", tokens, terms, candidates, normalizer
    )
    scores = {}
    for qid, pairs in candidates.items():
        pairs.sort()
        idf = np.array([
            np.log1p((n_docs - df[t] + 0.5) / (df[t] + 0.5)) if df[t]
            else math.log1p((n_docs + 0.5) / 0.5)
            for t in terms[qid]
        ])
        rows = [features[qid, doc_id] for _, doc_id in pairs]
        values = score_query(rows, idf, avg_title, avg_body)
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite ranking scores for query {qid}")
        scores.update((oid, score) for (oid, _), score in zip(pairs, values))

    args.submission_file.parent.mkdir(parents=True, exist_ok=True)
    with args.submission_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ObjectId", "Score"])
        writer.writerows((oid, format(scores[oid], ".17g")) for oid in objects)
    print(f"Saved {len(objects):,} scores in {time.perf_counter() - start:.1f} s")


if __name__ == "__main__":
    main()
