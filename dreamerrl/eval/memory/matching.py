"""Canonical maximum matching of group-disjoint, exact-input label contrasts."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import networkx as nx


@dataclass(frozen=True)
class ContrastFrame:
    episode: int
    time: int
    key: tuple
    positive: bool
    age: int | None


def exact_contrasts(frames):
    """Return canonical disjoint (negative, positive) pairs and all matchable rows.

    A lexicographic earliest edge is fixed only if a residual maximum
    matching can still achieve the original cardinality. Each edge uses
    its earliest qualifying frame. Score arrays never enter this function.
    """
    by_key = defaultdict(lambda: ([], []))
    for row in frames:
        by_key[row.key][int(row.positive)].append(row)
    edges = {}
    matchable = set()
    for negative, positive in by_key.values():
        for left, right in product(negative, positive):
            if left.episode == right.episode:
                continue
            matchable.update((left, right))
            edge = tuple(sorted((left.episode, right.episode)))
            ordering = (left.time, right.time, left.episode, right.episode)
            if edge not in edges or ordering < edges[edge][0]:
                edges[edge] = (ordering, (left, right))
    graph = nx.Graph()
    graph.add_edges_from(sorted(edges))

    def size(graph):
        return len(nx.max_weight_matching(graph, maxcardinality=True))

    remaining = size(graph)
    pairs = []
    for edge in sorted(edges):
        if not remaining:
            break
        if not graph.has_edge(*edge):
            continue
        reduced = graph.copy()
        reduced.remove_nodes_from(edge)
        if size(reduced) == remaining - 1:
            pairs.append(edges[edge][1])
            graph = reduced
            remaining -= 1
    if remaining:
        raise RuntimeError("Unable to construct a canonical maximum matching")
    return pairs, matchable
