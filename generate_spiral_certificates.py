#!/usr/bin/env python3
"""
Reproduce P58's Lean prime and row certificates from untrusted JSON data.

Run from the repository root. Generation never invokes Lean or searches for new
mathematical data. Every emitted proof must still pass the Lean kernel checks.
Use --check to compare both certificate directories without writing any files.
"""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from dataclasses import dataclass
import json
from math import prod
from pathlib import Path

BOUND = 13120
CHUNK_SIZE = 128
NUM = 1
DEN = 10
LIB = Path('ProjectEulerSolutions/Lib')
PRIME_DIR = LIB / 'SpiralPrimeCertificates'
ROW_DIR = LIB / 'SpiralRowCertificates'
PRIME_MODULE = 'ProjectEulerSolutions.Lib.SpiralPrimeCertificates'
ROW_MODULE = 'ProjectEulerSolutions.Lib.SpiralRowCertificates'


def integer(value, label):
    if type(value) is not int:
        raise ValueError(f'{label} must be an integer')
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_data(path):
    data = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_object)
    if not isinstance(data, dict) or not isinstance(data.get('metadata'), dict):
        raise ValueError('expected metadata, layers and prime_certificates objects')
    if data['metadata'].get('first_layer') != BOUND:
        raise ValueError(f'expected the fixed {BOUND}-layer data set')
    raw_nodes, layers = data.get('prime_certificates'), data.get('layers')
    if not isinstance(raw_nodes, dict) or not isinstance(layers, list) or len(layers) != BOUND:
        raise ValueError('invalid prime-node object or layer-list size')
    nodes = {}
    for key, node in raw_nodes.items():
        if not key.isdecimal() or str(int(key)) != key or not isinstance(node, dict):
            raise ValueError(f'invalid canonical prime-node key: {key}')
        number = int(key)
        if number < 2 or not isinstance(node.get('factors'), list):
            raise ValueError(f'invalid prime node {number}')
        integer(node.get('generator'), f'node {number} generator')
        nodes[number] = node
    seen = set()
    for number in sorted(nodes):
        factors = nodes[number]['factors']
        if any(type(factor) is not int or factor < 2 or factor >= number or factor not in seen
               for factor in factors):
            raise ValueError(f'node {number} has an invalid or forward factor reference')
        if prod(factors) != number - 1:
            raise ValueError(f'node {number} lacks a complete factorization')
        seen.add(number)
    previous = 0
    for expected, item in enumerate(layers, 1):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f'layer {expected} must have index, counter and three corners')
        layer, cumulative, corners = item
        integer(layer, f'layer {expected} index')
        integer(cumulative, f'layer {expected} cumulative count')
        if layer != expected or not isinstance(corners, list) or len(corners) != 3:
            raise ValueError(f'invalid layer index or corner count at {expected}')
        for offset, corner in enumerate(corners, 1):
            if not isinstance(corner, list) or len(corner) != 3:
                raise ValueError(f'invalid corner shape at {layer}:{offset}')
            number, flag, factor = corner
            for value, name in zip(corner, ('number', 'flag', 'factor')):
                integer(value, f'corner {layer}:{offset} {name}')
            if number != (2 * layer + 1) ** 2 - offset * 2 * layer or flag not in (0, 1):
                raise ValueError(f'invalid corner value or flag at {layer}:{offset}')
            if flag:
                if number not in nodes or factor != number:
                    raise ValueError(f'missing prime node or inconsistent prime marker {number}')
            elif not (2 <= factor < number and number % factor == 0):
                raise ValueError(f'invalid composite factor at {layer}:{offset}')
        if cumulative != previous + sum(corner[1] for corner in corners):
            raise ValueError(f'invalid cumulative count at layer {layer}')
        if (DEN * cumulative < NUM * (4 * layer + 1)) != (layer == BOUND):
            raise ValueError(f'invalid first-crossing data at layer {layer}')
        previous = cumulative
    if previous != 5248:
        raise ValueError('expected the certified endpoint counter 5248')
    # Other metadata are exploratory annotations, never premises of Lean proofs.
    return nodes, layers


def prime_files(nodes):
    files, numbers = {}, sorted(nodes)
    for start in range(0, len(numbers), CHUNK_SIZE):
        index = start // CHUNK_SIZE
        lines = ['import ProjectEulerSolutions.Lib.PrattCertificates']
        if index:
            lines.append(f'import {PRIME_MODULE}.Chunk{index - 1:03d}')
        lines.extend(['', '/-! Arithmetic certificates for the spiral corners; Lean checks every node. -/'])
        if index >= 35:
            lines.extend(['set_option maxRecDepth 4096', ''])
        lines.extend(['namespace ProjectEulerSolutions.P58PrimeNodes',
                      'open ProjectEulerSolutions.PrattCertificates', ''])
        for number in numbers[start:start + CHUNK_SIZE]:
            node = nodes[number]
            factors = ', '.join(f'p_{factor}' for factor in node['factors'])
            lines.extend([f'def p_{number} : CertifiedPrime :=',
                          f'  certify {number} ({node["generator"]}) [{factors}] (by decide)', ''])
        lines.append('end ProjectEulerSolutions.P58PrimeNodes')
        files[PRIME_DIR / f'Chunk{index:03d}.lean'] = '\n'.join(lines) + '\n'
    files[PRIME_DIR / 'All.lean'] = f'import {PRIME_MODULE}.Chunk{(len(numbers) - 1) // CHUNK_SIZE:03d}\n'
    return files


@dataclass(frozen=True)
class Segment:
    name: str
    first: int
    last: int
    initial: int
    final: int
    agreement: str


def balanced(segments, lines, prefix):
    next_join = 0

    def combine(items):
        nonlocal next_join
        if len(items) == 1:
            return items[0]
        middle = len(items) // 2
        left, right = combine(items[:middle]), combine(items[middle:])
        if left.last != right.first or left.final != right.initial:
            raise ValueError('segment boundaries disagree')
        name = f'{prefix}_{next_join:03d}'
        next_join += 1
        lines.extend([
            f'private def {name} : CountSegment 1 10 13120 {left.first} {right.last} {left.initial} {right.final} := CountSegment.join {left.name} {right.name}',
            f'private theorem {name}_agreement : RowAgreement {left.first} {right.last} := RowAgreement.join {left.agreement} {right.agreement}',
        ])
        return Segment(name, left.first, right.last, left.initial, right.final, name + '_agreement')

    return combine(segments)


def row_lines(layer, corners):
    lines = [f'def row_{layer} : RowCertificate {layer} where']
    for field, offset, (number, flag, factor) in zip(('first', 'second', 'third'), range(1, 4), corners):
        lines.extend([f'  {field} := {"true" if flag else "false"}', f'  {field}_correct := by'])
        if flag:
            lines.extend([
                f'    have prime : Nat.Prime (corner {layer} {offset}) := by change Nat.Prime {number}; exact ProjectEulerSolutions.P58PrimeNodes.p_{number}.prime',
                '    exact decide_eq_true prime',
            ])
        else:
            lines.extend([
                f'    have composite : ¬ Nat.Prime (corner {layer} {offset}) := by change ¬ Nat.Prime {number}; exact PrattCertificates.composite_of_product (factor := {factor}) (cofactor := {number // factor}) rfl (by decide) (by decide)',
                '    exact decide_eq_false composite',
            ])
    increment = sum(corner[1] for corner in corners)
    lines.extend([
        f'theorem row_count_{layer} : row {layer} = {increment} := (row_{layer}.checked).trans rfl',
        f'theorem model_count_{layer} : modelRow {layer} = {increment} := by decide',
        f'theorem modelrow_agrees_{layer} : modelRow {layer} = row {layer} := model_count_{layer}.trans row_count_{layer}.symm',
    ])
    return lines


def row_files(layers):
    files, chunks = {}, []
    for start in range(0, BOUND, CHUNK_SIZE):
        index = start // CHUNK_SIZE
        rows = layers[start:start + CHUNK_SIZE]
        lines = [
            'import ProjectEulerSolutions.Lib.SpiralPrimeSegments',
            f'import {PRIME_MODULE}.All',
            'import ProjectEulerSolutions.Lib.MillerRabin',
            'import ProjectEulerSolutions.Lib.SpiralMillerAgreement',
            '/-! Finite spiral row and counter certificates. Python execution is checked separately. -/',
            'namespace ProjectEulerSolutions.P58Numerical',
            'open SpiralPrimes',
            'set_option Elab.async false',
        ]
        if index in (61, 62) or index >= 64:
            lines.append('set_option maxRecDepth 4096')
        for layer, _, corners in rows:
            lines.extend(row_lines(layer, corners))
        atoms = []
        for layer, cumulative, _ in rows:
            previous = layers[layer - 2][1] if layer > 1 else 0
            atom = f'atomic_{layer}'
            lines.extend([
                f'private def {atom} : CountSegment 1 10 13120 {layer - 1} {layer} {previous} {cumulative} := CountSegment.single (first := {layer - 1}) (initial := {previous}) row_count_{layer} (by decide)',
                f'private theorem {atom}_agreement : RowAgreement {layer - 1} {layer} := RowAgreement.single (first := {layer - 1}) modelrow_agrees_{layer}',
            ])
            atoms.append(Segment(atom, layer - 1, layer, previous, cumulative, atom + '_agreement'))
        joined = balanced(atoms, lines, f'within_{index:03d}')
        name = f'segment_{index:03d}'
        agreement = f'agreement_{index:03d}'
        lines.extend([
            f'def {name} : CountSegment 1 10 13120 {joined.first} {joined.last} {joined.initial} {joined.final} := {joined.name}',
            f'theorem {agreement} : RowAgreement {joined.first} {joined.last} := {joined.agreement}',
            'end ProjectEulerSolutions.P58Numerical',
        ])
        files[ROW_DIR / f'Chunk{index:03d}.lean'] = '\n'.join(lines) + '\n'
        chunks.append(Segment(name, joined.first, joined.last, joined.initial, joined.final, agreement))
    lines = [*(f'import {ROW_MODULE}.Chunk{index:03d}' for index in range(len(chunks))),
             'namespace ProjectEulerSolutions.P58Numerical', 'open SpiralPrimes', 'set_option Elab.async false']
    joined = balanced(chunks, lines, 'joined')
    lines.extend([
        f'def completeSegment : CountSegment 1 10 13120 0 13120 0 5248 := {joined.name}',
        'def certifiedPrefix : CountPrefix 13120 := completeSegment.asPrefix',
        'theorem endpoint_count : count 13120 = 5248 := (certifiedPrefix.correct (layer := 13120) (by decide)).symm.trans completeSegment.ending',
        f'theorem model_rows_agree : RowAgreement 0 13120 := {joined.agreement}',
        'theorem first_crossing : FirstCrossing 1 10 13120 := completeSegment.firstCrossing (by decide)',
        'theorem reference_search : search 1 10 13120 1 = some 26241 := first_crossing.search',
        'theorem root_search : rootSearch 1 10 13120 0 0 = some 26241 := first_crossing.rootSearch',
        'end ProjectEulerSolutions.P58Numerical',
    ])
    files[ROW_DIR / 'All.lean'] = '\n'.join(lines) + '\n'
    return files


def mismatches(root, files):
    problems = []
    for relative, expected in files.items():
        path = root / relative
        if not path.is_file():
            problems.append(f'missing: {relative}')
        elif path.read_text(encoding='utf-8') != expected:
            problems.append(f'different: {relative}')
    for directory in (PRIME_DIR, ROW_DIR):
        if (root / directory).is_dir():
            for path in sorted((root / directory).rglob('*.lean')):
                if path.relative_to(root) not in files:
                    problems.append(f'unexpected: {path.relative_to(root)}')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/p58-certificates.json'))
    parser.add_argument('--out-root', type=Path, default=Path('.'))
    parser.add_argument('--check', action='store_true', help='Compare generated files without creating or changing files')
    args = parser.parse_args(argv)
    try:
        nodes, layers = read_data(args.data)
        files = {**prime_files(nodes), **row_files(layers)}
        root = args.out_root.resolve()
        if args.check:
            problems = mismatches(root, files)
            if problems:
                print('\n'.join(problems[:12]), file=sys.stderr)
                if len(problems) > 12:
                    print(f'... and {len(problems) - 12} more differences', file=sys.stderr)
                return 1
            print(f'All {len(files)} generated Lean files match; Lean kernel verification remains required.')
            return 0
        extras = [problem for problem in mismatches(root, files) if problem.startswith('unexpected:')]
        if extras:
            raise ValueError('refusing to remove unexpected Lean files: ' + ', '.join(extras[:5]))
        changed = 0
        for relative, text in files.items():
            path = root / relative
            if path.is_file() and path.read_text(encoding='utf-8') == text:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding='utf-8')
            changed += 1
        print(f'Reproduced {len(files)} Lean files ({changed} changed). Generation is not Lean kernel verification.')
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f'certificate generation failed: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
