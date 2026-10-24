"""Offline JSONL validation; standard library only. Never loads a model."""
import argparse
import hashlib
import json
import sys
import unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path


def normalize(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def validate(paths, expected_counts=None, near_threshold=0.88, require_system=False, min_rounds=2):
    errors, warnings, records, stats = [], [], [], []
    seen_conversations, seen_prompts, seen_answers, seen_ids = {}, {}, {}, {}
    for file_index, path in enumerate(map(Path, paths)):
        count, valid, rounds, categories = 0, 0, Counter(), Counter()
        try:
            raw = path.read_bytes()
            text = raw.decode('utf-8-sig')
        except (OSError, UnicodeError) as exc:
            errors.append(f'{path}: cannot read UTF-8 file: {exc}')
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            count += 1
            loc = f'{path.name}:{line_number}'
            if not line.strip():
                errors.append(f'{loc}: blank line')
                continue
            try:
                obj = json.loads(line, object_pairs_hook=unique_keys)
            except (ValueError, TypeError) as exc:
                errors.append(f'{loc}: invalid JSON: {exc}')
                continue
            if not isinstance(obj, dict) or not isinstance(obj.get('messages'), list):
                errors.append(f'{loc}: expected object with messages array')
                continue
            messages = obj['messages']
            start = 1 if messages and isinstance(messages[0], dict) and messages[0].get('role') == 'system' else 0
            turns = messages[start:]
            problems = []
            if require_system and not start:
                problems.append('required first system message is missing')
            if len(turns) < 2 * min_rounds or len(turns) % 2:
                problems.append(f'need at least {min_rounds} complete user/assistant rounds')
            for i, message in enumerate(messages):
                if not isinstance(message, dict):
                    problems.append(f'message {i}: expected object')
                    continue
                expected = 'system' if start and i == 0 else ('user' if (i-start) % 2 == 0 else 'assistant')
                if message.get('role') != expected:
                    problems.append(f'message {i}: expected role {expected}')
                content = message.get('content')
                if not isinstance(content, str) or not normalize(content):
                    problems.append(f'message {i}: content must be nonempty text')
                elif '\ufffd' in content or any(unicodedata.category(c) == 'Cc' and c not in '\n\t\r' for c in content):
                    problems.append(f'message {i}: replacement/control character')
            if problems:
                errors.extend(f'{loc}: {p}' for p in problems)
                continue
            valid += 1
            rounds[len(turns)//2] += 1
            categories[str(obj.get('category', 'unspecified'))] += 1
            if 'id' in obj:
                identifier = obj['id']
                if not isinstance(identifier, str) or not identifier.strip():
                    errors.append(f'{loc}: id must be nonempty string')
                elif identifier in seen_ids:
                    errors.append(f'{loc}: duplicate id with {seen_ids[identifier]}')
                else:
                    seen_ids[identifier] = loc
            conversation = tuple((m['role'], normalize(m['content'])) for m in turns)
            if conversation in seen_conversations:
                errors.append(f'{loc}: duplicate conversation with {seen_conversations[conversation]}')
            else:
                seen_conversations[conversation] = loc
            for role, registry in [('user', seen_prompts), ('assistant', seen_answers)]:
                local = set()
                for message in turns:
                    if message['role'] != role:
                        continue
                    key = normalize(message['content'])
                    if key in local:
                        errors.append(f'{loc}: repeated {role} utterance within conversation')
                    local.add(key)
                    previous = registry.get(key)
                    if previous and previous[0] != file_index:
                        errors.append(f'{loc}: cross-split {role} leakage with {previous[1]}')
                    elif previous:
                        warnings.append(f'{loc}: repeated {role} utterance with {previous[1]}')
                    else:
                        registry[key] = (file_index, loc)
            records.append((file_index, loc, conversation))
        if expected_counts is not None and count != expected_counts[file_index]:
            errors.append(f'{path.name}: expected {expected_counts[file_index]} records, got {count}')
        stats.append({'file': path.name, 'lines': count, 'valid_structure': valid,
                      'rounds': dict(rounds), 'categories': dict(categories),
                      'sha256': hashlib.sha256(raw).hexdigest()})
    for i, (split_a, loc_a, a) in enumerate(records):
        for split_b, loc_b, b in records[i+1:]:
            if a == b:
                continue
            # Compare all dialogue text and first prompt separately: catches rewrites
            # even when follow-up messages differ. A heuristic, not semantic proof.
            whole_a, whole_b = ''.join(x[1] for x in a), ''.join(x[1] for x in b)
            first_a, first_b = a[0][1], b[0][1]
            score = max(SequenceMatcher(None, whole_a, whole_b, autojunk=False).ratio(),
                        SequenceMatcher(None, first_a, first_b, autojunk=False).ratio())
            if score >= near_threshold:
                label = 'cross-split near match' if split_a != split_b else 'near duplicate'
                warnings.append(f'{loc_a} / {loc_b}: {label} ({score:.3f})')
    return {'ok': not errors, 'errors': errors, 'warnings': warnings, 'files': stats,
            'near_threshold': near_threshold, 'require_system': require_system, 'min_rounds': min_rounds,
            'limits': 'Lexical checks cannot prove semantic independence, originality, or language quality. Review scenarios manually.'}


def validate_scenarios(path, corpus_paths, near_threshold=0.88):
    errors, warnings, prompts, corpus_ids = [], [], [], set()
    for corpus_path in map(Path, corpus_paths):
        try:
            for line in corpus_path.read_text(encoding='utf-8-sig').splitlines():
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if isinstance(obj.get('id'), str):
                    corpus_ids.add(obj['id'])
                if isinstance(obj.get('messages'), list):
                    for m in obj['messages']:
                        if isinstance(m, dict) and m.get('role') == 'user' and isinstance(m.get('content'), str):
                            prompts.append((corpus_path.name, normalize(m['content'])))
        except (OSError, ValueError, UnicodeError) as exc:
            errors.append(f'{corpus_path}: unable to check holdout overlap: {exc}')
    try:
        raw = Path(path).read_bytes()
        lines = raw.decode('utf-8-sig').splitlines()
    except (OSError, UnicodeError) as exc:
        return {'errors': [f'{path}: cannot read scenarios: {exc}'], 'warnings': []}
    if len(lines) != 10:
        errors.append(f'{Path(path).name}: expected 10 scenarios, got {len(lines)}')
    ids, seen = set(), {}
    for i, line in enumerate(lines, 1):
        loc = f'{Path(path).name}:{i}'
        try:
            row = json.loads(line, object_pairs_hook=unique_keys)
        except ValueError as exc:
            errors.append(f'{loc}: invalid JSON: {exc}')
            continue
        if not isinstance(row, dict):
            errors.append(f'{loc}: expected scenario object')
            continue
        prompt_fields = ['first_user', 'follow_up_user'] + [
            key for key in row if key.endswith('_user') and key not in ('first_user', 'follow_up_user')
        ]
        required = ['id', 'scenario'] + prompt_fields
        if any(not isinstance(row.get(k), str) or not normalize(row[k]) for k in required):
            errors.append(f'{loc}: missing or empty scenario field')
            continue
        if row['id'] in ids or row['id'] in corpus_ids:
            errors.append(f'{loc}: duplicate scenario id')
        ids.add(row['id'])
        if not isinstance(row.get('checks'), list) or not row['checks'] or any(not isinstance(c, str) or not c.strip() for c in row['checks']):
            errors.append(f'{loc}: checks must be a nonempty list of text')
        if any(k in row for k in ('messages', 'assistant', 'answer', 'expected_answer')):
            errors.append(f'{loc}: reserved tests must not contain model answers')
        for field in prompt_fields:
            text = normalize(row[field])
            if text in seen:
                errors.append(f'{loc}: repeated holdout prompt with {seen[text]}')
            seen[text] = loc
            for source, prompt in prompts:
                if text == prompt:
                    errors.append(f'{loc}: holdout prompt leakage with {source}')
                elif SequenceMatcher(None, text, prompt, autojunk=False).ratio() >= near_threshold:
                    warnings.append(f'{loc}: near-match holdout prompt with {source}')
    return {'file': Path(path).name, 'count': len(lines), 'sha256': hashlib.sha256(raw).hexdigest(),
            'errors': errors, 'warnings': warnings}


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train', type=Path, default=root/'data/train.jsonl')
    parser.add_argument('--eval', type=Path, default=root/'data/eval.jsonl')
    parser.add_argument('--test-scenarios', type=Path, default=root/'data/test_scenarios.jsonl')
    parser.add_argument('--expected-train', type=int, default=200)
    parser.add_argument('--expected-eval', type=int, default=20)
    parser.add_argument('--require-system', action='store_true', help='Require the first message of every record to be system')
    parser.add_argument('--min-rounds', type=int, default=2)
    parser.add_argument('--near-threshold', type=float, default=0.88)
    parser.add_argument('--strict-near', action='store_true', help='Treat all duplicate/near-match warnings as failures')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    if not 0 <= args.near_threshold <= 1:
        parser.error('--near-threshold must be between 0 and 1')
    if args.min_rounds < 2:
        parser.error('--min-rounds must be at least 2')
    report = validate([args.train, args.eval], [args.expected_train, args.expected_eval],
                      args.near_threshold, args.require_system, args.min_rounds)
    holdout = validate_scenarios(args.test_scenarios, [args.train, args.eval], args.near_threshold)
    report['test_scenarios'] = holdout
    report['errors'].extend(holdout['errors'])
    report['warnings'].extend(holdout['warnings'])
    report['ok'] = not report['errors']
    if args.strict_near and report['warnings']:
        report['ok'] = False
    output = json.dumps(report, ensure_ascii=False, indent=2)
    if args.report:
        args.report.write_text(output + '\n', encoding='utf-8')
    # ensure output works even on legacy Windows console encodings
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    print(output)
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
