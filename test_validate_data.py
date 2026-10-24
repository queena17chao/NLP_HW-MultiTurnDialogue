import copy
import json
import tempfile
import unittest
from pathlib import Path
from validate_data import validate, validate_scenarios


class ValidatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='validator_test_', dir=Path(__file__).resolve().parent)
        self.root = Path(self.tmp.name)
        self.row = {'messages': [{'role': r, 'content': c} for r, c in zip(
            ['user', 'assistant', 'user', 'assistant'],
            ['今天下雨，說句情話。', '雨傘想替我靠近你。', '要自然一點。', '雨停了也想陪你走。'])]}

    def tearDown(self):
        self.tmp.cleanup()

    def check(self, train, evaluation=None):
        paths = []
        for name, rows in [('train', train), ('eval', evaluation or [])]:
            p = self.root / (name + '.jsonl')
            p.write_text('\n'.join(json.dumps(x, ensure_ascii=False) for x in rows) + ('\n' if rows else ''), encoding='utf-8')
            paths.append(p)
        return validate(paths, expected_counts=[len(train), len(evaluation or [])], near_threshold=0.9)

    def test_valid(self):
        self.assertEqual(self.check([self.row])['errors'], [])

    def test_invalid_records(self):
        bad = [None, {}, {'messages': 'wrong'}, {'messages': [None]}, {'messages': []}]
        for value in bad:
            with self.subTest(value=value):
                self.assertTrue(self.check([value])['errors'])

    def test_roles_rounds_and_empty(self):
        for change in ('role', 'round', 'empty', 'number', 'system'):
            row = copy.deepcopy(self.row)
            if change == 'role': row['messages'][1]['role'] = 'user'
            if change == 'round': row['messages'] = row['messages'][:2]
            if change == 'empty': row['messages'][2]['content'] = ' \t'
            if change == 'number': row['messages'][0]['content'] = 4
            if change == 'system': row['messages'].append({'role': 'system', 'content': '錯誤位置'})
            with self.subTest(change=change): self.assertTrue(self.check([row])['errors'])

    def test_optional_system(self):
        row = copy.deepcopy(self.row)
        row['messages'].insert(0, {'role': 'system', 'content': '請用繁體中文。'})
        self.assertEqual(self.check([row])['errors'], [])

    def test_duplicates_and_leakage(self):
        self.assertTrue(self.check([self.row, self.row])['errors'])
        self.assertTrue(self.check([self.row], [self.row])['errors'])
        row = copy.deepcopy(self.row)
        row['messages'][0]['content'] = '今天下雨 說句情話'
        self.assertTrue(self.check([self.row], [row])['errors'])

    def test_shared_answer_and_prompt(self):
        for role in ('user', 'assistant'):
            row = copy.deepcopy(self.row)
            for m in row['messages']:
                if m['role'] != role: m['content'] += '另外一種寫法'
            self.assertTrue(self.check([self.row], [row])['errors'])

    def test_file_errors(self):
        p = self.root / 'bad.jsonl'
        for content in ('{bad}\n', '\n', '{"messages":[],"messages":[]}\n'):
            p.write_text(content, encoding='utf-8')
            self.assertTrue(validate([p])['errors'])
        self.assertTrue(validate([self.root / 'missing'])['errors'])

    def test_count_mismatch(self):
        p = self.root / 'empty.jsonl'
        p.write_text('', encoding='utf-8')
        self.assertTrue(validate([p], expected_counts=[1])['errors'])

    def test_holdout_scenarios(self):
        rows = [{'id': f'test-{i}', 'scenario': f'獨立情境{i}', 'first_user': f'首輪甲乙{i}',
                 'follow_up_user': f'後續丙丁{i}', 'checks': ['承接前文']} for i in range(10)]
        p = self.root / 'test.jsonl'
        def run(values):
            p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values), encoding='utf-8')
            return validate_scenarios(p, [], near_threshold=1.0)
        self.assertEqual(run(rows)['errors'], [])
        self.assertTrue(run(rows[:9])['errors'])
        self.assertTrue(run([None]*10)['errors'])
        bad = copy.deepcopy(rows)
        bad[1]['id'] = bad[0]['id']
        self.assertTrue(run(bad)['errors'])
        bad = copy.deepcopy(rows)
        bad[1]['follow_up_user'] = ''
        self.assertTrue(run(bad)['errors'])
        bad = copy.deepcopy(rows)
        bad[1]['checks'] = 'wrong'
        self.assertTrue(run(bad)['errors'])

    def test_holdout_prompt_leakage(self):
        rows = [{'id': f'test-{i}', 'scenario': f'情境{i}', 'first_user': f'新的提示甲{i}',
                 'follow_up_user': f'新的要求乙{i}', 'checks': ['檢查']} for i in range(10)]
        rows[0]['first_user'] = self.row['messages'][0]['content']
        p = self.root / 'test.jsonl'
        train = self.root / 'train.jsonl'
        p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
        train.write_text(json.dumps(self.row, ensure_ascii=False)+'\n', encoding='utf-8')
        self.assertTrue(validate_scenarios(p, [train])['errors'])

    def test_required_system_and_round_floor(self):
        p = self.root / 'record.jsonl'
        def run(row, **kwargs):
            p.write_text(json.dumps(row, ensure_ascii=False)+'\n', encoding='utf-8')
            return validate([p], **kwargs)
        self.assertTrue(run(self.row, require_system=True)['errors'])
        row = copy.deepcopy(self.row)
        row['messages'].insert(0, {'role': 'system', 'content': '你是情話高手，請用臺灣繁體中文回答。'})
        self.assertEqual(run(row, require_system=True)['errors'], [])
        self.assertTrue(run(row, require_system=True, min_rounds=3)['errors'])
        row['messages'].extend([
            {'role': 'user', 'content': '她問雨停了還散步嗎，怎麼接？'},
            {'role': 'assistant', 'content': '你可以說：「雨是臨時的邀請，想跟妳走走才是原本的計畫。」'},
        ])
        self.assertEqual(run(row, require_system=True, min_rounds=3)['errors'], [])

    def test_holdout_third_turn_is_checked(self):
        rows = [{'id': f'test-{i}', 'scenario': f'第三輪情境{i}', 'first_user': f'不同首問{i}',
                 'follow_up_user': f'不同追問{i}', 'third_user': f'不同第三問{i}',
                 'checks': ['連續接招']} for i in range(10)]
        p = self.root / 'test.jsonl'
        train = self.root / 'train.jsonl'
        train.write_text(json.dumps(self.row, ensure_ascii=False)+'\n', encoding='utf-8')
        def run():
            p.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows), encoding='utf-8')
            return validate_scenarios(p, [train])
        self.assertEqual(run()['errors'], [])
        rows[0]['third_user'] = self.row['messages'][0]['content']
        self.assertTrue(run()['errors'])
        rows[0]['third_user'] = ''
        self.assertTrue(run()['errors'])


if __name__ == '__main__':
    unittest.main()
