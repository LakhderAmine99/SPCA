import argparse
import unittest

import torch

from index import make_imdb_masks, make_train_mask, parse_train_ratio, parse_training_budget
from src.splits import make_fixed_masks


class LegacyTrainRatioTests(unittest.TestCase):
    def setUp(self):
        # Match IMDB's node and original train counts, with three genres.
        self.labels = torch.arange(4278) % 3
        self.original_train_mask = torch.zeros(4278, dtype=torch.bool)
        self.original_train_mask[1000:1400] = True
        self.original_val_mask = torch.zeros(4278, dtype=torch.bool)
        self.original_val_mask[1400:1800] = True
        self.original_test_mask = ~(self.original_train_mask | self.original_val_mask)

    def split_masks(self, ratio, seed=42):
        return make_imdb_masks(
            self.labels, self.original_train_mask, self.original_val_mask,
            self.original_test_mask, ratio, seed=seed, split_protocol='legacy',
        )

    def test_requested_percentages_replace_mask_and_preserve_held_out_nodes(self):
        mask_before = self.original_train_mask.clone()
        for ratio, expected_count in [(0.01, 43), (0.05, 214)]:
            with self.subTest(ratio=ratio):
                train_mask = make_train_mask(self.labels, self.original_train_mask, ratio)
                self.assertEqual(int(train_mask.sum()), expected_count)
                self.assertFalse(torch.any(train_mask & ~self.original_train_mask))
                self.assertTrue(torch.equal(self.original_train_mask, mask_before))
                self.assertEqual(train_mask.dtype, torch.bool)

    def test_sampling_is_reproducible_and_seed_controls_selection(self):
        first = make_train_mask(self.labels, self.original_train_mask, 0.05, seed=42)
        same = make_train_mask(self.labels, self.original_train_mask, 0.05, seed=42)
        other = make_train_mask(self.labels, self.original_train_mask, 0.05, seed=43)
        self.assertTrue(torch.equal(first, same))
        self.assertFalse(torch.equal(first, other))

    def test_all_genres_are_represented_proportionally(self):
        for ratio, expected_count in [(0.01, 43), (0.05, 214)]:
            with self.subTest(ratio=ratio):
                mask = make_train_mask(self.labels, self.original_train_mask, ratio)
                counts = torch.bincount(self.labels[mask], minlength=3)
                expected = torch.bincount(self.labels[self.original_train_mask], minlength=3)
                expected = expected.float() * (expected_count / 400)
                self.assertTrue(torch.all(torch.abs(counts - expected) <= 1))

    def test_full_original_train_split_is_supported(self):
        mask = make_train_mask(self.labels, self.original_train_mask, 400 / 4278)
        self.assertTrue(torch.equal(mask, self.original_train_mask))

    def test_invalid_ratios_are_rejected(self):
        for ratio in [0, -0.05, 1.01, float('nan'), 0.00001]:
            with self.subTest(ratio=ratio):
                with self.assertRaises(ValueError):
                    make_train_mask(self.labels, self.original_train_mask, ratio)

    def test_larger_ratio_does_not_take_validation_or_test_nodes(self):
        with self.assertRaisesRegex(ValueError, 'larger ratio requires a new'):
            make_train_mask(self.labels, self.original_train_mask, 0.20)

    def test_small_ratios_keep_original_held_out_sets(self):
        for ratio in [0.01, 0.05, 400 / 4278]:
            with self.subTest(ratio=ratio):
                masks, strategy = self.split_masks(ratio)
                self.assertEqual(strategy, 'original')
                self.assertTrue(torch.equal(masks['val_mask'], self.original_val_mask))
                self.assertTrue(torch.equal(masks['test_mask'], self.original_test_mask))
                self.assertTrue(torch.equal(
                    masks['train_mask'],
                    make_train_mask(self.labels, self.original_train_mask, ratio),
                ))

    def test_larger_ratios_create_complete_disjoint_stratified_splits(self):
        before = [mask.clone() for mask in [self.original_train_mask,
                  self.original_val_mask, self.original_test_mask]]
        for ratio, expected_train in [(0.10, 428), (0.20, 856), (0.60, 2567)]:
            with self.subTest(ratio=ratio):
                masks, strategy = self.split_masks(ratio)
                self.assertEqual(strategy, 'stratified')
                self.assertEqual(int(masks['train_mask'].sum()), expected_train)
                self.assertEqual(int(masks['val_mask'].sum()), 400)
                self.assertEqual(int(masks['test_mask'].sum()), 4278 - expected_train - 400)
                # Every movie belongs to exactly one split.
                membership = torch.stack(list(masks.values())).sum(dim=0)
                self.assertTrue(torch.all(membership == 1))
                for mask in masks.values():
                    self.assertEqual(mask.dtype, torch.bool)
                    counts = torch.bincount(self.labels[mask], minlength=3)
                    self.assertTrue(torch.all(torch.abs(counts - int(mask.sum()) / 3) <= 1))
        for original, previous in zip([self.original_train_mask, self.original_val_mask,
                                       self.original_test_mask], before):
            self.assertTrue(torch.equal(original, previous))

    def test_new_splits_are_reproducible_and_seed_controls_all_masks(self):
        first, _ = self.split_masks(0.10, seed=42)
        same, _ = self.split_masks(0.10, seed=42)
        other, _ = self.split_masks(0.10, seed=43)
        for name in first:
            self.assertTrue(torch.equal(first[name], same[name]))
            self.assertFalse(torch.equal(first[name], other[name]))

    def test_new_split_rejects_invalid_or_oversized_ratios(self):
        for ratio in [0, -0.1, float('nan'), 1.01, 0.00001]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.split_masks(ratio)
        for ratio in [0.906, 0.95, 1.0]:
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, 'leave room'):
                self.split_masks(ratio)


class FixedTrainRatioTests(unittest.TestCase):
    def setUp(self):
        self.labels = torch.arange(4278) % 3
        self.train_mask = torch.zeros(4278, dtype=torch.bool)
        self.train_mask[:400] = True
        self.val_mask = torch.zeros(4278, dtype=torch.bool)
        self.val_mask[400:800] = True
        self.test_mask = ~(self.train_mask | self.val_mask)

    def split_masks(self, ratio, seed=42, test_ratio=0.10, test_count=None):
        return make_imdb_masks(
            self.labels, self.train_mask, self.val_mask, self.test_mask,
            ratio, seed=seed, test_ratio=test_ratio, test_count=test_count,
        )

    def test_all_ratios_use_identical_holdouts_and_exact_training_counts(self):
        baseline, _ = self.split_masks(0.01)
        for ratio, count in [(0.01, 43), (0.05, 214), (0.20, 856),
                             (0.60, 2567), (0.80, 3422)]:
            with self.subTest(ratio=ratio):
                masks, strategy = self.split_masks(ratio)
                self.assertEqual(strategy, 'fixed-nested')
                self.assertEqual(int(masks['train_mask'].sum()), count)
                self.assertEqual(int(masks['val_mask'].sum()), 400)
                self.assertEqual(int(masks['test_mask'].sum()), 428)
                for split in ['val_mask', 'test_mask']:
                    self.assertTrue(torch.equal(masks[split], baseline[split]))
                membership = torch.stack(list(masks.values())).sum(dim=0)
                self.assertTrue(torch.all(membership <= 1))
                self.assertEqual(int((membership == 0).sum()), 3450 - count)

    def test_training_sets_grow_by_adding_nodes_and_calls_are_order_independent(self):
        previous = torch.zeros_like(self.train_mask)
        for ratio in [0.01, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80]:
            masks, _ = self.split_masks(ratio)
            self.assertFalse(torch.any(previous & ~masks['train_mask']))
            previous = masks['train_mask']
        large, _ = self.split_masks(0.80)
        small, _ = self.split_masks(0.20)
        self.assertFalse(torch.any(small['train_mask'] & ~large['train_mask']))

    def test_masks_are_reproducible_and_seed_changes_membership(self):
        first, _ = self.split_masks(0.20)
        same, _ = self.split_masks(0.20)
        other, _ = self.split_masks(0.20, seed=43)
        for split in first:
            self.assertTrue(torch.equal(first[split], same[split]))
            self.assertFalse(torch.equal(first[split], other[split]))

    def test_imbalanced_training_prefixes_remain_proportionate(self):
        self.labels = torch.tensor([0] * 2400 + [1] * 1300 + [2] * 578)
        for ratio in [0.01, 0.05, 0.20, 0.60, 0.80]:
            with self.subTest(ratio=ratio):
                masks, _ = self.split_masks(ratio)
                pool = ~(masks['val_mask'] | masks['test_mask'])
                pool_counts = torch.bincount(self.labels[pool], minlength=3)
                actual = torch.bincount(self.labels[masks['train_mask']], minlength=3)
                expected = pool_counts.double() * int(masks['train_mask'].sum()) / int(pool.sum())
                self.assertTrue(torch.all(torch.abs(actual - expected) <= 1))

    def test_original_masks_and_global_rng_are_unchanged(self):
        before = [mask.clone() for mask in [self.train_mask, self.val_mask, self.test_mask]]
        rng_before = torch.get_rng_state().clone()
        self.split_masks(0.60)
        for original, copy in zip([self.train_mask, self.val_mask, self.test_mask], before):
            self.assertTrue(torch.equal(original, copy))
        self.assertTrue(torch.equal(torch.get_rng_state(), rng_before))

    def test_custom_test_ratio_is_independent_of_training_ratio(self):
        first, _ = self.split_masks(0.20, test_ratio=0.20)
        second, _ = self.split_masks(0.60, test_ratio=0.20)
        self.assertEqual(int(first['test_mask'].sum()), 856)
        for split in ['val_mask', 'test_mask']:
            self.assertTrue(torch.equal(first[split], second[split]))
        with self.assertRaisesRegex(ValueError, 'fixed training pool'):
            self.split_masks(0.80, test_ratio=0.20)

    def test_full_training_pool_has_no_unused_labels(self):
        numeric, _ = self.split_masks(3450 / 4278)
        full, _ = self.split_masks('full')
        eighty, _ = self.split_masks(0.80)
        self.assertEqual(int(full['train_mask'].sum()), 3450)
        self.assertTrue(torch.all(torch.stack(list(full.values())).sum(dim=0) == 1))
        self.assertFalse(torch.any(eighty['train_mask'] & ~full['train_mask']))
        for split in full:
            self.assertTrue(torch.equal(full[split], numeric[split]))
        for split in ['val_mask', 'test_mask']:
            self.assertTrue(torch.equal(full[split], eighty[split]))

    def test_exact_800_test_nodes_on_each_dataset_with_fixed_holdouts_and_nested_training(self):
        for dataset, total, val_count, classes in [('dblp', 4057, 400, 4),
                                                  ('imdb', 4278, 400, 3),
                                                  ('acm', 4025, 402, 3)]:
            with self.subTest(dataset=dataset):
                labels = torch.arange(total) % classes
                baseline = None
                previous = torch.zeros(total, dtype=torch.bool)
                for budget in [0.01, 0.20, 0.60, 'full']:
                    masks = make_fixed_masks(labels, budget, val_count, test_count=800)
                    self.assertEqual(int(masks['test_mask'].sum()), 800)
                    self.assertEqual(int(masks['val_mask'].sum()), val_count)
                    expected_train = total - val_count - 800 if budget == 'full' else round(total * budget)
                    self.assertEqual(int(masks['train_mask'].sum()), expected_train)
                    self.assertTrue(torch.all(torch.stack(list(masks.values())).sum(dim=0) <= 1))
                    self.assertFalse(torch.any(previous & ~masks['train_mask']))
                    previous = masks['train_mask']
                    if baseline is not None:
                        for split in ['val_mask', 'test_mask']:
                            self.assertTrue(torch.equal(masks[split], baseline[split]))
                    baseline = masks
                self.assertTrue(torch.all(torch.stack(list(baseline.values())).sum(dim=0) == 1))

    def test_exact_count_overrides_api_ratio_and_is_reproducible(self):
        first, _ = self.split_masks(0.60, test_count=800)
        same, _ = self.split_masks(0.60, test_ratio=0.50, test_count=800)
        other, _ = self.split_masks(0.60, seed=43, test_count=800)
        for split in first:
            self.assertTrue(torch.equal(first[split], same[split]))
            self.assertFalse(torch.equal(first[split], other[split]))

    def test_invalid_counts_and_training_budgets_that_cannot_fit_are_rejected(self):
        for count in [0, -1, 800.5, True, '800', 1, 4278]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.split_masks(0.20, test_count=count)
        with self.assertRaisesRegex(ValueError, 'fixed training pool'):
            self.split_masks(0.80, test_count=800)

    def test_exact_count_requires_fixed_protocol(self):
        with self.assertRaisesRegex(ValueError, 'test_count requires'):
            make_imdb_masks(self.labels, self.train_mask, self.val_mask, self.test_mask,
                            0.60, split_protocol='legacy', test_count=800)

    def test_full_budget_parser_preserves_numeric_inputs_and_rejects_full_test_ratio(self):
        self.assertEqual(parse_training_budget(' full '), 'full')
        self.assertEqual(parse_training_budget('FULL'), 'full')
        self.assertEqual(parse_training_budget('80%'), 0.80)
        self.assertEqual(parse_training_budget('0.8'), 0.80)
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_training_budget('missing')
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_train_ratio('full')

    def test_full_budget_requires_fixed_protocol(self):
        with self.assertRaisesRegex(ValueError, 'requires'):
            make_imdb_masks(
                self.labels, self.train_mask, self.val_mask, self.test_mask,
                'full', split_protocol='legacy',
            )

    def test_invalid_or_oversized_budgets_are_rejected(self):
        for ratio in [0, -0.1, float('nan'), 1.01, 0.00001, 2 / 4278]:
            with self.subTest(ratio=ratio), self.assertRaises(ValueError):
                self.split_masks(ratio)
        for ratio in [0.81, 0.90, 1.0]:
            with self.subTest(ratio=ratio), self.assertRaisesRegex(ValueError, 'fixed training pool'):
                self.split_masks(ratio)
        for ratio in [0, -0.1, float('nan'), 1.0, 1.01, 0.00001]:
            with self.subTest(test_ratio=ratio), self.assertRaises(ValueError):
                self.split_masks(0.20, test_ratio=ratio)


if __name__ == '__main__':
    unittest.main()
