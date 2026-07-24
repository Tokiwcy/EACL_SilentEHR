#!/usr/bin/env python3
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from silentehr import _train_lr, make_pilot
from sru_binary_core import (
    REFUTED,
    SUPPORTED,
    Y_REFUTED,
    Y_SUPPORTED,
    predict_sr,
    validate_direction,
    y_from_label,
)


class TestSRUBinaryCore(unittest.TestCase):
    def _fit_toy(self, n=40):
        rng = np.random.default_rng(0)
        X_pos = rng.normal(+1.0, 0.5, size=(n, 4))
        X_neg = rng.normal(-1.0, 0.5, size=(n, 4))
        X = np.vstack([X_neg, X_pos])
        y = np.array([Y_REFUTED] * n + [Y_SUPPORTED] * n)
        clf, sc = _train_lr(X, y)
        return clf, sc, X, y

    def test_explicit_negative_maps_to_refuted(self):
        clf, sc, X, y = self._fit_toy()
        neg_idx = np.where(y == Y_REFUTED)[0][:3]
        bundle = predict_sr(clf, sc, X[neg_idx])
        self.assertTrue(all(p == REFUTED for p in bundle["pred"]))

    def test_explicit_positive_maps_to_supported(self):
        clf, sc, X, y = self._fit_toy()
        pos_idx = np.where(y == Y_SUPPORTED)[0][:3]
        bundle = predict_sr(clf, sc, X[pos_idx])
        self.assertTrue(all(p == SUPPORTED for p in bundle["pred"]))

    def test_unknown_predictions_partition(self):
        clf, sc, X, y = self._fit_toy()
        holdout = np.vstack([X[:5], X[-5:]])
        bundle = predict_sr(clf, sc, holdout)
        pred_u = bundle["pred"]
        urr = np.mean(pred_u == REFUTED)
        usr = np.mean(pred_u == SUPPORTED)
        self.assertTrue(np.isclose(urr + usr, 1.0))

    def test_direction_sanity(self):
        clf, sc, X, y = self._fit_toy()
        bundle = predict_sr(clf, sc, X)
        diag = validate_direction(y, bundle["p_ref"], context="toy")
        self.assertTrue(diag["direction_ok"])
        self.assertGreater(diag["mean_p_ref_on_refuted"], diag["mean_p_ref_on_supported"])

    def test_classes_are_integer_labels(self):
        clf, sc, X, y = self._fit_toy()
        print("clf.classes_:", list(clf.classes_))
        bundle = predict_sr(clf, sc, X[:1])
        self.assertIn(Y_REFUTED, bundle["classes"])
        self.assertIn(Y_SUPPORTED, bundle["classes"])


class TestIIDFamilySplit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        out = ROOT / "data" / "controlled_full_test.jsonl"
        if not out.exists():
            make_pilot(full=True, out=out)
        cls.rows = [
            json.loads(line) for line in out.open(encoding="utf-8")
        ]

    def test_background_concept_family_single_split(self):
        from collections import defaultdict

        groups = defaultdict(list)
        for row in self.rows:
            key = (row["background_id"], row["concept_id"])
            groups[key].append(row["iid_split"])
        for key, splits in groups.items():
            with self.subTest(family=key):
                self.assertEqual(len(set(splits)), 1)


if __name__ == "__main__":
    unittest.main()
