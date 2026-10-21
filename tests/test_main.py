
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from main import create_dataset, DRIVERS, FEATURES


class ForestWatchTests(unittest.TestCase):

    def test_dataset_has_expected_size(self):
        df = create_dataset(n=70)
        self.assertEqual(len(df), 70)

    def test_dataset_has_required_columns(self):
        df = create_dataset()
        for column in FEATURES + ["driver"]:
            self.assertIn(column, df.columns)

    def test_all_categories_are_present(self):
        df = create_dataset()
        self.assertEqual(set(df["driver"]), set(DRIVERS))

    def test_data_generation_is_reproducible(self):
        first = create_dataset(n=20, seed=42)
        second = create_dataset(n=20, seed=42)
        self.assertTrue(first.equals(second))


if __name__ == "__main__":
    unittest.main()