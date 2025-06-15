import pandas as pd
from data.prepare_data import split_dataset
from twisted.trial import unittest


class TestSplitDataset(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "col2": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
        })

    def test_split_ratios(self):
        df_train, df_test, df_val = split_dataset(self.data, test_size=0.2, val_size=0.2, random_state=42)
        self.assertEqual(len(df_train) + len(df_test) + len(df_val), len(self.data))
        self.assertEqual(len(df_test), int(0.2 * len(self.data)))
        self.assertEqual(len(df_val), int(0.2 * len(self.data)))

    def test_random_state_stability(self):
        df_train_1, df_test_1, df_val_1 = split_dataset(self.data, test_size=0.2, val_size=0.2, random_state=42)
        df_train_2, df_test_2, df_val_2 = split_dataset(self.data, test_size=0.2, val_size=0.2, random_state=42)
        self.assertTrue(df_train_1.equals(df_train_2))
        self.assertTrue(df_test_1.equals(df_test_2))
        self.assertTrue(df_val_1.equals(df_val_2))

    def test_invalid_val_size(self):
        with self.assertRaises(ValueError):
            split_dataset(self.data, test_size=0.2, val_size=0.9)

    def test_empty_dataframe(self):
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            split_dataset(empty_data, test_size=0.2, val_size=0.2)

    def test_non_dataframe_input(self):
        non_df_input = [1, 2, 3, 4, 5]
        with self.assertRaises(TypeError):
            split_dataset(non_df_input, test_size=0.2, val_size=0.2)
