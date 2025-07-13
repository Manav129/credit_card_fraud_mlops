import unittest
import pandas as pd

from src.data_loader import load_data, preprocess_data


class TestDataLoader(unittest.TestCase):

    def test_load_data(self):
        df = load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn("Class", df.columns)

    def test_preprocess_data(self):
        df = load_data()
        X_train, X_test, y_train, y_test, scaler = preprocess_data(df)
        self.assertEqual(X_train.shape[0] + X_test.shape[0], df.shape[0])
        self.assertEqual(X_train.shape[1], X_test.shape[1])
        self.assertEqual(len(y_train) + len(y_test), df.shape[0])


if __name__ == "__main__":
    unittest.main()
