import unittest
from src import train

class TestTrain(unittest.TestCase):
    def test_training_runs(self):
        try:
            train.train()
        except Exception as e:
            self.fail(f"Training crashed with error: {e}")

if __name__ == "__main__":
    unittest.main()
