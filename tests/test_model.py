
import unittest
from src.model import create_katakana_model
import tensorflow as tf

class TestKatakanaModel(unittest.TestCase):

    def test_model_shapes(self):
        model = create_katakana_model()
        input_shape = (1, 64, 64)  # Example input shape for a single image
        output = model(tf.random.normal([1] + list(input_shape)))
        self.assertEqual(output.shape[1], 46)  # Check if output classes are 46

if __name__ == '__main__':
    unittest.main()
