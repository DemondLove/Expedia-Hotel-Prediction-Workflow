import sys
import os
import inspect
import numpy as np
import pandas as pd
import unittest
import calc


parentPath = '/'.join(sys.path[0].split('/')[:-1])


# Types of asserts: https://docs.python.org/3/library/unittest.html

class TestCalc(unittest.TestCase):
    
    def test_add(self):
        result = calc.add(10,5)
        self.assertEqual(result, 15)

if __name__ == '__main__':
    unittest.main()
