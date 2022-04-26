import pytest
import unittest
import os
import sys
from backend_react import react_app



class MyTestCase(unittest.TestCase):

    def setUp(self):
        react_app.app.testing = True
        self.app = react_app.app.test_client()

    def test_home(self):
        result = self.app.get('/')
        # Make your assertions