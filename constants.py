import os
import sys

PROJECT_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(PROJECT_DIR, 'data')

NUM_LABELS = 10

NETWORK_DEPTH = (18, 34, 50, 101, 152)