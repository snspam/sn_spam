import os
import sys
one_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
two_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(1, one_up)
sys.path.insert(2, two_up)

from app import config
from app.tests import test_utils
from independent.scripts import independent
from independent.scripts import classification
from independent.scripts import content_features
from independent.scripts import graph_features
from independent.scripts import relational_features
from analysis import util
