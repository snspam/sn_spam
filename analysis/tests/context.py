import os
import sys
one_level_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, one_level_up)

from app import config
from app.tests import test_utils
from relational.scripts import generator
from relational.scripts import pred_builder
from analysis import analysis
from analysis import connections
from analysis import evaluation
from analysis import interpretability
from analysis import label
from analysis import purity
from analysis import util
