import os
import sys
one_level_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(1, one_level_up)

from app import config
from app.tests import test_utils
from relational.scripts import comments
from relational.scripts import generator
from relational.scripts import pred_builder
from relational.scripts import psl
from relational.scripts import relational
from relational.scripts import tuffy
from relational.scripts import mrf
from analysis import util
