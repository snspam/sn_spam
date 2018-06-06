import os
import sys
one_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
two_up = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(1, one_up)
sys.path.insert(1, two_up)

from app import run
from app import runner
from app import config
from app.tests import test_utils
from independent.scripts import independent
from relational.scripts import relational
from analysis import analysis
