# read version from installed package
from importlib.metadata import version
from rswjax.losses import *
from rswjax.regularizers import *
from rswjax.solver import *
__version__ = version("rswjax")