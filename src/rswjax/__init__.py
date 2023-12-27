# read version from installed package
from importlib.metadata import version
from rswjax.losses import *
from rswjax.regularizers import *
__version__ = version("rswjax")