# read version from installed package
from importlib.metadata import version
from rswjax.losses import *
from rswjax.regularizers import *
from rswjax.solver import *
from rswjax.rsw import *
from rswjax.native_lambertw import *
__version__ = version("rswjax")