import warnings

warnings.simplefilter('ignore')

from ._model import scCAVAE
from ._module import scCAVAEModule
from ._task_auto import scCAVAEAutoTrainingPlan
from . import _plotting as pl
from ._api import scCAVAEAPI
from ._tuner import run_autotune

from importlib.metadata import version

package_name = "scCAVAE"
try:
    __version__ = version(package_name)
except:
    __version__ = "0.1.0"  

__all__ = [
    "scCAVAE",
    "scCAVAEModule",
    "scCAVAEAutoTrainingPlan",
    "scCAVAEAPI",
    "pl",
    "run_autotune"
]