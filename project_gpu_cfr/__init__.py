"""Import shim: expose the artifact root as the ``project_gpu_cfr`` package.

The test suite imports the solver as ``project_gpu_cfr.gpu_cfr.*`` (its layout
in the original research tree). In this artifact ``gpu_cfr/`` sits at the
root, so this package redirects its search path one level up, making
``project_gpu_cfr.gpu_cfr`` resolve to ``<artifact>/gpu_cfr`` unchanged.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parent.parent)]
