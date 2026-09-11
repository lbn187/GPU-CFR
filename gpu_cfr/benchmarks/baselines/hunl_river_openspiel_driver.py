# launcher stub: the implementation is compiled in hunl_river_openspiel_driver_impl
import sys
from . import hunl_river_openspiel_driver_impl as _impl
globals().update({k: v for k, v in vars(_impl).items() if k not in {"__name__", "__file__", "__spec__", "__loader__", "__package__", "__builtins__", "__cached__", "__path__"}})
if __name__ == "__main__":
    raise SystemExit(main())
else:
    sys.modules[__name__] = _impl
