# launcher stub: the implementation is compiled in scaling_sweep_impl
import sys
import importlib.util as _u, pathlib as _p
try:
    import scaling_sweep_impl as _impl
except ImportError:
    _s = _u.spec_from_file_location("scaling_sweep_impl", next(_p.Path(__file__).resolve().parent.glob("scaling_sweep_impl.*.so")))
    _impl = _u.module_from_spec(_s); sys.modules[_s.name] = _impl; _s.loader.exec_module(_impl)
globals().update({k: v for k, v in vars(_impl).items() if k not in {"__name__", "__file__", "__spec__", "__loader__", "__package__", "__builtins__", "__cached__", "__path__"}})
if __name__ == "__main__":
    main()
else:
    sys.modules[__name__] = _impl
