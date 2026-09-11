# launcher stub: the implementation is compiled in probe_variant_order_race_impl
import sys
import importlib.util as _u, pathlib as _p
try:
    import probe_variant_order_race_impl as _impl
except ImportError:
    _s = _u.spec_from_file_location("probe_variant_order_race_impl", next(_p.Path(__file__).resolve().parent.glob("probe_variant_order_race_impl.*.so")))
    _impl = _u.module_from_spec(_s); sys.modules[_s.name] = _impl; _s.loader.exec_module(_impl)
globals().update({k: v for k, v in vars(_impl).items() if k not in {"__name__", "__file__", "__spec__", "__loader__", "__package__", "__builtins__", "__cached__", "__path__"}})
if __name__ == "__main__":
    raise SystemExit(main())
else:
    sys.modules[__name__] = _impl
