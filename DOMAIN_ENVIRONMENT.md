# Reproducible offline environment

Upstream: https://github.com/python-streamz/streamz
Upstream SHA: 5267116f49fc2257e39b44abd8d1e5c5460715f8
Runtime: 3.13.14 (tags/v3.13.14:fd17997, Jun 10 2026, 13:03:48) [MSC v.1944 64 bit (AMD64)]
Platform: win32

Create an isolated virtual environment, then run from this checkout:
```
python -m pip install -r requirements-domain.lock
python -m pip install -e .
```
Set `PYTHONUTF8=1` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` before tests.
```
python -m pytest -q -o addopts= -o log_cli=false -p pytest_asyncio.plugin streamz/tests/test_core.py -m "not network and not slow" -k "not timed_window and not delay and not rate_limit and not map_async and not timeout"
```
Measured selected pytest process: 2.052 seconds.
The core selection excludes wall-clock-sensitive timing tests and map_async timing tests. Runtime needs tornado, toolz and zict; distributed is used only by upstream test fixtures. No Dask cluster is launched.

Tests use synthetic in-memory or temporary-file inputs under the upstream license.
No account, cloud service, browser or external database is needed. Recreate the
environment from this immutable commit to reset the starting state.
