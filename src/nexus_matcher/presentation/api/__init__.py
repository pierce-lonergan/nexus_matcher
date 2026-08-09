"""
nexus_matcher.presentation.api | Layer: PRESENTATION
REST API interface for NexusMatcher.

Importing this subpackage must NOT require the `api` extra.

It did until 2026-08-09. This file ran `from ...api.app import create_app, run_dev_server`
at module scope; app.py imports fastapi at module scope; fastapi ships only in `[api]`. So
on a default `pip install nexus-matcher`, merely naming a subpackage of the installed
package -- `import nexus_matcher.presentation.api` -- raised
`ModuleNotFoundError: No module named 'fastapi'`. That is NM-0007 one directory down: the
top-level package was taught to defer create_app, and this was left behind, so the fix held
for `from nexus_matcher import create_app` and not for the equally documented
`from nexus_matcher.presentation.api import create_app`.

Both names are resolved lazily below, and both are absent from `__all__` -- see the note
there -- so the subpackage imports, introspects and documents itself with only the core
dependencies present, and asking for a factory produces an error naming the extra.
"""

from __future__ import annotations

# Export -> the module that must be importable for it to RESOLVE. Both need fastapi:
# run_dev_server does not import it directly, but it lives in app.py, which does, so
# fetching either name executes that module.
#
# This is deliberately about resolution, not about calling. run_dev_server additionally
# does `import uvicorn` in its body, and uvicorn ships in the same `api` extra, so in
# practice the two arrive together -- but if uvicorn alone were missing, the name would
# still resolve here and fail when called. Widening this to cover call-time dependencies
# would make __dir__ hide a name that getattr() hands back perfectly well.
_OPTIONAL_EXPORT_REQUIRES = {
    "create_app": "fastapi",
    "run_dev_server": "fastapi",
}

# Reachable through __getattr__, deliberately NOT listed in __all__.
#
# __all__ is exactly the list `from ... import *` walks, and it is a promise that a name
# imports. Neither of these holds on a bare install, so listing them would reintroduce the
# defect above through the star-import door -- one name whose dependency lives in an extra
# takes the whole statement down. This mirrors the top-level package, where create_app is
# excluded from __all__ for the same reason and for the same evidence.
#
# The cost is real and accepted: with the extra installed,
# `from nexus_matcher.presentation.api import *` now binds nothing. Nothing in this repo
# used it, explicit imports and `dir()` are unaffected, and the alternative -- an __all__
# whose contents depend on what happens to be installed -- makes the promise conditional
# and defeats every static analyser that reads it.
__all__: list[str] = []


def __getattr__(name: str):
    """Resolve a factory on demand, with the missing extra named in the error."""
    if name not in _OPTIONAL_EXPORT_REQUIRES:
        # Not a lazy export: answer exactly as a module without __getattr__ would. A lazy
        # hook that returns something for every name would make hasattr() always true and
        # turn a typo into a confusing failure much later.
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    try:
        from nexus_matcher.presentation.api import app as _app
    except ModuleNotFoundError as exc:
        # Imported here, not at module scope, so this file never depends on the parent
        # package being finished initialising.
        from nexus_matcher import _EXTRA_FOR_MODULE

        extra = _EXTRA_FOR_MODULE.get((exc.name or "").partition(".")[0])
        if extra is None:
            raise
        raise ModuleNotFoundError(
            f"nexus_matcher.presentation.api.{name} needs the optional '{extra}' extra, "
            f"which is not installed (no module named {exc.name!r}). Install it with: "
            f"pip install nexus-matcher[{extra}]",
            name=exc.name,
        ) from exc

    return getattr(_app, name)


def __dir__() -> list[str]:
    """
    Advertise the factories -- but only when they will actually resolve.

    dir() is not merely a display list. inspect.getmembers(), help(), pydoc and rlcompleter
    tab-completion all walk it and getattr() every entry, so one name listed here that
    raises makes ALL of them blow up. Listing both names unconditionally would mean this
    module imports on a bare install and then breaks the moment anyone looks at it, which
    is a worse trade than not advertising: it turns a clear "install the extra" error into
    four broken tools. pydoc is the quiet one -- pydoc.doc() catches the ImportError and
    renders the error text where the documentation should be, without raising.

    The tempting alternative, an exception that is both ModuleNotFoundError (so the message
    still names the extra) and AttributeError (so getattr-based introspection skips it),
    does not exist: the two have incompatible C-level layouts and cannot be subclassed
    together -- `TypeError: multiple bases have instance lay-out conflict`. Deciding
    availability here is what is left, and it is also cheaper: find_spec only consults the
    import system's finders, so it neither executes fastapi nor pays its import cost.
    """
    from importlib.util import find_spec

    available = []
    for name, required in _OPTIONAL_EXPORT_REQUIRES.items():
        try:
            if find_spec(required) is not None:
                available.append(name)
        except (ImportError, ValueError):  # a broken or namespace-shadowed install
            continue
    return sorted({*globals(), *__all__, *available})
