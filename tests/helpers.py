"""Test helpers.

``load_app()`` imports ``app.py`` with a stubbed Streamlit runtime so the
retrieval and evaluation functions can be tested without a running server.
The real ``streamlit`` module is restored afterwards, which keeps the
end-to-end dashboard test (Streamlit ``AppTest``) working in the same
process.
"""

import importlib.util
import sys
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = PROJECT_ROOT / "app.py"
NOTE_PATH = PROJECT_ROOT / "sample_clinical_note.txt"

CLINICAL_NOTE = (
    "Sample Clinical Note\n"
    "The patient reports a headache for two days.\n"
    "The patient reports a sore throat for three days.\n"
    "The patient has a dry cough.\n"
    "The patient denies shortness of breath.\n"
    "The patient reports drinking warm water.\n"
)


class _SessionState(dict):
    """Minimal stand-in for ``st.session_state``."""


class _Stub:
    """Every unknown Streamlit attribute becomes a harmless stub."""

    session_state = _SessionState()

    def __init__(self, name="stub"):
        self._name = name

    def __call__(self, *a, **k):
        return _Stub()

    def __getattr__(self, name):
        return _Stub(name)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        return iter(())

    def __bool__(self):
        return False


def build_streamlit_stub():
    """A module that behaves enough like Streamlit to import ``app.py``."""

    module = types.ModuleType("streamlit")
    module.__getattr__ = lambda name: _Stub(name)
    module.session_state = _SessionState()

    def columns(spec=1, **kwargs):
        if isinstance(spec, int):
            return [_Stub() for _ in range(spec)]
        return [_Stub() for _ in spec]

    module.columns = columns
    module.file_uploader = lambda *a, **k: None
    module.radio = lambda *a, **k: None
    module.text_input = lambda *a, **k: ""
    module.text_area = lambda *a, **k: ""

    return module


def load_app():
    """Import ``app.py`` as a module and return it."""

    real_streamlit = sys.modules.get("streamlit")

    sys.modules["streamlit"] = build_streamlit_stub()

    try:
        spec = importlib.util.spec_from_file_location(
            "clinical_app_under_test", APP_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["clinical_app_under_test"] = module
        spec.loader.exec_module(module)
    finally:
        if real_streamlit is not None:
            sys.modules["streamlit"] = real_streamlit
        else:
            sys.modules.pop("streamlit", None)

    return module


def read_note():
    """The demonstration note that ships with the project."""

    if NOTE_PATH.exists():
        return NOTE_PATH.read_text(encoding="utf-8")

    return CLINICAL_NOTE
