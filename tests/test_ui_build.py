"""The UI must at least *construct* under the installed Gradio.

This exists because a component-signature mismatch is invisible until launch,
and launch was never part of the test suite. Gradio 6 removed both `type` and
`show_copy_button` from `Chatbot`, and the resulting
`TypeError: unexpected keyword argument 'type'` is identical to the one you get
on Gradio < 4.44 — where the argument does not exist *yet*. Same error, opposite
cause, so a version pin would not have caught it either way.

These tests do not launch a server or need the backend running; `build_ui()`
only constructs components, and its health probe fails closed.
"""

from __future__ import annotations

import inspect
import warnings

import gradio as gr
import pytest

from te_assistant.ui import gradio_app


def test_ui_builds() -> None:
    """The whole component tree constructs under this Gradio version."""
    demo = gradio_app.build_ui()
    assert isinstance(demo, gr.Blocks)
    assert demo.blocks, "no components were created"


def test_chatbot_kwargs_are_all_accepted() -> None:
    """Whatever we pass to Chatbot must exist in this version's signature."""
    params = inspect.signature(gr.Chatbot.__init__).parameters
    for name in gradio_app._chatbot_kwargs():
        assert name in params, f"Chatbot does not accept {name!r} in gradio {gr.__version__}"


def test_chatbot_kwargs_survive_a_future_removal(monkeypatch) -> None:
    """If a later Gradio drops another of these, we degrade rather than crash."""
    monkeypatch.setattr(gradio_app, "OPTIONAL_CHATBOT_KWARGS",
                        {"type": "messages", "not_a_real_argument": True})
    assert "not_a_real_argument" not in gradio_app._chatbot_kwargs()


def test_old_gradio_is_rejected_with_a_clear_message(monkeypatch) -> None:
    """Pre-4.44 uses tuple history and cannot work — say so explicitly."""
    class _OldChatbot:
        def __init__(self, label=None, height=None):   # no `type`
            pass

    monkeypatch.setattr(gr, "Chatbot", _OldChatbot)
    monkeypatch.setattr(gr, "__version__", "4.19.2")

    with pytest.raises(RuntimeError, match="too old"):
        gradio_app._chatbot_kwargs()


def test_gradio_6_needs_no_opt_in(monkeypatch) -> None:
    """On 6+, `type` is absent because messages is the only format — not an error."""
    class _NewChatbot:
        def __init__(self, label=None, height=None):   # no `type`
            pass

    monkeypatch.setattr(gr, "Chatbot", _NewChatbot)
    monkeypatch.setattr(gr, "__version__", "6.27.0")

    assert gradio_app._chatbot_kwargs() == {}


def test_event_handlers_are_wired_with_matching_arity() -> None:
    """Gradio warns rather than raises when inputs do not match the signature.

    `on_end_session` was wired with `None` inputs while taking a session_id, so
    ending a session would have evicted nothing — a warning at build time and a
    silent no-op at runtime.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        gradio_app.build_ui()

    arity = [str(w.message) for w in caught if "arguments for function" in str(w.message)]
    assert not arity, f"handler wiring mismatch: {arity}"
