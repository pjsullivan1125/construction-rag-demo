from rag.answer import NOT_FOUND, answer, build_prompt
from rag.chunking import Chunk


def fake_results():
    chunk = Chunk("RFI-004", "Provide 4,500 psi.", "rfi_log.csv", "rfi", "03 30 00", "RFI-004")
    return [(chunk, 0.9)]


def test_prompt_has_rules_sources_and_question():
    prompt = build_prompt("Which strength governs?", fake_results())
    assert "ONLY" in prompt and NOT_FOUND in prompt
    assert "[RFI-004]\nProvide 4,500 psi." in prompt
    assert prompt.endswith("QUESTION: Which strength governs?")


def test_without_api_key_returns_prompt_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    prompt, reply = answer("Q?", fake_results())
    assert reply is None and "SOURCES" in prompt
