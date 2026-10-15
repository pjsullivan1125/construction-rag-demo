"""Step 5: GROUNDED ANSWER.

"Grounded" means the model may only use the sources we hand it, must cite
their IDs, and must admit when the answer is not there. We build a prompt
from the retrieved chunks; if an API key exists we send it to Claude,
otherwise we just show the prompt.
"""
import os

from rag.chunking import Chunk

# Claude model to call. Haiku is the small, fast, cheap model: plenty for answering
# from three short, already-retrieved passages. Swap in "claude-sonnet-5-5" for more reasoning.
MODEL = "claude-haiku-5-5"

NOT_FOUND = "I cannot find this in the provided documents."

INSTRUCTIONS = (
    "Answer the question using ONLY the sources below. "
    "Cite the source ID in square brackets after each fact, like [RFI-004]. "
    f"If the sources do not contain the answer, reply exactly: {NOT_FOUND} "
    "Do not use outside knowledge."
)


def build_prompt(question: str, results: list[tuple[Chunk, float]]) -> str:
    """Instructions + numbered sources (each labelled with its ID) + the question."""
    sources = "\n\n".join(f"[{chunk.id}]\n{chunk.text}" for chunk, _score in results)
    return f"{INSTRUCTIONS}\n\nSOURCES:\n\n{sources}\n\nQUESTION: {question}"


def call_claude(prompt: str) -> str:
    """Send the prompt to Claude and return the text of its reply."""
    import anthropic  # imported here so the rest of the project works without the package

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,  # room for the model's internal reasoning plus a short answer
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return "(The model declined to answer this request.)"
    # The reply is a list of blocks; keep only the text ones.
    return "".join(block.text for block in response.content if block.type == "text")


def answer(question: str, results: list[tuple[Chunk, float]]) -> tuple[str, str | None]:
    """Return (prompt, answer). The answer is None when no API key is set."""
    prompt = build_prompt(question, results)
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return prompt, None
    return prompt, call_claude(prompt)
