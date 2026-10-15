"""Command line: python -m rag.cli "your question" [--strategy fixed] [--mode vector] [-k 3]"""
import argparse

from rag.answer import answer
from rag.retriever import retrieve


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question about the Lakeside project documents.")
    parser.add_argument("question")
    parser.add_argument("--strategy", choices=["structured", "fixed"], default="structured")
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default="hybrid")
    parser.add_argument("-k", type=int, default=3)
    args = parser.parse_args()

    results = retrieve(args.question, args.k, args.strategy, args.mode)
    print("RETRIEVED:")
    for chunk, score in results:
        print(f"  {score:.3f}  [{chunk.id}]  ({chunk.source_file})")

    prompt, reply = answer(args.question, results)
    if reply is None:
        print("\nNo ANTHROPIC_API_KEY set, so here is the prompt that WOULD be sent:\n")
        print(prompt)
    else:
        print("\nANSWER:\n" + reply)


if __name__ == "__main__":
    main()
