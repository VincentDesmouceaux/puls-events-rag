import json
from pathlib import Path

from scripts.rag_chain import answer_question


EVALUATION_FILE = Path("data/evaluation/rag_questions.json")


def load_questions() -> list[dict]:
    """Charge les scénarios d'évaluation RAG."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_answer(
    answer: str,
    expected_terms: list[str],
) -> dict:
    """Évalue simplement la présence des termes attendus."""
    normalized_answer = answer.lower()

    found_terms = [
        term
        for term in expected_terms
        if term.lower() in normalized_answer
    ]

    score = len(found_terms) / len(expected_terms)

    if score == 1:
        label = "correcte"
    elif score > 0:
        label = "partiellement correcte"
    else:
        label = "incorrecte"

    return {
        "score": score,
        "label": label,
        "found_terms": found_terms,
    }


def main() -> None:
    """Exécute les scénarios d'évaluation du chatbot RAG."""
    questions = load_questions()

    for item in questions:
        print("=" * 80)
        print(f"Scénario {item['id']}")
        print(f"Question : {item['question']}")

        answer = answer_question(
            item["question"]
        )

        evaluation = evaluate_answer(
            answer,
            item["expected_answer_contains"],
        )

        print("\nRéponse :")
        print(answer)

        print("\nÉvaluation :")
        print(
            f"Score : "
            f"{evaluation['score']:.2f}"
        )
        print(
            f"Classement : "
            f"{evaluation['label']}"
        )
        print(
            f"Termes trouvés : "
            f"{evaluation['found_terms']}"
        )

    print("=" * 80)


if __name__ == "__main__":
    main()
