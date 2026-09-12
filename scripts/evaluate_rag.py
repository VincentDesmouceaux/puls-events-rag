import json
import os
from pathlib import Path

from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness, LLMContextRecall, ResponseRelevancy

from scripts.rag_chain import answer_question_with_context


EVALUATION_FILE = Path("data/evaluation/rag_questions.json")


def load_questions() -> list[dict]:
    """Charge les scénarios d'évaluation RAG."""
    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def evaluate_terms(
    answer: str,
    expected_terms: list[str],
) -> dict:
    """Évalue la présence des termes attendus."""
    if not expected_terms:
        return {
            "score": 1.0,
            "found_terms": [],
        }

    normalized_answer = answer.lower()

    found_terms = [
        term
        for term in expected_terms
        if term.lower() in normalized_answer
    ]

    score = len(found_terms) / len(expected_terms)

    return {
        "score": score,
        "found_terms": found_terms,
    }


def evaluate_answer(
    answer: str,
    expected_terms: list[str],
    expected_facts: list[str] | None = None,
) -> dict:
    """Évalue une réponse avec les termes et faits attendus."""
    expected_facts = expected_facts or []

    terms_evaluation = evaluate_terms(
        answer,
        expected_terms,
    )

    facts_evaluation = evaluate_terms(
        answer,
        expected_facts,
    )

    if expected_facts:
        score = (
            terms_evaluation["score"]
            + facts_evaluation["score"]
        ) / 2
    else:
        score = terms_evaluation["score"]

    if score == 1:
        label = "correcte"
    elif score > 0:
        label = "partiellement correcte"
    else:
        label = "incorrecte"

    return {
        "score": score,
        "label": label,
        "found_terms": terms_evaluation["found_terms"],
        "fact_score": facts_evaluation["score"],
        "found_facts": facts_evaluation["found_terms"],
    }


def build_ragas_dataset(
    questions: list[dict],
) -> tuple[EvaluationDataset, list[dict]]:
    """
    Exécute le système RAG et construit le dataset attendu par Ragas.

    Retourne également les résultats intermédiaires afin de conserver
    l'évaluation métier déterministe existante.
    """
    ragas_samples = []
    generated_results = []

    for item in questions:
        rag_result = answer_question_with_context(
            item["question"]
        )

        business_evaluation = evaluate_answer(
            rag_result["answer"],
            item["expected_answer_contains"],
            item.get("expected_facts", []),
        )

        generated_results.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answer": rag_result["answer"],
                "reference_answer": item["reference_answer"],
                "retrieved_contexts": rag_result[
                    "retrieved_contexts"
                ],
                "business_evaluation": business_evaluation,
            }
        )

        ragas_samples.append(
            {
                "user_input": item["question"],
                "response": rag_result["answer"],
                "retrieved_contexts": rag_result[
                    "retrieved_contexts"
                ],
                "reference": item["reference_answer"],
            }
        )

    return (
        EvaluationDataset.from_list(ragas_samples),
        generated_results,
    )


def get_ragas_models():
    """Configure les modèles Mistral utilisés par Ragas."""
    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY is missing")

    evaluator_llm = LangchainLLMWrapper(
        ChatMistralAI(
            model="ministral-3b-latest",
            api_key=api_key,
            temperature=0,
        )
    )

    evaluator_embeddings = LangchainEmbeddingsWrapper(
        MistralAIEmbeddings(
            model="mistral-embed",
            api_key=api_key,
        )
    )

    return evaluator_llm, evaluator_embeddings


def evaluate_with_ragas(
    dataset: EvaluationDataset,
):
    """Évalue le dataset avec les métriques Ragas."""
    evaluator_llm, evaluator_embeddings = get_ragas_models()

    metrics = [
        Faithfulness(
            llm=evaluator_llm,
        ),
        ResponseRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
            strictness=1,
        ),
        LLMContextRecall(
            llm=evaluator_llm,
        ),
    ]

    return evaluate(
        dataset=dataset,
        metrics=metrics,
    )


def print_business_results(
    generated_results: list[dict],
) -> None:
    """Affiche les résultats de l'évaluation métier."""
    print("\n")
    print("=" * 80)
    print("ÉVALUATION MÉTIER")
    print("=" * 80)

    for item in generated_results:
        evaluation = item["business_evaluation"]

        print(f"\nScénario {item['id']}")
        print(f"Question : {item['question']}")

        print("\nRéponse IA :")
        print(item["answer"])

        print("\nRéponse de référence :")
        print(item["reference_answer"])

        print("\nÉvaluation métier :")
        print(
            f"Score global : "
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
        print(
            f"Score des faits : "
            f"{evaluation['fact_score']:.2f}"
        )
        print(
            f"Faits trouvés : "
            f"{evaluation['found_facts']}"
        )

        print("-" * 80)


def print_ragas_results(result) -> None:
    """Affiche les scores Ragas globaux."""
    print("\n")
    print("=" * 80)
    print("ÉVALUATION RAGAS")
    print("=" * 80)

    print(result)

    print("\nRésultats détaillés :")

    dataframe = result.to_pandas()

    columns = [
        column
        for column in [
            "user_input",
            "faithfulness",
            "answer_relevancy",
            "context_recall",
        ]
        if column in dataframe.columns
    ]

    print(
        dataframe[columns].to_string(
            index=False,
        )
    )


def main() -> None:
    """Exécute l'évaluation métier puis l'évaluation Ragas."""
    questions = load_questions()

    print(
        f"{len(questions)} scénarios "
        f"d'évaluation chargés."
    )

    dataset, generated_results = build_ragas_dataset(
        questions
    )

    print_business_results(
        generated_results
    )

    print("\nLancement de l'évaluation Ragas...")

    ragas_result = evaluate_with_ragas(
        dataset
    )

    print_ragas_results(
        ragas_result
    )


if __name__ == "__main__":
    main()