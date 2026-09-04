from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """Données reçues par l'endpoint POST /ask."""

    question: str = Field(
        ...,
        description="Question posée au système RAG.",
        examples=["Je cherche un événement jazz à Paris."],
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Refuse une question vide ou composée uniquement d'espaces."""
        question = value.strip()

        if not question:
            raise ValueError("La question ne peut pas être vide.")

        return question


class AskResponse(BaseModel):
    """Réponse générée par le système RAG."""

    question: str
    answer: str