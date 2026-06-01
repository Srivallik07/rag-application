"""Prompt construction for grounded answer generation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from rag_app.schemas import RetrievedDocument


SYSTEM_PROMPT = """You are a factual RAG assistant.
Answer the user using only the retrieved context.
If the context does not contain the answer, say that the provided documents do not contain enough information.
Do not invent facts. Cite sources inline using [source:rank] where rank is the context item number."""

HEALTHCARE_SYSTEM_PROMPT = """You are a healthcare domain RAG clinical information assistant.
Answer only from the retrieved healthcare corpus documents.
Provide clear, structured answers for clinicians and care teams.
If the retrieved context does not support an answer, say: "The indexed healthcare documents do not contain enough information."
Do not provide personalized medical diagnosis or emergency instructions; recommend professional evaluation when appropriate.
Do not invent drugs, doses, or guidelines. Cite evidence using [source:rank].
Use plain language and mention document titles or categories when helpful."""


class PromptBuilder:
    """Create dynamic prompts from query and retrieved context."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT) -> None:
        self.template = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Conversation history:\n{chat_history}\n\n"
                    "Question:\n{question}\n\n"
                    "Retrieved context:\n{context}\n\n"
                    "Answer:",
                ),
            ]
        )

    def build_context(self, documents: list[RetrievedDocument]) -> str:
        """Format retrieved documents into a prompt-ready context block."""
        if not documents:
            return "No context was retrieved."
        blocks: list[str] = []
        for document in documents:
            source = document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            location = f"{source}"
            if page is not None:
                location = f"{location} page={page}"
            blocks.append(
                f"[source:{document.rank}] {location} score={document.score:.6f}\n"
                f"{document.content}"
            )
        return "\n\n---\n\n".join(blocks)

    def build_messages(
        self,
        question: str,
        documents: list[RetrievedDocument],
        chat_history: str | None = None,
    ) -> list[object]:
        """Build chat messages for the configured LLM."""
        context = self.build_context(documents)
        return self.template.format_messages(
            question=question,
            context=context,
            chat_history=chat_history or "No prior conversation.",
        )

    def render_prompt(
        self,
        question: str,
        documents: list[RetrievedDocument],
        chat_history: str | None = None,
    ) -> str:
        """Render a readable prompt copy for logging and response metadata."""
        return "\n\n".join(
            str(message.content)
            for message in self.build_messages(question, documents, chat_history=chat_history)
        )
