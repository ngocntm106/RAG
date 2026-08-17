import logging
from typing import Dict, Any, Optional

try:
    from .neo4j_client import Neo4jClient
    from .embedder import VietnameseEmbedder
    from .retriever import MultiHopGraphRetriever
    from .generator import GeminiGenerator
except ImportError:
    from neo4j_client import Neo4jClient
    from embedder import VietnameseEmbedder
    from retriever import MultiHopGraphRetriever
    from generator import GeminiGenerator

logger = logging.getLogger(__name__)

class GraphRAGPipeline:
    """
    End-to-end Multi-hop Graph RAG Pipeline:
    1. Retrieve direct chunks and multi-hop graph relations from Neo4j.
    2. Format prompt with schema instructions and strict grounding.
    3. Generate answer using Gemini LLM.
    """

    def __init__(
        self,
        retriever: Optional[MultiHopGraphRetriever] = None,
        generator: Optional[GeminiGenerator] = None,
    ):
        self.retriever = retriever or MultiHopGraphRetriever()
        self.generator = generator or GeminiGenerator()

    def query(
        self,
        question: str,
        top_k: int = 3,
        max_hops: int = 1,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end QA with configurable multi-hop graph depth.
        """
        # 1. Retrieve context
        retrieval_result = self.retriever.retrieve_context(
            query=question,
            top_k=top_k,
            max_hops=max_hops,
        )

        context_text = retrieval_result["formatted_context"]

        # 2. Generate answer via LLM
        gen_result = self.generator.generate(
            question=question,
            context=context_text,
            temperature=temperature,
        )

        return {
            "question": question,
            "max_hops": max_hops,
            "top_k": top_k,
            "retrieval": retrieval_result,
            "answer": gen_result.get("answer", ""),
            "prompt": gen_result.get("prompt", ""),
            "system_prompt": gen_result.get("system_prompt", ""),
            "model": gen_result.get("model", ""),
            "status": gen_result.get("status", "unknown"),
        }
