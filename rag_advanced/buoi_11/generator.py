import logging
import os
from typing import Optional, Dict, Any

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

try:
    from .config import GEMINI_API_KEY, GEMINI_MODEL_NAME
    from .prompts import GRAPH_RAG_SYSTEM_PROMPT, format_graph_rag_prompt
except ImportError:
    from config import GEMINI_API_KEY, GEMINI_MODEL_NAME
    from prompts import GRAPH_RAG_SYSTEM_PROMPT, format_graph_rag_prompt

logger = logging.getLogger(__name__)

class GeminiGenerator:
    """
    LLM Generator using Gemini API with Graph RAG prompt structure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.model_name = model_name or GEMINI_MODEL_NAME or "gemini-flash-latest"
        self.client = None

        if self.api_key and genai is not None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Could not initialize Gemini client: {e}")

    def generate(
        self,
        question: str,
        context: str,
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        """
        Generate answer based on retrieved Graph RAG context.
        """
        prompt = format_graph_rag_prompt(question=question, context=context)

        if not self.api_key or genai is None:
            fallback_answer = self._synthesize_from_context(question, context)
            return {
                "answer": fallback_answer,
                "prompt": prompt,
                "system_prompt": GRAPH_RAG_SYSTEM_PROMPT,
                "model": "Graph RAG Synthesizer (Local)",
                "status": "fallback_synthesized",
            }

        if self.client is None:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception as e:
                fallback_answer = self._synthesize_from_context(question, context, error_notice=str(e))
                return {
                    "answer": fallback_answer,
                    "prompt": prompt,
                    "system_prompt": GRAPH_RAG_SYSTEM_PROMPT,
                    "model": "Graph RAG Synthesizer (Local)",
                    "status": "fallback_synthesized",
                    "error": str(e),
                }

        try:
            # Generate with system instruction
            config = None
            if types is not None:
                config = types.GenerateContentConfig(
                    system_instruction=GRAPH_RAG_SYSTEM_PROMPT,
                    temperature=temperature,
                )

            models_to_try = [self.model_name]
            for fallback in ["gemini-flash-latest", "gemini-pro-latest", "gemini-3.5-flash-lite"]:
                if fallback not in models_to_try:
                    models_to_try.append(fallback)

            last_error = None
            for model_id in models_to_try:
                try:
                    response = self.client.models.generate_content(
                        model=model_id,
                        contents=prompt,
                        config=config,
                    )

                    answer_text = ""
                    if hasattr(response, "text") and response.text:
                        answer_text = response.text
                    elif hasattr(response, "candidates") and response.candidates:
                        first_cand = response.candidates[0]
                        if hasattr(first_cand, "content"):
                            answer_text = str(first_cand.content)
                    else:
                        answer_text = str(response)

                    return {
                        "answer": answer_text,
                        "prompt": prompt,
                        "system_prompt": GRAPH_RAG_SYSTEM_PROMPT,
                        "model": model_id,
                        "status": "success",
                    }
                except Exception as ex:
                    last_error = ex
                    continue

            # If all cloud models failed due to 403 / quota, synthesize answer from graph context
            fallback_answer = self._synthesize_from_context(question, context, error_notice=str(last_error))
            return {
                "answer": fallback_answer,
                "prompt": prompt,
                "system_prompt": GRAPH_RAG_SYSTEM_PROMPT,
                "model": f"{self.model_name} (Graph Context Synthesizer)",
                "status": "fallback_synthesized",
                "error": str(last_error),
            }
        except Exception as e:
            logger.error(f"Error calling Gemini API ({self.model_name}): {e}")
            fallback_answer = self._synthesize_from_context(question, context, error_notice=str(e))
            return {
                "answer": fallback_answer,
                "prompt": prompt,
                "system_prompt": GRAPH_RAG_SYSTEM_PROMPT,
                "model": f"{self.model_name} (Graph Context Synthesizer)",
                "status": "fallback_synthesized",
                "error": str(e),
            }

    def _synthesize_from_context(self, question: str, context: str, error_notice: str = "") -> str:
        """
        Synthesize structured legal answer directly from retrieved Graph RAG context
        when cloud LLM API returns permission or quota errors.
        """
        lines = []
        lines.append("### ⚖️ Trả lời Dựa trên Đồ thị Tri thức (Multi-hop Graph Context):")
        
        # Parse direct matches and relations from context text
        has_direct = "Không tìm thấy phân đoạn khớp" not in context and "=== I." in context
        has_hops = "=== II." in context and "Không phát hiện thêm liên kết" not in context

        if not has_direct and not has_hops:
            return "Ngữ cảnh dữ liệu được cung cấp không có đủ thông tin để trả lời câu hỏi này."

        # Extract relationships if present
        if has_hops:
            lines.append("\n**1. Căn cứ và Mối quan hệ Pháp lý Đa bước (Graph Relations):**")
            for line in context.split("\n"):
                if "Chuỗi quan hệ:" in line or "Văn bản liên quan:" in line or "Bước nhảy:" in line:
                    lines.append(f"- {line.strip()}")

        if has_direct:
            lines.append("\n**2. Nội dung Phân đoạn Văn bản Liên quan (Direct Chunks):**")
            for line in context.split("\n"):
                if line.startswith("[") and "Tài liệu:" in line:
                    lines.append(f"\n* **{line.strip()}**")
                elif "Phân đoạn:" in line:
                    lines.append(f"  - *{line.strip()}*")
                elif "Nội dung:" in line:
                    lines.append(f"  - {line.strip()}")

        lines.append("\n---")
        lines.append(f"> 💡 *Lưu ý*: Câu trả lời được tổng hợp tự động từ cấu trúc Đồ thị Neo4j. Để dùng LLM sinh văn bản tự nhiên, vui lòng dán **GEMINI_API_KEY** hợp lệ vào thanh Sidebar bên trái.")
        return "\n".join(lines)
