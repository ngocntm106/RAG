import logging
from typing import List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

try:
    from .config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION
except ImportError:
    from config import EMBEDDING_MODEL_NAME, EMBEDDING_DIMENSION

logger = logging.getLogger(__name__)

class VietnameseEmbedder:
    """
    Singleton embedder using Vietnamese MSMARCO MiniLM model.
    Runs efficiently on CPU with PyTorch and HuggingFace transformers.
    """
    _instance: Optional["VietnameseEmbedder"] = None

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model_name = model_name
        logger.info(f"Loading Vietnamese embedding model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.dimension = EMBEDDING_DIMENSION

    @classmethod
    def get_instance(cls, model_name: str = EMBEDDING_MODEL_NAME) -> "VietnameseEmbedder":
        if cls._instance is None:
            cls._instance = cls(model_name=model_name)
        return cls._instance

    @staticmethod
    def _mean_pooling(model_output, attention_mask):
        """Mean pooling to produce fixed-size sentence vector from token embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single text query and return normalized vector of floats.
        """
        if not text or not text.strip():
            raise ValueError("Query text cannot be empty.")

        encoded_input = self.tokenizer(
            text.strip(),
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed multiple documents/chunks in batch.
        """
        if not texts:
            return []

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings.tolist()
