import logging
from typing import List, Dict, Any, Optional
try:
    from .neo4j_client import Neo4jClient
    from .embedder import VietnameseEmbedder
except ImportError:
    from neo4j_client import Neo4jClient
    from embedder import VietnameseEmbedder

logger = logging.getLogger(__name__)

DEFAULT_RELATION_TYPES = ["CAN_CU", "THAY_THE", "HOP_NHAT"]

class MultiHopGraphRetriever:
    """
    Multi-hop Graph RAG Retriever:
    1. Dense vector search over Chunk nodes in Neo4j (using cosine similarity on vector index).
    2. Multi-hop traversal over inter-document relationships (CAN_CU, THAY_THE, HOP_NHAT).
    3. Aggregation of direct chunks, document relationships, and extended context.
    """

    def __init__(
        self,
        neo4j_client: Optional[Neo4jClient] = None,
        embedder: Optional[VietnameseEmbedder] = None,
    ):
        self.client = neo4j_client or Neo4jClient()
        self.embedder = embedder or VietnameseEmbedder.get_instance()

    def vector_search(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search in Neo4j using chunk_vector_index.
        """
        query = """
        CALL db.index.vector.queryNodes('chunk_vector_index', $k, $query_vector)
        YIELD node, score
        MATCH (node)-[:PART_OF]->(doc:Document)
        RETURN 
            node.id AS chunk_id,
            node.title AS chunk_title,
            node.content AS chunk_content,
            node.level AS chunk_level,
            doc.doc_id AS doc_id,
            doc.title AS doc_title,
            doc.type AS doc_type,
            doc.year AS doc_year,
            score
        ORDER BY score DESC
        """
        results = []
        with self.client.get_session() as session:
            records = session.run(query, {"k": top_k, "query_vector": query_embedding}).data()
            for r in records:
                results.append({
                    "chunk_id": r.get("chunk_id"),
                    "title": r.get("chunk_title"),
                    "content": r.get("chunk_content"),
                    "level": r.get("chunk_level"),
                    "doc_id": r.get("doc_id"),
                    "doc_title": r.get("doc_title"),
                    "doc_type": r.get("doc_type"),
                    "doc_year": r.get("doc_year"),
                    "score": float(r.get("score", 0.0)),
                })
        return results

    def expand_multihop(
        self,
        seed_doc_ids: List[str],
        max_hops: int = 1,
        rel_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Traverse graph from seed documents up to max_hops through specified relationship types.
        Returns paths, related documents, and their chunks.
        """
        if not seed_doc_ids or max_hops <= 0:
            return {"paths": [], "related_docs": {}, "related_chunks": []}

        if rel_types is None:
            rel_types = DEFAULT_RELATION_TYPES

        rel_filter = "|".join(rel_types)
        
        # Cypher query for multi-hop graph expansion
        cypher_query = f"""
        MATCH (seed:Document)
        WHERE seed.doc_id IN $seed_doc_ids
        MATCH path = (seed)-[:{rel_filter}*1..{max_hops}]-(target:Document)
        WHERE target.doc_id <> seed.doc_id
        WITH path, seed, target, length(path) AS hops,
             [r in relationships(path) | type(r)] AS rel_names,
             [r in relationships(path) | {{
                 type: type(r),
                 from: startNode(r).doc_id,
                 from_title: startNode(r).title,
                 to: endNode(r).doc_id,
                 to_title: endNode(r).title
             }}] AS rel_details
        OPTIONAL MATCH (target)<-[:PART_OF]-(c:Chunk)
        RETURN 
            seed.doc_id AS seed_doc_id,
            seed.title AS seed_doc_title,
            target.doc_id AS target_doc_id,
            target.title AS target_doc_title,
            target.type AS target_doc_type,
            target.year AS target_doc_year,
            hops,
            rel_names,
            rel_details,
            collect(DISTINCT {{
                id: c.id,
                title: c.title,
                content: c.content,
                level: c.level
            }}) AS target_chunks
        ORDER BY hops ASC, target.doc_id
        """

        paths = []
        related_docs = {}
        all_related_chunks = []

        with self.client.get_session() as session:
            records = session.run(cypher_query, {
                "seed_doc_ids": seed_doc_ids,
            }).data()

            for r in records:
                target_id = r["target_doc_id"]
                if target_id not in related_docs:
                    related_docs[target_id] = {
                        "doc_id": target_id,
                        "title": r["target_doc_title"],
                        "type": r["target_doc_type"],
                        "year": r["target_doc_year"],
                        "min_hops": r["hops"],
                    }

                # Filter out empty chunk records if any
                valid_chunks = [
                    c for c in r.get("target_chunks", [])
                    if c and c.get("id") is not None
                ]

                paths.append({
                    "seed_id": r["seed_doc_id"],
                    "seed_title": r["seed_doc_title"],
                    "target_id": target_id,
                    "target_title": r["target_doc_title"],
                    "hops": r["hops"],
                    "rel_names": r["rel_names"],
                    "rel_details": r["rel_details"],
                })

                for chk in valid_chunks:
                    chk_entry = {**chk, "doc_id": target_id, "doc_title": r["target_doc_title"]}
                    if chk_entry not in all_related_chunks:
                        all_related_chunks.append(chk_entry)

        return {
            "paths": paths,
            "related_docs": list(related_docs.values()),
            "related_chunks": all_related_chunks,
        }

    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        max_hops: int = 1,
        rel_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Full retrieval pipeline:
        1. Query embedding via MSMARCO.
        2. Vector search in Neo4j.
        3. Multi-hop relation expansion for seed documents.
        4. Context formatting.
        """
        # 1. Embed query
        query_embedding = self.embedder.embed_query(query)

        # 2. Vector search
        vector_candidates = self.vector_search(query_embedding, top_k=top_k)

        # Extract seed document IDs
        seed_doc_ids = list(dict.fromkeys(
            c["doc_id"] for c in vector_candidates if c.get("doc_id")
        ))

        # 3. Multi-hop expansion
        multi_hop_data = {"paths": [], "related_docs": [], "related_chunks": []}
        if max_hops > 0 and seed_doc_ids:
            multi_hop_data = self.expand_multihop(
                seed_doc_ids=seed_doc_ids,
                max_hops=max_hops,
                rel_types=rel_types,
            )

        # 4. Format context text
        formatted_context = self._build_context_text(
            vector_candidates=vector_candidates,
            multi_hop_data=multi_hop_data,
            max_hops=max_hops,
        )

        return {
            "query": query,
            "top_k": top_k,
            "max_hops": max_hops,
            "vector_candidates": vector_candidates,
            "seed_documents": seed_doc_ids,
            "multi_hop": multi_hop_data,
            "formatted_context": formatted_context,
        }

    def _build_context_text(
        self,
        vector_candidates: List[Dict[str, Any]],
        multi_hop_data: Dict[str, Any],
        max_hops: int,
    ) -> str:
        """
        Build readable, structured context string for prompting LLMs.
        """
        lines = []
        lines.append("=== I. PHÂN ĐOẠN KHỚP TRỰC TIẾP TỪ VECTOR SEARCH (DIRECT MATCHES) ===")
        if not vector_candidates:
            lines.append("Không tìm thấy phân đoạn khớp trực tiếp.")
        else:
            for idx, c in enumerate(vector_candidates, 1):
                lines.append(f"[{idx}] Tài liệu: {c['doc_title']} (ID: {c['doc_id']})")
                lines.append(f"    Phân đoạn: {c['title']} | Độ tương đồng (Score): {c['score']:.4f}")
                lines.append(f"    Nội dung: {c['content']}")
                lines.append("")

        if max_hops > 0:
            lines.append(f"=== II. QUAN HỆ ĐỒ THỊ VĂN BẢN ĐA BƯỚC ({max_hops} HOP{'S' if max_hops > 1 else ''}) ===")
            paths = multi_hop_data.get("paths", [])
            if not paths:
                lines.append("Không phát hiện thêm liên kết văn bản qua các bước nhảy quan hệ.")
            else:
                for idx, p in enumerate(paths, 1):
                    rel_desc = []
                    for rd in p["rel_details"]:
                        rel_desc.append(f"({rd['from_title']}) -[:{rd['type']}]-> ({rd['to_title']})")
                    lines.append(f"[{idx}] Bước nhảy: {p['hops']} hop(s)")
                    lines.append(f"    Chuỗi quan hệ: {' -> '.join(rel_desc)}")
                    lines.append(f"    Văn bản liên quan: {p['target_title']} (ID: {p['target_id']})")
                    lines.append("")

            related_chunks = multi_hop_data.get("related_chunks", [])
            if related_chunks:
                lines.append("=== III. NỘI DUNG PHÂN ĐOẠN TỪ CÁC TÀI LIỆU LIÊN QUAN ===")
                for idx, rc in enumerate(related_chunks, 1):
                    lines.append(f"[{idx}] Tài liệu: {rc['doc_title']} (ID: {rc['doc_id']})")
                    lines.append(f"    Phân đoạn: {rc.get('title', 'N/A')}")
                    lines.append(f"    Nội dung: {rc.get('content', '')}")
                    lines.append("")

        return "\n".join(lines)
