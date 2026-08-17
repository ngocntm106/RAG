import sys
import json

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from retriever import MultiHopGraphRetriever

def run_tests():
    retriever = MultiHopGraphRetriever()
    
    test_queries = [
        ("Nghị định quản lý cơ sở dữ liệu đồ thị Neo4j căn cứ vào luật nào?", [0, 1, 2]),
        ("Quy định về bảo vệ dữ liệu cá nhân và an ninh mạng", [1]),
    ]

    for query, hop_configs in test_queries:
        print("\n" + "=" * 80)
        print(f"🔍 TEST QUERY: '{query}'")
        print("=" * 80)

        for hops in hop_configs:
            print(f"\n--- [KẾT QUẢ RETRIEVAL VỚI MAX_HOPS = {hops}] ---")
            result = retriever.retrieve_context(query=query, top_k=2, max_hops=hops)
            
            print(f"Direct vector candidates: {len(result['vector_candidates'])}")
            for c in result['vector_candidates']:
                print(f"  • [Score: {c['score']:.4f}] {c['doc_title']} -> {c['title']}")

            if hops > 0:
                paths = result['multi_hop']['paths']
                print(f"Discovered multi-hop paths: {len(paths)}")
                for p in paths:
                    print(f"  • ({p['seed_title']}) --{p['rel_names']}--> ({p['target_title']}) [Hops: {p['hops']}]")

            print("\n[FORMATTED CONTEXT OUTPUT]:")
            print("-" * 40)
            print(result["formatted_context"])
            print("-" * 40)

if __name__ == "__main__":
    run_tests()
