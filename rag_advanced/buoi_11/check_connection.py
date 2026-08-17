import sys
import json

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from neo4j_client import Neo4jClient

def main():
    print("=" * 60)
    print(" Neo4j Graph Database Connection - Bước 1")
    print("=" * 60)

    client = Neo4jClient()
    try:
        client.connect()
        print(f"✅ Kết nối thành công tới Neo4j Instance!")
        stats = client.get_database_statistics()
        
        print(f"\n[CẤU HÌNH KẾT NỐI]")
        print(f" - Connection URL: {stats['uri']}")
        print(f" - Database:       {stats['database']}")
        print(f" - Trạng thái:     Đã kết nối (Verified)")

        print(f"\n[THỐNG KÊ CƠ SỞ DỮ LIỆU]")
        print(f" - Tổng số Node:         {stats['total_nodes']}")
        for label, count in stats['node_counts'].items():
            print(f"   • Label {label}: {count}")

        print(f" - Tổng số Relationship: {stats['total_relationships']}")
        for rel_type, count in stats['relationship_counts'].items():
            print(f"   • Quan hệ [:{rel_type}]: {count}")

        print(f"\n[CHỈ MỤC (INDEXES)]")
        for idx in stats['indexes']:
            idx_name = idx.get('name')
            idx_type = idx.get('type')
            props = idx.get('properties') or []
            labels = idx.get('labels') or []
            print(f"   • {idx_name} ({idx_type}) -> Labels: {labels}, Props: {props}")

        print("\n" + "=" * 60)
        print(" Kết nối và xác thực thành công Bước 1!")
        print("=" * 60)
    except Exception as e:
        print(f"❌ Lỗi kết nối Neo4j: {e}")
        sys.exit(1)
    finally:
        client.close()

if __name__ == "__main__":
    main()
