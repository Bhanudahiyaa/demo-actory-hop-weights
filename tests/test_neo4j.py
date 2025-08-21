import sys
import os

# Add parent directory to sys.path so imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from neo4j_client import Neo4jClient


if __name__ == "__main__":
    client = Neo4jClient()

    # Test inserting a page
    client.upsert_page("http://example.com")

    # Test summary
    summary = client.get_summary()
    print(summary)

    client.close()