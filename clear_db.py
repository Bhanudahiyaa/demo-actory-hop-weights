#!/usr/bin/env python3
"""
Standalone script to clear the Neo4j database.
Usage: python clear_db.py
"""

import os
from dotenv import load_dotenv
from scraper.neo4j_client import Neo4jClient

def main():
    load_dotenv()
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pwd = os.getenv("NEO4J_PASSWORD", "neo4j")
    
    print("🗑️  Connecting to Neo4j database...")
    client = Neo4jClient()
    
    try:
        print("🗑️  Clearing all data from database...")
        client.clear_database()
        print("✅ Database cleared successfully!")
    except Exception as e:
        print(f"❌ Error clearing database: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    main()
