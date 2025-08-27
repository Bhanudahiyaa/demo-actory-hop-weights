#!/usr/bin/env python3
"""
Discover all unique URLs from Link nodes to see what pages exist but haven't been crawled.
"""

import os
from dotenv import load_dotenv
from scraper.neo4j_client import Neo4jClient

def discover_all_urls():
    """Discover all unique URLs from Link nodes."""
    load_dotenv()
    
    print("🔍 Discovering all URLs from Link nodes...")
    client = Neo4jClient()
    
    try:
        with client._driver.session() as session:
            # Get all unique URLs from Link nodes
            result = session.run(
                """
                MATCH (l:Link)
                RETURN DISTINCT l.href AS url, count(*) AS link_count
                ORDER BY link_count DESC
                """
            )
            
            urls = [record.data() for record in result]
            
            print(f"\n📊 Found {len(urls)} unique URLs in Link nodes:")
            print("="*80)
            
            # Group by domain
            domains = {}
            for url_data in urls:
                url = url_data['url']
                count = url_data['link_count']
                
                if url.startswith('http'):
                    from urllib.parse import urlparse
                    domain = urlparse(url).netloc
                    if domain not in domains:
                        domains[domain] = []
                    domains[domain].append((url, count))
                else:
                    # Relative URLs
                    if 'relative' not in domains:
                        domains['relative'] = []
                    domains['relative'].append((url, count))
            
            # Print by domain
            for domain, url_list in domains.items():
                print(f"\n🌐 {domain} ({len(url_list)} URLs):")
                print("-" * 60)
                for url, count in url_list[:20]:  # Show first 20
                    print(f"  {url:<50} (linked {count} times)")
                if len(url_list) > 20:
                    print(f"  ... and {len(url_list) - 20} more URLs")
            
            # Check which URLs are already crawled
            print(f"\n🔍 Checking crawl status...")
            crawled_urls = set()
            with client._driver.session() as session:
                result = session.run("MATCH (p:Page) RETURN p.url AS url")
                crawled_urls = {record.data()['url'] for record in result}
            
            print(f"✅ Already crawled: {len(crawled_urls)} pages")
            
            # Find uncrawled URLs
            all_urls = {url_data['url'] for url_data in urls if url_data['url'].startswith('http')}
            uncrawled = all_urls - crawled_urls
            
            if uncrawled:
                print(f"⏳ Not yet crawled: {len(uncrawled)} pages")
                print("="*80)
                for url in sorted(list(uncrawled))[:30]:  # Show first 30
                    print(f"  {url}")
                if len(uncrawled) > 30:
                    print(f"  ... and {len(uncrawled) - 30} more URLs")
            else:
                print("🎉 All discovered URLs have been crawled!")
                
    except Exception as e:
        print(f"❌ Error discovering URLs: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    discover_all_urls()
