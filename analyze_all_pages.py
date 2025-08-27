#!/usr/bin/env python3
"""
Analyze all pages in the database to get a complete website overview.
This shows the full website structure, not just hop-1 neighbors.
"""

import os
from dotenv import load_dotenv
from scraper.neo4j_client import Neo4jClient

def analyze_all_pages():
    """Analyze all pages in the database for complete website overview."""
    load_dotenv()
    
    print("🔍 Analyzing complete website structure...")
    client = Neo4jClient()
    
    try:
        with client._driver.session() as session:
            # Get all pages with their metadata
            result = session.run(
                """
                MATCH (p:Page)
                OPTIONAL MATCH (p)-[:HAS_FORM]->(form:Form)
                OPTIONAL MATCH (p)-[:HAS_BUTTON]->(button:Button)
                OPTIONAL MATCH (p)-[:HAS_INPUT]->(input:InputField)
                OPTIONAL MATCH (p)-[:HAS_IMAGE]->(image:Image)
                OPTIONAL MATCH (p)-[:USES_SCRIPT]->(script:Script)
                OPTIONAL MATCH (p)-[:USES_STYLESHEET]->(stylesheet:Stylesheet)
                OPTIONAL MATCH (p)-[:HAS_LINK]->(link:Link)
                OPTIONAL MATCH (p)-[:USES_MEDIA]->(media:MediaResource)
                RETURN p.url AS url,
                       count(DISTINCT form) AS formCount,
                       count(DISTINCT button) AS buttonCount,
                       count(DISTINCT input) AS inputCount,
                       count(DISTINCT image) AS imageCount,
                       count(DISTINCT script) AS scriptCount,
                       count(DISTINCT stylesheet) AS stylesheetCount,
                       count(DISTINCT link) AS linkCount,
                       count(DISTINCT media) AS mediaCount
                ORDER BY scriptCount DESC, formCount DESC, buttonCount DESC
                """
            )
            
            pages = [record.data() for record in result]
            
            if not pages:
                print("❌ No pages found in database.")
                return
            
            # Print complete website overview
            print("\n" + "="*100)
            print("🌐 COMPLETE WEBSITE STRUCTURE")
            print("="*100)
            
            # Table header
            print(f"{'URL':<60} {'Scripts':<8} {'Forms':<6} {'Buttons':<8} {'Images':<7} {'Links':<6}")
            print("-" * 100)
            
            for page in pages:
                url = page['url'][:57] + "..." if len(page['url']) > 60 else page['url']
                script_count = page['scriptCount']
                form_count = page['formCount']
                button_count = page['buttonCount']
                image_count = page['imageCount']
                link_count = page['linkCount']
                
                print(f"{url:<60} {script_count:<8} {form_count:<6} {button_count:<8} {image_count:<7} {link_count:<6}")
            
            # Print summary
            total_scripts = sum(p['scriptCount'] for p in pages)
            total_forms = sum(p['formCount'] for p in pages)
            total_buttons = sum(p['buttonCount'] for p in pages)
            total_images = sum(p['imageCount'] for p in pages)
            total_links = sum(p['linkCount'] for p in pages)
            
            print("\n" + "-" * 100)
            print(f"📊 WEBSITE SUMMARY: {len(pages)} pages | "
                  f"{total_scripts} scripts | "
                  f"{total_forms} forms | "
                  f"{total_buttons} buttons | "
                  f"{total_images} images | "
                  f"{total_links} links")
            
            # Print crawl priority plan
            print("\n" + "="*100)
            print("🗺️  CRAWL PRIORITY PLAN (All Pages Ranked)")
            print("="*100)
            
            # Calculate dynamic complexity score for each page
            for i, page in enumerate(pages, 1):
                # Dynamic complexity score: heavily weighted toward JS and interactivity
                dynamic_score = (
                    page['scriptCount'] * 0.5 +                    # JS: 50% weight
                    page['formCount'] * 0.25 +                     # Forms: 25% weight
                    (page['buttonCount'] + page['inputCount']) * 0.15 +  # Interactive: 15%
                    page['imageCount'] * 0.1                       # Images: 10% weight
                )
                
                print(f"{i:2d}. {page['url']}")
                print(f"    Dynamic Score: {dynamic_score:.2f} | "
                      f"JS Scripts: {page['scriptCount']} | "
                      f"Forms: {page['formCount']} | "
                      f"Interactive: {page['buttonCount'] + page['inputCount']} | "
                      f"Images: {page['imageCount']}")
                print()
                
    except Exception as e:
        print(f"❌ Error analyzing all pages: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    analyze_all_pages()
