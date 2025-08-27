#!/usr/bin/env python3
"""
Standalone module for analyzing hop-1 neighborhoods.
Keeps the scraper focused on data collection only.
"""

import os
from dotenv import load_dotenv
from scraper.neo4j_client import Neo4jClient


def analyze_neighborhood(page_url: str):
    """Analyze hop-1 neighborhood for a specific page."""
    load_dotenv()
    
    print(f"🔍 Analyzing hop-1 neighborhood for: {page_url}")
    client = Neo4jClient()
    
    try:
        neighborhood = client.get_hop1_neighborhood(page_url)
        
        if not neighborhood['neighbors']:
            print("❌ No neighbors found for this page.")
            return
        
        # Print Hop-1 Neighbors section
        print("\n" + "="*80)
        print("🏘️  HOP-1 NEIGHBORS")
        print("="*80)
        
        # Table header
        print(f"{'URL':<60} {'Status':<8} {'Forms':<6} {'Scripts':<8} {'Buttons':<8} {'Images':<7} {'Links':<6}")
        print("-" * 95)
        
        for neighbor in neighborhood['neighbors']:
            url = neighbor['url'][:57] + "..." if len(neighbor['url']) > 60 else neighbor['url']
            status = "✅ Crawled" if neighbor['page_exists'] else "⏳ Not Crawled"
            form_count = neighbor['formCount']
            script_count = neighbor['scriptCount']
            button_count = neighbor['buttonCount']
            image_count = neighbor['imageCount']
            link_count = neighbor['linkCount']
            
            print(f"{url:<60} {status:<8} {form_count:<6} {script_count:<8} {button_count:<8} {image_count:<7} {link_count:<6}")
        
        # Print summary
        summary = neighborhood['summary']
        print("\n" + "-" * 95)
        print(f"📊 SUMMARY: {summary['total_neighbors']} neighbors | "
              f"{summary['total_forms']} forms | "
              f"{summary['total_scripts']} scripts | "
              f"{summary['total_interactive']} interactive elements")
        
        # Print Crawl Plan section with dynamic/JS-focused weighting
        print("\n" + "="*80)
        print("🗺️  CRAWL PLAN (Prioritizing Dynamic/JS-Heavy Pages)")
        print("="*80)
        
        # Sort neighbors by dynamic complexity (JS-heavy, interactive, forms)
        priority_neighbors = sorted(
            neighborhood['neighbors'],
            key=lambda x: (
                x['scriptCount'],           # JavaScript complexity (highest priority)
                x['formCount'],             # Forms and user input
                x['buttonCount'] + x['inputCount'],  # Interactive elements
                x['imageCount']             # Media content
            ),
            reverse=True
        )
        
        for i, neighbor in enumerate(priority_neighbors, 1):
            # Dynamic complexity score: heavily weighted toward JS and interactivity
            dynamic_score = (
                neighbor['scriptCount'] * 0.5 +                    # JS: 50% weight
                neighbor['formCount'] * 0.25 +                     # Forms: 25% weight
                (neighbor['buttonCount'] + neighbor['inputCount']) * 0.15 +  # Interactive: 15%
                neighbor['imageCount'] * 0.1                       # Images: 10% weight
            )
            
            print(f"{i:2d}. {neighbor['url']}")
            print(f"    Dynamic Score: {dynamic_score:.2f} | "
                  f"JS Scripts: {neighbor['scriptCount']} | "
                  f"Forms: {neighbor['formCount']} | "
                  f"Interactive: {neighbor['buttonCount'] + neighbor['inputCount']}")
            print()
        
    except Exception as e:
        print(f"❌ Error analyzing neighborhood: {e}")
    finally:
        client.close()


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze hop-1 neighborhood for web pages")
    parser.add_argument("url", help="URL of the page to analyze")
    
    args = parser.parse_args()
    analyze_neighborhood(args.url)


if __name__ == "__main__":
    main()
