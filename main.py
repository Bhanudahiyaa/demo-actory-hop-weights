#!/usr/bin/env python3
"""
Actory AI Web Scraper - Intelligent Navigation Workflow
Clean, organized main entry point for DOM feature extraction and AI-powered navigation planning.

Features:
- Simple DOM Feature Extraction (no LLM complexity)
- Intelligent Navigation Planning with DSPy PlanAgent (GPT-OSS 20B)
- Neo4j Graph Database Storage
- Feature-based Weighting System
- Subpath Memoization & Route Compression
- Robust Fallback to Algorithmic Approach
"""

import asyncio
import sys
import os
import time
from typing import Optional

# Add project to path
sys.path.append('.')

async def run_intelligent_navigation_workflow(url: str, max_pages: Optional[int] = None, enable_llm: bool = False):
    """
    Run the complete intelligent navigation workflow:
    1. Simple Scraping (DOM features only)
    2. Intelligent Navigation Planning with DSPy PlanAgent (GPT-OSS 20B)
    """
    
    print("🚀 ACTORY AI WEB SCRAPER - INTELLIGENT NAVIGATION WORKFLOW")
    print("=" * 60)
    print(f"🎯 Target: {url}")
    print(f"📊 Max Pages: {max_pages if max_pages else 'Unlimited'}")
    print(f"🧠 AI Intelligence: {'Enabled' if enable_llm else 'Disabled'}")
    print(f"🗄️  Database: Neo4j")
    print(f"🤖 Route Discovery: DSPy PlanAgent (GPT-OSS 20B)")
    print("=" * 60)
    
    try:
        # Phase 1: Simple Scraping (DOM features only)
        print(f"\n📊 PHASE 1: Simple Scraping")
        print("-" * 40)
        
        from scraper.simple_run import simple_crawl_and_store
        from scraper.neo4j_client import Neo4jClient
        
        # Initialize Neo4j
        neo4j_client = Neo4jClient()
        
        # Run simple scraping
        pages = await simple_crawl_and_store(
            url=url,
            depth=5,  # Increased depth for full coverage
            max_pages=max_pages,
            include_external=False,
            neo4j=neo4j_client
        )
        
        print(f"✅ Phase 1 Complete: {len(pages)} pages scraped with DOM features")
        
        # Phase 2: Intelligent Navigation Planning with DSPy PlanAgent
        print(f"\n🤖 PHASE 2: Intelligent Navigation Planning with DSPy PlanAgent")
        print("-" * 70)
        
        from intelligent_navigation_planner import IntelligentNavigationPlanner
        
        # Your OpenRouter API key for GPT-OSS 20B
        API_KEY = "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"
        
        # Initialize intelligent navigation planner
        planner = IntelligentNavigationPlanner(
            api_key=API_KEY,
            max_depth=4,
            enable_memoization=True
        )
        
        print("✅ Intelligent Navigation Planner initialized successfully")
        
        # Use the new intelligent navigation planning method
        print("🧠 Creating intelligent navigation plan from actual scraped data...")
        start_planning = time.time()
        
        navigation_plan = planner.create_intelligent_navigation_plan(url, neo4j_client)
        
        planning_time = time.time() - start_planning
        
        if navigation_plan:
            print(f"✅ Intelligent navigation plan created in {planning_time:.2f}s")
            
            # Display the comprehensive navigation plan
            _display_navigation_plan(navigation_plan)
            
            # Create planning result structure
            planning_result = {
                'navigation_plan': navigation_plan,
                'total_pages': navigation_plan['metadata']['total_pages'],
                'total_neighborhoods': navigation_plan['metadata']['total_neighborhoods'],
                'total_ranked_pages': navigation_plan['metadata']['total_ranked_pages'],
                'planning_time': planning_time,
                'high_priority_pages': len(navigation_plan['summary']['high_priority_pages']),
                'critical_paths': len(navigation_plan['summary']['critical_paths']),
                'recommendations': len(navigation_plan['summary']['recommendations'])
            }
        else:
            print("❌ Failed to create intelligent navigation plan")
            planning_result = {
                'error': 'Navigation planning failed',
                'planning_time': planning_time
            }
        
        print(f"✅ Phase 2 Complete: Intelligent navigation planning completed")
        
        # Show results summary
        print(f"\n🎉 INTELLIGENT NAVIGATION WORKFLOW COMPLETE!")
        print("=" * 60)
        print("📊 Results Summary:")
        print(f"   🗺️  Pages Scraped: {len(pages)}")
        print(f"   🤖 AI Intelligence: {'Active' if enable_llm else 'Active (DSPy)'}")
        
        if 'error' not in planning_result:
            print(f"   📊 Total Ranked Pages: {planning_result['total_ranked_pages']}")
            print(f"   ⚡ Dynamic Content Analysis: Complete")
            print(f"   🛣️  Navigation Paths Generated: {planning_result['total_pages']}")
            print(f"   🔗 Neighbor Analysis: Complete")
            print(f"   ⏱️  Planning Time: {planning_result['planning_time']:.2f}s")
        else:
            print(f"   ❌ Planning Error: {planning_result['error']}")
        
        print("=" * 60)
        
        return planning_result
        
    except Exception as e:
        print(f"❌ Workflow failed: {e}")
        return None


def main():
    """Main function with command line interface."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Actory AI Web Scraper - Intelligent Navigation Workflow")
    parser.add_argument("url", help="Starting URL to scrape and analyze")
    parser.add_argument("-p", "--pages", type=int, help="Maximum pages to scrape (default: unlimited)")
    parser.add_argument("--enable-llm", action="store_true", help="Enable LLM intelligence")
    parser.add_argument("--clear-db", action="store_true", help="Clear Neo4j database before starting")
    
    args = parser.parse_args()
    
    # Clear database if requested
    if args.clear_db:
        print("🗑️  Clearing Neo4j database...")
        try:
            from scraper.neo4j_client import Neo4jClient
            neo4j_client = Neo4jClient()
            neo4j_client.clear_database()
            print("✅ Database cleared successfully")
        except Exception as e:
            print(f"⚠️  Database clear failed: {e}")
    
    # Run intelligent navigation workflow
    result = asyncio.run(run_intelligent_navigation_workflow(
        url=args.url,
        max_pages=args.pages,
        enable_llm=args.enable_llm
    ))
    
    if result:
        print("\n🎯 Intelligent navigation workflow completed successfully!")
        print("🚀 Your web scraping data is ready with AI-powered navigation planning!")
    else:
        print("\n❌ Workflow failed. Check the logs above for details.")


def _display_navigation_plan(navigation_plan: dict):
    """Display the AI Navigation Planner output following strict rules."""
    print("\n" + "="*100)
    print("🤖 AI NAVIGATION PLANNER - DSPy Dynamic Content Analysis & Comprehensive Path Enumeration")
    print("="*100)
    
    # Section 1: Priority Ranking (dynamic content only)
    print("\n" + "="*80)
    print("📊 SECTION 1: PRIORITY RANKING (Dynamic Content Only)")
    print("="*80)
    print("🎯 Pages ranked from highest to lowest by dynamic content score")
    print("⚡ Scoring: Forms(×10) + Inputs(×8) + Scripts(×6) + Buttons(×3)")
    print("-" * 80)
    
    for i, page_data in enumerate(navigation_plan['section_1_priority_ranking'], 1):
        print(f"\n🏆 RANK #{i}: {page_data['url']}")
        print(f"   ⚡ Dynamic Score: {page_data['dynamic_score']}")
        print(f"   📊 Features: Scripts({page_data['features']['scripts']}) | Forms({page_data['features']['forms']}) | Inputs({page_data['features']['inputs']}) | Buttons({page_data['features']['buttons']}) | Images({page_data['features']['images']}) | Links({page_data['features']['links']})")
        print(f"   🧮 Score Breakdown: {page_data['score_breakdown']}")
    
    # Section 2: Navigation Paths (for each page in ranking order)
    print("\n" + "="*80)
    print("🛣️  SECTION 2: NAVIGATION PATHS (All Possible Routes)")
    print("="*80)
    print("🎯 For each page in ranking order, ALL possible paths from landing → destination")
    print("📏 Paths ordered from SHORTEST to LONGEST")
    print("-" * 80)
    
    for page_url, page_info in navigation_plan['section_2_navigation_paths'].items():
        print(f"\n🎯 DESTINATION #{page_info['rank']}: {page_url}")
        print(f"   ⚡ Dynamic Score: {page_info['dynamic_score']}")
        print(f"   📊 Features: Scripts({page_info['features']['scripts']}) | Forms({page_info['features']['forms']}) | Inputs({page_info['features']['inputs']}) | Buttons({page_info['features']['buttons']})")
        print(f"   🧮 Score Breakdown: {page_info['score_breakdown']}")
        print(f"   🛣️  Total Paths Found: {len(page_info['paths'])}")
        
        for i, path_info in enumerate(page_info['paths'], 1):
            print(f"      {i:2d}. [{path_info['hops']:2d} hops] {path_info['path_string']}")
    
    # Section 3: Neighbors (for each visited page, repeat path expansion)
    print("\n" + "="*80)
    print("🔗 SECTION 3: NEIGHBORS (Directly Connected Pages)")
    print("="*80)
    print("🎯 For each visited page, ALL possible paths to its directly connected neighbors")
    print("📏 Paths ordered from SHORTEST to LONGEST")
    print("-" * 80)
    
    for page_url, page_info in navigation_plan['section_3_neighbors'].items():
        print(f"\n🔗 SOURCE PAGE: {page_url}")
        print(f"   📊 Direct Neighbors: {page_info['immediate_neighbor_count']}")
        
        for neighbor_url, neighbor_info in page_info['immediate_neighbors'].items():
            print(f"\n      🎯 NEIGHBOR: {neighbor_url}")
            print(f"         🛣️  Total Paths Found: {len(neighbor_info['all_possible_paths'])}")
            
            for i, path_info in enumerate(neighbor_info['all_possible_paths'], 1):
                print(f"            {i:2d}. [{path_info['hops']:2d} hops] {path_info['path_string']}")
    
    print("\n" + "="*100)
    print("✅ AI NAVIGATION PLANNER - COMPREHENSIVE ANALYSIS COMPLETE")
    print("="*100)


if __name__ == "__main__":
    main()
