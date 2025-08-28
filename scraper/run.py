import os
import asyncio
import argparse
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from crawl4ai.deep_crawling.bfs_strategy import BFSDeepCrawlStrategy
from crawl4ai.async_configs import CrawlerRunConfig

from .neo4j_client import Neo4jClient
from .extractors import (
    extract_buttons, 
    extract_images_from_result_media,
    extract_all_dom_features,
    extract_input_fields,
    extract_textareas,
    extract_links,
    extract_images,
    extract_stylesheets,
    extract_scripts,
    extract_fonts,
    extract_forms,
    extract_media_resources
)

# Import DSPy modules for autonomous crawling
try:
    from .crawl_agent import CrawlAgent
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available. Autonomous crawling disabled.")


def env_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


async def crawl_and_store(
    url: str,
    depth: int,
    max_pages: int,
    include_external: bool,
    neo4j: Optional[Neo4jClient],
):
    start_url = url  # Store the starting URL for LINKS_TO creation
    strategy = BFSDeepCrawlStrategy(
        max_depth=depth,
        include_external=include_external,
        max_pages=max_pages,
    )
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        exclude_external_links=not include_external,
        verbose=False,
        stream=True,
    )

    async with AsyncWebCrawler() as crawler:
        # With DeepCrawlDecorator, calling crawler.arun(url, config) will delegate to deep strategy
        results = await crawler.arun(url, config=config)
        pages: List[str] = []

        async for result in results:
            page_url = result.url
            pages.append(page_url)
            if neo4j:
                neo4j.upsert_page(page_url)

            # Extract all DOM features comprehensively
            dom_features = extract_all_dom_features(result.html)
            
            # LINKS - Enhanced link extraction
            internal = (result.links or {}).get("internal", [])
            external = (result.links or {}).get("external", [])
            
            # Only process external links if include_external is True
            links_to_process = internal + (external if include_external else [])
            for link in links_to_process:
                href = link.get("href") or ""
                text = link.get("text") or ""
                title = link.get("title") or ""
                if not href:
                    continue
                if neo4j:
                    neo4j.add_link(page_url, href, text, title)
                
                # Create LINKS_TO relationships for all discovered links to enable DSPy agent neighbor discovery
                # This ensures the graph has proper navigation paths between pages
                if href.startswith("http"):
                    # Check if this is an internal link (same domain) or if external crawling is enabled
                    current_domain = urlparse(page_url).netloc
                    link_domain = urlparse(href).netloc
                    
                    if include_external or current_domain == link_domain:
                        if neo4j:
                            neo4j.link_pages(page_url, href)
                            print(f"   🔗 Created LINKS_TO: {page_url} -> {href}")
                elif href.startswith("/"):
                    # Handle relative URLs by converting to absolute URLs
                    try:
                        parsed_url = urlparse(page_url)
                        absolute_href = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                        if neo4j:
                            neo4j.link_pages(page_url, absolute_href)
                            print(f"   🔗 Created LINKS_TO: {page_url} -> {absolute_href}")
                    except Exception as e:
                        print(f"   ⚠️  Warning: Could not process relative link {href}: {e}")

            # Store all extracted DOM features in Neo4j
            if neo4j:
                # BUTTONS
                for button in dom_features["buttons"]:
                    neo4j.add_button(page_url, button)

                # INPUT FIELDS
                for input_field in dom_features["inputs"]:
                    neo4j.add_input_field(page_url, input_field)

                # TEXTAREAS
                for textarea in dom_features["textareas"]:
                    neo4j.add_textarea(page_url, textarea)

                # IMAGES (from DOM + media results)
                for image in dom_features["images"]:
                    src = image.get("src") or ""
                    if src:
                        neo4j.add_image(page_url, src, image.get("alt", ""))
                
                # Also add images from result media (legacy support)
                for image in extract_images_from_result_media(result.media or {}):
                    src = image.get("src") or ""
                    if src:
                        neo4j.add_image(page_url, src, image.get("alt", ""))

                # STYLESHEETS
                for stylesheet in dom_features["stylesheets"]:
                    neo4j.add_stylesheet(page_url, stylesheet)

                # SCRIPTS
                for script in dom_features["scripts"]:
                    neo4j.add_script(page_url, script)

                # FONTS
                for font in dom_features["fonts"]:
                    neo4j.add_font(page_url, font)

                # FORMS
                for form in dom_features["forms"]:
                    neo4j.add_form(page_url, form)

                # MEDIA RESOURCES
                for media in dom_features["media"]:
                    neo4j.add_media_resource(page_url, media)

            # Print summary for this page
            print(f"📄 Processed: {page_url}")
            print(f"   🔗 Links: {len(dom_features['links'])}")
            print(f"   🔘 Buttons: {len(dom_features['buttons'])}")
            print(f"   📝 Inputs: {len(dom_features['inputs'])}")
            print(f"   📋 Textareas: {len(dom_features['textareas'])}")
            print(f"   🖼️  Images: {len(dom_features['images'])}")
            print(f"   🎨 Stylesheets: {len(dom_features['stylesheets'])}")
            print(f"   📜 Scripts: {len(dom_features['scripts'])}")
            print(f"   🔤 Fonts: {len(dom_features['fonts'])}")
            print(f"   📋 Forms: {len(dom_features['forms'])}")
            print(f"   🎵 Media: {len(dom_features['media'])}")
            
            # Create LINKS_TO relationship from start_url to this page if this is not the start page
            # This ensures the DSPy agent can discover all crawled pages as neighbors
            if neo4j and page_url != start_url:
                try:
                    neo4j.link_pages(start_url, page_url)
                    print(f"   🔗 Created LINKS_TO from start: {start_url} -> {page_url}")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not create LINKS_TO from start: {e}")
            
            print()


def run_dspy_crawl_agent(start_url: str, max_depth: int, max_pages: int):
    """Run the DSPy autonomous crawl agent."""
    if not DSPY_AVAILABLE:
        print("❌ DSPy not available. Cannot run autonomous crawling.")
        return
    
    print("🤖 Initializing DSPy Crawl Agent...")
    
    try:
        # Initialize Neo4j client
        neo4j_client = Neo4jClient()
        
        # Create and run crawl agent
        crawl_agent = CrawlAgent(neo4j_client)
        result = crawl_agent.forward(start_url, max_depth, max_pages)
        
        # Print crawl analysis
        print("\n" + "="*80)
        print("📊 CRAWL ANALYSIS")
        print("="*80)
        
        # Summary
        summary = result['analysis_report']['summary']
        print(f"📈 SUMMARY:")
        print(f"   Total Pages Crawled: {summary['total_pages_crawled']}")
        print(f"   Successful Crawls: {summary['successful_crawls']}")
        print(f"   Failed Crawls: {summary['failed_crawls']}")
        print(f"   Success Rate: {summary['success_rate']:.1f}%")
        print(f"   Max Depth Reached: {summary['max_depth_reached']}")
        print(f"   Crawl Duration: {summary['crawl_duration']:.2f}s")
        print(f"   Total URLs Discovered: {summary.get('total_urls_discovered', 'N/A')}")
        print(f"   Autonomous Expansions: {summary.get('autonomous_expansions', 'N/A')}")
        
        # Statistics
        stats = result['analysis_report']['statistics']
        print(f"\n📊 STATISTICS:")
        print(f"   Avg Response Time: {stats['avg_response_time']}s")
        print(f"   Max Response Time: {stats['max_response_time']}s")
        print(f"   Min Response Time: {stats.get('min_response_time', 'N/A')}s")
        print(f"   Avg Dynamic Score: {stats['avg_dynamic_score']}")
        print(f"   Max Dynamic Score: {stats['max_dynamic_score']}")
        print(f"   Min Dynamic Score: {stats.get('min_dynamic_score', 'N/A')}")
        print(f"   Total Scripts: {stats['total_scripts']}")
        print(f"   Total Forms: {stats['total_forms']}")
        print(f"   Total Buttons: {stats['total_buttons']}")
        print(f"   Total Images: {stats['total_images']}")
        print(f"   Total Links: {stats['total_links']}")
        print(f"   Total Stylesheets: {stats.get('total_stylesheets', 'N/A')}")
        print(f"   Total Media: {stats.get('total_media', 'N/A')}")
        
        # Autonomous Insights
        if 'autonomous_insights' in result['analysis_report']:
            insights = result['analysis_report']['autonomous_insights']
            print(f"\n🎯 AUTONOMOUS INSIGHTS:")
            print(f"   Discovery Efficiency: {insights.get('discovery_efficiency', {}).get('discovery_ratio', 'N/A'):.2f}")
            print(f"   Avg Depth: {insights.get('depth_exploration', {}).get('avg_depth', 'N/A'):.1f}")
            print(f"   Avg Scripts/Page: {insights.get('content_complexity', {}).get('avg_scripts_per_page', 'N/A'):.1f}")
            print(f"   Avg Forms/Page: {insights.get('content_complexity', {}).get('avg_forms_per_page', 'N/A'):.1f}")
            print(f"   Most Complex Page: {insights.get('content_complexity', {}).get('most_complex_page', 'N/A')}")
        
        # Top pages
        top_pages = result['analysis_report']['top_pages']
        if top_pages:
            print(f"\n🏆 TOP PAGES (by Dynamic Score):")
            for i, page in enumerate(top_pages[:5], 1):
                print(f"   {i}. {page['url']}")
                print(f"      Score: {page['dynamic_score']:.2f} | "
                      f"Scripts: {page['features']['scripts']} | "
                      f"Forms: {page['features']['forms']} | "
                      f"Images: {page['features']['images']}")
        
        # Sample list
        sample_list = result['analysis_report']['sample_list']
        if sample_list:
            print(f"\n📋 SAMPLE PAGES CRAWLED:")
            for i, page in enumerate(sample_list[:10], 1):
                print(f"   {i}. {page['url']}")
                print(f"      Status: {page['status_code']} | "
                      f"Score: {page['dynamic_score']:.2f} | "
                      f"Time: {page['response_time']:.2f}s | "
                      f"Size: {page['content_length']} chars")
        
        print("\n" + "="*80)
        print("✅ DSPy Crawl Agent completed successfully!")
        
    except Exception as e:
        print(f"❌ Error running DSPy Crawl Agent: {e}")
    finally:
        if 'neo4j_client' in locals():
            neo4j_client.close()


def clear_database_only():
    """Clear the Neo4j database without running any scraping."""
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


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("TARGET_URL", "https://example.com"))
    parser.add_argument("--depth", type=int, default=int(os.getenv("MAX_DEPTH", "3")))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("MAX_PAGES", "200")))
    parser.add_argument(
        "--include-external",
        action="store_true",
        default=env_bool(os.getenv("INCLUDE_EXTERNAL"), False),
        help="Include external-domain links during link discovery (default from INCLUDE_EXTERNAL env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run crawl and extraction without writing to Neo4j",
    )
    parser.add_argument(
        "--clear-db",
        action="store_true",
        help="Clear all data from the Neo4j database before starting",
    )
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Only clear the database without running any scraping",
    )
    
    # New DSPy Crawl Agent arguments
    parser.add_argument(
        "--crawl",
        action="store_true",
        help="Use DSPy autonomous crawl agent instead of traditional crawling",
    )
    parser.add_argument(
        "--start-url",
        type=str,
        help="Starting URL for DSPy crawl agent (overrides --url when using --crawl)",
    )
    
    args = parser.parse_args()

    # Handle clear-only option
    if args.clear_only:
        clear_database_only()
        return

    # Handle DSPy crawl agent
    if args.crawl:
        if not DSPY_AVAILABLE:
            print("❌ DSPy not available. Install with: pip install dspy-ai")
            return
        
        start_url = args.start_url or args.url
        print(f"🤖 Running DSPy Crawl Agent from: {start_url}")
        run_dspy_crawl_agent(start_url, args.depth, args.max_pages)
        return

    # Traditional crawling (existing functionality)
    client: Optional[Neo4jClient] = None if args.dry_run else Neo4jClient()
    
    # Clear database if requested
    if client and args.clear_db:
        print("🗑️  Clearing database...")
        client.clear_database()
    
    try:
        asyncio.run(crawl_and_store(args.url, args.depth, args.max_pages, args.include_external, client))
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()
