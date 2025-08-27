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
                # link to page if href is a page we visited as well later (optional: we can also create LINK_TO regardless)
                # We'll create LINKS_TO edges when we see the linked page or preemptively if same-domain
                # Only create page links for internal URLs or if external crawling is enabled
                if href.startswith("http"):
                    # Check if this is an internal link (same domain) or if external crawling is enabled
                    current_domain = urlparse(page_url).netloc
                    link_domain = urlparse(href).netloc
                    
                    if include_external or current_domain == link_domain:
                        if neo4j:
                            neo4j.link_pages(page_url, href)

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
            print()


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
    args = parser.parse_args()

    # Handle clear-only option
    if args.clear_only:
        clear_database_only()
        return

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
