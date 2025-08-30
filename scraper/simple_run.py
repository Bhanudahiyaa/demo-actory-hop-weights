#!/usr/bin/env python3
"""
Simple Web Scraping - DOM Feature Extraction Only
Clean, fast scraping without LLM complexity.
"""

import asyncio
from typing import List, Optional
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler
from crawl4ai.deep_crawling.bfs_strategy import BFSDeepCrawlStrategy
from crawl4ai.async_configs import CrawlerRunConfig

from .neo4j_client import Neo4jClient
from .extractors import extract_all_dom_features


async def simple_crawl_and_store(
    url: str,
    depth: int,
    max_pages: Optional[int] = None,
    include_external: bool = False,
    neo4j: Optional[Neo4jClient] = None,
):
    """
    Simple crawling with DOM feature extraction only.
    No LLM analysis, just clean feature counting and weights.
    """
    
    print(f"🚀 Starting Simple Crawl")
    print(f"🎯 Target: {url}")
    print(f"📊 Depth: {depth}")
    print(f"📄 Max Pages: {max_pages if max_pages else 'Unlimited'}")
    print(f"🔗 External Links: {'Yes' if include_external else 'No'}")
    print("-" * 60)
    
    # Configure crawler with BFS strategy to visit all pages (restore original working config)
    strategy = BFSDeepCrawlStrategy(
        max_depth=depth,  # Use original depth parameter
        include_external=include_external,
        max_pages=max_pages if max_pages else 1000  # Set a high limit if None
    )
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        exclude_external_links=not include_external,
        verbose=True,  # Enable verbose to see what's happening
        stream=True,
    )

    # Track statistics
    stats = {
        'total_pages': 0,
        'total_links': 0,
        'total_buttons': 0,
        'total_forms': 0,
        'total_images': 0,
        'total_scripts': 0
    }

    # Use the original working BFS crawling approach
    print("🚀 Starting BFS crawl with original strategy...")
    
    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, config=config)
        pages: List[str] = []
        
        async for result in results:
            page_url = result.url
            pages.append(page_url)
            stats['total_pages'] += 1
            
            if neo4j:
                neo4j.upsert_page(page_url)

            # Extract DOM features
            dom_features = extract_all_dom_features(result.html)
            
            # Update statistics
            stats['total_links'] += len(dom_features['links'])
            stats['total_buttons'] += len(dom_features['buttons'])
            stats['total_forms'] += len(dom_features['forms'])
            stats['total_images'] += len(dom_features['images'])
            stats['total_scripts'] += len(dom_features['scripts'])
            
            # Process links
            await _process_links(result, page_url, neo4j, include_external, url)
            
            # Store DOM features
            if neo4j:
                await _store_dom_features(neo4j, page_url, dom_features)

            # Print simple page summary
            _print_simple_summary(page_url, dom_features)
            
            # Create LINKS_TO relationship from start
            if neo4j and page_url != url:
                try:
                    neo4j.link_pages(url, page_url)
                    print(f"   🔗 Created LINKS_TO from start: {url} -> {page_url}")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not create LINKS_TO from start: {e}")
            
            print()

        # Print final summary
        _print_final_summary(stats)

        return pages


async def _process_links(result, page_url: str, neo4j: Optional[Neo4jClient], include_external: bool, start_url: str):
    """Process and store link information."""
    discovered_urls = set()
    
    if not neo4j:
        return discovered_urls
        
    internal = (result.links or {}).get("internal", [])
    external = (result.links or {}).get("external", [])
    
    links_to_process = internal + (external if include_external else [])
    for link in links_to_process:
        href = link.get("href") or ""
        text = link.get("text") or ""
        title = link.get("title") or ""
        
        if not href:
            continue
            
        neo4j.add_link(page_url, href, text, title)
        
        # Create LINKS_TO relationships and collect discovered URLs
        if href.startswith("http"):
            current_domain = urlparse(page_url).netloc
            link_domain = urlparse(href).netloc
            
            if include_external or current_domain == link_domain:
                neo4j.link_pages(page_url, href)
                discovered_urls.add(href)
                print(f"   🔗 Created LINKS_TO: {page_url} -> {href}")
        elif href.startswith("/"):
            try:
                parsed_url = urlparse(page_url)
                absolute_href = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                neo4j.link_pages(page_url, absolute_href)
                discovered_urls.add(absolute_href)
                print(f"   🔗 Created LINKS_TO: {page_url} -> {absolute_href}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not process relative link {href}: {e}")
    
    return discovered_urls


async def _store_dom_features(neo4j: Neo4jClient, page_url: str, dom_features: dict):
    """Store all extracted DOM features in Neo4j and update Page node with counts."""
    
    # Calculate feature counts for the Page node
    feature_counts = {
        'scriptCount': len(dom_features["scripts"]),
        'formCount': len(dom_features["forms"]),
        'buttonCount': len(dom_features["buttons"]),
        'inputCount': len(dom_features["inputs"]),
        'imageCount': len(dom_features["images"]),
        'linkCount': len(dom_features["links"]),
        'mediaCount': len(dom_features["media"]),
        'stylesheetCount': len(dom_features["stylesheets"]),
        'fontCount': len(dom_features["fonts"]),
        'textareaCount': len(dom_features["textareas"])
    }
    
    # Update the Page node with feature counts
    neo4j.update_page_features(page_url, feature_counts)
    
    # Store each feature type
    for button in dom_features["buttons"]:
        neo4j.add_button(page_url, button)
    
    for input_field in dom_features["inputs"]:
        neo4j.add_input_field(page_url, input_field)
    
    for textarea in dom_features["textareas"]:
        neo4j.add_textarea(page_url, textarea)
    
    for image in dom_features["images"]:
        src = image.get("src") or ""
        if src:
            neo4j.add_image(page_url, src, image.get("alt", ""))
    
    for stylesheet in dom_features["stylesheets"]:
        neo4j.add_stylesheet(page_url, stylesheet)
    
    for script in dom_features["scripts"]:
        neo4j.add_script(page_url, script)
    
    for font in dom_features["fonts"]:
        neo4j.add_font(page_url, font)
    
    for form in dom_features["forms"]:
        neo4j.add_form(page_url, form)
    
    for media in dom_features["media"]:
        neo4j.add_media_resource(page_url, media)


def _print_simple_summary(page_url: str, dom_features: dict):
    """Print simple page summary without LLM analysis."""
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


def _print_final_summary(stats: dict):
    """Print simple crawl summary."""
    print("\n" + "="*60)
    print("📊 SIMPLE CRAWL SUMMARY")
    print("="*60)
    print(f"📄 Total Pages Crawled: {stats['total_pages']}")
    print(f"🔗 Total Links Found: {stats['total_links']}")
    print(f"🔘 Total Buttons: {stats['total_buttons']}")
    print(f"📋 Total Forms: {stats['total_forms']}")
    print(f"🖼️  Total Images: {stats['total_images']}")
    print(f"📜 Total Scripts: {stats['total_scripts']}")
    print(f"🚀 Data ready for navigation planning!")
    print("="*60)


# Legacy compatibility
crawl_and_store = simple_crawl_and_store
