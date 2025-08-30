#!/usr/bin/env python3
"""
Enhanced Web Scraping with LLM Intelligence
Clean, organized module for intelligent content extraction and analysis.
"""

import asyncio
from typing import List, Optional
from urllib.parse import urlparse

from crawl4ai import AsyncWebCrawler
from crawl4ai.deep_crawling.bfs_strategy import BFSDeepCrawlStrategy
from crawl4ai.async_configs import CrawlerRunConfig

from .neo4j_client import Neo4jClient
from .extractors import extract_all_dom_features
from .llm_extractor import QwenLLMExtractor, calculate_enhanced_complexity_score


async def enhanced_crawl_and_store(
    url: str,
    depth: int,
    max_pages: int,
    include_external: bool,
    neo4j: Optional[Neo4jClient],
    enable_llm_analysis: bool = True,
):
    """
    Enhanced crawling with LLM-powered content intelligence.
    Provides business value scoring and UX analysis for better planning agent decisions.
    """
    
    print(f"🚀 Starting Enhanced Crawl with LLM Intelligence")
    print(f"🎯 Target: {url}")
    print(f"📊 Depth: {depth}, Pages: {max_pages}")
    print(f"🔗 External Links: {'Yes' if include_external else 'No'}")
    print(f"🧠 LLM Analysis: {'Yes' if enable_llm_analysis else 'No'}")
    print("-" * 80)
    
    # Initialize LLM extractor if enabled
    llm_extractor = None
    if enable_llm_analysis:
        try:
            llm_extractor = QwenLLMExtractor()
            print(f"🧠 GPT-OSS-20B Intelligence: {llm_extractor.get_extraction_stats()}")
        except Exception as e:
            print(f"⚠️  LLM initialization failed: {e}")
            llm_extractor = None

    # Configure crawler
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

    # Track statistics
    enhanced_stats = {
        'total_pages': 0,
        'llm_analyzed_pages': 0,
        'avg_business_score': 0,
        'avg_ux_score': 0,
        'total_business_score': 0,
        'total_ux_score': 0
    }

    async with AsyncWebCrawler() as crawler:
        results = await crawler.arun(url, config=config)
        pages: List[str] = []

        async for result in results:
            page_url = result.url
            pages.append(page_url)
            enhanced_stats['total_pages'] += 1
            
            if neo4j:
                neo4j.upsert_page(page_url)

            # Extract DOM features
            dom_features = extract_all_dom_features(result.html)
            
            # LLM-powered content intelligence extraction
            llm_intelligence = None
            enhanced_score = None
            if llm_extractor:
                try:
                    llm_intelligence = llm_extractor.extract_page_intelligence(
                        page_url=page_url,
                        html_content=result.html,
                        dom_features=dom_features
                    )
                    
                    # Calculate enhanced complexity score
                    enhanced_score = calculate_enhanced_complexity_score(dom_features, llm_intelligence)
                    
                    # Update statistics
                    enhanced_stats['llm_analyzed_pages'] += 1
                    enhanced_stats['total_business_score'] += llm_intelligence.get('business_value_score', 0)
                    enhanced_stats['total_ux_score'] += llm_intelligence.get('user_experience_score', 0)
                    
                    print(f"   🧠 GPT-OSS-20B Analysis: Business {llm_intelligence.get('business_value_score', 0):.1f}/10, UX {llm_intelligence.get('user_experience_score', 0):.1f}/10, Enhanced Score: {enhanced_score:.2f}")
                    
                except Exception as e:
                    print(f"   ⚠️  LLM extraction failed: {e}")
                    llm_intelligence = None
            
            # Process links
            await _process_links(result, page_url, neo4j, include_external, url)
            
            # Store DOM features
            if neo4j:
                await _store_dom_features(neo4j, page_url, dom_features)
                
                # Store LLM insights
                if llm_intelligence:
                    neo4j.add_llm_insights(page_url, llm_intelligence)

            # Print page summary
            _print_page_summary(page_url, dom_features, llm_intelligence, enhanced_score)
            
            # Create LINKS_TO relationship from start
            if neo4j and page_url != url:
                try:
                    neo4j.link_pages(url, page_url)
                    print(f"   🔗 Created LINKS_TO from start: {url} -> {page_url}")
                except Exception as e:
                    print(f"   ⚠️  Warning: Could not create LINKS_TO from start: {e}")
            
            print()

        # Calculate final statistics
        if enhanced_stats['llm_analyzed_pages'] > 0:
            enhanced_stats['avg_business_score'] = enhanced_stats['total_business_score'] / enhanced_stats['llm_analyzed_pages']
            enhanced_stats['avg_ux_score'] = enhanced_stats['total_ux_score'] / enhanced_stats['llm_analyzed_pages']

        # Print enhanced crawl summary
        _print_enhanced_summary(enhanced_stats)

        return pages


async def _process_links(result, page_url: str, neo4j: Optional[Neo4jClient], include_external: bool, start_url: str):
    """Process and store link information."""
    if not neo4j:
        return
        
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
        
        # Create LINKS_TO relationships
        if href.startswith("http"):
            current_domain = urlparse(page_url).netloc
            link_domain = urlparse(href).netloc
            
            if include_external or current_domain == link_domain:
                neo4j.link_pages(page_url, href)
                print(f"   🔗 Created LINKS_TO: {page_url} -> {href}")
        elif href.startswith("/"):
            try:
                parsed_url = urlparse(page_url)
                absolute_href = f"{parsed_url.scheme}://{parsed_url.netloc}{href}"
                neo4j.link_pages(page_url, absolute_href)
                print(f"   🔗 Created LINKS_TO: {page_url} -> {absolute_href}")
            except Exception as e:
                print(f"   ⚠️  Warning: Could not process relative link {href}: {e}")


async def _store_dom_features(neo4j: Neo4jClient, page_url: str, dom_features: dict):
    """Store all extracted DOM features in Neo4j."""
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


def _print_page_summary(page_url: str, dom_features: dict, llm_intelligence: Optional[dict], enhanced_score: Optional[float]):
    """Print comprehensive page summary."""
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
    
    if llm_intelligence:
        print(f"   🧠 Page Type: {llm_intelligence.get('page_type', 'unknown')}")
        print(f"   💼 Business Value: {llm_intelligence.get('business_value_score', 0):.1f}/10")
        print(f"   🎯 User Experience: {llm_intelligence.get('user_experience_score', 0):.1f}/10")
        print(f"   📊 Planning Priority: {llm_intelligence.get('planning_priority_score', 0):.1f}/10")
        print(f"   📈 Enhanced Score: {enhanced_score:.2f}")


def _print_enhanced_summary(stats: dict):
    """Print enhanced crawl summary."""
    print("\n" + "="*80)
    print("🧠 ENHANCED CRAWL SUMMARY WITH LLM INTELLIGENCE")
    print("="*80)
    print(f"📊 Total Pages Crawled: {stats['total_pages']}")
    print(f"🧠 LLM Analyzed Pages: {stats['llm_analyzed_pages']}")
    
    if stats['llm_analyzed_pages'] > 0:
        print(f"💼 Average Business Value: {stats['avg_business_score']:.1f}/10")
        print(f"🎯 Average UX Score: {stats['avg_ux_score']:.1f}/10")
        print(f"📈 LLM Enhancement Rate: {(stats['llm_analyzed_pages']/stats['total_pages']*100):.1f}%")
    
    print(f"🚀 Enhanced data ready for intelligent planning agent!")
    print("="*80)


# Legacy compatibility
crawl_and_store = enhanced_crawl_and_store
