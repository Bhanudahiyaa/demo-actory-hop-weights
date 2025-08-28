#!/usr/bin/env python3
"""
DSPy Crawl Agent for autonomous website crawling with Neo4j integration.
"""

import asyncio
import time
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from collections import deque
from datetime import datetime

import dspy
from dspy import Module, InputField, OutputField

from .neo4j_client import Neo4jClient
from .plan import make_plan_from_neighbors, analyze_plan_quality
from .extractors import extract_all_dom_features


class CrawlPlanner(Module):
    """
    DSPy module that plans crawl strategy based on neighbor analysis.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, center_url: str, neighbors: List[Dict], max_pages: int) -> Dict[str, Any]:
        """Generate crawl plan from neighbors."""
        # Use the planning helper to create optimal crawl order
        crawl_plan = make_plan_from_neighbors(neighbors)
        
        # Limit to max_pages
        if len(crawl_plan) > max_pages:
            crawl_plan = crawl_plan[:max_pages]
        
        # Analyze plan quality
        plan_analysis = analyze_plan_quality(neighbors, crawl_plan)
        
        return {
            'crawl_plan': crawl_plan,
            'plan_analysis': plan_analysis,
            'center_url': center_url,
            'total_neighbors': len(neighbors),
            'planned_pages': len(crawl_plan)
        }


class Crawler(Module):
    """
    DSPy module that fetches URLs and stores metadata.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__()
        self.neo4j = neo4j_client
    
    def forward(self, url: str) -> Dict[str, Any]:
        """Crawl a URL and return comprehensive metadata."""
        start_time = time.time()
        
        try:
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; DSPyCrawler/1.0)'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response_time = time.time() - start_time
            
            # Extract content features
            html_content = response.text
            dom_features = extract_all_dom_features(html_content)
            
            # Calculate dynamic complexity score
            dynamic_score = (
                len(dom_features['scripts']) * 0.5 +
                len(dom_features['forms']) * 0.25 +
                (len(dom_features['buttons']) + len(dom_features['inputs'])) * 0.15 +
                len(dom_features['images']) * 0.1
            )
            
            # Store in Neo4j
            self.neo4j.upsert_page(
                url=url,
                title=dom_features.get('title', ''),
                weight=dynamic_score
            )
            
            # Mark as visited
            self.neo4j.mark_page_visited(url)
            
            # Store DOM features
            self._store_dom_features(url, dom_features)
            
            crawl_result = {
                'url': url,
                'status_code': response.status_code,
                'response_time': response_time,
                'content_length': len(html_content),
                'dynamic_score': dynamic_score,
                'features': {
                    'scripts': len(dom_features['scripts']),
                    'forms': len(dom_features['forms']),
                    'buttons': len(dom_features['buttons']),
                    'inputs': len(dom_features['inputs']),
                    'images': len(dom_features['images']),
                    'links': len(dom_features['links']),
                    'stylesheets': len(dom_features['stylesheets']),
                    'media': len(dom_features['media'])
                },
                'crawled_at': datetime.now().isoformat(),
                'success': True
            }
            
        except Exception as e:
            response_time = time.time() - start_time
            crawl_result = {
                'url': url,
                'status_code': None,
                'response_time': response_time,
                'content_length': 0,
                'dynamic_score': 0.0,
                'features': {
                    'scripts': 0, 'forms': 0, 'buttons': 0, 'inputs': 0,
                    'images': 0, 'links': 0, 'stylesheets': 0, 'media': 0
                },
                'crawled_at': datetime.now().isoformat(),
                'success': False,
                'error': str(e)
            }
        
        return crawl_result
    
    def _store_dom_features(self, url: str, dom_features: Dict):
        """Store extracted DOM features in Neo4j."""
        try:
            # Store buttons
            for button in dom_features['buttons']:
                self.neo4j.add_button(url, button)
            
            # Store forms
            for form in dom_features['forms']:
                self.neo4j.add_form(url, form)
            
            # Store inputs
            for input_field in dom_features['inputs']:
                self.neo4j.add_input_field(url, input_field)
            
            # Store images
            for image in dom_features['images']:
                src = image.get('src', '')
                if src:
                    self.neo4j.add_image(url, src, image.get('alt', ''))
            
            # Store scripts
            for script in dom_features['scripts']:
                self.neo4j.add_script(url, script)
            
            # Store stylesheets
            for stylesheet in dom_features['stylesheets']:
                self.neo4j.add_stylesheet(url, stylesheet)
            
            # Store links
            for link in dom_features['links']:
                href = link.get('href', '')
                text = link.get('text', '')
                title = link.get('title', '')
                if href:
                    self.neo4j.add_link(url, href, text, title)
                    
        except Exception as e:
            print(f"Warning: Failed to store DOM features for {url}: {e}")


class Analyzer(Module):
    """
    DSPy module that analyzes crawl results and generates reports.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, crawl_results: List[Dict], crawl_metadata: Dict) -> Dict[str, Any]:
        """Generate comprehensive crawl analysis report with autonomous insights."""
        if not crawl_results:
            return {
                'summary': 'No pages crawled',
                'statistics': {},
                'top_pages': [],
                'sample_list': [],
                'autonomous_insights': {}
            }
        
        # Calculate statistics
        successful_crawls = [r for r in crawl_results if r.get('success', False)]
        failed_crawls = [r for r in crawl_results if not r.get('success', False)]
        
        total_pages = len(crawl_results)
        successful_pages = len(successful_crawls)
        failed_pages = len(failed_crawls)
        
        # Response time analysis
        response_times = [r.get('response_time', 0) for r in successful_crawls]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        # Dynamic score analysis
        dynamic_scores = [r.get('dynamic_score', 0) for r in successful_crawls]
        avg_dynamic_score = sum(dynamic_scores) / len(dynamic_scores) if dynamic_scores else 0
        max_dynamic_score = max(dynamic_scores) if dynamic_scores else 0
        min_dynamic_score = min(dynamic_scores) if dynamic_scores else 0
        
        # Content analysis
        total_scripts = sum(r['features']['scripts'] for r in successful_crawls)
        total_forms = sum(r['features']['forms'] for r in successful_crawls)
        total_buttons = sum(r['features']['buttons'] for r in successful_crawls)
        total_images = sum(r['features']['images'] for r in successful_crawls)
        total_links = sum(r['features']['links'] for r in successful_crawls)
        total_stylesheets = sum(r['features']['stylesheets'] for r in successful_crawls)
        total_media = sum(r['features']['media'] for r in successful_crawls)
        
        # Top pages by dynamic score
        top_pages = sorted(
            successful_crawls,
            key=lambda x: x.get('dynamic_score', 0),
            reverse=True
        )[:10]
        
        # Sample of crawled pages
        sample_list = successful_crawls[:20]  # First 20 successful crawls
        
        # Autonomous insights
        autonomous_insights = {
            'discovery_efficiency': {
                'total_discovered': crawl_metadata.get('total_urls_discovered', 0),
                'total_crawled': total_pages,
                'discovery_ratio': (crawl_metadata.get('total_urls_discovered', 0) / total_pages) if total_pages > 0 else 0
            },
            'depth_exploration': {
                'max_depth_reached': crawl_metadata.get('max_depth_reached', 0),
                'avg_depth': sum(r.get('depth', 0) for r in crawl_results) / len(crawl_results) if crawl_results else 0
            },
            'content_complexity': {
                'avg_scripts_per_page': total_scripts / total_pages if total_pages > 0 else 0,
                'avg_forms_per_page': total_forms / total_pages if total_pages > 0 else 0,
                'avg_images_per_page': total_images / total_pages if total_pages > 0 else 0,
                'most_complex_page': max(successful_crawls, key=lambda x: x.get('dynamic_score', 0))['url'] if successful_crawls else None
            }
        }
        
        # Generate report
        report = {
            'summary': {
                'total_pages_crawled': total_pages,
                'successful_crawls': successful_pages,
                'failed_crawls': failed_pages,
                'success_rate': (successful_pages / total_pages * 100) if total_pages > 0 else 0,
                'max_depth_reached': crawl_metadata.get('max_depth_reached', 0),
                'crawl_duration': crawl_metadata.get('crawl_duration', 0),
                'total_urls_discovered': crawl_metadata.get('total_urls_discovered', 0),
                'autonomous_expansions': crawl_metadata.get('autonomous_expansions', 0)
            },
            'statistics': {
                'avg_response_time': round(avg_response_time, 3),
                'max_response_time': round(max_response_time, 3),
                'min_response_time': round(min_response_time, 3),
                'avg_dynamic_score': round(avg_dynamic_score, 3),
                'max_dynamic_score': round(max_dynamic_score, 3),
                'min_dynamic_score': round(min_dynamic_score, 3),
                'total_scripts': total_scripts,
                'total_forms': total_forms,
                'total_buttons': total_buttons,
                'total_images': total_images,
                'total_links': total_links,
                'total_stylesheets': total_stylesheets,
                'total_media': total_media
            },
            'top_pages': [
                {
                    'url': page['url'],
                    'dynamic_score': page.get('dynamic_score', 0),
                    'response_time': page.get('response_time', 0),
                    'features': page['features']
                }
                for page in top_pages
            ],
            'sample_list': [
                {
                    'url': page['url'],
                    'status_code': page.get('status_code'),
                    'dynamic_score': page.get('dynamic_score', 0),
                    'response_time': page.get('response_time', 0),
                    'content_length': page.get('content_length', 0)
                }
                for page in sample_list
            ],
            'autonomous_insights': autonomous_insights
        }
        
        return report


class CrawlAgent(Module):
    """
    Main DSPy module that orchestrates autonomous website crawling.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__()
        self.neo4j = neo4j_client
        self.crawl_planner = CrawlPlanner()
        self.crawler = Crawler(neo4j_client)
        self.analyzer = Analyzer()
    
    def forward(self, start_url: str, max_depth: int, max_pages: int) -> Dict[str, Any]:
        """Execute fully autonomous website crawling with dynamic neighbor discovery."""
        start_time = time.time()
        
        # Initialize crawl state
        crawl_queue = deque([(start_url, 0)])  # (url, depth)
        visited_urls = set()
        crawl_results = []
        max_depth_reached = 0
        discovered_urls = set([start_url])  # Track all discovered URLs
        
        print(f"🚀 Starting FULLY AUTONOMOUS crawl from: {start_url}")
        print(f"📊 Limits: max_depth={max_depth}, max_pages={max_pages}")
        print("🎯 Agent will autonomously discover, plan, and crawl the entire site!")
        print("-" * 80)
        
        # Main autonomous crawl loop
        while crawl_queue and len(crawl_results) < max_pages:
            current_url, current_depth = crawl_queue.popleft()
            
            # Skip if already visited or depth exceeded
            if current_url in visited_urls or current_depth > max_depth:
                continue
            
            print(f"\n🕷️  CRAWLING [{current_depth}/{max_depth}]: {current_url}")
            print("-" * 60)
            
            # Mark as visited
            visited_urls.add(current_url)
            
            # Crawl the page with comprehensive analysis
            crawl_result = self.crawler.forward(current_url)
            crawl_results.append(crawl_result)
            
            # Update max depth reached
            max_depth_reached = max(max_depth_reached, current_depth)
            
            # Print comprehensive page analysis
            if crawl_result.get('success', False):
                self._print_page_analysis(crawl_result)
                
                # AUTONOMOUS NEIGHBOR DISCOVERY AND PLANNING
                if current_depth < max_depth:
                    self._autonomous_neighbor_expansion(
                        current_url, current_depth, crawl_queue, visited_urls, 
                        discovered_urls, max_pages, crawl_results
                    )
            else:
                print(f"   ❌ FAILED | Error: {crawl_result.get('error', 'Unknown')}")
            
            print()
        
        # Calculate crawl duration
        crawl_duration = time.time() - start_time
        
        # Generate comprehensive final analysis
        crawl_metadata = {
            'start_url': start_url,
            'max_depth': max_depth,
            'max_pages': max_pages,
            'max_depth_reached': max_depth_reached,
            'crawl_duration': crawl_duration,
            'total_pages_crawled': len(crawl_results),
            'total_urls_discovered': len(discovered_urls),
            'autonomous_expansions': len(discovered_urls) - len(crawl_results)
        }
        
        analysis_report = self.analyzer.forward(crawl_results, crawl_metadata)
        
        # Compile final result
        final_result = {
            'crawl_metadata': crawl_metadata,
            'crawl_results': crawl_results,
            'analysis_report': analysis_report,
            'discovery_summary': {
                'total_discovered': len(discovered_urls),
                'total_crawled': len(crawl_results),
                'autonomous_discoveries': len(discovered_urls) - len(crawl_results)
            }
        }
        
        return final_result
    
    def _print_page_analysis(self, crawl_result: Dict[str, Any]):
        """Print comprehensive analysis for a single crawled page."""
        features = crawl_result['features']
        print(f"   📊 PAGE ANALYSIS:")
        print(f"      🎯 Dynamic Score: {crawl_result['dynamic_score']:.2f}")
        print(f"      ⚡ Response Time: {crawl_result['response_time']:.3f}s")
        print(f"      📄 Content Size: {crawl_result['content_length']:,} chars")
        print(f"      🔧 Features:")
        print(f"         📜 Scripts: {features['scripts']} | 📋 Forms: {features['forms']}")
        print(f"         🔘 Buttons: {features['buttons']} | 📝 Inputs: {features['inputs']}")
        print(f"         🖼️  Images: {features['images']} | 🔗 Links: {features['links']}")
        print(f"         🎨 Stylesheets: {features['stylesheets']} | 🎵 Media: {features['media']}")
    
    def _autonomous_neighbor_expansion(self, current_url: str, current_depth: int, 
                                     crawl_queue: deque, visited_urls: set, 
                                     discovered_urls: set, max_pages: int, 
                                     crawl_results: List[Dict]):
        """Autonomously discover and plan neighbors for intelligent crawling."""
        try:
            print(f"   🔍 AUTONOMOUS NEIGHBOR DISCOVERY:")
            
            # Get neighbors from current page - include both visited and unvisited for comprehensive discovery
            neighbors = self.neo4j.get_neighbors_hop1(
                current_url, 
                direction="out", 
                limit=30,  # Increased limit for better discovery
                exclude_visited=False  # Include all neighbors for autonomous discovery
            )
            
            if neighbors:
                # Plan next crawl targets intelligently
                plan_result = self.crawl_planner.forward(
                    center_url=current_url,
                    neighbors=neighbors,
                    max_pages=max_pages - len(crawl_results)
                )
                
                # Add newly discovered URLs to discovered set
                for planned_url in plan_result['crawl_plan']:
                    discovered_urls.add(planned_url)
                
                # Add planned URLs to queue (excluding already visited)
                new_urls_added = 0
                for planned_url in plan_result['crawl_plan']:
                    if planned_url not in visited_urls:
                        crawl_queue.append((planned_url, current_depth + 1))
                        new_urls_added += 1
                
                # Show visited vs unvisited breakdown
                total_neighbors = len(neighbors)
                visited_neighbors = sum(1 for n in neighbors if n.get('visited', False))
                unvisited_neighbors = total_neighbors - visited_neighbors
                print(f"      📊 Neighbor Status: {visited_neighbors} visited, {unvisited_neighbors} unvisited")
                
                print(f"      📋 Found {len(neighbors)} neighbors")
                print(f"      🗺️  Planned {len(plan_result['crawl_plan'])} URLs")
                print(f"      ➕ Added {new_urls_added} new URLs to queue")
                print(f"      🎯 Queue size: {len(crawl_queue)} | Visited: {len(visited_urls)}")
                
                # Show top planned URLs
                top_planned = plan_result['crawl_plan'][:5]
                print(f"      🏆 Top planned URLs:")
                for i, url in enumerate(top_planned, 1):
                    print(f"         {i}. {url}")
                
            else:
                print(f"      ⚠️  No neighbors found - reached boundary")
                
        except Exception as e:
            print(f"      ❌ Error in neighbor expansion: {e}")
    
    def _smart_queue_management(self, crawl_queue: deque, visited_urls: set, 
                               discovered_urls: set, max_pages: int) -> deque:
        """Intelligently manage the crawl queue to explore new areas."""
        # Remove duplicates and already visited URLs
        unique_queue = deque()
        seen_in_queue = set()
        
        for url, depth in crawl_queue:
            if url not in seen_in_queue and url not in visited_urls:
                unique_queue.append((url, depth))
                seen_in_queue.add(url)
        
        # Sort by depth to maintain BFS-like exploration
        sorted_queue = deque(sorted(unique_queue, key=lambda x: x[1]))
        
        return sorted_queue

