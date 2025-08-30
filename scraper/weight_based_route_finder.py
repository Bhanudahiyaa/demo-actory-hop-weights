#!/usr/bin/env python3
"""
Weight-Based Route Finder with Priority-Based Discovery

This system:
1. Calculates page weights based on DOM features
2. Discovers routes starting from highest priority pages
3. Shows routes in format: (from this - to this - to this)
4. Sorts routes by ascending hop count
5. Prioritizes by page weights (highest to lowest)
"""

import sys
import os
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
from collections import defaultdict, deque
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.neo4j_client import Neo4jClient


@dataclass
class PageInfo:
    """Information about a page including its weight and features."""
    url: str
    weight: int
    script_count: int
    form_count: int
    button_count: int
    input_count: int
    image_count: int
    link_count: int
    depth: int = 0


@dataclass
class RouteInfo:
    """Information about a discovered route."""
    path: List[str]
    hop_count: int
    total_weight: int
    start_weight: int
    end_weight: int
    priority_score: float


class WeightBasedRouteFinder:
    """Route finder that prioritizes by page weights and shows routes in specific format."""
    
    def __init__(self):
        """Initialize the route finder with Neo4j connection."""
        self.neo4j = Neo4jClient()
        self.pages_cache = {}
        self.routes_cache = {}
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        
    def close(self):
        """Close Neo4j connection."""
        if hasattr(self, 'neo4j'):
            self.neo4j.close()
    
    def calculate_page_weight(self, page_data: dict) -> int:
        """Calculate page weight based on DOM features."""
        weight = 0
        
        # Scripts are highest priority (JavaScript-heavy pages)
        weight += page_data.get('scriptCount', 0) * 10
        
        # Forms indicate interactive functionality
        weight += page_data.get('formCount', 0) * 8
        
        # Buttons show user interaction points
        weight += page_data.get('buttonCount', 0) * 6
        
        # Inputs indicate data collection
        weight += page_data.get('inputCount', 0) * 5
        
        # Images show content richness
        weight += page_data.get('imageCount', 0) * 2
        
        # Links show navigation complexity
        weight += page_data.get('linkCount', 0) * 1
        
        return weight
    
    def fetch_pages_with_weights(self) -> List[PageInfo]:
        """Fetch all pages from Neo4j with calculated weights."""
        print("📊 Fetching pages and calculating weights...")
        
        query = """
        MATCH (p:Page)
        RETURN p.url as url,
               COALESCE(p.scriptCount, 0) as scriptCount,
               COALESCE(p.formCount, 0) as formCount,
               COALESCE(p.buttonCount, 0) as buttonCount,
               COALESCE(p.inputCount, 0) as inputCount,
               COALESCE(p.imageCount, 0) as imageCount,
               COALESCE(p.linkCount, 0) as linkCount
        """
        
        pages = []
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            for record in result:
                page_data = {
                    'scriptCount': record['scriptCount'],
                    'formCount': record['formCount'],
                    'buttonCount': record['buttonCount'],
                    'inputCount': record['inputCount'],
                    'imageCount': record['imageCount'],
                    'linkCount': record['linkCount']
                }
                
                weight = self.calculate_page_weight(page_data)
                page_info = PageInfo(
                    url=record['url'],
                    weight=weight,
                    script_count=page_data['scriptCount'],
                    form_count=page_data['formCount'],
                    button_count=page_data['buttonCount'],
                    input_count=page_data['inputCount'],
                    image_count=page_data['imageCount'],
                    link_count=page_data['linkCount']
                )
                pages.append(page_info)
        
        # Sort by weight (highest to lowest)
        pages.sort(key=lambda x: x.weight, reverse=True)
        
        print(f"✅ Found {len(pages)} pages with weights")
        print(f"🏆 Top 5 pages by weight:")
        for i, page in enumerate(pages[:5]):
            print(f"   {i+1}. {page.url} (Weight: {page.weight})")
        
        return pages
    
    def fetch_page_connections(self) -> Dict[str, List[str]]:
        """Fetch all page connections from Neo4j."""
        print("🔗 Fetching page connections...")
        
        query = """
        MATCH (p1:Page)-[:LINKS_TO]->(p2:Page)
        RETURN p1.url as from_url, p2.url as to_url
        """
        
        connections = defaultdict(list)
        with self.neo4j._driver.session() as session:
            result = session.run(query)
            for record in result:
                from_url = record['from_url']
                to_url = record['to_url']
                connections[from_url].append(to_url)
        
        print(f"✅ Found connections for {len(connections)} pages")
        return connections
    
    def find_weighted_routes(self, start_url: str, max_depth: int = 6, max_routes: int = 1000) -> List[RouteInfo]:
        """Find routes starting from highest priority pages, sorted by hop count."""
        print(f"\n🎯 Finding weight-based routes from {start_url}")
        print(f"📊 Max depth: {max_depth}, Max routes: {max_routes}")
        
        # Fetch pages and connections
        pages = self.fetch_pages_with_weights()
        connections = self.fetch_page_connections()
        
        # Create URL to page info mapping
        url_to_page = {page.url: page for page in pages}
        
        # Find all routes using BFS
        all_routes = []
        visited = set()
        
        # Start BFS from the start URL
        queue = deque([(start_url, [start_url], 0)])
        visited.add(start_url)
        
        while queue and len(all_routes) < max_routes:
            current_url, current_path, current_depth = queue.popleft()
            
            # Add current route if it has multiple hops
            if len(current_path) > 1:
                route_info = self._create_route_info(current_path, url_to_page)
                all_routes.append(route_info)
            
            # Stop if we've reached max depth
            if current_depth >= max_depth:
                continue
            
            # Explore neighbors
            for neighbor_url in connections.get(current_url, []):
                if neighbor_url not in visited:
                    visited.add(neighbor_url)
                    new_path = current_path + [neighbor_url]
                    queue.append((neighbor_url, new_path, current_depth + 1))
        
        # Sort routes by hop count (ascending) and then by priority score (descending)
        all_routes.sort(key=lambda x: (x.hop_count, -x.priority_score))
        
        print(f"✅ Found {len(all_routes)} routes")
        return all_routes
    
    def _create_route_info(self, path: List[str], url_to_page: Dict[str, PageInfo]) -> RouteInfo:
        """Create RouteInfo object from a path."""
        hop_count = len(path) - 1
        
        # Get page info for start and end
        start_page = url_to_page.get(path[0], PageInfo(path[0], 0, 0, 0, 0, 0, 0, 0))
        end_page = url_to_page.get(path[-1], PageInfo(path[-1], 0, 0, 0, 0, 0, 0, 0))
        
        # Calculate total weight and priority score
        total_weight = sum(url_to_page.get(url, PageInfo(url, 0, 0, 0, 0, 0, 0, 0)).weight for url in path)
        priority_score = (start_page.weight * 0.4) + (end_page.weight * 0.3) + (total_weight * 0.3)
        
        return RouteInfo(
            path=path,
            hop_count=hop_count,
            total_weight=total_weight,
            start_weight=start_page.weight,
            end_weight=end_page.weight,
            priority_score=priority_score
        )
    
    def display_routes_in_format(self, routes: List[RouteInfo], max_display: int = 50):
        """Display routes in the requested format: (from this - to this - to this)"""
        print(f"\n🗺️  DISPLAYING ROUTES (Format: from this - to this - to this)")
        print(f"📊 Showing first {min(max_display, len(routes))} routes")
        print("=" * 80)
        
        # Group routes by hop count
        routes_by_hops = defaultdict(list)
        for route in routes:
            routes_by_hops[route.hop_count].append(route)
        
        # Display routes by hop count (ascending)
        for hop_count in sorted(routes_by_hops.keys()):
            hop_routes = routes_by_hops[hop_count]
            
            # Sort routes within same hop count by priority score
            hop_routes.sort(key=lambda x: -x.priority_score)
            
            print(f"\n🔄 HOP COUNT: {hop_count} ({len(hop_routes)} routes)")
            print("-" * 60)
            
            for i, route in enumerate(hop_routes[:max_display]):
                # Format: (from this - to this - to this)
                formatted_path = " - ".join(route.path)
                print(f"   {i+1:2d}. ({formatted_path})")
                print(f"       🏆 Priority: {route.priority_score:.1f}, Weight: {route.total_weight}")
                
                if i >= max_display - 1:
                    remaining = len(hop_routes) - max_display
                    if remaining > 0:
                        print(f"       ... and {remaining} more routes")
                    break
    
    def find_routes_to_targets(self, start_url: str, target_urls: List[str], max_depth: int = 6) -> List[RouteInfo]:
        """Find routes from start URL to specific target URLs, prioritized by weights."""
        print(f"\n🎯 Finding routes to specific targets:")
        for target in target_urls:
            print(f"   - {target}")
        
        # Fetch pages and connections
        pages = self.fetch_pages_with_weights()
        connections = self.fetch_page_connections()
        url_to_page = {page.url: page for page in pages}
        
        # Find routes to each target
        all_routes = []
        for target_url in target_urls:
            routes_to_target = self._find_routes_to_single_target(
                start_url, target_url, connections, url_to_page, max_depth
            )
            all_routes.extend(routes_to_target)
        
        # Sort by hop count and priority
        all_routes.sort(key=lambda x: (x.hop_count, -x.priority_score))
        
        print(f"✅ Found {len(all_routes)} routes to targets")
        return all_routes
    
    def _find_routes_to_single_target(self, start_url: str, target_url: str, 
                                    connections: Dict[str, List[str]], 
                                    url_to_page: Dict[str, PageInfo], 
                                    max_depth: int) -> List[RouteInfo]:
        """Find all routes from start to a single target."""
        routes = []
        queue = deque([(start_url, [start_url], 0)])
        visited = set()
        
        while queue:
            current_url, current_path, current_depth = queue.popleft()
            
            # Check if we reached the target
            if current_url == target_url and len(current_path) > 1:
                route_info = self._create_route_info(current_path, url_to_page)
                routes.append(route_info)
                continue
            
            # Stop if we've reached max depth
            if current_depth >= max_depth:
                continue
            
            # Explore neighbors
            for neighbor_url in connections.get(current_url, []):
                if neighbor_url not in visited:
                    visited.add(neighbor_url)
                    new_path = current_path + [neighbor_url]
                    queue.append((neighbor_url, new_path, current_depth + 1))
        
        return routes


def main():
    """Example usage of the weight-based route finder."""
    print("🚀 WEIGHT-BASED ROUTE FINDER DEMO")
    print("=" * 50)
    
    # Initialize finder
    finder = WeightBasedRouteFinder()
    
    try:
        # Find all routes from zineps.com
        print("\n1️⃣  Finding all weight-based routes...")
        all_routes = finder.find_weighted_routes("https://zineps.com", max_depth=5, max_routes=500)
        
        # Display routes in requested format
        finder.display_routes_in_format(all_routes, max_display=30)
        
        # Find routes to specific targets
        print("\n2️⃣  Finding routes to specific targets...")
        target_urls = [
            "https://www.zineps.ai/pricing",
            "https://www.zineps.ai/contact",
            "https://www.zineps.ai/support"
        ]
        
        target_routes = finder.find_routes_to_targets("https://zineps.com", target_urls, max_depth=5)
        finder.display_routes_in_format(target_routes, max_display=20)
        
    finally:
        finder.close()


if __name__ == "__main__":
    main()
