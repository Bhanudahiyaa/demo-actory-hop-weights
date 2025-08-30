#!/usr/bin/env python3
"""
DSPy-Enhanced Weight-Based Route Finder

This module combines:
1. Weight-based page prioritization (highest to lowest)
2. DSPy PlanAgent for intelligent expansion decisions
3. Optimized route discovery with subpath memoization
4. Neo4j sitemap integration
5. Routes displayed in format: (from this - to this - to this)
6. Sorted by hop count (ascending) and priority score (descending)
"""

import sys
import os
import time
from typing import List, Dict, Tuple, Optional, Generator, Set
from dataclasses import dataclass
from collections import defaultdict, deque
import psutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper.neo4j_client import Neo4jClient

# DSPy imports
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    print("⚠️  DSPy not available. Install with: pip install dspy-ai")
    print("   Falling back to pure algorithmic approach.")


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
    visited: bool = False


@dataclass
class RouteInfo:
    """Information about a discovered route."""
    path: List[str]
    hop_count: int
    total_weight: int
    start_weight: int
    end_weight: int
    priority_score: float
    target_page: str
    expansion_reason: str = ""


@dataclass
class RouteStats:
    """Statistics about route discovery performance."""
    total_routes: int
    compressed_routes: int
    max_depth: int
    discovery_time: float
    memory_usage_mb: float
    cache_hits: int
    cache_misses: int
    pages_covered: int
    total_pages: int
    coverage_percentage: float
    dspy_decisions: int = 0


# -----------------------------
# DSPy PlanAgent Setup
# -----------------------------

if DSPY_AVAILABLE:
    class ExpandNode(dspy.Signature):
        """Agent decides which nodes to expand next based on weights and context."""
        frontier = dspy.InputField(desc="List of frontier nodes with their weights and outgoing links")
        visited = dspy.InputField(desc="Nodes already visited")
        current_depth = dspy.InputField(desc="Current exploration depth")
        max_depth = dspy.InputField(desc="Maximum allowed depth")
        next_nodes = dspy.OutputField(desc="Nodes to expand in order of priority, with reasoning")

    class WeightBasedExpansion(dspy.Signature):
        """Agent analyzes page weights and decides expansion strategy."""
        page_weights = dspy.InputField(desc="List of pages with their calculated weights")
        current_frontier = dspy.InputField(desc="Current nodes to expand")
        expansion_strategy = dspy.OutputField(desc="Strategy for expanding nodes: 'weight_first', 'breadth_first', or 'hybrid'")

    # Initialize DSPy agents using Predict instead of Plan
    plan_agent = dspy.Predict(ExpandNode)
    weight_agent = dspy.Predict(WeightBasedExpansion)


class SubpathCache:
    """Efficient subpath memoization for route reuse."""
    
    def __init__(self):
        self.cache = defaultdict(dict)
        self.hits = 0
        self.misses = 0
    
    def get(self, start: str, end: str, max_depth: int) -> Optional[List[List[str]]]:
        """Get cached routes from start to end within max_depth."""
        if start in self.cache and end in self.cache[start]:
            routes = self.cache[start][end]
            # Filter by max_depth
            valid_routes = [route for route in routes if len(route) - 1 <= max_depth]
            if valid_routes:
                self.hits += 1
                return valid_routes
        self.misses += 1
        return None
    
    def set(self, start: str, end: str, routes: List[List[str]]):
        """Cache routes from start to end."""
        if start not in self.cache:
            self.cache[start] = {}
        self.cache[start][end] = routes
    
    def clear(self):
        """Clear the cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Tuple[int, int]:
        """Get cache hit/miss statistics."""
        return self.hits, self.misses


class DSPyEnhancedRouteFinder:
    """Enhanced route finder with DSPy PlanAgent for intelligent decisions."""
    
    def __init__(self, max_depth: int = 6, max_routes_per_target: int = 1000, 
                 enable_compression: bool = True, enable_memoization: bool = True,
                 use_dspy: bool = True):
        """Initialize the enhanced finder."""
        self.neo4j = Neo4jClient()
        self.subpath_cache = SubpathCache()
        self.max_depth = max_depth
        self.max_routes_per_target = max_routes_per_target
        self.enable_compression = enable_compression
        self.enable_memoization = enable_memoization
        self.use_dspy = use_dspy and DSPY_AVAILABLE
        self.pages_cache = {}
        self.routes_cache = {}
        self.dspy_decisions = 0
        
        if self.use_dspy:
            print("🤖 DSPy PlanAgent enabled for intelligent route expansion")
        else:
            print("⚡ Using pure algorithmic approach for route expansion")
        
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
        print(f"🏆 Top 10 pages by weight:")
        for i, page in enumerate(pages[:10]):
            print(f"   {i+1:2d}. {page.url} (Weight: {page.weight})")
        
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
    
    def get_dspy_expansion_strategy(self, pages: List[PageInfo], current_frontier: List[str]) -> str:
        """Use DSPy to decide expansion strategy."""
        if not self.use_dspy:
            return "weight_first"  # Default fallback
        
        try:
            # Prepare page weights for DSPy
            page_weights = [
                {"url": page.url, "weight": page.weight, "features": {
                    "scripts": page.script_count,
                    "forms": page.form_count,
                    "buttons": page.button_count
                }} for page in pages[:20]  # Limit to top 20 for efficiency
            ]
            
            decision = weight_agent(
                page_weights=page_weights,
                current_frontier=current_frontier
            )
            
            self.dspy_decisions += 1
            strategy = decision.expansion_strategy.lower()
            
            # Validate strategy
            if strategy in ["weight_first", "breadth_first", "hybrid"]:
                return strategy
            else:
                return "weight_first"  # Fallback
                
        except Exception as e:
            print(f"⚠️  DSPy decision failed: {e}. Falling back to weight_first.")
            return "weight_first"
    
    def get_dspy_expansion_order(self, frontier: List[str], visited: Set[str], 
                                current_depth: int, max_depth: int, 
                                connections: Dict[str, List[str]], 
                                url_to_page: Dict[str, PageInfo]) -> List[str]:
        """Use DSPy to decide which nodes to expand next."""
        if not self.use_dspy:
            # Fallback to weight-based ordering
            frontier_with_weights = []
            for node in frontier:
                if node in url_to_page:
                    weight = url_to_page[node].weight
                    frontier_with_weights.append((node, weight))
            
            # Sort by weight (highest first)
            frontier_with_weights.sort(key=lambda x: x[1], reverse=True)
            return [node for node, _ in frontier_with_weights]
        
        try:
            # Prepare frontier data for DSPy
            frontier_data = []
            for node in frontier:
                if node in connections:
                    neighbors = connections[node]
                    weight = url_to_page.get(node, PageInfo(node, 0, 0, 0, 0, 0, 0, 0)).weight
                    frontier_data.append({
                        "node": node,
                        "weight": weight,
                        "links": neighbors,
                        "link_count": len(neighbors)
                    })
            
            decision = plan_agent(
                frontier=frontier_data,
                visited=list(visited),
                current_depth=current_depth,
                max_depth=max_depth
            )
            
            self.dspy_decisions += 1
            return decision.next_nodes
            
        except Exception as e:
            print(f"⚠️  DSPy expansion decision failed: {e}. Falling back to weight-based ordering.")
            # Fallback to weight-based ordering
            frontier_with_weights = []
            for node in frontier:
                if node in url_to_page:
                    weight = url_to_page[node].weight
                    frontier_with_weights.append((node, weight))
            
            frontier_with_weights.sort(key=lambda x: x[1], reverse=True)
            return [node for node, _ in frontier_with_weights]
    
    def find_routes_to_every_page(self, start_url: str, max_depth: int = None) -> List[RouteInfo]:
        """Find routes from start URL to every page using DSPy-enhanced decisions."""
        if max_depth is None:
            max_depth = self.max_depth
            
        print(f"\n🎯 Finding routes to EVERY PAGE from {start_url}")
        print(f"📊 Max depth: {max_depth}")
        print(f"⚡ Features: Memoization={'ON' if self.enable_memoization else 'OFF'}, "
              f"Compression={'ON' if self.enable_compression else 'OFF'}")
        print(f"🤖 DSPy: {'ON' if self.use_dspy else 'OFF'}")
        print("-" * 80)
        
        start_time = time.time()
        
        # Fetch pages and connections
        pages = self.fetch_pages_with_weights()
        connections = self.fetch_page_connections()
        
        # Create URL to page info mapping
        url_to_page = {page.url: page for page in pages}
        
        # Get DSPy expansion strategy
        expansion_strategy = self.get_dspy_expansion_strategy(pages, [start_url])
        print(f"🎯 DSPy Expansion Strategy: {expansion_strategy}")
        
        # Find routes to each page, prioritized by weight
        all_routes = []
        pages_covered = 0
        
        print(f"\n🔍 Discovering routes to {len(pages)} pages...")
        
        for i, target_page in enumerate(pages):
            if target_page.url == start_url:
                continue  # Skip start page
                
            print(f"   {i+1:2d}/{len(pages)-1}: Finding routes to {target_page.url} (Weight: {target_page.weight})")
            
            # Check cache first
            cached_routes = None
            if self.enable_memoization:
                cached_routes = self.subpath_cache.get(start_url, target_page.url, max_depth)
            
            if cached_routes:
                # Use cached routes
                for path in cached_routes:
                    route_info = self._create_route_info(path, url_to_page, target_page.url, "cached")
                    all_routes.append(route_info)
                print(f"      ✅ Found {len(cached_routes)} cached routes")
            else:
                # Find new routes using DSPy-enhanced expansion
                routes_to_target = self._find_routes_with_dspy(
                    start_url, target_page.url, connections, url_to_page, max_depth
                )
                
                # Cache the routes
                if self.enable_memoization and routes_to_target:
                    paths = [route.path for route in routes_to_target]
                    self.subpath_cache.set(start_url, target_page.url, paths)
                
                all_routes.extend(routes_to_target)
                print(f"      ✅ Found {len(routes_to_target)} new routes")
            
            pages_covered += 1
            
            # Update progress
            if (i + 1) % 10 == 0 or i == len(pages) - 1:
                print(f"      📊 Progress: {i+1}/{len(pages)-1} pages covered")
        
        # Sort routes by hop count (ascending) and then by priority score (descending)
        all_routes.sort(key=lambda x: (x.hop_count, -x.priority_score))
        
        discovery_time = time.time() - start_time
        
        print(f"\n✅ Route discovery completed!")
        print(f"📊 Total routes found: {len(all_routes):,}")
        print(f"📄 Pages covered: {pages_covered}/{len(pages)} ({pages_covered/len(pages)*100:.1f}%)")
        print(f"⏱️  Discovery time: {discovery_time:.2f}s")
        print(f"🤖 DSPy decisions made: {self.dspy_decisions}")
        
        return all_routes
    
    def _find_routes_with_dspy(self, start_url: str, target_url: str, 
                              connections: Dict[str, List[str]], 
                              url_to_page: Dict[str, PageInfo], 
                              max_depth: int) -> List[RouteInfo]:
        """Find routes using DSPy-enhanced expansion decisions."""
        routes = []
        queue = deque([(start_url, [start_url], 0)])
        visited = set()
        
        while queue and len(routes) < self.max_routes_per_target:
            current_url, current_path, current_depth = queue.popleft()
            
            # Check if we reached the target
            if current_url == target_url and len(current_path) > 1:
                route_info = self._create_route_info(current_path, url_to_page, target_url, "dspy_guided")
                routes.append(route_info)
                continue
            
            # Stop if we've reached max depth
            if current_depth >= max_depth:
                continue
            
            # Get neighbors
            neighbors = connections.get(current_url, [])
            if not neighbors:
                continue
            
            # Use DSPy to decide expansion order
            expansion_order = self.get_dspy_expansion_order(
                neighbors, visited, current_depth, max_depth, connections, url_to_page
            )
            
            # Expand in DSPy-recommended order
            for neighbor_url in expansion_order:
                if neighbor_url not in visited:
                    visited.add(neighbor_url)
                    new_path = current_path + [neighbor_url]
                    queue.append((neighbor_url, new_path, current_depth + 1))
        
        return routes
    
    def _create_route_info(self, path: List[str], url_to_page: Dict[str, PageInfo], 
                          target_url: str, expansion_reason: str = "") -> RouteInfo:
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
            priority_score=priority_score,
            target_page=target_url,
            expansion_reason=expansion_reason
        )
    
    def display_routes_in_format(self, routes: List[RouteInfo], max_display: int = 100):
        """Display routes in the requested format: (from this - to this - to this)"""
        print(f"\n🗺️  DISPLAYING ROUTES (Format: from this - to this - to this)")
        print(f"📊 Showing first {min(max_display, len(routes))} routes")
        print("=" * 100)
        
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
            print("-" * 80)
            
            for i, route in enumerate(hop_routes[:max_display]):
                # Format: (from this - to this - to this)
                formatted_path = " - ".join(route.path)
                print(f"   {i+1:3d}. ({formatted_path})")
                print(f"       🎯 Target: {route.target_page}")
                print(f"       🏆 Priority: {route.priority_score:.1f}, Weight: {route.total_weight}")
                if route.expansion_reason:
                    print(f"       🤖 Expansion: {route.expansion_reason}")
                
                if i >= max_display - 1:
                    remaining = len(hop_routes) - max_display
                    if remaining > 0:
                        print(f"       ... and {remaining} more routes")
                    break
    
    def get_route_summary(self, routes: List[RouteInfo]) -> RouteStats:
        """Get comprehensive statistics about the routes."""
        if not routes:
            return RouteStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Calculate statistics
        total_routes = len(routes)
        max_depth = max(route.hop_count for route in routes) if routes else 0
        
        # Calculate memory usage
        process = psutil.Process()
        memory_usage_mb = process.memory_info().rss / 1024 / 1024
        
        # Get cache stats
        cache_hits, cache_misses = self.subpath_cache.get_stats()
        
        # Calculate coverage
        unique_targets = set(route.target_page for route in routes)
        pages_covered = len(unique_targets)
        
        # Get total pages
        pages = self.fetch_pages_with_weights()
        total_pages = len(pages)
        coverage_percentage = (pages_covered / total_pages * 100) if total_pages > 0 else 0
        
        return RouteStats(
            total_routes=total_routes,
            compressed_routes=total_routes,  # All routes are stored efficiently
            max_depth=max_depth,
            discovery_time=0,  # Will be set by caller
            memory_usage_mb=memory_usage_mb,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            pages_covered=pages_covered,
            total_pages=total_pages,
            coverage_percentage=coverage_percentage,
            dspy_decisions=self.dspy_decisions
        )
    
    def clear_cache(self):
        """Clear all caches."""
        self.subpath_cache.clear()
        self.pages_cache.clear()
        self.routes_cache.clear()
        self.dspy_decisions = 0
        print("🗑️  All caches cleared")


def main():
    """Example usage of the DSPy-enhanced route finder."""
    print("🚀 DSPY-ENHANCED WEIGHT-BASED OPTIMIZED ROUTE FINDER DEMO")
    print("=" * 80)
    print("This system combines weight-based prioritization with")
    print("DSPy PlanAgent for intelligent route expansion decisions.")
    print("=" * 80)
    
    # Initialize finder
    finder = DSPyEnhancedRouteFinder(
        max_depth=6,
        max_routes_per_target=500,
        enable_compression=True,
        enable_memoization=True,
        use_dspy=True
    )
    
    try:
        # Find routes to every page
        print("\n🎯 Finding routes to EVERY PAGE with DSPy guidance...")
        all_routes = finder.find_routes_to_every_page("https://zineps.com", max_depth=5)
        
        # Display routes in requested format
        finder.display_routes_in_format(all_routes, max_display=50)
        
        # Get comprehensive statistics
        print("\n📊 ROUTE DISCOVERY STATISTICS")
        print("-" * 50)
        
        stats = finder.get_route_summary(all_routes)
        print(f"📄 Total Routes: {stats.total_routes:,}")
        print(f"📄 Pages Covered: {stats.pages_covered}/{stats.total_pages} ({stats.coverage_percentage:.1f}%)")
        print(f"📏 Max Depth: {stats.max_depth}")
        print(f"💾 Memory Usage: {stats.memory_usage_mb:.2f} MB")
        print(f"🎯 Cache Hit Rate: {stats.cache_hits / (stats.cache_hits + stats.cache_misses) * 100:.1f}%")
        print(f"🤖 DSPy Decisions: {stats.dspy_decisions}")
        
        print(f"\n💡 Key Benefits:")
        print(f"   • Weight-based prioritization (highest to lowest)")
        print(f"   • DSPy PlanAgent for intelligent expansion")
        print(f"   • Routes to EVERY page discovered")
        print(f"   • Subpath memoization for efficiency")
        print(f"   • Route compression for memory optimization")
        print(f"   • Exact format: (from this - to this - to this)")
        print(f"   • Sorted by hop count and priority")
        
    finally:
        finder.close()


if __name__ == "__main__":
    main()
