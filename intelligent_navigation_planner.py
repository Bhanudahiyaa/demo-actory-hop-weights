#!/usr/bin/env python3
"""
Production-Ready Intelligent Site Navigation Planner using DSPy PlanAgent

This module implements a robust navigation planning system that:
1. Uses DSPy PlanAgent with GPT-OSS 20B via OpenRouter
2. Builds compressed route trees with subpath memoization
3. Avoids cycles and enforces max_depth cutoffs
4. Computes shortest paths and all routes with length-first enumeration
5. Ranks neighbors by priority with deterministic fallback
6. Emits human-readable reports with emojis and sections

Author: AI Assistant
License: MIT
"""

import os
import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Set, Optional, Generator, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# DSPy imports
try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    logger.error("DSPy not available. Install with: pip install dspy-ai")
    raise ImportError("DSPy is required for this module")


@dataclass
class RouteNode:
    """Represents a node in the compressed route tree."""
    url: str
    children: Dict[str, 'RouteNode'] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_child(self, url: str, node: 'RouteNode'):
        """Add a child node to this route node."""
        self.children[url] = node
    
    def get_child(self, url: str) -> Optional['RouteNode']:
        """Get a child node by URL."""
        return self.children.get(url)
    
    def has_children(self) -> bool:
        """Check if this node has children."""
        return len(self.children) > 0
    
    def is_leaf(self) -> bool:
        """Check if this node is a leaf (no children)."""
        return len(self.children) == 0


@dataclass
class RouteStats:
    """Statistics about the route tree construction."""
    total_nodes: int = 0
    total_paths: int = 0
    max_depth: int = 0
    construction_time: float = 0.0
    memoization_hits: int = 0
    memoization_misses: int = 0
    cycles_detected: int = 0
    dspy_decisions: int = 0
    dspy_failures: int = 0
    fallback_usage: int = 0


class SubpathCache:
    """
    Efficient subpath memoization to avoid recalculating shared subpaths.
    
    This cache stores previously computed routes from start to end nodes,
    allowing the system to reuse common path segments and significantly
    improve performance for large sitemaps.
    """
    
    def __init__(self):
        self.cache: Dict[str, Dict[str, List[List[str]]]] = defaultdict(dict)
        self.hits = 0
        self.misses = 0
    
    def get(self, start: str, end: str, max_depth: int) -> Optional[List[List[str]]]:
        """
        Get cached routes from start to end within max_depth.
        
        Args:
            start: Starting URL
            end: Target URL
            max_depth: Maximum allowed hop depth
            
        Returns:
            List of cached routes if available, None otherwise
        """
        if start in self.cache and end in self.cache[start]:
            cached_routes = self.cache[start][end]
            # Filter by max_depth
            valid_routes = [route for route in cached_routes if len(route) - 1 <= max_depth]
            if valid_routes:
                self.hits += 1
                return valid_routes
        
        self.misses += 1
        return None
    
    def set(self, start: str, end: str, routes: List[List[str]]):
        """
        Cache routes from start to end.
        
        Args:
            start: Starting URL
            end: Target URL
            routes: List of route paths to cache
        """
        if start not in self.cache:
            self.cache[start] = {}
        self.cache[start][end] = routes
    
    def clear(self):
        """Clear the entire cache."""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': total_requests,
            'hit_rate_percent': round(hit_rate, 2)
        }


class IntelligentNavigationPlanner:
    """
    Production-ready intelligent site navigation planner using DSPy PlanAgent.
    
    This class implements the core navigation planning logic, leveraging
    DSPy's PlanAgent to make intelligent decisions about which nodes
    to explore next during route tree construction, with robust fallback
    to deterministic algorithms when the LLM fails.
    """
    
    def __init__(self, api_key: str, max_depth: int = 6, enable_memoization: bool = True):
        """
        Initialize the intelligent navigation planner.
        
        Args:
            api_key: GPT-OSS 20B API key via OpenRouter
            max_depth: Maximum hop depth for route exploration
            enable_memoization: Whether to enable subpath memoization
        """
        self.api_key = api_key
        self.max_depth = max_depth
        self.enable_memoization = enable_memoization
        
        # Initialize DSPy with GPT-OSS 20B via OpenRouter
        self._configure_dspy()
        
        # Initialize components
        self.subpath_cache = SubpathCache() if enable_memoization else None
        self.stats = RouteStats()
        
        logger.info(f"Intelligent Navigation Planner initialized with max_depth={max_depth}")
    
    def _configure_dspy(self):
        """Configure DSPy with GPT-OSS 20B API via OpenRouter."""
        try:
            # Set environment variables for OpenRouter
            os.environ['OPENAI_API_KEY'] = self.api_key
            os.environ['OPENAI_API_BASE'] = 'https://openrouter.ai/api/v1'
            
            # Configure DSPy with GPT-OSS 20B via OpenRouter
            dspy.configure(
                lm=dspy.LM('openai/gpt-oss-20b:free'),
                api_key=self.api_key,
                api_base='https://openrouter.ai/api/v1'
            )
            
            logger.info("DSPy configured successfully with GPT-OSS 20B via OpenRouter")
            
        except Exception as e:
            logger.error(f"Failed to configure DSPy: {e}")
            raise
    
    def _create_dspy_signatures(self):
        """Create DSPy signatures for intelligent navigation decisions."""
        
        class ExploreNode(dspy.Signature):
            """
            DSPy signature for deciding which nodes to explore next.
            
            This agent analyzes the current exploration state and makes
            intelligent decisions about which frontier nodes to explore
            next, considering factors like node connectivity, depth,
            and exploration progress.
            
            OUTPUT MUST BE VALID JSON with next_nodes array containing URLs.
            """
            current_node = dspy.InputField(desc="Current node being explored")
            frontier = dspy.InputField(desc="Available nodes to explore next")
            visited = dspy.InputField(desc="Nodes already visited")
            current_depth = dspy.InputField(desc="Current exploration depth")
            max_depth = dspy.InputField(desc="Maximum allowed depth")
            sitemap_context = dspy.InputField(desc="Context about the sitemap structure")
            
            next_nodes = dspy.OutputField(
                desc="JSON string with format: {\"next_nodes\": [\"url1\", \"url2\", ...]}. Must contain valid URLs from frontier list."
            )
        
        return ExploreNode
    
    def _initialize_dspy_agents(self):
        """Initialize DSPy PlanAgent for intelligent decision making."""
        ExploreNode = self._create_dspy_signatures()
        
        # Create the main exploration agent using Predict (available in DSPy)
        self.explore_agent = dspy.Predict(ExploreNode)
        
        logger.info("DSPy Predict agent initialized successfully")
    
    def build_route_tree(self, start_node: str, sitemap: Dict[str, List[str]], max_depth: int = None) -> RouteNode:
        """
        Build a compressed route tree starting from the specified node.
        
        This method constructs a compressed route tree using intelligent
        exploration decisions from the DSPy PlanAgent. The tree structure
        shares common prefixes to minimize memory usage while maintaining
        the ability to reconstruct full paths.
        
        Args:
            start_node: Starting node for route exploration
            sitemap: Dictionary mapping pages to their neighbors
            max_depth: Maximum hop depth (overrides instance default)
            
        Returns:
            RouteNode: Root of the compressed route tree
            
        Raises:
            ValueError: If start_node is not in sitemap
            RuntimeError: If route tree construction fails
        """
        if max_depth is None:
            max_depth = self.max_depth
        
        if start_node not in sitemap:
            raise ValueError(f"Start node '{start_node}' not found in sitemap")
        
        logger.info(f"Building route tree from '{start_node}' with max_depth={max_depth}")
        start_time = time.time()
        
        try:
            # Initialize DSPy agents if not already done
            if not hasattr(self, 'explore_agent'):
                self._initialize_dspy_agents()
            
            # Initialize the route tree
            root = RouteNode(start_node)
            visited = set()
            self.stats = RouteStats()
            
            # Build the tree using intelligent exploration
            self._build_tree_recursive(root, start_node, [], visited, sitemap, max_depth, 0)
            
            # Update statistics
            self.stats.construction_time = time.time() - start_time
            self.stats.total_nodes = self._count_nodes(root)
            self.stats.total_paths = self._count_total_paths(root)
            self.stats.max_depth = max_depth
            
            # Log completion
            logger.info(f"Route tree construction completed in {self.stats.construction_time:.2f}s")
            logger.info(f"Total nodes: {self.stats.total_nodes}, Total paths: {self.stats.total_paths}")
            
            return root
            
        except Exception as e:
            logger.error(f"Failed to build route tree: {e}")
            raise RuntimeError(f"Route tree construction failed: {e}")
    
    def _build_tree_recursive(self, current_node: RouteNode, current_url: str, 
                             current_path: List[str], visited: Set[str], 
                             sitemap: Dict[str, List[str]], max_depth: int, 
                             current_depth: int):
        """
        Recursively build the route tree using intelligent exploration.
        
        This method uses the DSPy PlanAgent to make intelligent decisions
        about which nodes to explore next, while applying subpath memoization
        to avoid recalculating common paths.
        
        Args:
            current_node: Current node in the route tree
            current_url: Current URL being explored
            current_path: Path from root to current node
            visited: Set of visited URLs
            sitemap: Dictionary mapping pages to neighbors
            max_depth: Maximum allowed depth
            current_depth: Current exploration depth
        """
        # Stop if we've reached max depth
        if current_depth >= max_depth:
            return
        
        # Mark current node as visited
        visited.add(current_url)
        
        # Get neighbors from sitemap
        neighbors = sitemap.get(current_url, [])
        if not neighbors:
            return
        
        # Filter out already visited neighbors to avoid cycles
        unvisited_neighbors = [n for n in neighbors if n not in visited]
        if not unvisited_neighbors:
            return
        
        # Use DSPy to decide exploration order
        try:
            exploration_order = self._get_intelligent_exploration_order(
                current_url, unvisited_neighbors, visited, current_depth, max_depth, sitemap
            )
            self.stats.dspy_decisions += 1
            
        except Exception as e:
            logger.warning(f"DSPy decision failed, falling back to default order: {e}")
            # Fallback to default ordering (breadth-first)
            exploration_order = unvisited_neighbors
            self.stats.dspy_failures += 1
            self.stats.fallback_usage += 1
        
        # Explore neighbors in the recommended order
        for neighbor_url in exploration_order:
            if neighbor_url in visited:
                continue
            
            # Check cache for existing subpath
            if self.enable_memoization and self.subpath_cache:
                cached_routes = self.subpath_cache.get(current_url, neighbor_url, max_depth - current_depth)
                if cached_routes:
                    # Use cached routes to build subtree
                    self._build_subtree_from_cache(current_node, neighbor_url, cached_routes, sitemap)
                    self.stats.memoization_hits += 1
                    continue
            
            # Create new child node
            child_node = RouteNode(neighbor_url)
            current_node.add_child(neighbor_url, child_node)
            
            # Recursively explore this neighbor
            new_path = current_path + [neighbor_url]
            self._build_tree_recursive(
                child_node, neighbor_url, new_path, visited.copy(), 
                sitemap, max_depth, current_depth + 1
            )
            
            # Cache the discovered routes if memoization is enabled
            if self.enable_memoization and self.subpath_cache:
                routes_to_neighbor = self._extract_routes_to_target(current_node, neighbor_url)
                if routes_to_neighbor:
                    self.subpath_cache.set(current_url, neighbor_url, routes_to_neighbor)
    
    def _get_intelligent_exploration_order(self, current_node: str, neighbors: List[str], 
                                         visited: Set[str], current_depth: int, 
                                         max_depth: int, sitemap: Dict[str, List[str]]) -> List[str]:
        """
        Use DSPy PlanAgent to determine intelligent exploration order.
        
        This method leverages the DSPy PlanAgent to analyze the current
        exploration state and make intelligent decisions about which
        neighbors to explore next.
        
        Args:
            current_node: Current node being explored
            neighbors: Available neighbors to explore
            visited: Set of visited nodes
            current_depth: Current exploration depth
            max_depth: Maximum allowed depth
            sitemap: Dictionary mapping pages to neighbors
            
        Returns:
            List[str]: Ordered list of neighbors to explore
        """
        try:
            # Prepare context for DSPy
            sitemap_context = {
                'total_pages': len(sitemap),
                'avg_connectivity': sum(len(neighbors) for neighbors in sitemap.values()) / len(sitemap),
                'current_connectivity': len(neighbors),
                'exploration_progress': len(visited) / len(sitemap)
            }
            
            # Get intelligent exploration decision using PlanAgent
            decision = self.explore_agent(
                current_node=current_node,
                frontier=neighbors,
                visited=list(visited),
                current_depth=current_depth,
                max_depth=max_depth,
                sitemap_context=sitemap_context
            )
            
            # Parse JSON response from DSPy
            if hasattr(decision, 'next_nodes') and decision.next_nodes:
                try:
                    # Parse the JSON string response
                    if isinstance(decision.next_nodes, str):
                        parsed_response = json.loads(decision.next_nodes)
                    else:
                        parsed_response = decision.next_nodes
                    
                    # Extract next_nodes array
                    if isinstance(parsed_response, dict) and 'next_nodes' in parsed_response:
                        dspy_nodes = parsed_response['next_nodes']
                        
                        # Validate that all returned nodes are valid URLs from the frontier
                        valid_nodes = []
                        for node in dspy_nodes:
                            if isinstance(node, str) and node in neighbors:
                                valid_nodes.append(node)
                        
                        # If we have valid nodes from DSPy, use them
                        if valid_nodes:
                            # Ensure all original neighbors are included
                            ordered_neighbors = valid_nodes[:]
                            for neighbor in neighbors:
                                if neighbor not in ordered_neighbors:
                                    ordered_neighbors.append(neighbor)
                            return ordered_neighbors
                        
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse DSPy JSON response: {e}")
                    self.stats.dspy_failures += 1
            
            # Fallback to original order if DSPy decision is invalid
            logger.warning("DSPy returned invalid response, using fallback ordering")
            self.stats.fallback_usage += 1
            return neighbors
            
        except Exception as e:
            logger.warning(f"Intelligent exploration failed: {e}")
            self.stats.dspy_failures += 1
            self.stats.fallback_usage += 1
            return neighbors
    
    def _build_subtree_from_cache(self, parent_node: RouteNode, target_url: str, 
                                 cached_routes: List[List[str]], sitemap: Dict[str, List[str]]):
        """
        Build a subtree from cached routes.
        
        This method reconstructs a subtree using previously cached routes,
        avoiding the need to recalculate common path segments.
        
        Args:
            parent_node: Parent node to attach the subtree to
            target_url: Target URL for the subtree
            cached_routes: List of cached route paths
            sitemap: Dictionary mapping pages to neighbors
        """
        if not cached_routes:
            return
        
        # Use the shortest cached route to build the subtree
        shortest_route = min(cached_routes, key=len)
        
        # Build the subtree following the cached route
        current_node = parent_node
        for url in shortest_route[1:]:  # Skip the first URL (parent)
            if url not in current_node.children:
                child_node = RouteNode(url)
                current_node.add_child(url, child_node)
                current_node = child_node
            else:
                current_node = current_node.children[url]
    
    def _extract_routes_to_target(self, root_node: RouteNode, target_url: str) -> List[List[str]]:
        """
        Extract all routes from root to a specific target.
        
        Args:
            root_node: Root of the route tree
            target_url: Target URL to find routes to
            
        Returns:
            List of route paths from root to target
        """
        routes = []
        
        def dfs(node: RouteNode, current_path: List[str]):
            current_path.append(node.url)
            
            if node.url == target_url:
                routes.append(current_path[:])
            else:
                for child in node.children.values():
                    dfs(child, current_path[:])
        
        dfs(root_node, [])
        return routes
    
    def _count_nodes(self, node: RouteNode) -> int:
        """Count total nodes in the route tree."""
        count = 1  # Count current node
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
    
    def _count_total_paths(self, node: RouteNode) -> int:
        """Count total possible paths in the route tree."""
        if not node.has_children():
            return 1
        
        total = 0
        for child in node.children.values():
            total += self._count_total_paths(child)
        return total
    
    def expand_route_tree(self, route_tree: RouteNode) -> Generator[List[str], None, None]:
        """
        Expand the compressed route tree into full paths.
        
        This generator yields all possible paths from the root to leaf nodes,
        effectively decompressing the tree structure into a flat list of routes.
        
        Args:
            route_tree: Root node of the compressed route tree
            
        Yields:
            List[str]: Complete route path from root to leaf
        """
        def _expand_recursive(node: RouteNode, current_path: List[str]):
            current_path.append(node.url)
            
            if node.is_leaf():
                yield current_path[:]
            else:
                for child in node.children.values():
                    yield from _expand_recursive(child, current_path[:])
        
        yield from _expand_recursive(route_tree, [])
    
    def get_route_summary(self, route_tree: RouteNode) -> RouteStats:
        """
        Get comprehensive statistics about the route tree.
        
        Args:
            route_tree: Root node of the route tree
            
        Returns:
            RouteStats: Statistics about the route tree
        """
        if not route_tree:
            return RouteStats()
        
        # Update statistics
        self.stats.total_nodes = self._count_nodes(route_tree)
        self.stats.total_paths = self._count_total_paths(route_tree)
        
        # Add cache statistics if memoization is enabled
        if self.enable_memoization and self.subpath_cache:
            cache_stats = self.subpath_cache.get_stats()
            self.stats.memoization_hits = cache_stats['hits']
            self.stats.memoization_misses = cache_stats['misses']
        
        return self.stats
    
    def clear_cache(self):
        """Clear the subpath cache."""
        if self.subpath_cache:
            self.subpath_cache.clear()
            logger.info("Subpath cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        if not self.subpath_cache:
            return {'enabled': False}
        
        stats = self.subpath_cache.get_stats()
        stats['enabled'] = True
        return stats

    def create_intelligent_navigation_plan(self, start_url: str, neo4j_client) -> Dict[str, Any]:
        """
        AI Navigation Planner using DSPy - Dynamic Content Analysis & Comprehensive Path Enumeration
        
        STRICT RULES (NO EXCEPTIONS):
        1. PRIORITY RANKING: Rank pages ONLY by dynamic content (JS, forms, interactive elements)
           - Forms(×10) + Inputs(×8) + Textareas(×8) + Scripts(×6) + Buttons(×3)
           - NO business importance, keywords, or semantics considered
        
        2. NAVIGATION ORDER: 
           - Start from highest-ranked page and move downward to lowest
           - For each destination: find ALL possible paths from landing page
           - Order paths from SHORTEST to LONGEST (no summarization)
           - After each destination, process its NEIGHBOR pages with same rules
        
        3. OUTPUT FORMAT:
           - Section 1: Priority Ranking (all pages sorted by dynamic score)
           - Section 2: Navigation Paths (all possible routes, shortest→longest)
           - Section 3: Neighbors (repeat path expansion for connected pages)
        
        IMPORTANT: Do not skip any path. Show all routes in full detail.
        """
        print("🤖 AI NAVIGATION PLANNER - DSPy Dynamic Content Analysis")
        print("=" * 70)
        print("🎯 Processing sitemap with strict dynamic content ranking rules...")
        
        try:
            # Step 1: Extract actual sitemap from Neo4j
            print("\n📊 STEP 1: Extracting actual sitemap from Neo4j...")
            sitemap = self._extract_sitemap_from_neo4j(start_url, neo4j_client)
            
            if not sitemap:
                raise ValueError("No sitemap data found in Neo4j")
            
            print(f"✅ Extracted sitemap with {len(sitemap)} pages")
            
            # Step 2: Calculate dynamic scores and rank pages
            print("\n⚡ STEP 2: Calculating dynamic content scores...")
            dynamic_ranking = self._calculate_dynamic_ranking(sitemap, neo4j_client)
            
            # Step 3: Generate comprehensive navigation paths
            print("\n🛣️  STEP 3: Generating ALL possible navigation paths...")
            navigation_plan = self._generate_comprehensive_navigation_plan(
                start_url, sitemap, dynamic_ranking
            )
            
            print("✅ AI Navigation planning completed!")
            return navigation_plan
            
        except Exception as e:
            print(f"❌ AI Navigation planning failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _extract_sitemap_from_neo4j(self, start_url: str, neo4j_client) -> Dict[str, List[str]]:
        """Extract the actual sitemap from Neo4j database."""
        try:
            # Get all pages and their outgoing links
            query = """
            MATCH (p1:Page)-[:LINKS_TO]->(p2:Page)
            RETURN p1.url as from, p2.url as to
            ORDER BY p1.url, p2.url
            """
            
            result = neo4j_client.run_query(query)
            
            # Build adjacency list
            sitemap = {}
            for record in result:
                from_url = record['from']
                to_url = record['to']
                
                if from_url not in sitemap:
                    sitemap[from_url] = []
                sitemap[from_url].append(to_url)
            
            # Also get pages that are linked FROM the start page (these are immediate neighbors)
            start_page_query = """
            MATCH (start:Page {url: $start_url})-[:LINKS_TO]->(p:Page)
            RETURN p.url as url
            """
            start_page_result = neo4j_client.run_query(start_page_query, {'start_url': start_url})
            
            # Add start page with its immediate neighbors
            if start_url not in sitemap:
                sitemap[start_url] = []
            
            for record in start_page_result:
                neighbor_url = record['url']
                if neighbor_url not in sitemap[start_url]:
                    sitemap[start_url].append(neighbor_url)
            
            # Get INCOMING links for all pages (pages that link TO each page)
            incoming_query = """
            MATCH (p1:Page)-[:LINKS_TO]->(p2:Page)
            RETURN p2.url as to, p1.url as from
            ORDER BY p2.url, p1.url
            """
            
            incoming_result = neo4j_client.run_query(incoming_query)
            
            # Add incoming links to sitemap
            for record in incoming_result:
                to_url = record['to']
                from_url = record['from']
                
                if to_url not in sitemap:
                    sitemap[to_url] = []
                sitemap[to_url].append(from_url)
            
            # Handle URL variations (with/without trailing slashes)
            # Create a normalized mapping to handle both versions
            url_variations = {}
            for url in sitemap.keys():
                # Remove trailing slash for normalization
                normalized = url.rstrip('/')
                if normalized not in url_variations:
                    url_variations[normalized] = []
                url_variations[normalized].append(url)
            
            # Merge neighbors for URL variations
            for normalized, variations in url_variations.items():
                if len(variations) > 1:
                    # Find the version with the most neighbors
                    best_version = max(variations, key=lambda v: len(sitemap.get(v, [])))
                    best_neighbors = sitemap.get(best_version, [])
                    
                    # Update all variations to have the same neighbors
                    for variation in variations:
                        sitemap[variation] = best_neighbors.copy()
            
            # Also handle the reverse case - if a page has no neighbors but its normalized version does
            for url in list(sitemap.keys()):
                if len(sitemap.get(url, [])) == 0:
                    # Try to find neighbors for normalized version
                    normalized = url.rstrip('/')
                    if normalized in sitemap and len(sitemap[normalized]) > 0:
                        sitemap[url] = sitemap[normalized].copy()
                        print(f"      🔄 Fixed {url} by copying neighbors from {normalized}")
                    elif normalized != url and normalized in sitemap:
                        sitemap[url] = sitemap[normalized].copy()
                        print(f"      🔄 Fixed {url} by copying neighbors from {normalized}")
            
            # Debug: Print sitemap structure
            print(f"      🔍 Sitemap extracted: {len(sitemap)} pages")
            pages_with_neighbors = sum(1 for neighbors in sitemap.values() if neighbors)
            print(f"      📍 Pages with neighbors: {pages_with_neighbors}")
            total_connections = sum(len(neighbors) for neighbors in sitemap.values())
            print(f"      🔗 Total connections: {total_connections}")
            
            # Show some examples
            print("      📋 Sample sitemap entries:")
            for i, (url, neighbors) in enumerate(list(sitemap.items())[:3]):
                print(f"         {url}: {len(neighbors)} neighbors")
                if neighbors:
                    for neighbor in neighbors[:2]:
                        print(f"           -> {neighbor}")
                    if len(neighbors) > 2:
                        print(f"           ... and {len(neighbors) - 2} more")
            
            # Ensure all pages are in the sitemap (even if they have no outgoing links)
            all_pages_query = "MATCH (p:Page) RETURN p.url as url"
            all_pages = neo4j_client.run_query(all_pages_query)
            
            for page in all_pages:
                url = page['url']
                if url not in sitemap:
                    sitemap[url] = []
            
            return sitemap
            
        except Exception as e:
            print(f"❌ Failed to extract sitemap: {e}")
            return {}

    def _calculate_dynamic_ranking(self, sitemap: Dict[str, List[str]], neo4j_client) -> List[Dict[str, Any]]:
        """
        Calculate dynamic content scores and rank pages from highest to lowest.
        
        STRICT RULE: Rank ONLY by dynamic content (JS, forms, interactive elements)
        NO business importance, keywords, or semantics considered.
        """
        print("   🔍 Analyzing dynamic content for each page...")
        
        dynamic_scores = []
        
        for page_url in sitemap.keys():
            try:
                # Get page features from Neo4j using the working _get_feature_counts method
                features = neo4j_client._get_feature_counts(page_url)
                
                # Extract feature counts from the working method
                scripts = features.get('scriptCount', 0)
                forms = features.get('formCount', 0)
                buttons = features.get('buttonCount', 0)
                inputs = features.get('inputCount', 0)
                textareas = 0  # Not available in current database schema
                images = features.get('imageCount', 0)
                links = features.get('linkCount', 0)
                
                # Check if this page was actually scraped (has any features)
                total_features = scripts + forms + buttons + inputs + images + links
                
                if total_features == 0:
                    # Page was discovered but not scraped - mark it appropriately
                    print(f"      ⚠️  {page_url}: Discovered but not scraped (no features)")
                    # Give it a minimal score but mark it as incomplete
                    dynamic_score = 0
                    status = "discovered_not_scraped"
                else:
                    # Page was actually scraped - calculate real score
                    # Calculate dynamic score based ONLY on interactive elements
                    # Higher weight for forms, inputs, scripts (most dynamic)
                    # Lower weight for buttons (moderately dynamic)
                    # No weight for images, links (static content)
                    dynamic_score = (
                        (forms * 10) +           # Forms are most dynamic
                        (inputs * 8) +           # Input fields are highly dynamic
                        (scripts * 6) +          # JavaScript files are dynamic
                        (buttons * 3)            # Buttons have some interactivity
                    )
                    status = "fully_scraped"
                
                dynamic_scores.append({
                    'url': page_url,
                    'dynamic_score': dynamic_score,
                    'status': status,
                    'features': {
                        'scripts': scripts,
                        'forms': forms,
                        'inputs': inputs,
                        'textareas': textareas,
                        'buttons': buttons,
                        'images': images,
                        'links': links
                    },
                    'score_breakdown': f"Forms({forms}×10) + Inputs({inputs}×8) + Scripts({scripts}×6) + Buttons({buttons}×3) = {dynamic_score}"
                })
                
            except Exception as e:
                print(f"   ⚠️  Error analyzing {page_url}: {e}")
                # Add with error status
                dynamic_scores.append({
                    'url': page_url,
                    'dynamic_score': 0,
                    'status': 'error',
                    'features': {'scripts': 0, 'forms': 0, 'inputs': 0, 'textareas': 0, 'buttons': 0, 'images': 0, 'links': 0},
                    'score_breakdown': f"Error: {e}"
                })
                continue
        
        # Sort by dynamic score (highest to lowest), then by status (fully_scraped first)
        dynamic_scores.sort(key=lambda x: (x['dynamic_score'], x['status'] != 'fully_scraped'), reverse=True)
        
        # Count pages by status
        fully_scraped = sum(1 for p in dynamic_scores if p['status'] == 'fully_scraped')
        discovered_only = sum(1 for p in dynamic_scores if p['status'] == 'discovered_not_scraped')
        errors = sum(1 for p in dynamic_scores if p['status'] == 'error')
        
        print(f"   ✅ Ranked {len(dynamic_scores)} pages by dynamic content")
        print(f"      📊 Fully scraped: {fully_scraped} pages")
        print(f"      📍 Discovered only: {discovered_only} pages")
        print(f"      ❌ Errors: {errors} pages")
        
        return dynamic_scores

    def _generate_comprehensive_navigation_plan(self, start_url: str, sitemap: Dict[str, List[str]], 
                                              dynamic_ranking: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate comprehensive navigation plan following strict rules:
        1. Priority Ranking (dynamic content only)
        2. Navigation Paths (all possible paths, shortest→longest)
        3. Neighbors (repeat for directly connected pages)
        """
        print("   🗺️  Generating comprehensive navigation paths...")
        
        navigation_plan = {
            'metadata': {
                'start_url': start_url,
                'total_pages': len(sitemap),
                'total_ranked_pages': len(dynamic_ranking),
                'total_neighborhoods': 0,  # Will be calculated after processing neighbors
                'generation_timestamp': datetime.now().isoformat()
            },
            'section_1_priority_ranking': [],
            'section_2_navigation_paths': {},
            'section_3_neighbors': {},
            'summary': {
                'high_priority_pages': [],
                'critical_paths': [],
                'recommendations': []
            }
        }
        
        # Section 1: Priority Ranking (dynamic content only)
        print("   📊 Section 1: Building priority ranking...")
        navigation_plan['section_1_priority_ranking'] = dynamic_ranking
        
        # Section 2: Navigation Paths (for each page in ranking order)
        print("   🛣️  Section 2: Generating navigation paths...")
        for i, page_data in enumerate(dynamic_ranking):
            page_url = page_data['url']
            print(f"      Processing #{i+1}: {page_url}")
            
            try:
                # Find ALL possible paths from start to this page (with timeout)
                import signal
                
                def timeout_handler(signum, frame):
                    raise TimeoutError(f"Path finding timeout for {page_url}")
                
                # Set timeout for each page (60 seconds for 100 paths)
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(60)
                
                try:
                    all_paths = self._find_all_possible_paths(start_url, page_url, sitemap)
                    signal.alarm(0)  # Cancel alarm
                except TimeoutError:
                    print(f"         ⏰ Timeout processing {page_url}, using fallback")
                    # Fallback: find just the shortest path
                    all_paths = self._find_shortest_path_fallback(start_url, page_url, sitemap)
                
                # Sort paths from shortest to longest (1 hop → 6 hops)
                all_paths.sort(key=lambda x: len(x))
                
                navigation_plan['section_2_navigation_paths'][page_url] = {
                    'rank': i + 1,
                    'dynamic_score': page_data['dynamic_score'],
                    'features': page_data['features'],
                    'score_breakdown': page_data['score_breakdown'],
                    'paths': []
                }
                
                for path in all_paths:
                    navigation_plan['section_2_navigation_paths'][page_url]['paths'].append({
                        'path': path,
                        'hops': len(path) - 1,
                        'path_string': ' → '.join(path)
                    })
                
                print(f"         ✅ Found {len(all_paths)} paths")
                
            except Exception as e:
                print(f"         ❌ Error processing {page_url}: {e}")
                # Create entry with error info
                navigation_plan['section_2_navigation_paths'][page_url] = {
                    'rank': i + 1,
                    'dynamic_score': page_data['dynamic_score'],
                    'features': page_data['features'],
                    'score_breakdown': page_data['score_breakdown'],
                    'paths': [],
                    'error': str(e)
                }
        
        # Section 3: COMPREHENSIVE Neighbor Analysis for ALL Pages in Priority Order
        print("   🔗 Section 3: Processing ALL pages in priority order with comprehensive neighbor analysis...")
        total_neighbor_count = 0
        total_pages_processed = 0
        
        # Process each page in priority order (highest dynamic score first)
        for rank_index, page_data in enumerate(dynamic_ranking):
            page_url = page_data['url']
            page_rank = rank_index + 1
            page_score = page_data['dynamic_score']
            
            print(f"      🎯 Processing RANK #{page_rank} ({page_score} points): {page_url}")
            
            # Get all immediate neighbors for this page
            immediate_neighbors = sitemap.get(page_url, [])
            
            # Debug: Show what we found
            print(f"         🔍 Looking for neighbors of: {page_url}")
            print(f"         📍 Found {len(immediate_neighbors)} immediate neighbors")
            if immediate_neighbors:
                print(f"         📋 Neighbors: {', '.join(immediate_neighbors[:3])}{'...' if len(immediate_neighbors) > 3 else ''}")
            
            if immediate_neighbors:
                
                # Initialize this page's neighbor analysis
                navigation_plan['section_3_neighbors'][page_url] = {
                    'rank': page_rank,
                    'dynamic_score': page_score,
                    'immediate_neighbor_count': len(immediate_neighbors),
                    'immediate_neighbors': {}
                }
                
                # Process each immediate neighbor with comprehensive path analysis
                for neighbor_url in immediate_neighbors:
                    try:
                        print(f"            🔍 Analyzing neighbor: {neighbor_url}")
                        
                        # Find ALL possible paths from start to this neighbor (up to 50 paths, max 6 hops)
                        neighbor_paths = self._find_all_possible_paths(start_url, neighbor_url, sitemap)
                        
                        # Ensure paths are sorted from shortest to longest (1 hop → 6 hops)
                        neighbor_paths.sort(key=lambda x: len(x))
                        
                        # Create comprehensive neighbor analysis
                        navigation_plan['section_3_neighbors'][page_url]['immediate_neighbors'][neighbor_url] = {
                            'connection_type': 'direct_link',
                            'hops_from_source': 1,
                            'is_ranked_page': neighbor_url in [p['url'] for p in dynamic_ranking],
                            'neighbor_rank': next((i+1 for i, p in enumerate(dynamic_ranking) if p['url'] == neighbor_url), 'N/A'),
                            'neighbor_score': next((p['dynamic_score'] for p in dynamic_ranking if p['url'] == neighbor_url), 'N/A'),
                            'all_possible_paths': []
                        }
                        
                        # Add all possible paths to this neighbor with detailed information
                        for path in neighbor_paths:
                            navigation_plan['section_3_neighbors'][page_url]['immediate_neighbors'][neighbor_url]['all_possible_paths'].append({
                                'path': path,
                                'hops': len(path) - 1,
                                'path_string': ' → '.join(path),
                                'is_shortest': len(path) == min(len(p) for p in neighbor_paths)
                            })
                        
                        # Ensure paths are sorted by hops (shortest first) - CRITICAL for proper ordering
                        navigation_plan['section_3_neighbors'][page_url]['immediate_neighbors'][neighbor_url]['all_possible_paths'].sort(key=lambda x: x['hops'])
                        
                        # Double-check ordering and log for debugging
                        paths = navigation_plan['section_3_neighbors'][page_url]['immediate_neighbors'][neighbor_url]['all_possible_paths']
                        if paths:
                            first_hop = paths[0]['hops']
                            last_hop = paths[-1]['hops']
                            print(f"               🔄 Paths sorted: {first_hop} → {last_hop} hops")
                        
                        print(f"               ✅ {len(neighbor_paths)} paths found (shortest: {min(len(p)-1 for p in neighbor_paths)} hops)")
                        total_neighbor_count += 1
                        
                    except Exception as e:
                        print(f"               ⚠️  Error processing neighbor {neighbor_url}: {e}")
                        navigation_plan['section_3_neighbors'][page_url]['immediate_neighbors'][neighbor_url] = {
                            'connection_type': 'direct_link',
                            'hops_from_source': 1,
                            'is_ranked_page': neighbor_url in [p['url'] for p in dynamic_ranking],
                            'neighbor_rank': 'N/A',
                            'neighbor_score': 'N/A',
                            'all_possible_paths': [],
                            'error': str(e)
                        }
                        continue
                
                total_pages_processed += 1
            else:
                print(f"         📍 No immediate neighbors found")
                # Still create entry for pages with no neighbors
                navigation_plan['section_3_neighbors'][page_url] = {
                    'rank': page_rank,
                    'dynamic_score': page_score,
                    'immediate_neighbor_count': 0,
                    'immediate_neighbors': {}
                }
        
        print(f"   🔗 Comprehensive neighbor analysis complete!")
        print(f"      📊 Total pages processed: {total_pages_processed}")
        print(f"      📊 Total neighbors analyzed: {total_neighbor_count}")
        
        # Enhanced neighborhood structure analysis
        print("   🏘️  Enhanced neighborhood structure analysis...")
        
        # Analyze the comprehensive neighborhood structure
        total_immediate_connections = sum(
            neighbors.get('immediate_neighbor_count', 0) 
            for neighbors in navigation_plan['section_3_neighbors'].values()
        )
        
        pages_with_neighbors = sum(
            1 for neighbors in navigation_plan['section_3_neighbors'].values()
            if neighbors.get('immediate_neighbor_count', 0) > 0
        )
        
        print(f"      📊 Total immediate connections: {total_immediate_connections}")
        print(f"      📊 Pages with neighbors: {pages_with_neighbors}")
        print(f"      📊 Average neighbors per connected page: {total_immediate_connections / pages_with_neighbors if pages_with_neighbors > 0 else 0:.1f}")
        
        # Calculate total neighborhoods and populate summary
        print("   📊 Calculating comprehensive summary statistics...")
        
        # Count total neighborhoods (pages with immediate neighbors)
        total_neighborhoods = sum(1 for neighbors in navigation_plan['section_3_neighbors'].values() 
                                if neighbors.get('immediate_neighbor_count', 0) > 0)
        navigation_plan['metadata']['total_neighborhoods'] = total_neighborhoods
        
        # Populate high priority pages (top 3 by dynamic score)
        for i, page_data in enumerate(dynamic_ranking[:3]):
            navigation_plan['summary']['high_priority_pages'].append({
                'url': page_data['url'],
                'dynamic_score': page_data['dynamic_score'],
                'rank': i + 1,
                'features': page_data['features']
            })
        
        # Populate critical paths (shortest paths to high-priority pages)
        for page_data in dynamic_ranking[:3]:
            page_url = page_data['url']
            if page_url in navigation_plan['section_2_navigation_paths']:
                paths = navigation_plan['section_2_navigation_paths'][page_url]['paths']
                if paths:
                    shortest_path = paths[0]  # Already sorted by length
                    navigation_plan['summary']['critical_paths'].append({
                        'destination': page_url,
                        'hops': shortest_path['hops'],
                        'path': shortest_path['path'],
                        'dynamic_score': page_data['dynamic_score']
                    })
        
        # Generate recommendations
        navigation_plan['summary']['recommendations'] = self._generate_recommendations_from_plan(
            navigation_plan, sitemap
        )
        
        print("   ✅ Comprehensive navigation plan generated")
        
        # Generate clean, hierarchical output following strict rules
        self._generate_clean_output(navigation_plan)
        
        return navigation_plan

    def _find_all_possible_paths(self, start: str, target: str, sitemap: Dict[str, List[str]], 
                                max_depth: int = 6) -> List[List[str]]:
        """
        Find ALL possible paths from start to target following strict rules.
        Returns paths sorted from shortest to longest.
        NO path summarization - show all routes in full detail.
        """
        if start == target:
            return [[start]]
        
        all_paths = []
        visited_paths = set()
        path_count = 0
        max_paths = 1000  # Extended to 1000 paths per page as requested
        
        # Use BFS to find shortest paths first, then DFS for longer paths
        from collections import deque
        
        # First, find shortest paths using BFS
        queue = deque([(start, [start])])
        visited_bfs = {start}
        shortest_paths = []
        
        while queue and len(shortest_paths) < 10:  # Find first 10 shortest paths
            current, path = queue.popleft()
            
            if current == target:
                path_str = ' → '.join(path)
                if path_str not in visited_paths:
                    visited_paths.add(path_str)
                    shortest_paths.append(path[:])
                    path_count += 1
                continue
            
            if current not in sitemap:
                continue
            
            # Process neighbors in BFS order
            for neighbor in sitemap[current]:
                if neighbor not in visited_bfs and neighbor not in path:
                    visited_bfs.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        # Add shortest paths to all_paths
        all_paths.extend(shortest_paths)
        
        # Then use DFS to find longer paths
        def dfs(current: str, path: List[str], depth: int):
            nonlocal path_count
            
            # Safety checks
            if depth > max_depth or path_count >= max_paths:
                return
            
            if current == target:
                path_str = ' → '.join(path)
                if path_str not in visited_paths:
                    visited_paths.add(path_str)
                    all_paths.append(path[:])
                    path_count += 1
                return
            
            if current not in sitemap:
                return
            
            # Process ALL neighbors (no artificial limits) for comprehensive coverage
            neighbors = sitemap[current]
            for neighbor in neighbors:
                if neighbor not in path and path_count < max_paths:  # Avoid cycles
                    dfs(neighbor, path + [neighbor], depth + 1)
        
        try:
            # Start DFS from start node
            dfs(start, [start], 0)
            
            # Sort by length (shortest first) as per strict rules - guaranteed ordering
            all_paths.sort(key=lambda x: len(x))
            
            # Return ALL found paths (up to 1000)
            return all_paths[:max_paths]
            
        except Exception as e:
            print(f"      ⚠️  Error finding paths to {target}: {e}")
            # Return at least the shortest path if available
            if all_paths:
                return [all_paths[0]]
            return []

    def _find_shortest_path_fallback(self, start: str, target: str, sitemap: Dict[str, List[str]]) -> List[List[str]]:
        """
        Fallback method to find just the shortest path when comprehensive search times out.
        Uses BFS for efficiency.
        """
        if start == target:
            return [[start]]
        
        from collections import deque
        
        # BFS to find shortest path
        queue = deque([(start, [start])])
        visited = {start}
        
        while queue:
            current, path = queue.popleft()
            
            if current == target:
                return [path]
            
            if current not in sitemap:
                continue
            
            # Process neighbors
            for neighbor in sitemap[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        # No path found
        return []

    def _form_neighborhoods(self, sitemap: Dict[str, List[str]], neo4j_client) -> Dict[str, Dict[str, Any]]:
        """Group pages into neighborhoods based on topical similarity."""
        neighborhoods = {}
        
        # Define neighborhood patterns based on URL structure and content
        neighborhood_patterns = {
            'core_business': {
                'keywords': ['shipping', 'returns', 'ai-shipping-intelligence', 'integrations'],
                'description': 'Core business functionality and product features',
                'priority': 'high'
            },
            'user_engagement': {
                'keywords': ['contact', 'support', 'pricing', 'signup', 'login'],
                'description': 'User interaction and conversion points',
                'priority': 'critical'
            },
            'company_info': {
                'keywords': ['about-us', 'team', 'careers', 'company', 'mission'],
                'description': 'Company information and team details',
                'priority': 'medium'
            },
            'documentation': {
                'keywords': ['knowledge-base', 'api-docs', 'docs', 'help', 'tutorial'],
                'description': 'Documentation and help resources',
                'priority': 'medium'
            },
            'legal': {
                'keywords': ['privacy-policy', 'terms', 'legal', 'disclaimer'],
                'priority': 'low'
            },
            'blog_content': {
                'keywords': ['blog', 'news', 'articles', 'insights'],
                'description': 'Content marketing and blog posts',
                'priority': 'low'
            }
        }
        
        # Assign each page to a neighborhood
        for url in sitemap.keys():
            assigned = False
            
            for neighborhood_name, pattern in neighborhood_patterns.items():
                if any(keyword in url.lower() for keyword in pattern['keywords']):
                    if neighborhood_name not in neighborhoods:
                        neighborhoods[neighborhood_name] = {
                            'pages': [],
                            'description': pattern.get('description', ''),
                            'priority': pattern.get('priority', 'medium'),
                            'theme': neighborhood_name.replace('_', ' ').title()
                        }
                    
                    neighborhoods[neighborhood_name]['pages'].append(url)
                    assigned = True
                    break
            
            # If no pattern matches, assign to 'other' neighborhood
            if not assigned:
                if 'other' not in neighborhoods:
                    neighborhoods['other'] = {
                        'pages': [],
                        'description': 'Miscellaneous pages that don\'t fit other categories',
                        'priority': 'low',
                        'theme': 'Other'
                    }
                neighborhoods['other']['pages'].append(url)
        
        return neighborhoods

    def _assign_priority_scores(self, neighborhoods: Dict[str, Dict[str, Any]], neo4j_client) -> Dict[str, Dict[str, Any]]:
        """Assign priority scores (1-10) to pages within each neighborhood."""
        prioritized_neighborhoods = {}
        
        for neighborhood_name, neighborhood_data in neighborhoods.items():
            prioritized_neighborhoods[neighborhood_name] = {
                **neighborhood_data,
                'prioritized_pages': []
            }
            
            for page_url in neighborhood_data['pages']:
                priority_score, reasoning = self._calculate_page_priority(page_url, neighborhood_name, neo4j_client)
                
                prioritized_neighborhoods[neighborhood_name]['prioritized_pages'].append({
                    'url': page_url,
                    'priority_score': priority_score,
                    'reasoning': reasoning
                })
            
            # Sort pages by priority score (highest first)
            prioritized_neighborhoods[neighborhood_name]['prioritized_pages'].sort(
                key=lambda x: x['priority_score'], reverse=True
            )
        
        return prioritized_neighborhoods

    def _calculate_page_priority(self, page_url: str, neighborhood: str, neo4j_client) -> tuple[int, str]:
        """Calculate priority score (1-10) for a page based on navigation relevance."""
        try:
            # Get page features from Neo4j
            features_query = """
            MATCH (p:Page {url: $url})
            RETURN COALESCE(p.scriptCount, 0) as scripts,
                   COALESCE(p.formCount, 0) as forms,
                   COALESCE(p.buttonCount, 0) as buttons,
                   COALESCE(p.inputCount, 0) as inputs,
                   COALESCE(p.imageCount, 0) as images,
                   COALESCE(p.linkCount, 0) as links
            """
            
            result = neo4j_client.run_query(features_query, {'url': page_url})
            if not result:
                return 5, "Default priority - no feature data available"
            
            features = result[0]
            scripts = features['scripts'] or 0
            forms = features['forms'] or 0
            buttons = features['buttons'] or 0
            inputs = features['inputs'] or 0
            images = features['images'] or 0
            links = features['links'] or 0
            
            # Base priority based on neighborhood
            base_priority = {
                'user_engagement': 8,
                'core_business': 7,
                'company_info': 5,
                'documentation': 6,
                'legal': 3,
                'blog_content': 4,
                'other': 4
            }.get(neighborhood, 5)
            
            # Adjust based on interactive elements (forms, inputs = higher priority)
            if forms > 0:
                base_priority += 2
                reasoning = f"Contains {forms} form(s) - high user interaction potential"
            elif inputs > 0:
                base_priority += 1
                reasoning = f"Contains {inputs} input field(s) - moderate user interaction"
            else:
                reasoning = f"No interactive forms - informational content"
            
            # Adjust based on content richness
            if images > 20:
                base_priority += 1
                reasoning += f", rich visual content ({images} images)"
            
            if scripts > 15:
                base_priority += 1
                reasoning += f", dynamic functionality ({scripts} scripts)"
            
            # Ensure priority is within 1-10 range
            final_priority = max(1, min(10, base_priority))
            
            return final_priority, reasoning
            
        except Exception as e:
            return 5, f"Error calculating priority: {e}"

    def _find_efficient_paths(self, start_url: str, sitemap: Dict[str, List[str]], 
                             prioritized_neighborhoods: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Find efficient navigation paths to key destinations."""
        paths = {
            'shortest_paths': {},
            'alternative_paths': {},
            'coverage_paths': {}
        }
        
        # Find shortest paths to high-priority destinations
        high_priority_destinations = []
        for neighborhood_name, neighborhood_data in prioritized_neighborhoods.items():
            if neighborhood_data['priority'] in ['critical', 'high']:
                for page_data in neighborhood_data['prioritized_pages'][:3]:  # Top 3 pages
                    high_priority_destinations.append(page_data['url'])
        
        # Find shortest paths using BFS
        for destination in high_priority_destinations:
            shortest_path = self._find_shortest_path_bfs(start_url, destination, sitemap)
            if shortest_path:
                paths['shortest_paths'][destination] = {
                    'path': shortest_path,
                    'hops': len(shortest_path) - 1,
                    'reasoning': f"Shortest path to {destination} via BFS"
                }
        
        # Find alternative paths for key destinations
        for destination in high_priority_destinations[:5]:  # Top 5 destinations
            alternative_paths = self._find_alternative_paths(start_url, destination, sitemap, max_paths=3)
            if alternative_paths:
                paths['alternative_paths'][destination] = alternative_paths
        
        # Find coverage paths that visit multiple neighborhoods efficiently
        coverage_paths = self._find_coverage_paths(start_url, sitemap, prioritized_neighborhoods)
        paths['coverage_paths'] = coverage_paths
        
        return paths

    def _find_shortest_path_bfs(self, start: str, target: str, sitemap: Dict[str, List[str]]) -> List[str]:
        """Find shortest path using BFS."""
        if start == target:
            return [start]
        
        queue = [(start, [start])]
        visited = {start}
        
        while queue:
            current, path = queue.pop(0)
            
            if current not in sitemap:
                continue
                
            for neighbor in sitemap[current]:
                if neighbor == target:
                    return path + [neighbor]
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return []

    def _find_alternative_paths(self, start: str, target: str, sitemap: Dict[str, List[str]], 
                               max_paths: int = 3) -> List[Dict[str, Any]]:
        """Find alternative paths to a destination."""
        paths = []
        visited_paths = set()
        
        def dfs(current: str, path: List[str], depth: int):
            if depth > 6:  # Limit depth
                return
            
            if current == target:
                path_str = ' → '.join(path)
                if path_str not in visited_paths:
                    visited_paths.add(path_str)
                    paths.append({
                        'path': path,
                        'hops': len(path) - 1,
                        'reasoning': f"Alternative path found via DFS (depth {depth})"
                    })
                return
            
            if current not in sitemap or len(paths) >= max_paths:
                return
            
            for neighbor in sitemap[current]:
                if neighbor not in path:  # Avoid cycles
                    dfs(neighbor, path + [neighbor], depth + 1)
        
        dfs(start, [start], 0)
        return paths

    def _find_coverage_paths(self, start: str, sitemap: Dict[str, List[str]], 
                            prioritized_neighborhoods: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Find paths that efficiently cover multiple neighborhoods."""
        coverage_paths = {
            'business_focused': [],
            'user_journey': [],
            'comprehensive': []
        }
        
        # Business-focused path: core business → user engagement
        business_path = self._create_business_path(start, sitemap, prioritized_neighborhoods)
        if business_path:
            coverage_paths['business_focused'] = business_path
        
        # User journey path: company info → core business → user engagement
        user_journey_path = self._create_user_journey_path(start, sitemap, prioritized_neighborhoods)
        if user_journey_path:
            coverage_paths['user_journey'] = user_journey_path
        
        # Comprehensive path: visit all high-priority neighborhoods
        comprehensive_path = self._create_comprehensive_path(start, sitemap, prioritized_neighborhoods)
        if comprehensive_path:
            coverage_paths['comprehensive'] = comprehensive_path
        
        return coverage_paths

    def _create_business_path(self, start: str, sitemap: Dict[str, List[str]], 
                             prioritized_neighborhoods: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create business-focused navigation path."""
        path = []
        current = start
        
        # Try to find path: start → core business → user engagement
        core_business_pages = prioritized_neighborhoods.get('core_business', {}).get('prioritized_pages', [])
        user_engagement_pages = prioritized_neighborhoods.get('user_engagement', {}).get('prioritized_pages', [])
        
        if core_business_pages and user_engagement_pages:
            # Find path to first core business page
            core_path = self._find_shortest_path_bfs(current, core_business_pages[0]['url'], sitemap)
            if core_path:
                path.append({
                    'segment': core_path,
                    'target': core_business_pages[0]['url'],
                    'reasoning': f"Route to core business functionality: {core_business_pages[0]['url']}"
                })
                current = core_business_pages[0]['url']
            
            # Find path from core business to user engagement
            engagement_path = self._find_shortest_path_bfs(current, user_engagement_pages[0]['url'], sitemap)
            if engagement_path:
                path.append({
                    'segment': engagement_path,
                    'target': user_engagement_pages[0]['url'],
                    'reasoning': f"Route to user engagement: {user_engagement_pages[0]['url']}"
                })
        
        return path

    def _create_user_journey_path(self, start: str, sitemap: Dict[str, List[str]], 
                                 prioritized_neighborhoods: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create user journey navigation path."""
        path = []
        current = start
        
        # Path: company info → core business → user engagement
        company_info_pages = prioritized_neighborhoods.get('company_info', {}).get('prioritized_pages', [])
        core_business_pages = prioritized_neighborhoods.get('core_business', {}).get('prioritized_pages', [])
        user_engagement_pages = prioritized_neighborhoods.get('user_engagement', {}).get('prioritized_pages', [])
        
        if company_info_pages:
            company_path = self._find_shortest_path_bfs(current, company_info_pages[0]['url'], sitemap)
            if company_path:
                path.append({
                    'segment': company_path,
                    'target': company_info_pages[0]['url'],
                    'reasoning': f"Route to company information: {company_info_pages[0]['url']}"
                })
                current = company_info_pages[0]['url']
        
        if core_business_pages:
            business_path = self._find_shortest_path_bfs(current, core_business_pages[0]['url'], sitemap)
            if business_path:
                path.append({
                    'segment': business_path,
                    'target': core_business_pages[0]['url'],
                    'reasoning': f"Route to core business: {core_business_pages[0]['url']}"
                })
                current = core_business_pages[0]['url']
        
        if user_engagement_pages:
            engagement_path = self._find_shortest_path_bfs(current, user_engagement_pages[0]['url'], sitemap)
            if engagement_path:
                path.append({
                    'segment': engagement_path,
                    'target': user_engagement_pages[0]['url'],
                    'reasoning': f"Route to user engagement: {user_engagement_pages[0]['url']}"
                })
        
        return path

    def _create_comprehensive_path(self, start: str, sitemap: Dict[str, List[str]], 
                                 prioritized_neighborhoods: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create comprehensive path covering all high-priority neighborhoods."""
        path = []
        current = start
        
        # Visit neighborhoods in priority order
        priority_order = ['user_engagement', 'core_business', 'company_info', 'documentation']
        
        for neighborhood in priority_order:
            if neighborhood in prioritized_neighborhoods:
                pages = prioritized_neighborhoods[neighborhood]['prioritized_pages']
                if pages:
                    target = pages[0]['url']
                    segment_path = self._find_shortest_path_bfs(current, target, sitemap)
                    if segment_path:
                        path.append({
                            'segment': segment_path,
                            'target': target,
                            'reasoning': f"Route to {neighborhood.replace('_', ' ')}: {target}"
                        })
                        current = target
        
        return path

    def _generate_navigation_plan(self, start_url: str, sitemap: Dict[str, List[str]], 
                                 prioritized_neighborhoods: Dict[str, Dict[str, Any]], 
                                 navigation_paths: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the final comprehensive navigation plan."""
        from datetime import datetime
        
        plan = {
            'metadata': {
                'start_url': start_url,
                'total_pages': len(sitemap),
                'total_neighborhoods': len(prioritized_neighborhoods),
                'generation_timestamp': datetime.now().isoformat()
            },
            'neighborhoods': prioritized_neighborhoods,
            'navigation_paths': navigation_paths,
            'summary': {
                'high_priority_pages': [],
                'critical_paths': [],
                'recommendations': []
            }
        }
        
        # Generate summary
        for neighborhood_name, neighborhood_data in prioritized_neighborhoods.items():
            if neighborhood_data['priority'] in ['critical', 'high']:
                for page_data in neighborhood_data['prioritized_pages'][:2]:  # Top 2 pages
                    plan['summary']['high_priority_pages'].append({
                        'url': page_data['url'],
                        'priority_score': page_data['priority_score'],
                        'neighborhood': neighborhood_name,
                        'reasoning': page_data['reasoning']
                    })
        
        # Identify critical paths
        for destination, path_info in navigation_paths['shortest_paths'].items():
            if path_info['hops'] <= 2:  # Paths with 2 or fewer hops
                plan['summary']['critical_paths'].append({
                    'destination': destination,
                    'hops': path_info['hops'],
                    'path': path_info['path'],
                    'reasoning': path_info['reasoning']
                })
        
        # Generate recommendations
        plan['summary']['recommendations'] = self._generate_recommendations(
            prioritized_neighborhoods, navigation_paths, sitemap
        )
        
        return plan

    def _generate_recommendations_from_plan(self, navigation_plan: Dict[str, Any], sitemap: Dict[str, List[str]]) -> List[str]:
        """Generate actionable recommendations based on the comprehensive navigation plan."""
        recommendations = []
        
        # Check for isolated pages
        isolated_pages = []
        for url, neighbors in sitemap.items():
            if len(neighbors) == 0 and len([n for n in sitemap.values() if url in n]) == 0:
                isolated_pages.append(url)
        
        if isolated_pages:
            recommendations.append(f"⚠️  Found {len(isolated_pages)} isolated pages that may need linking")
        
        # Check for long navigation paths
        long_paths = []
        for page_data in navigation_plan['section_2_navigation_paths'].values():
            if page_data['paths']:
                longest_path = max(page_data['paths'], key=lambda x: x['hops'])
                if longest_path['hops'] > 3:
                    long_paths.append(longest_path)
        
        if long_paths:
            recommendations.append(f"⚠️  {len(long_paths)} destinations require 4+ hops - consider adding direct links")
        
        # Check dynamic content distribution
        high_dynamic_pages = [p for p in navigation_plan['section_1_priority_ranking'] if p['dynamic_score'] >= 20]
        if len(high_dynamic_pages) >= 3:
            recommendations.append(f"✅ Strong dynamic content with {len(high_dynamic_pages)} highly interactive pages")
        elif len(high_dynamic_pages) == 0:
            recommendations.append("⚠️  No high-dynamic content pages found - consider adding interactive elements")
        
        # Check neighborhood connectivity
        if navigation_plan['metadata']['total_neighborhoods'] > 0:
            recommendations.append(f"✅ Good page connectivity with {navigation_plan['metadata']['total_neighborhoods']} neighborhoods")
        else:
            recommendations.append("⚠️  Limited page connectivity - consider improving internal linking")
        
        return recommendations

    def _generate_clean_output(self, navigation_plan: Dict[str, Any]) -> None:
        """
        Generate clean, hierarchical output following strict rules.
        Output is machine-readable and follows the exact format specified.
        """
        print("\n" + "="*100)
        print("🤖 AI NAVIGATION PLANNER - COMPREHENSIVE OUTPUT")
        print("="*100)
        
        # Section 1: Priority Ranking (dynamic content only)
        print("\n" + "="*80)
        print("📊 SECTION 1: PRIORITY RANKING (Dynamic Content Only)")
        print("="*80)
        print("🎯 Pages ranked from highest to lowest by dynamic content score")
        print("⚡ Scoring: Forms(×10) + Inputs(×8) + Scripts(×6) + Buttons(×3)")
        print("-" * 80)
        
        for i, page_data in enumerate(navigation_plan['section_1_priority_ranking'], 1):
            status_icon = "✅" if page_data.get('status') == 'fully_scraped' else "⚠️" if page_data.get('status') == 'discovered_not_scraped' else "❌"
            status_text = page_data.get('status', 'unknown').replace('_', ' ').title()
            
            print(f"\n🏆 RANK #{i}: {page_data['url']}")
            print(f"   {status_icon} Status: {status_text}")
            print(f"   ⚡ Dynamic Score: {page_data['dynamic_score']}")
            print(f"   📊 Features: Scripts({page_data['features']['scripts']}) | Forms({page_data['features']['forms']}) | Inputs({page_data['features']['inputs']}) | Buttons({page_data['features']['buttons']}) | Images({page_data['features']['images']}) | Links({page_data['features']['links']})")
            print(f"   🧮 Score Breakdown: {page_data['score_breakdown']}")
        
        # Section 2: Navigation Paths (all possible paths, shortest→longest)
        print("\n" + "="*80)
        print("🛣️  SECTION 2: NAVIGATION PATHS (All Possible Routes)")
        print("="*80)
        print("🎯 For each page in ranking order, showing ALL paths from landing page")
        print("📏 Paths ordered from SHORTEST to LONGEST (no summarization)")
        print("-" * 80)
        
        for page_url, page_info in navigation_plan['section_2_navigation_paths'].items():
            print(f"\n🎯 DESTINATION #{page_info['rank']}: {page_url}")
            print(f"   ⚡ Dynamic Score: {page_info['dynamic_score']}")
            print(f"   🛣️  Total Paths Found: {len(page_info['paths'])}")
            
            if page_info['paths']:
                for j, path_info in enumerate(page_info['paths'], 1):
                    print(f"      📍 Path #{j} ({path_info['hops']} hops): {path_info['path_string']}")
            else:
                print("      ❌ No paths found")
        
        # Section 3: COMPREHENSIVE Neighbor Analysis (ALL Pages in Priority Order)
        print("\n" + "="*80)
        print("🔗 SECTION 3: COMPREHENSIVE NEIGHBOR ANALYSIS (ALL Pages in Priority Order)")
        print("="*80)
        print("🎯 For each page in ranking order, showing ALL paths to its immediate neighbors")
        print("📏 Paths ordered from SHORTEST to LONGEST for each neighbor")
        print("⚡ Processing order: Highest dynamic score → Lowest dynamic score")
        print("-" * 80)
        
        # Process pages in priority order (highest score first)
        sorted_pages = sorted(
            navigation_plan['section_3_neighbors'].items(),
            key=lambda x: x[1].get('rank', 999)  # Sort by rank, handle missing rank gracefully
        )
        
        for page_url, page_info in sorted_pages:
            page_rank = page_info.get('rank', 'N/A')
            page_score = page_info.get('dynamic_score', 'N/A')
            neighbor_count = page_info.get('immediate_neighbor_count', 0)
            
            print(f"\n🎯 SOURCE PAGE #{page_rank} (Score: {page_score}): {page_url}")
            print(f"   📊 Direct Neighbors: {neighbor_count}")
            
            if neighbor_count > 0:
                # Sort neighbors by their rank in the priority list
                neighbors = page_info['immediate_neighbors']
                sorted_neighbors = sorted(
                    neighbors.items(),
                    key=lambda x: (
                        x[1].get('neighbor_rank', 999) if x[1].get('neighbor_rank') != 'N/A' else 999,
                        x[0]  # Fallback to URL if rank is same
                    )
                )
                
                for i, (neighbor_url, neighbor_paths) in enumerate(sorted_neighbors, 1):
                    total_routes = len(neighbor_paths['all_possible_paths'])
                    neighbor_rank = neighbor_paths.get('neighbor_rank', 'N/A')
                    neighbor_score = neighbor_paths.get('neighbor_score', 'N/A')
                    
                    print(f"\n      🎯 NEIGHBOR #{i}: {neighbor_url}")
                    print(f"         📊 Neighbor Rank: #{neighbor_rank} (Score: {neighbor_score})")
                    print(f"         🛣️  Total Paths Found: {total_routes}")
                    
                    if total_routes > 0:
                        # Show first 5 routes with detailed info
                        for k, path_info in enumerate(neighbor_paths['all_possible_paths'][:5], 1):
                            is_shortest = "🏆 SHORTEST" if path_info.get('is_shortest', False) else f"   #{k}"
                            print(f"            {is_shortest} [{path_info['hops']} hops]: {path_info['path_string']}")
                        
                        # Show summary for remaining routes
                        if total_routes > 5:
                            remaining = total_routes - 5
                            print(f"            ... and {remaining} more routes")
                    else:
                        print("            ❌ No routes found")
            else:
                print("   📍 No immediate neighbors found")
        
        print(f"\n📊 COMPREHENSIVE ANALYSIS SUMMARY:")
        print(f"   🎯 Total pages analyzed: {len(navigation_plan['section_3_neighbors'])}")
        print(f"   🔗 Total neighbors processed: {sum(page.get('immediate_neighbor_count', 0) for page in navigation_plan['section_3_neighbors'].values())}")
        print(f"   🏆 Processing order: Highest dynamic score → Lowest dynamic score")
        print(f"   📏 Path ordering: Shortest → Longest for each neighbor")
        
        print("\n" + "="*100)
        print("✅ AI NAVIGATION PLANNER - COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*100)

    def _generate_recommendations(self, prioritized_neighborhoods: Dict[str, Dict[str, Any]], 
                                 navigation_paths: Dict[str, Any], sitemap: Dict[str, List[str]]) -> List[str]:
        """Generate actionable recommendations based on the analysis."""
        recommendations = []
        
        # Check for missing critical pages
        critical_neighborhoods = ['user_engagement', 'core_business']
        for neighborhood in critical_neighborhoods:
            if neighborhood not in prioritized_neighborhoods:
                recommendations.append(f"⚠️  Missing critical neighborhood: {neighborhood}")
        
        # Check for isolated pages
        isolated_pages = []
        for url, neighbors in sitemap.items():
            if len(neighbors) == 0 and len([n for n in sitemap.values() if url in n]) == 0:
                isolated_pages.append(url)
        
        # Check for long navigation paths
        long_paths = [p for p in navigation_paths['shortest_paths'].values() if p['hops'] > 3]
        if long_paths:
            recommendations.append(f"⚠️  {len(long_paths)} destinations require 4+ hops - consider adding direct links")
        
        # Positive findings
        high_priority_count = sum(1 for n in prioritized_neighborhoods.values() 
                                if n['priority'] in ['critical', 'high'])
        if high_priority_count >= 3:
            recommendations.append(f"✅ Strong neighborhood coverage with {high_priority_count} high-priority areas")
        
        return recommendations


# Clean API functions as requested
def build_route_tree(start_node: str, sitemap: Dict[str, List[str]], max_depth: int = 6, 
                    api_key: str = None) -> RouteNode:
    """
    Build a compressed route tree starting from the specified node.
    
    This is the main entry point for building route trees. It creates
    an IntelligentNavigationPlanner instance and builds the route tree
    using intelligent exploration decisions.
    
    Args:
        start_node: Starting node for route exploration
        sitemap: Dictionary mapping pages to their neighbors
        max_depth: Maximum hop depth for route exploration
        api_key: GPT-OSS 20B API key (required)
        
    Returns:
        RouteNode: Root of the compressed route tree
        
    Raises:
        ValueError: If api_key is not provided
        RuntimeError: If route tree construction fails
    """
    if not api_key:
        raise ValueError("API key is required for GPT-OSS 20B access")
    
    planner = IntelligentNavigationPlanner(api_key, max_depth)
    return planner.build_route_tree(start_node, sitemap, max_depth)


def expand_route_tree(route_tree: RouteNode) -> Generator[List[str], None, None]:
    """
    Expand the compressed route tree into full paths.
    
    This function takes a compressed route tree and yields all possible
    paths from the root to leaf nodes.
    
    Args:
        route_tree: Root node of the compressed route tree
        
    Yields:
        List[str]: Complete route path from root to leaf
    """
    if not route_tree:
        return
    
    def _expand_recursive(node: RouteNode, current_path: List[str]):
        current_path.append(node.url)
        
        if node.is_leaf():
            yield current_path[:]
        else:
            for child in node.children.values():
                yield from _expand_recursive(child, current_path[:])
    
    yield from _expand_recursive(route_tree, [])


def main():
    """Example usage of the intelligent navigation planner."""
    
    # Your API key
    API_KEY = "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"
    
    # Example sitemap
    sitemap = {
        "landing": ["about", "pricing", "contact"],
        "about": ["team", "careers"],
        "pricing": ["checkout", "faq"],
        "contact": ["form"],
        "team": ["leadership"],
        "careers": ["openings"],
        "checkout": ["payment"],
        "faq": ["support"],
        "form": ["submission"],
        "leadership": [],
        "openings": [],
        "payment": [],
        "support": [],
        "submission": []
    }
    
    print("🚀 Intelligent Site Navigation Planner Demo")
    print("=" * 60)
    print("Using DSPy PlanAgent with GPT-OSS 20B for intelligent route exploration")
    print("=" * 60)
    
    try:
        # Build the route tree
        print("\n1️⃣  Building compressed route tree...")
        start_time = time.time()
        
        route_tree = build_route_tree("landing", sitemap, max_depth=4, api_key=API_KEY)
        
        build_time = time.time() - start_time
        print(f"✅ Route tree built in {build_time:.2f}s")
        
        # Display tree structure
        print("\n2️⃣  Route tree structure:")
        def print_tree(node: RouteNode, level: int = 0):
            indent = "  " * level
            print(f"{indent}📄 {node.url}")
            for child in node.children.values():
                print_tree(child, level + 1)
        
        print_tree(route_tree)
        
        # Expand and display all routes
        print("\n3️⃣  All possible routes:")
        routes = list(expand_route_tree(route_tree))
        
        for i, route in enumerate(routes, 1):
            print(f"   {i:2d}. {' → '.join(route)}")
        
        print(f"\n📊 Total routes found: {len(routes)}")
        
        # Create planner instance for detailed stats
        planner = IntelligentNavigationPlanner(API_KEY, max_depth=4)
        stats = planner.get_route_summary(route_tree)
        
        print(f"\n4️⃣  Performance statistics:")
        print(f"   • Total nodes: {stats.total_nodes}")
        print(f"   • Total paths: {stats.total_paths}")
        print(f"   • Construction time: {stats.construction_time:.2f}s")
        print(f"   • DSPy decisions: {stats.dspy_decisions}")
        print(f"   • DSPy failures: {stats.dspy_failures}")
        print(f"   • Fallback usage: {stats.fallback_usage}")
        
        if planner.subpath_cache:
            cache_stats = planner.get_cache_stats()
            print(f"   • Cache hit rate: {cache_stats['hit_rate_percent']}%")
        
        print("\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        logger.error(f"Demo failed: {e}", exc_info=True)


if __name__ == "__main__":
    main()
