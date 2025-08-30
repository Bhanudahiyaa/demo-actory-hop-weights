#!/usr/bin/env python3
"""
Ultra-Fast Planning Agent - Weight-Based Route Discovery
Uses threading, advanced caching, and smart algorithms for finding ALL possible routes by weight priority.
"""

import time
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlparse
from collections import deque, defaultdict
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import dspy
from dspy import Module

from .neo4j_client import Neo4jClient
from .plan import make_hierarchical_plan_from_neighbors


class UltraFastPathCache:
    """
    Ultra-fast path caching with memory optimization and thread-safe access.
    """
    
    def __init__(self):
        self.path_cache = {}
        self.sub_path_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_paths_per_pair = 1000   # Limited for speed
        self.max_total_paths = 10000     # Limited for speed
        
        # Thread-safe operations
        self._lock = threading.Lock()
    
    def cache_path(self, start: str, end: str, paths: List[List[str]]):
        """Cache paths with reasonable limits for faster route discovery."""
        with self._lock:
            if start not in self.path_cache:
                self.path_cache[start] = {}
            
            # Store ALL paths - no limits for complete coverage
            self.path_cache[start][end] = paths
            
            # Cache sub-paths for reuse
            for path in paths:
                self._cache_sub_paths(path)
    
    def _cache_sub_paths(self, full_path: List[str]):
        """Cache sub-paths for intelligent reuse."""
        for i in range(len(full_path)):
            for j in range(i + 1, len(full_path) + 1):
                sub_path = tuple(full_path[i:j])
                if sub_path not in self.sub_path_cache:
                    self.sub_path_cache[sub_path] = []
                self.sub_path_cache[sub_path].append({
                    'full_path': full_path,
                    'start_index': i,
                    'end_index': j
                })
    
    def is_path_cached(self, start: str, end: str) -> bool:
        """Check if we have cached paths."""
        return (start in self.path_cache and 
                end in self.path_cache[start] and 
                len(self.path_cache[start][end]) > 0)
    
    def get_cached_paths(self, start: str, end: str) -> List[List[str]]:
        """Get cached paths."""
        if self.is_path_cached(start, end):
            self.cache_hits += 1
            return self.path_cache[start][end]
        self.cache_misses += 1
        return []
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache performance statistics."""
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
            'cached_paths': len(self.path_cache),
            'sub_paths': len(self.sub_path_cache)
        }


class WeightBasedRouteFinder:
    """
    Weight-based route finder using threading and smart algorithms.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.path_cache = UltraFastPathCache()
        self.route_count = 0
        
        # Pre-computed graph structure
        self._graph_cache = None
        self._graph_cache_time = 0
        self._graph_cache_ttl = 300  # 5 minutes
        
        # Thread-safe graph access
        self._graph_lock = threading.Lock()
        
    def find_all_routes(self, start: str, target: str, max_hops: int = 8) -> Dict[str, Any]:
        """Find ALL routes with unlimited discovery."""
        
        # Check cache first
        if self.path_cache.is_path_cached(start, target):
            cached_paths = self.path_cache.get_cached_paths(start, target)
            return {
                'all_routes': cached_paths,
                'total_routes': len(cached_paths),
                'status': 'from_cache',
                'optimization_note': f'Retrieved {len(cached_paths)} routes from cache'
            }
        
        # Get graph structure (cached)
        graph = self._get_cached_graph_structure()
        if not graph or start not in graph or target not in graph:
            return {
                'all_routes': [],
                'total_routes': 0,
                'status': 'not_found',
                'error': 'Start or target not in graph'
            }
        
        print(f"      🚀 Finding ALL routes: {start} → {target} (max {max_hops} hops)")
        
        # Use threaded BFS for unlimited route discovery
        all_routes = self._threaded_bfs_all_routes(graph, start, target, max_hops)
        
        # Cache results
        self.path_cache.cache_path(start, target, all_routes)
        
        # Sort by length (shortest first) - ALL routes in ascending order
        all_routes.sort(key=lambda x: len(x))
        
        return {
            'all_routes': all_routes,
            'total_routes': len(all_routes),
            'status': 'found',
            'shortest_route': all_routes[0] if all_routes else [],
            'shortest_distance': len(all_routes[0]) - 1 if all_routes else -1,
            'optimization_note': f'Found {len(all_routes)} routes using unlimited BFS'
        }
    
    def _get_cached_graph_structure(self) -> Dict[str, Set[str]]:
        """Get cached graph structure to avoid repeated database queries."""
        with self._graph_lock:
            current_time = time.time()
            
            if (self._graph_cache is None or 
                current_time - self._graph_cache_time > self._graph_cache_ttl):
                
                self._graph_cache = self._fetch_graph_structure()
                self._graph_cache_time = current_time
                
            return self._graph_cache
    
    def _fetch_graph_structure(self) -> Dict[str, Set[str]]:
        """Fetch graph structure from Neo4j."""
        try:
            query = """
            MATCH (p:Page)
            OPTIONAL MATCH (p)-[:LINKS_TO]->(neighbor:Page)
            RETURN p.url AS page_url, 
                   collect(DISTINCT neighbor.url) AS neighbors
            """
            
            with self.neo4j._driver.session() as session:
                result = session.run(query)
                graph = {}
                for record in result:
                    page_url = record['page_url']
                    neighbors = record['neighbors']
                    valid_neighbors = {url for url in neighbors if url and url.strip()}
                    graph[page_url] = valid_neighbors
                return graph
                
        except Exception as e:
            print(f"      ⚠️  Failed to get graph: {e}")
            return {}
    
    def _threaded_bfs_all_routes(self, graph: Dict[str, Set[str]], start: str, target: str, max_hops: int) -> List[List[str]]:
        """Threaded BFS that finds ALL possible routes without limits."""
        
        if start == target:
            return [[start]]
        
        # Use threading for route discovery
        with ThreadPoolExecutor(max_workers=min(6, threading.active_count() + 3)) as executor:
            # Split the work into hop levels
            futures = []
            
            # Submit threaded BFS tasks for different hop levels
            for hop_level in range(1, max_hops + 1):
                future = executor.submit(
                    self._bfs_all_routes_at_hop_level, 
                    graph, start, target, max_hops, hop_level
                )
                futures.append(future)
            
            # Collect results
            all_routes = []
            for future in as_completed(futures):
                try:
                    routes = future.result()
                    all_routes.extend(routes)
                except Exception as e:
                    print(f"      ⚠️  Threaded BFS error: {e}")
            
            # Remove duplicates and sort
            unique_routes = list({tuple(route) for route in all_routes})
            return [list(route) for route in unique_routes]
    
    def _bfs_all_routes_at_hop_level(self, graph: Dict[str, Set[str]], start: str, target: str, max_hops: int, current_hop: int) -> List[List[str]]:
        """BFS at specific hop level for limited route discovery (max 1000 routes)."""
        if current_hop > max_hops:
            return []
        
        # Use a set to avoid duplicate routes
        routes = set()
        queue = deque([(start, [start])])
        
        while queue and len(routes) < 1000:  # Limit to 1000 routes for speed
            current, path = queue.popleft()
            
            if current == target and len(path) == current_hop:
                routes.add(tuple(path))
                continue
            
            if len(path) >= current_hop:
                continue
            
            # Get neighbors from Neo4j
            neighbors = graph.get(current, [])
            
            for neighbor in neighbors:
                if neighbor not in path:  # Avoid cycles
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))
        
        # Convert back to list of lists
        return [list(route) for route in routes]


class WeightBasedNavigationPlanner(Module):
    """
    Weight-based navigation planner that prioritizes by page weights.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, landing_url: str, neighbors: List[Dict], max_pages: int) -> Dict[str, Any]:
        """Generate weight-based navigation plan starting from landing page."""
        
        # Calculate weights for each neighbor
        weighted_neighbors = self._calculate_weights(neighbors)
        
        # Sort by weight in descending order (highest weight first)
        weighted_neighbors.sort(key=lambda x: x['total_weight'], reverse=True)
        
        # Create navigation plan with weight priority
        navigation_plan = [neighbor['url'] for neighbor in weighted_neighbors]
        
        # Limit to max_pages
        if len(navigation_plan) > max_pages:
            navigation_plan = navigation_plan[:max_pages]
        
        # Show weight-based ordering
        print(f"🎯 Weight-Based Navigation Plan (Highest Weight First):")
        for i, neighbor in enumerate(weighted_neighbors[:10]):  # Show top 10
            print(f"   {i+1}. {neighbor['url']} - Weight: {neighbor['total_weight']}")
            print(f"      Scripts: {neighbor['scriptCount']}, Forms: {neighbor['formCount']}, Links: {neighbor['linkCount']}")
        
        if len(weighted_neighbors) > 10:
            print(f"   ... and {len(weighted_neighbors) - 10} more pages")
        
        return {
            'navigation_plan': navigation_plan,
            'planned_pages': len(navigation_plan),
            'weighted_neighbors': weighted_neighbors
        }
    
    def _calculate_weights(self, neighbors: List[Dict]) -> List[Dict]:
        """Calculate comprehensive weights for each neighbor."""
        weighted_neighbors = []
        
        for neighbor in neighbors:
            # Calculate total weight based on all features
            total_weight = (
                neighbor['scriptCount'] * 10 +      # Scripts are high value
                neighbor['formCount'] * 8 +         # Forms are high value
                neighbor['buttonCount'] * 5 +       # Buttons are medium value
                neighbor['inputCount'] * 5 +        # Inputs are medium value
                neighbor['imageCount'] * 3 +        # Images are medium value
                neighbor['linkCount'] * 2 +         # Links are lower value
                neighbor['mediaCount'] * 4          # Media is medium-high value
            )
            
            weighted_neighbors.append({
                **neighbor,
                'total_weight': total_weight
            })
        
        return weighted_neighbors


class WeightBasedPlanningAgent(Module):
    """
    Weight-Based Planning Agent that finds ALL routes prioritized by weights.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__()
        self.neo4j = neo4j_client
        self.navigation_planner = WeightBasedNavigationPlanner()
        self.route_finder = WeightBasedRouteFinder(neo4j_client)
        
        # Track state
        self.current_position = None
        self.navigation_history = []
        self.discovered_routes = {}
        
        # Performance tracking
        self.start_time = None
        self.route_discovery_times = []
    
    def forward(self, landing_url: str, max_pages: int = 50, exploration_strategy: str = "weight_based") -> Dict[str, Any]:
        """Execute weight-based navigation planning with ALL routes."""
        
        self.start_time = time.time()
        self.current_position = landing_url
        
        print(f"🚀 Starting WEIGHT-BASED Navigation Planning from: {landing_url}")
        print(f"📊 Target: {max_pages} pages | Strategy: {exploration_strategy}")
        print("⚡ Agent will find ALL possible routes prioritized by page weights!")
        print("🎯 Highest weight pages (pricing, forms, scripts) will be processed FIRST!")
        print("-" * 80)
        
        # Phase 1: Get initial neighbors and create weight-based navigation plan
        print(f"\n📋 PHASE 1: Creating Weight-Based Navigation Plan")
        print("-" * 60)
        
        initial_neighbors = self._get_initial_neighbors(landing_url, 100)
        
        if not initial_neighbors:
            return {
                'error': 'No neighbors found for landing page',
                'landing_url': landing_url
            }
        
        # Create weight-based navigation plan
        plan_result = self.navigation_planner.forward(
            landing_url=landing_url,
            neighbors=initial_neighbors,
            max_pages=max_pages
        )
        
        navigation_plan = plan_result['navigation_plan']
        weighted_neighbors = plan_result['weighted_neighbors']
        print(f"🗺️  Created weight-based navigation plan with {len(navigation_plan)} URLs")
        
        # Phase 2: Weight-based route discovery (highest weight first)
        print(f"\n⚡ PHASE 2: Weight-Based Route Discovery (Highest Weight First)")
        print("-" * 70)
        
        route_discovery_results = []
        total_routes_found = 0
        
        # Process routes in weight order (highest weight first)
        for i, target_url in enumerate(navigation_plan[:max_pages]):
            # Get weight info for this target
            target_weight_info = next((n for n in weighted_neighbors if n['url'] == target_url), {})
            target_weight = target_weight_info.get('total_weight', 0)
            
            print(f"\n🎯 TARGET {i+1}: {target_url}")
            print(f"   ⚖️  Weight: {target_weight} (Scripts: {target_weight_info.get('scriptCount', 0)}, Forms: {target_weight_info.get('formCount', 0)})")
            print("-" * 70)
            
            # Discover ALL routes for this target
            result = self._discover_all_routes_for_target(landing_url, target_url, i + 1)
            route_discovery_results.append(result)
            
            if result['route_result']['all_routes']:
                total_routes_found += result['route_result']['total_routes']
        
        # Phase 3: Generate comprehensive performance report
        print(f"\n📊 PHASE 3: Comprehensive Performance Analysis")
        print("-" * 50)
        
        navigation_duration = time.time() - self.start_time
        
        # Analyze performance
        performance_analysis = self._analyze_performance(route_discovery_results, total_routes_found)
        
        # Show performance metrics
        print(f"⚡ PERFORMANCE METRICS:")
        print(f"   Total Duration: {navigation_duration:.2f}s")
        print(f"   Routes Found: {total_routes_found:,}")
        print(f"   Routes/Second: {total_routes_found/navigation_duration:.0f}")
        print(f"   Avg Time per Target: {navigation_duration/len(route_discovery_results):.2f}s")
        
        # Cache performance
        cache_stats = self.route_finder.path_cache.get_cache_stats()
        print(f"   Cache Hit Rate: {cache_stats['hit_rate']:.1%}")
        print(f"   Cache Hits: {cache_stats['hits']}")
        print(f"   Cache Misses: {cache_stats['misses']}")
        
        # Show weight-based summary
        print(f"\n🎯 WEIGHT-BASED SUMMARY:")
        print(f"   Highest Weight Page: {weighted_neighbors[0]['url']} (Weight: {weighted_neighbors[0]['total_weight']})")
        print(f"   Lowest Weight Page: {weighted_neighbors[-1]['url']} (Weight: {weighted_neighbors[-1]['total_weight']})")
        print(f"   Average Weight: {sum(n['total_weight'] for n in weighted_neighbors) / len(weighted_neighbors):.1f}")
        
        # Compile final result
        final_result = {
            'landing_url': landing_url,
            'navigation_plan': navigation_plan,
            'weighted_neighbors': weighted_neighbors,
            'route_discovery_results': route_discovery_results,
            'performance_analysis': performance_analysis,
            'discovered_routes': self.discovered_routes,
            'navigation_history': self.navigation_history,
            'metadata': {
                'max_pages': max_pages,
                'exploration_strategy': exploration_strategy,
                'navigation_duration': round(navigation_duration, 2),
                'total_routes_found': total_routes_found,
                'routes_per_second': round(total_routes_found/navigation_duration, 0),
                'weight_based_ordering': True
            }
        }
        
        print(f"\n🎉 WEIGHT-BASED Navigation Planning completed!")
        print(f"⏱️  Duration: {navigation_duration:.2f}s")
        print(f"🎯 Explored: {len(route_discovery_results)} pages by weight priority")
        print(f"🗺️  Total Routes: {total_routes_found:,} discovered")
        print(f"⚡ Speed: {total_routes_found/navigation_duration:.0f} routes/second")
        print(f"🎯 Weight Priority: Highest value pages processed first!")
        
        return final_result
    
    def _discover_all_routes_for_target(self, landing_url: str, target_url: str, target_index: int) -> Dict:
        """Discover ALL routes for a single target with unlimited discovery."""
        
        # Find ALL routes with timing
        route_start = time.time()
        route_result = self.route_finder.find_all_routes(
            start=landing_url,
            target=target_url,
            max_hops=8  # Increased for complete coverage
        )
        route_duration = time.time() - route_start
        
        # Store discovered routes
        if route_result['all_routes']:
            self.discovered_routes[target_url] = route_result['all_routes']
        
        # Show comprehensive route summary
        all_routes = route_result.get('all_routes', [])
        shortest_route = route_result.get('shortest_route', [])
        total_routes = route_result.get('total_routes', 0)
        
        print(f"      🏆 SHORTEST: {len(shortest_route)-1} hops")
        print(f"      📊 Total Routes: {total_routes:,}")
        print(f"      ⚡ Discovery Time: {route_duration:.2f}s")
        
        # Show route distribution by hop count
        if all_routes:
            hop_distribution = defaultdict(int)
            for route in all_routes:
                hop_distribution[len(route) - 1] += 1
            
            print(f"      📊 Route Distribution by Hops:")
            for hops in sorted(hop_distribution.keys()):
                count = hop_distribution[hops]
                print(f"         {hops} hops: {count:,} routes")
        
        # Show sample routes (first 10)
        if all_routes:
            print(f"      🗺️  Sample Routes (First 10):")
            for j, route in enumerate(all_routes[:10]):
                route_str = " → ".join(route[-2:])  # Show only last 2 nodes
                hops = len(route) - 1
                print(f"         {j+1}. ({hops} hops): ... → {route_str}")
            
            if len(all_routes) > 10:
                print(f"         ... and {len(all_routes) - 10} more routes")
        
        # Track performance
        self.route_discovery_times.append(route_duration)
        
        # Compile result
        route_discovery = {
            'target_url': target_url,
            'route_result': route_result,
            'discovery_order': target_index,
            'discovery_time': route_duration
        }
        
        # Update current position
        self.current_position = target_url
        self.navigation_history.append({
            'url': target_url,
            'timestamp': time.time(),
            'order': target_index
        })
        
        return route_discovery
    
    def _get_initial_neighbors(self, landing_url: str, limit: int) -> List[Dict]:
        """Get initial neighbors for navigation planning."""
        try:
            query = """
            MATCH (p:Page {url: $landing_url})-[r:LINKS_TO]->(neighbor:Page)
            RETURN neighbor.url as url,
                   COALESCE(neighbor.scriptCount, 0) as scriptCount,
                   COALESCE(neighbor.formCount, 0) as formCount,
                   COALESCE(neighbor.buttonCount, 0) as buttonCount,
                   COALESCE(neighbor.inputCount, 0) as inputCount,
                   COALESCE(neighbor.imageCount, 0) as imageCount,
                   COALESCE(neighbor.linkCount, 0) as linkCount,
                   COALESCE(neighbor.mediaCount, 0) as mediaCount,
                   COALESCE(neighbor.visited, false) as visited
            LIMIT $limit
            """
            
            records = self.neo4j._driver.execute_query(
                query, 
                {"landing_url": landing_url, "limit": limit}
            ).records
            
            neighbors = []
            for record in records:
                neighbor = {
                    'url': record["url"],
                    'scriptCount': record.get("scriptCount", 0) or 0,
                    'formCount': record.get("formCount", 0) or 0,
                    'buttonCount': record.get("buttonCount", 0) or 0,
                    'inputCount': record.get("inputCount", 0) or 0,
                    'imageCount': record.get("imageCount", 0) or 0,
                    'linkCount': record.get("linkCount", 0) or 0,
                    'mediaCount': record.get("mediaCount", 0) or 0,
                    'visited': record.get("visited", False) or False
                }
                neighbors.append(neighbor)
            
            return neighbors
            
        except Exception as e:
            print(f"      ⚠️  Failed to get initial neighbors: {e}")
            return []
    
    def _analyze_performance(self, route_discovery_results: List[Dict], total_routes: int) -> Dict[str, Any]:
        """Analyze performance metrics."""
        
        if not self.route_discovery_times:
            return {}
        
        return {
            'total_duration': time.time() - self.start_time,
            'total_routes': total_routes,
            'routes_per_second': total_routes / (time.time() - self.start_time),
            'avg_discovery_time': sum(self.route_discovery_times) / len(self.route_discovery_times),
            'min_discovery_time': min(self.route_discovery_times),
            'max_discovery_time': max(self.route_discovery_times),
            'cache_performance': self.route_finder.path_cache.get_cache_stats()
        }


# Alias for backward compatibility
UltraFastPlanningAgent = WeightBasedPlanningAgent
UltraFastRouteFinder = WeightBasedRouteFinder
UltraFastNavigationPlanner = WeightBasedNavigationPlanner
UltraFastPathCache = UltraFastPathCache
