#!/usr/bin/env python3
"""
Smart Planning Agent - Find EVERY Possible Route Efficiently
Uses intelligent path caching, route reuse, and early termination to avoid endless loops.
"""

import time
import json
from typing import List, Dict, Any, Optional, Set, Tuple
from urllib.parse import urlparse
from collections import deque, defaultdict
import heapq

import dspy
from dspy import Module

from .neo4j_client import Neo4jClient
from .plan import make_hierarchical_plan_from_neighbors


class SmartPathCache:
    """
    Intelligent path caching system that prevents endless loops and reuses common paths.
    """
    
    def __init__(self):
        self.path_cache = {}  # {start: {end: [all_paths]}}
        self.sub_path_cache = {}  # {path_segment: [extensions]}
        self.visited_combinations = set()  # Prevent infinite loops
        self.max_paths_per_pair = 1000  # Limit paths per start-end pair
        self.max_total_paths = 10000  # Global path limit
        
    def cache_path(self, start: str, end: str, paths: List[List[str]]):
        """Cache paths with limits to prevent memory explosion."""
        if start not in self.path_cache:
            self.path_cache[start] = {}
        
        # Limit paths per pair to prevent endless loops
        if len(paths) > self.max_paths_per_pair:
            paths = paths[:self.max_paths_per_pair]
            print(f"      ⚠️  Limited paths from {start} to {end} to {self.max_paths_per_pair}")
        
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
            return self.path_cache[start][end]
        return []
    
    def check_path_reuse(self, current_path: List[str], target: str) -> Optional[List[str]]:
        """Check if we can reuse cached paths to reach target."""
        for i in range(len(current_path)):
            sub_path = tuple(current_path[i:])
            if sub_path in self.sub_path_cache:
                for path_info in self.sub_path_cache[sub_path]:
                    cached_path = path_info['full_path']
                    start_index = path_info['start_index']
                    
                    if cached_path[-1] == target:
                        reuse_start = start_index + len(sub_path)
                        reusable_part = cached_path[reuse_start:]
                        if reusable_part:
                            return reusable_part
        return None


class SmartRouteFinder:
    """
    Smart route finder that discovers every possible route efficiently.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.path_cache = SmartPathCache()
        self.route_count = 0
        
    def find_all_routes(self, start: str, target: str, max_hops: int = 8) -> Dict[str, Any]:
        """Find every possible route from start to target efficiently."""
        
        # Check cache first
        if self.path_cache.is_path_cached(start, target):
            cached_paths = self.path_cache.get_cached_paths(start, target)
            return {
                'all_routes': cached_paths,
                'total_routes': len(cached_paths),
                'status': 'from_cache',
                'optimization_note': f'Retrieved {len(cached_paths)} routes from cache'
            }
        
        # Get graph structure
        graph = self._get_graph_structure()
        if not graph or start not in graph or target not in graph:
            return {
                'all_routes': [],
                'total_routes': 0,
                'status': 'not_found',
                'error': 'Start or target not in graph'
            }
        
        print(f"      🧠 Finding ALL routes from {start} to {target} (max {max_hops} hops)")
        
        # Use smart BFS with early termination
        all_routes = self._smart_bfs_all_routes(graph, start, target, max_hops)
        
        # Cache results
        self.path_cache.cache_path(start, target, all_routes)
        
        # Sort by length (shortest first)
        all_routes.sort(key=lambda x: len(x))
        
        return {
            'all_routes': all_routes,
            'total_routes': len(all_routes),
            'status': 'found',
            'shortest_route': all_routes[0] if all_routes else [],
            'shortest_distance': len(all_routes[0]) - 1 if all_routes else -1,
            'optimization_note': f'Found {len(all_routes)} routes using smart BFS'
        }
    
    def _get_graph_structure(self) -> Dict[str, Set[str]]:
        """Get graph structure from Neo4j efficiently."""
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
    
    def _smart_bfs_all_routes(self, graph: Dict[str, Set[str]], start: str, target: str, max_hops: int) -> List[List[str]]:
        """Smart BFS that finds all routes with early termination."""
        
        if start == target:
            return [[start]]
        
        # Queue: (current_url, path, hops)
        queue = deque([(start, [start], 0)])
        all_routes = []
        visited_paths = set()
        
        # Progress tracking
        routes_found = 0
        search_iterations = 0
        max_iterations = 50000  # Prevent infinite loops
        
        while queue and search_iterations < max_iterations:
            current, path, hops = queue.popleft()
            search_iterations += 1
            
            # Progress indicator
            if search_iterations % 1000 == 0:
                print(f"         ⏳ Search: {search_iterations} iterations, {routes_found} routes found")
            
            # Check if we reached target
            if current == target and len(path) > 1:
                path_tuple = tuple(path)
                if path_tuple not in visited_paths:
                    all_routes.append(path.copy())
                    visited_paths.add(path_tuple)
                    routes_found += 1
                    
                    # Show progress
                    if routes_found % 10 == 0:
                        print(f"         🎯 Found {routes_found} routes so far...")
            
            # Continue exploring if we haven't reached max hops
            if hops < max_hops:
                for neighbor in graph.get(current, []):
                    if neighbor not in path:  # Avoid cycles
                        new_path = path + [neighbor]
                        
                        # Check for path reuse opportunity
                        reused_path = self.path_cache.check_path_reuse(new_path, target)
                        if reused_path:
                            # We can reuse part of this path!
                            combined_path = new_path + reused_path
                            path_tuple = tuple(combined_path)
                            if path_tuple not in visited_paths:
                                all_routes.append(combined_path)
                                visited_paths.add(path_tuple)
                                routes_found += 1
                                print(f"         🔄 Path reuse: Combined {len(new_path)} + {len(reused_path)} = {len(combined_path)} nodes")
                        else:
                            # Continue normal BFS
                            queue.append((neighbor, new_path, hops + 1))
            
            # Early termination if we have too many routes
            if len(all_routes) >= self.path_cache.max_total_paths:
                print(f"         ⚠️  Early termination: Reached max total paths limit ({self.path_cache.max_total_paths})")
                break
        
        if search_iterations >= max_iterations:
            print(f"         ⚠️  Search terminated: Reached max iterations ({max_iterations})")
        
        print(f"         ✅ Smart BFS complete: {routes_found} routes found in {search_iterations} iterations")
        return all_routes


class SimpleNavigationPlanner(Module):
    """
    Simple navigation planner that creates hierarchical plans.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, landing_url: str, neighbors: List[Dict], max_pages: int) -> Dict[str, Any]:
        """Generate navigation plan starting from landing page."""
        
        # Create hierarchical plan
        navigation_plan = make_hierarchical_plan_from_neighbors(neighbors, landing_url)
        
        # Limit to max_pages
        if len(navigation_plan) > max_pages:
            navigation_plan = navigation_plan[:max_pages]
        
        return {
            'navigation_plan': navigation_plan,
            'planned_pages': len(navigation_plan)
        }


class SmartPlanningAgent(Module):
    """
    Smart Planning Agent that finds EVERY possible route efficiently.
    """
    
    def __init__(self, neo4j_client: Neo4jClient):
        super().__init__()
        self.neo4j = neo4j_client
        self.navigation_planner = SimpleNavigationPlanner()
        self.route_finder = SmartRouteFinder(neo4j_client)
        
        # Track state
        self.current_position = None
        self.navigation_history = []
        self.discovered_routes = {}
    
    def forward(self, landing_url: str, max_pages: int = 50, exploration_strategy: str = "hierarchical") -> Dict[str, Any]:
        """Execute smart navigation planning that finds every possible route."""
        
        start_time = time.time()
        self.current_position = landing_url
        
        print(f"🧭 Starting Smart Navigation Planning from: {landing_url}")
        print(f"📊 Target: {max_pages} pages | Strategy: {exploration_strategy}")
        print("🧠 Agent will find EVERY possible route using smart path caching!")
        print("-" * 80)
        
        # Phase 1: Get initial neighbors and create navigation plan
        print(f"\n📋 PHASE 1: Creating Navigation Plan")
        print("-" * 40)
        
        initial_neighbors = self._get_initial_neighbors(landing_url, 100)
        
        if not initial_neighbors:
            return {
                'error': 'No neighbors found for landing page',
                'landing_url': landing_url
            }
        
        # Create navigation plan
        plan_result = self.navigation_planner.forward(
            landing_url=landing_url,
            neighbors=initial_neighbors,
            max_pages=max_pages
        )
        
        navigation_plan = plan_result['navigation_plan']
        print(f"🗺️  Created navigation plan with {len(navigation_plan)} URLs")
        
        # Phase 2: Find every possible route for each target
        print(f"\n🧠 PHASE 2: Finding EVERY Possible Route")
        print("-" * 60)
        
        route_discovery_results = []
        total_routes_found = 0
        
        for i, target_url in enumerate(navigation_plan[:max_pages]):
            print(f"\n🧭 ROUTE DISCOVERY [{i+1}/{len(navigation_plan)}]: {target_url}")
            print("-" * 60)
            
            # Find every possible route from landing to target
            route_result = self.route_finder.find_all_routes(
                start=landing_url,
                target=target_url,
                max_hops=8
            )
            
            # Store discovered routes
            if route_result['all_routes']:
                self.discovered_routes[target_url] = route_result['all_routes']
                total_routes_found += route_result['total_routes']
            
            # Show route summary
            all_routes = route_result.get('all_routes', [])
            shortest_route = route_result.get('shortest_route', [])
            total_routes = route_result.get('total_routes', 0)
            
            print(f"   🎯 Target: {target_url}")
            print(f"   🏆 SHORTEST ROUTE ({len(shortest_route)-1} hops): {' → '.join(shortest_route)}")
            print(f"   📊 Total Routes Found: {total_routes}")
            print(f"   ⚡ {route_result.get('optimization_note', '')}")
            
            # Show ALL routes (shortest to longest)
            if all_routes:
                print(f"   🗺️  EVERY POSSIBLE ROUTE (shortest to longest):")
                for j, route in enumerate(all_routes[:20]):  # Show first 20 routes
                    route_str = " → ".join(route)
                    hops = len(route) - 1
                    print(f"      {j+1}. ({hops} hops): {route_str}")
                
                if len(all_routes) > 20:
                    print(f"      ... and {len(all_routes) - 20} more routes")
            
            # Compile result
            route_discovery = {
                'target_url': target_url,
                'route_result': route_result,
                'discovery_order': i + 1
            }
            route_discovery_results.append(route_discovery)
            
            # Update current position
            self.current_position = target_url
            self.navigation_history.append({
                'url': target_url,
                'timestamp': time.time(),
                'order': i + 1
            })
        
        # Phase 3: Generate comprehensive route report
        print(f"\n📊 PHASE 3: Route Analysis")
        print("-" * 40)
        
        navigation_duration = time.time() - start_time
        
        # Analyze route distribution
        route_distribution = self._analyze_route_distribution()
        
        route_analysis = {
            'navigation_efficiency': {
                'total_pages_planned': len(navigation_plan),
                'total_routes_discovered': total_routes_found,
                'avg_routes_per_target': total_routes_found / len(route_discovery_results) if route_discovery_results else 0,
                'navigation_coverage': len(route_discovery_results) / len(navigation_plan) if navigation_plan else 0
            },
            'route_optimization': {
                'shortest_routes': {url: len(routes[0]) if routes else 0 for url, routes in list(self.discovered_routes.items())[:5]},
                'route_distribution': route_distribution
            },
            'smart_path_cache_stats': {
                'cached_paths': len(self.route_finder.path_cache.path_cache),
                'sub_paths': len(self.route_finder.path_cache.sub_path_cache)
            }
        }
        
        print(f"📈 ROUTE DISCOVERY SUMMARY:")
        print(f"   Pages Planned: {route_analysis['navigation_efficiency']['total_pages_planned']}")
        print(f"   Routes Discovered: {route_analysis['navigation_efficiency']['total_routes_discovered']}")
        print(f"   Avg Routes per Target: {route_analysis['navigation_efficiency']['avg_routes_per_target']:.1f}")
        print(f"   Navigation Coverage: {route_analysis['navigation_efficiency']['navigation_coverage']:.1%}")
        
        # Show route distribution
        print(f"\n🗺️  ROUTE DISTRIBUTION:")
        for hop_count, count in route_distribution.items():
            print(f"   {hop_count}: {count} routes")
        
        # Compile final result
        final_result = {
            'landing_url': landing_url,
            'navigation_plan': navigation_plan,
            'route_discovery_results': route_discovery_results,
            'route_analysis': route_analysis,
            'discovered_routes': self.discovered_routes,
            'navigation_history': self.navigation_history,
            'metadata': {
                'max_pages': max_pages,
                'exploration_strategy': exploration_strategy,
                'navigation_duration': round(navigation_duration, 2),
                'total_routes_found': total_routes_found
            }
        }
        
        print(f"\n✅ Smart Navigation Planning completed successfully!")
        print(f"⏱️  Duration: {navigation_duration:.2f}s")
        print(f"🎯 Explored: {len(route_discovery_results)} pages")
        print(f"🗺️  Total Routes: {total_routes_found} discovered")
        print(f"🚀 Smart Path Caching: {route_analysis['smart_path_cache_stats']['cached_paths']} paths cached for efficiency!")
        
        return final_result
    
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
    
    def _analyze_route_distribution(self) -> Dict[str, int]:
        """Analyze distribution of route lengths."""
        route_lengths = []
        for routes in self.discovered_routes.values():
            for route in routes:
                route_lengths.append(len(route) - 1)
        
        distribution = defaultdict(int)
        for length in route_lengths:
            distribution[f"{length}_hops"] += 1
        
        return dict(distribution)
