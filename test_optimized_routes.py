#!/usr/bin/env python3
"""
Test script for the Optimized Route Finder

This script demonstrates the new route finding system with:
- Subpath memoization
- Route compression
- Tree-like structures
- Memory-efficient exploration
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper.optimized_route_finder import OptimizedRouteFinder, build_route_tree, expand_route_tree


def test_basic_route_discovery():
    """Test basic route discovery functionality."""
    print("🧪 Testing Basic Route Discovery")
    print("=" * 50)
    
    try:
        # Build route tree with depth 4
        route_tree = build_route_tree("https://zineps.com", max_depth=4)
        
        # Get summary using the finder's method
        from scraper.optimized_route_finder import OptimizedRouteFinder
        with OptimizedRouteFinder() as finder:
            summary = finder.get_route_summary(route_tree)
            print(f"✅ Route tree built successfully!")
            print(f"   Root: {summary['url']}")
            print(f"   Max Depth: {summary['depth']}")
            print(f"   Total Children: {summary['children_count']}")
        
        return route_tree
        
    except Exception as e:
        print(f"❌ Basic route discovery failed: {e}")
        return None


def test_route_expansion(route_tree):
    """Test route expansion functionality."""
    print("\n🧪 Testing Route Expansion")
    print("=" * 50)
    
    try:
        # Expand first 10 routes using the finder's method
        from scraper.optimized_route_finder import OptimizedRouteFinder
        with OptimizedRouteFinder() as finder:
            routes = []
            for i, route in enumerate(finder.expand_route_tree(route_tree)):
                if i >= 10:
                    break
                routes.append(route)
        
        print(f"✅ Expanded {len(routes)} routes:")
        for i, route in enumerate(routes):
            print(f"   {i+1:2d}. {' → '.join(route)}")
        
        return routes
        
    except Exception as e:
        print(f"❌ Route expansion failed: {e}")
        return None


def test_targeted_discovery():
    """Test targeted route discovery to specific pages."""
    print("\n🧪 Testing Targeted Route Discovery")
    print("=" * 50)
    
    try:
        # Focus on specific high-value pages
        target_pages = [
            "https://www.zineps.ai/pricing",
            "https://www.zineps.ai/contact",
            "https://www.zineps.ai/support"
        ]
        
        route_tree = build_route_tree(
            "https://zineps.com", 
            max_depth=5, 
            target_nodes=target_pages
        )
        
        # Get summary using the finder's method
        from scraper.optimized_route_finder import OptimizedRouteFinder
        with OptimizedRouteFinder() as finder:
            summary = finder.get_route_summary(route_tree)
            print(f"✅ Targeted route tree built successfully!")
            print(f"   Root: {summary['url']}")
            print(f"   Max Depth: {summary['depth']}")
            print(f"   Children: {summary['children_count']}")
        
        # Show target pages
        print(f"   🎯 Target Pages: {len(target_pages)}")
        for target in target_pages:
            print(f"      - {target}")
        
        return route_tree
        
    except Exception as e:
        print(f"❌ Targeted discovery failed: {e}")
        return None


def test_performance_comparison():
    """Test performance with different depths."""
    print("\n🧪 Testing Performance Comparison")
    print("=" * 50)
    
    depths = [3, 4, 5]
    results = {}
    
    for depth in depths:
        try:
            print(f"   Testing depth {depth}...")
            start_time = time.time()
            
            route_tree = build_route_tree("https://zineps.com", max_depth=depth)
            
            end_time = time.time()
            discovery_time = end_time - start_time
            
            # Count routes using the finder's method
            from scraper.optimized_route_finder import OptimizedRouteFinder
            with OptimizedRouteFinder() as finder:
                route_count = sum(1 for _ in finder.expand_route_tree(route_tree))
            
            results[depth] = {
                'time': discovery_time,
                'routes': route_count,
                'routes_per_second': route_count / discovery_time if discovery_time > 0 else 0
            }
            
            print(f"      ✅ Depth {depth}: {route_count} routes in {discovery_time:.2f}s")
            
        except Exception as e:
            print(f"      ❌ Depth {depth} failed: {e}")
            results[depth] = None
    
    # Print performance summary
    print(f"\n📊 Performance Summary:")
    print(f"   {'Depth':<6} {'Routes':<8} {'Time (s)':<10} {'Routes/s':<12}")
    print(f"   {'-'*6} {'-'*8} {'-'*10} {'-'*12}")
    
    for depth, result in results.items():
        if result:
            print(f"   {depth:<6} {result['routes']:<8} {result['time']:<10.2f} {result['routes_per_second']:<12.1f}")
        else:
            print(f"   {depth:<6} {'FAILED':<8} {'N/A':<10} {'N/A':<12}")
    
    return results


def main():
    """Main test function."""
    print("🚀 OPTIMIZED ROUTE FINDER TEST SUITE")
    print("=" * 60)
    print("This test demonstrates the new route finding system with:")
    print("• Subpath memoization prevents recalculation")
    print("• Route compression saves memory")
    print("• Tree structures enable efficient traversal")
    print("• Configurable cutoffs prevent explosion")
    print("• Clean APIs for easy integration")
    print("=" * 60)
    
    # Test 1: Basic route discovery
    route_tree = test_basic_route_discovery()
    if not route_tree:
        print("❌ Cannot continue without basic route tree")
        return
    
    # Test 2: Route expansion
    test_route_expansion(route_tree)
    
    # Test 3: Targeted discovery
    test_targeted_discovery()
    
    # Test 4: Performance comparison
    test_performance_comparison()
    
    print("\n🎉 All tests completed!")
    print("\n💡 Key Benefits of the New System:")
    print("   • Subpath memoization prevents recalculation")
    print("   • Route compression saves memory")
    print("   • Tree structures enable efficient traversal")
    print("   • Configurable cutoffs prevent explosion")
    print("   • Clean APIs for easy integration")


if __name__ == "__main__":
    import time
    main()
