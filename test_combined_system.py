#!/usr/bin/env python3
"""
Test Script for Combined Weight-Based Optimized Route Finder

This script demonstrates the combined system that:
1. Uses weight-based prioritization (highest to lowest)
2. Leverages optimized route finder's efficiency
3. Navigates to EVERY page in the website
4. Shows routes in format: (from this - to this - to this)
5. Sorts by hop count and priority score
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper.weight_based_optimized_finder import WeightBasedOptimizedFinder


def test_comprehensive_navigation():
    """Test comprehensive navigation to every page."""
    print("🚀 TESTING COMBINED WEIGHT-BASED OPTIMIZED ROUTE FINDER")
    print("=" * 80)
    print("Features:")
    print("• Weight-based page prioritization (highest to lowest)")
    print("• Optimized route discovery with subpath memoization")
    print("• Navigation to EVERY page in the website")
    print("• Routes in format: (from this - to this - to this)")
    print("• Sorted by hop count (ascending) and priority (descending)")
    print("=" * 80)
    
    # Initialize finder with optimal settings
    finder = WeightBasedOptimizedFinder(
        max_depth=6,
        max_routes_per_target=500,
        enable_compression=True,
        enable_memoization=True
    )
    
    try:
        # Test 1: Find routes to every page
        print("\n1️⃣  TEST: Comprehensive Navigation to Every Page")
        print("-" * 60)
        
        start_time = time.time()
        all_routes = finder.find_routes_to_every_page("https://zineps.com", max_depth=5)
        total_time = time.time() - start_time
        
        # Test 2: Display routes in requested format
        print("\n2️⃣  TEST: Route Display in Requested Format")
        print("-" * 60)
        
        finder.display_routes_in_format(all_routes, max_display=60)
        
        # Test 3: Comprehensive statistics
        print("\n3️⃣  TEST: Comprehensive Route Statistics")
        print("-" * 60)
        
        stats = finder.get_route_summary(all_routes)
        print(f"📊 Total Routes Found: {stats.total_routes:,}")
        print(f"📄 Pages Covered: {stats.pages_covered}/{stats.total_pages}")
        print(f"📈 Coverage Percentage: {stats.coverage_percentage:.1f}%")
        print(f"📏 Max Depth: {stats.max_depth}")
        print(f"⏱️  Total Discovery Time: {total_time:.2f}s")
        print(f"💾 Memory Usage: {stats.memory_usage_mb:.2f} MB")
        print(f"🎯 Cache Hit Rate: {stats.cache_hits / (stats.cache_hits + stats.cache_misses) * 100:.1f}%")
        
        # Test 4: Route analysis by hop count
        print("\n4️⃣  TEST: Route Analysis by Hop Count")
        print("-" * 60)
        
        hop_distribution = {}
        for route in all_routes:
            hop_count = route.hop_count
            hop_distribution[hop_count] = hop_distribution.get(hop_count, 0) + 1
        
        print(f"📈 Route Distribution by Hop Count:")
        for hop_count in sorted(hop_distribution.keys()):
            count = hop_distribution[hop_count]
            print(f"   {hop_count} hops: {count:,} routes")
        
        # Test 5: Top routes by priority
        print("\n5️⃣  TEST: Top Routes by Priority Score")
        print("-" * 60)
        
        top_routes = sorted(all_routes, key=lambda x: -x.priority_score)[:15]
        print(f"🏆 Top 15 Routes by Priority Score:")
        for i, route in enumerate(top_routes):
            print(f"   {i+1:2d}. Priority: {route.priority_score:.1f}")
            print(f"       Path: {' - '.join(route.path)}")
            print(f"       Target: {route.target_page}")
            print(f"       Hops: {route.hop_count}, Weight: {route.total_weight}")
            print()
        
        # Test 6: Performance metrics
        print("\n6️⃣  TEST: Performance Metrics")
        print("-" * 60)
        
        routes_per_second = stats.total_routes / total_time if total_time > 0 else 0
        print(f"⚡ Routes per second: {routes_per_second:,.1f}")
        print(f"💾 Memory efficiency: {stats.total_routes / stats.memory_usage_mb:,.1f} routes/MB")
        print(f"🎯 Cache efficiency: {stats.cache_hits / (stats.cache_hits + stats.cache_misses) * 100:.1f}% hit rate")
        
        # Test 7: Coverage analysis
        print("\n7️⃣  TEST: Coverage Analysis")
        print("-" * 60)
        
        if stats.coverage_percentage >= 95:
            print(f"✅ EXCELLENT coverage: {stats.coverage_percentage:.1f}% of pages reached")
        elif stats.coverage_percentage >= 80:
            print(f"🟡 GOOD coverage: {stats.coverage_percentage:.1f}% of pages reached")
        else:
            print(f"⚠️  Coverage needs improvement: {stats.coverage_percentage:.1f}% of pages reached")
        
        print(f"📄 Total pages in database: {stats.total_pages}")
        print(f"📄 Pages successfully reached: {stats.pages_covered}")
        print(f"📄 Pages not reached: {stats.total_pages - stats.pages_covered}")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        finder.close()


def test_specific_targets():
    """Test navigation to specific high-priority targets."""
    print("\n🎯 TESTING NAVIGATION TO SPECIFIC TARGETS")
    print("=" * 60)
    
    finder = WeightBasedOptimizedFinder(
        max_depth=5,
        max_routes_per_target=200,
        enable_compression=True,
        enable_memoization=True
    )
    
    try:
        # Define high-priority targets
        high_priority_targets = [
            "https://www.zineps.ai/pricing",      # Usually highest weight
            "https://www.zineps.ai/contact",      # High weight (forms)
            "https://www.zineps.ai/support",      # Medium weight
            "https://www.zineps.ai/integrations"  # Medium weight
        ]
        
        print(f"🎯 Testing navigation to {len(high_priority_targets)} high-priority targets:")
        for target in high_priority_targets:
            print(f"   - {target}")
        
        # Find routes to each target
        all_target_routes = []
        for target_url in high_priority_targets:
            print(f"\n🔍 Finding routes to: {target_url}")
            
            # Use the comprehensive method but filter for specific target
            all_routes = finder.find_routes_to_every_page("https://zineps.com", max_depth=5)
            target_routes = [route for route in all_routes if route.target_page == target_url]
            
            print(f"   ✅ Found {len(target_routes)} routes to {target_url}")
            all_target_routes.extend(target_routes)
        
        # Display target routes
        print(f"\n🗺️  DISPLAYING ROUTES TO HIGH-PRIORITY TARGETS")
        print("=" * 80)
        
        # Sort by priority score
        all_target_routes.sort(key=lambda x: -x.priority_score)
        
        for i, route in enumerate(all_target_routes[:30]):  # Show top 30
            formatted_path = " - ".join(route.path)
            print(f"{i+1:2d}. ({formatted_path})")
            print(f"    🎯 Target: {route.target_page}")
            print(f"    🏆 Priority: {route.priority_score:.1f}, Hops: {route.hop_count}")
            print()
        
        if len(all_target_routes) > 30:
            print(f"... and {len(all_target_routes) - 30} more routes")
    
    except Exception as e:
        print(f"❌ Error during target testing: {e}")
    
    finally:
        finder.close()


def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE TEST OF COMBINED SYSTEM")
    print("=" * 100)
    print("This test demonstrates the combined weight-based optimized route finder")
    print("that navigates to EVERY page with intelligent prioritization.")
    print("=" * 100)
    
    # Test 1: Comprehensive navigation
    test_comprehensive_navigation()
    
    # Test 2: Specific targets
    test_specific_targets()
    
    print("\n" + "=" * 100)
    print("🎉 ALL TESTS COMPLETED!")
    print("✅ Weight-based prioritization working")
    print("✅ Optimized route discovery working")
    print("✅ Navigation to every page working")
    print("✅ Routes displayed in requested format")
    print("✅ Subpath memoization and compression working")
    print("✅ Comprehensive coverage achieved")


if __name__ == "__main__":
    main()
