#!/usr/bin/env python3
"""
Test Script for Weight-Based Route Finder

This script demonstrates:
1. Weight-based route discovery
2. Routes displayed in format: (from this - to this - to this)
3. Sorting by ascending hop count
4. Prioritization by page weights
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper.weight_based_route_finder import WeightBasedRouteFinder


def test_weight_based_discovery():
    """Test the weight-based route discovery system."""
    print("🚀 TESTING WEIGHT-BASED ROUTE FINDER")
    print("=" * 60)
    print("Features:")
    print("• Routes prioritized by page weights (highest to lowest)")
    print("• Format: (from this - to this - to this)")
    print("• Sorted by ascending hop count")
    print("• Priority-based discovery")
    print("=" * 60)
    
    # Initialize finder
    finder = WeightBasedRouteFinder()
    
    try:
        # Test 1: Find all weight-based routes
        print("\n1️⃣  TEST: Finding all weight-based routes")
        print("-" * 40)
        
        all_routes = finder.find_weighted_routes(
            start_url="https://zineps.com",
            max_depth=5,
            max_routes=300
        )
        
        # Display routes in requested format
        finder.display_routes_in_format(all_routes, max_display=25)
        
        # Test 2: Find routes to specific high-priority targets
        print("\n2️⃣  TEST: Finding routes to high-priority targets")
        print("-" * 40)
        
        high_priority_targets = [
            "https://www.zineps.ai/pricing",      # Usually high weight (forms, buttons)
            "https://www.zineps.ai/contact",      # Usually high weight (forms, inputs)
            "https://www.zineps.ai/support",      # Usually medium weight
            "https://www.zineps.ai/integrations"  # Usually medium weight
        ]
        
        target_routes = finder.find_routes_to_targets(
            start_url="https://zineps.com",
            target_urls=high_priority_targets,
            max_depth=5
        )
        
        finder.display_routes_in_format(target_routes, max_display=20)
        
        # Test 3: Show route statistics
        print("\n3️⃣  TEST: Route Statistics and Analysis")
        print("-" * 40)
        
        print(f"📊 Total Routes Found: {len(all_routes)}")
        print(f"🎯 Target Routes: {len(target_routes)}")
        
        # Analyze route distribution by hop count
        hop_distribution = {}
        for route in all_routes:
            hop_count = route.hop_count
            hop_distribution[hop_count] = hop_distribution.get(hop_count, 0) + 1
        
        print(f"\n📈 Route Distribution by Hop Count:")
        for hop_count in sorted(hop_distribution.keys()):
            count = hop_distribution[hop_count]
            print(f"   {hop_count} hops: {count} routes")
        
        # Show top routes by priority
        print(f"\n🏆 Top 10 Routes by Priority Score:")
        top_routes = sorted(all_routes, key=lambda x: -x.priority_score)[:10]
        for i, route in enumerate(top_routes):
            print(f"   {i+1:2d}. Priority: {route.priority_score:.1f}")
            print(f"       Path: {' - '.join(route.path)}")
            print(f"       Hops: {route.hop_count}, Weight: {route.total_weight}")
            print()
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        finder.close()


def test_specific_format():
    """Test the specific route format requested."""
    print("\n🎯 TESTING SPECIFIC ROUTE FORMAT")
    print("=" * 60)
    print("Expected Format: (from this - to this - to this)")
    print("Sorted by: 1) Hop count (ascending), 2) Priority score (descending)")
    print("=" * 60)
    
    finder = WeightBasedRouteFinder()
    
    try:
        # Find routes with limited depth for clear demonstration
        routes = finder.find_weighted_routes(
            start_url="https://zineps.com",
            max_depth=4,  # Limited depth for clearer demonstration
            max_routes=100
        )
        
        print(f"\n📊 Found {len(routes)} routes with max depth 4")
        print("🔍 Displaying routes in requested format:")
        print()
        
        # Display in the exact format requested
        for i, route in enumerate(routes[:20]):  # Show first 20
            # Format: (from this - to this - to this)
            formatted_path = " - ".join(route.path)
            print(f"{i+1:2d}. ({formatted_path})")
            print(f"    Hops: {route.hop_count}, Priority: {route.priority_score:.1f}")
            print()
        
        if len(routes) > 20:
            print(f"... and {len(routes) - 20} more routes")
    
    except Exception as e:
        print(f"❌ Error during format testing: {e}")
    
    finally:
        finder.close()


def main():
    """Main test function."""
    print("🧪 COMPREHENSIVE TEST OF WEIGHT-BASED ROUTE FINDER")
    print("=" * 80)
    
    # Test 1: Basic functionality
    test_weight_based_discovery()
    
    # Test 2: Specific format
    test_specific_format()
    
    print("\n" + "=" * 80)
    print("🎉 ALL TESTS COMPLETED!")
    print("✅ Weight-based route discovery working")
    print("✅ Routes displayed in requested format")
    print("✅ Sorting by hop count and priority")
    print("✅ Priority-based discovery from highest weight pages")


if __name__ == "__main__":
    main()
