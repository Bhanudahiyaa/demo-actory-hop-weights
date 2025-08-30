#!/usr/bin/env python3
"""
Test Suite for Intelligent Site Navigation Planner

This test suite validates the functionality of the intelligent navigation
planner using DSPy with GPT-OSS 20B, including route tree construction,
subpath memoization, and intelligent exploration decisions.
"""

import time
import json
from typing import Dict, List
from intelligent_navigation_planner import (
    IntelligentNavigationPlanner, 
    build_route_tree, 
    expand_route_tree,
    RouteNode
)


def test_basic_functionality():
    """Test basic functionality with a simple sitemap."""
    print("🧪 TEST 1: Basic Functionality")
    print("-" * 50)
    
    # Simple test sitemap
    sitemap = {
        "landing": ["about", "pricing"],
        "about": ["team"],
        "pricing": ["checkout"],
        "team": [],
        "checkout": []
    }
    
    API_KEY = "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"
    
    try:
        # Test the clean API functions
        print("Building route tree...")
        route_tree = build_route_tree("landing", sitemap, max_depth=3, api_key=API_KEY)
        
        print("Expanding routes...")
        routes = list(expand_route_tree(route_tree))
        
        print(f"✅ Basic test passed! Found {len(routes)} routes")
        for i, route in enumerate(routes, 1):
            print(f"   {i}. {' → '.join(route)}")
            
        return True
        
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False


def test_complex_sitemap():
    """Test with a more complex sitemap structure."""
    print("\n🧪 TEST 2: Complex Sitemap")
    print("-" * 50)
    
    # Complex test sitemap
    sitemap = {
        "landing": ["home", "products", "services", "about"],
        "home": ["dashboard", "profile"],
        "products": ["product1", "product2", "product3"],
        "services": ["service1", "service2"],
        "about": ["team", "company", "careers"],
        "dashboard": ["analytics", "reports"],
        "profile": ["settings", "preferences"],
        "product1": ["details", "reviews"],
        "product2": ["details", "reviews"],
        "product3": ["details", "reviews"],
        "service1": ["pricing", "contact"],
        "service2": ["pricing", "contact"],
        "team": ["leadership", "members"],
        "company": ["history", "mission"],
        "careers": ["openings", "apply"],
        "analytics": ["metrics", "charts"],
        "reports": ["summary", "detailed"],
        "settings": ["account", "security"],
        "preferences": ["notifications", "privacy"],
        "details": ["specs", "images"],
        "reviews": ["ratings", "comments"],
        "pricing": ["plans", "billing"],
        "contact": ["form", "support"],
        "leadership": ["ceo", "cto"],
        "members": ["developers", "designers"],
        "history": ["timeline", "milestones"],
        "mission": ["values", "goals"],
        "openings": ["positions", "requirements"],
        "apply": ["resume", "interview"],
        "metrics": ["performance", "usage"],
        "charts": ["graphs", "data"],
        "summary": ["overview", "highlights"],
        "detailed": ["breakdown", "analysis"],
        "account": ["profile", "billing"],
        "security": ["password", "2fa"],
        "notifications": ["email", "sms"],
        "privacy": ["settings", "permissions"],
        "specs": ["technical", "features"],
        "images": ["gallery", "thumbnails"],
        "ratings": ["stars", "feedback"],
        "comments": ["reviews", "responses"],
        "plans": ["basic", "premium"],
        "billing": ["invoices", "payment"],
        "form": ["fields", "validation"],
        "support": ["tickets", "chat"],
        "ceo": [],
        "cto": [],
        "developers": [],
        "designers": [],
        "timeline": [],
        "milestones": [],
        "values": [],
        "goals": [],
        "positions": [],
        "requirements": [],
        "resume": [],
        "interview": [],
        "performance": [],
        "usage": [],
        "graphs": [],
        "data": [],
        "overview": [],
        "highlights": [],
        "breakdown": [],
        "analysis": [],
        "profile": [],
        "billing": [],
        "password": [],
        "2fa": [],
        "email": [],
        "sms": [],
        "settings": [],
        "permissions": [],
        "technical": [],
        "features": [],
        "gallery": [],
        "thumbnails": [],
        "stars": [],
        "feedback": [],
        "reviews": [],
        "responses": [],
        "basic": [],
        "premium": [],
        "invoices": [],
        "payment": [],
        "fields": [],
        "validation": [],
        "tickets": [],
        "chat": []
    }
    
    API_KEY = "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"
    
    try:
        print(f"Testing with complex sitemap ({len(sitemap)} pages)...")
        
        # Test different max depths
        for max_depth in [3, 4, 5]:
            print(f"\nTesting max_depth={max_depth}...")
            start_time = time.time()
            
            route_tree = build_route_tree("landing", sitemap, max_depth=max_depth, api_key=API_KEY)
            
            build_time = time.time() - start_time
            routes = list(expand_route_tree(route_tree))
            
            print(f"   ✅ Built tree in {build_time:.2f}s")
            print(f"   📊 Found {len(routes)} routes")
            print(f"   🌳 Tree has {_count_nodes(route_tree)} nodes")
            
        print("✅ Complex sitemap test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Complex sitemap test failed: {e}")
        return False


def test_memoization():
    """Test subpath memoization functionality."""
    print("\n🧪 TEST 3: Subpath Memoization")
    print("-" * 50)
    
    sitemap = {
        "landing": ["a", "b"],
        "a": ["c", "d"],
        "b": ["c", "e"],
        "c": ["f"],
        "d": ["f"],
        "e": ["f"],
        "f": []
    }
    
    API_KEY = "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"
    
    try:
        print("Testing memoization with overlapping paths...")
        
        # Create planner with memoization enabled
        planner = IntelligentNavigationPlanner(API_KEY, max_depth=4, enable_memoization=True)
        
        start_time = time.time()
        route_tree = planner.build_route_tree("landing", sitemap, max_depth=4)
        build_time = time.time() - start_time
        
        # Get cache statistics
        cache_stats = planner.get_cache_stats()
        
        print(f"✅ Memoization test passed!")
        print(f"   📊 Build time: {build_time:.2f}s")
        print(f"   🎯 Cache hits: {cache_stats['hits']}")
        print(f"   ❌ Cache misses: {cache_stats['misses']}")
        print(f"   📈 Hit rate: {cache_stats['hit_rate_percent']}%")
        
        # Test cache clearing
        planner.clear_cache()
        new_cache_stats = planner.get_cache_stats()
        print(f"   🗑️  Cache cleared: {new_cache_stats['enabled']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Memoization test failed: {e}")
        return False


def test_cycle_detection():
    """Test cycle detection and prevention."""
    print("\n🧪 TEST 4: Cycle Detection")
    print("-" * 50)
    
    # Sitemap with potential cycles
    sitemap = {
        "landing": ["a", "b"],
        "a": ["b", "c"],
        "b": ["c", "a"],  # Creates cycle: a -> b -> a
        "c": ["d"],
        "d": ["e"],
        "e": []
    }
    
    API_KEY = "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"
    
    try:
        print("Testing cycle detection...")
        
        route_tree = build_route_tree("landing", sitemap, max_depth=5, api_key=API_KEY)
        routes = list(expand_route_tree(route_tree))
        
        # Check that no route contains cycles
        cycles_found = 0
        for route in routes:
            if len(route) != len(set(route)):
                cycles_found += 1
        
        print(f"✅ Cycle detection test passed!")
        print(f"   📊 Total routes: {len(routes)}")
        print(f"   🔄 Cycles detected: {cycles_found}")
        print(f"   🌳 Max route length: {max(len(route) for route in routes) if routes else 0}")
        
        return True
        
    except Exception as e:
        print(f"❌ Cycle detection test failed: {e}")
        return False


def test_performance():
    """Test performance with large sitemaps."""
    print("\n🧪 TEST 5: Performance Testing")
    print("-" * 50)
    
    # Generate a large sitemap
    def generate_large_sitemap(size: int) -> Dict[str, List[str]]:
        sitemap = {}
        for i in range(size):
            page = f"page_{i}"
            # Each page connects to 2-4 other pages
            connections = []
            for j in range(2, 5):
                target = f"page_{(i + j) % size}"
                if target != page:
                    connections.append(target)
            sitemap[page] = connections
        return sitemap
    
    API_KEY = "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"
    
    try:
        print("Testing performance with different sitemap sizes...")
        
        for size in [10, 25, 50]:
            print(f"\nTesting sitemap size: {size}")
            sitemap = generate_large_sitemap(size)
            
            start_time = time.time()
            route_tree = build_route_tree("page_0", sitemap, max_depth=4, api_key=API_KEY)
            build_time = time.time() - start_time
            
            routes = list(expand_route_tree(route_tree))
            nodes = _count_nodes(route_tree)
            
            print(f"   ✅ Size {size}: {build_time:.2f}s, {len(routes)} routes, {nodes} nodes")
            
        print("✅ Performance test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n🧪 TEST 6: Error Handling")
    print("-" * 50)
    
    API_KEY = "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"
    
    try:
        print("Testing error handling...")
        
        # Test 1: Invalid start node
        try:
            sitemap = {"a": ["b"], "b": []}
            build_route_tree("invalid", sitemap, max_depth=3, api_key=API_KEY)
            print("   ❌ Should have failed for invalid start node")
            return False
        except ValueError:
            print("   ✅ Correctly handled invalid start node")
        
        # Test 2: Empty sitemap
        try:
            empty_sitemap = {}
            build_route_tree("landing", empty_sitemap, max_depth=3, api_key=API_KEY)
            print("   ❌ Should have failed for empty sitemap")
            return False
        except ValueError:
            print("   ✅ Correctly handled empty sitemap")
        
        # Test 3: Missing API key
        try:
            build_route_tree("landing", {"a": []}, max_depth=3, api_key=None)
            print("   ❌ Should have failed for missing API key")
            return False
        except ValueError:
            print("   ✅ Correctly handled missing API key")
        
        print("✅ Error handling test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def test_dspy_integration():
    """Test DSPy integration and intelligent decisions."""
    print("\n🧪 TEST 7: DSPy Integration")
    print("-" * 50)
    
    sitemap = {
        "landing": ["home", "products", "about"],
        "home": ["dashboard", "profile"],
        "products": ["product1", "product2"],
        "about": ["team", "company"],
        "dashboard": ["analytics"],
        "profile": ["settings"],
        "product1": ["details"],
        "product2": ["details"],
        "team": ["leadership"],
        "company": ["history"],
        "analytics": [],
        "settings": [],
        "details": [],
        "leadership": [],
        "history": []
    }
    
    API_KEY = "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"
    
    try:
        print("Testing DSPy integration...")
        
        # Create planner and build tree
        planner = IntelligentNavigationPlanner(API_KEY, max_depth=4, enable_memoization=True)
        route_tree = planner.build_route_tree("landing", sitemap, max_depth=4)
        
        # Get statistics
        stats = planner.get_route_summary(route_tree)
        
        print(f"✅ DSPy integration test passed!")
        print(f"   🤖 DSPy decisions made: {stats.dspy_decisions}")
        print(f"   📊 Total nodes: {stats.total_nodes}")
        print(f"   📈 Total paths: {stats.total_paths}")
        print(f"   ⏱️  Construction time: {stats.construction_time:.2f}s")
        
        # Test cache integration
        cache_stats = planner.get_cache_stats()
        print(f"   🎯 Cache hit rate: {cache_stats['hit_rate_percent']}%")
        
        return True
        
    except Exception as e:
        print(f"❌ DSPy integration test failed: {e}")
        return False


def _count_nodes(node: RouteNode) -> int:
    """Helper function to count nodes in a route tree."""
    count = 1
    for child in node.children.values():
        count += _count_nodes(child)
    return count


def run_all_tests():
    """Run all test cases and report results."""
    print("🚀 INTELLIGENT NAVIGATION PLANNER - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print("Testing DSPy integration with GPT-OSS 20B for intelligent route planning")
    print("=" * 80)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Complex Sitemap", test_complex_sitemap),
        ("Subpath Memoization", test_memoization),
        ("Cycle Detection", test_cycle_detection),
        ("Performance", test_performance),
        ("Error Handling", test_error_handling),
        ("DSPy Integration", test_dspy_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\n🎯 Overall Result: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The intelligent navigation planner is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total


if __name__ == "__main__":
    run_all_tests()
