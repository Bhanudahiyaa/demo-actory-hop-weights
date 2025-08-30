#!/usr/bin/env python3
"""
Speed Benchmark Script - Compare Old vs Ultra-Fast Performance
Tests the speed improvements of the new ultra-fast planning agent.
"""

import time
import asyncio
import sys
from typing import Dict, Any

# Add project to path
sys.path.append('.')

async def benchmark_old_vs_ultra_fast(url: str, max_pages: int = 20):
    """
    Benchmark old smart planning vs new ultra-fast planning.
    """
    
    print("🏁 SPEED BENCHMARK: Old vs Ultra-Fast Planning")
    print("=" * 80)
    print(f"🎯 Target URL: {url}")
    print(f"📊 Max Pages: {max_pages}")
    print("=" * 80)
    
    try:
        from scraper.neo4j_client import Neo4jClient
        from scraper.smart_planning_agent import SmartPlanningAgent
        from scraper.ultra_fast_planning_agent import UltraFastPlanningAgent
        
        # Initialize Neo4j
        neo4j_client = Neo4jClient()
        
        # Test 1: Old Smart Planning Agent
        print(f"\n🧠 TEST 1: Old Smart Planning Agent")
        print("-" * 50)
        
        old_agent = SmartPlanningAgent(neo4j_client)
        old_start = time.time()
        
        old_result = old_agent.forward(
            landing_url=url,
            max_pages=max_pages,
            exploration_strategy="hierarchical"
        )
        
        old_duration = time.time() - old_start
        old_routes = old_result.get('metadata', {}).get('total_routes_found', 0)
        old_speed = old_routes / old_duration if old_duration > 0 else 0
        
        print(f"⏱️  Duration: {old_duration:.2f}s")
        print(f"🗺️  Routes Found: {old_routes:,}")
        print(f"🚀 Speed: {old_speed:.0f} routes/second")
        
        # Test 2: New Ultra-Fast Planning Agent
        print(f"\n⚡ TEST 2: Ultra-Fast Planning Agent")
        print("-" * 50)
        
        ultra_agent = UltraFastPlanningAgent(neo4j_client)
        ultra_start = time.time()
        
        ultra_result = ultra_agent.forward(
            landing_url=url,
            max_pages=max_pages,
            exploration_strategy="hierarchical"
        )
        
        ultra_duration = time.time() - ultra_start
        ultra_routes = ultra_result.get('metadata', {}).get('total_routes_found', 0)
        ultra_speed = ultra_routes / ultra_duration if ultra_duration > 0 else 0
        
        print(f"⏱️  Duration: {ultra_duration:.2f}s")
        print(f"🗺️  Routes Found: {ultra_routes:,}")
        print(f"🚀 Speed: {ultra_speed:.0f} routes/second")
        
        # Performance Comparison
        print(f"\n📊 PERFORMANCE COMPARISON")
        print("=" * 50)
        
        if old_duration > 0 and ultra_duration > 0:
            speed_improvement = old_duration / ultra_duration
            efficiency_gain = (ultra_speed / old_speed) if old_speed > 0 else float('inf')
            
            print(f"⚡ Speed Improvement: {speed_improvement:.1f}x faster")
            print(f"🚀 Efficiency Gain: {efficiency_gain:.1f}x more routes/second")
            print(f"⏱️  Time Saved: {old_duration - ultra_duration:.2f}s")
            
            if speed_improvement >= 2:
                print(f"🎉 MASSIVE IMPROVEMENT! Ultra-fast is {speed_improvement:.1f}x faster!")
            elif speed_improvement >= 1.5:
                print(f"🚀 SIGNIFICANT IMPROVEMENT! Ultra-fast is {speed_improvement:.1f}x faster!")
            else:
                print(f"📈 MODERATE IMPROVEMENT! Ultra-fast is {speed_improvement:.1f}x faster!")
        
        # Route Quality Comparison
        print(f"\n🎯 ROUTE QUALITY COMPARISON")
        print("-" * 50)
        
        old_plan = old_result.get('navigation_plan', [])
        ultra_plan = ultra_result.get('navigation_plan', [])
        
        print(f"📋 Navigation Plan Size:")
        print(f"   Old Agent: {len(old_plan)} URLs")
        print(f"   Ultra-Fast: {len(ultra_plan)} URLs")
        
        print(f"\n🗺️  Route Discovery:")
        print(f"   Old Agent: {old_routes:,} routes")
        print(f"   Ultra-Fast: {ultra_routes:,} routes")
        
        # Cache Performance (if available)
        if 'performance_analysis' in ultra_result:
            cache_stats = ultra_result['performance_analysis'].get('cache_performance', {})
            if cache_stats:
                print(f"\n💾 Cache Performance:")
                print(f"   Hit Rate: {cache_stats.get('hit_rate', 0):.1%}")
                print(f"   Cache Hits: {cache_stats.get('hits', 0)}")
                print(f"   Cache Misses: {cache_stats.get('misses', 0)}")
        
        return {
            'old_performance': {
                'duration': old_duration,
                'routes': old_routes,
                'speed': old_speed
            },
            'ultra_performance': {
                'duration': ultra_duration,
                'routes': ultra_routes,
                'speed': ultra_speed
            },
            'improvement': {
                'speed_improvement': old_duration / ultra_duration if ultra_duration > 0 else 0,
                'efficiency_gain': (ultra_speed / old_speed) if old_speed > 0 else 0
            }
        }
        
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return None


def main():
    """Main benchmark function."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Speed Benchmark: Old vs Ultra-Fast Planning")
    parser.add_argument("url", help="URL to benchmark")
    parser.add_argument("-p", "--pages", type=int, default=20, help="Maximum pages to test (default: 20)")
    parser.add_argument("--clear-db", action="store_true", help="Clear Neo4j database before testing")
    
    args = parser.parse_args()
    
    # Clear database if requested
    if args.clear_db:
        print("🗑️  Clearing Neo4j database for clean benchmark...")
        try:
            from scraper.neo4j_client import Neo4jClient
            neo4j_client = Neo4jClient()
            neo4j_client.clear_database()
            print("✅ Database cleared successfully")
        except Exception as e:
            print(f"⚠️  Database clear failed: {e}")
    
    # Run benchmark
    print(f"🚀 Starting speed benchmark with {args.pages} pages...")
    result = asyncio.run(benchmark_old_vs_ultra_fast(args.url, args.pages))
    
    if result:
        print(f"\n🎯 Benchmark completed successfully!")
        print(f"📊 Check the results above for performance comparison.")
    else:
        print(f"\n❌ Benchmark failed. Check the logs above for details.")


if __name__ == "__main__":
    main()
