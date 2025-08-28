#!/usr/bin/env python3
"""
DSPy Website Analysis Agent - Works with existing crawl4ai infrastructure
This agent analyzes complete websites crawled by crawl4ai and creates intelligent plans.
"""

import time
from typing import List, Dict, Any, Set
from urllib.parse import urlparse
from scraper.neo4j_client import Neo4jClient
from scraper.plan import make_plan_from_neighbors, analyze_plan_quality


class WebsiteAnalysisAgent:
    """DSPy-style agent that analyzes complete websites and creates intelligent plans."""
    
    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j = neo4j_client
        self.analysis_results = {}
        self.target_domain = ""
        
    def analyze_complete_website(self, start_url: str) -> Dict[str, Any]:
        """Analyze the complete website and create intelligent plans."""
        print(f"🔍 WEBSITE ANALYSIS AGENT")
        print(f"🎯 Analyzing website: {start_url}")
        print("=" * 80)
        
        # Set target domain for filtering
        self.target_domain = urlparse(start_url).netloc
        print(f"🌐 Target Domain: {self.target_domain}")
        print()
        
        # Get complete website structure (filtered by domain)
        website_structure = self._get_complete_website_structure()
        
        if not website_structure['pages']:
            print(f"❌ No pages found for domain: {self.target_domain}")
            print("💡 This domain may not have been crawled yet, or has no traditional navigation links.")
            return {}
        
        print(f"📊 Website Structure Found for {self.target_domain}:")
        print(f"   Total Pages: {len(website_structure['pages'])}")
        print(f"   Total Relationships: {len(website_structure['relationships'])}")
        print(f"   Total Features: {website_structure['total_features']}")
        print()
        
        # Analyze 1-hop neighborhoods for target domain pages only
        neighborhood_analysis = self._analyze_all_neighborhoods(website_structure['pages'])
        
        # Create priority-based plans for target domain only
        priority_plans = self._create_priority_based_plans(neighborhood_analysis)
        
        # Generate navigation flow for target domain only
        navigation_flow = self._generate_navigation_flow(priority_plans)
        
        # Final analysis
        self.analysis_results = {
            'website_structure': website_structure,
            'neighborhood_analysis': neighborhood_analysis,
            'priority_plans': priority_plans,
            'navigation_flow': navigation_flow,
            'summary': self._create_analysis_summary(website_structure, neighborhood_analysis, priority_plans)
        }
        
        # Display results
        self._display_analysis_results()
        
        return self.analysis_results
    
    def _get_complete_website_structure(self) -> Dict[str, Any]:
        """Get complete website structure from Neo4j, filtered by target domain."""
        with self.neo4j._driver.session() as session:
            # Get all pages for target domain only
            pages_result = session.run("""
                MATCH (p:Page)
                WHERE p.url CONTAINS $domain
                RETURN p.url AS url, p.title AS title, p.weight AS weight, 
                       p.visited AS visited, p.lastCrawledAt AS lastCrawledAt
                ORDER BY p.weight DESC
            """, domain=self.target_domain)
            pages = [record.data() for record in pages_result]
            
            # Get all relationships within target domain only
            relationships_result = session.run("""
                MATCH (from:Page)-[:LINKS_TO]->(to:Page)
                WHERE from.url CONTAINS $domain AND to.url CONTAINS $domain
                RETURN from.url AS from_url, to.url AS to_url
            """, domain=self.target_domain)
            relationships = [record.data() for record in relationships_result]
            
            # Get total feature counts for target domain only
            features_result = session.run("""
                MATCH (p:Page)
                WHERE p.url CONTAINS $domain
                OPTIONAL MATCH (p)-[:HAS_LINK]->(l:Link)
                OPTIONAL MATCH (p)-[:HAS_BUTTON]->(b:Button)
                OPTIONAL MATCH (p)-[:HAS_FORM]->(f:Form)
                OPTIONAL MATCH (p)-[:HAS_IMAGE]->(img:Image)
                OPTIONAL MATCH (p)-[:USES_SCRIPT]->(s:Script)
                RETURN count(DISTINCT p) AS total_pages,
                       count(DISTINCT l) AS total_links,
                       count(DISTINCT b) AS total_buttons,
                       count(DISTINCT f) AS total_forms,
                       count(DISTINCT img) AS total_images,
                       count(DISTINCT s) AS total_scripts
            """, domain=self.target_domain)
            features = features_result.single().data()
            
            return {
                'pages': pages,
                'relationships': relationships,
                'total_features': features
            }
    
    def _analyze_all_neighborhoods(self, pages: List[Dict]) -> Dict[str, Any]:
        """Analyze 1-hop neighborhoods for target domain pages only."""
        print(f"🔍 Analyzing 1-hop neighborhoods for {self.target_domain}...")
        
        neighborhood_data = {}
        total_neighbors = 0
        
        for page in pages:
            page_url = page['url']
            
            # Get outbound neighbors (pages this page links to) - only within same domain
            outbound_neighbors = self._get_domain_neighbors(page_url, "out")
            
            # Get inbound neighbors (pages that link to this page) - only within same domain
            inbound_neighbors = self._get_domain_neighbors(page_url, "in")
            
            # Calculate neighborhood metrics
            outbound_count = len(outbound_neighbors)
            inbound_count = len(inbound_neighbors)
            total_neighbor_count = outbound_count + inbound_count
            
            # Calculate dynamic complexity score
            dynamic_score = page.get('weight', 0.0)
            
            neighborhood_data[page_url] = {
                'url': page_url,
                'title': page.get('title', ''),
                'dynamic_score': dynamic_score,
                'outbound_neighbors': outbound_neighbors,
                'inbound_neighbors': inbound_neighbors,
                'outbound_count': outbound_count,
                'inbound_count': inbound_count,
                'total_neighbor_count': total_neighbor_count,
                'neighbor_ratio': total_neighbor_count / max(len(pages), 1),  # Normalized ratio
                'importance_score': (total_neighbor_count * 0.6) + (dynamic_score * 0.4)  # Combined score
            }
            
            total_neighbors += total_neighbor_count
            
            print(f"   📄 {page_url}")
            print(f"      🔗 Outbound: {outbound_count} | Inbound: {inbound_count} | Total: {total_neighbor_count}")
            print(f"      ⚖️  Dynamic Score: {dynamic_score:.2f} | Importance: {neighborhood_data[page_url]['importance_score']:.2f}")
        
        print(f"\n📊 Neighborhood Analysis Complete for {self.target_domain}:")
        print(f"   Total Pages Analyzed: {len(pages)}")
        print(f"   Total Neighbor Relationships: {total_neighbors}")
        if len(pages) > 0:
            print(f"   Average Neighbors per Page: {total_neighbors / len(pages):.1f}")
        
        return neighborhood_data
    
    def _get_domain_neighbors(self, url: str, direction: str) -> List[Dict]:
        """Get neighbors within the same domain only."""
        with self.neo4j._driver.session() as session:
            if direction == "out":
                query = """
                    MATCH (start:Page {url: $url})-[:LINKS_TO]->(p:Page)
                    WHERE p.url CONTAINS $domain
                    RETURN p.url AS url, 
                           coalesce(p.title, '') AS title,
                           coalesce(p.weight, 0.0) AS weight,
                           coalesce(p.visited, false) AS visited,
                           p.lastCrawledAt AS lastCrawledAt
                    ORDER BY p.weight DESC
                """
            else:  # inbound
                query = """
                    MATCH (p:Page)-[:LINKS_TO]->(start:Page {url: $url})
                    WHERE p.url CONTAINS $domain
                    RETURN p.url AS url, 
                           coalesce(p.title, '') AS title,
                           coalesce(p.weight, 0.0) AS weight,
                           coalesce(p.visited, false) AS visited,
                           p.lastCrawledAt AS lastCrawledAt
                    ORDER BY p.weight DESC
                """
            
            result = session.run(query, url=url, domain=self.target_domain)
            neighbors = [record.data() for record in result]
            
            # Enrich with feature counts
            enriched_neighbors = []
            for neighbor in neighbors:
                neighbor_url = neighbor['url']
                feature_counts = self._get_feature_counts(neighbor_url)
                enriched_neighbors.append({
                    **neighbor,
                    **feature_counts
                })
            
            return enriched_neighbors
    
    def _get_feature_counts(self, url: str) -> Dict[str, int]:
        """Get feature counts for a specific page."""
        with self.neo4j._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Page {url: $url})
                OPTIONAL MATCH (p)-[:HAS_FORM]->(form:Form)
                OPTIONAL MATCH (p)-[:HAS_BUTTON]->(button:Button)
                OPTIONAL MATCH (p)-[:HAS_INPUT]->(input:InputField)
                OPTIONAL MATCH (p)-[:HAS_IMAGE]->(image:Image)
                OPTIONAL MATCH (p)-[:USES_SCRIPT]->(script:Script)
                OPTIONAL MATCH (p)-[:HAS_LINK]->(link:Link)
                OPTIONAL MATCH (p)-[:USES_STYLESHEET]->(stylesheet:Stylesheet)
                OPTIONAL MATCH (p)-[:USES_MEDIA]->(media:MediaResource)
                RETURN count(DISTINCT form) AS formCount,
                       count(DISTINCT button) AS buttonCount,
                       count(DISTINCT input) AS inputCount,
                       count(DISTINCT image) AS imageCount,
                       count(DISTINCT script) AS scriptCount,
                       count(DISTINCT link) AS linkCount,
                       count(DISTINCT stylesheet) AS stylesheetCount,
                       count(DISTINCT media) AS mediaCount
                """,
                url=url
            )
            record = result.single()
            if record:
                return record.data()
            return {
                'formCount': 0, 'buttonCount': 0, 'inputCount': 0, 'imageCount': 0,
                'scriptCount': 0, 'linkCount': 0, 'stylesheetCount': 0, 'mediaCount': 0
            }
    
    def _create_priority_based_plans(self, neighborhood_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create priority-based plans for the target domain only."""
        print(f"\n📋 Creating priority-based plans for {self.target_domain}...")
        
        # Sort pages by importance score
        sorted_pages = sorted(
            neighborhood_analysis.values(),
            key=lambda x: x['importance_score'],
            reverse=True
        )
        
        # Create plans for top pages
        plans = {}
        for i, page in enumerate(sorted_pages[:10]):  # Top 10 pages
            page_url = page['url']
            
            # Get all neighbors for this page
            all_neighbors = page['outbound_neighbors'] + page['inbound_neighbors']
            
            if all_neighbors:
                # Create crawl plan using existing plan.py
                crawl_plan = make_plan_from_neighbors(all_neighbors)
                plan_analysis = analyze_plan_quality(all_neighbors, crawl_plan)
                
                plans[page_url] = {
                    'page_url': page_url,
                    'title': page['title'],
                    'importance_rank': i + 1,
                    'importance_score': page['importance_score'],
                    'dynamic_score': page['dynamic_score'],
                    'neighbor_count': page['total_neighbor_count'],
                    'crawl_plan': crawl_plan,
                    'plan_analysis': plan_analysis,
                    'priority_level': self._get_priority_level(page['importance_score'])
                }
                
                print(f"   📋 Plan {i+1}: {page_url}")
                print(f"      🏆 Rank: {i+1} | Priority: {plans[page_url]['priority_level']}")
                print(f"      🔗 Neighbors: {page['total_neighbor_count']} | Score: {page['importance_score']:.2f}")
                print(f"      📋 Planned URLs: {len(crawl_plan)}")
        
        return plans
    
    def _get_priority_level(self, importance_score: float) -> str:
        """Determine priority level based on importance score."""
        if importance_score >= 15.0:
            return "🔥 CRITICAL"
        elif importance_score >= 10.0:
            return "⚡ HIGH"
        elif importance_score >= 5.0:
            return "📈 MEDIUM"
        else:
            return "📊 LOW"
    
    def _generate_navigation_flow(self, priority_plans: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate navigation flow based on priority plans."""
        print(f"\n🗺️  Generating navigation flow for {self.target_domain}...")
        
        # Sort plans by importance rank
        sorted_plans = sorted(priority_plans.values(), key=lambda x: x['importance_rank'])
        
        navigation_flow = []
        for i, plan in enumerate(sorted_plans):
            if i == 0:
                # First page - starting point
                navigation_flow.append({
                    'step': i + 1,
                    'action': 'START',
                    'url': plan['page_url'],
                    'title': plan['title'],
                    'priority': plan['priority_level'],
                    'reason': f"Starting point - {plan['priority_level']} priority page"
                })
            else:
                # Subsequent pages - navigation steps
                navigation_flow.append({
                    'step': i + 1,
                    'action': 'GO TO',
                    'url': plan['page_url'],
                    'title': plan['title'],
                    'priority': plan['priority_level'],
                    'reason': f"Navigate to {plan['priority_level']} priority page"
                })
            
            # Add planned destinations for this page
            for j, planned_url in enumerate(plan['crawl_plan'][:5]):  # Show first 5 planned destinations
                navigation_flow.append({
                    'step': f"{i+1}.{j+1}",
                    'action': 'EXPLORE',
                    'url': planned_url,
                    'title': f"Neighbor of {plan['title']}",
                    'priority': '🔍 EXPLORE',
                    'reason': f"1-hop neighbor of {plan['title']}"
                })
        
        return navigation_flow
    
    def _create_analysis_summary(self, website_structure: Dict, neighborhood_analysis: Dict, priority_plans: Dict) -> Dict[str, Any]:
        """Create comprehensive analysis summary for target domain only."""
        pages = website_structure['pages']
        features = website_structure['total_features']
        
        # Calculate summary metrics
        total_pages = len(pages)
        total_relationships = len(website_structure['relationships'])
        
        # Find most important pages
        most_important_pages = sorted(
            neighborhood_analysis.values(),
            key=lambda x: x['importance_score'],
            reverse=True
        )[:5]
        
        # Calculate average metrics
        avg_dynamic_score = sum(p.get('weight', 0) for p in pages) / total_pages if total_pages > 0 else 0
        avg_neighbors = sum(p['total_neighbor_count'] for p in neighborhood_analysis.values()) / total_pages if total_pages > 0 else 0
        
        return {
            'target_domain': self.target_domain,
            'total_pages': total_pages,
            'total_relationships': total_relationships,
            'total_features': features,
            'avg_dynamic_score': avg_dynamic_score,
            'avg_neighbors_per_page': avg_neighbors,
            'most_important_pages': most_important_pages,
            'priority_distribution': {
                'critical': len([p for p in priority_plans.values() if p['priority_level'] == '🔥 CRITICAL']),
                'high': len([p for p in priority_plans.values() if p['priority_level'] == '⚡ HIGH']),
                'medium': len([p for p in priority_plans.values() if p['priority_level'] == '📈 MEDIUM']),
                'low': len([p for p in priority_plans.values() if p['priority_level'] == '📊 LOW'])
            }
        }
    
    def _display_analysis_results(self):
        """Display comprehensive analysis results for target domain only."""
        summary = self.analysis_results['summary']
        priority_plans = self.analysis_results['priority_plans']
        navigation_flow = self.analysis_results['navigation_flow']
        
        print("\n" + "="*80)
        print(f"🎯 WEBSITE ANALYSIS COMPLETE FOR {summary['target_domain'].upper()}")
        print("="*80)
        
        # Summary
        print(f"📊 WEBSITE SUMMARY:")
        print(f"   Target Domain: {summary['target_domain']}")
        print(f"   Total Pages: {summary['total_pages']}")
        print(f"   Total Relationships: {summary['total_relationships']}")
        print(f"   Average Dynamic Score: {summary['avg_dynamic_score']:.2f}")
        print(f"   Average Neighbors per Page: {summary['avg_neighbors_per_page']:.1f}")
        
        # Feature counts
        features = summary['total_features']
        print(f"\n🔧 FEATURE COUNTS:")
        print(f"   Links: {features['total_links']}")
        print(f"   Buttons: {features['total_buttons']}")
        print(f"   Forms: {features['total_forms']}")
        print(f"   Images: {features['total_images']}")
        print(f"   Scripts: {features['total_scripts']}")
        
        # Priority distribution
        priority_dist = summary['priority_distribution']
        print(f"\n🏆 PRIORITY DISTRIBUTION:")
        print(f"   🔥 CRITICAL: {priority_dist['critical']}")
        print(f"   ⚡ HIGH: {priority_dist['high']}")
        print(f"   📈 MEDIUM: {priority_dist['medium']}")
        print(f"   📊 LOW: {priority_dist['low']}")
        
        # Most important pages
        if summary['most_important_pages']:
            print(f"\n🏆 MOST IMPORTANT PAGES:")
            for i, page in enumerate(summary['most_important_pages'][:5], 1):
                print(f"   {i}. {page['url']}")
                print(f"      Score: {page['importance_score']:.2f} | Neighbors: {page['total_neighbor_count']} | Dynamic: {page['dynamic_score']:.2f}")
        
        # Navigation flow
        if navigation_flow:
            print(f"\n🗺️  NAVIGATION FLOW:")
            for step in navigation_flow[:15]:  # Show first 15 steps
                print(f"   {step['step']}. {step['action']}: {step['url']}")
                print(f"      Priority: {step['priority']} | Reason: {step['reason']}")
            
            if len(navigation_flow) > 15:
                print(f"   ... and {len(navigation_flow) - 15} more navigation steps")
        else:
            print(f"\n🗺️  NAVIGATION FLOW:")
            print(f"   No navigation flow generated - this domain may have limited internal linking")
        
        print("\n" + "="*80)
        print(f"✅ Analysis complete for {summary['target_domain']}!")
        print("💡 All data remains in Neo4j database for cross-website analysis")
        print("="*80)


def main():
    """Main function to run the website analysis agent."""
    import sys
    
    # Get URL from command line or use default
    if len(sys.argv) > 1:
        start_url = sys.argv[1]
    else:
        start_url = input("🌐 Enter website URL to analyze (e.g., https://example.com): ").strip()
    
    print(f"\n🔍 Starting Website Analysis for: {start_url}")
    print("📊 This will analyze the website structure from Neo4j (filtered by domain)")
    print("💡 All previous crawl data remains in the database")
    print("=" * 80)
    
    # Initialize Neo4j client
    neo4j_client = Neo4jClient()
    
    try:
        # Create and run analysis agent
        agent = WebsiteAnalysisAgent(neo4j_client)
        result = agent.analyze_complete_website(start_url)
        
        if result:
            print(f"\n🎉 WEBSITE ANALYSIS COMPLETED!")
            print(f"✅ Analyzed {result['summary']['total_pages']} pages for {result['summary']['target_domain']}")
            print(f"✅ Found {result['summary']['total_relationships']} relationships")
            print(f"✅ Created {len(result['priority_plans'])} priority plans")
            print(f"✅ Generated {len(result['navigation_flow'])} navigation steps")
        else:
            print("❌ Analysis failed. Please ensure the website has been crawled first.")
        
    except Exception as e:
        print(f"❌ Error during website analysis: {e}")
        import traceback
        traceback.print_exc()
    finally:
        neo4j_client.close()


if __name__ == "__main__":
    main()
