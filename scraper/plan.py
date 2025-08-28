#!/usr/bin/env python3
"""
Planning helper for DSPy Crawl Agent.
Generates ordered crawl plans from enriched neighbor data.
"""

from typing import List, Dict
from urllib.parse import urlparse


def make_plan_from_neighbors(neighbors: List[Dict]) -> List[str]:
    """
    Generate an ordered list of URLs to crawl from enriched neighbors.
    
    Args:
        neighbors: List of neighbor dictionaries with feature counts
        
    Returns:
        Ordered list of URLs to crawl, prioritized by dynamic complexity
    """
    if not neighbors:
        return []
    
    # Calculate dynamic complexity score for each neighbor
    scored_neighbors = []
    for neighbor in neighbors:
        # Dynamic complexity scoring (heavily weighted toward JS and interactivity)
        dynamic_score = (
            neighbor.get('scriptCount', 0) * 0.5 +                    # JS: 50% weight
            neighbor.get('formCount', 0) * 0.25 +                     # Forms: 25% weight
            (neighbor.get('buttonCount', 0) + neighbor.get('inputCount', 0)) * 0.15 +  # Interactive: 15%
            neighbor.get('imageCount', 0) * 0.1                       # Images: 10% weight
        )
        
        # Additional factors
        link_density = neighbor.get('linkCount', 0) / 100.0  # Normalize link count
        media_richness = neighbor.get('mediaCount', 0) / 50.0  # Normalize media count
        
        # Final score combines dynamic complexity with content richness
        final_score = dynamic_score + (link_density * 0.05) + (media_richness * 0.05)
        
        scored_neighbors.append({
            'url': neighbor['url'],
            'score': final_score,
            'dynamic_score': dynamic_score,
            'link_density': link_density,
            'media_richness': media_richness,
            'metadata': {
                'title': neighbor.get('title', ''),
                'scriptCount': neighbor.get('scriptCount', 0),
                'formCount': neighbor.get('formCount', 0),
                'buttonCount': neighbor.get('buttonCount', 0),
                'inputCount': neighbor.get('inputCount', 0),
                'imageCount': neighbor.get('imageCount', 0),
                'linkCount': neighbor.get('linkCount', 0),
                'mediaCount': neighbor.get('mediaCount', 0)
            }
        })
    
    # Sort by final score (descending), then by URL for consistency
    scored_neighbors.sort(key=lambda x: (-x['score'], x['url']))
    
    # Extract ordered URLs
    ordered_urls = [neighbor['url'] for neighbor in scored_neighbors]
    
    return ordered_urls


def analyze_plan_quality(neighbors: List[Dict], plan: List[str]) -> Dict:
    """
    Analyze the quality and characteristics of the generated crawl plan.
    
    Args:
        neighbors: Original neighbor data
        plan: Generated crawl plan (ordered URLs)
        
    Returns:
        Dictionary with plan analysis metrics
    """
    if not plan:
        return {
            'total_urls': 0,
            'avg_dynamic_score': 0.0,
            'js_heavy_pages': 0,
            'form_rich_pages': 0,
            'interactive_pages': 0,
            'media_rich_pages': 0,
            'plan_distribution': {}
        }
    
    # Create URL to neighbor mapping
    neighbor_map = {n['url']: n for n in neighbors}
    
    # Analyze plan characteristics
    total_dynamic_score = 0.0
    js_heavy_count = 0
    form_rich_count = 0
    interactive_count = 0
    media_rich_count = 0
    
    plan_analysis = []
    
    for i, url in enumerate(plan):
        neighbor = neighbor_map.get(url, {})
        
        # Calculate scores
        dynamic_score = (
            neighbor.get('scriptCount', 0) * 0.5 +
            neighbor.get('formCount', 0) * 0.25 +
            (neighbor.get('buttonCount', 0) + neighbor.get('inputCount', 0)) * 0.15 +
            neighbor.get('imageCount', 0) * 0.1
        )
        
        total_dynamic_score += dynamic_score
        
        # Categorize pages
        if neighbor.get('scriptCount', 0) > 10:
            js_heavy_count += 1
        if neighbor.get('formCount', 0) > 2:
            form_rich_count += 1
        if (neighbor.get('buttonCount', 0) + neighbor.get('inputCount', 0)) > 15:
            interactive_count += 1
        if neighbor.get('imageCount', 0) > 50:
            media_rich_count += 1
        
        plan_analysis.append({
            'position': i + 1,
            'url': url,
            'title': neighbor.get('title', ''),
            'dynamic_score': dynamic_score,
            'scriptCount': neighbor.get('scriptCount', 0),
            'formCount': neighbor.get('formCount', 0),
            'buttonCount': neighbor.get('buttonCount', 0),
            'inputCount': neighbor.get('inputCount', 0),
            'imageCount': neighbor.get('imageCount', 0)
        })
    
    avg_dynamic_score = total_dynamic_score / len(plan) if plan else 0.0
    
    return {
        'total_urls': len(plan),
        'avg_dynamic_score': avg_dynamic_score,
        'js_heavy_pages': js_heavy_count,
        'form_rich_pages': form_rich_count,
        'interactive_pages': interactive_count,
        'media_rich_pages': media_rich_count,
        'plan_distribution': plan_analysis
    }


def filter_neighbors_by_domain(neighbors: List[Dict], base_domain: str) -> List[Dict]:
    """
    Filter neighbors to only include URLs from the same domain.
    
    Args:
        neighbors: List of neighbor dictionaries
        base_domain: Base domain to filter by
        
    Returns:
        Filtered list of neighbors from the same domain
    """
    if not base_domain:
        return neighbors
    
    filtered = []
    for neighbor in neighbors:
        try:
            neighbor_domain = urlparse(neighbor['url']).netloc
            if neighbor_domain == base_domain:
                filtered.append(neighbor)
        except Exception:
            # Skip invalid URLs
            continue
    
    return filtered


def prioritize_by_content_type(neighbors: List[Dict], content_priority: str = "dynamic") -> List[Dict]:
    """
    Re-prioritize neighbors based on specific content type preferences.
    
    Args:
        neighbors: List of neighbor dictionaries
        content_priority: Priority type ("dynamic", "forms", "media", "interactive")
        
    Returns:
        Re-prioritized list of neighbors
    """
    if content_priority == "dynamic":
        # Already handled by default scoring
        return neighbors
    elif content_priority == "forms":
        neighbors.sort(key=lambda x: (-x.get('formCount', 0), -x.get('scriptCount', 0), x['url']))
    elif content_priority == "media":
        neighbors.sort(key=lambda x: (-x.get('imageCount', 0), -x.get('mediaCount', 0), x['url']))
    elif content_priority == "interactive":
        neighbors.sort(key=lambda x: (-(x.get('buttonCount', 0) + x.get('inputCount', 0)), -x.get('scriptCount', 0), x['url']))
    
    return neighbors
