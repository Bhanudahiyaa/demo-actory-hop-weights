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


def visualize_hierarchical_plan(neighbors: List[Dict], home_url: str = None) -> Dict:
    """
    Visualize the hierarchical structure of the crawl plan.
    
    Args:
        neighbors: List of neighbor dictionaries
        home_url: The home page URL to use as the starting point
        
    Returns:
        Dictionary with hierarchical visualization data
    """
    if not neighbors:
        return {}
    
    # Identify home page if not provided
    if not home_url:
        home_candidates = sorted(neighbors, key=lambda x: (len(x['url']), -x.get('linkCount', 0)))
        home_url = home_candidates[0]['url'] if home_candidates else neighbors[0]['url']
    
    # Parse home URL to get base domain
    try:
        home_parsed = urlparse(home_url)
        home_domain = home_parsed.netloc
    except Exception:
        home_domain = ""
    
    # Build hierarchical tree
    hierarchy = {
        'home': {
            'url': home_url,
            'level': 0,
            'children': {}
        }
    }
    
    # Categorize and organize neighbors
    for neighbor in neighbors:
        url = neighbor['url']
        
        try:
            parsed = urlparse(url)
            neighbor_domain = parsed.netloc
            neighbor_path = parsed.path.rstrip('/') or '/'
            
            # Skip if it's the home page itself
            if url == home_url:
                continue
                
            # Only process same-domain URLs for hierarchy
            if neighbor_domain != home_domain:
                continue
            
            # Determine level and parent
            path_parts = [p for p in neighbor_path.split('/') if p]
            level = len(path_parts)
            
            if level == 1:  # Direct child of home
                parent_key = 'home'
            elif level == 2:  # Child of main section
                parent_key = f"main_{path_parts[0]}"
            elif level >= 3:  # Deep nested
                parent_key = f"sub_{path_parts[0]}_{path_parts[1]}"
            else:
                parent_key = 'home'
            
            # Create node
            node = {
                'url': url,
                'level': level,
                'path': neighbor_path,
                'title': neighbor.get('title', ''),
                'dynamic_score': calculate_dynamic_score(neighbor)['dynamic_score'],
                'scriptCount': neighbor.get('scriptCount', 0),
                'formCount': neighbor.get('formCount', 0),
                'buttonCount': neighbor.get('buttonCount', 0),
                'imageCount': neighbor.get('imageCount', 0),
                'children': {}
            }
            
            # Add to hierarchy
            if parent_key not in hierarchy:
                hierarchy[parent_key] = node
            else:
                hierarchy[parent_key]['children'][url] = node
                
        except Exception:
            continue
    
    return hierarchy


def analyze_plan_quality(neighbors: List[Dict], plan: List[str], home_url: str = None) -> Dict:
    """
    Analyze the quality and characteristics of the generated crawl plan.
    Updated to work with hierarchical structure.
    
    Args:
        neighbors: Original neighbor data
        plan: Generated crawl plan (ordered URLs)
        home_url: The home page URL used for hierarchy
        
    Returns:
        Dictionary with plan analysis metrics including hierarchy info
    """
    if not plan:
        return {
            'total_urls': 0,
            'avg_dynamic_score': 0.0,
            'js_heavy_pages': 0,
            'form_rich_pages': 0,
            'interactive_pages': 0,
            'media_rich_pages': 0,
            'hierarchy_levels': {},
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
    
    hierarchy_levels = {
        'home': 0,
        'main_sections': 0,
        'sub_sections': 0,
        'deep_pages': 0,
        'media_pages': 0,
        'external': 0
    }
    
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
        
        # Determine hierarchy level
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.rstrip('/').split('/') if p]
            level = len(path_parts)
            
            if level == 0:
                hierarchy_levels['home'] += 1
            elif level == 1:
                hierarchy_levels['main_sections'] += 1
            elif level == 2:
                hierarchy_levels['sub_sections'] += 1
            elif level >= 3:
                hierarchy_levels['deep_pages'] += 1
            else:
                hierarchy_levels['main_sections'] += 1
        except Exception:
            hierarchy_levels['main_sections'] += 1
        
        plan_analysis.append({
            'position': i + 1,
            'url': url,
            'title': neighbor.get('title', ''),
            'dynamic_score': dynamic_score,
            'hierarchy_level': level if 'level' in locals() else 0,
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
        'hierarchy_levels': hierarchy_levels,
        'plan_distribution': plan_analysis
    }


# Helper function for dynamic score calculation (used by other functions)
def calculate_dynamic_score(neighbor):
    """Calculate dynamic complexity score for a neighbor."""
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
    
    return {
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


def make_hierarchical_plan_from_neighbors(neighbors: List[Dict], home_url: str = None) -> List[str]:
    """
    Generate an ordered list of URLs to crawl from enriched neighbors.
    Creates a hierarchical structure starting from the home page as the central hub.
    
    Args:
        neighbors: List of neighbor dictionaries with feature counts
        home_url: The home page URL to use as the starting point
        
    Returns:
        Ordered list of URLs to crawl, organized hierarchically from home page
    """
    if not neighbors:
        return []
    
    # Identify home page if not provided
    if not home_url:
        # Try to find the most likely home page (shortest URL or highest link count)
        home_candidates = sorted(neighbors, key=lambda x: (len(x['url']), -x.get('linkCount', 0)))
        home_url = home_candidates[0]['url'] if home_candidates else neighbors[0]['url']
    
    # Parse home URL to get base domain and path
    try:
        home_parsed = urlparse(home_url)
        home_domain = home_parsed.netloc
        home_path = home_parsed.path.rstrip('/') or '/'
    except Exception:
        home_domain = ""
        home_path = "/"
    
    # Categorize neighbors by their relationship to home page
    categorized_neighbors = {
        'home': [],
        'main_sections': [],      # Direct children of home (e.g., /energydrink, /events)
        'sub_sections': [],       # Second level (e.g., /energydrink/red-bull-energy-drink)
        'deep_pages': [],         # Third level and deeper
        'media_pages': [],        # Image/media pages
        'external': []            # External domain pages
    }
    
    for neighbor in neighbors:
        url = neighbor['url']
        
        try:
            parsed = urlparse(url)
            neighbor_domain = parsed.netloc
            neighbor_path = parsed.path.rstrip('/') or '/'
            
            # Skip if it's the home page itself
            if url == home_url:
                continue
                
            # Categorize by domain and path depth
            if neighbor_domain != home_domain:
                categorized_neighbors['external'].append(neighbor)
            elif neighbor_path == '/':
                categorized_neighbors['home'].append(neighbor)
            elif neighbor_path.count('/') == 1:  # e.g., /energydrink
                categorized_neighbors['main_sections'].append(neighbor)
            elif neighbor_path.count('/') == 2:  # e.g., /energydrink/red-bull
                categorized_neighbors['sub_sections'].append(neighbor)
            elif neighbor_path.count('/') >= 3:  # Deep nested pages
                categorized_neighbors['deep_pages'].append(neighbor)
            else:
                categorized_neighbors['main_sections'].append(neighbor)
                
        except Exception:
            # If parsing fails, put in main sections
            categorized_neighbors['main_sections'].append(neighbor)
    
    # Calculate dynamic complexity score for each neighbor
    def calculate_dynamic_score(neighbor):
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
        
        return {
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
        }
    
    # Score and sort each category
    for category in categorized_neighbors:
        if categorized_neighbors[category]:
            scored = [calculate_dynamic_score(n) for n in categorized_neighbors[category]]
            scored.sort(key=lambda x: (-x['score'], x['url']))
            categorized_neighbors[category] = scored
    
    # Build hierarchical crawl plan
    ordered_urls = []
    
    # 1. Start with home page (if not already in neighbors)
    if home_url not in [n['url'] for n in neighbors]:
        ordered_urls.append(home_url)
    
    # 2. Add main navigation sections (highest priority)
    ordered_urls.extend([n['url'] for n in categorized_neighbors['main_sections']])
    
    # 3. Add sub-sections (second priority)
    ordered_urls.extend([n['url'] for n in categorized_neighbors['sub_sections']])
    
    # 4. Add deep pages (third priority)
    ordered_urls.extend([n['url'] for n in categorized_neighbors['deep_pages']])
    
    # 5. Add media pages (lower priority)
    ordered_urls.extend([n['url'] for n in categorized_neighbors['media_pages']])
    
    # 6. Add external pages (lowest priority)
    ordered_urls.extend([n['url'] for n in categorized_neighbors['external']])
    
    return ordered_urls
