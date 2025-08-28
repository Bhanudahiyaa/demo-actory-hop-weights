# DSPy Crawl Agent - Autonomous Website Crawling

## 🚀 Overview

The DSPy Crawl Agent is an intelligent, autonomous website crawling system that uses DSPy modules to plan, execute, and analyze website crawls. It integrates seamlessly with Neo4j for graph-based data storage and analysis.

## 🏗️ Architecture

### Core Components

1. **CrawlPlanner** - Plans crawl strategy based on neighbor analysis
2. **Crawler** - Fetches URLs and extracts metadata
3. **Analyzer** - Generates comprehensive crawl reports
4. **CrawlAgent** - Orchestrates the entire crawling process

### Data Flow

```
Start URL → CrawlPlanner → Neighbor Analysis → Crawl Queue → Crawler → Neo4j → Analyzer → Report
```

## 📋 Requirements

- Python 3.13+
- Neo4j database
- DSPy AI framework
- Required Python packages (see requirements.txt)

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install DSPy
pip install dspy-ai

# Ensure Neo4j is running
docker-compose up -d
```

## 🎯 Usage

### Command Line Interface

```bash
# Basic autonomous crawling
python -m scraper.run --crawl --start-url https://example.com --depth 2 --max-pages 50

# With custom parameters
python -m scraper.run --crawl --start-url https://apple.in --depth 3 --max-pages 100

# Traditional crawling (existing functionality)
python -m scraper.run --url https://example.com --depth 2 --max-pages 50
```

### Programmatic Usage

```python
from scraper.crawl_agent import CrawlAgent
from scraper.neo4j_client import Neo4jClient

# Initialize
neo4j_client = Neo4jClient()
crawl_agent = CrawlAgent(neo4j_client)

# Run autonomous crawl
result = crawl_agent.forward(
    start_url="https://example.com",
    max_depth=2,
    max_pages=50
)

# Access results
print(f"Pages crawled: {result['crawl_metadata']['total_pages_crawled']}")
print(f"Analysis: {result['analysis_report']['summary']}")
```

## 🔧 Configuration

### Environment Variables

```bash
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=please_change_me

# Crawling Defaults
TARGET_URL=https://example.com
MAX_DEPTH=3
MAX_PAGES=200
INCLUDE_EXTERNAL=false
```

### Neo4j Schema

```cypher
// Page nodes with metadata
(:Page {
  url: STRING,
  title: STRING,
  weight: FLOAT,
  visited: BOOLEAN,
  lastCrawledAt: DATETIME
})

// Navigation relationships
(Page)-[:LINKS_TO]->(Page)

// Feature relationships
(Page)-[:HAS_LINK]->(Link)
(Page)-[:HAS_BUTTON]->(Button)
(Page)-[:HAS_FORM]->(Form)
(Page)-[:USES_SCRIPT]->(Script)
(Page)-[:HAS_IMAGE]->(Image)
```

## 🧠 Intelligence Features

### Dynamic Scoring

The agent calculates dynamic complexity scores based on:

- **JavaScript Scripts** (50% weight) - Dynamic functionality
- **Forms** (25% weight) - User interaction
- **Interactive Elements** (15% weight) - Buttons, inputs
- **Images** (10% weight) - Media content

### Autonomous Planning

1. **Neighbor Discovery** - Finds connected pages
2. **Feature Analysis** - Analyzes page complexity
3. **Priority Scoring** - Ranks pages by importance
4. **Queue Management** - Maintains crawl order
5. **Depth Control** - Respects crawl limits

### Adaptive Crawling

- **Weight-based Prioritization** - Crawls important pages first
- **Feature Extraction** - Comprehensive DOM analysis
- **Error Handling** - Graceful failure recovery
- **Progress Tracking** - Real-time crawl monitoring

## 📊 Output & Analysis

### Crawl Summary

```
📊 CRAWL ANALYSIS
📈 SUMMARY:
   Total Pages Crawled: 25
   Successful Crawls: 24
   Failed Crawls: 1
   Success Rate: 96.0%
   Max Depth Reached: 2
   Crawl Duration: 45.23s
```

### Statistics

```
📊 STATISTICS:
   Avg Response Time: 1.847s
   Max Response Time: 8.234s
   Avg Dynamic Score: 12.456
   Max Dynamic Score: 45.678
   Total Scripts: 156
   Total Forms: 23
   Total Buttons: 89
   Total Images: 234
   Total Links: 567
```

### Top Pages

```
🏆 TOP PAGES (by Dynamic Score):
   1. https://example.com/dashboard
      Score: 45.68 | Scripts: 23 | Forms: 5 | Images: 12
   2. https://example.com/admin
      Score: 38.92 | Scripts: 18 | Forms: 8 | Images: 6
```

## 🔍 Testing

### Test Script

```bash
# Run basic test
python test_dspy_agent.py

# Test with specific URL
python -m scraper.run --crawl --start-url https://example.com --depth 1 --max-pages 5
```

### Validation

The agent validates:

- URL accessibility
- Neo4j connectivity
- Feature extraction accuracy
- Crawl plan quality
- Analysis report completeness

## 🚨 Error Handling

### Common Issues

1. **Neo4j Connection** - Check database status and credentials
2. **URL Accessibility** - Verify target website availability
3. **Memory Limits** - Adjust max_pages for large sites
4. **Rate Limiting** - Respect website crawling policies

### Recovery Strategies

- **Automatic Retry** - Failed pages are logged and skipped
- **Graceful Degradation** - Continues crawling despite individual failures
- **Error Logging** - Comprehensive error tracking and reporting
- **Resource Cleanup** - Proper cleanup of connections and resources

## 🔮 Future Enhancements

### Planned Features

- **Machine Learning Integration** - Learn from crawl patterns
- **Advanced Scheduling** - Intelligent crawl timing
- **Content Classification** - AI-powered content analysis
- **Performance Optimization** - Parallel crawling and caching
- **API Integration** - RESTful API for external access

### Extensibility

The modular design allows for:

- **Custom Scoring Algorithms** - Implement domain-specific logic
- **Plugin Architecture** - Add new feature extractors
- **Multi-Strategy Support** - Different crawling approaches
- **Custom Analysis** - Domain-specific reporting

## 📚 API Reference

### CrawlAgent

```python
class CrawlAgent(dspy.Module):
    def forward(self, start_url: str, max_depth: int, max_pages: int) -> Dict[str, Any]
```

### CrawlPlanner

```python
class CrawlPlanner(dspy.Module):
    def forward(self, center_url: str, neighbors: List[Dict], max_pages: int) -> Dict[str, Any]
```

### Crawler

```python
class Crawler(dspy.Module):
    def forward(self, url: str) -> Dict[str, Any]
```

### Analyzer

```python
class Analyzer(dspy.Module):
    def forward(self, crawl_results: List[Dict], crawl_metadata: Dict) -> Dict[str, Any]
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch**
3. **Implement your changes**
4. **Add tests and documentation**
5. **Submit a pull request**

## 📄 License

This project is part of the Actory AI Scrape system. See the main LICENSE file for details.

## 🆘 Support

For issues and questions:

1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information
4. Include logs and error messages

---

**Happy Autonomous Crawling! 🕷️🤖**
