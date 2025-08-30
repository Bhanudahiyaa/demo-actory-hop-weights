# 🚀 Actory AI Web Scraper - Enhanced with LLM Intelligence

A clean, organized web scraping system powered by **GPT-OSS-20B** for intelligent content analysis and navigation planning.

## ✨ **Features**

- **🧠 LLM-Enhanced Scraping**: Business value scoring and UX analysis
- **🗺️ Intelligent Navigation**: Smart route planning with path optimization
- **🗄️ Neo4j Graph Database**: Efficient storage and relationship mapping
- **⚡ Performance Optimized**: Clean architecture for maximum efficiency
- **🎯 Business Intelligence**: Content classification and conversion analysis

## 🏗️ **Clean Architecture**

```
actory-ai-scrape-bhanu/
├── main.py                 # 🚀 Main entry point
├── scraper/               # 🧹 Core modules
│   ├── enhanced_run.py    # 📊 LLM-enhanced scraping
│   ├── planningagent.py   # 🧠 Intelligent navigation
│   ├── llm_extractor.py   # 🤖 Qwen3 LLM integration
│   ├── neo4j_client.py    # 🗄️ Database interface
│   ├── extractors.py      # 🔍 DOM feature extraction
│   ├── plan.py           # 📋 Navigation planning
│   └── crawl_agent.py    # 🤖 Autonomous crawling
├── tests/                 # 🧪 Test suite
├── requirements.txt       # 📦 Dependencies
└── README.md             # 📚 This file
```

## 🚀 **Quick Start**

### **1. Setup Environment**

```bash
# Clone repository
git clone <your-repo>
cd actory-ai-scrape-bhanu

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Start Neo4j Database**

```bash
# Using Docker
docker-compose up -d

# Or start Neo4j manually
# Make sure Neo4j is running on localhost:7687
```

### **3. Run Enhanced Workflow**

```bash
# Basic usage
python main.py "https://example.com"

# With custom settings
python main.py "https://example.com" -p 20 --clear-db

# Disable LLM (faster, less intelligent)
python main.py "https://example.com" --no-llm
```

## 🧠 **LLM Intelligence Features**

### **Content Analysis**

- **Business Value Scoring** (1-10): Conversion potential and revenue impact
- **User Experience Analysis** (1-10): Navigation clarity and interaction quality
- **Content Classification**: Page types and business categories
- **Planning Priority**: Route importance for navigation optimization

### **Enhanced Scoring**

- **DOM Complexity**: Scripts, forms, buttons, inputs, images
- **LLM Multipliers**: Business value, UX, and planning priority
- **Smart Optimization**: Intelligent route prioritization

## 📊 **Workflow Phases**

### **Phase 1: LLM-Enhanced Scraping**

```python
from scraper.enhanced_run import enhanced_crawl_and_store

# Scrape with LLM intelligence
pages = await enhanced_crawl_and_store(
    url="https://example.com",
    depth=3,
    max_pages=20,
    include_external=False,
    neo4j=neo4j_client,
    enable_llm_analysis=True
)
```

### **Phase 2: Intelligent Navigation Planning**

```python
from scraper.planningagent import PlanningAgent

# Create intelligent navigation plan
planning_agent = PlanningAgent(neo4j_client)
result = planning_agent.forward(
    landing_url="https://example.com",
    max_pages=20,
    exploration_strategy="hierarchical"
)
```

## 🗄️ **Neo4j Database Schema**

### **Nodes**

- **Page**: Web pages with enhanced metadata
- **Link**: Hyperlinks with text and title
- **Button**: Interactive buttons
- **Form**: Web forms
- **Image**: Images with alt text
- **Script**: JavaScript files
- **Stylesheet**: CSS files

### **Relationships**

- **LINKS_TO**: Page navigation paths
- **HAS_LINK**: Page contains link
- **HAS_BUTTON**: Page contains button
- **HAS_FORM**: Page contains form
- **HAS_IMAGE**: Page contains image

### **Enhanced Properties**

- **business_value_score**: LLM business analysis (1-10)
- **user_experience_score**: LLM UX analysis (1-10)
- **planning_priority_score**: LLM planning analysis (1-10)
- **page_type**: Content classification
- **enhanced_scoring_active**: LLM enhancement status

## 🔧 **Configuration**

### **Environment Variables**

```bash
# Neo4j Connection
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM API (OpenRouter)
OPENROUTER_API_KEY=your-api-key
```

### **LLM Model Configuration**

```python
# Default: GPT-OSS-20B
model_name = "openai/gpt-oss-20b:free"
api_token = "your-openrouter-api-key"
```

## 🧪 **Testing**

```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_extractors.py

# Run with coverage
pytest --cov=scraper
```

## 📈 **Performance & Optimization**

### **Smart Path Caching**

- **Route Reuse**: Intelligent path segment caching
- **LLM Optimization**: AI-powered route selection
- **Memory Management**: Efficient cache strategies

### **Parallel Processing**

- **Async Scraping**: Non-blocking web requests
- **Batch Processing**: Efficient LLM analysis
- **Database Optimization**: Bulk Neo4j operations

## 🚨 **Troubleshooting**

### **Common Issues**

1. **Neo4j Connection**: Check database is running and credentials
2. **LLM API Errors**: Verify OpenRouter API key and quota
3. **Memory Issues**: Reduce max_pages or enable garbage collection

### **Debug Mode**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 **Contributing**

1. **Fork** the repository
2. **Create** a feature branch
3. **Make** your changes
4. **Test** thoroughly
5. **Submit** a pull request

## 📄 **License**

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 **Acknowledgments**

- **GPT-OSS-20B**: Advanced LLM intelligence
- **Crawl4AI**: High-performance web scraping
- **Neo4j**: Graph database technology
- **DSPy**: AI agent framework

---

**🚀 Ready to scrape with intelligence? Run `python main.py "https://your-site.com"` and experience the power of LLM-enhanced web scraping!**
