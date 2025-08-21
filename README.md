## Domain-agnostic web scraper with Crawl4AI → Neo4j

This project crawls any website up to a configurable depth (default 3), extracts pages, links, buttons, and images, and stores them as a graph in Neo4j.

### Prerequisites

- Python 3.13
- Neo4j running and reachable (default: `bolt://localhost:7687`)

### Setup

```bash
# In project root
uv venv -p 3.13
source .venv/bin/activate

# Install deps
uv pip install -r requirements.txt

# Install Playwright browsers for Crawl4AI
python -m playwright install chromium

# Configure environment
cp .env.example .env
```

Edit `.env` as needed. Example defaults target `https://example.com` and local Neo4j.

### Run a crawl

```bash
source .venv/bin/activate
python -m scraper.run --url https://zineps.com --depth 3 --max-pages 200
```

### Testing

```bash
source .venv/bin/activate
pytest -q
```

### Output

Nodes created in Neo4j:

- Page (unique by `url`)
- Link (unique by `href` + `text`)
- Button (unique per page via `uid`)
- Image (unique by `src` + `alt`)

Relationships:

- (Page)-[:CONTAINS]->(Link|Button|Image)
- (Page)-[:LINKS_TO]->(Page)
- (Link)-[:TARGETS]->(Page) when target page is known

Optionally, we can add PyVis visualization later.
