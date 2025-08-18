import os
import asyncio
import argparse
from typing import List, Optional

from dotenv import load_dotenv
from crawl4ai import AsyncWebCrawler
from crawl4ai.deep_crawling.bfs_strategy import BFSDeepCrawlStrategy
from crawl4ai.async_configs import CrawlerRunConfig

from .neo4j_client import Neo4jClient
from .extractors import extract_buttons, extract_images_from_result_media


def env_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


async def crawl_and_store(
    url: str,
    depth: int,
    max_pages: int,
    include_external: bool,
    neo4j: Optional[Neo4jClient],
):
    strategy = BFSDeepCrawlStrategy(
        max_depth=depth,
        include_external=include_external,
        max_pages=max_pages,
    )
    config = CrawlerRunConfig(
        deep_crawl_strategy=strategy,
        exclude_external_links=not include_external,
        verbose=False,
        stream=True,
    )

    async with AsyncWebCrawler() as crawler:
        # With DeepCrawlDecorator, calling crawler.arun(url, config) will delegate to deep strategy
        results = await crawler.arun(url, config=config)
        pages: List[str] = []

        async for result in results:
            page_url = result.url
            pages.append(page_url)
            if neo4j:
                neo4j.upsert_page(page_url)

            # LINKS
            internal = (result.links or {}).get("internal", [])
            external = (result.links or {}).get("external", [])
            for link in internal + external:
                href = link.get("href") or ""
                text = link.get("text") or ""
                title = link.get("title") or ""
                if not href:
                    continue
                if neo4j:
                    neo4j.add_link(page_url, href, text, title)
                # link to page if href is a page we visited as well later (optional: we can also create LINK_TO regardless)
                # We'll create LINKS_TO edges when we see the linked page or preemptively if same-domain
                if href.startswith("http"):
                    if neo4j:
                        neo4j.link_pages(page_url, href)

            # BUTTONS
            for button in extract_buttons(result.html):
                if neo4j:
                    neo4j.add_button(page_url, button)

            # IMAGES
            for image in extract_images_from_result_media(result.media or {}):
                src = image.get("src") or ""
                if not src:
                    continue
                if neo4j:
                    neo4j.add_image(page_url, src, image.get("alt", ""))


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=os.getenv("TARGET_URL", "https://example.com"))
    parser.add_argument("--depth", type=int, default=int(os.getenv("MAX_DEPTH", "3")))
    parser.add_argument("--max-pages", type=int, default=int(os.getenv("MAX_PAGES", "200")))
    parser.add_argument(
        "--include-external",
        action="store_true",
        default=env_bool(os.getenv("INCLUDE_EXTERNAL"), False),
        help="Include external-domain links during link discovery (default from INCLUDE_EXTERNAL env)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run crawl and extraction without writing to Neo4j",
    )
    args = parser.parse_args()

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_pwd = os.getenv("NEO4J_PASSWORD", "neo4j")

    client: Optional[Neo4jClient] = None if args.dry_run else Neo4jClient(neo4j_uri, neo4j_user, neo4j_pwd)
    try:
        asyncio.run(crawl_and_store(args.url, args.depth, args.max_pages, args.include_external, client))
    finally:
        if client:
            client.close()


if __name__ == "__main__":
    main()


