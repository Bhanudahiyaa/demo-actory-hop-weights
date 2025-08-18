from neo4j import GraphDatabase
from typing import Optional, Dict, Any


class Neo4jClient:
    def __init__(self, uri: str, user: str, password: str):
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self._driver.close()

    def upsert_page(self, url: str):
        with self._driver.session() as session:
            session.execute_write(self._merge_page, url)

    def link_pages(self, from_url: str, to_url: str):
        with self._driver.session() as session:
            session.execute_write(self._merge_page_link, from_url, to_url)

    def add_link(self, page_url: str, href: str, text: Optional[str] = None, title: Optional[str] = None):
        with self._driver.session() as session:
            session.execute_write(self._merge_link, page_url, href, text, title)

    def add_button(self, page_url: str, button: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_button,
                page_url,
                button.get("text", ""),
                button.get("id", ""),
                button.get("name", ""),
                button.get("type", ""),
            )

    def add_image(self, page_url: str, src: str, alt: Optional[str]):
        with self._driver.session() as session:
            session.execute_write(self._merge_image, page_url, src, alt or "")

    @staticmethod
    def _merge_page(tx, url: str):
        tx.run("MERGE (p:Page {url: $url})", url=url)

    @staticmethod
    def _merge_page_link(tx, from_url: str, to_url: str):
        tx.run(
            """
            MERGE (from:Page {url: $from_url})
            MERGE (to:Page {url: $to_url})
            MERGE (from)-[:LINKS_TO]->(to)
            """,
            from_url=from_url,
            to_url=to_url,
        )

    @staticmethod
    def _merge_link(tx, page_url: str, href: str, text: Optional[str], title: Optional[str]):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (l:Link {href: $href, text: coalesce($text, ''), title: coalesce($title, '')})
            MERGE (p)-[:CONTAINS]->(l)
            """,
            page_url=page_url,
            href=href,
            text=text,
            title=title,
        )

    @staticmethod
    def _merge_button(tx, page_url: str, text: str, button_id: str, name: str, btn_type: str):
        # uid makes button unique per page
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (b:Button {uid: $uid})
            SET b.text = $text, b.id = $button_id, b.name = $name, b.type = $btn_type
            MERGE (p)-[:CONTAINS]->(b)
            """,
            page_url=page_url,
            uid=f"{page_url}:::{button_id or text}",
            text=text,
            button_id=button_id,
            name=name,
            btn_type=btn_type,
        )

    @staticmethod
    def _merge_image(tx, page_url: str, src: str, alt: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (i:Image {src: $src, alt: $alt})
            MERGE (p)-[:CONTAINS]->(i)
            """,
            page_url=page_url,
            src=src,
            alt=alt,
        )


