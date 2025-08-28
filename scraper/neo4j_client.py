import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import Optional, Dict, Any, List, Literal
from datetime import datetime

load_dotenv()


class Neo4jClient:
    def __init__(self):
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        password = os.getenv("NEO4J_PASSWORD")  # change if your password differs

        if not uri or not user or not password:
            raise ValueError("Missing Neo4j connection parameters in environment variables")
        
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        if self._driver:
            self._driver.close()

    # ---------------- Enhanced methods for comprehensive DOM features ---------------- #

    def upsert_page(self, url: str, title: Optional[str] = None, weight: Optional[float] = None):
        """Create or update a page node with metadata."""
        with self._driver.session() as session:
            session.execute_write(self._merge_page_with_metadata, url, title, weight)

    def mark_page_visited(self, url: str):
        """Mark a page as visited and set lastCrawledAt timestamp."""
        with self._driver.session() as session:
            session.execute_write(self._mark_visited, url)

    def get_neighbors_hop1(self, url: str, direction: Literal["out", "in", "both"] = "out", 
                          limit: int = 25, exclude_visited: bool = True) -> List[Dict]:
        """
        Fetch immediate neighbors ordered by weight DESC, url ASC.
        Optionally enrich with feature counts and exclude visited pages.
        """
        with self._driver.session() as session:
            result = session.run(
                self._get_neighbors_query(direction, exclude_visited),
                url=url,
                limit=limit
            )
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
        with self._driver.session() as session:
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

    def _get_neighbors_query(self, direction: str, exclude_visited: bool) -> str:
        """Generate the appropriate Cypher query based on direction and visited filter."""
        visited_filter = "AND NOT p.visited" if exclude_visited else ""
        
        if direction == "out":
            return f"""
                MATCH (start:Page {{url: $url}})-[:LINKS_TO]->(p:Page)
                WHERE p.url IS NOT NULL {visited_filter}
                RETURN p.url AS url, 
                       coalesce(p.title, '') AS title,
                       coalesce(p.weight, 0.0) AS weight,
                       coalesce(p.visited, false) AS visited,
                       p.lastCrawledAt AS lastCrawledAt
                ORDER BY p.weight DESC, p.url ASC
                LIMIT $limit
            """
        elif direction == "in":
            return f"""
                MATCH (p:Page)-[:LINKS_TO]->(start:Page {{url: $url}})
                WHERE p.url IS NOT NULL {visited_filter}
                RETURN p.url AS url, 
                       coalesce(p.title, '') AS title,
                       coalesce(p.weight, 0.0) AS weight,
                       coalesce(p.visited, false) AS visited,
                       p.lastCrawledAt AS lastCrawledAt
                ORDER BY p.weight DESC, p.url ASC
                LIMIT $limit
            """
        else:  # both
            return f"""
                MATCH (start:Page {{url: $url}})-[:LINKS_TO]->(p:Page)
                WHERE p.url IS NOT NULL {visited_filter}
                RETURN p.url AS url, 
                       coalesce(p.title, '') AS title,
                       coalesce(p.weight, 0.0) AS weight,
                       coalesce(p.visited, false) AS visited,
                       p.lastCrawledAt AS lastCrawledAt
                UNION
                MATCH (p:Page)-[:LINKS_TO]->(start:Page {{url: $url}})
                WHERE p.url IS NOT NULL {visited_filter}
                RETURN p.url AS url, 
                       coalesce(p.title, '') AS title,
                       coalesce(p.weight, 0.0) AS weight,
                       coalesce(p.visited, false) AS visited,
                       p.lastCrawledAt AS lastCrawledAt
                ORDER BY weight DESC, url ASC
                LIMIT $limit
            """

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

    def add_input_field(self, page_url: str, input_data: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_input_field,
                page_url,
                input_data.get("id", ""),
                input_data.get("name", ""),
                input_data.get("type", ""),
                input_data.get("placeholder", ""),
                input_data.get("required", False),
                input_data.get("value", ""),
            )

    def add_textarea(self, page_url: str, textarea_data: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_textarea,
                page_url,
                textarea_data.get("id", ""),
                textarea_data.get("name", ""),
                textarea_data.get("placeholder", ""),
                textarea_data.get("rows", ""),
                textarea_data.get("cols", ""),
                textarea_data.get("required", False),
            )

    def add_image(self, page_url: str, src: str, alt: Optional[str]):
        with self._driver.session() as session:
            session.execute_write(self._merge_image, page_url, src, alt or "")

    def add_stylesheet(self, page_url: str, stylesheet_data: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_stylesheet,
                page_url,
                stylesheet_data.get("url", ""),
                stylesheet_data.get("type", ""),
                stylesheet_data.get("media", ""),
                stylesheet_data.get("integrity", ""),
                stylesheet_data.get("crossorigin", ""),
                stylesheet_data.get("content", ""),
            )

    def add_script(self, page_url: str, script_data: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_script,
                page_url,
                script_data.get("url", ""),
                script_data.get("type", ""),
                script_data.get("async", False),
                script_data.get("defer", False),
                script_data.get("integrity", ""),
                script_data.get("crossorigin", ""),
                script_data.get("content", ""),
            )

    def add_font(self, page_url: str, font_data: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_font,
                page_url,
                font_data.get("url", ""),
                font_data.get("type", ""),
                font_data.get("format", ""),
                font_data.get("crossorigin", ""),
            )

    def add_form(self, page_url: str, form_data: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_form,
                page_url,
                form_data.get("id", ""),
                form_data.get("name", ""),
                form_data.get("action", ""),
                form_data.get("method", ""),
                form_data.get("enctype", ""),
                form_data.get("target", ""),
            )

    def add_media_resource(self, page_url: str, media_data: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_media_resource,
                page_url,
                media_data.get("url", ""),
                media_data.get("type", ""),
                media_data.get("width", ""),
                media_data.get("height", ""),
                media_data.get("controls", False),
                media_data.get("autoplay", False),
                media_data.get("loop", False),
            )

    def add_resource(self, page_url: str, res: Dict[str, Any]):
        with self._driver.session() as session:
            session.execute_write(
                self._merge_resource,
                page_url,
                res.get("url", ""),
                res.get("method", ""),
                res.get("resource_type", "")
            )

    # ---------------- Legacy methods for backward compatibility ---------------- #

    def add_input(self, page_url: str, input_data: Dict[str, Any]):
        """Legacy method - use add_input_field instead"""
        self.add_input_field(page_url, input_data)

    # ---------------- Utility / Summary methods ---------------- #

    def get_summary(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Returns a comprehensive summary of pages with all their linked elements,
        resources, and interactive components.
        """
        with self._driver.session() as session:
            result = session.run(
                """
                MATCH (p:Page)
                OPTIONAL MATCH (p)-[:LINKS_TO]->(linked:Page)
                OPTIONAL MATCH (p)-[:HAS_LINK]->(l:Link)
                OPTIONAL MATCH (p)-[:HAS_BUTTON]->(b:Button)
                OPTIONAL MATCH (p)-[:HAS_INPUT]->(i:InputField)
                OPTIONAL MATCH (p)-[:HAS_TEXTAREA]->(t:Textarea)
                OPTIONAL MATCH (p)-[:HAS_IMAGE]->(img:Image)
                OPTIONAL MATCH (p)-[:USES_STYLESHEET]->(s:Stylesheet)
                OPTIONAL MATCH (p)-[:USES_SCRIPT]->(scr:Script)
                OPTIONAL MATCH (p)-[:USES_FONT]->(f:Font)
                OPTIONAL MATCH (p)-[:HAS_FORM]->(form:Form)
                OPTIONAL MATCH (p)-[:USES_MEDIA]->(m:MediaResource)
                OPTIONAL MATCH (p)-[:USES_RESOURCE]->(r:Resource)
                RETURN p.url AS page,
                       collect(DISTINCT linked.url) AS linked_pages,
                       collect(DISTINCT l) AS links,
                       collect(DISTINCT b) AS buttons,
                       collect(DISTINCT i) AS inputs,
                       collect(DISTINCT t) AS textareas,
                       collect(DISTINCT img) AS images,
                       collect(DISTINCT s) AS stylesheets,
                       collect(DISTINCT scr) AS scripts,
                       collect(DISTINCT f) AS fonts,
                       collect(DISTINCT form) AS forms,
                       collect(DISTINCT m) AS media,
                       collect(DISTINCT r) AS resources
                LIMIT $limit
                """,
                limit=limit,
            )
            return [record.data() for record in result]

    def get_hop1_neighborhood(self, page_url: str) -> Dict[str, Any]:
        """
        Get hop-1 neighborhood information for a specific page.
        Returns neighbors with their metadata and crawl planning information.
        """
        with self._driver.session() as session:
            # First, get all pages that this page links to (via Link nodes)
            result = session.run(
                """
                MATCH (source:Page {url: $page_url})-[:HAS_LINK]->(link:Link)
                WITH link.href AS target_url, link.text AS link_text
                OPTIONAL MATCH (target:Page {url: target_url})
                OPTIONAL MATCH (target)-[:HAS_FORM]->(form:Form)
                OPTIONAL MATCH (target)-[:HAS_BUTTON]->(button:Button)
                OPTIONAL MATCH (target)-[:HAS_INPUT]->(input:InputField)
                OPTIONAL MATCH (target)-[:HAS_IMAGE]->(image:Image)
                OPTIONAL MATCH (target)-[:USES_SCRIPT]->(script:Script)
                OPTIONAL MATCH (target)-[:USES_STYLESHEET]->(stylesheet:Stylesheet)
                OPTIONAL MATCH (target)-[:HAS_LINK]->(target_link:Link)
                OPTIONAL MATCH (target)-[:USES_MEDIA]->(media:MediaResource)
                RETURN target_url AS url,
                       link_text AS link_text,
                       target.url IS NOT NULL AS page_exists,
                       count(DISTINCT form) AS formCount,
                       count(DISTINCT button) AS buttonCount,
                       count(DISTINCT input) AS inputCount,
                       count(DISTINCT image) AS imageCount,
                       count(DISTINCT script) AS scriptCount,
                       count(DISTINCT stylesheet) AS stylesheetCount,
                       count(DISTINCT target_link) AS linkCount,
                       count(DISTINCT media) AS mediaCount
                ORDER BY page_exists DESC, formCount DESC, scriptCount DESC, buttonCount DESC
                """,
                page_url=page_url,
            )
            neighbors = [record.data() for record in result]
            
            # Filter out None values
            neighbors = [n for n in neighbors if n['url'] is not None]
            
            # Calculate total counts for crawl planning
            total_forms = sum(n['formCount'] for n in neighbors)
            total_scripts = sum(n['scriptCount'] for n in neighbors)
            total_interactive = sum(n['buttonCount'] + n['inputCount'] for n in neighbors)
            
            return {
                'source_url': page_url,
                'neighbors': neighbors,
                'summary': {
                    'total_neighbors': len(neighbors),
                    'total_forms': total_forms,
                    'total_scripts': total_scripts,
                    'total_interactive': total_interactive,
                    'avg_weight': 0.0  # Default weight since we don't have weight property
                }
            }

    # ---------------- Enhanced Cypher queries ---------------- #

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
            MERGE (l:Link {href: $href})
            SET l.text = coalesce($text, ''), l.title = coalesce($title, '')
            MERGE (p)-[:HAS_LINK]->(l)
            """,
            page_url=page_url,
            href=href,
            text=text,
            title=title,
        )

    @staticmethod
    def _merge_button(tx, page_url: str, text: str, button_id: str, name: str, btn_type: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (b:Button {uid: $uid})
            SET b.text = $text, b.id = $button_id, b.name = $name, b.type = $btn_type
            MERGE (p)-[:HAS_BUTTON]->(b)
            """,
            page_url=page_url,
            uid=f"{page_url}:::{button_id or text}",
            text=text,
            button_id=button_id,
            name=name,
            btn_type=btn_type,
        )

    @staticmethod
    def _merge_input_field(tx, page_url: str, input_id: str, name: str, input_type: str, placeholder: str, required: bool, value: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (i:InputField {uid: $uid})
            SET i.id = $input_id, i.name = $name, i.type = $input_type, i.placeholder = $placeholder, i.required = $required, i.value = $value
            MERGE (p)-[:HAS_INPUT]->(i)
            """,
            page_url=page_url,
            uid=f"{page_url}:::{input_id or name or placeholder}",
            input_id=input_id,
            name=name,
            input_type=input_type,
            placeholder=placeholder,
            required=required,
            value=value,
        )

    @staticmethod
    def _merge_textarea(tx, page_url: str, ta_id: str, name: str, placeholder: str, rows: str, cols: str, required: bool):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (t:Textarea {uid: $uid})
            SET t.id = $ta_id, t.name = $name, t.placeholder = $placeholder, t.rows = $rows, t.cols = $cols, t.required = $required
            MERGE (p)-[:HAS_TEXTAREA]->(t)
            """,
            page_url=page_url,
            uid=f"{page_url}:::{ta_id or name or placeholder}",
            ta_id=ta_id,
            name=name,
            placeholder=placeholder,
            rows=rows,
            cols=cols,
            required=required,
        )

    @staticmethod
    def _merge_image(tx, page_url: str, src: str, alt: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (i:Image {src: $src})
            SET i.alt = $alt
            MERGE (p)-[:HAS_IMAGE]->(i)
            """,
            page_url=page_url,
            src=src,
            alt=alt,
        )

    @staticmethod
    def _merge_stylesheet(tx, page_url: str, url: str, stylesheet_type: str, media: str, integrity: str, crossorigin: str, content: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (s:Stylesheet {url: $url})
            SET s.type = $stylesheet_type, s.media = $media, s.integrity = $integrity, s.crossorigin = $crossorigin, s.content = $content
            MERGE (p)-[:USES_STYLESHEET]->(s)
            """,
            page_url=page_url,
            url=url,
            stylesheet_type=stylesheet_type,
            media=media,
            integrity=integrity,
            crossorigin=crossorigin,
            content=content,
        )

    @staticmethod
    def _merge_script(tx, page_url: str, url: str, script_type: str, async_flag: bool, defer_flag: bool, integrity: str, crossorigin: str, content: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (s:Script {url: $url})
            SET s.type = $script_type, s.async = $async_flag, s.defer = $defer_flag, s.integrity = $integrity, s.crossorigin = $crossorigin, s.content = $content
            MERGE (p)-[:USES_SCRIPT]->(s)
            """,
            page_url=page_url,
            url=url,
            script_type=script_type,
            async_flag=async_flag,
            defer_flag=defer_flag,
            integrity=integrity,
            crossorigin=crossorigin,
            content=content,
        )

    @staticmethod
    def _merge_font(tx, page_url: str, url: str, font_type: str, format_type: str, crossorigin: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (f:Font {url: $url})
            SET f.type = $font_type, f.format = $format_type, f.crossorigin = $crossorigin
            MERGE (p)-[:USES_FONT]->(f)
            """,
            page_url=page_url,
            url=url,
            font_type=font_type,
            format_type=format_type,
            crossorigin=crossorigin,
        )

    @staticmethod
    def _merge_form(tx, page_url: str, form_id: str, name: str, action: str, method: str, enctype: str, target: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (f:Form {uid: $uid})
            SET f.id = $form_id, f.name = $name, f.action = $action, f.method = $method, f.enctype = $enctype, f.target = $target
            MERGE (p)-[:HAS_FORM]->(f)
            """,
            page_url=page_url,
            uid=f"{page_url}:::{form_id or name}",
            form_id=form_id,
            name=name,
            action=action,
            method=method,
            enctype=enctype,
            target=target,
        )

    @staticmethod
    def _merge_media_resource(tx, page_url: str, url: str, media_type: str, width: str, height: str, controls: bool, autoplay: bool, loop: bool):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (m:MediaResource {url: $url})
            SET m.type = $media_type, m.width = $width, m.height = $height, m.controls = $controls, m.autoplay = $autoplay, m.loop = $loop
            MERGE (p)-[:USES_MEDIA]->(m)
            """,
            page_url=page_url,
            url=url,
            media_type=media_type,
            width=width,
            height=height,
            controls=controls,
            autoplay=autoplay,
            loop=loop,
        )

    @staticmethod
    def _merge_resource(tx, page_url: str, res_url: str, method: str, res_type: str):
        tx.run(
            """
            MERGE (p:Page {url: $page_url})
            MERGE (r:Resource {url: $res_url})
            SET r.method = $method, r.type = $res_type
            MERGE (p)-[:USES_RESOURCE {type: $res_type}]->(r)
            """,
            page_url=page_url,
            res_url=res_url,
            method=method,
            res_type=res_type,
        )

    # ---------------- Legacy methods for backward compatibility ---------------- #

    @staticmethod
    def _merge_input(tx, page_url: str, input_id: str, name: str, input_type: str, placeholder: str):
        """Legacy method - use _merge_input_field instead"""
        Neo4jClient._merge_input_field(tx, page_url, input_id, name, input_type, placeholder, False, "")

    @staticmethod
    def _merge_page_with_metadata(tx, url: str, title: Optional[str], weight: Optional[float]):
        """Create or update page with metadata."""
        tx.run(
            """
            MERGE (p:Page {url: $url})
            SET p.title = coalesce($title, p.title),
                p.weight = coalesce($weight, p.weight),
                p.visited = coalesce(p.visited, false)
            """,
            url=url, title=title, weight=weight
        )

    @staticmethod
    def _mark_visited(tx, url: str):
        """Mark page as visited with timestamp."""
        tx.run(
            """
            MATCH (p:Page {url: $url})
            SET p.visited = true, p.lastCrawledAt = datetime()
            """,
            url=url
        )

    # ---------------- Database Management Methods ---------------- #

    def clear_database(self):
        """Clear all data from the database."""
        with self._driver.session() as session:
            session.execute_write(self._clear_all_data)
            print("🗑️  Database cleared successfully!")

    @staticmethod
    def _clear_all_data(tx):
        """Clear all nodes and relationships from the database."""
        tx.run("MATCH (n) DETACH DELETE n")