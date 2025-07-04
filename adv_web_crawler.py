from langflow.custom import Component
from langflow.io import StrInput, BoolInput, IntInput, DropdownInput, MultiselectInput, Output
from langflow.schema import Data, DataFrame
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from collections import deque
import re
import time
import pandas as pd

try:
    from requests_html import HTMLSession
    RENDERING_AVAILABLE = True
except ImportError:
    RENDERING_AVAILABLE = False

class WebCrawlerComponent(Component):
    display_name = "Advanced Web Crawler"
    description = "Fetches HTML or JSON from a URL and extracts structured content. Supports crawling depth, domain filtering, JS rendering, robots.txt and sitemap parsing."
    icon = "language-html5"
    name = "AdvancedWebCrawler"
    field_order = [
        "url", "user_agent", "max_content_length", "content_type_preference",
        "include_metadata", "include_headings", "include_paragraphs",
        "extract_images", "extract_links", "output_format",
        "max_depth", "same_domain_only", "enable_js_rendering"
    ]

    inputs = [
        StrInput(name="url", display_name="URL Address", info="The web address of the page to fetch.", required=True),
        StrInput(name="user_agent", display_name="User-Agent", info="Custom user-agent string for the request.", value="LangflowWebCrawler/1.0"),
        IntInput(name="max_content_length", display_name="Max Content Length", info="Max characters from text.", value=10000),
        DropdownInput(name="content_type_preference", display_name="Preferred Content Type", info="Content type to prioritize.", options=["auto", "json", "html"], value="auto"),
        BoolInput(name="include_metadata", display_name="Include Metadata", info="Extract metadata (title, description, etc.).", value=True),
        BoolInput(name="include_headings", display_name="Include Headings", info="Extract headings (h1-h6).", value=True),
        BoolInput(name="include_paragraphs", display_name="Include Paragraphs", info="Extract paragraph content.", value=True),
        BoolInput(name="extract_images", display_name="Extract Images", info="Extract image sources.", value=False),
        BoolInput(name="extract_links", display_name="Extract Links", info="Extract internal and external links.", value=False),
        DropdownInput(name="output_format", display_name="Output Format", info="Format for output data.", options=["structured", "flat_text", "table"], value="structured"),
        IntInput(name="max_depth", display_name="Max Crawl Depth", info="Max number of link levels to follow.", value=0),
        BoolInput(name="same_domain_only", display_name="Same Domain Only", info="Limit crawling to the same domain.", value=True),
        BoolInput(name="enable_js_rendering", display_name="Enable JS Rendering", info="Use JS rendering (requires requests-html).", value=False),
    ]

    outputs = [
        Output(name="structured_data", display_name="Structured Data", method="get_structured_data"),
        Output(name="table_data", display_name="Table Output", method="get_table_output"),
    ]

    def fetch_page(self, url):
        headers = {"User-Agent": self.user_agent}
        try:
            if self.enable_js_rendering and RENDERING_AVAILABLE:
                session = HTMLSession()
                response = session.get(url, headers=headers, timeout=10)
                response.html.render(timeout=20, sleep=2)
            else:
                response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except Exception as e:
            self.log(f"Failed to fetch {url}: {e}")
            return None

    def parse_html(self, html, url):
        soup = BeautifulSoup(html, "html.parser")
        data = {"url": url}

        if self.include_metadata:
            data["metadata"] = {
                "title": soup.title.string if soup.title else "",
                "description": "",
            }
            description = soup.find("meta", attrs={"name": "description"})
            if description and description.get("content"):
                data["metadata"]["description"] = description["content"]

        if self.include_headings:
            headings = {}
            for i in range(1, 7):
                headings[f"h{i}"] = [tag.get_text(strip=True).replace("\n", "") for tag in soup.find_all(f"h{i}")]
            data["headings"] = headings

        if self.include_paragraphs:
            paragraphs = [p.get_text(strip=True).replace("\n", "") for p in soup.find_all("p")]
            full_text = " ".join(paragraphs)
            if len(full_text) > self.max_content_length:
                full_text = full_text[:self.max_content_length] + "..."
            data["paragraphs"] = full_text

        if self.extract_images:
            data["images"] = [img.get("src") for img in soup.find_all("img") if img.get("src")]

        if self.extract_links:
            data["links"] = [urljoin(url, a.get("href")) for a in soup.find_all("a", href=True)]

        return data

    def get_robots_and_sitemap(self, base_url):
        domain = urlparse(base_url).scheme + "://" + urlparse(base_url).netloc
        robots_url = urljoin(domain, "/robots.txt")
        sitemap = []
        try:
            res = requests.get(robots_url, headers={"User-Agent": self.user_agent}, timeout=5)
            res.raise_for_status()
            for line in res.text.splitlines():
                if line.lower().startswith("sitemap"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        sitemap.append(parts[1].strip())
        except Exception as e:
            self.log(f"No robots.txt or error parsing it: {e}")
        return sitemap

    def crawl(self):
        visited = set()
        queue = deque([(self.url, 0)])
        domain = urlparse(self.url).netloc
        results = []

        sitemaps = self.get_robots_and_sitemap(self.url)
        if sitemaps:
            for sitemap_url in sitemaps:
                queue.append((sitemap_url, 0))

        while queue:
            current_url, depth = queue.popleft()
            if current_url in visited or depth > self.max_depth:
                continue

            self.log(f"Crawling {current_url} at depth {depth}")
            response = self.fetch_page(current_url)
            if not response:
                continue

            content_type = response.headers.get("Content-Type", "")
            visited.add(current_url)

            if self.content_type_preference == "json" or ("application/json" in content_type and self.content_type_preference == "auto"):
                try:
                    json_data = response.json()
                    results.append({"url": current_url, "json": json_data})
                    continue
                except Exception as e:
                    self.log(f"Error parsing JSON at {current_url}: {e}")
                    continue

            parsed_data = self.parse_html(response.text, current_url)
            results.append(parsed_data)

            if depth < self.max_depth:
                soup = BeautifulSoup(response.text, "html.parser")
                for a in soup.find_all("a", href=True):
                    link = urljoin(current_url, a["href"])
                    if self.same_domain_only and urlparse(link).netloc != domain:
                        continue
                    if link not in visited:
                        queue.append((link, depth + 1))
        return results

    def get_structured_data(self) -> Data:
        results = self.crawl()
        if self.output_format == "flat_text":
            flat = []
            for page in results:
                if "metadata" in page:
                    flat.append(f"Title: {page['metadata'].get('title', '')}")
                    flat.append(f"Description: {page['metadata'].get('description', '')}")
                if "headings" in page:
                    for level, items in page["headings"].items():
                        flat.extend(items)
                if "paragraphs" in page:
                    flat.append(page["paragraphs"])
            return Data(data={"text": "\n".join(flat)})
        return Data(data={"pages": results})

    def get_table_output(self) -> DataFrame:
        results = self.crawl()
        rows = []
        for page in results:
            row = {
                "url": page.get("url"),
                "metadata": page.get("metadata"),
                "paragraphs": page.get("paragraphs", ""),
                "headings": page.get("headings")
            }
            rows.append(row)
        df = pd.DataFrame(rows)
        return DataFrame(df)
