from pydantic_ai import RunContext
import httpx
from bs4 import BeautifulSoup
import re
from typing import Optional


class SearxNGClient:
    def __init__(self, base_url: str, num_results: int = 10, excluded_domains: Optional[list[str]] = None):
        """Configure a SearxNG-backed web search tool."""
        self.base_url = base_url.rstrip("/")
        self.num_results = num_results
        self.excluded_domains = excluded_domains or ["reddit.com"]

    async def search_web(self, ctx: RunContext[None], query: str) -> str:
        """Search the web using SearXNG and summarise the results."""
        return await self._search(query)

    async def _search(self, query: str) -> str:
        search_url = f"{self.base_url}/search"
        params = {
            "q": query,
            "number_of_results": self.num_results,
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.get(search_url, params=params)
                response.raise_for_status()
                
                # Try JSON first, fall back to HTML parsing
                content_type = response.headers.get("content-type", "")
                if "application/json" in content_type:
                    try:
                        data = response.json()
                        return self._format_json_results(query, data)
                    except Exception:
                        pass
                
                # Parse HTML response
                return self._parse_html_results(query, response.text)
                
        except Exception as exc:  # pragma: no cover - network errors
            return f"Search failed: {exc}"

    def _clean_snippet(self, text: str) -> str:
        """Clean up snippet text to ensure complete sentences."""
        if not text or text == "No description":
            return text
        
        # Remove trailing ellipsis and incomplete sentences
        text = text.rstrip()
        
        # Remove trailing ... or …
        text = re.sub(r'\.{3,}$', '', text)
        text = re.sub(r'…+$', '', text)
        
        # If text doesn't end with sentence-ending punctuation, find last complete sentence
        if text and not re.search(r'[.!?]$', text):
            # Find the last sentence-ending punctuation
            last_sentence_end = max(
                text.rfind('.'),
                text.rfind('!'),
                text.rfind('?')
            )
            if last_sentence_end > 0:
                # Keep only up to the last complete sentence
                text = text[:last_sentence_end + 1]
        
        return text.strip()

    def _is_excluded_url(self, url: str) -> bool:
        """Check if URL should be excluded based on domain."""
        if not url:
            return False
        url_lower = url.lower()
        return any(domain in url_lower for domain in self.excluded_domains)

    def _format_json_results(self, query: str, data: dict) -> str:
        """Format JSON results from SearXNG."""
        results = data.get("results", [])
        if not results:
            return f"No results for '{query}'."

        lines = [f"Search results for '{query}':", ""]
        result_count = 0
        for item in results:
            url = item.get("url", "")
            
            # Skip excluded domains
            if self._is_excluded_url(url):
                continue
            
            result_count += 1
            if result_count > self.num_results:
                break
                
            title = item.get("title", "No title")
            content = item.get("content", "No description")
            content = self._clean_snippet(content)
            lines.extend(
                [
                    f"{result_count}. {title}",
                    f"{url}",
                    f"{content}\n",
                ]
            )
        
        if result_count == 0:
            return f"No results for '{query}'."
        
        return "\n".join(lines).rstrip()

    def _parse_html_results(self, query: str, html: str) -> str:
        """Parse HTML results from SearXNG."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Find all result articles
        results = soup.find_all("article", class_="result")
        
        if not results:
            return f"No results for '{query}'."
        
        lines = [f"Search results for '{query}':", ""]
        result_count = 0
        
        for result in results:
            # Extract URL first to check if it should be excluded
            url_elem = result.find("a", href=True)
            url = str(url_elem["href"]) if url_elem else ""
            
            # Skip excluded domains
            if self._is_excluded_url(url):
                continue
            
            result_count += 1
            if result_count > self.num_results:
                break
            
            # Extract title
            title_elem = result.find("h3")
            title = title_elem.get_text(strip=True) if title_elem else "No title"
            
            # Extract content/description
            content_elem = result.find("p", class_="content")
            content = content_elem.get_text(strip=True) if content_elem else "No description"
            
            # Clean up content (remove extra whitespace)
            content = re.sub(r'\s+', ' ', content)
            content = self._clean_snippet(content)
            
            lines.extend(
                [
                    f"{result_count}. {title}",
                    f"{url}",
                    f"{content}\n",
                ]
            )
        
        if result_count == 0:
            return f"No results for '{query}'."
        
        return "\n".join(lines).rstrip()

    def get_tool(self):
        """Return the search function for agent tool registration."""
        return self.search_web

if __name__ == "__main__":
    import asyncio

    # Test with your GCP instance
    SEARXNG_URL = "http://35.185.180.33:8080"
    searxng_client = SearxNGClient(SEARXNG_URL, num_results=20)
    
    # Run the async function directly using the internal _search method
    result = asyncio.run(
        searxng_client._search("CCS policy developments 2025")
    )
    print(type(result))
    print(result)