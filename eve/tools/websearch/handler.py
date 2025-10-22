from playwright.async_api import async_playwright
import asyncio
from collections import Counter


async def wait_for_content(page) -> bool:
    """Wait for page to load and stabilize."""
    try:
        # Wait for body to have content
        await page.wait_for_selector("body", timeout=3000)
        # Wait a bit for dynamic content
        await page.wait_for_timeout(1000)
        return True
    except:
        return False


async def get_text_content(page) -> str:
    """Extract all visible text content from the page."""
    script = """
    () => {
        // Helper function to check if element is visible
        const isVisible = el => {
            if (!el.offsetParent && el.tagName !== 'BODY') return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && 
                   style.visibility !== 'hidden' && 
                   style.opacity !== '0' &&
                   style.width !== '0px' &&
                   style.height !== '0px';
        };

        // Helper to check if element is likely navigation/footer
        const isBoilerplate = el => {
            const tag = el.tagName.toLowerCase();
            if (['nav', 'header', 'footer'].includes(tag)) return true;
            
            const classes = el.className.toLowerCase();
            const id = el.id.toLowerCase();
            const boilerplateTerms = ['menu', 'nav', 'header', 'footer', 'sidebar', 'cookie', 'popup', 'modal'];
            
            return boilerplateTerms.some(term => 
                classes.includes(term) || id.includes(term)
            );
        };

        // Get all text nodes in the document
        const walker = document.createTreeWalker(
            document.body,
            NodeFilter.SHOW_TEXT,
            {
                acceptNode: function(node) {
                    // Skip if parent is hidden
                    if (!isVisible(node.parentElement)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    
                    // Skip if parent is boilerplate
                    if (isBoilerplate(node.parentElement)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    
                    // Skip script and style contents
                    if (['SCRIPT', 'STYLE', 'NOSCRIPT'].includes(node.parentElement.tagName)) {
                        return NodeFilter.FILTER_REJECT;
                    }
                    
                    // Accept if node has meaningful content
                    return node.textContent.trim().length > 0 
                        ? NodeFilter.FILTER_ACCEPT 
                        : NodeFilter.FILTER_REJECT;
                }
            }
        );

        const textNodes = [];
        while (walker.nextNode()) {
            const node = walker.currentNode;
            const text = node.textContent.trim();
            
            // Get the closest block-level ancestor
            let ancestor = node.parentElement;
            while (ancestor && window.getComputedStyle(ancestor).display === 'inline') {
                ancestor = ancestor.parentElement;
            }
            
            // Add text with its structural context
            if (ancestor) {
                const tag = ancestor.tagName.toLowerCase();
                if (tag.match(/^h[1-6]$/)) {
                    textNodes.push(`\n## ${text}\n`);
                } else if (tag === 'p') {
                    textNodes.push(`${text}\n\n`);
                } else {
                    textNodes.push(`${text} `);
                }
            }
        }

        return textNodes.join('').trim();
    }
    """
    try:
        return await page.evaluate(script)
    except:
        return ""


async def handler(args: dict, user: str = None, agent: str = None, session: str = None):
    """
    Performance-optimized web scraper using Playwright.
    """
    url = args["url"]
    max_links = int(args.get("max_links", 5))
    max_chars = int(args.get("max_chars", 20000))

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"],
        )

        context = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
        )

        try:
            page = await context.new_page()
            await page.goto(url, wait_until="networkidle", timeout=10000)

            # Wait for content to load
            await wait_for_content(page)

            # Extract data in parallel
            title_future = page.title()
            content_future = get_text_content(page)

            # Modified link extraction to be more generic
            links_script = """
                () => {
                    const isVisible = el => {
                        if (!el.offsetParent && el.tagName !== 'BODY') return false;
                        const style = window.getComputedStyle(el);
                        return style.display !== 'none' && 
                               style.visibility !== 'hidden' && 
                               style.opacity !== '0';
                    };
                    
                    return Array.from(document.querySelectorAll('a[href]'))
                        .filter(a => isVisible(a))
                        .map(a => ({
                            text: (a.innerText || a.textContent || '').trim(),
                            href: a.href
                        }))
                        .filter(({text, href}) => 
                            text && 
                            text.length > 1 &&
                            href.startsWith('http') &&
                            !href.includes('/cdn-cgi/') &&
                            !text.match(/^(Accept|Cookie|Got it|×|Close)/i)
                        );
                }
            """
            links_future = page.evaluate(links_script)

            # Wait for all operations to complete
            title, text_content, links = await asyncio.gather(
                title_future, content_future, links_future
            )

            # Format output
            output = f"""# Page Analysis: {url}

## Title
{title}

## Content Summary
{text_content[:max_chars]}{"..." if len(text_content) > max_chars else ""}

## Top Links
"""
            # Count frequency of each link
            link_counts = Counter(link["href"] for link in links)

            # Sort links by frequency, then by text length for ties
            sorted_links = sorted(
                links, key=lambda x: (-link_counts[x["href"]], -len(x["text"]))
            )

            # Add links with their counts
            seen_links = set()
            for link in sorted_links[:max_links]:
                if link["href"] not in seen_links:
                    count = link_counts[link["href"]]
                    output += (
                        f"- [{link['text']}]({link['href']}) (appears {count} times)\n"
                    )
                    seen_links.add(link["href"])

            return {"output": output}

        except Exception as e:
            return {"output": f"Error processing webpage: {str(e)}"}

        finally:
            await browser.close()
