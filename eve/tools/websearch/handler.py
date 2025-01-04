import os
from typing import Dict, Optional, List, Any, Tuple
from playwright.async_api import async_playwright
import json
import logging

async def safe_evaluate(page, script: str, default_value: Any) -> Tuple[Any, str]:
    """
    Safely evaluate JavaScript on the page with error handling.
    """
    try:
        result = await page.evaluate(script)
        return result, ""
    except Exception as e:
        return default_value, str(e)

async def handler(args: dict, env: str = None) -> Dict[str, str]:
    """
    Handler function for the websearch tool that scrapes content from specified URLs.
    
    Args:
        args (dict): Dictionary containing:
            - url (str): Required. The URL to scrape
            - max_links (int): Optional. Maximum number of links to extract (default: 15)
            - max_chars (int): Optional. Maximum number of characters to include in content summary (default: 2000)
        env (str): Optional environment parameter
    
    Returns:
        Dict[str, str]: Dictionary containing the scraped content and any errors
    """
    url = args.get('url')
    if not url:
        raise ValueError("URL parameter is required")
        
    # Get configurable limits from args with defaults
    max_links = int(args.get('max_links', 15))
    max_chars = int(args.get('max_chars', 2000))

    page_content = {
        "title": "",
        "text": "",
        "links": []
    }
    
    errors = []
    
    try:
        async with async_playwright() as p:
            # Launch browser with security options
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-gpu', '--no-sandbox', '--disable-dev-shm-usage']
            )
            
            # Create new page (timeout is set per operation, not during page creation)
            context = await browser.new_context(
                viewport={'width': 1280, 'height': 800}
            )
            page = await context.new_page()
            
            # Set default timeout for all operations
            page.set_default_timeout(30000)
            
            # Navigate to URL with retry
            for attempt in range(3):
                try:
                    response = await page.goto(
                        url, 
                        wait_until='domcontentloaded',
                        timeout=10000
                    )
                    if response and response.ok:
                        break
                except Exception as e:
                    if attempt == 2:
                        return {"output": f"Error loading page after 3 attempts: {str(e)}"}
                    continue
            
            # Extract page title (with fallback)
            try:
                page_content["title"] = await page.title()
            except Exception as e:
                page_content["title"] = "Title extraction failed"
                errors.append(f"Title error: {str(e)}")
            
            # Extract visible text content with multiple strategies
            text_extraction_script = """
                () => {
                    try {
                        let text = Array.from(document.body.querySelectorAll('p, h1, h2, h3, h4, h5, h6, article, section, main'))
                            .map(element => element.textContent.trim())
                            .filter(text => text.length > 0)
                            .join('\\n\\n');
                            
                        if (text.length < 100) {
                            text = Array.from(document.body.getElementsByTagName('*'))
                                .map(element => element.textContent.trim())
                                .filter(text => text.length > 20)
                                .join('\\n\\n');
                        }
                        
                        return text;
                    } catch (error) {
                        return '';
                    }
                }
            """
            
            text_content, text_error = await safe_evaluate(page, text_extraction_script, "")
            if text_error:
                errors.append(f"Text extraction error: {text_error}")
            page_content["text"] = text_content if text_content else "No text content could be extracted"
            
            # Extract links with fallback strategies
            links_script = """
                () => {
                    try {
                        let links = Array.from(document.links)
                            .map(link => ({
                                text: link.textContent.trim(),
                                href: link.href
                            }))
                            .filter(link => link.text && link.href.startsWith('http'));
                            
                        if (links.length === 0) {
                            links = Array.from(document.getElementsByTagName('a'))
                                .map(a => ({
                                    text: a.textContent.trim(),
                                    href: a.getAttribute('href')
                                }))
                                .filter(link => link.text && link.href && link.href.startsWith('http'));
                        }
                        
                        return links;
                    } catch (error) {
                        return [];
                    }
                }
            """
            
            links, links_error = await safe_evaluate(page, links_script, [])
            if links_error:
                errors.append(f"Links extraction error: {links_error}")
            page_content["links"] = links[:max_links] if links else []
            
            # Close browser
            await browser.close()
            
            # Format output with error reporting
            output = f"""
# Page Analysis: {url}

## Title
{page_content['title']}

## Content Summary
{page_content['text'][:max_chars]}{'...' if len(page_content['text']) > max_chars else ''}

## Top Links
"""
            if page_content["links"]:
                for link in page_content["links"]:
                    output += f"- [{link['text']}]({link['href']})\n"
            else:
                output += "No links were extracted\n"

            if errors:
                output += "\n## Extraction Issues\n"
                output += "Some content may be incomplete due to the following issues:\n"
                for error in errors:
                    output += f"- {error}\n"
            
            return {
                "output": output
            }
            
    except Exception as e:
        return {
            "output": f"Critical error processing webpage: {str(e)}\n\nPartial content (if any):\n{json.dumps(page_content, indent=2)}"
        }