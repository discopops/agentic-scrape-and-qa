import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from openai import AsyncOpenAI
from supabase import create_client, Client

load_dotenv()

# Initialize OpenAI and Supabase clients
openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, Any]:
    """Extract title, summary, and metadata using GPT-4."""
    system_prompt = """You are an expert at analyzing documentation content.
    For the given content chunk, analyze and provide the following information in this exact format:

    TITLE: (create a descriptive title, or use the actual title if this is a page start)
    SUMMARY: (write a detailed summary of the key points)
    TOPICS: (list the main topics discussed, comma-separated)
    TYPE: (specify the content type: tutorial, reference, guide, api, overview, etc.)
    TECHNOLOGIES: (list any programming languages, frameworks, tools mentioned, comma-separated)

    Keep titles concise but informative.
    For summaries, focus on key points and technical details.
    For topics and technologies, be specific and thorough.
    """
    
    try:
        response = await openai_client.chat.completions.create(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
            ]
        )

        # Parse the structured response
        content = response.choices[0].message.content
        result = {}
        current_key = None
        
        for line in content.split('\n'):
            if line.strip():
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    key = key.strip().lower()
                    result[key] = value.strip()
                elif current_key:
                    result[current_key] += ' ' + line.strip()
                    
        # Extract topics and technologies for metadata
        topics = [t.strip() for t in result.get('topics', '').split(',') if t.strip()]
        technologies = [t.strip() for t in result.get('technologies', '').split(',') if t.strip()]
        content_type = result.get('type', 'unknown').lower()
        
        return {
            "title": result.get('title', 'Error processing title'),
            "summary": result.get('summary', 'Error processing summary'),
            "content_type": content_type,
            "topics": topics,
            "technologies": technologies
        }
        
    except Exception as e:
        print(f"Error getting title and summary: {e}")
        return {
            "title": "Error processing title",
            "summary": "Error processing summary",
            "content_type": "unknown",
            "topics": [],
            "technologies": []
        }

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from OpenAI."""
    try:
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return [0] * 1536  # Return zero vector on error

async def process_chunk(chunk: str, chunk_number: int, url: str, source_type: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title, summary, and metadata
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": source_type,
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path,
        "domain": urlparse(url).netloc,
        "content_type": extracted['content_type'],
        "topics": extracted['topics'],
        "technologies": extracted['technologies']
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            "url": chunk.url,
            "chunk_number": chunk.chunk_number,
            "title": chunk.title,
            "summary": chunk.summary,
            "content": chunk.content,
            "metadata": chunk.metadata,
            "embedding": chunk.embedding
        }
        
        result = supabase.table("site_pages").insert(data).execute()
        print(f"Inserted chunk {chunk.chunk_number} for {chunk.url}")
        return result
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str, source_type: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url, source_type) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_website(base_url: str, source_type: str, max_concurrent: int = 5):
    """Crawl website starting from base URL."""
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        # Try to get sitemap first
        sitemap_urls = []
        try:
            sitemap_url = f"{base_url}/sitemap.xml"
            response = requests.get(sitemap_url)
            response.raise_for_status()
            root = ElementTree.fromstring(response.content)
            namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
            sitemap_urls = [loc.text for loc in root.findall('.//ns:loc', namespace)]
        except:
            # If no sitemap, just use the base URL
            sitemap_urls = [base_url]
            print("No sitemap found, crawling from base URL")

        print(f"Found {len(sitemap_urls)} URLs to crawl")
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown, source_type)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        await asyncio.gather(*[process_url(url) for url in sitemap_urls])
    finally:
        await crawler.close()

def get_available_sites() -> List[Dict[str, Any]]:
    """Get list of unique sites that have been crawled."""
    try:
        # Get all distinct sources and domains from metadata
        result = supabase.table('site_pages')\
            .select('metadata, created_at, url')\
            .execute()
            
        sites = {}
        for row in result.data:
            source = row['metadata'].get('source')
            domain = row['metadata'].get('domain')
            if source and domain:
                if source not in sites:
                    sites[source] = {'source': source, 'domain': domain}
        return list(sites.values())
    except Exception as e:
        print(f"Error getting available sites: {e}")
        return []

async def delete_site_content(source: str) -> bool:
    """Delete all content for a specific source from Supabase."""
    try:
        # First get count of records to be deleted
        count_result = supabase.table('site_pages')\
            .select('*', count='exact')\
            .eq('metadata->>source', source)\
            .execute()
            
        if not count_result.count:
            print(f"No content found for source: {source}")
            return False
            
        # Delete the records
        result = supabase.table('site_pages')\
            .delete()\
            .eq('metadata->>source', source)\
            .execute()
        print(f"Deleted {count_result.count} records for source: {source}")
        return True
    except Exception as e:
        print(f"Error deleting content: {e}")
        return False
async def main():
    while True:
        print("\nAgentic RAG System")
        print("1. Crawl a new website")
        print("2. Perform Q&A on existing site")
        print("3. Delete site content")
        print("4. Exit")
        
        choice = input("Choose an option (1-4): ")
        
        if choice == "1":
            base_url = input("\nEnter the base URL to crawl (e.g., https://example.com): ")
            source_type = input("Enter a name/identifier for this documentation source: ")
            
            print(f"\nStarting crawl of {base_url}")
            await crawl_website(base_url, source_type)
            print("Crawl complete!")
            
        elif choice == "2":
            # Get available sites
            sites = get_available_sites()
            if not sites:
                print("\nNo sites available for Q&A. Please crawl a site first.")
                continue
                
            print("\nAvailable sites:")
            for i, site in enumerate(sites, 1):
                print(f"{i}. {site['source']} ({site['domain']})")
            
            site_choice = input("\nChoose a site number: ")
            try:
                selected_site = sites[int(site_choice) - 1]
                print(f"\nSelected: {selected_site['source']}")
                
                # Import here to avoid circular imports
                from pydantic_ai_expert import pydantic_ai_expert, PydanticAIDeps
                
                deps = PydanticAIDeps(
                    supabase=supabase,
                    openai_client=openai_client,
                    selected_source=selected_site['source']  # Set the selected source
                )

                while True:
                    question = input("\nEnter your question (or 'exit' to go back to main menu): ")
                    if question.lower() == 'exit':
                        break
                        
                    async with pydantic_ai_expert.run_stream(
                        question,
                        deps=deps
                    ) as result:
                        async for chunk in result.stream_text(delta=True):
                            print(chunk, end='', flush=True)
                        print("\n")
                        
            except (ValueError, IndexError):
                print("Invalid selection")
                
        elif choice == "3":
            # Get available sites
            sites = get_available_sites()
            if not sites:
                print("\nNo sites available to delete.")
                continue
                
            print("\nAvailable sites:")
            for i, site in enumerate(sites, 1):
                print(f"{i}. {site['source']} ({site['domain']})")
            
            site_choice = input("\nChoose a site number to delete (or 'cancel'): ")
            
            if site_choice.lower() == 'cancel':
                continue
                
            try:
                selected_site = sites[int(site_choice) - 1]
                confirm = input(f"\nAre you sure you want to delete all content for {selected_site['source']}? (yes/no): ")
                
                if confirm.lower() == 'yes':
                    await delete_site_content(selected_site['source'])
                
            except (ValueError, IndexError):
                print("Invalid selection")
                
        elif choice == "4":
            print("\nGoodbye!")
            break
            
        else:
            print("\nInvalid choice, please try again")

if __name__ == "__main__":
    asyncio.run(main())