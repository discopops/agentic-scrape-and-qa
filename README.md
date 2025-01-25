# Agentic Scrape and QA

A tool for scraping documentation websites and performing intelligent Q&A using agentic RAG (Retrieval-Augmented Generation).

## Features

- **Website Crawling**: Automatically crawls documentation websites, with support for sitemap.xml
- **Semantic Chunking**: Intelligently splits content into meaningful chunks while preserving context
- **Rich Metadata**: Extracts and stores metadata like topics, technologies, and content types
- **Vector Search**: Uses OpenAI embeddings for semantic search
- **Agentic RAG**: Leverages LLMs for intelligent question answering with context
- **Source Management**: Manage multiple documentation sources independently

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/agentic-scrape-and-qa.git
cd agentic-scrape-and-qa
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables by copying the example:
```bash
cp .env.example .env
```

4. Edit `.env` with your API keys and configure your Supabase database

5. Run the SQL setup script in your Supabase database (site_pages.sql)

## Usage

Run the program:
```bash
python agentic_rag.py
```

### Features:

1. **Crawl a New Website**
   - Enter the base URL (e.g., https://docs.example.com)
   - Provide a unique identifier for this documentation
   - System will crawl pages, extract content, and store with metadata

2. **Q&A on Existing Documentation**
   - Select from available documentation sets
   - Ask questions naturally
   - Get context-aware responses with source URLs

3. **Manage Documentation Sets**
   - View all stored documentation sets
   - Delete specific sets when needed
   - Clean up outdated content

## Technical Details

- Uses OpenAI embeddings for semantic search
- Stores content and metadata in Supabase
- Implements vector similarity search
- Preserves code blocks and formatting
- Handles pagination and rate limiting

## Requirements

- Python 3.8+
- OpenAI API key
- Supabase account
- Packages listed in requirements.txt
