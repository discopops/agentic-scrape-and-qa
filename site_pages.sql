-- Enable the pgvector extension
create extension if not exists vector;

-- Create the documentation chunks table
create table site_pages (
    id bigserial primary key,
    url varchar not null,
    chunk_number integer not null,
    title varchar not null,
    summary varchar not null,
    content text not null,
    metadata jsonb not null default '{}'::jsonb,
    embedding vector(1536),  -- OpenAI embeddings are 1536 dimensions
    created_at timestamp with time zone default timezone('utc'::text, now()) not null,
    
    -- Add a unique constraint to prevent duplicate chunks for the same URL
    unique(url, chunk_number)
);

-- Create an index for better vector similarity search performance
create index on site_pages using ivfflat (embedding vector_cosine_ops);

-- Create an index on metadata for faster filtering
create index idx_site_pages_metadata on site_pages using gin (metadata);

-- Create a function to search for documentation chunks
create function match_site_pages (
  query_embedding vector(1536),
  match_count int default 10,
  filter jsonb DEFAULT '{}'::jsonb
) returns table (
  id bigint,
  url varchar,
  chunk_number integer,
  title varchar,
  summary varchar,
  content text,
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    url,
    chunk_number,
    title,
    summary,
    content,
    metadata,
    1 - (site_pages.embedding <=> query_embedding) as similarity
  from site_pages
  where metadata @> filter
  order by site_pages.embedding <=> query_embedding
  limit match_count;
end;
$$;

-- Create a function to get unique sources and their domains
create function get_unique_sources()
returns table (
  source text,
  domain text,
  url_count bigint,
  last_crawled timestamptz
)
language plpgsql
as $$
begin
  return query
  select 
    metadata->>'source' as source,
    metadata->>'domain' as domain,
    count(distinct url) as url_count,
    max(created_at) as last_crawled
  from site_pages
  where metadata->>'source' is not null
  group by metadata->>'source', metadata->>'domain'
  order by max(created_at) desc;
end;
$$;

-- Enable RLS but make it unrestricted since this is a local development setup
alter table site_pages enable row level security;
create policy "Unrestricted access" on site_pages for all using (true);