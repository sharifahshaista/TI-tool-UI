"""
Test script to verify date filtering and analytical query handling
"""
from pathlib import Path
from datetime import datetime
import re

def parse_query_for_filters(query):
    """Parse natural language query to extract metadata filters and date ranges"""
    filters = {}
    date_filters = {}
    query_lower = query.lower()
    
    # Import datetime for date parsing
    from datetime import datetime, timedelta
    import re
    
    # Enhanced date parsing
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # Specific month/year patterns (e.g., "Nov 2025", "November 2025", "in 2025")
    month_patterns = {
        r'(january|jan)\s+(\d{4})': 1,
        r'(february|feb)\s+(\d{4})': 2,
        r'(march|mar)\s+(\d{4})': 3,
        r'(april|apr)\s+(\d{4})': 4,
        r'(may)\s+(\d{4})': 5,
        r'(june|jun)\s+(\d{4})': 6,
        r'(july|jul)\s+(\d{4})': 7,
        r'(august|aug)\s+(\d{4})': 8,
        r'(september|sept|sep)\s+(\d{4})': 9,
        r'(october|oct)\s+(\d{4})': 10,
        r'(november|nov)\s+(\d{4})': 11,
        r'(december|dec)\s+(\d{4})': 12
    }
    
    for pattern, month_num in month_patterns.items():
        match = re.search(pattern, query_lower)
        if match:
            year = int(match.group(2))
            # Date range for that month
            date_filters['year'] = year
            date_filters['month'] = month_num
            date_filters['date_after'] = f"{year}-{month_num:02d}-01"
            # Calculate last day of month
            if month_num == 12:
                next_month = 1
                next_year = year + 1
            else:
                next_month = month_num + 1
                next_year = year
            date_filters['date_before'] = f"{next_year}-{next_month:02d}-01"
            break
    
    # Year only patterns (e.g., "in 2025", "from 2025")
    if not date_filters:
        year_match = re.search(r'\b(in|from|during)\s+(\d{4})\b', query_lower)
        if year_match:
            year = int(year_match.group(2))
            date_filters['year'] = year
            date_filters['date_after'] = f"{year}-01-01"
            date_filters['date_before'] = f"{year + 1}-01-01"
    
    return filters, date_filters

# Test queries
test_queries = [
    "What are the technologies in trend (mentioned the most) in the articles in your knowledge base that are published in Nov 2025?",
    "Show me articles from November 2025",
    "What happened in 2025?",
    "Articles from December 2025",
    "Tell me about AI in Nov 2025",
    "Latest carbon tech news",
    "What are recent hydrogen developments?",
    "Show me TechCrunch articles from Oct 2025"
]

print("=" * 80)
print("DATE FILTER PARSING TESTS")
print("=" * 80)

for query in test_queries:
    print(f"\n Query: {query}")
    print("-" * 80)
    
    metadata_filters, date_filters = parse_query_for_filters(query)
    
    if date_filters:
        print(f"✓ Date filters detected:")
        for key, value in date_filters.items():
            print(f"   - {key}: {value}")
    else:
        print("  No date filters detected")
    
    if metadata_filters:
        print(f"  Metadata filters:")
        for key, value in metadata_filters.items():
            print(f"   - {key}: {value}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)

# Now test date filtering logic
print("\n" + "=" * 80)
print("DATE FILTERING LOGIC TEST")
print("=" * 80)

# Sample article dates
test_articles = [
    {"title": "Article 1", "date": "2025-11-15"},
    {"title": "Article 2", "date": "2025-10-20"},
    {"title": "Article 3", "date": "2025-11-28"},
    {"title": "Article 4", "date": "2024-11-15"},
    {"title": "Article 5", "date": "2025-12-01"},
]

# Test with Nov 2025 filter
_, date_filters = parse_query_for_filters("articles from Nov 2025")

print(f"\nFilter: Nov 2025")
print(f"Date range: {date_filters.get('date_after')} to {date_filters.get('date_before')}")
print("\nFiltered articles:")

for article in test_articles:
    pub_date_obj = datetime.strptime(article['date'], '%Y-%m-%d')
    
    # Check date filters
    include = True
    if 'date_after' in date_filters:
        after_date = datetime.strptime(date_filters['date_after'], '%Y-%m-%d')
        if pub_date_obj < after_date:
            include = False
    
    if 'date_before' in date_filters:
        before_date = datetime.strptime(date_filters['date_before'], '%Y-%m-%d')
        if pub_date_obj >= before_date:
            include = False
    
    status = "✓ INCLUDED" if include else "✗ EXCLUDED"
    print(f"  {status}: {article['title']} ({article['date']})")

print("\n" + "=" * 80)
