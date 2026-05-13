#!/usr/bin/env python3
"""Quick test for openalex_data module."""
import sys, json
sys.path.insert(0, ".")
from api.openalex_data import get_real_stats, search_openalex, build_real_graph, get_authors, get_author_by_slug, get_works_for_author

# Test 1: Stats
stats = get_real_stats()
print("=== STATS ===")
print(f"Authors: {stats['total_authors']}")
print(f"Works: {stats['total_works']}")
print(f"Topics: {stats['total_topics']}")
print(f"Institutions: {stats['total_institutions']}")
print(f"Citations: {stats['total_citations']}")
print(f"Top cited: {stats['top_cited_papers'][0]['title'][:60]}... ({stats['top_cited_papers'][0]['citations']} cites)")

# Test 2: Search
print("\n=== SEARCH: relativity ===")
r = search_openalex("relativity")
print(f"Authors: {len(r['authors'])}, Papers: {len(r['papers'])}, Topics: {len(r['topics'])}")
if r["authors"]:
    print(f"  First: {r['authors'][0]['display_name']}")
    print(f"  Explanation: {r['authors'][0]['explanation']}")
if r["papers"]:
    print(f"  First paper: {r['papers'][0]['title'][:60]}...")
    print(f"  Explanation: {r['papers'][0]['explanation']}")

# Test 3: Author lookup
print("\n=== AUTHOR LOOKUP ===")
a = get_author_by_slug("Miguel_Alcubierre")
print(f"Found: {a['display_name'] if a else 'NOT FOUND'}")
w = get_works_for_author("Miguel_Alcubierre")
print(f"Works: {len(w)}")

# Test 4: Graph
print("\n=== GRAPH ===")
g = build_real_graph()
s = g["stats"]
print(f"Nodes: {s['total_nodes']}, Edges: {s['total_edges']}")
print(f"  Authors: {s['authors']}, Papers: {s['papers']}, Institutions: {s['institutions']}, Topics: {s['topics']}")

# Test 5: Search UNAM
print("\n=== SEARCH: UNAM ===")
r2 = search_openalex("unam")
print(f"Authors: {len(r2['authors'])}, Papers: {len(r2['papers'])}, Institutions: {len(r2['institutions'])}")
if r2["authors"]:
    print(f"  First: {r2['authors'][0]['display_name']}, explanation: {r2['authors'][0]['explanation']}")

print("\n=== ALL TESTS PASSED ===")
