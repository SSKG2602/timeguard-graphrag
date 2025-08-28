# Graph storage layer for entity relationships and knowledge representation
# Provides both in-memory NetworkX graph and optional Neo4j persistence

from __future__ import annotations
import os
import networkx as nx
from typing import Dict, Any, List, Optional

# Graph structure:
# Nodes: Entity:{type='entity', name}, Claim:{type='claim', chunk_id}, Source:{type='source', uri}  
# Edges: (claim)-[MENTIONS]->(entity), (claim)-[FROM_SOURCE]->(source)


class GraphStore:
    """
    Hybrid graph storage system supporting both in-memory and persistent graph operations
    Uses NetworkX for fast in-memory operations with optional Neo4j backend for persistence
    """
    
    def __init__(self):
        # Primary in-memory graph using NetworkX MultiDiGraph
        # MultiDiGraph allows multiple edges between same nodes with different keys
        self.g = nx.MultiDiGraph()
        
        # Optional Neo4j connection for persistent storage
        self.neo = None
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")  
        pw = os.getenv("NEO4J_PASS")
        
        # Initialize Neo4j connection if credentials provided
        if uri and user and pw:
            try:
                from neo4j import GraphDatabase
                self.neo = GraphDatabase.driver(uri, auth=(user, pw))
            except Exception:
                # Silently continue without Neo4j if connection fails
                self.neo = None

    def _neo_run(self, cypher: str, **params):
        """
        Execute Cypher query on Neo4j database if available
        Fails silently if Neo4j is not configured or connection fails
        
        Args:
            cypher: Cypher query string
            **params: Query parameters
        """
        if not self.neo:
            return
        try:
            with self.neo.session() as s:
                s.run(cypher, **params)
        except Exception:
            # Fail silently - graph operations continue with in-memory only
            pass

    def add_entity(self, name: str):
        """
        Add entity node to the graph
        Entities represent named entities extracted from documents (people, orgs, places, etc.)
        
        Args:
            name: Entity name/identifier
        """
        name = name.strip()
        if not self.g.has_node(("entity", name)):
            # Add to in-memory graph
            self.g.add_node(("entity", name), type="entity", name=name)
            # Persist to Neo4j if available
            self._neo_run("MERGE (:Entity {name:$name})", name=name)

    def add_source(self, uri: str):
        """
        Add source node to the graph
        Sources represent document origins or provenance information
        
        Args:
            uri: Source URI or identifier
        """
        if not uri:
            return
        if not self.g.has_node(("source", uri)):
            # Add to in-memory graph
            self.g.add_node(("source", uri), type="source", uri=uri)
            # Persist to Neo4j if available
            self._neo_run("MERGE (:Source {uri:$uri})", uri=uri)

    def add_claim(self, chunk_id: str, uri: Optional[str], entities: List[str]):
        """
        Add claim node and establish relationships to entities and sources
        Claims represent chunks of text that mention entities and come from sources
        
        Args:
            chunk_id: Unique identifier for the text chunk
            uri: Source URI (optional)
            entities: List of entity names mentioned in this claim
        """
        # Add claim node to in-memory graph
        self.g.add_node(("claim", chunk_id), type="claim", chunk_id=chunk_id)
        
        # Link claim to source if provided
        if uri:
            self.add_source(uri)
            self.g.add_edge(("claim", chunk_id),
                            ("source", uri), key="FROM_SOURCE")
            # Persist relationship to Neo4j
            self._neo_run("""
                MERGE (c:Claim {chunk_id:$cid})
                MERGE (s:Source {uri:$uri})
                MERGE (c)-[:FROM_SOURCE]->(s)
            """, cid=chunk_id, uri=uri)
        
        # Link claim to all mentioned entities
        for e in entities:
            en = e.strip()
            if not en:
                continue
            
            self.add_entity(en)
            self.g.add_edge(("claim", chunk_id),
                            ("entity", en), key="MENTIONS")
            # Persist relationship to Neo4j
            self._neo_run("""
                MERGE (c:Claim {chunk_id:$cid})
                MERGE (e:Entity {name:$name})
                MERGE (c)-[:MENTIONS]->(e)
            """, cid=chunk_id, name=en)

    def top_entities_from_chunks(self, chunk_entities: Dict[str, List[str]], top_n: int = 5) -> List[str]:
        """
        Find most frequently mentioned entities across a set of chunks
        Used for query expansion in multi-hop retrieval
        
        Args:
            chunk_entities: Mapping from chunk_id to list of entities
            top_n: Maximum number of top entities to return
        
        Returns:
            List of entity names sorted by frequency (descending)
        """
        # Count entity frequencies across all chunks
        freq = {}
        for ents in chunk_entities.values():
            for e in ents:
                freq[e] = freq.get(e, 0) + 1
        
        # Return top entities by frequency
        return sorted(freq, key=freq.get, reverse=True)[:top_n]

    def summary(self) -> Dict[str, Any]:
        """
        Generate summary statistics about the current graph state
        Useful for monitoring and debugging graph growth
        
        Returns:
            Dictionary with graph statistics and entity list
        """
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "entities": [n[1] for n in self.g.nodes if isinstance(n, tuple) and n[0] == "entity"]
        }
