from __future__ import annotations
import os
import networkx as nx
from typing import Dict, Any, List, Optional

# Simple in-memory graph with optional Neo4j passthrough.
# Nodes: Entity:{type='entity', name}, Claim:{type='claim', chunk_id}, Source:{type='source', uri}
# Edges: (claim)-[MENTIONS]->(entity), (claim)-[FROM_SOURCE]->(source)


class GraphStore:
    def __init__(self):
        self.g = nx.MultiDiGraph()
        self.neo = None
        uri = os.getenv("NEO4J_URI")
        user = os.getenv("NEO4J_USER")
        pw = os.getenv("NEO4J_PASS")
        if uri and user and pw:
            try:
                from neo4j import GraphDatabase
                self.neo = GraphDatabase.driver(uri, auth=(user, pw))
            except Exception:
                self.neo = None

    def _neo_run(self, cypher: str, **params):
        if not self.neo:
            return
        try:
            with self.neo.session() as s:
                s.run(cypher, **params)
        except Exception:
            pass

    def add_entity(self, name: str):
        name = name.strip()
        if not self.g.has_node(("entity", name)):
            self.g.add_node(("entity", name), type="entity", name=name)
            self._neo_run("MERGE (:Entity {name:$name})", name=name)

    def add_source(self, uri: str):
        if not uri:
            return
        if not self.g.has_node(("source", uri)):
            self.g.add_node(("source", uri), type="source", uri=uri)
            self._neo_run("MERGE (:Source {uri:$uri})", uri=uri)

    def add_claim(self, chunk_id: str, uri: Optional[str], entities: List[str]):
        self.g.add_node(("claim", chunk_id), type="claim", chunk_id=chunk_id)
        if uri:
            self.add_source(uri)
            self.g.add_edge(("claim", chunk_id),
                            ("source", uri), key="FROM_SOURCE")
            self._neo_run("""
                MERGE (c:Claim {chunk_id:$cid})
                MERGE (s:Source {uri:$uri})
                MERGE (c)-[:FROM_SOURCE]->(s)
            """, cid=chunk_id, uri=uri)
        for e in entities:
            en = e.strip()
            if not en:
                continue
            self.add_entity(en)
            self.g.add_edge(("claim", chunk_id),
                            ("entity", en), key="MENTIONS")
            self._neo_run("""
                MERGE (c:Claim {chunk_id:$cid})
                MERGE (e:Entity {name:$name})
                MERGE (c)-[:MENTIONS]->(e)
            """, cid=chunk_id, name=en)

    def top_entities_from_chunks(self, chunk_entities: Dict[str, List[str]], top_n: int = 5) -> List[str]:
        freq = {}
        for ents in chunk_entities.values():
            for e in ents:
                freq[e] = freq.get(e, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:top_n]

    def summary(self) -> Dict[str, Any]:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "entities": [n[1] for n in self.g.nodes if isinstance(n, tuple) and n[0] == "entity"]
        }
