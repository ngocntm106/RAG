import logging
from typing import Optional, Dict, Any, List
from neo4j import GraphDatabase, Driver, Session
try:
    from .config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE
except ImportError:
    from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, NEO4J_DATABASE

logger = logging.getLogger(__name__)

class Neo4jClient:
    """
    Manages connections and basic operations with Neo4j graph database.
    """

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        database: str = NEO4J_DATABASE,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver: Optional[Driver] = None
        self._active_database: Optional[str] = None

    def connect(self) -> Driver:
        """Establish connection and verify connectivity to Neo4j instance."""
        if self.driver is None:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            
            # Determine available database
            self._active_database = self._resolve_database()
            logger.info(f"Connected to Neo4j at {self.uri}, using database: {self._active_database}")
        return self.driver

    def _resolve_database(self) -> str:
        """Verify if specified database exists, fallback to available default if needed."""
        try:
            with self.driver.session(database=self.database) as session:
                session.run("RETURN 1").consume()
                return self.database
        except Exception:
            # Fallback to default session/database
            try:
                with self.driver.session() as session:
                    res = session.run("SHOW DATABASES").data()
                    for db in res:
                        if db.get("name") == self.database:
                            return self.database
                        if db.get("default"):
                            return db.get("name", "neo4j")
            except Exception:
                pass
            return "neo4j"

    def get_session(self) -> Session:
        """Get an active session for the resolved database with auto-reconnect."""
        try:
            if self.driver is None:
                self.connect()
            else:
                self.driver.verify_connectivity()
        except Exception:
            if self.driver:
                try:
                    self.driver.close()
                except Exception:
                    pass
                self.driver = None
            self.connect()
        return self.driver.session(database=self._active_database)

    def close(self):
        """Close Neo4j driver connection."""
        if self.driver:
            self.driver.close()
            self.driver = None

    def get_database_statistics(self) -> Dict[str, Any]:
        """Fetch statistics: counts of nodes, labels, relationships, and indexes."""
        if self.driver is None:
            self.connect()

        stats = {
            "uri": self.uri,
            "database": self._active_database,
            "connected": True,
            "total_nodes": 0,
            "total_relationships": 0,
            "node_counts": {},
            "relationship_counts": {},
            "indexes": [],
        }

        with self.get_session() as session:
            # Total Nodes
            res = session.run("MATCH (n) RETURN count(n) AS total_nodes")
            record = res.single()
            stats["total_nodes"] = record["total_nodes"] if record else 0

            # Total Relationships
            res = session.run("MATCH ()-[r]->() RETURN count(r) AS total_rels")
            record = res.single()
            stats["total_relationships"] = record["total_rels"] if record else 0

            # Node Labels breakdown
            res = session.run("MATCH (n) RETURN labels(n) AS labels, count(*) AS count")
            for r in res:
                label_name = ":".join(r["labels"]) if r["labels"] else "Unlabeled"
                stats["node_counts"][label_name] = r["count"]

            # Relationship Types breakdown
            res = session.run("MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS count")
            for r in res:
                stats["relationship_counts"][r["rel_type"]] = r["count"]

            # Indexes
            res = session.run("SHOW INDEXES")
            for r in res:
                stats["indexes"].append({
                    "name": r.get("name"),
                    "type": r.get("type"),
                    "labels": r.get("labelsOrTypes"),
                    "properties": r.get("properties"),
                })

        return stats
