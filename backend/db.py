"""Neo4j database connection manager for AtmoGraph with fast health probing."""
from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from dotenv import load_dotenv
from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError, ServiceUnavailable

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env", override=True)


class Neo4jConnection:
    """Manages a Neo4j driver instance across the FastAPI lifecycle."""

    def __init__(self) -> None:
        self._driver: Driver | None = None

    @property
    def database(self) -> str:
        return os.environ.get("NEO4J_DATABASE", "neo4j")

    def connect(self) -> Driver:
        if self._driver is None:
            uri = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
            user = os.environ.get("NEO4J_USER", "neo4j")
            password = os.environ.get("NEO4J_PASSWORD", "")
            if not password:
                raise ServiceUnavailable(
                    "NEO4J_PASSWORD is not set in f:\\atomgraph\\.env."
                )
            self._driver = GraphDatabase.driver(
                uri,
                auth=(user, password),
                connection_timeout=1.5,
                connection_acquisition_timeout=2.0,
                max_transaction_retry_time=1.0,
            )
        return self._driver

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def query(
        self, cypher: str, parameters: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        uri = os.environ.get("NEO4J_URI", "neo4j://127.0.0.1:7687")
        parsed = urlparse(uri)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 7687
        try:
            with socket.create_connection((host, port), timeout=0.25):
                pass
        except OSError as exc:
            self.close()
            raise ServiceUnavailable(f"Neo4j port {host}:{port} is not reachable") from exc

        try:
            driver = self.connect()
            with driver.session(database=self.database) as session:
                result = session.run(cypher, parameters or {})
                return [record.data() for record in result]
        except (ServiceUnavailable, Neo4jError, OSError):
            self.close()
            raise


db = Neo4jConnection()
