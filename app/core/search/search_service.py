from elasticsearch import Elasticsearch
from typing import Dict, Any, List
from datetime import datetime, timedelta
from app.config import settings
from app.models.domain.data_catalog_model import DataCatalogModel

class DataCatalogSearch:
    def __init__(self, es_client=None, security_manager=None):
        """Initialize search service with optional client for testing"""
        self.es = es_client if es_client is not None else Elasticsearch([settings.ELASTICSEARCH_URL])
        self.security_manager = security_manager
        self.index_name = "data_catalog"
        self._ensure_index()

    def _ensure_index(self):
        """Ensure index exists with proper mappings"""
        if not self.es.indices.exists(index=self.index_name):
            mappings = {
                "mappings": {
                    "properties": {
                        "name": {"type": "text", "analyzer": "standard"},
                        "description": {"type": "text", "analyzer": "standard"},
                        "schema": {"type": "object"},
                        "meta_info": {
                            "type": "object",
                            "properties": {
                                "level": {"type": "keyword"},
                                "data_layer": {"type": "keyword"},
                                "sensitivity": {"type": "keyword"}
                            }
                        },
                        "level_context": {"type": "object"},
                        "lineage": {"type": "object"},
                        "data_lake_id": {"type": "keyword"},
                        "created_at": {"type": "date"},
                        "updated_at": {"type": "date"}
                    }
                }
            }
            self.es.indices.create(index=self.index_name, body=mappings)

    def index_entry(self, entry: DataCatalogModel):
        """Index a catalog entry for searching"""
        document = {
            "name": entry.name,
            "description": entry.description,
            "schema": entry.schema_definition if hasattr(entry, 'schema_definition') else None,
            "meta_info": entry.meta_info,
            "level_context": entry.level_context,
            "lineage": entry.lineage if hasattr(entry, 'lineage') else None,
            "data_lake_id": entry.data_lake_id,
            "created_at": entry.created_at.isoformat() if entry.created_at else None,
            "updated_at": entry.updated_at.isoformat() if entry.updated_at else None
        }
        self.es.index(index=self.index_name, document=document)

    def bulk_index(self, entries: List[DataCatalogModel]):
        """Bulk index multiple catalog entries"""
        actions = []
        for entry in entries:
            # Add the action
            actions.append({"index": {"_index": self.index_name}})
            # Add the document with consistent structure
            document = {
                "name": entry.name,
                "description": entry.description,
                "schema": entry.schema_definition if hasattr(entry, 'schema_definition') else None,
                "meta_info": entry.meta_info,
                "level_context": entry.level_context,
                "lineage": entry.lineage if hasattr(entry, 'lineage') else None,
                "data_lake_id": entry.data_lake_id,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "updated_at": entry.updated_at.isoformat() if entry.updated_at else None
            }
            actions.append(document)
        if actions:
            self.es.bulk(body=actions)

    def search(self, query: str, filters: Dict[str, Any] = None, user_id: str = None) -> List[Dict[str, Any]]:
        """Search catalog entries with optional filters and security checks"""
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["name^2", "description", "schema.*"]
                            }
                        }
                    ]
                }
            }
        }

        if filters:
            for field, value in filters.items():
                search_body["query"]["bool"].setdefault("filter", []).append(
                    {"term": {f"meta_info.{field}": value}}
                )

        results = self.es.search(index=self.index_name, body=search_body)
        hits = results["hits"]["hits"]

        # Apply security filtering if security manager is available
        if self.security_manager and user_id:
            filtered_hits = []
            for hit in hits:
                source = hit["_source"]
                if self.security_manager.check_access(user_id, f"catalog:{source['data_lake_id']}", "read"):
                    filtered_hits.append(source)
            return filtered_hits
        
        return [hit["_source"] for hit in hits]

    def search_by_level(self, level: str, query: str = None, user_id: str = None) -> List[Dict[str, Any]]:
        """Search entries by data level with optional text query and security checks"""
        search_body = {
            "query": {
                "bool": {
                    "must": [{"term": {"meta_info.level": level}}]
                }
            }
        }

        if query:
            search_body["query"]["bool"]["must"].append({
                "multi_match": {
                    "query": query,
                    "fields": ["name^2", "description"]
                }
            })

        results = self.es.search(index=self.index_name, body=search_body)
        hits = results["hits"]["hits"]

        # Apply security filtering if security manager is available
        if self.security_manager and user_id:
            filtered_hits = []
            for hit in hits:
                source = hit["_source"]
                if self.security_manager.check_access(user_id, f"catalog:{source['data_lake_id']}", "read"):
                    filtered_hits.append(source)
            return filtered_hits

        return [hit["_source"] for hit in hits]

    def get_recent_updates(self, days: int = 7, user_id: str = None) -> List[Dict[str, Any]]:
        """Get recently updated catalog entries with security checks"""
        cutoff_date = datetime.now() - timedelta(days=days)
        search_body = {
            "query": {
                "range": {
                    "updated_at": {
                        "gte": cutoff_date.isoformat()
                    }
                }
            },
            "sort": [{"updated_at": "desc"}]
        }
        results = self.es.search(index=self.index_name, body=search_body)
        hits = results["hits"]["hits"]

        # Apply security filtering if security manager is available
        if self.security_manager and user_id:
            filtered_hits = []
            for hit in hits:
                source = hit["_source"]
                if self.security_manager.check_access(user_id, f"catalog:{source['data_lake_id']}", "read"):
                    filtered_hits.append(source)
            return filtered_hits

        return [hit["_source"] for hit in hits]

    def search_with_facets(self, query: str, facets: List[str] = None, filters: Dict[str, Any] = None, user_id: str = None) -> Dict[str, Any]:
        """Advanced search with faceted results and aggregations.
        
        Args:
            query: Search query string
            facets: List of fields to facet on
            filters: Dictionary of filters to apply
            user_id: Optional user ID for security filtering
            
        Returns:
            Dict containing hits and facets
        """
        search_body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["name^2", "description", "schema.*"]
                            }
                        }
                    ]
                }
            },
            "aggs": {}
        }

        # Add facet aggregations
        if facets:
            for facet in facets:
                search_body["aggs"][facet] = {
                    "terms": {"field": f"meta_info.{facet}"}
                }

        # Add filters
        if filters:
            for field, value in filters.items():
                search_body["query"]["bool"].setdefault("filter", []).append(
                    {"term": {f"meta_info.{field}": value}}
                )

        results = self.es.search(index=self.index_name, body=search_body)
        hits = results["hits"]["hits"]
        
        # Apply security filtering if needed
        if self.security_manager and user_id:
            filtered_hits = []
            for hit in hits:
                source = hit["_source"]
                if self.security_manager.check_access(user_id, f"catalog:{source['data_lake_id']}", "read"):
                    filtered_hits.append(source)
            hits = filtered_hits

        return {
            "hits": [hit["_source"] for hit in hits],
            "facets": {
                name: {
                    "buckets": bucket["buckets"]
                } for name, bucket in results.get("aggregations", {}).items()
            }
        }