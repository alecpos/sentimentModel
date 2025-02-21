from elasticsearch import Elasticsearch, NotFoundError
from app.config import settings
from typing import List, Dict, Any
from app.models import DataCatalogModel

class DataCatalogSearch:
    def __init__(self, es_host: str = settings.ELASTICSEARCH_HOST):
        self.es = Elasticsearch([es_host])
        self.index_name = "data_catalog"

    def search(self, query: str, max_results: int = 10) -> Dict[str, Any]:
        try:
            return self.es.search(
                index=self.index_name,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["name^3", "description", "schema.fields.name"]
                        }
                    },
                    "size": max_results
                }
            )
        except NotFoundError:
            return {"hits": {"hits": []}}
        except Exception as e:
            raise RuntimeError(f"Search error: {str(e)}")

    def index_entry(self, catalog_entry: 'DataCatalogModel'):
        try:
            self.es.index(
                index=self.index_name,
                id=catalog_entry.id,
                body={
                    "name": catalog_entry.name,
                    "description": catalog_entry.description,
                    "schema": catalog_entry.schema_definition,
                    "lineage": catalog_entry.lineage,
                    "version": catalog_entry.version
                }
            )
        except Exception as e:
            raise RuntimeError(f"Indexing error: {str(e)}")