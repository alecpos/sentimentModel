import pytest
from unittest.mock import MagicMock
from app.core.search.search_service import DataCatalogSearch
from app.models.domain.data_catalog_model import DataCatalogModel
from app.models.domain.data_lake_model import DataLakeModel
from datetime import datetime, timedelta
import json

@pytest.fixture
def mock_es():
    """Create a mock Elasticsearch client"""
    mock = MagicMock()
    # Mock index exists check
    mock.indices.exists.return_value = False
    
    # Mock index operation
    def mock_index(index, document, **kwargs):
        return {"_index": index, "_id": "test_id", "result": "created"}
    mock.index = MagicMock(side_effect=mock_index)
    
    # Mock bulk operation
    def mock_bulk(body, **kwargs):
        return {
            "took": 30,
            "errors": False,
            "items": [{"index": {"_index": "test_index", "_id": "test_id", "status": 201}} for _ in range(len(body)//2)]
        }
    mock.bulk = MagicMock(side_effect=mock_bulk)
    
    # Mock search results
    mock.search.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [{
                "_source": {
                    "name": "Test Dataset",
                    "description": "Test description",
                    "meta_info": {"level": "campaign"}
                }
            }]
        }
    }
    return mock

@pytest.fixture
def search_service(mock_es):
    """Create search service with mocked Elasticsearch"""
    return DataCatalogSearch(es_client=mock_es)

@pytest.fixture
def sample_entries(db_session):
    """Create sample catalog entries for testing"""
    entries = []
    
    # Create a data lake entry first
    lake_entry = DataLakeModel(
        name="test_lake",
        data=b"test_data",
        meta_info={"level": "campaign", "data_layer": "raw"}
    )
    db_session.add(lake_entry)
    db_session.flush()

    # Create multiple catalog entries with different characteristics
    entry_data = [
        {
            "name": "Campaign Performance Data",
            "description": "Daily campaign metrics and KPIs",
            "meta_info": {"level": "campaign", "data_layer": "raw"},
            "level_context": {
                "start_date": "2024-01-01",
                "end_date": "2024-12-31",
                "target_demographic": "all"
            }
        },
        {
            "name": "User Behavior Analysis",
            "description": "User interaction and engagement metrics",
            "meta_info": {"level": "user", "data_layer": "processed"},
            "level_context": {
                "user_segments": ["active", "new"],
                "age_range": "18-35",
                "geography": "global"
            }
        },
        {
            "name": "System Performance Logs",
            "description": "Infrastructure and application metrics",
            "meta_info": {"level": "environment", "data_layer": "raw"},
            "level_context": {
                "infra_version": "v2.1",
                "deployment_region": "us-west"
            }
        }
    ]

    for data in entry_data:
        entry = DataCatalogModel(
            data_lake_id=lake_entry.id,
            **data
        )
        entries.append(entry)
        db_session.add(entry)
    
    db_session.commit()
    return entries

class TestSearchService:
    """Test suite for catalog search functionality"""

    def test_index_creation(self, search_service, mock_es):
        """Test proper index creation with mappings"""
        # Verify index creation was attempted
        mock_es.indices.exists.assert_called_once_with(index=search_service.index_name)
        mock_es.indices.create.assert_called_once()
        
        # Verify the service is initialized
        assert search_service is not None
        assert search_service.index_name == "data_catalog"

    def test_entry_indexing(self, search_service, mock_es, sample_entries):
        """Test indexing of catalog entries"""
        # Index the sample entries
        for entry in sample_entries:
            search_service.index_entry(entry)
        
        # Verify index calls
        assert mock_es.index.call_count == len(sample_entries)
        
        # Verify the last index call arguments
        last_entry = sample_entries[-1]
        mock_es.index.assert_called_with(
            index=search_service.index_name,
            document={
                'name': last_entry.name,
                'description': last_entry.description,
                'meta_info': last_entry.meta_info,
                'level_context': last_entry.level_context,
                'data_lake_id': last_entry.data_lake_id,
                'created_at': last_entry.created_at.isoformat() if last_entry.created_at else None,
                'updated_at': last_entry.updated_at.isoformat() if last_entry.updated_at else None,
                'schema': last_entry.schema_definition if hasattr(last_entry, 'schema_definition') else None,
                'lineage': last_entry.lineage if hasattr(last_entry, 'lineage') else None
            }
        )

    def test_basic_search(self, search_service, mock_es):
        """Test basic search functionality"""
        # Setup mock search response
        mock_es.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "name": "Test Dataset",
                        "description": "Test description",
                        "meta_info": {"level": "campaign"}
                    }
                }]
            }
        }
        
        # Perform search
        results = search_service.search("test")
        
        # Verify search was called with correct parameters
        mock_es.search.assert_called_once()
        assert len(results) == 1
        assert results[0]["name"] == "Test Dataset"

    def test_filtered_search(self, search_service, mock_es):
        """Test search with filters"""
        # Setup mock response for filtered search
        mock_es.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "name": "Campaign Data",
                        "meta_info": {"level": "campaign"}
                    }
                }]
            }
        }
        
        # Perform filtered search
        results = search_service.search("campaign", filters={"level": "campaign"})
        
        # Verify search call and results
        mock_es.search.assert_called_once()
        assert len(results) == 1
        assert results[0]["meta_info"]["level"] == "campaign"

    def test_level_specific_search(self, search_service, mock_es):
        """Test level-specific search functionality"""
        # Setup mock response
        mock_es.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_source": {
                            "name": "Campaign 1",
                            "meta_info": {"level": "campaign"}
                        }
                    },
                    {
                        "_source": {
                            "name": "Campaign 2",
                            "meta_info": {"level": "campaign"}
                        }
                    }
                ]
            }
        }
        
        # Search for campaign level entries
        results = search_service.search_by_level("campaign")
        
        # Verify search parameters and results
        mock_es.search.assert_called_once()
        assert len(results) == 2
        assert all(r["meta_info"]["level"] == "campaign" for r in results)

    def test_recent_updates(self, search_service, mock_es):
        """Test retrieval of recently updated entries"""
        # Setup mock response
        mock_es.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "name": "Recent Entry",
                        "updated_at": datetime.now().isoformat()
                    }
                }]
            }
        }
        
        # Get recent updates
        results = search_service.get_recent_updates(days=7)
        
        # Verify search call and results
        mock_es.search.assert_called_once()
        assert len(results) == 1
        assert "updated_at" in results[0]

    def test_search_result_format(self, search_service, mock_es):
        """Test search result formatting"""
        # Setup detailed mock response
        mock_es.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "name": "Test Entry",
                        "description": "Test Description",
                        "meta_info": {"level": "campaign"},
                        "level_context": {"date": "2024-01-01"},
                        "created_at": "2024-01-01T00:00:00",
                        "updated_at": "2024-01-02T00:00:00"
                    }
                }]
            }
        }
        
        # Perform search
        results = search_service.search("test")
        
        # Verify result format
        assert len(results) == 1
        result = results[0]
        assert all(key in result for key in [
            "name", "description", "meta_info", 
            "level_context", "created_at", "updated_at"
        ])

    def test_search_relevance(self, search_service, mock_es):
        """Test search result relevance scoring"""
        # Setup mock response with scores
        mock_es.search.return_value = {
            "hits": {
                "total": {"value": 2},
                "hits": [
                    {
                        "_score": 2.0,
                        "_source": {"name": "Exact Match"}
                    },
                    {
                        "_score": 1.0,
                        "_source": {"name": "Partial Match"}
                    }
                ]
            }
        }
        
        # Perform search
        results = search_service.search("match")
        
        # Verify results are ordered by relevance
        assert len(results) == 2
        assert results[0]["name"] == "Exact Match"  # Higher score should be first

    def test_search_indexing(self, search_service, mock_es, sample_entries):
        """Test bulk indexing functionality"""
        # Perform bulk indexing
        search_service.bulk_index(sample_entries)
        
        # Verify bulk indexing call
        mock_es.bulk.assert_called_once()
        
        # Verify the number of entries indexed
        call_args = mock_es.bulk.call_args
        assert call_args is not None
        bulk_actions = call_args.kwargs.get('body', [])
        assert len(bulk_actions) == len(sample_entries) * 2  # Each entry needs an action and a source

    def test_level_filtering(self, search_service, mock_es):
        """Test filtering by data level"""
        # Setup mock response for level filtering
        mock_es.search.return_value = {
            "hits": {
                "total": {"value": 1},
                "hits": [{
                    "_source": {
                        "name": "Campaign Entry",
                        "meta_info": {"level": "campaign", "data_layer": "raw"}
                    }
                }]
            }
        }
        
        # Search with level and layer filters
        results = search_service.search(
            "campaign",
            filters={
                "level": "campaign",
                "data_layer": "raw"
            }
        )
        
        # Verify filter application
        mock_es.search.assert_called_once()
        assert len(results) == 1
        assert results[0]["meta_info"]["level"] == "campaign"
        assert results[0]["meta_info"]["data_layer"] == "raw"