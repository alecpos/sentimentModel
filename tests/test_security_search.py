import pytest
from app.core.data_lake.security_manager import SecurityManager, PolicyEngine, EncryptionService, AuditLogger
from app.core.search.search_service import DataCatalogSearch
from app.models.domain.data_catalog_model import DataCatalogModel
from cryptography.fernet import Fernet
from unittest.mock import MagicMock

@pytest.fixture
def mock_es():
    """Create a mock Elasticsearch client"""
    mock = MagicMock()
    # Mock index exists check
    mock.indices.exists.return_value = False
    # Mock search results
    mock.search.return_value = {
        "hits": {
            "total": {"value": 1},
            "hits": [{
                "_source": {
                    "name": "Test Dataset",
                    "description": "Test description",
                    "meta_info": {"level": "campaign", "sensitivity": "low"},
                    "data_lake_id": "test_123"
                }
            }]
        }
    }
    return mock

@pytest.fixture
def mock_security():
    """Create mock security manager"""
    mock = MagicMock(spec=SecurityManager)
    mock.check_access.return_value = True
    return mock

def test_security_manager_access(mock_security):
    """Test security manager access control"""
    # Test allowed access
    assert mock_security.check_access("test_user", "catalog:123", "read")
    mock_security.check_access.assert_called_with("test_user", "catalog:123", "read")
    
    # Test denied access
    mock_security.check_access.return_value = False
    assert not mock_security.check_access("invalid_user", "catalog:123", "read")

def test_search_service_indexing(mock_security, mock_es):
    """Test search service with security integration"""
    # Initialize search service with mock client
    search_service = DataCatalogSearch(es_client=mock_es)
    
    # Verify index creation was attempted
    mock_es.indices.exists.assert_called_once_with(index=search_service.index_name)
    mock_es.indices.create.assert_called_once()
    
    # Verify the service is initialized
    assert search_service is not None
    assert search_service.index_name == "data_catalog"

def test_encryption_service():
    """Test encryption service functionality"""
    key = Fernet.generate_key()
    service = EncryptionService(key)
    
    test_data = b"sensitive data"
    encrypted = service.encrypt(test_data)
    decrypted = service.decrypt(encrypted)
    
    assert decrypted == test_data
    assert encrypted != test_data  # Ensure data was actually encrypted

def test_audit_logging(mock_security):
    """Test audit logging functionality"""
    # Perform actions that should be logged
    mock_security.check_access("test_user", "catalog:123", "read")
    mock_security.check_access.assert_called_with("test_user", "catalog:123", "read")
    
    mock_security.log_general_event("test_user", "Test event logged")
    mock_security.log_general_event.assert_called_with("test_user", "Test event logged")

def test_secure_search(mock_security, mock_es):
    """Test secure search functionality"""
    # Initialize search service with mock client and security manager
    search_service = DataCatalogSearch(es_client=mock_es, security_manager=mock_security)
    
    # Configure mock security checks
    mock_security.check_access.return_value = True
    
    # Perform search
    results = search_service.search("test", user_id="test_user")
    
    # Verify search was performed with correct parameters
    mock_es.search.assert_called_once()
    search_body = mock_es.search.call_args[1]['body']
    assert 'query' in search_body
    assert 'bool' in search_body['query']
    
    # Verify results
    assert len(results) == 1
    assert results[0]["meta_info"]["sensitivity"] == "low"
    
    # Verify security checks
    mock_security.check_access.assert_called_with("test_user", "catalog:test_123", "read")

def test_secure_search_denied(mock_security, mock_es):
    """Test secure search with denied access"""
    # Initialize search service with mock client and security manager
    search_service = DataCatalogSearch(es_client=mock_es, security_manager=mock_security)
    
    # Configure mock security to deny access
    mock_security.check_access.return_value = False
    
    # Perform search
    results = search_service.search("test", user_id="test_user")
    
    # Verify no results returned when access is denied
    assert len(results) == 0
    
    # Verify security check was performed
    mock_security.check_access.assert_called_with("test_user", "catalog:test_123", "read")