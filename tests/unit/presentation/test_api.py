"""
tests.unit.presentation.test_api | Layer: TEST
Unit tests for API endpoints.

## Relationships
# TESTS → presentation/api/app :: health endpoints
"""

import pytest
from fastapi.testclient import TestClient

from nexus_matcher.presentation.api.app import create_app


@pytest.fixture
def client():
    """Create test client."""
    app = create_app(configure_logs=False)
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns service info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "nexus-matcher"
        assert data["version"] == "2.0.0"
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("healthy", "degraded")
        assert "timestamp" in data
        assert data["version"] == "2.0.0"

    def test_liveness_endpoint(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "alive"

    def test_request_id_header(self, client):
        """Test that request ID is returned in response."""
        response = client.get("/health")
        assert "X-Request-ID" in response.headers
        assert "X-Response-Time-Ms" in response.headers

    def test_custom_request_id(self, client):
        """Test that custom request ID is preserved."""
        custom_id = "test-12345"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.headers["X-Request-ID"] == custom_id
