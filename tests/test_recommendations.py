import pytest

@pytest.mark.integration
class TestRecommendations:
    """Test recommendations functionality"""
    
    def test_recommendations_page_requires_auth(self, client):
        """Test recommendations page requires authentication"""
        response = client.get('/recommendations')
        assert response.status_code in [302, 401]
    
    def test_recommendations_page_loads(self, auth_client):
        """Test authenticated user can access recommendations"""
        response = auth_client.get('/recommendations')
        assert response.status_code == 200
    