import pytest

@pytest.mark.unit
class TestSettings:
    """Test settings routes"""
    
    def test_settings_page_requires_auth(self, client):
        """Test settings page requires authentication"""
        response = client.get('/settings')
        assert response.status_code in [302, 401]
    
    def test_settings_page_loads(self, auth_client):
        """Test authenticated user can access settings"""
        response = auth_client.get('/settings')
        assert response.status_code == 200
    