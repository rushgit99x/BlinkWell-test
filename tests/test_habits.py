import pytest

@pytest.mark.unit
class TestHabits:
    """Test habits routes"""
    
    def test_habits_page_requires_auth(self, client):
        """Test habits page requires authentication"""
        response = client.get('/habits')
        assert response.status_code in [302, 401]
    
    def test_habits_page_loads(self, auth_client):
        """Test authenticated user can access habits"""
        response = auth_client.get('/habits')
        assert response.status_code == 200
    
    def test_get_available_habits(self, auth_client):
        """Test getting available habits"""
        response = auth_client.get('/api/habits/available')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'success' in data or 'habits' in data
    
    def test_get_user_habits(self, auth_client):
        """Test getting user's habits"""
        response = auth_client.get('/api/habits/user-habits')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'success' in data or 'habits' in data
