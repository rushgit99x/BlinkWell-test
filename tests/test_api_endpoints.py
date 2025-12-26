import pytest

@pytest.mark.api
class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_habits_api_available(self, auth_client):
        """Test habits API is available"""
        response = auth_client.get('/api/habits/available')
        assert response.status_code == 200
        assert response.content_type == 'application/json'
    
    def test_habits_api_dashboard_stats(self, auth_client):
        """Test habits dashboard stats API"""
        response = auth_client.get('/api/habits/dashboard-stats')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'success' in data or 'stats' in data
    
    def test_chatbot_api_message(self, client):
        """Test chatbot message API"""
        response = client.post('/api/chatbot/message', json={
            'message': 'Hello'
        })
        
        assert response.status_code in [200, 503]
        assert response.content_type == 'application/json'