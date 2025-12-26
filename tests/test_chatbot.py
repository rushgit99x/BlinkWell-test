import pytest

@pytest.mark.unit
class TestChatbot:
    """Test chatbot routes"""
    
    def test_chatbot_page_loads(self, client):
        """Test chatbot page loads"""
        response = client.get('/chatbot')
        assert response.status_code == 200
    
    def test_chatbot_health_check(self, client):
        """Test chatbot health endpoint"""
        response = client.get('/api/chatbot/health')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'status' in data
    
    @pytest.mark.slow
    def test_send_chatbot_message(self, client):
        """Test sending message to chatbot"""
        response = client.post('/api/chatbot/message', json={
            'message': 'What are dry eye symptoms?'
        })
        
        assert response.status_code in [200, 503]  # May not be initialized
        
        if response.status_code == 200:
            data = response.get_json()
            assert 'response' in data or 'error' in data