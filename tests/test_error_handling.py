import pytest

@pytest.mark.unit
class TestErrorHandling:
    """Test error handling"""
    
    def test_404_error(self, client):
        """Test 404 error for non-existent route"""
        response = client.get('/nonexistent-route')
        assert response.status_code == 404
    
    def test_invalid_method(self, client):
        """Test invalid HTTP method"""
        response = client.delete('/register')
        assert response.status_code in [405, 404]
    
    def test_malformed_json(self, auth_client):
        """Test handling of malformed JSON"""
        response = auth_client.post('/api/chatbot/message',
                                   data='invalid json',
                                   content_type='application/json')
        
        assert response.status_code in [400, 500]
