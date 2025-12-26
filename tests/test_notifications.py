import pytest

@pytest.mark.unit
class TestEmailService:
    """Test email service"""
    
    def test_email_test_page(self, auth_client):
        """Test email test page loads"""
        response = auth_client.get('/email-test')
        assert response.status_code == 200
    
    @pytest.mark.slow
    def test_send_test_email(self, auth_client, mocker):
        """Test sending test email"""
        # Mock the email service
        mock_send = mocker.patch('services.email_service.email_service.send_email')
        mock_send.return_value = True
        
        response = auth_client.post('/api/notifications/test-email', json={
            'email_type': 'welcome'
        })
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['success'] is True


@pytest.mark.integration
class TestNotificationScheduler:
    """Test notification scheduler"""
    
    def test_scheduler_status(self, auth_client):
        """Test getting scheduler status"""
        response = auth_client.get('/api/notifications/status')
        assert response.status_code == 200
        
        data = response.get_json()
        assert 'status' in data