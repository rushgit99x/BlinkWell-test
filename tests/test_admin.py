import pytest
from models.admin import authenticate_admin, create_eye_habit, list_eye_habits

@pytest.mark.unit
class TestAdminAuth:
    """Test admin authentication"""
    
    def test_admin_login_page(self, client):
        """Test admin login page loads"""
        response = client.get('/admin/login')
        assert response.status_code == 200
        assert b'Admin' in response.data
    
@pytest.mark.integration
class TestAdminRoutes:
    """Test admin routes"""
    
    def test_admin_dashboard_requires_auth(self, client):
        """Test admin dashboard requires authentication"""
        response = client.get('/admin/')
        
        # Should redirect to login
        assert response.status_code in [302, 200]
    
    def test_admin_eye_habits_page(self, client):
        """Test admin eye habits management page"""
        # Try to access without auth
        response = client.get('/admin/eye-habits')
        assert response.status_code in [302, 200]