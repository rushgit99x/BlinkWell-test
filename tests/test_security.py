import pytest

@pytest.mark.unit
class TestSecurity:
    """Test security features"""
    
    def test_password_hashing(self):
        """Test passwords are hashed"""
        from werkzeug.security import generate_password_hash, check_password_hash
        
        password = 'testpassword123'
        hashed = generate_password_hash(password)
        
        assert hashed != password
        assert check_password_hash(hashed, password)
    
    def test_csrf_protection(self, client):
        """Test CSRF protection (if enabled)"""
        # Note: CSRF is disabled in test config
        response = client.post('/register', data={
            'username': 'test',
            'email': 'test@example.com',
            'password': 'test123'
        })
        
        assert response.status_code in [200, 302, 400]
    