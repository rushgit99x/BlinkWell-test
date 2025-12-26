import pytest

@pytest.mark.auth
class TestRegistration:
    """Test user registration"""

    def test_register_page_loads(self, client):
        """Test registration page loads successfully"""
        response = client.get('/register')
        assert response.status_code == 200
        assert b'Create Account' in response.data or b'Register' in response.data

    def test_register_valid_user(self, client):
        """Test successful user registration"""
        response = client.post('/register', data={
            'username': 'newuser',
            'email': 'newuser@example.com',
            'password': 'password123'
        }, follow_redirects=True)

        assert response.status_code == 200
        assert b'Success' in response.data or b'Login' in response.data

    def test_register_invalid_email(self, client):
        """Test registration with invalid email"""
        response = client.post('/register', data={
            'username': 'invaliduser',
            'email': 'invalidemail',
            'password': 'password123'
        }, follow_redirects=True)
        assert response.status_code == 200

    def test_register_short_password(self, client):
        """Test registration with short password"""
        response = client.post('/register', data={
            'username': 'shortpass',
            'email': 'short@example.com',
            'password': '123'
        }, follow_redirects=True)
        assert response.status_code == 200


@pytest.mark.auth
class TestLogin:
    """Test user login"""

    def test_login_page_loads(self, client):
        """Test login page loads successfully"""
        response = client.get('/login')
        assert response.status_code == 200
        assert b'Login' in response.data or b'Sign in' in response.data

    def test_login_invalid_user(self, client):
        """Test login with invalid credentials"""
        response = client.post('/login', data={
            'username': 'nonexistent',
            'password': 'password123'
        }, follow_redirects=True)
        assert response.status_code == 200
        assert b'Invalid' in response.data or b'Error' in response.data
