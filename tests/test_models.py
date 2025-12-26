import pytest
from models.user import User, register_user, authenticate_user
from werkzeug.security import check_password_hash

@pytest.mark.database
class TestUserModel:
    """Test User model"""
    
    def test_user_creation(self):
        """Test creating a User object"""
        user = User(1, 'testuser', 'test@example.com')
        
        assert user.id == 1
        assert user.username == 'testuser'
        assert user.email == 'test@example.com'
    
    def test_user_is_authenticated(self):
        """Test User is_authenticated property"""
        user = User(1, 'testuser', 'test@example.com')
        assert user.is_authenticated is True
    
    def test_user_is_active(self):
        """Test User is_active property"""
        user = User(1, 'testuser', 'test@example.com')
        assert user.is_active is True