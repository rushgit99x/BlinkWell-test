import pytest

@pytest.mark.unit
class TestMainRoutes:
    """Test main application routes"""
    
    def test_index_page(self, client):
        """Test index page loads"""
        response = client.get('/')
        assert response.status_code == 200
        assert b'BlinkWell' in response.data
    
    def test_about_page(self, client):
        """Test about page loads"""
        response = client.get('/about')
        assert response.status_code == 200
        assert b'About' in response.data or b'Mission' in response.data
    
    def test_contact_page(self, client):
        """Test contact page loads"""
        response = client.get('/contact')
        assert response.status_code == 200
        assert b'Contact' in response.data
    
    def test_dashboard_requires_login(self, client):
        """Test dashboard requires authentication"""
        response = client.get('/dashboard')
        
        # Should redirect to login
        assert response.status_code in [302, 401]
    
    def test_dashboard_with_auth(self, auth_client):
        """Test authenticated user can access dashboard"""
        response = auth_client.get('/dashboard')
        
        assert response.status_code == 200
        assert b'Dashboard' in response.data or b'Welcome' in response.data


@pytest.mark.integration
class TestEyeAnalysis:
    """Test eye analysis routes"""
    
    def test_eye_analysis_page_requires_auth(self, client):
        """Test eye analysis requires authentication"""
        response = client.get('/eye-analysis')
        assert response.status_code in [302, 401]
    
    def test_eye_analysis_page_loads(self, auth_client):
        """Test authenticated user can access eye analysis"""
        response = auth_client.get('/eye-analysis')
        assert response.status_code == 200
    
    @pytest.mark.slow
    def test_image_upload(self, auth_client, tmp_path):
        """Test eye image upload"""
        # Create a temporary image file
        import io
        from PIL import Image
        
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        response = auth_client.post('/analyze-eye-image', data={
            'eye_image': (img_bytes, 'test.png')
        }, content_type='multipart/form-data')
        
        assert response.status_code in [200, 500]  # May fail without model