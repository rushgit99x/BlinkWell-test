from authlib.integrations.flask_client import OAuth
from authlib.common.security import generate_token

def init_oauth(app):
    oauth = OAuth(app)
    
    # Configure Google OAuth
    oauth.register(
        name='google',
        client_id=app.config['GOOGLE_CLIENT_ID'],
        client_secret=app.config['GOOGLE_CLIENT_SECRET'],
        server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
        client_kwargs={
            'scope': 'openid email profile'
        }
    )
    
    # Store oauth client in app config for access in routes
    app.config['OAUTH'] = oauth