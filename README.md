# BlinkWell AI Chatbot System

A comprehensive AI-powered chatbot system for the BlinkWell web application, designed to provide instant answers to frequently asked questions about eye health, app features, and general support.

## Features

### 🤖 AI-Powered Chatbot
- **Intelligent Response Generation**: Uses TF-IDF vectorization and cosine similarity to find the best matching FAQ responses
- **Natural Language Processing**: Understands user intent and provides contextual responses
- **Real-time Chat Interface**: Modern, responsive chat UI with typing indicators and message history
- **Smart Suggestions**: Provides quick-access buttons for common questions

### 📚 Knowledge Base Management
- **Comprehensive FAQ System**: Organized by categories (General, Eye Detection, Habits, Account, Technical, Security, AI, Support)
- **Admin Interface**: Full CRUD operations for managing knowledge base items
- **Import/Export**: JSON-based import/export functionality for bulk operations
- **Search & Filtering**: Advanced search capabilities with category and status filters

### 🎨 Modern User Interface
- **Responsive Design**: Works seamlessly on desktop and mobile devices
- **Beautiful Animations**: Smooth transitions and hover effects
- **Accessibility**: Keyboard navigation and screen reader support
- **Real-time Updates**: Live chat with typing indicators and message timestamps

## Installation & Setup

### 1. Database Setup
Run the updated database schema to create the required tables:

```sql
-- Execute the contents of database.sql
-- This will create:
-- - knowledge_base table for FAQ storage
-- - chat_history table for conversation tracking
-- - Sample FAQ data
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

The chatbot requires these additional packages:
- `scikit-learn` - For TF-IDF vectorization and similarity matching
- `numpy` - For numerical operations

### 3. Configuration
Ensure your `config.py` contains the necessary database configuration:

```python
class Config:
    MYSQL_HOST = 'your_host'
    MYSQL_USER = 'your_username'
    MYSQL_PASSWORD = 'your_password'
    MYSQL_DB = 'your_database'
    # ... other config options
```

## Usage

### For Users

#### Accessing the Chatbot
1. **From Dashboard**: Click on "AI Assistant" in the sidebar
2. **Direct URL**: Navigate to `/chat`
3. **Quick Access**: Use the "Ask AI Assistant" button on the dashboard

#### Using the Chat Interface
1. **Type your question** in the chat input
2. **Use suggestion buttons** for quick access to common questions
3. **View real-time responses** with confidence scores
4. **Browse FAQ categories** for organized information

#### FAQ Page
- Access via `/chat/faq` or the FAQ button in the chat interface
- **Search functionality** to find specific questions
- **Category filtering** to browse by topic
- **Expandable answers** for better readability

### For Administrators

#### Knowledge Base Management
Access the admin panel at `/admin/knowledge-base`

**Adding New FAQ Items:**
1. Click "Add New Item"
2. Fill in question, answer, category, and keywords
3. Save to immediately update the chatbot

**Editing Existing Items:**
1. Click the "Edit" button on any FAQ item
2. Modify question, answer, category, or keywords
3. Toggle active/inactive status
4. Save changes

**Bulk Operations:**
- **Import**: Upload JSON files with multiple FAQ items
- **Export**: Download current knowledge base as JSON
- **Format**: Follow the provided JSON structure

#### Managing Categories
- Create new categories by typing them in the category field
- Organize FAQs by logical groupings
- Use consistent naming conventions

## Technical Architecture

### Backend Components

#### 1. ChatbotAI Class (`models/chatbot.py`)
- **TF-IDF Vectorization**: Converts text to numerical vectors for similarity matching
- **Cosine Similarity**: Calculates similarity between user input and knowledge base
- **Response Generation**: Provides contextual responses based on confidence thresholds
- **Chat History**: Stores conversation data for analytics

#### 2. API Endpoints (`routes/chatbot.py`)
- `POST /chat/api/send` - Process user messages and return AI responses
- `GET /chat/api/suggestions` - Get quick-access question suggestions
- `GET /chat/api/faq` - Retrieve FAQ data
- `POST /chat/api/refresh` - Refresh knowledge base from database

#### 3. Admin Routes (`routes/admin.py`)
- Full CRUD operations for knowledge base management
- Import/export functionality
- Search and filtering capabilities

### Frontend Components

#### 1. Chat Interface (`templates/chatbot/chat.html`)
- Real-time chat with typing indicators
- Message history and timestamps
- Suggestion buttons for common questions
- Responsive design for all devices

#### 2. FAQ Page (`templates/chatbot/faq.html`)
- Searchable FAQ database
- Category-based organization
- Expandable question/answer pairs
- Clean, intuitive interface

#### 3. Admin Panel (`templates/admin/knowledge_base.html`)
- Comprehensive management interface
- Real-time search and filtering
- Modal-based editing
- Statistics and analytics

## Customization

### Adding New FAQ Categories
1. Add new category names in the admin interface
2. Update the category tabs in `templates/chatbot/faq.html`
3. Consider adding category-specific icons

### Modifying Response Logic
Edit the `ChatbotAI.get_response()` method in `models/chatbot.py`:
- Add new greeting patterns
- Implement custom response logic
- Adjust confidence thresholds

### Styling Changes
- Modify CSS in the template files
- Update color schemes and gradients
- Adjust responsive breakpoints

## Performance Considerations

### Database Optimization
- Index the `question`, `answer`, and `category` columns
- Use `is_active` flag for soft deletes
- Consider pagination for large knowledge bases

### Caching Strategy
- Cache knowledge base vectors in memory
- Implement Redis for session storage
- Use CDN for static assets

### Scalability
- The TF-IDF approach scales well with knowledge base size
- Consider implementing vector databases for very large datasets
- Use async processing for chat history storage

## Security Features

### Access Control
- All chatbot routes require user authentication
- Admin functions restricted to authenticated users
- CSRF protection on form submissions

### Input Validation
- Sanitize user input to prevent XSS
- Validate file uploads for import functionality
- Rate limiting on API endpoints

## Monitoring & Analytics

### Chat Metrics
- Track user engagement and question patterns
- Monitor response confidence scores
- Analyze popular questions and categories

### Performance Metrics
- Response time monitoring
- Knowledge base hit rates
- User satisfaction scores

## Troubleshooting

### Common Issues

#### 1. Knowledge Base Not Loading
- Check database connection
- Verify table structure
- Ensure sample data is inserted

#### 2. Chatbot Not Responding
- Check API endpoint availability
- Verify scikit-learn installation
- Review error logs

#### 3. Admin Panel Access Issues
- Confirm user authentication
- Check route registration
- Verify blueprint registration

### Debug Mode
Enable Flask debug mode for detailed error messages:
```python
app.run(debug=True)
```

## Future Enhancements

### Planned Features
- **Multi-language Support**: Internationalization for global users
- **Voice Integration**: Speech-to-text and text-to-speech
- **Advanced NLP**: Integration with external AI services
- **Analytics Dashboard**: Detailed usage statistics and insights
- **A/B Testing**: Test different response strategies

### Integration Opportunities
- **Slack/Discord Bots**: Extend chatbot to team communication platforms
- **Mobile App**: Native mobile chatbot interface
- **Email Support**: Automated email response system
- **CRM Integration**: Connect with customer support systems

## Support & Contributing

### Getting Help
- Check the FAQ system first
- Review error logs and console output
- Test with sample data

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

### Code Standards
- Follow PEP 8 Python style guidelines
- Include docstrings for all functions
- Write comprehensive tests
- Update documentation for new features

---

**Note**: This chatbot system is designed to complement human support, not replace it. Always provide clear escalation paths to human agents for complex issues.