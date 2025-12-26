from flask import Blueprint, send_file, current_app
from flask_login import login_required, current_user
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch
import os
from datetime import datetime
from models.user import load_user, User

reports_bp = Blueprint('reports', __name__)

def create_user_report_pdf(user_id):
    """Generate a detailed PDF report for the user"""
    # Get user data using project's load_user helper
    user = load_user(user_id)
    if not user:
        return None
    
    # Create the PDF file path
    filename = f"user_report_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    pdf_path = os.path.join(current_app.root_path, 'static', 'exports', filename)
    
    # Create the PDF document
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Container for PDF elements
    elements = []
    
    # Styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    # Add title
    elements.append(Paragraph("BlinkWell Health Report", title_style))
    elements.append(Spacer(1, 12))
    
    # Add user information
    elements.append(Paragraph(f"User: {user.username}", styles["Heading2"]))
    elements.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles["Normal"]))
    elements.append(Spacer(1, 20))
    
    # Fetch recommendations and latest health stats from DB
    try:
        conn = current_app.config['get_db_connection']()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, category, recommendation_text, priority, status, created_at, updated_at, completed_at
            FROM user_recommendations
            WHERE user_id = %s
            ORDER BY created_at DESC, id DESC
        """, (user_id,))
        rec_results = cursor.fetchall()

        cursor.execute("""
            SELECT dry_eye_disease, risk_score
            FROM user_eye_health_data
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 1
        """, (user_id,))
        health_result = cursor.fetchone()

        cursor.close()
        conn.close()
    except Exception as e:
        current_app.logger.error(f"Error fetching user recommendations for PDF: {e}")
        rec_results = []
        health_result = None

    # Summary / Stats section
    elements.append(Paragraph("Summary", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    total_recs = len(rec_results)
    completed = sum(1 for r in rec_results if r[4] == 'completed')
    in_progress = sum(1 for r in rec_results if r[4] == 'in_progress')
    pending = total_recs - completed - in_progress
    has_dry_eyes = False
    risk_score = 0
    if health_result:
        has_dry_eyes = health_result[0] == 'Y'
        try:
            risk_score = float(health_result[1]) if health_result[1] is not None else 0
        except Exception:
            risk_score = 0

    stats_table = [
        ["Metric", "Value"],
        ["Total Recommendations", str(total_recs)],
        ["Completed", str(completed)],
        ["In Progress", str(in_progress)],
        ["Pending", str(pending)],
        ["Has Dry Eyes", "Yes" if has_dry_eyes else "No"],
        ["Risk Score", f"{risk_score}"],
    ]

    table = Table(stats_table, colWidths=[3.5*inch, 3.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 11),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.black),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 16))

    # Recommendations details
    elements.append(Paragraph("Recommendations Detail", styles["Heading2"]))
    elements.append(Spacer(1, 8))

    if not rec_results:
        elements.append(Paragraph("No recommendations available for this user.", styles['Normal']))
    else:
        # Group by category and render
        categories = {
            'immediate_actions': 'Immediate Actions',
            'medical_advice': 'Medical Advice',
            'lifestyle_changes': 'Lifestyle Changes',
            'monitoring': 'Monitoring'
        }

        recs_by_cat = {k: [] for k in categories}
        for r in rec_results:
            rec_obj = {
                'id': r[0],
                'category': r[1],
                'text': r[2],
                'priority': r[3],
                'status': r[4],
                'created_at': r[5].strftime('%Y-%m-%d %H:%M:%S') if r[5] else ''
            }
            if rec_obj['category'] in recs_by_cat:
                recs_by_cat[rec_obj['category']].append(rec_obj)

        for key, title in categories.items():
            items = recs_by_cat.get(key, [])
            elements.append(Paragraph(title, styles['Heading3']))
            elements.append(Spacer(1, 6))
            if not items:
                elements.append(Paragraph("No items.", styles['Normal']))
            else:
                # Render each recommendation as a small table row for clarity
                for idx, item in enumerate(items, 1):
                    rec_paragraph = Paragraph(f"{idx}. {item['text']}", styles['Normal'])
                    meta = Paragraph(f"Priority: {item['priority'].title()} — Status: {item['status'].replace('_', ' ').title()} — Created: {item['created_at']}", styles['Italic'])
                    elements.append(rec_paragraph)
                    elements.append(meta)
                    elements.append(Spacer(1, 6))
            elements.append(Spacer(1, 10))
    
    # Build the PDF
    doc.build(elements)
    
    return filename

@reports_bp.route('/download-report')
@login_required
def download_report():
    """Handle the download report request"""
    try:
        # Generate the report
        filename = create_user_report_pdf(current_user.id)
        if not filename:
            return "Error generating report", 500
            
        # Get the file path
        file_path = os.path.join(current_app.root_path, 'static', 'exports', filename)
        
        # Send the file
        return send_file(
            file_path,
            as_attachment=True,
            download_name=filename,
            mimetype='application/pdf'
        )
        
    except Exception as e:
        current_app.logger.error(f"Error generating report: {str(e)}")
        return "Error generating report", 500