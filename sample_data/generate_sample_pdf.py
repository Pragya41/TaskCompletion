from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

def create_sample_pdf(output_path: str):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter

    # Page 1
    c.setFont("Helvetica-Bold", 18)
    c.drawString(72, height - 72, "Palm Mind AI - Engineering & Candidate Handbook")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 110, "1. Welcome to Palm Mind AI")
    c.setFont("Helvetica", 11)
    text_p1 = [
        "Welcome to the Palm Mind AI Engineering team. We design resilient AI systems",
        "for enterprise retrieval, conversational intelligence, and automated operations.",
        "Candidates interviewing with Palm Mind AI are evaluated on clean code, software",
        "architecture, and deep understanding of modern RAG pipelines.",
    ]
    y = height - 130
    for line in text_p1:
        c.drawString(72, y, line)
        y -= 18

    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y - 20, "2. Interview Scheduling Protocol")
    y -= 40
    text_p2 = [
        "Technical interviews can be scheduled directly through our conversational agent.",
        "Candidates must provide their full name, email address, proposed interview date",
        "(in YYYY-MM-DD format), and preferred time slot (e.g., 14:00 or 2:00 PM).",
        "Our hiring managers review submissions and confirm bookings within 48 hours.",
    ]
    for line in text_p2:
        c.drawString(72, y, line)
        y -= 18

    c.showPage()

    # Page 2
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, height - 72, "3. Core Values and Technical Stack")
    y = height - 100
    text_p3 = [
        "Our backend services are built with Python, FastAPI, and SQLAlchemy.",
        "For semantic indexing, we rely on Qdrant vector databases with cosine distance.",
        "For multi-turn session tracking, we leverage Redis memory buffers.",
        "We prioritize type safety, modular architecture, and end-to-end testing.",
    ]
    for line in text_p3:
        c.drawString(72, y, line)
        y -= 18

    c.save()
    print(f"Generated sample PDF at: {output_path}")

if __name__ == "__main__":
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_file = os.path.join(out_dir, "palm_mind_handbook.pdf")
    create_sample_pdf(out_file)
