"""
Generate a clean synthetic English Master Services Agreement PDF for end-to-end testing.
"""
import os
import fitz  # PyMuPDF


def create_sample_contract(pdf_path: str = "data/sample_contract.pdf"):
    os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)  # A4

    text = """MASTER SERVICES AGREEMENT

This Master Services Agreement ("Agreement") is entered into on January 15, 2024 ("Effective Date"), by and between Acme Corp, a Delaware corporation ("Client"), and TechSolutions LLC, a California limited liability company ("Provider").

ARTICLE 1 - SCOPE OF SERVICES
Provider agrees to perform professional cloud engineering, software architecture, and AI development services as described in each Statement of Work ("SOW").

ARTICLE 2 - TERM AND DURATION
The term of this Agreement shall commence on the Effective Date and shall continue for an initial period of thirty-six (36) months, unless earlier terminated in accordance with Article 6.

ARTICLE 3 - FEES AND PAYMENT TERMS
Client shall pay Provider an annual service fee of $120,000 USD, payable in equal monthly installments of $10,000 USD within thirty (30) days of invoice receipt. Late payments shall accrue interest at the rate of 1.5% per month.

ARTICLE 4 - INTELLECTUAL PROPERTY RIGHTS
All deliverables, work product, source code, and inventions developed by Provider under this Agreement shall be deemed "work made for hire" and shall be the exclusive property of Client.

ARTICLE 5 - INDEMNIFICATION AND LIABILITIES
Provider shall indemnify, defend, and hold harmless Client against any third-party claims arising out of intellectual property infringement or gross negligence. Late delivery penalties shall be assessed at $150 USD per business day of delay.

ARTICLE 6 - TERMINATION
Either party may terminate this Agreement for material breach upon thirty (30) days' prior written notice if the breach remains uncured. Client may terminate for convenience upon sixty (60) days' written notice.

ARTICLE 7 - GOVERNING LAW AND JURISDICTION
This Agreement shall be governed by and construed in accordance with the laws of the State of Delaware, without regard to its conflict of law principles. Any dispute shall be resolved exclusively in the state and federal courts located in Wilmington, Delaware.
"""

    rect = fitz.Rect(50, 50, 545, 792)
    page.insert_textbox(rect, text, fontsize=10, fontname="helv", align=0)

    doc.save(pdf_path)
    doc.close()
    print(f"✅ Generated sample English contract at: {pdf_path}")


if __name__ == "__main__":
    create_sample_contract()
