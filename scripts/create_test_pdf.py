"""
Generate a realistic small test PDF for French contract parsing and RAG testing
with proper text wrapping.
"""
import fitz  # PyMuPDF
import os

def create_sample_contract(output_path: str = "data/test_contract.pdf"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = fitz.open()

    # Page 1
    page1 = doc.new_page()
    text_p1 = """CONTRAT DE BAIL COMMERCIAL ET PHOTOVOLTAÏQUE

Entre les soussignés :
La société SOLAIRE PROVENCE SAS, au capital de 100 000 €, dont le siège social est situé à Lyon (69000), ci-après désignée "Le Preneur".
Et :
Monsieur Jean DUPONT, propriétaire du site situé à Lentilly (69210), ci-après désigné "Le Bailleur".

ARTICLE 1 - OBJET ET DESIGNATION DU SITE
Le présent bail a pour objet la mise à disposition par le Bailleur au Preneur de la toiture du bâtiment agricole d'une superficie de 1 200 m², situé au 15 Route de Paris, 69210 Lentilly, pour l'installation et l'exploitation d'une centrale photovoltaïque d'une puissance crête de 250 kWc.

ARTICLE 2 - DUREE DU BAIL
Le présent contrat de bail est conclu pour une durée ferme et irrévocable de trente (30) ans à compter de la date de signature des présentes.
Il prendra effet au 1er septembre 2024 et expirera le 31 août 2054.

ARTICLE 3 - LOYER ET CONDITIONS FINANCIERES
En contrepartie de la mise à disposition du site, le Preneur s'engage à verser au Bailleur une redevance annuelle forfaitaire de 12 500 € (douze mille cinq cents euros) hors taxes.
Ce loyer sera indexé annuellement sur l'indice ILAT publié par l'INSEE. Le premier paiement interviendra à la date de mise en service de la centrale.
"""
    rect1 = fitz.Rect(50, 50, 550, 800)
    page1.insert_textbox(rect1, text_p1, fontsize=10, fontname="helv")

    # Page 2
    page2 = doc.new_page()
    text_p2 = """ARTICLE 4 - OBLIGATIONS ET ASSURANCES DU PRENEUR
Le Preneur s'engage à concevoir, construire, exploiter et entretenir à ses frais exclusifs l'installation photovoltaïque.
Le Preneur devra souscrire une police d'assurance responsabilité civile professionnelle couvrant les dommages matériels et corporels à hauteur minimale de 5 000 000 € par sinistre.

ARTICLE 5 - PENALITES ET RETARD DE MISE EN SERVICE
En cas de retard de mise en service imputable au Preneur excédant un délai de six (6) mois après l'obtention des autorisations administratives purgées de tout recours, une pénalité forfaitaire de 50 € par jour de retard sera due au Bailleur.

ARTICLE 6 - RESILIATION ET FIN DE CONTRAT
Le contrat pourra être résilié de plein droit par l'une des parties en cas d'inexécution grave d'une obligation contractuelle, trente (30) jours après l'envoi d'une mise en demeure par lettre recommandée avec accusé de réception restée sans effet.
A l'expiration du bail, le Preneur procédera à ses frais au démantèlement complet de l'installation et à la remise en état de la toiture dans un délai de 6 mois.

Fait à Lyon, le 15 août 2024, en deux exemplaires originaux.
"""
    rect2 = fitz.Rect(50, 50, 550, 800)
    page2.insert_textbox(rect2, text_p2, fontsize=10, fontname="helv")

    doc.save(output_path)
    doc.close()
    print(f"✅ Created sample contract with text wrapping at {output_path}")

if __name__ == "__main__":
    create_sample_contract()
