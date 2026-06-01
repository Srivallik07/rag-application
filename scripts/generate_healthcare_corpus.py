"""Generate 50+ healthcare domain source documents for the RAG corpus."""

from __future__ import annotations

import json
from pathlib import Path

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "documents" / "healthcare"

TOPICS: list[dict[str, str]] = [
    {"id": "01", "title": "Type 2 Diabetes Management", "category": "Endocrinology"},
    {"id": "02", "title": "Hypertension Clinical Guidelines", "category": "Cardiology"},
    {"id": "03", "title": "Childhood Immunization Schedule", "category": "Pediatrics"},
    {"id": "04", "title": "Asthma Action Plan Basics", "category": "Pulmonology"},
    {"id": "05", "title": "Antibiotic Stewardship in Primary Care", "category": "Infectious Disease"},
    {"id": "06", "title": "Postoperative Pain Control", "category": "Anesthesiology"},
    {"id": "07", "title": "Heart Failure with Reduced Ejection Fraction", "category": "Cardiology"},
    {"id": "08", "title": "Chronic Kidney Disease Staging", "category": "Nephrology"},
    {"id": "09", "title": "Major Depressive Disorder Screening", "category": "Psychiatry"},
    {"id": "10", "title": "Osteoporosis Prevention in Older Adults", "category": "Geriatrics"},
    {"id": "11", "title": "Migraine Acute Treatment Options", "category": "Neurology"},
    {"id": "12", "title": "Prenatal Folic Acid Supplementation", "category": "Obstetrics"},
    {"id": "13", "title": "Community Acquired Pneumonia Treatment", "category": "Pulmonology"},
    {"id": "14", "title": "Rheumatoid Arthritis First-Line Therapy", "category": "Rheumatology"},
    {"id": "15", "title": "Pediatric Fever Without Source", "category": "Pediatrics"},
    {"id": "16", "title": "Atrial Fibrillation Stroke Prevention", "category": "Cardiology"},
    {"id": "17", "title": "COPD Exacerbation Management", "category": "Pulmonology"},
    {"id": "18", "title": "Iron Deficiency Anemia Workup", "category": "Hematology"},
    {"id": "19", "title": "Generalized Anxiety Disorder Treatment", "category": "Psychiatry"},
    {"id": "20", "title": "Gestational Diabetes Screening", "category": "Obstetrics"},
    {"id": "21", "title": "Hepatitis C Screening Recommendations", "category": "Gastroenterology"},
    {"id": "22", "title": "Hypothyroidism Levothyroxine Dosing", "category": "Endocrinology"},
    {"id": "23", "title": "Sepsis Early Recognition Bundle", "category": "Critical Care"},
    {"id": "24", "title": "Low Back Pain Non-Surgical Care", "category": "Orthopedics"},
    {"id": "25", "title": "Breast Cancer Screening Mammography", "category": "Oncology"},
    {"id": "26", "title": "Colorectal Cancer Screening Options", "category": "Gastroenterology"},
    {"id": "27", "title": "Pediatric Asthma Controller Therapy", "category": "Pediatrics"},
    {"id": "28", "title": "Dyslipidemia Statin Indications", "category": "Cardiology"},
    {"id": "29", "title": "Urinary Tract Infection Empiric Therapy", "category": "Urology"},
    {"id": "30", "title": "Acute Coronary Syndrome Initial Care", "category": "Emergency Medicine"},
    {"id": "31", "title": "Stroke Thrombolysis Eligibility", "category": "Neurology"},
    {"id": "32", "title": "Pediatric Dehydration Oral Rehydration", "category": "Pediatrics"},
    {"id": "33", "title": "Contraception Counseling Overview", "category": "Reproductive Health"},
    {"id": "34", "title": "Sleep Apnea Diagnosis Pathway", "category": "Sleep Medicine"},
    {"id": "35", "title": "Celiac Disease Serologic Testing", "category": "Gastroenterology"},
    {"id": "36", "title": "Gout Acute Flare Management", "category": "Rheumatology"},
    {"id": "37", "title": "Attention Deficit Hyperactivity Disorder", "category": "Psychiatry"},
    {"id": "38", "title": "Influenza Vaccination High-Risk Groups", "category": "Preventive Medicine"},
    {"id": "39", "title": "Pressure Injury Prevention in Hospitals", "category": "Nursing"},
    {"id": "40", "title": "Opioid Use Disorder Buprenorphine", "category": "Addiction Medicine"},
    {"id": "41", "title": "Eczema Topical Treatment Ladder", "category": "Dermatology"},
    {"id": "42", "title": "Glaucoma First-Line Eye Drops", "category": "Ophthalmology"},
    {"id": "43", "title": "HIV Pre-Exposure Prophylaxis PrEP", "category": "Infectious Disease"},
    {"id": "44", "title": "Palliative Dyspnea Symptom Relief", "category": "Palliative Care"},
    {"id": "45", "title": "Food Allergy Anaphylaxis Epinephrine", "category": "Allergy"},
    {"id": "46", "title": "Venous Thromboembolism Prophylaxis", "category": "Hematology"},
    {"id": "47", "title": "Alcohol Withdrawal CIWA Protocol", "category": "Addiction Medicine"},
    {"id": "48", "title": "Thyroid Nodule Evaluation Steps", "category": "Endocrinology"},
    {"id": "49", "title": "Pelvic Inflammatory Disease Treatment", "category": "Gynecology"},
    {"id": "50", "title": "Vitamin D Deficiency Replacement", "category": "Primary Care"},
    {"id": "51", "title": "Acute Otitis Media in Children", "category": "ENT"},
    {"id": "52", "title": "Parkinson Disease Motor Symptoms", "category": "Neurology"},
    {"id": "53", "title": "Burn Wound Initial Emergency Care", "category": "Emergency Medicine"},
    {"id": "54", "title": "Chronic Hepatitis B Monitoring", "category": "Gastroenterology"},
    {"id": "55", "title": "Fall Prevention in Older Adults", "category": "Geriatrics"},
]


def build_document(topic: dict[str, str]) -> str:
    title = topic["title"]
    category = topic["category"]
    doc_id = topic["id"]
    return f"""Healthcare Knowledge Base Document
Document ID: HC-{doc_id}
Title: {title}
Category: {category}
Domain: Healthcare / Clinical Reference

Overview
This clinical reference summarizes evidence-informed guidance for {title.lower()} used in hospital and primary-care settings. Content is intended for clinician education and patient counseling support within a retrieval-augmented healthcare assistant.

Key Clinical Points
- Patients presenting with concerns related to {title.lower()} should receive structured assessment including history, focused examination, and appropriate diagnostics.
- First-line management should follow specialty consensus where available, with attention to comorbidities, renal function, hepatic function, pregnancy status, and drug interactions.
- Red-flag symptoms requiring urgent escalation include sudden severe pain, neurologic deficits, respiratory distress, hemodynamic instability, altered mental status, and signs of sepsis.

Diagnostic Considerations
Clinicians should document onset, duration, severity, triggers, prior episodes, medication adherence, allergies, and social determinants of health. Laboratory and imaging decisions should be guided by pretest probability and shared decision-making.

Treatment and Monitoring
Treatment plans should specify medication name, dose, route, frequency, duration, monitoring labs, and follow-up interval. Non-pharmacologic interventions such as lifestyle counseling, physical therapy, vaccination, or care coordination should be included when relevant to {category.lower()} care pathways.

Patient Counseling
Explain expected benefits, common adverse effects, warning signs that require immediate care, and adherence strategies. Provide written instructions when feasible and confirm health literacy using teach-back methods.

Quality and Safety
Verify allergy status before prescribing antibiotics or contrast agents. Use weight-based dosing in pediatrics. Reconcile medications at every visit. Document shared decisions for screening and preventive services.

Documentation Keywords
{title}; {category}; clinical guideline; patient education; monitoring; contraindications; follow-up; healthcare RAG corpus HC-{doc_id}.
"""


def main() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    for topic in TOPICS:
        filename = f"hc_{topic['id']}_{topic['title'].lower().replace(' ', '_')[:40]}.txt"
        path = CORPUS_DIR / filename
        path.write_text(build_document(topic), encoding="utf-8")
        manifest.append(
            {
                "document_id": f"HC-{topic['id']}",
                "filename": filename,
                "title": topic["title"],
                "category": topic["category"],
            }
        )

    manifest_path = CORPUS_DIR / "corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(TOPICS)} healthcare documents to {CORPUS_DIR}")


if __name__ == "__main__":
    main()
