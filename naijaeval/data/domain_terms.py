"""Built-in domain-critical terminology lists for African deployment contexts.

Each entry maps a canonical term to a list of acceptable surface forms
(including abbreviations, common synonyms, and local variants).

These lists are intentionally concise and high-precision.  Contributions
for additional domains and local variants are welcomed — see CONTRIBUTING.md.
"""

from __future__ import annotations

MEDICAL: dict[str, list[str]] = {
    "malaria": ["malaria", "maleria"],
    "tuberculosis": ["tuberculosis", "TB", "t.b.", "t.b"],
    "HIV": ["HIV", "H.I.V", "human immunodeficiency virus"],
    "AIDS": ["AIDS", "A.I.D.S", "acquired immunodeficiency syndrome"],
    "hypertension": ["hypertension", "high blood pressure", "hbp"],
    "diabetes": ["diabetes", "diabetes mellitus", "sugar disease"],
    "anemia": ["anaemia", "anemia", "low blood"],
    "typhoid": ["typhoid", "typhoid fever"],
    "cholera": ["cholera"],
    "meningitis": ["meningitis"],
    "pneumonia": ["pneumonia"],
    "diarrhea": ["diarrhoea", "diarrhea", "running stomach", "stooling"],
    "vaccination": ["vaccination", "vaccine", "immunisation", "immunization"],
    "antibiotic": ["antibiotic", "antibiotics"],
    "dosage": ["dosage", "dose", "dosing"],
    "prescription": ["prescription", "prescribed"],
    "emergency": ["emergency", "urgent care"],
    "hospital": ["hospital", "clinic", "health centre", "health center"],
    "surgery": ["surgery", "operation", "surgical procedure"],
    "blood pressure": ["blood pressure", "BP"],
}

LEGAL: dict[str, list[str]] = {
    "plaintiff": ["plaintiff", "claimant", "complainant"],
    "defendant": ["defendant", "accused", "respondent"],
    "jurisdiction": ["jurisdiction"],
    "indictment": ["indictment", "charge", "charges"],
    "affidavit": ["affidavit"],
    "judgment": ["judgment", "judgement", "ruling", "verdict"],
    "appeal": ["appeal", "appellate"],
    "evidence": ["evidence", "exhibit"],
    "witness": ["witness", "witnesses", "testimony"],
    "bail": ["bail", "bond"],
    "injunction": ["injunction", "restraining order"],
    "constitution": ["constitution", "constitutional"],
    "statute": ["statute", "act", "legislation", "law"],
    "contract": ["contract", "agreement", "deed"],
    "liability": ["liability", "liable"],
    "damages": ["damages", "compensation", "redress"],
    "prosecution": ["prosecution", "prosecute", "prosecutor"],
    "acquittal": ["acquittal", "acquitted", "not guilty"],
    "convict": ["conviction", "convicted", "guilty verdict"],
    "tribunal": ["tribunal", "court", "bench"],
}

FINANCIAL: dict[str, list[str]] = {
    "collateral": ["collateral", "security", "pledge"],
    "interest rate": ["interest rate", "interest", "APR", "annual rate"],
    "principal": ["principal", "loan amount", "capital"],
    "amortization": ["amortization", "amortisation", "repayment schedule"],
    "equity": ["equity", "ownership stake"],
    "dividend": ["dividend", "payout"],
    "inflation": ["inflation", "price increase", "cost of living"],
    "exchange rate": ["exchange rate", "forex", "FX rate"],
    "remittance": ["remittance", "transfer", "diaspora transfer"],
    "microfinance": ["microfinance", "micro-finance", "microloan"],
    "mobile money": ["mobile money", "MoMo", "M-Pesa", "Paga"],
    "account": ["account", "bank account"],
    "transaction": ["transaction", "transfer", "payment"],
    "fraud": ["fraud", "scam", "419", "advance fee fraud"],
    "naira": ["naira", "NGN", "₦"],
    "cedi": ["cedi", "GHS", "GH₵"],
    "shilling": ["shilling", "KES", "UGX", "TZS"],
    "rand": ["rand", "ZAR", "R"],
}

CUSTOMER_SUPPORT: dict[str, list[str]] = {
    "refund": ["refund", "money back", "reimbursement"],
    "verification": ["verification", "verify", "KYC", "know your customer"],
    "complaint": ["complaint", "issue", "problem", "report"],
    "resolution": ["resolution", "resolve", "fix"],
    "escalation": ["escalation", "escalate", "supervisor", "manager"],
    "delivery": ["delivery", "shipping", "dispatch"],
    "tracking": ["tracking", "track", "order status"],
    "cancellation": ["cancellation", "cancel", "terminate"],
    "subscription": ["subscription", "plan", "package"],
    "authentication": ["authentication", "login", "sign in", "password"],
    "account": ["account", "profile"],
    "transaction": ["transaction", "payment", "transfer"],
}

DOMAIN_TERMS: dict[str, dict[str, list[str]]] = {
    "medical": MEDICAL,
    "legal": LEGAL,
    "financial": FINANCIAL,
    "customer_support": CUSTOMER_SUPPORT,
}
