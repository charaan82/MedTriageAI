# verifier.py
import re

# Matches technique IDs such as:
# T1049
# T1016.001
# T1590.004
TECHNIQUE_PATTERN = r"\bT\d{4}(?:\.\d{3})?\b"


def extract_technique_ids(text):
    """
    Extract MITRE ATT&CK technique IDs from text.
    """
    if not text:
        return []

    return sorted(
        set(re.findall(TECHNIQUE_PATTERN, text))
    )


def verify_citations(llm_response, retrieved_context):
    """
    Check whether every MITRE technique ID mentioned
    by the LLM exists in the retrieved MITRE context.
    """
    cited_ids = extract_technique_ids(llm_response)
    retrieved_ids = extract_technique_ids(retrieved_context)

    valid_ids = [
        technique_id
        for technique_id in cited_ids
        if technique_id in retrieved_ids
    ]

    fabricated_ids = [
        technique_id
        for technique_id in cited_ids
        if technique_id not in retrieved_ids
    ]

    if cited_ids:
        fabricated_rate = (
            len(fabricated_ids) / len(cited_ids)
        )
    else:
        fabricated_rate = 0.0

    return {
        "cited_ids": cited_ids,
        "retrieved_ids": retrieved_ids,
        "valid_ids": valid_ids,
        "fabricated_ids": fabricated_ids,
        "fabricated_id_rate": fabricated_rate
    }


def evaluate_with_abstention(llm_response, retrieved_context, confidence_score, confidence_threshold=0.65):
    """
    Combines citation verification with confidence-based abstention (Section 22).
    Escalates low-confidence alerts to human review rather than forcing an unconfident mapping.
    """
    # 1. Check confidence-based abstention threshold
    if confidence_score < confidence_threshold:
        return {
            "decision": "Human Analyst Review Required",
            "status": "ABSTAINED",
            "confidence": float(confidence_score),
            "threshold": confidence_threshold,
            "reason": f"Confidence score ({confidence_score:.2f}) is below the mandatory abstention threshold ({confidence_threshold}).",
            "cited_ids": extract_technique_ids(llm_response),
            "retrieved_ids": extract_technique_ids(retrieved_context),
            "valid_ids": [],
            "fabricated_ids": [],
            "fabricated_id_rate": 0.0
        }

    # 2. Run citation verification if confidence is sufficient
    verification_results = verify_citations(llm_response, retrieved_context)
    
    if verification_results["fabricated_id_rate"] > 0.05:
        decision = "Flagged for Verification (High Fabricated-ID Rate)"
        status = "WARNING"
    else:
        decision = "Approved (Evidence-Grounded)"
        status = "PROCESSED"

    verification_results["decision"] = decision
    verification_results["status"] = status
    verification_results["confidence"] = float(confidence_score)
    verification_results["threshold"] = confidence_threshold
    
    return verification_results


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retrieved_context = """
    1. T1016.001 - Internet Connection Discovery
    2. T1049 - System Network Connections Discovery
    3. T1590.004 - Network Topology
    """

    llm_response = """
    The alert is related to T1049 and T1016.001.
    It may also be related to T1059.
    """

    # Test with high confidence (Passes threshold)
    result_high_conf = evaluate_with_abstention(llm_response, retrieved_context, confidence_score=0.85)

    print("====================================")
    print("MITRE CITATION VERIFIER & ABSTENTION TEST")
    print("====================================")

    print("\nDecision (High Confidence):", result_high_conf["decision"])
    print(
        f"Fabricated-ID Rate: "
        f"{result_high_conf['fabricated_id_rate'] * 100:.2f}%"
    )

    # Test with low confidence (Triggers Abstention)
    result_low_conf = evaluate_with_abstention(llm_response, retrieved_context, confidence_score=0.45)
    print("\nDecision (Low Confidence):", result_low_conf["decision"])
    print("Reason:", result_low_conf["reason"])

    print("\nTarget: Fabricated-ID Rate <= 5.00%")
    if result_high_conf["fabricated_id_rate"] <= 0.05:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")