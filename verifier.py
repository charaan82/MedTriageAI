import re


# Matches:
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

    result = verify_citations(
        llm_response,
        retrieved_context
    )

    print("====================================")
    print("MITRE CITATION VERIFIER TEST")
    print("====================================")

    print("\nCited IDs:")
    print(result["cited_ids"])

    print("\nRetrieved IDs:")
    print(result["retrieved_ids"])

    print("\nValid IDs:")
    print(result["valid_ids"])

    print("\nFabricated IDs:")
    print(result["fabricated_ids"])

    print(
        f"\nFabricated-ID Rate: "
        f"{result['fabricated_id_rate'] * 100:.2f}%"
    )

    print("\nTarget: <= 5.00%")

    if result["fabricated_id_rate"] <= 0.05:
        print("STATUS: PASS")
    else:
        print("STATUS: FAIL")