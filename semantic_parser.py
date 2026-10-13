def parse_alert(row):
    """
    Convert network telemetry into a contextual natural-language
    description for MITRE ATT&CK retrieval.
    """

    proto = str(row.get("proto", "unknown"))
    service = str(row.get("service", "unknown"))
    state = str(row.get("state", "unknown"))

    dur = float(row.get("dur", 0) or 0)
    spkts = float(row.get("spkts", 0) or 0)
    dpkts = float(row.get("dpkts", 0) or 0)
    sbytes = float(row.get("sbytes", 0) or 0)
    dbytes = float(row.get("dbytes", 0) or 0)
    rate = float(row.get("rate", 0) or 0)

    # Calculate simple contextual indicators
    total_packets = spkts + dpkts
    total_bytes = sbytes + dbytes

    if rate > 1000:
        traffic_pattern = "very high network traffic rate"
    elif rate > 100:
        traffic_pattern = "high network traffic rate"
    elif rate > 10:
        traffic_pattern = "moderate network traffic rate"
    else:
        traffic_pattern = "low network traffic rate"

    if total_packets > 1000:
        packet_pattern = "large packet volume"
    elif total_packets > 100:
        packet_pattern = "moderate packet volume"
    else:
        packet_pattern = "small packet volume"

    if total_bytes > 1000000:
        byte_pattern = "large data transfer volume"
    elif total_bytes > 100000:
        byte_pattern = "moderate data transfer volume"
    else:
        byte_pattern = "small data transfer volume"

    description = (
        f"A network event using {proto} protocol "
        f"with {service} service and connection state {state}. "
        f"The connection lasted {dur:.2f} seconds and generated "
        f"{spkts:.0f} source packets and {dpkts:.0f} destination packets, "
        f"with {sbytes:.0f} source bytes and {dbytes:.0f} destination bytes. "
        f"The observed traffic has a {traffic_pattern}, "
        f"{packet_pattern}, and {byte_pattern}."
    )

    return description


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    test_row = {
        "proto": "tcp",
        "service": "http",
        "state": "FIN",
        "dur": 2.5,
        "spkts": 150,
        "dpkts": 120,
        "sbytes": 25000,
        "dbytes": 18000,
        "rate": 250
    }

    print("====================================")
    print("SEMANTIC PARSER TEST")
    print("====================================")

    result = parse_alert(test_row)

    print("\nGenerated semantic description:")
    print("------------------------------------")
    print(result)