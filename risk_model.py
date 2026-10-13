# risk_model.py
import numpy as np

class RiskAssessmentModel:
    """
    Implements the Mathematical Risk-Based Severity Model as defined in 
    Section 14 of the TriageMind Revised Synopsis.
    R(x) = w1*P_A + w2*C_A + w3*E_X + w4*I_P + w5*T_I
    """
    def __init__(self, weights=None):
        # Default weights summing to 1.0
        self.weights = weights or {
            'w1_prob': 0.30,  # P_A: Attack Probability from Random Forest
            'w2_crit': 0.25,  # C_A: Attack-Category Criticality
            'w3_expl': 0.20,  # E_X: Exploitation Evidence
            'w4_pers': 0.15,  # I_P: Persistence/Impact Evidence
            'w5_int':  0.10   # T_I: Traffic/Behavioral Intensity
        }

    def compute_risk(self, attack_prob, category_criticality, exploit_ev, persist_ev, traffic_intensity):
        """Calculates the normalized risk score R(x) between 0.0 and 1.0."""
        r_score = (
            self.weights['w1_prob'] * attack_prob +
            self.weights['w2_crit'] * category_criticality +
            self.weights['w3_expl'] * exploit_ev +
            self.weights['w4_pers'] * persist_ev +
            self.weights['w5_int']  * traffic_intensity
        )
        return float(np.clip(r_score, 0.0, 1.0))

    def get_severity_label(self, risk_score):
        """Maps risk score to severity intervals defined in Section 14."""
        if risk_score < 0.25:
            return "Low"
        elif risk_score < 0.50:
            return "Medium"
        elif risk_score < 0.75:
            return "High"
        else:
            return "Critical"