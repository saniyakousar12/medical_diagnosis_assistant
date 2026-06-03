def get_risk_level(probability):
    """Convert probability to risk level with color and advice"""
    
    if probability < 0.3:
        return {
            'level': 'Low Risk',
            'color': 'green',
            'advice': '✅ Maintain healthy lifestyle. Regular check-ups recommended.'
        }
    elif probability < 0.6:
        return {
            'level': 'Medium Risk',
            'color': 'orange',
            'advice': '⚠️ Consult a doctor. Consider lifestyle changes and monitoring.'
        }
    else:
        return {
            'level': 'High Risk',
            'color': 'red',
            'advice': '🚨 Seek medical attention immediately. Further tests required.'
        }