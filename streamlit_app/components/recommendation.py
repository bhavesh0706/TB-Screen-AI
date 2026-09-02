# streamlit_app/components/recommendation.py

def get_patient_recommendation(
    tb_probability,
    age,
    gender,
    pregnant=False,
    symptoms=None,
    conditions=None,
    emergency=None
):
    """
    WHO & NTEP aligned rule-based recommendation engine.
    Chest X-ray AI Screening only (not a diagnosis).
    Returns a dictionary containing clean, plain-text strings for UI injection.
    """

    symptoms = symptoms or {}
    conditions = conditions or {}
    emergency = emergency or {}

    # -----------------------------
    # Age Group
    # -----------------------------
    if age <= 1:
        age_group = "Infant"
    elif age <= 14:
        age_group = "Child"
    elif age <= 59:
        age_group = "Adult"
    else:
        age_group = "Senior"

    # -----------------------------
    # Emergency Override
    # -----------------------------
    emergency_steps = []

    if emergency.get("coughing_blood"):
        emergency_steps = [
            "Seek emergency medical care immediately.",
            "Avoid delaying hospital evaluation.",
            "Inform healthcare staff that you are coughing blood.",
            "Follow emergency medical instructions promptly."
        ]

    elif emergency.get("breathing_difficulty"):
        emergency_steps = [
            "Go to the nearest emergency department immediately.",
            "Avoid strenuous activity while seeking care.",
            "Inform healthcare providers about your breathing symptoms.",
            "Follow emergency evaluation without delay."
        ]

    elif emergency.get("blue_lips"):
        emergency_steps = [
            "Seek emergency medical care immediately.",
            "Do not wait for a routine appointment.",
            "Inform emergency staff about breathing symptoms.",
            "Follow emergency treatment instructions."
        ]

    elif emergency.get("confusion"):
        emergency_steps = [
            "Seek emergency medical care immediately.",
            "Have someone accompany you if possible.",
            "Inform emergency staff about your recent symptoms.",
            "Do not delay hospital evaluation."
        ]

    if emergency_steps:
        return {
            "risk": "Emergency",
            "color": "#991B1B",
            "summary": "Emergency symptoms require immediate medical attention.",
            "steps": emergency_steps,
            "age_group": age_group,
            "gender": gender,
            "confidence_text": "Emergency care required"
        }

    # -----------------------------
    # Confidence Score
    # -----------------------------
    if tb_probability <= 24:
        risk = "Very Low Suspicion"
        color = "#22C55E"
        confidence_text = "Routine monitoring"
        summary = "Low likelihood of TB-related abnormalities on this chest X-ray."

        steps = [
            "Continue normal daily activities unless symptoms worsen.",
            "Seek medical evaluation if cough persists for two weeks or longer, or if fever, weight loss, or night sweats develop.",
            "Repeat clinical assessment if symptoms continue despite a low screening score.",
            "Maintain good respiratory hygiene and adequate indoor ventilation if you have a cough."
        ]

    elif tb_probability <= 49:
        risk = "Low Suspicion"
        color = "#0EA5E9"
        confidence_text = "Clinical review advised"
        summary = "Mild abnormalities require clinical review."

        steps = [
            "Schedule a healthcare consultation within the next few days.",
            "A healthcare provider may recommend CBNAAT/GeneXpert or sputum examination if clinically indicated.",
            "Monitor symptoms closely and avoid delaying evaluation if they worsen.",
            "Share this screening report with the examining clinician."
        ]

    elif tb_probability <= 74:
        risk = "Moderate Suspicion"
        color = "#EAB308"
        confidence_text = "Further testing recommended"
        summary = "Findings require prompt confirmatory testing."

        steps = [
            "Visit a TB diagnostic center within 24–48 hours.",
            "Complete recommended confirmatory tests such as CBNAAT/GeneXpert.",
            "Keep indoor spaces well ventilated and cover your mouth while coughing.",
            "Attend all recommended follow-up appointments until TB is ruled out or confirmed."
        ]

    elif tb_probability <= 89:
        risk = "High Suspicion"
        color = "#F97316"
        confidence_text = "Urgent medical evaluation"
        summary = "High suspicion requiring urgent medical evaluation."

        steps = [
            "Seek same-day medical evaluation if possible.",
            "Complete confirmatory TB testing without unnecessary delay.",
            "Minimize close contact with others if you have a persistent cough until evaluated.",
            "Return immediately if breathing difficulty or coughing blood develops."
        ]

    else:
        risk = "Very High Suspicion"
        color = "#DC2626"
        confidence_text = "Immediate evaluation recommended"
        summary = "Very high suspicion requiring immediate clinical assessment."

        steps = [
            "Visit the nearest hospital or TB diagnostic center immediately.",
            "Complete same-day confirmatory testing if available.",
            "Follow healthcare provider instructions promptly after evaluation.",
            "Seek emergency care immediately if severe breathing difficulty or coughing blood occurs."
        ]

    # -----------------------------
    # Age Recommendations
    # -----------------------------
    age_matrix = {
        "Infant": [
            "A parent or guardian should seek immediate pediatric evaluation if TB is suspected.",
            "Inform the pediatrician about any household member diagnosed with TB.",
            "Follow pediatric testing recommendations, which may differ from adult testing methods.",
            "Seek urgent care if poor feeding, difficulty breathing, or unusual sleepiness occurs."
        ],

        "Child": [
            "Arrange prompt evaluation by a pediatric healthcare provider.",
            "Inform the clinician about close contact with anyone who has TB.",
            "Complete pediatric-appropriate TB testing if recommended.",
            "Return immediately if breathing difficulty or persistent fever develops."
        ],

        "Adult": [
            "Follow the confidence-based screening recommendation.",
            "Complete recommended diagnostic testing without delay.",
            "Attend follow-up appointments until evaluation is complete.",
            "Inform your healthcare provider about ongoing respiratory symptoms."
        ],

        "Senior": [
            "Seek earlier medical evaluation because symptoms may be less obvious.",
            "Inform clinicians about existing medical conditions and medications.",
            "Complete recommended testing promptly.",
            "Seek immediate care if breathing becomes difficult or symptoms rapidly worsen."
        ]
    }

    steps.extend(age_matrix[age_group])

    # -----------------------------
    # Gender & Pregnancy
    # -----------------------------
    if gender.lower() == "female":

        if pregnant:
            steps.extend([
                "Inform your healthcare provider immediately that you are pregnant.",
                "Do not delay recommended TB evaluation or confirmatory testing.",
                "Follow guidance from both TB and obstetric care providers.",
                "Seek urgent medical review if breathing difficulty or persistent fever develops."
            ])

        else:
            steps.extend([
                "Follow the confidence-based recommendation.",
                "Complete recommended confirmatory testing.",
                "Inform your healthcare provider about persistent symptoms.",
                "Return for follow-up if symptoms continue or worsen."
            ])

    else:
        steps.extend([
            "Follow the confidence-based recommendation.",
            "Complete recommended TB testing if advised.",
            "Report persistent respiratory symptoms to your healthcare provider.",
            "Attend follow-up appointments until evaluation is completed."
        ])

    # -----------------------------
    # Comorbidity Recommendations
    # -----------------------------
    disease_matrix = {
        "diabetes": [
            "Inform your healthcare provider that you have diabetes.",
            "Complete TB testing promptly because diabetes increases TB risk.",
            "Continue managing blood sugar during evaluation.",
            "Attend all recommended follow-up appointments."
        ],

        "hiv": [
            "Seek prompt medical evaluation.",
            "Complete rapid confirmatory TB testing if recommended.",
            "Inform the healthcare provider about HIV-related medications.",
            "Follow specialist recommendations throughout evaluation."
        ],

        "smoker": [
            "Complete recommended TB testing without delay.",
            "Inform the healthcare provider about your smoking history.",
            "Consider smoking cessation support during evaluation.",
            "Return if cough or breathing symptoms worsen."
        ],

        "copd": [
            "Inform the clinician about your lung condition.",
            "Complete recommended TB evaluation promptly.",
            "Continue prescribed respiratory medications unless instructed otherwise.",
            "Seek urgent care if breathing becomes significantly worse."
        ],

        "kidney_disease": [
            "Inform your healthcare provider about kidney disease.",
            "Complete recommended TB investigations promptly.",
            "Discuss current medications during evaluation.",
            "Attend all scheduled follow-up visits."
        ],

        "immunosuppressed": [
            "Inform your healthcare provider about ongoing treatment affecting immunity.",
            "Seek prompt TB evaluation because immunity may be reduced.",
            "Complete recommended confirmatory testing.",
            "Follow specialist advice regarding ongoing treatment."
        ]
    }

    for disease, recs in disease_matrix.items():
        if conditions.get(disease):
            steps.extend(recs)

    # -----------------------------
    # Symptom Recommendations
    # -----------------------------
    symptom_matrix = {
        "cough_2_weeks": [
            "Visit a healthcare facility for TB evaluation.",
            "Complete recommended sputum or molecular testing if advised.",
            "Cover your mouth while coughing.",
            "Keep living spaces well ventilated."
        ],

        "fever": [
            "Arrange medical evaluation if fever persists.",
            "Inform your clinician about accompanying cough or weight loss.",
            "Stay hydrated while awaiting evaluation.",
            "Return promptly if fever becomes severe or persistent."
        ],

        "weight_loss": [
            "Seek medical evaluation without unnecessary delay.",
            "Inform the healthcare provider about recent unintentional weight loss.",
            "Complete recommended diagnostic testing.",
            "Attend follow-up appointments until the cause is identified."
        ],

        "night_sweats": [
            "Schedule medical evaluation promptly.",
            "Inform your clinician if night sweats occur with cough or fever.",
            "Complete recommended TB investigations.",
            "Continue follow-up until symptoms are explained."
        ]
    }

    for symptom, recs in symptom_matrix.items():
        if symptoms.get(symptom):
            steps.extend(recs)

    # -----------------------------
    # Remove Duplicate Recommendations
    # -----------------------------
    unique_steps = []
    seen = set()

    for step in steps:
        if step not in seen:
            seen.add(step)
            unique_steps.append(step)

    return {
        "risk": risk,
        "color": color,
        "summary": summary,
        "steps": unique_steps,
        "age_group": age_group,
        "gender": gender,
        "confidence_text": confidence_text,
    }