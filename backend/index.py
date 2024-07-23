from app.ai.ner_model import extract_symptoms_from_text

while True:
    user_input = input('How can I help\n')
    symptoms = extract_symptoms_from_text(user_input)

    print("Identified symptoms: ", symptoms)
