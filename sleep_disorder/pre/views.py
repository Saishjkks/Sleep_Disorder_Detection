import joblib
import numpy as np
import os
from django.shortcuts import render

model_path = os.path.join(os.path.dirname(__file__), "sleep_model.pkl")
model_package = joblib.load(model_path)

model = model_package["model"]  # Trained model
scaler = model_package["scaler"]  # StandardScaler
encoders = model_package["encoders"]  # Dictionary of categorical encoders
target_encoder = model_package["target_encoder"]  # Target label encoder

def index(request):
    return render(request, 'index.html')


def index1(request):
    return render(request, 'index1.html')  # Ensure index1.html is inside the templates folder


def predict(request):
    if request.method == "POST":
        try:
            # Get form data
            input_data = {
                "Gender": request.POST.get("Gender"),
                "Age": request.POST.get("Age"),
                "Occupation": request.POST.get("Occupation"),
                "Sleep Duration": request.POST.get("Sleep Duration"),
                "Quality of Sleep": request.POST.get("Quality of Sleep"),
                "Physical Activity Level": request.POST.get("Physical Activity Level"),
                "Stress Level": request.POST.get("Stress Level"),
                "BMI Category": request.POST.get("BMI Category"),
                "Heart Rate": request.POST.get("Heart Rate"),
                "Daily Steps": request.POST.get("Daily Steps"),
                "Systolic": request.POST.get("Systolic"),
                "Diastolic": request.POST.get("Diastolic"),
            }

            # Convert categorical data using encoders
            for feature, encoder in encoders.items():
                input_data[feature] = encoder.transform([input_data[feature]])[0]

            # Convert numeric values to float
            numeric_features = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                                "Stress Level", "Heart Rate", "Daily Steps", "Systolic", "Diastolic"]

            for feature in numeric_features:
                input_data[feature] = float(input_data[feature])

            # Convert input to NumPy array
            input_array = np.array(list(input_data.values())).reshape(1, -1)

            # Scale numerical data
            input_array = scaler.transform(input_array)

            # Make prediction
            prediction = model.predict(input_array)

            # Decode the predicted class
            decoded_prediction = target_encoder.inverse_transform(prediction)[0]
           
            if decoded_prediction == "Sleep Apnea":
                show_suggestions = True
                suggestions_content = [
                "\u2705 Use a CPAP Machine – A Continuous Positive Airway Pressure (CPAP) machine keeps your airway open while sleeping.",
                "\u2705 Maintain a Healthy Weight – Losing excess weight can significantly reduce airway obstruction.",
                "\u2705 Change Sleep Position – Sleeping on your side instead of your back helps keep the airway open.",
                "\u2705 Avoid Alcohol & Sedatives – These relax throat muscles, increasing airway collapse risk.",
                "\u2705 Quit Smoking – Smoking increases inflammation and fluid retention in the airway.",
                "\ud83e\uddec Consider Oral Appliances – Special mouthpieces help adjust jaw position for better airflow.",
                "\ud83e\uddec Explore Surgery (If Necessary) – Procedures like UPPP (Uvulopalatopharyngoplasty) or Inspire Therapy may be options in severe cases.",
                "\ud83e\uddec Use a Humidifier – Dry air can irritate the airway; a humidifier adds moisture for easier breathing.",
                "\ud83d\udca4 Follow a Sleep Schedule – Sleep and wake up at the same time daily to regulate your body clock.",
                "\ud83d\udca4 Avoid Heavy Meals Before Bed – A full stomach can put pressure on the diaphragm and worsen breathing.",
                "\ud83d\udca4 Exercise Regularly – Moderate activity can improve overall sleep quality and reduce symptoms."

                ]
            elif decoded_prediction == 'nan' or isinstance(decoded_prediction, float) and np.isnan(decoded_prediction):
                show_suggestions = True
                suggestions_content = [
                "Maintain this life style"
                "✅ Stick to a Regular Sleep Schedule – Go to bed and wake up at the same time daily, even on weekends.",
                "✅ Limit Caffeine & Stimulants – Avoid coffee, tea, and nicotine in the afternoon and evening.",
                "✅ Stay Active – Engage in daily physical activity, but avoid intense workouts close to bedtime.",
                "✅ Get Natural Light Exposure – Spend time outside during the day to regulate your body's circadian rhythm.",
                "✅ Take Short Naps – Limit naps to 20-30 minutes to avoid disrupting nighttime sleep.",
                "😴 Keep Your Room Cool & Dark – A comfortable temperature (around 18-22°C) and blackout curtains improve sleep.",
                "😴 Use Comfortable Bedding – Invest in a quality mattress and pillows suited to your sleep position.",
                "😴 Reduce Noise Levels – Use earplugs or a white noise machine to block disturbances.",
                "😴 Limit Electronics Before Bed – Avoid screens at least an hour before sleeping to reduce blue light exposure.",
                "🌙 Establish a Wind-Down Routine – Read a book, meditate, or listen to calming music before bed.",
                "🌙 Avoid Heavy Meals Before Sleep – Eating too close to bedtime can cause indigestion and disrupt rest.",
                "🌙 Try Relaxation Techniques – Deep breathing exercises or progressive muscle relaxation can help you unwind."
                ]    
            else:
                
                show_suggestions = True
                suggestions_content = [
                "✅ Establish a Consistent Sleep Schedule – Go to bed and wake up at the same time every day, even on weekends.",
                "✅ Limit Caffeine & Nicotine Intake – Avoid stimulants like coffee, tea, and cigarettes, especially in the evening.",
                "✅ Avoid Alcohol Before Bed – While it may help you fall asleep, alcohol can disrupt sleep cycles.",
                "✅ Reduce Evening Screen Time – Avoid blue light from phones, tablets, and computers at least an hour before bed.",
                "✅ Exercise Regularly – Engage in moderate physical activity during the day to improve sleep quality.",
                "😴 Create a Relaxing Bedtime Routine – Engage in calming activities like reading, meditating, or taking a warm bath before bed.",
                "😴 Optimize Your Sleep Environment – Keep your bedroom dark, quiet, and at a comfortable temperature.",
                "😴 Use Your Bed Only for Sleep – Avoid working, watching TV, or using your phone in bed to strengthen the brain’s sleep association.",
                "😴 Avoid Large Meals Late at Night – Eating too close to bedtime can cause discomfort and disrupt sleep.",
                "🩺 Try Cognitive Behavioral Therapy for Insomnia (CBT-I) – This therapy helps address negative thoughts and behaviors affecting sleep.",
                "🩺 Consider Melatonin Supplements – Consult a doctor before using melatonin or other sleep aids.",
                "🩺 Manage Stress & Anxiety – Practices like mindfulness meditation, yoga, and journaling can help reduce stress and improve sleep.",
                "🩺 Seek Professional Help – If insomnia persists for weeks, consult a sleep specialist to identify underlying causes."
                ]

            return render(request, 'index1.html', {
                'prediction': f"Detected Sleep Disorder: {decoded_prediction}",
                'show_suggestions': show_suggestions,
                'suggestions_content': suggestions_content
            })
            


            #return render(request, 'index1.html', {'prediction': f"Detected Sleep Disorder: {decoded_prediction}"})

        except Exception as e:
            return render(request, 'index1.html', {'prediction': f"Error: {str(e)}"})
        
        

    return render(request, 'index1.html')











