import bayespy as bp
import numpy as np
import csv
from colorama import init
from colorama import Fore, Back, Style

init()

# Encodings for categorical variables
ageEnum = {'SuperSeniorCitizen': 0, 'SeniorCitizen': 1, 'MiddleAged': 2, 'Youth': 3, 'Teen': 4}
genderEnum = {'Male': 0, 'Female': 1}
familyHistoryEnum = {'Yes': 0, 'No': 1}
dietEnum = {'High': 0, 'Medium': 1, 'Low': 2}
lifestyleEnum = {'Sedentary': 0, 'Active': 1}
cholesterolEnum = {'High': 0, 'Borderline': 1, 'Normal': 2}
heartDiseaseEnum = {'Yes': 0, 'No': 1}

# Load CSV data
data = []
with open('7.csv') as csvfile:
    lines = csv.reader(csvfile)
    dataset = list(lines)[1:]  # skip header
    for x in dataset:
        data.append([
            ageEnum[x[0]],
            genderEnum[x[1]],
            familyHistoryEnum[x[2]],
            dietEnum[x[3]],
            lifestyleEnum[x[4]],
            cholesterolEnum[x[5]],
            heartDiseaseEnum[x[6]]
        ])

data = np.array(data)
N = len(data)

# Define priors and observations
P_age = bp.nodes.Dirichlet(1.0 + np.ones(5))
age = bp.nodes.Categorical(P_age, plates=(N,))
age.observe(data[:, 0])

P_gender = bp.nodes.Dirichlet(1.0 + np.ones(2))
gender = bp.nodes.Categorical(P_gender, plates=(N,))
gender.observe(data[:, 1])

P_family_history = bp.nodes.Dirichlet(1.0 + np.ones(2))
family_history = bp.nodes.Categorical(P_family_history, plates=(N,))
family_history.observe(data[:, 2])

P_diet = bp.nodes.Dirichlet(1.0 + np.ones(3))
diet = bp.nodes.Categorical(P_diet, plates=(N,))
diet.observe(data[:, 3])

P_lifestyle = bp.nodes.Dirichlet(1.0 + np.ones(2))
lifestyle = bp.nodes.Categorical(P_lifestyle, plates=(N,))
lifestyle.observe(data[:, 4])

P_cholesterol = bp.nodes.Dirichlet(1.0 + np.ones(3))
cholesterol = bp.nodes.Categorical(P_cholesterol, plates=(N,))
cholesterol.observe(data[:, 5])

# Define conditional distribution for heart disease
P_heartdisease = bp.nodes.Dirichlet(np.ones((2, 2, 2, 3, 2, 3)))
heartdisease = bp.nodes.Categorical(
    P_heartdisease[age, gender, family_history, diet, lifestyle, cholesterol]
)
heartdisease.observe(data[:, 6])

P_heartdisease.update()

# Inference
m = 0
while m == 0:
    print(Fore.YELLOW + "\nEnter patient's data to predict Heart Disease Probability:\n" + Style.RESET_ALL)
    res = [
        list(ageEnum.values()).index(ageEnum[input("Enter Age Group (Teen/Youth/MiddleAged/SeniorCitizen/SuperSeniorCitizen): ")]),
        list(genderEnum.values()).index(genderEnum[input("Enter Gender (Male/Female): ")]),
        list(familyHistoryEnum.values()).index(familyHistoryEnum[input("Family History (Yes/No): ")]),
        list(dietEnum.values()).index(dietEnum[input("Diet (High/Medium/Low): ")]),
        list(lifestyleEnum.values()).index(lifestyleEnum[input("Lifestyle (Sedentary/Active): ")]),
        list(cholesterolEnum.values()).index(cholesterolEnum[input("Cholesterol (High/Borderline/Normal): ")])
    ]

    q = bp.nodes.Categorical(
        P_heartdisease[res[0], res[1], res[2], res[3], res[4], res[5]]
    )
    print(Fore.GREEN + "Probability of Heart Disease: {:.2f}%".format(q.get_moments()[0][0] * 100) + Style.RESET_ALL)

    m = int(input(Fore.CYAN + "Enter 0 to continue, 1 to exit: " + Style.RESET_ALL))