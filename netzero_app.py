import streamlit as st
import pandas as pd
import numpy as np
import pickle


with open('./climate_commitment.pkl', 'rb')  as f:
    model = pickle.load(f)

#body Text
st.title('Will the Company Committ To NetZero?')
st.subheader('A Climate Action Check In the MSCI ACWI Universe')
st.write('\n\n')
st.text('Select below company characteristics to find out')


#First Predictor
option1 = st.selectbox(
     '1. Sector',
     ('Communication Services', 'Consumer Discretionary', 'Consumer Staples', 'Energy', 'Health Care','Industrials', 'Information Technology', 'Materials','Real Estate' , 'Utilities'))

#st.write('You selected:', option)

#Second Predictor
option2 = st.selectbox(
     '2. Region',
     ('Africa/Mideast', 'Asia/Pacific Ex Japan', 'Europe', 'Japan','Latin America', 'North America'))

#st.write('You selected:', option)


#Third Predictor

option3 = st.selectbox(
     '3. Market Capitalization',
     ('Large', 'Medium', 'Small'))

#st.write('You selected:', option)


#Fourth Predictor

option4 = st.selectbox(
     '4. Country Economy',
     ('Developed', 'Emerging'))

#Fifth Predictor

# option5 = st.selectbox(
#      '5. Climate Target',
#      ('committed', 'set'))

#Sixth Predictor

option6 = st.slider(
     '5. Scope 3 Emissions (KT Co2)',
     0, 230000, 0)
#st.write('Values:', option6)


#Predictor 7th

option7 = st.slider(
     '6. Scope 1 and 2 Emissions (KT Co2)',
     0, 30000, 0)
#st.write('Values:', option7)


option1 = 'gics_sector_name_' + option1
option2 = 'region_' + option2
option3 = 'company_cap_' + option3.lower()
option4 = 'country_economy_' + option4.lower()
#option5 = 'target_status_class_year_Targets Set' + option5



# now I need to create my row
lst_columns = ['gics_sector_name_Communication Services',
       'gics_sector_name_Consumer Discretionary',
       'gics_sector_name_Consumer Staples', 'gics_sector_name_Energy',
       'gics_sector_name_Financials', 'gics_sector_name_Health Care',
       'gics_sector_name_Industrials',
       'gics_sector_name_Information Technology', 'gics_sector_name_Materials',
       'gics_sector_name_Real Estate', 'gics_sector_name_Utilities',
       'region_Africa/Mideast', 'region_Asia/Pacific Ex Japan',
       'region_Europe', 'region_Japan', 'region_Latin America',
       'region_North America', 'company_cap_large', 'company_cap_medium',
       'company_cap_small', 'country_economy_developed',
       'country_economy_emerging','target_status_class_year_committed',
       'target_status_class_year_targets_set']


lst = [option1,option2,option3,option4]
lst2 = []
for val in lst_columns:
    if val not in lst:
        lst2.append(0)
    else:
        lst2.append(1)


lst_features = []
lst_features.append(option6)
lst_features.append(option7)
for val in lst2:
  lst_features.append(int(val))

lst4 = np.array([lst_features])


pred = model.predict(lst4)
answer = 'YES' if pred == 1 else 'NO'
prob = model.predict_proba(lst4)
proba_yes =(prob[0][1]) * 100
proba_yes = round(proba_yes, 2)
proba_no = (prob[0][0]) * 100
proba_no = round(proba_no, 2)

if answer == 'YES':
  st.write('Is The Company Committing To NetZero ?', f'\n\n{answer}!')
  st.metric('Probability', f'{proba_yes}%')
  from PIL import Image
  image = Image.open('./images/action2.jpg')
  st.image(image, caption='The Company Is Protecting Our Home!')

elif answer == 'NO':
  st.write('Is The Company Committing To NetZero ?', f'\n\n{answer}!')
  st.metric('Probability', f'{proba_no}%')
  from PIL import Image
  image = Image.open('./images/join_movement.jpg')
  st.image(image, caption='Join The NetZero Movement Soon!')
