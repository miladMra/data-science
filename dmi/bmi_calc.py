import streamlit as st
st.title('BMI Calculate')

weight = st.number_input('Enter your number (kg) :')
height = st.number_input('Enter your number (cm) :')

try:
    bmi = int(height)/((height/100)**2)
except:
    pass

btn = st.button('calculate')

if btn:
    st.text('your BMI is' + str(bmi))
    if bmi<16:
        st.error('You are Underwighted')
    if bmi>=16 and bmi<18.5:
        st.warning('You are Underwighted')
    if bmi>=18.5 and bmi<25:
        st.success('You are Perfect')
    if bmi>25:
        st.error('You are Overwighted')