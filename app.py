import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
import platform

# Platformani aniqlash
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath

# Title va qisqacha tavsif
st.title("Telefon, Yo'l Belgisi va Soatni Klassifikatsiya Qiluvchi Model")
st.markdown("""
Bu web app yordamida siz tasvirlarni yuklab, ularni modelga yuborib, avtomatik ravishda turli obyektlarni (telefon, yo'l belgisi, soat) klassifikatsiya qilishingiz mumkin.
Yuqoridagi qismga rasm yuklab, bashoratlarni ko'rishingiz mumkin.
""")

# Rasmni yuklash uchun fayl yuklagich
file = st.file_uploader('Rasm yuklash', type=['png', 'jpeg', 'jpg', 'gif', 'svg'])
if file:
    # Yuklangan rasmlarni ko'rsatish
    st.image(file, caption='Yuklangan rasm', use_column_width=True)
    
    # Rasmni PIL formatiga o'tkazish
    img = PILImage.create(file)
    
    # Modelni yuklash
    model = load_learner('practice_model.pkl')
    
    # Bashorat qilish
    pred, pred_id, probs = model.predict(img)
    
    # Foydalanuvchiga natijani ko'rsatish
    st.success(f"Bashorat: {pred}")
    st.info(f'Ehtimollik: {probs[pred_id]*100:.1f}%')

    # Grafikni chizish
    fig = px.bar(x=model.dls.vocab, y=probs*100, labels={'x': 'Klasslar', 'y': 'Ehtimollik (%)'})
    fig.update_layout(title="Model Ehtimolliklari",
                      xaxis_title="Klasslar",
                      yaxis_title="Ehtimollik (%)",
                      plot_bgcolor="rgba(0,0,0,0)", 
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig)
