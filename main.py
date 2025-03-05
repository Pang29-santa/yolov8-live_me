import streamlit as st
import os

st.title("🎯 Object Detection Dashboard")

st.write("กดปุ่มด้านล่างเพื่อเริ่มตรวจจับวัตถุ")

if st.button("🔍 เปิดระบบตรวจจับวัตถุ"):
    os.system("streamlit run app.py")
