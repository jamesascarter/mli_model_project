import os
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import cv2
import pandas as pd
from streamlit_drawable_canvas import st_canvas
from sqlalchemy import create_engine, text
from PIL import Image
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DB_URL = "postgresql://postgres:password@db:5432/mydatabase"

engine = create_engine(DB_URL)

Session = sessionmaker(bind=engine)
session = Session()

Base = declarative_base()

class Log(Base):
    __tablename__ = 'logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    predicted_digit = Column(Integer, nullable=False)
    actual_digit = Column(String, nullable=False)
    timestamp = Column(DateTime, default=func.now(), nullable=False)


Base.metadata.create_all(engine)

def insert_log(predicted_digit, actual_digit):
    try:
        insert_query = text("INSERT INTO logs (predicted_digit, actual_digit, timestamp) VALUES (:predicted_digit, :actual_digit, NOW())")
        session.execute(insert_query, {"predicted_digit": predicted_digit, "actual_digit": actual_digit})
        session.commit()
        print("Log inserted successfully!")
    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()

def get_logs():
    try:
        result = session.execute(text("SELECT * FROM logs"))
        logs = result.fetchall()
        
        return logs
    except Exception as e:
        print(f"Error fetching logs: {e}")
        return []
    finally:
        session.close()

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)  # More neurons
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = NeuralNet().to(device)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This is /project_root/frontend
MODEL_PATH = os.path.abspath(os.path.join(BASE_DIR, 'mnist_model.pth'))

print(f"Looking for model at: {BASE_DIR}")

print(f"Looking for model at: {MODEL_PATH}")

print("File exists?" , os.path.exists(MODEL_PATH))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

st.title("Digit Recognizer")

canvas = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

actual_digit = st.text_input("True label:", "")

def preprocess_image(img):
    img = cv2.resize(img, (28, 28))
    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)  # Convert to grayscale
    img = img / 255.0  # Normalize
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return img

# Make prediction
if st.button("Submit"):
    if canvas.image_data is not None:
        img = np.array(canvas.image_data, dtype=np.uint8)  # Convert to NumPy array
        img = img[:, :, :3]
        img = preprocess_image(img)
        img = img.to(device)
        
        output = model(img)
        probabilities = F.softmax(output, dim=1)  # Apply softmax to get probabilities
        predicted_digit = torch.argmax(probabilities, 1).item()
        confidence = probabilities[0, predicted_digit].item() * 100  # percentage
        insert_log(predicted_digit, actual_digit)
        logs = get_logs()
        df = pd.DataFrame(logs, columns=["ID", "Predicted", "Actual", "Timestamp"])
        df = df.reset_index(drop=True)
        st.write(f"Prediction: {predicted_digit}")
        st.write(f"Confidence: {confidence:.2f}%")
        st.dataframe(df)

