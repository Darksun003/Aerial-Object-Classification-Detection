<div align="center">

# 🛰️⚡ Aerial Object Classification Detection   
### **Bird vs Drone — Smart Vision for the Sky**

<img src="https://img.shields.io/badge/Deep%20Learning-TensorFlow-blue?logo=tensorflow&style=flat-square">
<img src="https://img.shields.io/badge/Object%20Detection-YOLOv8-red?logo=python&style=flat-square">
<img src="https://img.shields.io/badge/UI-Streamlit-green?logo=streamlit&style=flat-square">
<img src="https://img.shields.io/badge/Status-Production%20Ready-success?style=flat-square">

🧠 A real-time AI system that **detects and identifies Birds & Drones from aerial images** —  
designed for ✈️ airports, 🛡️ security zones, 🏞️ wildlife monitoring & 🚁 UAV surveillance.

</div>

---

## 🚀 What this AI system can do

🔹 Classify an uploaded image as **Bird or Drone**  
🔹 Detect **multiple Birds + Drones** in the **same scene** using YOLOv8  
🔹 Visualize bounding boxes, labels & confidence scores  
🔹 Provide a clean, interactive **web UI for instant results**

---

## 🧠 Tech Behind the System
```

| Component         |   Technology Used                    |
|-------------------|--------------------------------------|
| Framewor        k | TensorFlow / Keras                   |
| Transfer Learning | MobileNetV2                          |
| Object Detection  | YOLOv8                               |
| Interface         | Streamlit                            |
| Language          | Python                               |
| Dataset           | Custom — Bird vs Drone (YOLO Format) |
```
---

## 🎯 Real-World Applications

✔ Airport bird strike prevention  
✔ Identify unauthorized drones in **no-fly zones**  
✔ Drone-based wildlife monitoring  
✔ Military & border surveillance  
✔ Smart city & traffic aerial monitoring  

---

## 🖥️ Live Workflow
![alt text](<sample results (custom)/classificaion.png>)
![alt text](<sample results (custom)/yolo.png>)


🟢 **Classification Mode** → Bird / Drone (single object)  
🔵 **YOLO Detection Mode** → Detects & counts **each bird and drone** in the scene  

---

## 📂 Project Structure
```
📁 Aerial Object Classification & Detection
│
├── 🗂 dataset/
│ └── object_detection_Dataset (train, valid, test, data.yaml)
│
├── 🤖 models/
│ ├── classification/best_transfer_model.h5
│ └── detection/yolov8n_birddrone/weights/best.pt
│
├── 🧾 src/
│ ├── train_transfer.py
│ ├── eval_classification.py
│ ├── yolo_train.py
│ ├── yolo_infer.py
│ ├── utils.py
│ └── check_paths.py
│
└── 🌐 streamlit_app/
└── app.py
```
---

## 🏆 Results Snapshot
```
| Model                      | Outcome                         |
|----------------------------|---------------------------------|
| **MobileNetV2 Classifier** | Predicts *Bird vs Drone*        |
| **YOLOv8 Detection**       | Detects **both** simultaneously |
```
📌 *The system automatically switches based on user selection.*

---

## 🔧 Setup & Execution

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
2️⃣ Run application
```
cd streamlit_app
streamlit run app.py
```
🔮 Future Upgrade Ideas

🟣 Add Bird Species Identification
🟣 Deploy to cloud (AWS / Azure / Streamlit Cloud)
🟣 Add live webcam drone alert system
🟣 Integrate geo-fencing & buzzer warning

👨‍💻 GV Jayanth
AI & ML Developer | Computer Vision | Generative AI
🔗 LinkedIn: https://www.linkedin.com/in/gv-jayanth

If this project inspires you, please ⭐ star the repository — it motivates future innovation!

<div align="center">
✨ Giving AI the eyes to protect our skies 🦅🛰️
</div> ```
