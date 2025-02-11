# **Real-Time Data Collection and Anomaly Detection Using a Fine-Tuned Long Short-Term Memory (LSTM) Model**

## **Overview**
This directory contains the **real-time data extraction and anomaly detection** Python scripts utilizing a **fine-tuned LSTM model** trained on **60-second feature data** extracted from the **Cold Spray Machine**.  

### **Key Features Monitored:**
- **Main Gas Flow Rate**  
- **Chamber Pressure**  
- **Powder Feeder Gas Flow Rate**  
- **Powder Feeder Pressure**  

The model predicts whether the incoming data points indicate an **anomaly**, enabling **predictive maintenance** and **quality assurance** in real time.

### **Enhancements in This Implementation:**
✅ **Real-time anomaly detection** to monitor the Cold Spray Machine.  
✅ **Clock synchronization fixes** to ensure data integrity between the **server clock** and **client clock**.  
✅ **Seamless data extraction and processing** for enhanced monitoring accuracy.  

---

## **Prerequisites**
- Python **3.9** or later  
- OPC UA communication protocol enabled for the **Cold Spray Machine**  

---

## **Setup**  
1. **Install all required dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```

2. Run the python scrip in terminal to start data extraction (different type of nozzles are fine-tuned using different training dataset):
    - for glass nozzle 
    ```bash
    python glass_data_acquisition.py
    ```

    - for polymer nozzle
    ```bash
    python polymer_data_acquisition.py
    ```
