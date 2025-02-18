# **Training Pipeline for Long Short-Term Memory (LSTM) Model Using Machine In-Process Parameters for Predictive Quality and Outcome**

## **Overview**
In this directory, you will find the **training pipeline** for the **LSTM model**, trained using **brand-new nozzle data** and four key features extracted from the Cold Spray Machine:

- **Main Gas Flow Rate**  
- **Chamber Pressure**  
- **Powder Feeder Gas Flow Rate**  
- **Powder Feeder Pressure**  

Additionally, this directory includes **inference results** on unseen data, consisting of a separate set of **good and bad nozzle data**.

The results successfully demonstrate that, even with different nozzle types, the **LSTM model can identify a higher number of anomalous data points** in the **bad-conditioned nozzle** compared to the **good-conditioned nozzle**.

---

## **Prerequisites**
- Python **3.9** or later  
- CUDA-supported PyTorch (for GPU acceleration)  

