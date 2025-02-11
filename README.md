# Predictive Quality & Maintenance through Long-Short-Term Memory (Cold Spray Machine)

## Description
Conventionally, the quality of additive manufacturing processes—such as Cold Spray—is assessed using test coupons. However, these methods are often destructive, labor-intensive, and time-consuming. Additionally, machine health is typically evaluated based on criteria such as porosity and tensile strength, requiring extensive manual analysis. To address these challenges, we introduce a data-driven solution that leverages in-process parameters to assess process quality in real time. Since in-process parameters are time-series data, each current value inherently influences future readings. Long Short-Term Memory (LSTM), widely used in Natural Language Processing (NLP) for capturing contextual dependencies in sentences, is applied here to model the sequential dependencies in process data.

We have developed a scalable machine learning pipeline for:

- Training on in-process data
- Evaluation (to be implemented)
- Real-time inference for deployment

This approach enables a non-destructive, automated, and efficient method for quality prediction and machine health monitoring in Cold Spray processes.

## Quick Start

If you are using virtual environment: 

1. Create a virtual environment to containerize your working environment

    Run `py -m venv .venv`

2. Activate your virtual environment

    Run `.venv\Scripts\activate`

3. Download all the libraries

    Run `pip install -r requirements.txt`

If you are using conda:

1. Create a virtual environment to containerize your working environment

    Run `conda create -n coldspray python=3.9`

2. Activate your conda environement

    Run `conda activate coldspray`

3. Download all the libraries

    Run `conda env create -f requirements.yml`

To run the opcua-client GUI:

1. Install opcua-client

    Run `pip3 install opcua-client`

2. To run the GUI

    Run `opcua-client`
