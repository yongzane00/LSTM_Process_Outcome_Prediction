### Make sure you have Python > 3.7 and pip installed

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
