<h1 styles="color:red"><b>Plant diesease detection system</b></h1>
<h3>training on plant Diesease  classification for AICTE internship cycle 4</h3>
Dataset is taken from:<a href="https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset">https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset</a><br>

## Run locally

Install the app dependencies and start Streamlit:

```bash
python -m pip install -r requirements.txt
streamlit run test.py
```

The Gemini advice feature is optional. Set `GEMINI_API_KEY` in your environment to enable it.

## Deploy to Streamlit Community Cloud

This repository is prepared for Streamlit Community Cloud. The model file is tracked with Git LFS; ensure Git LFS is installed and the model has been added and pushed as an LFS object. Then create an app at <https://share.streamlit.io> using this repository, the `main` branch, and `test.py` as the app file. In the app's Advanced settings, add `GEMINI_API_KEY` as a secret to enable Gemini advice. The disease classifier works without that key.


