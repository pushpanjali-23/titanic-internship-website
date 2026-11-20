# Titanic Data Analysis & Machine Learning

A Streamlit website companion for the Honeywell × Bharath Cares AI internship Titanic case study.

## Public deployment (free tier)
1. Create a GitHub account and a **public** repository.
2. Upload `app.py` and `requirements.txt` to the repository root.
3. Open https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app**, select your repository, branch `main`, and main file `app.py`.
5. Deploy. Streamlit Community Cloud provides a public `*.streamlit.app` URL.

## Dataset
The app attempts to load a public copy of the Titanic training data. If that fails, download `train.csv` from https://www.kaggle.com/competitions/titanic/data and upload it in the app sidebar.

## Notes
- Uses the seven features from the internship presentation: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked.
- Trains a real Logistic Regression pipeline with an 80:20 stratified split.
- Metrics are calculated at runtime and are not hardcoded to the presentation's 81%.
- Predictions are educational estimates based on historical data.
