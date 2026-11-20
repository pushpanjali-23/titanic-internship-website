import streamlit as st
import pandas as pd, numpy as np
import plotly.express as px
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

st.set_page_config(page_title="Titanic | Data & ML",page_icon="🚢",layout="wide")
st.markdown("""<style>
.block-container{max-width:1250px;padding-top:1.6rem}
.hero{padding:2rem 2.2rem;border-radius:20px;background:linear-gradient(120deg,#10233f,#1c5270);color:white}
.hero h1{color:white;font-size:2.45rem}.hero p{color:#e5eff7}
.kpi{background:#f1f5f9;border:1px solid #dbe4ed;padding:1rem;border-radius:14px}
.kpi small{color:#526273}.kpi strong{display:block;font-size:1.8rem;color:#142b45}
.note{padding:.9rem 1rem;background:#eef6fb;border-left:4px solid #277da1;border-radius:8px}
</style>""",unsafe_allow_html=True)

@st.cache_data
def default_data():
    url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    d=pd.read_csv(url)
    return d

@st.cache_resource
def train(d):
    cols=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    num=["Pclass","Age","SibSp","Parch","Fare"]; cat=["Sex","Embarked"]
    X=d[cols].copy(); y=d.Survived.astype(int)
    prep=ColumnTransformer([
      ("num",Pipeline([("impute",SimpleImputer(strategy="median")),("scale",StandardScaler())]),num),
      ("cat",Pipeline([("impute",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)
    ])
    model=Pipeline([("prep",prep),("lr",LogisticRegression(max_iter=1500,random_state=42))])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    model.fit(Xtr,ytr); pred=model.predict(Xte)
    metrics={"accuracy":accuracy_score(yte,pred),"precision":precision_score(yte,pred,zero_division=0),
             "recall":recall_score(yte,pred,zero_division=0),"f1":f1_score(yte,pred,zero_division=0),
             "y":yte,"pred":pred,"ntr":len(Xtr),"nte":len(Xte)}
    return model,metrics

with st.sidebar:
    st.title("🚢 Titanic Project")
    st.caption("Honeywell × Bharath Cares | AI Internship")
    page=st.radio("Navigate",["Project Overview","Dataset Explorer","Data Preparation","Survival Insights","Machine Learning","Try a Prediction","Conclusion & Viva"])
    upload=st.file_uploader("Upload Kaggle train.csv (optional)",type="csv")
try:
    df=pd.read_csv(upload) if upload else default_data()
except Exception as e:
    st.title("Titanic Data Analysis & Machine Learning")
    st.error("Could not load the sample dataset. Upload Kaggle's train.csv using the sidebar.")
    st.markdown("[Get the Titanic dataset from Kaggle](https://www.kaggle.com/competitions/titanic/data)")
    st.stop()
required=["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
if not set(required).issubset(df.columns):
    st.error("Dataset must contain: "+", ".join(required)); st.stop()
for c in ["Survived","Pclass","Age","SibSp","Parch","Fare"]: df[c]=pd.to_numeric(df[c],errors="coerce")
df["Sex"]=df["Sex"].astype("string").str.lower().str.strip()
df["Embarked"]=df["Embarked"].astype("string").str.upper().str.strip()
data=df.dropna(subset=["Survived"]).copy(); data["Survived"]=data.Survived.astype(int)
if data.Survived.nunique()!=2: st.error("Survived must contain both 0 and 1."); st.stop()
surv=data.Survived.mean()

if page=="Project Overview":
    st.markdown("""<div class="hero"><h1>🚢 Titanic Data Analysis & Machine Learning</h1>
    <p>From historical passenger records to data-driven survival predictions.</p>
    <p><b>Summer Internship–II · 2026–2027 · Department of CSE, UVCE</b></p>
    <p>Honeywell × Bharath Cares | Artificial Intelligence Internship Case Study</p></div>""",unsafe_allow_html=True)
    st.write("This project studies the Kaggle Titanic passenger manifest and frames survival prediction as a **supervised binary classification** problem.")
    cols=st.columns(4)
    for c,label,val in zip(cols,["Passenger records","Predictor features","Target","Overall survival"],[f"{len(data):,}","7","Survived",f"{surv:.1%}"]):
        c.markdown(f'<div class="kpi"><small>{label}</small><strong>{val}</strong></div>',unsafe_allow_html=True)
    st.subheader("Five-step workflow")
    for h,t in [("01 · Understand","Define Survived: 0 = No, 1 = Yes."),("02 · Explore","Inspect rows, columns, data types and distributions."),("03 · Prepare","Handle missing values and encode categorical features."),("04 · Model","Train Logistic Regression with an 80:20 split."),("05 · Evaluate","Measure performance on unseen test records.")]:
        with st.container(border=True): st.markdown(f"**{h}**"); st.write(t)
    st.markdown('<div class="note">Educational model based on historical data. Predictions are statistical estimates, not explanations of an individual passenger’s fate.</div>',unsafe_allow_html=True)

elif page=="Dataset Explorer":
    st.title("Dataset Explorer")
    a,b,c=st.columns(3); a.metric("Rows",len(df)); b.metric("Columns",len(df.columns)); c.metric("Missing cells",int(df.isna().sum().sum()))
    st.subheader("Sample passenger records"); st.dataframe(df.head(15),use_container_width=True)
    st.subheader("Schema and missingness")
    s=pd.DataFrame({"Column":df.columns,"Data type":df.dtypes.astype(str).values,"Missing":df.isna().sum().values,"Missing %":(df.isna().mean()*100).round(1).values})
    st.dataframe(s,use_container_width=True,hide_index=True)
    st.subheader("Statistical summary"); st.dataframe(df.describe(include="all").T,use_container_width=True)

elif page=="Data Preparation":
    st.title("Data Preparation")
    miss=df.isna().sum().sort_values(ascending=False); miss=miss[miss>0]
    if len(miss): st.plotly_chart(px.bar(x=miss.index,y=miss.values,labels={"x":"Column","y":"Missing values"},title="Missing values by column"),use_container_width=True)
    st.write("Preparation approach described in the internship presentation:")
    cabin=f"{int(df.Cabin.isna().sum())} missing" if "Cabin" in df else "Column not present"
    rows=[["Cabin",cabin,"Dropped from initial baseline due to high sparsity (687 / 891 = 77.1% in the presentation)."],
          ["Age",f"{int(df.Age.isna().sum())} missing","Median imputation."],
          ["Embarked",f"{int(df.Embarked.isna().sum())} missing","Mode imputation; presentation identifies Southampton ('S')."],
          ["Sex","Categorical","Binary encoding."],["Embarked","Categorical","One-hot encoding."]]
    st.dataframe(pd.DataFrame(rows,columns=["Field","Observed status","Preparation approach"]),use_container_width=True,hide_index=True)
    st.subheader("Seven model features"); st.write("Pclass, Sex, Age, SibSp, Parch, Fare and Embarked.")
    st.caption("In this implementation, numeric values are median-imputed and scaled; categorical values are most-frequent imputed and one-hot encoded. Cabin, Name, Ticket and PassengerId are not predictors.")

elif page=="Survival Insights":
    st.title("Survival Insights")
    st.caption("Charts are calculated from the dataset currently loaded.")
    sex=data.groupby("Sex",dropna=False).Survived.mean().reset_index(); sex["Rate (%)"]=sex.Survived*100
    cls=data.groupby("Pclass").Survived.mean().reset_index(); cls["Rate (%)"]=cls.Survived*100
    a,b=st.columns(2)
    a.plotly_chart(px.bar(sex,x="Sex",y="Rate (%)",color="Sex",title="Survival rate by sex",text_auto=".1f"),use_container_width=True)
    b.plotly_chart(px.bar(cls,x="Pclass",y="Rate (%)",title="Survival rate by passenger class",text_auto=".1f"),use_container_width=True)
    ages=data.dropna(subset=["Age"]).copy(); ages["Age group"]=pd.cut(ages.Age,[0,12,18,35,60,100],labels=["0–12","13–18","19–35","36–60","60+"])
    ag=ages.groupby("Age group",observed=False).Survived.mean().reset_index(); ag["Rate (%)"]=ag.Survived*100
    st.plotly_chart(px.bar(ag,x="Age group",y="Rate (%)",title="Survival rate by age group",text_auto=".1f"),use_container_width=True)
    fam=data.copy(); fam["Family size"]=fam.SibSp.fillna(0)+fam.Parch.fillna(0)+1
    fam["Family group"]=np.select([fam["Family size"]==1,fam["Family size"].between(2,4)],["Alone","Small family"],default="Larger family")
    fg=fam.groupby("Family group").Survived.mean().reset_index(); fg["Rate (%)"]=fg.Survived*100
    st.plotly_chart(px.bar(fg,x="Family group",y="Rate (%)",title="Survival rate by family group",text_auto=".1f"),use_container_width=True)
    fr=data.loc[data.Sex=="female","Survived"].mean(); mr=data.loc[data.Sex=="male","Survived"].mean()
    st.write(f"Female survival: **{fr:.1%}** · Male survival: **{mr:.1%}** · Difference: **{(fr-mr)*100:.1f} percentage points**")
    st.caption("Presentation values include 74.2% female, 18.9% male, and class rates of 63.0%, 47.3% and 24.2%. Actual charts depend on the loaded data.")

elif page=="Machine Learning":
    st.title("Logistic Regression Model")
    st.write("Target: Survived (0 = did not survive, 1 = survived).")
    model,m=train(data)
    cs=st.columns(4)
    for c,l,v in zip(cs,["Accuracy","Precision","Recall","F1-score"],[m["accuracy"],m["precision"],m["recall"],m["f1"]]): c.metric(l,f"{v:.1%}")
    st.write(f"Train rows: **{m['ntr']}** · Test rows: **{m['nte']}** · Split: **80:20** · Random state: **42**")
    baseline=max(data.Survived.mean(),1-data.Survived.mean()); st.write(f"Majority-class baseline: **{baseline:.1%}**")
    cm=confusion_matrix(m["y"],m["pred"],labels=[0,1])
    st.plotly_chart(px.imshow(cm,text_auto=True,x=["Predicted No","Predicted Yes"],y=["Actual No","Actual Yes"],color_continuous_scale="Blues",title="Confusion matrix"),use_container_width=True)
    st.subheader("Classification report"); st.dataframe(pd.DataFrame(classification_report(m["y"],m["pred"],output_dict=True,zero_division=0)).T.round(3),use_container_width=True)
    st.info("The presentation reports 81.0% accuracy on 179 held-out records. This app displays its own measured score and does not hardcode 81%.")
    st.caption("Pipeline: median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categories; Logistic Regression classifier.")

elif page=="Try a Prediction":
    st.title("Try a Passenger Survival Prediction")
    st.write("Enter hypothetical passenger details. The trained model will estimate survival probability.")
    model,_=train(data)
    with st.form("predict"):
        l,r=st.columns(2)
        pclass=l.selectbox("Passenger class",[1,2,3],index=2); sex=l.selectbox("Sex",["female","male"]); age=l.slider("Age",0,80,28); fare=l.number_input("Fare",0.0,600.0,32.0)
        sibsp=r.number_input("Siblings / spouses aboard",0,10,0); parch=r.number_input("Parents / children aboard",0,10,0); embarked=r.selectbox("Port",["S","C","Q"],format_func=lambda x:{"S":"Southampton","C":"Cherbourg","Q":"Queenstown"}[x])
        go=st.form_submit_button("Predict survival",type="primary",use_container_width=True)
    if go:
        x=pd.DataFrame([{"Pclass":pclass,"Sex":sex,"Age":age,"SibSp":sibsp,"Parch":parch,"Fare":fare,"Embarked":embarked}])
        probability=float(model.predict_proba(x)[0][1]); pred=int(model.predict(x)[0])
        if pred: st.success(f"Prediction: Survived · Estimated probability: {probability:.1%}")
        else: st.warning(f"Prediction: Did not survive · Estimated survival probability: {probability:.1%}")
        st.progress(probability)
        st.caption("This is a statistical estimate from historical data, not a reliable judgement about a real person's fate.")

else:
    st.title("Conclusion & Future Scope")
    st.write("The project demonstrates a structured workflow from historical data exploration to a measurable binary classification model.")
    st.subheader("Key takeaways")
    st.markdown("- Survival was associated with passenger demographics and ticket class.\\n- The presentation identifies gender as a strong predictive signal.\\n- Data quality checks and missing-value handling are essential.\\n- A held-out test set helps estimate performance on unseen records.\\n- Precision, recall, F1-score and confusion matrix complement accuracy.")
    st.subheader("Future scope from the presentation")
    st.markdown("1. Explore Random Forest.\\n2. Engineer Family Size = SibSp + Parch + 1.\\n3. Build an interactive Streamlit dashboard.\\n4. Organise the project on GitHub.")
    st.subheader("Quick viva questions")
    qa=[("What is the target variable?","Survived: 0 = No, 1 = Yes."),("Why supervised learning?","Training records contain known labels that the model learns to predict."),("Why Logistic Regression?","It estimates the probability of a binary outcome."),("Why split the data?","To evaluate the model on records it did not train on."),("What is precision?","Among positive predictions, the fraction that are correct."),("What is recall?","Among actual positive cases, the fraction correctly identified."),("What is a limitation?","The dataset does not include every evacuation factor, such as lifeboat proximity or crew decisions.")]
    for q,a in qa:
        with st.expander(q): st.write(a)

st.divider()
st.caption("Titanic Data Analysis & Machine Learning · Educational internship project")
