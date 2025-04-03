import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

def main():
    st.title("Naïve Bayes Classifier")
    st.write("Upload an Excel or CSV dataset to train and test a Naïve Bayes model.")
    
    uploaded_file = st.file_uploader("Upload File", type=["xls", "xlsx", "csv"])
    
    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        st.write("### Preview of Dataset:")
        st.write(df.head())
        
        target_column = st.selectbox("Select the target column (outcome variable):", df.columns)
        
        if target_column:
            feature_columns = [col for col in df.columns if col != target_column]
            X = df[feature_columns]
            y = df[target_column]
            
            # Handle missing values
            imputer = SimpleImputer(strategy='most_frequent')
            X = pd.DataFrame(imputer.fit_transform(X), columns=feature_columns)
            y = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())
            
            # Encode categorical variables
            label_encoders = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
            
            # Convert numerical target to categorical if necessary
            if y.dtype in ['int64', 'float64']:
                num_bins = len(set(pd.qcut(y, q=4, duplicates='drop').cat.categories))
                labels = ["Low", "Medium", "High", "Very High"][:num_bins]
                y = pd.qcut(y, q=num_bins, labels=labels, duplicates='drop')
                y = y.astype(str)  # Convert categories to string
                y_encoder = LabelEncoder()
                y = y_encoder.fit_transform(y)
            elif y.dtype == 'object':
                y_encoder = LabelEncoder()
                y = y_encoder.fit_transform(y)
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model_type = st.selectbox("Choose Naïve Bayes Model:", ["GaussianNB", "MultinomialNB", "BernoulliNB"])
            
            if model_type == "GaussianNB":
                model = GaussianNB()
            elif model_type == "MultinomialNB":
                model = MultinomialNB()
            else:
                model = BernoulliNB()
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            st.write("### Model Accuracy:")
            st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            
            st.write("### Classification Report:")
            report = classification_report(y_test, y_pred, target_names=y_encoder.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write(report_df.style.format(precision=2))
            
            st.write("### Make a Prediction:")
            input_data = []
            for col in feature_columns:
                val = st.text_input(f"Enter value for {col}:")
                input_data.append(val)
            
            if st.button("Predict"):
                input_df = pd.DataFrame([input_data], columns=feature_columns)
                
                for col in label_encoders:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].apply(lambda x: x if x in label_encoders[col].classes_ else label_encoders[col].classes_[0])
                        input_df[col] = label_encoders[col].transform(input_df[col])
                
                input_df = input_df.apply(pd.to_numeric, errors='coerce')
                input_df = input_df.fillna(input_df.mode().iloc[0])
                
                prediction = model.predict(input_df)
                predicted_label = y_encoder.inverse_transform(prediction) if 'y_encoder' in locals() else prediction
                
                st.write(f"Predicted Outcome: {predicted_label[0]}")

if __name__ == "__main__":
    main()