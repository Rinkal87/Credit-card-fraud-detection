import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

# Streamlit app
st.title('Credit Card Fraud Detection')

uploaded_file = st.file_uploader('Upload your CSV file with transaction data', type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file,skiprows=[x for x in range(1,280000)])
    st.write('Uploaded Data:')
    st.write(data.head())
    
    # Preprocess the data
    X = data.drop(columns=['Class'])
    y = data['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    if st.button('Random Forest'):
        # Train a RandomForestClassifier
        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        # Make predictions on the uploaded data
        predictions = model.predict(scaler.transform(X))
        data['Fraud_Prediction'] = predictions
        st.subheader('Prediction Results:')
        st.write(data)
        st.write('Fraudulent transactions are marked with 1, non-fraudulent with 0')
        c=0
        for i in range(len(data)):
            if data.loc[i,'Class']==data.loc[i,'Fraud_Prediction']:
                c+=1
        st.write(f'Correctly classified records: {c} out of {len(data)}')
        # Evaluate the model
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation:")
        st.text(classification_report(y_test, y_pred))
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)
        fig, ax = plt.subplots()
        correctly_classified = cm[0][0] + cm[1][1]
        st.write(f'Correctly classified records: {correctly_classified}')
        # Feature Importance
        st.subheader('Feature Importance')
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        feature_importance.plot(kind='bar', ax=ax)
        ax.set_ylabel('Importance')
        st.pyplot(fig)
        
        
        
    if st.button('gradient boosting'):
        # Train a RandomForestClassifier
        model = GradientBoostingClassifier()
        model.fit(X_train, y_train)

        # Make predictions on the uploaded data
        predictions = model.predict(scaler.transform(X))
        data['Fraud_Prediction'] = predictions
        st.subheader('Prediction Results:')
        st.write(data)
        st.write('Fraudulent transactions are marked with 1, non-fraudulent with 0')
        c=0
        for i in range(len(data)):
            if data.loc[i,'Class']==data.loc[i,'Fraud_Prediction']:
                c+=1
        st.write(f'Correctly classified records: {c} out of {len(data)}')
        # Evaluate the model
        y_pred = model.predict(X_test)
        st.subheader("Model Evaluation:")
        st.text(classification_report(y_test, y_pred))
        st.subheader('Confusion Matrix')
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        st.pyplot(fig)
        fig, ax = plt.subplots()
        correctly_classified = cm[0][0] + cm[1][1]
        st.write(f'Correctly classified records: {correctly_classified}')
        # Feature Importance
        st.subheader('Feature Importance')
        feature_importance = pd.Series(model.feature_importances_, index=X.columns)
        feature_importance = feature_importance.sort_values(ascending=False)
        feature_importance.plot(kind='bar', ax=ax)
        ax.set_ylabel('Importance')
        st.pyplot(fig)

        
        
