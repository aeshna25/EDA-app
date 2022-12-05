
import streamlit as st 
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# from streamlit_pandas_profiling import st_profile_report
# import pandas_profiling
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn import model_selection
st.set_option('deprecation.showPyplotGlobalUse', False)
matplotlib.use('Agg')
from pathlib import Path
from PIL import Image

st.title("Data analytics and modeling")
current_dir = Path(__file__).parent if "__file__" in locals() else Path.cwd()
image=  Image.open(r'Outlier.gif')
st.image(image,use_column_width=True)

def main():
    activities=['EDA','Visualization','Modeling','About us'] #'Pandas-Profiling'
    options= st.sidebar.selectbox('Select option: ',activities)

    if options=='EDA':
        st.subheader('Expolaritary data analysis')
        data = st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        if data is not None:
            st.success('Data successfully loaded!')
        else:
            st.error('Data not loaded')
        if data is not None:
            df = pd.read_csv(data)
            st.dataframe(df.head(25))
            if st.checkbox("Display rows and columns"):
                st.write(df.shape)
            if st.checkbox("Display columns"):
                st.write(df.columns)
            if st.checkbox("Select multiple columns"):
                selected_columns = st.multiselect("Select prefered columns:",df.columns)
                df1 = df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Summary'):
                st.write(df1.describe().T) #transpose
            if st.checkbox('Display Null values'):
                st.write(df1.isnull().sum())
            if st.checkbox("Display data types"):
                st.write(df1.dtypes)
            if st.checkbox('Display correlation of data various columns'):
                st.write(df1.corr())
    # elif options=='Pandas-Profiling':


    elif options=='Visualization':
        st.subheader('Data Visualization')
        data = st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        if data is not None:
            st.success('Data successfully loaded!')
        else:
            st.error('Data not loaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(25))
            if st.checkbox('Select Multiple columns to plot'):
                selected_columns=st.multiselect('Select your preferred columns',df.columns)
                df1=df[selected_columns]
                st.dataframe(df1)
            if st.checkbox('Display Heatmap'):
                st.write(sns.heatmap(df1.corr(),vmax=1,square=True,annot=True,cmap='viridis'))
                st.pyplot()
            if st.checkbox('Display Pairplot'):
                st.write(sns.pairplot(df1,diag_kind='kde'))
                st.pyplot()
            if st.checkbox('Display Pie Chart'):
                all_columns=df.columns.to_list()
                pie_columns=st.selectbox("select column to display",all_columns)
                pieChart=df[pie_columns].value_counts().plot.pie(autopct="%1.1f%%")
                st.write(pieChart)
                st.pyplot()


    elif options=='Modeling':
        st.subheader("Model Building")
        data=st.file_uploader("Upload dataset:",type=['csv','xlsx','txt','json'])
        if data is not None:
            st.success('Data successfully loaded!')
        else:
            st.error('Data not loaded')
        if data is not None:
            df=pd.read_csv(data)
            st.dataframe(df.head(50))
            if st.checkbox('Select Multiple columns'):
                new_data=st.multiselect("Select your preferred columns. NB: Let your target variable be the last column to be selected",df.columns)
                df1=df[new_data]
                st.dataframe(df1)


				#Dividing my data into X and y variables
                X=df1.iloc[:,0:-1]
                y=df1.iloc[:,-1]

            seed = st.sidebar.slider('Seed',1,200) #random state
            classifier_name= st.sidebar.selectbox('Select your preferred classifier:',('KNN','SVM','LR','naive_bayers','Decision Trees'))

            def add_parameter(name_of_clf):
                params=dict()
                if name_of_clf=='SVM':
                    C=st.sidebar.slider('C',0.01, 15.0)
                    params['C']=C
                else:
                    name_of_clf=='KNN'
                    K=st.sidebar.slider('K',1,15)
                    params['K']=K
                    return params
            params=add_parameter(classifier_name) #calling the function
			
#defing a function for our classifier

            def get_classifier(name_of_clf,params):
                clf= None
                if name_of_clf=='SVM':
                    clf=SVC(C=params['C'])
                elif name_of_clf=='KNN':
                    clf=KNeighborsClassifier(n_neighbors=params['K'])
                elif name_of_clf=='LR':
                    clf=LogisticRegression()
                elif name_of_clf=='naive_bayes':
                    clf=GaussianNB()
                elif name_of_clf=='decision tree':
                    clf=DecisionTreeClassifier()
                else:
                    st.warning('Select your choice of algorithm')
                return clf
            clf=get_classifier(classifier_name,params)


            X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=seed)

            clf.fit(X_train,y_train) #80% for training
            y_pred=clf.predict(X_test)
            st.write('Predictions:',y_pred)
            accuracy=accuracy_score(y_test,y_pred)
            st.write('Name of classifier:',classifier_name)
            st.write('Accuracy',accuracy)


    
    elif options=='About us':
        st.markdown('This analysis here is to demonstrate how we can present our work to our stakeholders in an interactive way by building a web app using Python')
        st.write('This project is build with the help on Full stack Data scientist Bootcamp')
        st.markdown('Name: Aeshna Gupta')
        st.markdown('Job profile: Senior analyst at Dell Technologies')
        st.markdown('Reach out to me on [Linkedin](https://www.linkedin.com/)')
        




if __name__=='__main__':
    main()