import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit_dynamic_filters import DynamicFilters

hide_st_style = """
                <style>
                #MainMenu {visibility : hidden;}
                footer {visibility : hidden;}
                header {visibility : hidden;}
                </style>
                """

st.set_page_config(
    page_title="PayLater User Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Hide Streamlit's default styles
st.markdown(hide_st_style, unsafe_allow_html=True)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="PayLater User Prediction",
        options=["Home", "Analysis", "Predictions"],
        default_index=0,
    )

# Fetch data
@st.cache_data
def fetch_data():
    df = pd.read_csv('initial_cleaning_rounded.csv')
    return df

df = fetch_data()

# ========= HOME TAB =========
if selected == "Home":
    st.title('WELCOME TO PAYLATER USER PREDICTION TOOLS')
    st.divider()
    st.header("About")
    st.markdown('''
    Welcome to the **PayLater User Prediction Tools**!  
    This application is designed to help businesses analyze customer behavior and predict potential PayLater users.  
    Using this tool, you can:  

    - Gain insights into survey data and make data-driven decisions to optimize marketing strategies.  
    - Predict the likelihood of a user opting for PayLater services.  

    This page is using the template from [github.com/Divvyanshiii/SMART](https://github.com/Divvyanshiii/SMART/blob/main/app.py).
    ''')

# ========= ANALYSIS TAB =========
if selected == "Analysis":
    st.title("Data Analysis 📊")  

    dynamic_filters = DynamicFilters(df, filters=['Gender', 'Location', "E-Paylater User Status","Job Status", "Monthly Income"])

    with st.sidebar:
        st.header("Filters")
        dynamic_filters.display_filters()
    
    # Get the filtered data
    filtered_df = dynamic_filters.df 

    # Table
    dynamic_filters.display_df()

    st.divider()

    col1, col2, col3  = st.columns(3)

    with col1:
        # Crosstab
        education_paylater = pd.crosstab(filtered_df['Educational Background'], filtered_df['E-Paylater User Status'])

        # Education vs PayLater
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        education_paylater.plot(kind='bar', ax=ax1, color=['skyblue', 'salmon'], edgecolor='black')
        ax1.set_title('Pengguna PayLater Berdasarkan Tingkat Pendidikan', fontsize=16)
        ax1.set_xlabel('Educational Background', fontsize=12)
        ax1.set_ylabel('Jumlah Responden', fontsize=12)
        ax1.legend(title='PayLater User Status', labels=['No', 'Yes'])
        plt.xticks(rotation=45)
        st.pyplot(fig1)
        
    with col2:
        # Crosstab
        paylater_vs_gender = pd.crosstab(filtered_df['Gender'], filtered_df['E-Paylater User Status'])

        # Gender vs PayLater
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        paylater_vs_gender.plot(kind='bar', ax=ax2, color=['skyblue', 'salmon'], edgecolor='black')
        ax2.set_title('Distribusi Pengguna PayLater Berdasarkan Gender', fontsize=16)
        ax2.set_xlabel('Gender', fontsize=12)
        ax2.set_ylabel('Jumlah Pengguna', fontsize=12)
        ax2.legend(title='PayLater User Status')
        plt.xticks(rotation=0)
        st.pyplot(fig2)

    with col3:
        # Crosstab
        paylater_vs_income = pd.crosstab(filtered_df['Monthly Income'], filtered_df['E-Paylater User Status'])

        # Income vs PayLater
        fig3, ax3 = plt.subplots(figsize=(8, 6))
        paylater_vs_income.plot(kind='bar', ax=ax3, color=['skyblue', 'salmon'], edgecolor='black')
        ax3.set_title('Distribusi Pengguna PayLater Berdasarkan Income', fontsize=16)
        ax3.set_xlabel('Income', fontsize=12)
        ax3.set_ylabel('Jumlah Pengguna', fontsize=12)
        ax3.legend(title='PayLater User Status')
        plt.xticks(rotation=75)
        st.pyplot(fig3)

# ========= PREDICTIONS TAB =========
if selected == "Predictions":
    st.title('PayLater User Predictions') 
    
    def user_input():
        education = st.sidebar.selectbox('Educational Background', df['Educational Background'].unique(), key='education')
        job_status = st.sidebar.selectbox('Job Status', df['Job Status'].unique(), key='job_status')
        monthly_income = st.sidebar.selectbox('Monthly Income', df['Monthly Income'].unique(), key='monthly_income')
        expenditure = st.sidebar.selectbox('Online Shopping Expenditure Percentage', df['Online Shopping Expenditure Percentage'].unique(), key='expenditure')
        impulsive_score = st.sidebar.selectbox('Impulsive Buying Behavior Score', df['Impulsive Buying Behavior Score'].unique(), key='impulsive_score')
        promotion_score = st.sidebar.selectbox('Promotion Score', df['Promotion Score'].unique(), key='promotion_score')
        social_score = st.sidebar.selectbox('Social Influence Score', df['Social Influence Score'].unique(), key='social_score')
        self_control_score = st.sidebar.selectbox('Self Control Score', df['Self Control Score'].unique(), key='self_control_score')  # Unique key added

        data = {
            'Educational Background': education,
            'Job Status': job_status,
            'Monthly Income': monthly_income,
            'Online Shopping Expenditure Percentage': expenditure,
            'Impulsive Buying Behavior Score': impulsive_score,
            'Promotion Score': promotion_score,
            'Social Influence Score': social_score,
            'Self Control Score': self_control_score
            }

        features = pd.DataFrame(data, index=[0])
        return features

    input_data = user_input()

    st.subheader('User Input')
    st.write(input_data)

    # Load the model and make predictions
    load_model = joblib.load("best_model_pipeline.pkl")
    prediction = load_model.predict(input_data)

    st.subheader('Will They Be A Potential User?')
    st.write('Based on the input, the user is: ')
    if prediction == 1:
        st.success('Potential User')

        st.divider()

        st.subheader('Recommendation Tools')
        st.markdown('''
        How we built these tools:
        - Spending Limit Recommendation is built based on the customer's Monthly Income, Online Shopping Expenditure, and studi literature [Reference](chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://kredivocorp.com/wp-content/uploads/2024/06/Laporan-Perilaku-Pengguna-Paylater-Indonesia-2024-Kredivo.pdf).
        - Promotion Channel Recommendation is based on [Target Internet: How Different Age Groups Are Using Social Media 2024](https://targetinternet.com/resources/how-different-age-groups-are-using-social-media-2024/)
        ''')
        
        monthly_income = input_data['Monthly Income'].iloc[0]
        if monthly_income == "Rp 7,500,001 - Rp 10,000,000":
            rate = 0.2
            income = 8750000
            spending_limit = rate * income
        elif monthly_income ==  "Rp 5,000,001 - Rp 7,500,000":
            rate = 0.1
            income = 6250000
            spending_limit = rate * income
        else:
            spending_limit = 500000

        st.subheader('Spending Limit Recommendation')
        st.markdown(f'Recommended spending limit for your user is **Rp {spending_limit:,}**')

        # Promotion Channel Recommendation
        st.subheader('Promotion Channel Recommendation')
        age = st.number_input('Enter the user\'s age:', min_value=0, max_value=120, step=1)
        this_year = pd.Timestamp.now().year
        birth_year = this_year - age

        if birth_year <= 1964:
            channels = ["Facebook"]
        elif 1965 <= birth_year <= 1980:
            channels = ["Facebook", "Instagram", "Pinterest"]
        elif 1981 <= birth_year <= 1996:
            channels = ["Facebook", "Instagram", "Twitter"]
        else:
            channels = ["TikTok", "Instagram"]

        st.write('Recommended promotion channels for this user are ranked as follows:')
        st.markdown("".join(f"{i+1}. {channel}\n" for i, channel in enumerate(channels)))

    else:
        st.error('Not Potential User')
    
    