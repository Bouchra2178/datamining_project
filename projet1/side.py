import streamlit as st
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from streamlit_lottie import st_lottie
from projet1_ import *
from projet__1 import *
from algo_classification import *
from algo_clustering import *
from app import *
try:
    from streamlit_option_menu import option_menu
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit_option_menu"])
    from streamlit_option_menu import option_menu
    

def dataset1_options():
    st.sidebar.subheader("Dataset1 Options")
    dataset1_option = st.sidebar.selectbox("Choose an option", ["-", "Manipulation", "Analysis", "Preprocessing"])

    if dataset1_option == "-":
        st.header("Welcome to the DataSet1 !")
    elif dataset1_option == "Manipulation":
        Manipultion()
    elif dataset1_option == "Analysis":
         Analysis()
    elif dataset1_option == "Preprocessing":
         Preprocessing()

def dataset3_options():
    st.sidebar.subheader("Dataset3 Options")
    with st.sidebar:
        dataset2_option = option_menu("Main", ["-", 'Discretiser',"APRIORI","FP-GROWTH","ECLAT"], 
            icons=['cogs', 'tools'], menu_icon="cast", default_index=0)
        
    # dataset1_option = st.sidebar.selectbox("Choose an option", ["-", "Discrétiser", "Apriori"])
    if dataset2_option == "-":
        st.header("Welcome to the DataSet3 !")
    elif dataset2_option == "Discretiser":
        Discretiser()
    elif dataset2_option == "APRIORI":
        if "Data" not in st.session_state:
            st.warning("Please discretize the data first.")
        else:
            Apriorit()
    elif dataset2_option == "FP-GROWTH":
        if "Data" not in st.session_state:
            st.warning("Please discretize the data first.")
        else:
            st.write("Helooooooooo")
    elif dataset2_option == "ECLAT":
        if "Data" not in st.session_state:
            st.warning("Please discretize the data first.")
        else:
            st.write("Heloooooooooo")

def Discretiser():
    st.title("Manipulation des données Dataset3")
    st.subheader("The head of dataset:")
    Data,Head=visualiser()
    st.table(Head)
    selected_column = st.selectbox("Select a Methode:", ["-","Equal-Frequency", "Equal-Width"],index=0)
    dis = st.selectbox("Select a Column:", ["Temperature","Humidity","Rainfall"],index=0)
    st.session_state.choix=dis
    if selected_column=="Equal-Width":
        N_class = st.number_input("Donner le nombre de classe:", min_value=2, max_value=20, step=1,value=8)
    if selected_column=="Equal-Frequency":
        N_class = st.number_input("Donner le nombre de classe:", min_value=2, max_value=20, step=1,value=5)
    if st.button("Discretiser"):
        if selected_column=="Equal-Frequency":
            Data=dicreminisation_frequance(Data,N_class,dis)
            Data = Data[['Temperature',"Humidity","Rainfall", 'Soil', 'Crop', 'Fertilizer']]
        else:
            Data=dicreminisation2(Data,dis,int(N_class))
        st.table(Data.head())
        st.session_state.Data=Data 

def Apriorit():
    st.title("Apriori des données Dataset3")
    choix=st.session_state.choix
    st.subheader("The head of dataset:")
    DATA=st.session_state.Data
    st.table(DATA.head())
    Nbr_transaction=nombre_transaction(DATA)
    min_supp = st.number_input("Entrez la valeur du min_sup entre 0 et 1 :", min_value=0.02, max_value=1.0,value=0.1, step=0.01)
    min_conf = st.number_input("Entrez la valeur du min_conf entre 0 et 1 :", min_value=0.02, max_value=1.0,value=0.1, step=0.01)
    if st.button("Apriori"):
        st.session_state.F_Patterns = Apriori(DATA, float(min_supp), Nbr_transaction, choix)
        st.session_state.Regles, st.session_state.Rules = Get_Rules(DATA, st.session_state.F_Patterns, float(min_conf), choix, "Corelation")
        st.session_state.Apriori_clicked = True 

    if st.session_state.get("Apriori_clicked", False):
        st.subheader("Frequent Patterns:")
        if st.button("Show_patterns"):
            st.session_state.Show_patterns = not st.session_state.get("Show_patterns", False)
        if st.session_state.get("Show_patterns", False):
            st.table(pd.DataFrame(st.session_state.F_Patterns, columns=["Frequent Patterns"]).tail())

        st.subheader("Les Regles d'Association:")
        if st.button("Show_Rules"):
            st.session_state.Show_Rules = not st.session_state.get("Show_Rules", False)
        if st.session_state.get("Show_Rules", False):
            st.table(st.session_state.Regles.tail())

    st.subheader("Test Utilisateurs:")
    selected_options_0 = st.multiselect("Temperature",DATA["Temperature"].unique().tolist(), key="select_columns_0")
    selected_options_1 = st.multiselect("Soil",DATA["Soil"].unique().tolist(), key="select_columns_1")
    selected_options_2 = st.multiselect("Crop",DATA["Crop"].unique().tolist(), key="select_columns_2")
    selected_options_3 = st.multiselect("Fertilizer",DATA["Fertilizer"].unique().tolist(), key="select_columns_3")
    predict1 =st.button("Predict")
    if predict1:
        res=predict(DATA,choix,st.session_state.Rules,selected_options_0,selected_options_1,selected_options_2,selected_options_3)
        st.table(res)
         
def Manipultion():
    st.title("Manipulation des données Dataset1")
    if "dataset1" not in st.session_state:
        st.session_state.dataset1,shape, head, info, describe, null = info_data("Dataset1.csv")
        df=st.session_state.dataset1
    else:
        st.session_state.dataset1,shape, head, info, describe, null= info_data(st.session_state.dataset1)
        df=st.session_state.dataset1
        
    st.subheader("The head of dataset:")
    
    but_head=st.button("Show Head")
    if 'show_head' not in st.session_state:
        st.session_state.show_head = False
    if but_head:
        st.session_state.show_head = not st.session_state.show_head
    if st.session_state.show_head:
        st.write(head)
         
    st.subheader("The information of dataset:")
    
    but_info=st.button("Show Information")
    if 'show_info' not in st.session_state:
        st.session_state.show_info = True
    if but_info:
        st.session_state.show_info = not st.session_state.show_info
    if st.session_state.show_info:
        info_df = parse_info(info)
        st.table(info_df)
        
    
    st.subheader("The shape of dataset")
    but_shape=st.button("Show Shape")
    if 'show_shape' not in st.session_state:
        st.session_state.show_shape = True
    if but_shape:
        st.session_state.show_shape = not st.session_state.show_shape
    if st.session_state.show_shape:
        st.success(f"The number of lines : {shape[0]}.")
        st.success(f"The number of columns : {shape[1]}.")
        # st.markdown('<p style="color: red;"> Success </p>', unsafe_allow_html=True)


    st.subheader("Description of dataset:")
    but_dis=st.button("Show Description")
    if 'show_data' not in st.session_state:
        st.session_state.show_data = True
    if but_dis:
        st.session_state.show_data = not st.session_state.show_data
    if st.session_state.show_data:
        st.table(describe)


    st.subheader("Count of Zeros for Each Column")
    but_null=st.button("Show Null")
    if 'show_null' not in st.session_state:
        st.session_state.show_null = True
    if but_null:
        st.session_state.show_null = not st.session_state.show_null
    if st.session_state.show_null:
        st.table(null)
    
    
    st.subheader("Numbers of OutLiers for Each Column")
    but_out=st.button("Show Outliers")
    if 'show_out' not in st.session_state:
        st.session_state.show_out = True
    if but_out:
        st.session_state.show_out = not st.session_state.show_out
        
    if st.session_state.show_out:
        st.table(count_outliers(st.session_state.dataset1))
        
    st.header("Generate Summary and Plot") 
    # Checkbox to choose whether to analyze all attributes or specific ones
    all_columns_checkbox = st.checkbox("Analyze All Attributes")

    # Multi-select box to choose specific attributes
    selected_columns = st.multiselect("Select Columns:", df.columns)

    # Button to trigger the action
    if st.button("Generate"):
        # Call the function based on user choices
        if all_columns_checkbox:
            g=generate_summary_and_plot2(df,columns=None)
            st.table(g)
        elif selected_columns:
            print(selected_columns)
            plot,g=generate_summary_and_plot2(df, columns=selected_columns[0])
            st.subheader(f"Display the summary DataFrame of the attribute : {selected_columns[0]} ")    
            st.table(g)
            st.subheader(f"Display La Boite a Moustache : {selected_columns[0]} ") 
            st.table(Boite_Mous(df[selected_columns[0]]))
            st.subheader(f"Display the Plot of the attribute : {selected_columns[0]}")
            if plot:
              st.pyplot(plot)
        else:
            st.warning("Please select at least one column.")

def Analysis():
    st.title("Analyse des caractéristiques des attributs")
    if "dataset1" not in st.session_state:
        st.session_state.dataset1 = info_data("Dataset1.csv")[0]
    else:
        st.session_state.dataset1= info_data(st.session_state.dataset1)[0]
    
    df = st.session_state.dataset1

    st.subheader("Measurements of central tendency and deduce symmetries")
    all_columns_checkbox = st.checkbox("Analyze All Attributes",key="1")
    selected_columns = st.multiselect("Select Columns:", df.columns, key="select_columns_1")

    if st.button("Generate Summary and Plot"):
        if all_columns_checkbox:
            g = generate_summary_and_plot2(df, columns=None)
            # ce code pour afficher en noire installe la bib plotly_chart
            # config = {'displayModeBar': False, 'responsive': True, 'staticPlot': False}
            # st.plotly_chart(px.line(g, x="X", y="Y", line_shape="linear", render_mode="svg", title="Graph", labels={"X": "X-axis", "Y": "Y-axis"}), config=config)
            st.line_chart(g)
        elif selected_columns:
            plot, g = generate_summary_and_plot2(df, columns=selected_columns[0])
            st.subheader(f"Display the summary DataFrame of the attribute: {selected_columns[0]} ")
            st.dataframe(g)

            st.subheader(f"Display the Plot of the attribute: {selected_columns[0]}")
            if plot:
                st.pyplot(plot)
        else:
            st.warning("Please select at least one column.")

    st.subheader("Boxplot and the outliers")
    selected_columns2 = st.multiselect("Select Columns:", df.columns, key="select_columns_2")

    if st.button("Plot the Boxplot :"):
        st.subheader(f"Display the BoxPlot of the attribute: {selected_columns2[0]}")
        boxplot = boxplot_with_outliers(df, selected_columns2[0])
        if boxplot:
            st.pyplot(boxplot)
    st.subheader("Histogram Plot")
    selected_columns3 = st.multiselect("Select Columns:", df.columns, key="select_columns_3")

    if st.button("Plot the Histogram :"):
        st.subheader(f"Display the Histogram of the attribute: {selected_columns3[0]}")
        plot = plot_histogram(df, selected_columns3[0])
        if plot:
            st.pyplot(plot)
    st.subheader("Scatter Plot")
    selected_columns4 = st.multiselect("Select 2 columns:", df.columns, key="select_columns_4")

    if st.button("Plot the Scatter Plot"):
        if len(selected_columns4) != 2:
            st.warning("Please select exactly two columns.")
        else:
            st.subheader(f"Display the Scatter Plot with correlation: {selected_columns4[0]} and {selected_columns4[1]}")
            plot = plot_scatter_with_correlation(df, selected_columns4[0],selected_columns4[1])
            if plot:
                st.pyplot(plot)
    st.header("Heat map ")
    all_columns_checkbox2 = st.checkbox("heat map for all Attributes",key="2")
    if st.button("Plot heat map"):
        if all_columns_checkbox2:
            st.subheader(f"Display the HeatMap : ")
            plot=heatmap(df)
            if plot:
                st.pyplot(plot)

        else:
            st.warning("press the checkbox")

def Preprocessing():
    st.title("Preprocessing") 
    if "dataset1" not in st.session_state:
        st.session_state.dataset1 = info_data("Dataset1.csv")[0]
    else:
        st.session_state.dataset1= info_data(st.session_state.dataset1)[0]
        
    df = info_data("Dataset1.csv")[0]
    st.header("Handling missing values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode"], key="select_columns_2")
    if selected_method:
            st.session_state.dataset1=replace_missing(st.session_state.dataset1,selected_method[0])
            st.subheader("Display the Data without missing values:  ")
            st.write(st.session_state.dataset1)
            # Display a table with a custom header for null values
            st.subheader("Null Values Check:")
            st.table(pd.DataFrame({ "Null Values": st.session_state.dataset1.isnull().sum()}))
    st.header("Handling outliers values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode","Q1","Q3","Delete","Regression"], key="select_columns_3")
    if selected_method:
            if selected_method==["Regression"]:
                st.session_state.dataset1=replace_outliers_with_regression(st.session_state.dataset1)
                st.subheader("Display the Data without outliers values:  ")
                st.write(st.session_state.dataset1)
                st.subheader("Outliers Check:")
                count=count_outliers(st.session_state.dataset1)
                st.write(count)
                
            else: 
                st.session_state.dataset1=replace_outliers(st.session_state.dataset1,selected_method[0].lower())
                st.subheader("Display the Data without outliers values:  ")
                st.write(st.session_state.dataset1)
                # Display a table with a custom header for null values
                st.subheader("Outliers Check:")
                count=count_outliers(st.session_state.dataset1)
                st.write(count)
            
    
    st.header("Discretisation")
    method4=st.multiselect("Select a method:", ["Discretisation par Width","Discretisation par Frequency"])
    if method4==["Discretisation par Width"]:
        N_class = st.number_input("Donner le nombre de classe:",key="in454545" ,min_value=2, max_value=20, step=1)
    if st.button("Discretisation"):
        if method4=="Discretisation par Width":
            Data=st.session_state.dataset1
            for column in st.session_state.dataset1.columns:
                Data=dicreminisation(Data,column)
                #ajouter le nombre de class ici comme parametre
        else:
            Data=st.session_state.dataset1
            for column in st.session_state.dataset1.columns:
                Data=dicreminisation(Data,column)
        st.session_state.dataset1=Data
        st.success("Discretisation Successfully")
    
      
    st.header("Data reduction")
    st.subheader("Reduction avec Les Donner Redendantes")
    method=st.multiselect("Select a method:", ["Data reduction par Column","Data reduction par Ligne"])
    # col1, col2 = st.columns(2)
    if st.button("Reduce",key="j4444jdjd"):
        if method[0]=="Data reduction par Column":
            st.session_state.dataset1,d=remove_redundant_column(st.session_state.dataset1)
            st.success(f'Nombre de Columns éliminées : {d}')

        if method[0]=="Data reduction par Ligne":
            st.session_state.dataset1,d=remove_redundant_ligne(st.session_state.dataset1)
            st.success(f'Nombre de Ligne éliminées : {d}')
       
    st.subheader("Reduction Avec Corelation")
    seuil = st.slider("Sélectionnez une valeur entre 0 et 1", 0.0, 1.0, step=0.01)
    if st.button("Reduce",key="jjdjd"):
        st.session_state.dataset1,removed=reduction_par_correlation(st.session_state.dataset1,seuil)
        if len(removed)>0:
            st.success(f'Nombre de Columns eliminer : {len(removed)}')
            st.table(removed)
        else:
            st.success('Aucune Column eliminer ')
    
    st.subheader("Reduction Avec Variance")
    seuil = st.slider("Sélectionnez une valeur entre 0 et 1", 0.0, 0.5, step=0.01,key="3")
    if st.button("Reduce",key="jjdjjd"):
            st.session_state.dataset1,removed=reduction_par_faible_variance(st.session_state.dataset1,seuil)
            if len(removed)>0:
                st.success(f'Nombre de Columns eliminer : {len(removed)}')
                st.table(removed)
            else:
                st.success('Aucune Column eliminer ') 
                                   
    st.header("Data normalization:")
    selected_method = st.multiselect("Select a method:", ["Min-Max-Scaler","Z-score","Logarithme","Racine Carre","Robuste Scalaire"], key="select_columns_5")
    norm=st.button("Start")
    if norm:
        out = st.session_state.dataset1.pop('Fertility')
        if selected_method==["Min-Max-Scaler"]:
            st.session_state.dataset1=min_max(st.session_state.dataset1)
        elif selected_method==["Z-score"] :
            st.session_state.dataset1=z_score(st.session_state.dataset1)
        elif selected_method==["Logarithme"] :
            st.session_state.dataset1=Normalisation_log(st.session_state.dataset1)
        elif selected_method==["Racine Carre"]:
            st.session_state.dataset1=racine_carre(st.session_state.dataset1)
        elif selected_method==["Robuste Scalaire"]:
            st.session_state.dataset1=robuste_scalaire(st.session_state.dataset1)
            
        st.session_state.dataset1["Fertility"]=out
        st.write(st.session_state.dataset1)
    
    col1,col3=st.columns([3,1])
    with col1:
        st.subheader("Restore the Original Data")
    with col3:
        if st.button("Restore"):
            st.session_state.dataset1 = info_data("Dataset1.csv")[0]
 
def dataset2_options():
    st.sidebar.subheader("Dataset2 Options")
    dataset1_option = st.sidebar.selectbox("Choose an option", ["-", "Manipulation",  "Preprocessing","Visualisation"])

    if dataset1_option == "-":
        st.header("Welcome to the DataSet2 !")
    elif dataset1_option == "Manipulation":
        Manipultion2()
    elif dataset1_option == "Preprocessing":
        preprocessing2()
    elif dataset1_option == "Visualisation":
        visualisation()   

def preprocessing2():
    if "dataset2" not in st.session_state:
        st.session_state.dataset2= update_data("Dataset2.csv")
        st.session_state.dataset2=ZCTA_TO_ZONE(st.session_state.dataset2)
        df=st.session_state.dataset2
    else:
        st.session_state.dataset2= update_data(st.session_state.dataset2)
        df=st.session_state.dataset2
    
    st.title("Preprocessing and Visualisation data")
    st.header("Handling missing values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode"], key="select_columns_2")
    if selected_method:
            st.session_state.dataset2=replace_missing(df,selected_method[0])
            df=st.session_state.dataset2
            st.subheader("Display the Data without missing values:  ")
            st.write(df)
            # Display a table with a custom header for null values
            st.subheader("Null Values Check:")
            st.table(pd.DataFrame({ "Null Values": df.isnull().sum()}))
    st.header("Handling outliers values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode","IQR","Delete"], key="select_columns_3")
    if selected_method:
            st.session_state.dataset2=replace_outliers(df,selected_method[0].lower())
            df=st.session_state.dataset2
            st.subheader("Display the Data without outliers values:  ")
            st.write(df)
            # Display a table with a custom header for null values
            st.subheader("Outliers Check after:")
            count=count_outliers(df)
            st.table(count)
    
    col1,col3=st.columns([3,1])
    with col1:
        st.subheader("Restore the Original Data")
    with col3:
        Restore2=st.button("Restore")
        if Restore2:
            st.session_state.dataset2 = update_data("Dataset2.csv")
            st.session_state.dataset2=ZCTA_TO_ZONE(st.session_state.dataset2)
            df=st.session_state.dataset2
   
def visualisation():
    if "dataset2" not in st.session_state:
        st.session_state.dataset2= update_data("Dataset2.csv")
        st.session_state.dataset2=ZCTA_TO_ZONE(st.session_state.dataset2)
        df=st.session_state.dataset2
    else:
        st.session_state.dataset2= update_data(st.session_state.dataset2)
        df=st.session_state.dataset2
    st.header("Data Visualisation")
    st.subheader("Distribution of the total number of confirmed cases and positive tests by area")
    plot=plot_distribution_by_zone(df)
    if plot:
        st.pyplot(plot)
    
    animated=animated_Plot(df)
    if animated:
        st.write(animated)
    st.subheader("Weekly, monthly and annual evolution of COVID-19 tests, positive tests and the number of cases for a chosen area")
    selected_columns4 = st.multiselect("Select the plot type:", ["Weekly","Monthly","Yearly"], key="select_columns_5")
    selected_columns5 = st.multiselect("Select the zone:", df['zcta'].unique(), key="select_columns_4")
    print(df['zcta'].unique())
    if st.button("plot"):
       if selected_columns4 and selected_columns5:
            # animated_plot(df,selected_columns5[0],selected_columns4[0].lower())
            plot=line(df,selected_columns5[0],selected_columns4[0].lower())
            if plot:
                st.pyplot(plot)
       else: 
           st.warning("Please select at least one column.")
    st.subheader("Distribution of positive COVID-19 cases by area and by year")
    plot=plot_positive_cases_distribution(df)
    if plot:
            st.pyplot(plot)
    st.subheader("Graphical analysis of the relationship between population and number of COVID-19 tests performed")
    plot=plot_population_test_relation(df)
    if plot:
            st.pyplot(plot)
    st.subheader("Top 5 Zones Most Impacted by COVID-19 ")
    selected_columns6 = st.multiselect("Select by:", ["Case count","Positive tests"], key="select_columns_6")
    if st.button("plot",key="2"):
        if selected_columns6:
            with st.spinner("Waiting..."):
                plot=plot_top_zones_impacted(df,selected_columns6[0].lower())
                if plot:
                        st.pyplot(plot)
                else: 
                    st.warning("Please select a column.")
        else:
            st.warning("Please select a column.")
            
    st.subheader(" Le rapport entre les cas confirmés les tests effectués et positif temps pour chaque zone:")
    if st.button("plot",key="42"):
        animated2=animated_Plot2(df)
        if animated2:
            st.write(animated2)  

def Manipultion2():
    
    if "dataset2" not in st.session_state:
        st.session_state.dataset2= update_data("Dataset2.csv")
        st.session_state.dataset2=ZCTA_TO_ZONE(st.session_state.dataset2)
        df=st.session_state.dataset2
    else:
        st.session_state.dataset2= update_data(st.session_state.dataset2)
        df=st.session_state.dataset2
    st.title("Manipulation des données Dataset2")
    
   
    st.subheader("The head of dataset:")
    st.write(df.head())
    
    st.subheader("The tail of dataset:")
    st.write(df.tail())

    st.subheader("The information of dataset:")
    buffer = StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    info_df = parse_info(info_str)

    
    # Display the DataFrame in Streamlit
    st.table(info_df)
    st.subheader("The shape of dataset")
    shape=df.shape
    st.success(f"The number of lines : {shape[0]}.")
    st.success(f"The number of columns : {shape[1]}.")

    st.subheader("Description of dataset:")
    st.write(df.describe())

    st.subheader("Count of Zeros for Each Column")
    st.table(pd.DataFrame(list(isnull(df).items()), columns=['Attributes', 'Null Count']))

def dataset4_options():
    st.sidebar.subheader("Application Options")
    dataset1_option = st.sidebar.selectbox("Choose an option", ["-", "Classification", "Clustering","Evaluation des Models"])

    if dataset1_option == "-":
        st.header("Apprentisage Automatique")
    elif dataset1_option == "Classification":
        Classification()
    elif dataset1_option == "Clustering":
        Clustering()
    elif dataset1_option == "Evaluation des Models":
        evaluation_models()
      
            
def evaluation_models():
    if "models" not in st.session_state:
        st.warning("Train your Models First")
        return 0
    if "classification" not in st.session_state:
        st.session_state.classification=pd.DataFrame(columns=["Algo",'EXACTITUDE','SPÉCIFICITÉ', 'PRÉCISION', 'RAPPEL', 'F-SCORE',"Learning Time"])
    if "clustering" not in st.session_state:
        st.session_state.clustering=pd.DataFrame(columns=['Algo', 'Adjusted Rand Score', 'Silhouette Score',"Davies Bouldin Score","Calinski Harabasz Score","Learning Time"])
    
    st.session_state.classification=pd.DataFrame(columns=["Algo",'EXACTITUDE','SPÉCIFICITÉ', 'PRÉCISION', 'RAPPEL', 'F-SCORE',"Learning Time"])
    st.session_state.clustering=pd.DataFrame(columns=['Algo', 'Adjusted Rand Score', 'Silhouette Score',"Davies Bouldin Score","Calinski Harabasz Score","Learning Time"])   
    st.header("Evaluation et comparaison des differantes Models utiliser")
    i=0
    j=0
    for model in st.session_state.models:
        if model[0] in ["Knn","Decision Tree","RandomForest"]:
            df=performance(model[4],model[1],model[2],1)
            st.session_state.classification = st.session_state.classification._append(df, ignore_index=True)
            st.session_state.classification.loc[i,"Algo"]=model[0]
            st.session_state.classification.loc[i,"Learning Time"]=model[3]
            i+=1
        else:
            df=modele_Evaluation(model[0],model[2],model[3],model[4],model[5])
            st.session_state.clustering = st.session_state.clustering._append(df, ignore_index=True)
            st.session_state.clustering.loc[j,"Learning Time"]=model[1]
            j+=1
            
    
    st.subheader("Classification Models :")
    st.table(st.session_state.classification)
    st.subheader("Clustering Models :")
    st.table(st.session_state.clustering)
def Classification():
    if "models" not in st.session_state:
        st.session_state.models=[]
    if "dataset1" not in st.session_state:
        st.session_state.dataset1,shape, head, info, describe, null = info_data("Dataset1.csv")
        st.session_state.dataset1=replace_missing(st.session_state.dataset1,"Mean")
        data=st.session_state.dataset1
    else:
        st.session_state.dataset1,shape, head, info, describe, null= info_data(st.session_state.dataset1)
        st.session_state.dataset1 = st.session_state.dataset1.reset_index(drop=True)
        data=st.session_state.dataset1
    if "model_trained" not in st.session_state:
        st.session_state.model_trained = False
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    st.title("Visualisation des Donnees")
    data_description2(X,Y)
    st.title("Exploitation des differents algorithems de Classification")
    st.subheader("Separation des Donnes")
    test_size=float(st.number_input("Definir la taille des Donner du Test :", value=0.2, step=0.1))
    st.subheader("Choisire un Algorithme")
    select_algo = st.multiselect("Select Algorithme:",["Knn","Decision Tree","RandomForest"])
    if select_algo==["Knn"]:
        train,test=train_test_split(data,shuffle=False,test_size=test_size)
        X_train = train.iloc[:, :-1]
        Y_train= train.iloc[:, -1]
        X_test = test.iloc[:, :-1]
        Y_test = test.iloc[:, -1]
        user_input2 = st.number_input("Entrez le Nombre des Neighbours", value=3, step=1)
        distance_options = ["euclidean", "manhattan", "minkowski", "cosine", "hamming"]
        default_distance = "euclidean"
        selected_distance = st.selectbox("Select The Distance measure:", distance_options, index=distance_options.index(default_distance))
        classifier= KNeighborsClassifier(k=user_input2, distance_metric=selected_distance)
    elif select_algo==["Decision Tree"]:
        gains = ["entropy","gini","C4.5"]
        gain = "entropy"
        gain = st.selectbox("Select Algorithme:", gains, index=gains.index(gain))
        user_input2 = st.number_input("Entrez le Min_Simples", value=3, step=1)
        user_input3 = st.number_input("Entrez Le Max-Depth", value=6, step=1)
        classifier = DecisionTreeClassifier(method=gain,min_samples_split=user_input2, max_depth=user_input3)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=41)   
    elif select_algo==["RandomForest"]:
        user_input = st.number_input("Donner le Nombre des Arbres", value=5, step=1)
        user_input2 = st.number_input("Entrez le Min_Simples", value=3, step=1)
        user_input3 = st.number_input("Entrez Le Max-Depth", value=6, step=1)
        classifier = RandomForestClassifier(n_trees=user_input, min_samples_split=user_input2, max_depth=user_input3)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=41) 
       
    fit=st.button("Fit")
    if fit:
        with st.spinner("Learning..."):
            start_time = time.time()
            classifier.fit(X_train,Y_train)
            end_time = time.time()
            execution_time = end_time - start_time
            st.session_state.Trained_model=classifier
            st.session_state.models.append([select_algo[0],X_test,Y_test,round(execution_time,2),classifier])
            st.success(select_algo[0]+" Successfully trained in "+str(round(execution_time,2))+" seconds.")
            performance(st.session_state.Trained_model,X_test,Y_test)
            if select_algo==["Knn"]:
                Y_pred = classifier.predict(data.iloc[:, :-1])
                performance3(data.iloc[:, :-1].values,data.iloc[:, -1].values.reshape(-1,1),Y_pred)
        st.session_state.model_trained = True
    
    user_inputs=[]
    st.header("Insertion d'une instance et Evaluation :")
    for d in data.columns[:-1]:
        user_input = st.number_input(f"Entrez la valeur de {d}", value=float(data[d].sample().values[0]), step=0.4)
        user_inputs.append(user_input)  
    
    user_inputs=np.array(user_inputs)
    user_inputs = user_inputs.reshape((1, -1))
    predict = st.button("Predict")
    if st.session_state.model_trained and predict:
        Result = st.session_state.Trained_model.predict(user_inputs)
        st.success("Le Resultat de la prediction est "+str(Result))

def Clustering():
    if "models" not in st.session_state:
        st.session_state.models=[]
    if "dataset1" not in st.session_state:
        st.session_state.dataset1,shape, head, info, describe, null = info_data("Dataset1.csv")
        st.session_state.dataset1=replace_missing(st.session_state.dataset1,"Mean")
        data=st.session_state.dataset1
    else:
        st.session_state.dataset1,shape, head, info, describe, null= info_data(st.session_state.dataset1)
        st.session_state.dataset1 = st.session_state.dataset1.reset_index(drop=True)
        data=st.session_state.dataset1
    st.write(data)
    if "model_trained2" not in st.session_state:
        st.session_state.model_trained2 = False
    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values.reshape(-1,1)
    st.title("Visualisation des Donnees")
    data_description2(X,Y)
    st.title("Exploitation des differents algorithems de Clustering")
    st.subheader("Choisire un Algorithme")
    select_algo = st.multiselect("Select Algorithme:",["K-Means","DBSCAN"])
    if select_algo==["K-Means"]:
        numeric_data = data.iloc[:, :-1].values.tolist()
        numeric_data = [np.array(instance) for instance in numeric_data]
        st.session_state.X = numeric_data
        # scaler = StandardScaler()
        # data_scaled = scaler.fit_transform(data)
        # pca = PCA(n_components=7)
        # data_pca = pca.fit_transform(data_scaled)
        # pca_df = pd.DataFrame(data=data_pca, columns=[f'PC{j+1}' for j in range(7)])
        # st.session_state.X=pca_df
        # st.session_state.X = st.session_state.X.reset_index(drop=True)
        # #lorsque utilise pas pca change ici avec iloc
        # numeric_data = st.session_state.X.values.tolist()
        # numeric_data = [np.array(instance) for instance in numeric_data]
        # st.session_state.X = numeric_data
        distance_options = ["euclidean", "manhattan", "minkowski", "cosine", "hamming"]
        default_distance = "euclidean"
        selected_distance = st.selectbox("Select The Distance measure:", distance_options, index=distance_options.index(default_distance))
        user_input2 = st.number_input("Entrez le Nombre de K", value=3, step=1)
        clustering=KMeans(k=user_input2,distance_metric=selected_distance)   
    elif select_algo==["DBSCAN"]:
        # scaler = StandardScaler()
        # data_scaled = scaler.fit_transform(data)
        # pca = PCA(n_components=7)
        # data_pca = pca.fit_transform(data_scaled)
        # pca_df = pd.DataFrame(data=data_pca, columns=[f'PC{j+1}' for j in range(7)])
        # numeric_data = pca_df.values
        # st.session_state.X = numeric_data
        numeric_data = data.iloc[:, :-1].values
        st.session_state.X = numeric_data
        user_input2 = st.number_input("Entrez le Min_Simples", value=3, step=1)
        user_input3 = st.number_input("Entrez Le Epsilon", value=0.5, step=0.1)
        clustering=DBSCAN(eps=user_input3,min_samples=user_input2)
        from sklearn.metrics import silhouette_score   

    fit=st.button("Fit")
    if fit:
        with st.spinner("Learning..."):
            if select_algo==["K-Means"]:
                start_time = time.time()
                clustering.fit(st.session_state.X)
                end_time = time.time()
                execution_time = end_time - start_time
                st.session_state.Trained_model=clustering
                labels = st.session_state.Trained_model.predict(st.session_state.X)
                silhouette_avg=st.session_state.Trained_model.calculate_silhouette_score(st.session_state.X)
                # st.success("Inter cluster et Iner Cluster pour K-Means : "+str(st.session_state.Trained_model.evaluate_clustering()))
                st.success(select_algo[0]+" Successfully trained in "+str(round(execution_time,2))+" seconds.")
            else:
                start_time = time.time()
                labels=clustering.fit_predict(st.session_state.X)
                labels=np.array(labels)
                end_time = time.time()
                execution_time = end_time - start_time
                st.session_state.Trained_model=clustering
                silhouette_avg = silhouette_score(data.iloc[:, :-1], labels)
                # print(labels)
                # resultat = np.where(labels != -1, labels - 1, -1)
                # print(resultat)
                # ari_score = adjusted_rand_score(data.iloc[:, -1], labels)
                # st.success("Silhouette Score for DBSCAN: "+str(silhouette_avg))
                st.success(select_algo[0]+" Successfully trained in "+str(round(execution_time,2))+" seconds.")
        
            #mesure pour comparer les clas reel avec leur cluster
            ari_score= adjusted_rand_index(data.iloc[:, -1], labels,select_algo[0])
            db_score = davies_bouldin_index(data.iloc[:, :-1], labels,select_algo[0])
            ch_score = calinski_harabasz_index(data.iloc[:, :-1].values, labels)
            st.session_state.models.append([select_algo[0],round(execution_time,2),ari_score,silhouette_avg,db_score,ch_score])
            st.table(modele_Evaluation(select_algo[0],ari_score,silhouette_avg,db_score,ch_score))
            # st.session_state.models.append([select_algo[0],round(execution_time,2),ari_score,silhouette_avg,db_score,ch_score,selected_distance])
            # st.table(modele_Evaluation(select_algo[0],ari_score,silhouette_avg,db_score,ch_score,selected_distance))
            performance2(data.iloc[:, :-1].values,data.iloc[:, -1].values.reshape(-1,1),labels)
            
             
        st.session_state.model_trained2 = True
    user_inputs=[]
    st.header("Insertion d'une instance et Evaluation :")
    for d in data.columns[:-1]:
        user_input = st.number_input(f"Entrez la valeur de {d}", value=float(data[d].sample().values[0]), step=0.4)
        user_inputs.append(user_input)
    predict = st.button("Predict")
    if st.session_state.model_trained2 and predict:
        if select_algo==["K-Means"]:
            labels = st.session_state.Trained_model.predict2(user_inputs)
            st.success("Le Resultat de la prediction est : Class : "+str(labels))
        else:
            X_array = np.array(user_inputs)
            X_array=X_array.reshape((1, 13))
            new_array = np.vstack((st.session_state.X, X_array))
            labels=clustering.fit(new_array)
            if labels[-1]==-1:
                st.success("L'element inserer est un outlier")
            else:
                st.success("Le Resultat de la prediction est :Class "+str(labels[-1]))
                     
def main():
    col1,col2=st.columns([1,2])
    with col1:
        st.title('Statistical Summary App')
    with col2:
        st_lottie(
        lottie_coding6,
        speed=1,
        reverse=False,
        loop=True,
        quality="hight", # medium ; high
        height=200,
        width=600,
        key="animation",
        )
    with st.sidebar:
        st_lottie(
        lottie_coding3,
        speed=1,
        reverse=False,
        loop=True,
        quality="hight", # medium ; high
        height=None,
        width=None,
        key="animation3",
        )
    nav_option = st.sidebar.selectbox("DataSet", ["Home","Dataset1","Dataset2", "Dataset3","Apprentisage"])
    # st.sidebar.title("Options Menus")
    # choix=st.sidebar.radio("Sélectionnez une option", ["Option 1", "Option 2", "Option 3"])
    if nav_option == "Home":
        st.title("Home Page")
        st.write("Welcome to the home page!")
    elif nav_option == "Dataset1":
        dataset1_options()
    elif nav_option == "Dataset2":
        dataset2_options()
    elif nav_option == "Dataset3":
        dataset3_options()
    elif nav_option=="Apprentisage":
        dataset4_options()
    st_lottie(
        lottie_coding,
        speed=1,
        reverse=False,
        loop=True,
        quality="hight", # medium ; high
        height=300,
        width=None,
        key="animation2",
        )
if __name__ == '__main__':
    main()
