import streamlit as st
import pandas as pd
from io import StringIO
from projet1 import info_data,generate_summary_and_plot2,boxplot_with_outliers, plot_histogram,parse_info,isnull,replace_outliers
from projet1 import plot_scatter_with_correlation,heatmap,replace_missing,count_outliers,remove_redundant_columns,min_max,z_score
from projet1 import update_data,plot_distribution_by_zone,line,plot_positive_cases_distribution
from projet1 import plot_population_test_relation,plot_top_zones_impacted

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

def Manipultion():

    st.title("Manipulation des données Dataset1")

    # Assuming info_data is correctly implemented and returns info, describe, and null
    df,shape, head, info, describe, null = info_data("Dataset1.csv")

    
    st.subheader("The head of dataset:")
    st.write(head)

    st.subheader("The information of dataset:")
    
    # Parse the info_str and create a DataFrame
    info_df = parse_info(info)
    
    # Display the DataFrame in Streamlit
    st.table(info_df)
    st.subheader("The shape of dataset")
    st.write(f"The number of lines : {shape[0]}.")
    st.write(f"The number of columns : {shape[1]}.")

    st.subheader("Description of dataset:")
    st.write(describe)

    st.subheader("Count of Zeros for Each Column")
    columns = ["Column","Count Null Values"]

    st.table(null)
    
def Analysis():
    st.title("Analyse des caractéristiques des attributs")
    df = info_data("Dataset1.csv")[0]

    st.subheader("Measurements of central tendency and deduce symmetries")
    all_columns_checkbox = st.checkbox("Analyze All Attributes",key="1")
    selected_columns = st.multiselect("Select Columns:", df.columns, key="select_columns_1")

    if st.button("Generate Summary and Plot"):
        if all_columns_checkbox:
            g = generate_summary_and_plot2(df, columns=None)
            st.write(g)
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
    df = info_data("Dataset1.csv")[0]
    st.header("Handling missing values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode"], key="select_columns_2")
    if selected_method:
            df2=replace_missing(df,selected_method[0])
            st.subheader("Display the Data without missing values:  ")
            st.write(df2)
            # Display a table with a custom header for null values
            st.subheader("Null Values Check:")
            st.table(pd.DataFrame({ "Null Values": df2.isnull().sum()}))
    st.header("Handling outliers values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode","IQR","Delete"], key="select_columns_3")
    if selected_method:
            df3=replace_outliers(df2,selected_method[0].lower())
            st.subheader("Display the Data without outliers values:  ")
            st.write(df3)
            # Display a table with a custom header for null values
            st.subheader("Outliers Check:")
            count=count_outliers(df3)
            st.write(count)
    st.header("Data reduction")
    check = st.checkbox("Data reduction h/v",key="4")
    if check:
            df4=remove_redundant_columns(df3)
            st.write(df4)
            shape=df4.shape
            st.subheader("The shape of dataset after reduction")
            st.write(f"The number of lines : {shape[0]}.")
            st.write(f"The number of columns : {shape[1]}.")
    st.header("Data normalization:")
    selected_method = st.multiselect("Select a method:", ["Min-Max-Scaler","Z-score"], key="select_columns_5")
    if selected_method:
        if selected_method=="Min-Max-Scaler" and check:
                st.write(min_max(df4))
        elif selected_method=="Min-Max-Scaler" and not check:
                st.write(min_max(df3))
        elif selected_method=="Z-score" and check :st.write(z_score(df4))
        else : st.write(z_score(df3))


def Manipultion2():
    df=update_data("Dataset2.csv")
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
    st.write(f"The number of lines : {shape[0]}.")
    st.write(f"The number of columns : {shape[1]}.")

    st.subheader("Description of dataset:")
    st.write(df.describe())

    st.subheader("Count of Zeros for Each Column")
    columns = ["Column","Count Null Values"]
    st.table(pd.DataFrame({ "Null Values": df.isnull().sum()}))

def preprocessing2():
    df=update_data("Dataset2.csv")
    st.title("Preprocessing and Visualisation data")
    st.header("Handling missing values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode"], key="select_columns_2")
    if selected_method:
            df=replace_missing(df,selected_method[0])
            st.subheader("Display the Data without missing values:  ")
            st.write(df)
            # Display a table with a custom header for null values
            st.subheader("Null Values Check:")
            st.table(pd.DataFrame({ "Null Values": df.isnull().sum()}))
    st.header("Handling outliers values")
    selected_method = st.multiselect("Select a method:", ["Mean","Median","Mode","IQR","Delete"], key="select_columns_3")
    if selected_method:
            df=replace_outliers(df,selected_method[0].lower())
            st.subheader("Display the Data without outliers values:  ")
            st.write(df)
            # Display a table with a custom header for null values
            st.subheader("Outliers Check after:")
            count=count_outliers(df)
            st.write(count)
   
def visualisation():
    df=update_data("Dataset2.csv")
    st.header("Data Visualisation")
    st.subheader("Distribution of the total number of confirmed cases and positive tests by area")
    plot=plot_distribution_by_zone(df)
    if plot:
            st.pyplot(plot)
    st.subheader("Weekly, monthly and annual evolution of COVID-19 tests, positive tests and the number of cases for a chosen area")
    selected_columns4 = st.multiselect("Select the plot type:", ["Weekly","Monthly","Yearly"], key="select_columns_5")
    selected_columns5 = st.multiselect("Select the zone:", df['zcta'].unique(), key="select_columns_4")
    if st.button("plot"):
       if selected_columns4 and selected_columns5:
                    plot=line(df,selected_columns5[0],selected_columns4[0].lower())
                    if plot:
                        st.pyplot(plot)
       else: st.warning("Please select at least one column.")
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
        with st.spinner("Waiting..."):
            plot=plot_top_zones_impacted(df,selected_columns6[0].lower())
            if plot:
                    st.pyplot(plot)
            else: st.warning("Please select at least one column.")  
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


# Function to display information for Dataset3
def show_dataset3_info():
    st.title("Dataset3 Info")
    st.write("Information about Dataset3 goes here.")

def main():
    st.title('Statistical Summary App')
    nav_option = st.sidebar.selectbox("DataSet", ["Home", "Dataset1", "Dataset2", "Dataset3"])

    # Display content based on the selected option
    if nav_option == "Home":
        st.title("Home Page")
        st.write("Welcome to the home page!")
    elif nav_option == "Dataset1":
        dataset1_options()
    elif nav_option == "Dataset2":
        dataset2_options()
    elif nav_option == "Dataset3":
        show_dataset3_info()

if __name__ == '__main__':
    main()
