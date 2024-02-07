import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm, lognorm
from math import ceil
import joblib
from sklearn.ensemble import RandomForestRegressor


# Load the CSV file
file_path = 'SteelDesignProccessAgg.pickle'
MODEL_PATH = 'my_random_forest.joblib'
dfMechanicals = pd.read_csv('mechanicals.csv')
df = pd.read_pickle(file_path)

default_columns = [
    'coil_thickness', 'HM_AMOUNT_C', 'HM_AMOUNT_MN', 'HM_AMOUNT_NB', 'HM_AMOUNT_V',
    'HM_AMOUNT_TI', 'HM_AMOUNT_N', 'HM_AMOUNT_SI', 'HM_AMOUNT_AL', 'average_finish_temp',
    'INTERM_TEMP_AVG', 'average_coiler_temp'
]

def generate_data(distribution, mean, std_dev, count):
    if distribution == 'Normal':
        data = np.random.normal(mean, std_dev, count)
    elif distribution == 'Log-Normal':
        # For log-normal distribution, mean and std_dev are the mean and
        # standard deviation of the underlying normal distribution
        # from which exp(value) is drawn.
        mu = np.log(mean**2 / np.sqrt(std_dev**2 + mean**2))
        sigma = np.sqrt(np.log(std_dev**2 / mean**2 + 1))
        data = np.random.lognormal(mu, sigma, count)
    
    if mean > 0:
            data = np.clip(data, a_min=0, a_max=None)

    return data

# def plot_distribution(mean, std_dev, count, distribution):
#     # Generate random data following a normal distribution
#     data = generate_data(distribution, mean, std_dev, count)

#     # Create a histogram of the data
#     fig = go.Figure(data=[go.Histogram(x=data, histnorm='probability')])

#     # Update layout for a better view
#     fig.update_layout(
#         title=f'Normal Distribution with mean={mean}, std_dev={std_dev}',
#         xaxis_title='Value',
#         yaxis_title='Probability',
#         bargap=0.2,
#     )

#     # Show the plot
#     fig.show()


# Make 3 x n grid of dist plots
def makeDistPlotGrid(filtered_columns, dfSteelDesign):
    ncols = 3  # Number of columns in the grid
    nrows = ceil(len(filtered_columns) / ncols)  # Calculate the number of rows needed
    variables = st.session_state.variables
    # Create a subplot figure with a 3-column layout
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=filtered_columns)
    for idx, colName in enumerate(filtered_columns):
        
        row = (idx // ncols) + 1
        col = (idx % ncols) + 1

        var = variables[colName]
        data = generate_data(var['distribution'], var['mean'], var['std_dev'], var['count'])



        fig.add_trace(go.Histogram(x=data, histnorm='probability', showlegend=False),
                    row=row, col=col)
    # Update layout for a better view
    fig.update_layout(
        title_text='Distributions Grid',
        height=900,
        width=900,
        showlegend=False
    )
    st.plotly_chart(fig)

def load_model(path):
    return joblib.load(path)

def predictions_page(model):
    st.title('Model Tensile Strength Predictions')
    st.write(f"# SteelDesign: {st.session_state['steelDesign']}")
    # Displaying Historical Data for Steel Design
    historicalData = dfMechanicals.loc[dfMechanicals['steel_design_description']==st.session_state['steelDesign']]
    if historicalData.shape[0] == 0:
        figHistorical = None

    else:
        histTensile = historicalData['TensileStrength']
        figHistorical = go.Figure(data=[go.Histogram(x=historicalData['TensileStrength'], histnorm='percent')])
        figHistorical.update_layout(title='Historical TS')
        count_hist = histTensile.count()
        mean_hist = np.mean(histTensile)
        std_dev_hist = np.std(histTensile)
        min_hist = np.min(histTensile)
        max_hist = np.max(histTensile)
            
    variables = st.session_state.variables
    dfInput = pd.DataFrame()
    for i in variables:
        var = variables[i]
        x = generate_data(var['distribution'], var['mean'], var['std_dev'], var['count'])
        dfInput[var['name']] = x
    # Assuming you have a way to generate feature data from your variables
    # For demonstration, let's say each variable generates one feature value
    # You should adjust this part to match your actual feature extraction process
    # feature_data = np.array([var['mean'] for var in variables]).reshape(1, -1)
    # Generate predictions
    predictions = model.predict(dfInput)

    # Display a histogram of the predictions
    fig = go.Figure(data=[go.Histogram(x=predictions, histnorm='percent')])
    fig.update_layout(title='Predicted TS')

    # Calculate and display statistics
    mean_pred = np.mean(predictions)
    std_dev_pred = np.std(predictions)
    min_pred = np.min(predictions)
    max_pred = np.max(predictions)

    col1, col2 = st.columns(2)
    # Display the charts in their respective columns
    with col1:
        if figHistorical:
            st.write(f"Numb Historical Test Results: {count_hist:.4f}")
            st.write(f"Mean Historical: {mean_hist:.4f}")
            st.write(f"STDev of Historical: {std_dev_hist:.4f}")
            st.write(f"Minimum Historical: {min_hist:.4f}")
            st.write(f"Maximum Historical: {max_hist:.4f}")
            st.plotly_chart(figHistorical, use_container_width=True)
        else:
            st.write(f"No Historical Data")

    with col2:
        st.write(f'Numb Predictions: {predictions.shape[0]}')
        st.write(f"Mean Predictions: {mean_pred:.4f}")
        st.write(f"STDev Predictions: {std_dev_pred:.4f}")
        st.write(f"Minimum Prediction: {min_pred:.4f}")
        st.write(f"Maximum Prediction: {max_pred:.4f}")
        st.plotly_chart(fig, use_container_width=True)



    

# def steel_design_freq(d):
#     steelDesigns = df['steel_design_description']
#     counts = df['coil_weight']['count']
#     steelDisgnsCounts = []
#     for i in range(len(steelDesigns)):
#         steelDisgnsCounts.append(f'{steelDesigns[i]} {float(counts[i])}')
#     return steelDisgnsCounts


def main_page():
    # Streamlit app
    st.title('Variable Distribtution Selection')

    # Filter DataFrame columns based on the predefined list
    filtered_columns = [col for col in default_columns if col in df.columns]

    #Hard coded like everything else Shiiiiittttttt fuck it
    # options_with_counts = steel_design_freq(df)
    # counts = df['steel_design_description'].value_counts().to_dict()
    # options_with_counts = [f'{category} {count}' for category, count in counts.items()]
    options = list(df['steel_design_description'].unique())
    default_index1 = options.index(st.session_state.get('steelDesign', options[0]))  # Get the index of the saved state
    selected_steelDesign = st.selectbox("Select Steel Design", options, index=default_index1)
    st.session_state['steelDesign'] = selected_steelDesign  # Save the selected option to session state
    # st.write(f"Your selection on Page 1: {st.session_state['input1']}")


    # selected_steelDesign = st.selectbox('Select a variable', df['steel_design_description'].unique())
    # st.session_state['steelDesign'] = selected_steelDesign

    dfSteelDesign = df.loc[df['steel_design_description']==st.session_state['steelDesign']]


    # # Allow the user to select a column
    # selected_column = st.selectbox('Select a variable', df.columns[1:].map(lambda x: x[0]).drop_duplicates())
    if filtered_columns:
        count = st.number_input("Number of Data Points for Each Variable", min_value=1, value=int(dfSteelDesign[filtered_columns[0]]['count']), step=100)
        col1, col2 = st.columns(2)
        st.session_state['count'] = count 
       
        if 'variables' not in st.session_state:
            st.session_state.variables = {} 
        middle = round(len(filtered_columns) / 2)
        with col1:
            for i in range(middle):  # First 5 variables
                with st.expander(f"{filtered_columns[i]}"):
                    
                    if filtered_columns[i] in st.session_state.variables and st.session_state['steelDesign'] == st.session_state.variables[filtered_columns[i]]['steelDesign']:
                            curVar = st.session_state.variables[filtered_columns[i]]
                            curDist = curVar['distribution']
                            curMean = curVar['mean']
                            curSTD = curVar['std_dev']
                    else:
                        curDist = 'Normal'
                        curMean = float(dfSteelDesign[filtered_columns[i]]['mean'])
                        curSTD = float(dfSteelDesign[filtered_columns[i]]['std'])

                    dist_type = st.selectbox(
                        "Distribution Type",
                        ['Normal', 'Log-Normal'],
                        placeholder=curDist,
                        key=f'dist_type_{i}'
                    )
                    mean = st.number_input("Mean", value=curMean,
                                        key=f'mean_{i}',  format="%0.3f")
                    std_dev = st.number_input("Standard Deviation", value=curSTD,
                                            min_value=0.000, key=f'std_dev_{i}',  format="%0.3f")


                    st.session_state.variables[filtered_columns[i]] = {
                        'name': f'{filtered_columns[i]}',
                        'distribution': dist_type,
                        'mean': mean,
                        'std_dev': std_dev,
                        'steelDesign': st.session_state['steelDesign'],
                        'count': count

                                                            }
        with col2:
            for i in range(middle, len(filtered_columns)):  # First 5 variables
                with st.expander(f"{filtered_columns[i]}"):

                    dist_type = st.selectbox(
                        "Distribution Type",
                        ['Normal', 'Log-Normal'],
                        key=f'dist_type_{i}'
                    )
                    mean = st.number_input("Mean", value=float(dfSteelDesign[filtered_columns[i]]['mean']),
                                        key=f'mean_{i}',  format="%0.3f")
                    std_dev = st.number_input("Standard Deviation", value=float(dfSteelDesign[filtered_columns[i]]['std']),
                                            min_value=0.000, key=f'std_dev_{i}',  format="%0.3f")

                    st.session_state.variables[filtered_columns[i]] ={
                        'name': f'{filtered_columns[i]}',
                        'distribution': dist_type,
                        'mean': mean,
                        'std_dev': std_dev,
                        'count': count
                                                            }

        #     mean_std_count_list.append((mean, std, n))
        # # Create a DataFrame for the table
        # df = pd.DataFrame(mean_std_count_list, columns=['Mean', 'StdDev', 'Count'])
        # df.index += 1
        # df.reset_index(inplace=True)
        # df.rename(columns={'index': 'Var'}, inplace=True)
        # df['Var'] = df['Var'].apply(lambda x: f'{filtered_columns[x-1]}')

        # Display the table
        # st.table(df)

        if st.button("Plot Distributions"):
            makeDistPlotGrid(filtered_columns, dfSteelDesign)
        # Show the plot in the Streamlit app
# Load your model (assuming it's a regression model)
model = joblib.load(MODEL_PATH)

# Simple navigation
st.sidebar.title('Navigation')
page = st.sidebar.radio("Select a Page:", ["Main", "Predictions"])

if page == "Main":
    main_page()
elif page == "Predictions":
    # Pass dummy variables and count for demonstration
    # You'll need to adapt how variables and count are shared between pages
    variables = [{'name': 'Var 1', 'distribution': 'Normal', 'mean': 1.0, 'std_dev': 0.5}]
    count = 1000
    predictions_page(model)



# dfSteelDesign = df.loc[df['steel_design_description']==selected_steelDesign]
# # Calculate mean and standard deviation of the selected column
# mean = float(dfSteelDesign[selected_column]['mean'])
# std = float(dfSteelDesign[selected_column]['std'])
# n = int(dfSteelDesign[selected_column]['count'])


# # Display the mean and standard deviation
# st.write(f'Number Data Points: {n}, Mean: {mean:.2f}, Standard Deviation: {std:.3f}')

# # Button to plot the distribution
# if st.button('Plot Normal Distribution'):
#     plot_normal_dist(mean, std, f'Normal Distribution of {selected_column}')
#     # Streamlit does not support plt.show(), so we use st.pyplot() to render the matplotlib plot
#     st.pyplot(plt)


