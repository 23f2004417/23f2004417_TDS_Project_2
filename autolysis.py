# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "uuid",
#   "seaborn",
#   "matplotlib",
#   "requests",
#   "statsmodels"
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import uuid
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json
import requests
import base64

#Checking the Existence of AIPROXY_TOKEN
try:
    AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]
    print(f"AIPROXY_TOKEN: {AIPROXY_TOKEN}")
except KeyError:
    print("AIPROXY_TOKEN is not set.")
    sys.exit(1)

url="https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
model="gpt-4o-mini"
headers={"Content-Type":"application/json", "Authorization":f"Bearer {AIPROXY_TOKEN}"}

#Use LLM to extract the Meta Data of the Fields in the .csv file.
def extractMetadata(fileName):
    encodingsList = ['utf-8', 'latin1', 'utf-16', 'utf-32', 'iso-8859-1', 'cp1252']

    for encoding in encodingsList:
        try:
            with open(file=fileName, mode='r', encoding=encoding) as f:
                sampleData = ''.join([f.readline() for i in range(10)])
            break  # Exit the loop if successful
        except (UnicodeDecodeError, FileNotFoundError) as e:
            print(f"Failed to read with encoding {encoding}: {e}")
            continue

    prompt = f"You are given with a sample of a dataset {fileName}. The first line is the header. The fields may have uncleansed values, Ignore such cells. Based on the majority of the column classify the fields as Integer, Float, Double, Boolean, Strings, Date, Datetime etc."
    functions = [
        {
            "name": "get_field_datatype",
            "description" : f"Identify field names and their data types from the sample dataset {fileName}",
            "parameters" : {
                "type": "object",
                "properties": {
                    "column_metadata": {
                        "type": "array",
                        "description": "Metadata for each column",
                        "items": {
                            "type": "object",
                            "properties": {
                                "column_name": {
                                    "type": "string",
                                    "description": "The Name of the Column"
                                },
                                "column_type": {
                                    "type": "string",
                                    "description": "The Data Type of the Column Integer, Float, Double, Boolean, Strings, Date, Datetime etc."
                                }
                            },
                            "required": ["column_name", "column_type"]
                        },
                        "minItems": 1
                    }
                },
                "required": ["column_metadata"]
            }
        }
    ]

    json_data = {
        "model":model,
        "messages":[
            {"role":"system","content":prompt},
            {"role":"user","content":sampleData}
        ],
        "functions":functions,
        "function_call":{"name":"get_field_datatype"}
    }

    result = requests.post(url=url, headers=headers, json=json_data)
    metadata = json.loads(result.json()['choices'][0]['message']['function_call']['arguments'])['column_metadata']
    return metadata

# Function to clean columns based on their type
def cleanData(column, column_type):
    if column_type == 'Integer':
        return pd.to_numeric(column, errors='coerce')  # Convert to numeric, coercing errors to NaN
    elif column_type == 'String':
        return column.astype(str).str.strip().fillna('')  # Strip whitespace and fill NaNs with empty strings
    elif column_type == 'Float':
        return pd.to_numeric(column, errors='coerce')  # Convert to float, coercing errors to NaN
    elif column_type == 'Boolean':
        return column.astype(str).str.lower().replace({'true': True, 'false': False, '1': True, '0': False, '': np.nan}).astype('boolean')  # Convert to boolean
    elif column_type == 'Date' or column_type == 'Datetime':
        return pd.to_datetime(column, errors='coerce')  # Convert to datetime, coercing errors to NaT
    else:
        return column  # Return the column unchanged if type is unknown

#Use LLM to get 3 best Analysis that can be performed on the given dataset.
def suggestAnalysis(fileName, columnMetadata):
    # Prepare header and data rows
    header = [item['column_name'] for item in columnMetadata]
    types = [item['column_type'] for item in columnMetadata]

    # Create CSV format as a string
    header_types_string = ','.join(header) + '\n' + ','.join(types)
    print(header_types_string)

    prompt = (
    f"Given the field names and their types for {fileName}"
    "Suggest 3 best analysis that can be performed on them. (Outlier Detection/Correlation Analysis/Regression Analysis/Time Series Analysis). Perform Regression and Time Series Analysis on only 2 fields"
    "Also provide the fields required for the analysis as a list"
    "For Regression Analysis let the first field name be the dependent variable."
    "For Time Series Analysis let the first field name be the date field name."
    )
    functions = [
        {
            "name": "suggest_analysis",
            "description" : "Use field Names and their types to guess the best possible analysis required for the given dataset (Outlier Detection/Correlation Analysis/Regression Analysis), Perform Regression and Time Series Analysis on only 2 fields, For Regression Analysis let the first field name be the dependent variable, For Time Series Analysis let the first field name be the date field name.",
            "parameters" : {
                "type": "object",
                "properties": {
                    "three_best_analysis": {
                        "type": "array",
                        "description": "3 Best Possible analysis choose only (Outlier Detection/Correlation Analysis/Regression Analysis/Time Series Analysis), Perform Regression and Time Series Analysis on only 2 fields",
                        "items": {
                            "type": "object",
                            "properties": {
                                "analysis_name": {
                                    "type": "string",
                                    "description": "Name of the Analysis"
                                },
                                "field_list": {
                                    "type": "string",
                                    "description": "The list of all field names required for the analysis in comma seperated, For Regression Analysis let the first field name be the dependent variable, For Time Series Analysis let the first field name be the date field name."
                                }
                            },
                            "required": ["analysis_name", "field_list"]
                        },
                        "minItems": 1
                    }
                },
                "required": ["three_best_analysis"]
            }
        }
    ]

    json_data = {
        "model":model,
        "messages":[
            {"role":"system","content":prompt},
            {"role":"user","content":header_types_string}
        ],
        "functions":functions,
        "function_call":{"name":"suggest_analysis"}
    }

    result = requests.post(url=url, headers=headers, json=json_data)
    three_best_analysis = json.loads(result.json()['choices'][0]['message']['function_call']['arguments'])['three_best_analysis']
    return three_best_analysis

#Function to Detect and Plot Outliers
def detect_outliers_boxplot(dataframe, field_names, save_directory):
    outliers = {}
    
    # Prepare data for box plot
    data_to_plot = [dataframe[field] for field in field_names]
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.boxplot(data_to_plot, vert=True, labels=field_names)
    plt.title('Box Plot for Multiple Fields with Outlier Detection')
    plt.ylabel('Values')
    
    # Initialize a variable to check if any outliers are found
    outlier_found = False
    
    # Identify outliers
    for i, field in enumerate(field_names):
        # Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR
        Q1 = dataframe[field].quantile(0.25)
        Q3 = dataframe[field].quantile(0.75)
        IQR = Q3 - Q1
        
        # Determine outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Identify outliers
        outlier_indices = dataframe[(dataframe[field] < lower_bound) | (dataframe[field] > upper_bound)].index
        outliers[field] = outlier_indices
        
        # Debug: Print outlier indices
        print(f"Outliers for {field}: {outlier_indices}")
        
        # Mark outliers on the box plot
        for outlier_index in outlier_indices:
            # Use loc instead of iloc
            if outlier_index in dataframe.index:
                plt.scatter(i + 1, dataframe[field].loc[outlier_index], color='red', label='Outliers' if not outlier_found else "")
                outlier_found = True  # Set to True if at least one outlier is found
            
    # Add a dummy artist if no outliers were found to prevent legend warning
    if not outlier_found:
        plt.scatter([], [], color='red', label='Outliers')  # Dummy artist
    
    plt.legend()
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    
    # Generate a unique filename
    output_filename = f"outliers_boxplot_{uuid.uuid4()}.png"
    output_path = os.path.join(save_directory, output_filename)
    
    # Save the plot
    plt.savefig(output_path)
    plt.close()  # Close the plot to free memory
    
    print(f"Outlier Box Plot saved as {output_path}")

#Function to perform Correlation Analysis and Plot Heat Maps
def correlation_analysis(df, field_names, output_directory):
    # Load the dataset
    data = df

    # Select the specified fields
    selected_data = data[field_names]

    # Compute the correlation matrix
    correlation_matrix = selected_data.corr()

    # Set the size of the plot
    plt.figure(figsize=(10, 8))

    # Create a heatmap to visualize the correlation matrix
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})

    # Set the title of the heatmap
    plt.title('Correlation Heatmap', fontsize=16)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate a unique filename using UUID
    output_filename = f"correlation_analysis_{uuid.uuid4()}.png"
    output_path = os.path.join(output_directory, output_filename)

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    print(f"Correlation heatmap saved as {output_path}")

#Function to perform Regression Analysis and Plot graphs
def regression_analysis(df, field_names, output_directory):
    # Load the dataset
    data = df
    dependent_var = field_names[0]
    independent_vars = field_names[1:]
    # Prepare the data for regression
    X = data[independent_vars]  # Independent variables
    y = data[dependent_var]      # Dependent variable
    X = sm.add_constant(X)       # Add a constant term to the predictor

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Print the summary of the regression results
    print(model.summary())

    # Set up the plot for the first independent variable
    plt.figure(figsize=(10, 6))
    plt.scatter(data[independent_vars[0]], y, color='blue', label='Data points')
    
    # Predict values for the regression line
    predictions = model.predict(X)
    plt.plot(data[independent_vars[0]], predictions, color='red', label='Regression line')

    # Labeling the plot
    plt.title('Regression Analysis', fontsize=16)
    plt.xlabel(independent_vars[0], fontsize=14)
    plt.ylabel(dependent_var, fontsize=14)
    plt.legend()

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate a unique filename using UUID
    output_filename = f"regression_analysis_{uuid.uuid4()}.png"
    output_path = os.path.join(output_directory, output_filename)

    # Save the figure
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    print(f"Regression plot saved as {output_path}")

#Function to perform Time Series Analysis and Plot graphs
def time_series_analysis(df, field_names, output_directory):
    # Extract the datetime and value fields
    datetime_field = field_names[0]
    value_field = field_names[1]

    # Ensure the datetime field is in datetime format
    df[datetime_field] = pd.to_datetime(df[datetime_field])

    # Sort the DataFrame by the datetime field
    df = df.sort_values(by=datetime_field)

    # Check if the value field is numeric, and convert if necessary
    df[value_field] = pd.to_numeric(df[value_field], errors='coerce')

    # Drop any rows with NaN values in the datetime or value fields
    df = df.dropna(subset=[datetime_field, value_field])

    # Generate the time series plot
    plt.figure(figsize=(12, 6))
    plt.plot(df[datetime_field], df[value_field], marker='o', linestyle='-')
    plt.title('Time Series Analysis')
    plt.xlabel(datetime_field)
    plt.ylabel(value_field)
    plt.grid()
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate a unique filename using UUID for the time series plot
    time_series_plot_filename = f"time_series_plot_{uuid.uuid4()}.png"
    time_series_plot_path = os.path.join(output_directory, time_series_plot_filename)
    
    # Save the time series plot
    plt.savefig(time_series_plot_path, bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    print(f"Time series plot saved as {time_series_plot_path}")

#Use LLM to narrate story based on the given details and images. Also, store the content into the README.md file    
def narrateStory(fileName, columnMetadata):
    # Prepare header and data rows
    header = [item['column_name'] for item in columnMetadata]
    types = [item['column_type'] for item in columnMetadata]

    # Create CSV format as a string
    header_types_string = ','.join(header) + '\n' + ','.join(types)
    print(header_types_string)

    # Get the current working directory
    current_directory = os.getcwd()
    # Construct the full path to the folder
    folder_path = os.path.join(current_directory, fileName[:-4])
    # Check if the folder exists
    base64_image = []
    if os.path.isdir(folder_path):
        # List all files in the folder
        files = os.listdir(folder_path)
        # Filter out only .png files
        png_files = [os.path.join(folder_path, f) for f in files if f.endswith('.png') and os.path.isfile(os.path.join(folder_path, f))]
        for each in png_files:
            encoded_image = encode_image(each)
            base64_image.append(encoded_image)

    json_data = {
        "model":model,
        "messages":[
            {
                "role": "user",
                "content": [
                {
                    "type": "text",
                    "text": f"Describe the data {fileName} briefly, The analysis carried out and insights that was discovered using the provided plots and metadata {header_types_string}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "detail": "low",
                    "url": f"data:image/png;base64,{base64_image[0]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "detail": "low",
                    "url": f"data:image/png;base64,{base64_image[1]}"
                    }
                },
                {
                    "type": "image_url",
                    "image_url": {
                    "detail": "low",
                    "url": f"data:image/png;base64,{base64_image[1]}"
                    }
                }
                ]
            }
            ]
    }

    result = requests.post(url=url, headers=headers, json=json_data)
    content = result.json()['choices'][0]['message']['content']
    print(result.text)
    return content

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <filename>")
        sys.exit(1)

    fileName = sys.argv[1]

    encodingsList = ['utf-8', 'latin1', 'utf-16', 'utf-32', 'iso-8859-1', 'cp1252']
    for encoding in encodingsList:
        try:
            df = pd.read_csv(fileName, encoding=encoding)
            break  # Exit the loop if successful
        except UnicodeDecodeError:
            print(f"Failed to read with encoding {encoding}. Trying next encoding...")
        except FileNotFoundError:
            print(f"The file {fileName} was not found.")
            break  # Exit if the file doesn't exist
        except Exception as e:
            print(f"An error occurred while reading the file: {e}")
            break  # Exit on any other unexpected error
    
    #Get Column Meta Data
    columnMetadata = extractMetadata(fileName)
    print(columnMetadata)
    #Cleaning the file
    for meta in columnMetadata:
        column_name = meta['column_name']
        column_type = meta['column_type']
        if column_name in df.columns:
            df[column_name] = cleanData(df[column_name], column_type)
    #Dropping Nans
    df.dropna(inplace=True)
    print(df)
    #Get Best Analysis
    bestAnalysis = suggestAnalysis(fileName, columnMetadata)
    print(bestAnalysis)
    #Perform Analysis and Plot them
    directoryName = fileName[:-4]
    for each in bestAnalysis:
        if each["analysis_name"] == 'Outlier Detection':
            detect_outliers_boxplot(df, each["field_list"].split(','), directoryName)
        elif each["analysis_name"] == 'Correlation Analysis':
            correlation_analysis(df, each["field_list"].split(','), directoryName)
        elif each["analysis_name"] == 'Regression Analysis':
            regression_analysis(df, each["field_list"].split(','), directoryName)
        elif each["analysis_name"] == 'Time Series Analysis':
            time_series_analysis(df, each["field_list"].split(','), directoryName)
        else:
            pass
    #Narrate Story and Write README.md
    content = narrateStory(fileName, columnMetadata)
    with open(f'{directoryName}/README.md', 'w') as readme_file:
        readme_file.write(content)
    print("README.md file has been created successfully.")
                        
    
    





