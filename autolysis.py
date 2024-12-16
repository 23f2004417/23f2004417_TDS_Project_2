# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "numpy",
#   "uuid",
#   "seaborn",
#   "matplotlib",
#   "requests",
# ]
# ///

import os
import sys
import pandas as pd
import numpy as np
import uuid
import seaborn as sns
# from sklearn.cluster import KMeans
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

def extractMetadata(fileName):
    try:
        with open(file=fileName,mode='r',encoding='utf-8') as f:
            sampleData = ''.join([f.readline() for i in range(10)])
    except UnicodeDecodeError:
        with open(file=fileName,mode='r',encoding='latin1') as f:
            sampleData = ''.join([f.readline() for i in range(10)])

    prompt = "You are given with a sample of a dataset. The first line is the header. The fields may have uncleansed values, Ignore such cells. Based on the majority of the column classify the fields as Integer, Float, Double, Boolean, Strings, etc."
    functions = [
        {
            "name": "get_field_datatype",
            "description" : "Identify field names and their data types from the dataset",
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
                                    "description": "The Data Type of the Column Integer, Float, Double, Boolean, Strings, etc."
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
    else:
        return column  # Return the column unchanged if type is unknown

def suggestAnalysis(fileName, columnMetadata):
    # Prepare header and data rows
    header = [item['column_name'] for item in columnMetadata]
    types = [item['column_type'] for item in columnMetadata]

    # Create CSV format as a string
    header_types_string = ','.join(header) + '\n' + ','.join(types)
    print(header_types_string)

    prompt = (
    "Given the field names and their types"
    "Suggest 3 best analysis that can be performed on them. (Outlier Detection/Correlation Analysis)"
    "Also provide the fields required for the analysis as a list"
    )
    functions = [
        {
            "name": "suggest_analysis",
            "description" : "Use field Names and their types to guess the best possible analysis required for the given dataset (Outlier Detection/Correlation Analysis)",
            "parameters" : {
                "type": "object",
                "properties": {
                    "three_best_analysis": {
                        "type": "array",
                        "description": "3 Best Possible analysis choose only (Outlier Detection/Correlation Analysis)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "analysis_name": {
                                    "type": "string",
                                    "description": "Name of the Analysis"
                                },
                                "field_list": {
                                    "type": "string",
                                    "description": "The list of all field names required for the analysis in comma seperated"
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
    folder_path = os.path.join(current_directory, fileName)
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
                    "text": f"Describe the data briefly, The analysis carried out and insights that was discovered using the provided plots and metadata {header_types_string}",
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
    print(result)
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
    try:
        df = pd.read_csv(fileName, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, fall back to Latin-1
        df = pd.read_csv(fileName, encoding='latin1')

    columnMetadata = extractMetadata(fileName)
    print(columnMetadata)

    for meta in columnMetadata:
        column_name = meta['column_name']
        column_type = meta['column_type']
        if column_name in df.columns:
            df[column_name] = cleanData(df[column_name], column_type)
    #Dropping Nans
    df.dropna(inplace=True)
    print(df)

    bestAnalysis = suggestAnalysis(fileName, columnMetadata)
    print(bestAnalysis)

    directoryName = fileName[:-4]
    for each in bestAnalysis:
        if each["analysis_name"] == 'Outlier Detection':
            detect_outliers_boxplot(df, each["field_list"].split(','), directoryName)
        elif each["analysis_name"] == 'Correlation Analysis':
            correlation_analysis(df, each["field_list"].split(','), directoryName)
        else:
            pass
    content = narrateStory(directoryName, columnMetadata)
    with open(f'{directoryName}/README.md', 'w') as readme_file:
        readme_file.write(content)
    print("README.md file has been created successfully.")
                        
    
    





