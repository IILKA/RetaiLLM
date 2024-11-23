'''
treat this file as a flexible template for storing the prompts in a hierarchical structure.

'''
import yaml 

agent_config = yaml.safe_load(open("./config.yaml"))["Agent"]

class PromptsTree: 
    '''
    This class is used to store the prompts in hierarchical structure.

    '''
    def __init__(self, user_input = None):
        '''
        Initialize the prompts tree.

        '''
        self.BackGround = "You are a specialized AI assistant focused on retail analysis. Your primary objective is to assist users by providing insights and solutions tailored to their retail-related needs, guided by the system instructions."
       
    
    def GetDataSubTree(self, user_input, datalist = [], data_info = None):
        '''
        Get the data subtree from the user input.
        this prompt subtree only contains the data-related prompts.
        strucure of the subtree:
        ClassifyingTasks: root node for classifying the tasks
        USER_INPUT
            - sourcefile_or_extract
            - Extract_from_user_input
        PREVIOUS_CONVO_DATA
            - choose_from_previous
                *datalist is required*
        WEB_SCRAPE
            - generate_web_scrape_keys
        DB_QUERY
            - choose_table: choose the most relevant tables based on the descriptions of tables.
                *datalist is required*
        NO_DATA
            - respond_user: provide the response to the user.
        
        '''
        SubPromptTree = {
            "ClassifyingTasks": {
                "prompt": f"""
                    Based on the user's request, identify the most suitable approach to obtain data and return only one tag from the following list:

                    Tags: 
                    - USER_INPUT: The data can be directly extracted from the user's input or the user has specified a given data source enclosed in <data></data>.  
                    - PREVIOUS_CONVO_DATA: It seems that the data required is already available from the previous conversation.
                    - WEB_SCRAPE: The user's requests need additional web scraping to retrieve external textual data, if the user does not state clearly that we should perform numerical analysis, choose this tag.
                    - DB_QUERY: The user;s requrests need additional query to a database to retrieve structured data. 
                    - NO_DATA: No data is required, only provide a response based on the user's input. 
                    - MENU: The user is asking how the assistant can help them. 

                    User Request: {user_input}

                    Output only the single, most appropriate tag from the list above (["USER_INPUT", "PREVIOUS_CONVO_DATA", "WEB_SCRAPE", "DB_QUERY", "NO_DATA", MENU]). 
                """,

                "options": ["USER_INPUT", "PREVIOUS_CONVO_DATA", "WEB_SCRAPE", "DB_QUERY", "NO_DATA", "MENU"],
                "max_length": 10, 
                "model": "Qwen/Qwen2.5-72B-Instruct"
            }, 

            "USER_INPUT": {

                "sourcefile_or_extract": {
                    "prompt": f"""
                        The data can be directly extracted from the user's input or the user has specified a given data source enclosed in <data></data>. 
                        Please determine which condition is most appropriate for the user's request.

                        - EXTRACT_DIRECTLY: The data can be directly extracted from the user's input text.
                        - SPECIFIED_SOURCE: The user has specified a given data source enclosed in <data></data>.

                        User Request: {user_input}
                        return only one tag from the following list (["EXTRACT_DIRECTLY", "SPECIFIED_SOURCE"]).
                    """,
                    "options": ["EXTRACT_DIRECTLY", "SPECIFIED_SOURCE"],
                    "max_length": 20, 
                    "model": "Qwen/Qwen2.5-72B-Instruct"
                }, 

                "Extract_from_user_input": {
                    "prompt": f"""
                        The data can be directly extracted from the user's input text.
                        Your task is to **extract attributes and their corresponding values** from the given user input and **return them in the following JSON format**:

                        {{"attribute1": ["value1", "value2", ...], "attribute2": ["value1", "value2", ...], ...}}

                        **Please follow these guidelines**:

                        - **Attribute Keys**: Use the exact attribute names as they appear in the user input.
                        - **Values**: Collect all values for each attribute into a list, preserving their original order.
                        - **Data Types**: If a value is numeric (integer or float), represent it as a number; otherwise, keep it as a string.
                        - **Consistency**: Ensure all lists are of the same length. If an attribute is missing a value, use `null` (without quotes) to represent it.
                        - **Output Format**: Do not include any additional text, explanations, or formatting. Return **only** the JSON object.

                        **Example**:

                        User Input:
                        I want to analysis the information of some customers.
                        Name: Alice, Bob, Charlie
                        Age: 30, 25, 35
                        City: New York, Los Angeles, Chicago

                        Expected Output: {{"Name": ["Alice", "Bob", "Charlie"], "Age": [30, 25, 35],"City": ["New York", "Los Angeles", "Chicago"]}}

                        Now, based on the following user input, only output the extracted json object:

                        **User Input**:
                        {user_input}
                        """,
                    "options": None,
                    "max_length": 512,
                    "model": "Qwen/Qwen2.5-72B-Instruct"
                }
            },
            #datalist [{"name": "data1", "description": "description1"}, {"name": "data2", "description": "description2"}]}
            "PREVIOUS_CONVO_DATA": {
                "choose_from_previous": {
                    "prompt": f"""
                        The required data is available from the previous conversation. 
                        Select the most appropriate data based on the user input provided below. 
                        The data options, including their names and descriptions, are as follows:
                        {datalist}
                        User input: {user_input}
                        Return only the name of the most appropriate data from this list: {[data["name"] for data in datalist]}.
                    """, 
                    "options": [data["name"] for data in datalist],
                    "max_length": 10,
                    "model": "Qwen/Qwen2.5-72B-Instruct"
                }, 
                "extract_from_previous": {
                    "prompt": f"""
                        Generate a description of the possible data provided by the user based on the user input. 
                        User input: {user_input}
                        Data information: 
                        {data_info}
                        return in the format: {{"data_name": name of data, "description": description of the data}}.
                    """,
                    "options": None,
                    "max_length": 64,
                    "model": "Qwen/Qwen2.5-72B-Instruct"
                }
            },
            
            "WEB_SCRAPE":{
                "generate_web_scrape_keys":{
                    "prompt": f"""
                        Based on the user's request, extract the most relevant keywords for a web scraping task to retrieve external textual data. 
                        The keywords should be concise and directly related to the user's input.

                        Example:
                        User Request: "Find articles about climate change policies in Europe."
                        Output: ["American", "Europe", "articles"]

                        Now process the following request:
                        User Request: {user_input}
                        Provide the keywords as a concise list of strings.
                    """ ,
                    "options": None,
                    "max_length": 50,
                    "model": "Qwen/Qwen2.5-72B-Instruct"
                }

            },
            "DB_QUERY":{
                "choose_table":{
                    "prompt": f"""
                        Based on the user's request, choose the most relavent tables based on the descriptions of tables.
                        The tale options, including their names and descriptions, are as follows:
                        {datalist} 

                        Now process the following request:
                        User input: {user_input}
                        Return only the name of the most appropriate data from this list: {[data["name"] for data in datalist]}.
                    """ ,
                    "options": [data["name"] for data in datalist],
                    "max_length": 50,
                    "model": "Qwen/Qwen2.5-72B-Instruct"
                }

            }, 
            "NO_DATA":{
                "respond_user":{
                    "prompt": f"""
                        Give helpful response based on the user's request. 
                        Your response should be no more than {512} words.
                        User Request: {user_input}
                    """ ,
                    "options": None,
                    "max_length": 512,
                    "model": "Qwen/Qwen2.5-72B-Instruct"
                }

            },
            
            }
        return SubPromptTree
    
    def GetResponseSubTree(self, user_input, summary = "", menu_return_length = agent_config["menu_return_length"], response_return_length = agent_config["response_return_length"]):
        SubPromptTree = {
            "MENU":{
                "prompt": f"""
                    The user is asking how the assistant can help them. 
                    
                    The basic functions of the assistant are as follows: 
                    - The model can judge the user's request and choose the most appropriate tools to getdata and process data. 
                    - Tools: 
                        webscraping: get data from the web and model will review the summarized data and provide insights, also an sentiment analysis is performed on the given keywords. 
                        db query: get data from the database and model will review the summarized data and provide insights, also an sentiment analysis is performed on the given keywords.
                        data analysis tools: 
                            - data visualization: model will provide the visualization of the data. 
                            - data summary: model will provide the summary of the data. 
                            - data prediction: model will provide the prediction of the data.
                        specific tools: 
                            time series analysis: model will provide the time series analysis of the data.
                                ARIMA
                                seasonal decomposition 
                                DeepLearning: transformer
                            correlation analysis: model will provide the correlation analysis of the data.
                                Spearman
                                multiple correlation
                            Regression: regression analysis 
                            Clustering analysis: model will provide the clustering analysis of the data.
                                K-means
                                DBSCAN
                                hierarchical clustering
                            Survival analysis: model will provide the survival analysis of the data.
                                Kaplan-Meier
                                Cox regression
                    And more precise description of the tools, more helpful will this assistant be.
                    Base on the user's request, please explain how the assistant can help the user.
                    Your response should be strictly no more than {menu_return_length} words, please be as concise as possible.
                    user request: {user_input}
                    """ ,
                "options": None,
                "max_length": menu_return_length + 100,
                "model": "Qwen/Qwen2.5-72B-Instruct"
                },
            "response":{
                "prompt": f"""
                    Present the summary generated by the system to user, as if you have done the analysis, and comment on the summary, and please notice that you are speaking to the user rather than the system. 
                    Act if you are presenting the summary to the user. 
                    Summary: {summary}
                    And the user is asking about: {user_input}
                    Your response should be no more than {response_return_length} words.
                """ ,
                "options": None,
                "max_length": response_return_length + 100,
                "model": "Qwen/Qwen2.5-72B-Instruct"
            }
        }
        return SubPromptTree

    def GetAnalysisSubTree(self, user_input, data_info=None):
        analysis_tags_list = [
            "Time_series_forecast",
            "Correlation_Spearman",
            "Correlation_Pearson",
            "Correlation_Kendall",
            "Regression_Ridge",
            "Regression_Lasso",
            "Regression_RandomForest",
            "Clustering_KMeans",
            "Clustering_DBSCAN",
            "Clustering_Hierarchical",
            "Kaplan_Meier",
            "Cox_Proportional_Hazards"
        ]
        # print(data_info)
        # print(data_info["attribute names"])
        # print("analysis_tag_list", analysis_tag_list)
        SubPromptTree = {
            "ClassifyDataTools":{
                "prompt": f"""
                    ###Instructions: 
                    1. Review the user's request and the data information provided. 
                    2. Identify the predictor (independent) variable(s) and the target (dependent) variable(s) in the data.
                    3. Select the most relevant analysis method by choosing the appropriate tag from the list below. 
                    4. Provide only the method tag, the predictor variable(s), and the target variable(s) in the format: 
                    {{"method": "method_tag", "predictor": ["predictor1", "predictor2", ...], "target": ["target1", "target2", ...]}}

                    ###Method Tags:
                    - Time_series_forecast: Time Series Analysis model, you should output this tag if you think users requires time series analysis.
                    - Correlation_Spearman: Correlation Analysis using Spearman's rank correlation coefficient.
                    - Correlation_Pearson: Correlation Analysis using Pearson's correlation coefficient.
                    - Correlation_Kendall: Correlation Analysis using Kendall's tau coefficient.
                    - Regression_Ridge: Regression Analysis using Ridge Regression.
                    - Regression_Lasso: Regression Analysis using Lasso Regression.
                    - Regression_RandomForest: Regression Analysis using Random Forest.
                    - Clustering_KMeans: Clustering Analysis using K-Means Clustering.
                    - Clustering_DBSCAN: Clustering Analysis using DBSCAN Clustering.
                    - Clustering_Hierarchical: Clustering Analysis using Hierarchical Clustering.
                    - Kaplan_Meier: Survival Analysis using Kaplan-Meier method.
                    - Cox_Proportional_Hazards: Survival Analysis using Cox Proportional Hazards model.

                    ### Examples: 

                    #### Example 1:

                    **Input:**
                    User Request: "Forecast the next 6 months of sales based on historical data."
                    Data Summary: {{
                        "description": "sales and date",
                        "attribute names": ["Sales", "Date"],
                        "dtypes": [timestamp, int64],
                        "shape": (36, 2),
                        "data_head": {{
                            "Sales": [500, 600, 550, 650, 780],
                            "Date": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04", "2021-01-05"]
                        }}
                    }}

                    **Output:**
                    {{"method": "TimeSeries_ARIMA", "predictor": ["Sales"], "target": ["Sales"]}}

                    #### Example 2:

                    **Input:**
                    User Request: "Analyze the relationship between advertising spend and sales."
                    Data Summary: {{
                        "description": "sales and advertising data",
                        "attribute names": ["Sales", "Advertising"],
                        "dtypes": [int64, int64],
                        "shape": (128, 2),
                        "data_head": {{
                            "Sales": [500, 600, 550, 650, 780],
                            "Advertising": [100, 120, 110, 130, 150]
                        }}
                    }}

                    **Output:**
                    {{"method": "Correlation_Spearman", "predictor": ["Advertising"], "target": ["Sales"]}}

                    #### Example 3:

                    **Input:**
                    User Request: "Segment customers based on their annual spending and purchase frequency."
                    Data Summary: {{
                        "description": "customer segmentation data",
                        "attribute names": ["CustomerID", "Spending", "Frequency"],
                        "dtypes": [int64, int64, int64],
                        "shape": (200, 3),
                        "data_head": {{
                            "CustomerID": [1, 2, 3, 4, 5],
                            "Spending": [500, 600, 550, 650, 780],
                            "Frequency": [2, 3, 2, 4, 5]
                        }}
                    }}

                    **Output:**
                    {{"method": "Clustering_KMeans", "predictor": ["Spending", "Frequency"], "target": ["CustomerID"]}}

                    ### Task: 

                    User Request: {user_input}

                    Data Summary: {data_info}

                    **Provide** 
                    - method tag: one of the tags from the list {analysis_tags_list}
                    - predictor variables: list of predictor variables from {data_info["attribute names"]}
                    - target variables: list of target variables from {data_info["attribute names"]}
                """,
                "options": analysis_tags_list,
                "max_length": 50,
                "model": "Qwen/Qwen2.5-72B-Instruct", 
            }
        }

        return SubPromptTree

 

        
    
        
   
       
   


        


    

