from openai import OpenAI
import pandas as pd 
import numpy as np
import ast

class QwenVllm:
    def __init__(self):
        self.model_id = "Qwen2.5-72B-Instruct"
        self.api_key = "xxx"
        self.base_url = "http://127.0.0.1:6006/v1/"
        self.client = OpenAI(api_key = self.api_key,base_url = self.base_url)

    def inference(self, message, max_length=512):

        chat_completion = self.client.chat.completions.create(
            messages=message,
            model="Qwen2.5-72B-Instruct",
            max_tokens=max_length
        )
        return chat_completion.choices[0].message.content
    


if __name__ == "__main__":
    from utils.PromptTree import PromptsTree
    from utils.DataNode import DataNode, DataContainer

    test = ["perform linear regression on X = [1,2,3,4,5,6], Y = [6,6,6,6,6,6]",
            "perform person correlation test on sales and sentiment <data> sales.csv <data>",
            "How about recent market in selling shoes?", 
            "can you give me some suggestion on making advertisement for my shoes?", 
            "I want to know the trend of selling shoes in the past 3 months", 
    ]
    datalist = [
    {"name": "Sales_Report_Q1", "description": "Detailed sales data for the first quarter, including revenue and units sold by region."},
    {"name": "Customer_Demographics", "description": "Insights into customer age, gender, and geographic distribution based on recent surveys."},
    {"name": "Product_Performance", "description": "Analysis of product performance metrics such as sales, returns, and customer reviews."},
    {"name": "Inventory_Levels", "description": "Current stock levels for all products, categorized by warehouse location."},
    {"name": "Marketing_Campaigns", "description": "Effectiveness data for recent marketing campaigns, including click-through rates and conversions."},
    {"name": "Competitor_Analysis", "description": "Comparison of pricing, product offerings, and market positioning with key competitors."},
    {"name": "Revenue_Projections", "description": "Forecasted revenue growth for the next fiscal year based on historical trends and market analysis."},
    {"name": "Customer_Feedback", "description": "Aggregated feedback from surveys and reviews, highlighting customer satisfaction and areas for improvement."},
    {"name": "Website_Analytics", "description": "Metrics for website traffic, user behavior, and conversion rates over the past six months."},
    {"name": "Supplier_Performance", "description": "Evaluation of supplier reliability, delivery times, and quality metrics."}
    ]

    prompt_tree = PromptsTree()
    LLMmodel = QwenVllm()
    test_classifying_tasks = False
    test_DataExtraction = False
    test_PreviousConvoData = False
    test_webscrape = True
    test_response = False

    test_get_data_tool = 0 
    



    while True:
       
        if test_classifying_tasks:
            user_input = input("test classifying tasks:")
            GetDataSubTree = prompt_tree.GetDataSubTree(user_input, datalist = datalist)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+GetDataSubTree["ClassifyingTasks"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=GetDataSubTree["ClassifyingTasks"]["max_length"])
            print("masked_response: ", masked_response)
        if test_DataExtraction:
            user_input = input("test data extraction:")
            GetDataSubTree = prompt_tree.GetDataSubTree(user_input, datalist = datalist)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+GetDataSubTree["USER_INPUT"]["sourcefile_or_extract"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=GetDataSubTree["USER_INPUT"]["sourcefile_or_extract"]["max_length"])
            print("USER_INPUT sourcefile_or_extract","masked_response: ", masked_response)
            messages2 = [
                {"role": "system", "content": prompt_tree.BackGround+" "+GetDataSubTree["USER_INPUT"]["Extract_from_user_input"]["prompt"]},
            ]
            unmasked_response, masked_response2 = LLMmodel.inference(messages2, max_tokens=GetDataSubTree["USER_INPUT"]["Extract_from_user_input"]["max_length"])
            print("USER_INPUT Edxtract_from_user_input", "masked_response: ", masked_response2)
        if test_PreviousConvoData: 
            user_input = input("test previous convo data:")
            GetDataSubTree = prompt_tree.GetDataSubTree(user_input, datalist = datalist)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+GetDataSubTree["PREVIOUS_CONVO_DATA"]["choose_from_previous"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=GetDataSubTree["PREVIOUS_CONVO_DATA"]["choose_from_previous"]["max_length"])
            print("PREVIOUS_CONVO_DATA choose_from_previous", "masked_response: ", masked_response)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+GetDataSubTree["PREVIOUS_CONVO_DATA"]["extract_from_previous"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=GetDataSubTree["PREVIOUS_CONVO_DATA"]["extract_from_previous"]["max_length"])
            print("PREVIOUS_CONVO_DATA extract_from_previous", "masked_response: ", masked_response)
        if test_webscrape:
            user_input = input("test web scrape:")
            GetDataSubTree = prompt_tree.GetDataSubTree(user_input, datalist = datalist)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+GetDataSubTree["WEB_SCRAPE"]["generate_web_scrape_keys"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=GetDataSubTree["WEB_SCRAPE"]["generate_web_scrape_keys"]["max_length"])
            print("WEB_SCRAPE generate_web_scrape_keys", "masked_response: ", masked_response)
            print("This is an keys to scrape: ", type(masked_response))
            print("This is an keys to scrape: ", type(ast.literal_eval(masked_response)), ast.literal_eval(masked_response))
            
            

        if test_response: 
            user_input = input("test response:")
            summary = input("summary:")
            tree = prompt_tree.GetResponseSubTree(user_input=user_input, summary=summary)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+tree["response"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=tree["response"]["max_length"])
            print("response", "masked_response: ", masked_response)

            user_input = input("test menu")
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+tree["MENU"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=tree["MENU"]["max_length"])
            print("menu", "masked_response: ", masked_response)

        if test_get_data_tool:

            

            # 时间序列数据生成
            date_range = pd.date_range(start="2021-01-01", periods=36, freq="ME")
            np.random.seed(0)
            sales = 100 + np.arange(36) * 10 + np.sin(np.linspace(0, 6 * np.pi, 36)) * 20 + np.random.normal(0, 5, 36)

            time_series_data = pd.DataFrame({
                "Date": date_range,
                "Sales": sales
            })

            # 数据摘要
            time_series_summary = {
                "description": "sales and date",
                "attribute names": ["Sales", "Date"],
                "dtypes": ["timestamp", "float64"],
                "shape": time_series_data.shape,
                "data_head": time_series_data.head().to_dict(orient="list")
            }
            user_input = "I want to analyze the sales trend over time" 
            tree = prompt_tree.GetAnalysisSubTree(user_input, time_series_summary)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+tree["ClassifyDataTools"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=tree["ClassifyDataTools"]["max_length"])
            print("cluster ", masked_response)

            np.random.seed(1)
            advertising = np.random.uniform(1000, 5000, 128)
            sales = advertising * 3 + np.random.normal(0, 10000, 128)

            correlation_data = pd.DataFrame({
                "Advertising": advertising,
                "Sales": sales
            })

            # 数据摘要
            correlation_summary = {
                "description": "sales and advertising data",
                "attribute names": ["Sales", "Advertising"],
                "dtypes": ["float64", "float64"],
                "shape": correlation_data.shape,
                "data_head": correlation_data.head().to_dict(orient="list")
            }
            user_input = "I want to analyze the correlation between sales and advertising"
            tree = prompt_tree.GetAnalysisSubTree(user_input, correlation_summary)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+tree["ClassifyDataTools"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=tree["ClassifyDataTools"]["max_length"])
            print("correlation ", masked_response)

            np.random.seed(2)
            price = np.random.uniform(10, 50, 100)
            advertising = np.random.uniform(500, 2000, 100)
            sales = 5000 - 50 * price + 3 * advertising + np.random.normal(0, 1000, 100)

            regression_data = pd.DataFrame({
                "Price": price,
                "Advertising": advertising,
                "Sales": sales
            })

            # 数据摘要
            regression_summary = {
                "description": "sales based on price and advertising",
                "attribute names": ["Price", "Advertising", "Sales"],
                "dtypes": ["float64", "float64", "float64"],
                "shape": regression_data.shape,
                "data_head": regression_data.head().to_dict(orient="list")
            }
            user_input = "I want to predict sales based on price and advertising"
            tree = prompt_tree.GetAnalysisSubTree(user_input, regression_summary)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+tree["ClassifyDataTools"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=tree["ClassifyDataTools"]["max_length"])
            print("regression ", masked_response)
            # print(masked_response)
            print("regression data frame",pd.DataFrame(regression_data))

            from sklearn.datasets import make_blobs

            # 聚类数据
            X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=3)
            cluster_data = pd.DataFrame(X, columns=["Annual_Spending", "Purchase_Frequency"])
            cluster_data["CustomerID"] = range(1, 201)

            # 数据摘要
            cluster_summary = {
                "description": "customer segmentation data",
                "attribute names": ["CustomerID", "Annual_Spending", "Purchase_Frequency"],
                "dtypes": ["int64", "float64", "float64"],
                "shape": cluster_data.shape,
                "data_head": cluster_data.head().to_dict(orient="list")
            }
            user_input = "I want to segment customers based on annual spending and purchase frequency"
            tree = prompt_tree.GetAnalysisSubTree(user_input, cluster_summary)
            messages = [
                {"role": "system", "content": prompt_tree.BackGround+" "+tree["ClassifyDataTools"]["prompt"]},
            ]
            unmasked_response, masked_response = LLMmodel.inference(messages, max_tokens=tree["ClassifyDataTools"]["max_length"])
            print("cluster ", masked_response)





            

        







    
        

