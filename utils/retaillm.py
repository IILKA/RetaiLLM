from utils.PromptTree import PromptsTree
from utils.DataNode import DataNode, DataContainer
from utils.text_match import get_closest_match
from LLM import QwenVllm
import pandas as pd
import os
import re
import ast
import json


class RetaiLLM: 
    '''
    This is the main class for the RetaiLLM agent

    '''

    def __init__(self, api_key, base_url, model_id, debug = True):
        self.prompt_tree = PromptsTree()
        self.background = self.prompt_tree.BackGround
        self.data_node = DataNode()
        self.llm = QwenVllm(api_key=api_key, base_url=base_url, model_id=model_id)
        self.root_dir = self.data_node.root_dir
        self._debug = debug
        self.check_data_source_manually = False
    
    def _wrap_message(self,text) -> list[dict]:
        '''
        used to enclose the message in the background and message

        '''
        return [{"role" : "system", "content": self.background + " " +  text}]

    def _extract_description(self, user_input, data_info) -> str:
        '''
        This is a helper function to extract the description of the data from the user input
        
        '''
    
        description = self.llm.inference(
                        self._wrap_message(
                            self.prompt_tree.GetDataSubTree(
                                user_input,
                                datalist = [],
                                data_info = data_info
                                )["PREVIOUS_CONVO_DATA"]["extract_from_previous"]["prompt"]
                        ),
                        max_length = self.prompt_tree.GetDataSubTree(
                                        user_input,
                                        datalist = [],
                                        data_info = data_info
                                    )["PREVIOUS_CONVO_DATA"]["extract_from_previous"]["max_length"]
                    )
        return description
    
    def _test_webscrape(self, webkeys):
        '''
        This is a test function to test the web scraping function

        '''
        return "This is a test function for web scraping"
        
        

    def chat(self, user_input) -> str:
        '''
        This is the main chat function that is called by the user

        '''
        # first classify the data getting method
        GetDataTree = self.prompt_tree.GetDataSubTree(
                            user_input,
                            datalist = self.data_node.get_data_list(),
                            data_info = self.data_node.get_data_description
                        ) 
        DataGetMethod = get_closest_match(
                            self.llm.inference(
                                self._wrap_message(GetDataTree["ClassifyingTasks"]["prompt"]), 
                                max_length = GetDataTree["ClassifyingTasks"]["max_length"]
                            ), 
                            GetDataTree["ClassifyingTasks"]["options"]
                        )
        if self._debug:
            print("DataGetMethod: ", DataGetMethod)
        #["USER_INPUT", "PREVIOUS_CONVO_DATA", "WEB_SCRAPE", "DB_QUERY", "NO_DATA", "MENU"]
        # Then get the data 
        if DataGetMethod == "USER_INPUT":
            user_input_tree = GetDataTree["USER_INPUT"]
            if self._debug:
                print("entering USER_INPUT")

            if not self.check_data_source_manually:
                sourcefile_or_extract = get_closest_match(
                                            self.llm.inference(
                                                self._wrap_message(user_input_tree["sourcefile_or_extract"]["prompt"]), 
                                                max_length = user_input_tree["sourcefile_or_extract"]["max_length"]
                                            ),
                                            user_input_tree["sourcefile_or_extract"]["options"]
                                        ) 
                if self._debug:  
                    print("sourcefile_or_extract: ", sourcefile_or_extract)
        
            #check if there is <data></data> in the user input
            if self.check_data_source_manually:
                if "<data>" in user_input and "</data>" in user_input:
                    sourcefile_or_extract = "SPECIFIED_SOURCE"
                else:
                    sourcefile_or_extract = "EXTRACT_DIRECTLY"
            
            if sourcefile_or_extract == "EXTRACT_DIRECTLY":     
                data_in_text = self.llm.inference(
                                    self._wrap_message(user_input_tree["Extract_from_user_input"]["prompt"]),
                                    max_length = user_input_tree["Extract_from_user_input"]["max_length"]
                                )
                if self._debug:
                    print("data_in_text: ", data_in_text)
                data = pd.DataFrame(json.loads(data_in_text))
                new_data = DataContainer(df=data)
                #extract a description of the data 
                if self._debug:
                    print("new_data: ", new_data.get_info())
                description = self._extract_description(user_input, new_data.get_info())
                new_data.description = description
                #add the data to the data node
                self.data_node.add_data(new_data)
                
                if self._debug:
                    #print current activate information 
                    print("Extracting directly from user input")
                    print("Now the activated id list is:",self.data_node.get_current_data())

            elif sourcefile_or_extract == "SPECIFIED_SOURCE":
                # the file path is in the user input and it's assumed to be enclosed by <data></data>
                try:
                    data_file_path = re.search(r'<data>(.*?)</data>', user_input).group(1)
                except:
                    return "It seems that your file path is missing"
                # the file path is assumed to be relative to the root directory
                data_file_path = os.path.join(self.root_dir, data_file_path)
                new_data = DataContainer()
                new_data.from_file(file_path = data_file_path)
                #extract a description of the data 
                description = self._extract_description(user_input, new_data.get_info())
                new_data.description = description
                #add the data to the data node
                self.data_node.add_data(new_data)
                
                if self._debug:
                    #print current activate information 
                    print("entering SPECIFIED_SOURCE")
                    print("Now the activated id list is:", self.data_node.get_current_data())
                    
            else: 
                return "I am sorry, I cannot understand the data source"
        elif DataGetMethod == "PREVIOUS_CONVO_DATA":
            previous_convo_tree = GetDataTree["PREVIOUS_CONVO_DATA"]
            #choose from previous data should return the id of the data  
            new_data_id = get_closest_match(
                        self.llm.inference(
                            self._wrap_message(previous_convo_tree["choose_from_previous"]["prompt"]),
                            max_length = previous_convo_tree["choose_from_previous"]["max_length"]
                        ),
                        self.data_node.get_data_list(id_only = True)
                    )
            #change the activated id to the new_data_id
            self.data_node.update_activate_id(new_data_id)
            if self._debug:
                print("entering PREVIOUS_CONVO_DATA")
                print("Now the activated id list is:", self.data_node.get_current_data())

        elif DataGetMethod == "WEB_SCRAPE":
            generate_keys_prompt = GetDataTree["WEB_SCRAPE"]["generate_web_scrape_keys"]

            webkeys = ast.literal_eval(
                self.llm.inference(
                    self._wrap_message(generate_keys_prompt["prompt"]),
                    max_length = generate_keys_prompt["max_length"]
                )
            )
            scraped_data = self._test_webscrape(webkeys)
            GetResponseTree = self.prompt_tree.GetResponseSubTree(
                                user_input,
                                summary = scraped_data
                            )
            #feed the scraped data into the model and return the response
            if self._debug:
                print("entering WEB_SCRAPE")
                print("webkeys: ", webkeys) 
                print("scraped_data: ", scraped_data)

            return self.llm.inference(
                        self._wrap_message(GetResponseTree["response"]["prompt"]),
                        max_length = GetResponseTree["response"]["max_length"]
                    )
        elif DataGetMethod == "DB_QUERY":
            #temporary list for test only
            db_datalist = [
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
            
            db_query_tree = self.prompt_tree.GetDataSubTree(user_input, datalist = db_datalist)
            table_name = get_closest_match(
                            self.llm.inference(
                                self._wrap_message(db_query_tree["DB_QUERY"]["choose_table"]["prompt"]),
                                max_length = db_query_tree["DB_QUERY"]["choose_table"]["max_length"]
                            ),
                            db_query_tree["DB_QUERY"]["choose_table"]["options"]
                        )
            #and the data is assumed to be in the database
            #not finished yet 
            print("temporarily note finished yet")

            if self._debug:
                print("entering DB_QUERY")
                print("table_name: ", table_name)
            
        elif DataGetMethod == "NO_DATA":
            #no data is needed in this case 
            response = self.llm.inference(
                            self._wrap_message(GetDataTree["NO_DATA"]["respond_user"]["prompt"]),
                            max_length = GetDataTree["NO_DATA"]["respond_user"]["max_length"]
                        )
            if self._debug:
                print("entering NO_DATA")
            return response
        
        elif DataGetMethod == "MENU":
            #the manual 
            menu_prompt = self.prompt_tree.GetResponseSubTree(user_input)["MENU"]
            response = self.llm.inference(
                            self._wrap_message(menu_prompt["prompt"]),
                            max_length = menu_prompt["max_length"]
                        )
            if self._debug:
                print("entering MENU")
            return response
        else:
            return "I am sorry, I cannot understand the data getting method"

        #update the method to process the data 
        GetAnalysisTree = self.prompt_tree.GetAnalysisSubTree(
                                user_input,
                                data_info = self.data_node.Data[self.data_node.activate_id].get_info()
                            )
        AnalysisMethod = self.llm.inference(
                            self._wrap_message(GetAnalysisTree["ClassifyDataTools"]["prompt"]),
                            max_length = GetAnalysisTree["ClassifyDataTools"]["max_length"]
                        )
                    
        return AnalysisMethod

    
        
        




        
            



            



            

        

            

        
            



                


        



