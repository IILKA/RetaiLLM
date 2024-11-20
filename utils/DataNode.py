import pandas as pd
class DataContainer: 
    '''
    DataContainer is a class that holds the data and its other information
    '''
    def __init__(self, df = None, description = None, type=None, name = None):
        self.df = df 
        self.description = description
        self.names = df.columns.tolist() if df is not None else []
        self.dtypes = df.dtypes.tolist() if df is not None else []
        self.id = "_".join(self.names) if name is None else name
        self.num = 0 # used to keep the sequence of the data 
        self.method = None
    
    def from_file(self, file_path, description=None):
        self.df = pd.read_csv(file_path)
        self.description = description
        self.names = self.df.columns.tolist()
        self.dtypes = self.df.dtypes.tolist()
        self.id = file_path.split("/")[-1].split(".")[0]

    def get_info(self):
        return {
            "description": self.description,
            "attribute names": self.names,
            "dtypes": self.dtypes, 
            "shape": self.df.shape,
            "data_head": self.df.head().to_dict()
        }
    
    def __str__(self):
        return str(self.get_info())

class DataNode: 
    """
    DataNode is singleton class that handle the all the data in the system 
    Data: a dictionary of DataContainer {id: DataContainer}
    id_list: a list of id of the data
    activate_id: the id of the data that is currently activated
    root_dir: the root directory of the user uploaded files

    """
    _instance = None
    _is_initialized = False

    def __new__(cls):
        #singleton pattern
        if cls._instance is None: 
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, root_dir = ".", debug = True):
        #initialize the model
        if not self._is_initialized:
            self.Data = {}
            self.id_list = []
            self._is_initialized = True
            self.activate_id = None 
            self._debug = debug
            DataNode._is_initialized = True
            self.count = 0
            self.root_dir = root_dir
        
    
    def add_data(self, data: DataContainer):
        #check if the id is already in the data
        if data.id in self.Data:
            data.id = data.id + "_copy"
            self.Data[data.id] = data
        else:
            self.Data[data.id] = data
        self.id_list.append(data.id)
        self.activate_id = data.id


    def get_data(self, id):
        return self.Data[id]

    def get_current_data(self):
        #return the current activated data
        if self._debug: 
            print("Activate id is:", self.activate_id)
            print("the id list is:", self.id_list)
            print("the data is:", self.Data)

        return self.Data[self.activate_id]
    
    def get_data_description(self, id):
        #return all the description of the data
        return {self.Data[id].id: self.Data[id].description for id in self.id_list}
    
    def update_activate_id(self, id):
        self.activate_id = id
        if self._debug:
            print("Activate id is updated to:", self.activate_id)

    def get_data_list(self, id_only = False):
        #return a list of {name: name, description: description}
        if id_only: 
            return self.id_list
        else: 
            return [{"name": id, "description": self.Data[id].description} for id in self.id_list]
        
    def __len__(self):
        return len(self.Data)
    
    def __print__(self):
        return self.get_data_description(self.activate_id)

    def save(self, name = "None"):
        '''
        Save this Data node into a file for next login conversation

        '''
        pass

    def read(self, file_path):
        '''
        Read form this a file_path for next login conversation

        '''
        pass

    

    




    

    
    
