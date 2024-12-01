from rapidfuzz import fuzz, process
import re 
import pandas as pd

def get_closest_match(query, choices, threshold=0):
    """
    Get the closest match from a list of choices
    """
    print("query:", query)
    match = process.extractOne(query, choices, scorer=fuzz.ratio)
    print("match:", match)
    if match[1] > threshold:
        return match[0]
    else:
        return None
    
def string2df(text):
    pattern = r'"(\w+)":\s*\[([\d,\s]+)\]'
    matches = re.findall(pattern, text)
    result = {key: [int(v) for v in values.split(",")] for key, values in matches}
    df = pd.DataFrame(result)
    return df

def method2dict(text):
    pattern = r'"method":\s*"([^"]+)",\s*"predictor":\s*(\[[^\]]*\]),\s*"target":\s*(\[[^\]]*\])'

    match = re.search(pattern, text)
    if match:
        method = match.group(1)
        predictor = eval(match.group(2))  
        target = eval(match.group(3))  
        return {
            "method": method,
            "predictor": predictor,
            "target": target
        }
    else:
        raise ValueError("Your model is too stupid to for this task. Use something smarter.")



if __name__ == "__main__":
    test_text_match = False
    test_trans = True
    if test_text_match:
        query = "-USER_INPUT: The data can be directly extracted from the user's input or the user has specified a given data source enclosed in <data></data>.-"
        choices = ["USER_INPUT", "PREVIOUS_CONVO_DATA", "WEB_SCRAPE", "DB_QUERY", "NO_DATA", "MENU"]
        print(get_closest_match(query, choices))
    if test_trans: 
        text = """```json
        {"method": "linear_regression", "predictor": ["X"], "target": ["Y"]}
        ```"""
        print(method2dict(text))
        print(string2df(text))
        