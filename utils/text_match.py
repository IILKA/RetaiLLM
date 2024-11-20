from rapidfuzz import fuzz, process 

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


if __name__ == "__main__":
    test_text_match = True

    if test_text_match:
        query = "-USER_INPUT: The data can be directly extracted from the user's input or the user has specified a given data source enclosed in <data></data>.-"
        choices = ["USER_INPUT", "PREVIOUS_CONVO_DATA", "WEB_SCRAPE", "DB_QUERY", "NO_DATA", "MENU"]
        print(get_closest_match(query, choices))