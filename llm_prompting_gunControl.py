import os
import numpy as np
import pandas as pd
import re
from langchain import PromptTemplate
from langchain_community.llms import Ollama
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from pprint import pprint
import logging
from datetime import datetime, timedelta
import ast
import json
import time

def get_response_llm(tweet1, tweet2, topic, topic_categories, model, index):
    print(index)
    response = {
        'tweet1_stance_explanation': '',
        'tweet1_stance': '',        
        'tweets_agreement_explanation': '',
        'tweets_agreement': '',        
        'tweet1_affect_explanation': '',
        'tweet1_affect': '',                
    }     
    if tweet1 and tweet2:
        model = model.bind(
            functions=[
                {
                    "name": "get_affective_polarization",
                    "description": (
                        "Analyze the content of the provided tweets to assess their stance and emotional tone with respect to the topic: gun access. " +  
                        "Provide detailed classifications for stance of the tweets, agreement between tweets, and affective polarization (emotional negativity towards opposing views). "
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tweet1_stance_explanation": {
                                "type": "string",
                                "description": (
                                    "Provide a brief explanation of tweet1's stance on the topic: gun access. " + 
                                    "If tweet1 supports or in favor of gun access, then classify it's stance as pro. " + 
                                    "If tweet1 opposes or against of gun access, then classify it's stance as anti. " + 
                                    "Also explain the reasoning behind the classification of the stance. " + 
                                    "If the stance is not clear, label the stance as don't know."
                                )
                            },
                            "tweet1_stance": {
                                "type": "string",
                                "description": (
                                    "Classify tweet1's stance on the topic: gun access. " + 
                                    "Possible values: pro (if supports or in favor of gun access), anti (if opposes or against gun access), don't know (if it is not clear whether the tweet supports or opposes gun access). This classification should be based on the explanation provided in 'tweet1_stance_explanation'. "
                                ),
                                "enum": topic_categories
                            },
                            "tweets_agreement_explanation": {
                                "type": "string",
                                "description": (
                                    "Provide an explanation of whether tweet1 and tweet2 agree or disagree on the topic: gun access. " + 
                                    "Also explain the reasoning. Agreement indicates similar views, disagreement means opposing views. " + 
                                    "If tweet2 is not available, state 'not applicable'. If the agreement is unclear, provide reasoning."
                                )
                            },
                            "tweets_agreement": {
                                "type": "string",
                                "description": (
                                    "Classify the agreement between tweet1 and tweet2 with respect to the topic: gun access" + 
                                    ". Possible values: yes (agreement), no (disagreement), don't know (unclear)."
                                ),
                                "enum": ["yes", "no", "don't know"]
                            },
                            "tweet1_affect_explanation": {
                                "type": "string",
                                "description": (
                                    "Explain whether tweet1 contains affective polarization i.e. deeply negative emotions or attitudes specifically towards people who hold opposing views on the topic: gun access. The focus is on emotional negativity beyond the stance itself."
                                )
                            },
                            "tweet1_affect": {
                                "type": "string",
                                "description": (
                                    "Classify tweet1's affective polarization, i.e., emotional negativity specifically towards opposing views on the topic: gun access. " + 
                                    "Possible values: yes (contains affective polarization), no (doesn't contain affective polarization), don't know (uncertain about affective polarization)."
                                ),
                                "enum": ["yes", "no", "don't know"]
                            },
                        },
                        "required": [
                            "tweet1_stance_explanation", "tweet1_stance", 
                            "tweets_agreement_explanation", "tweets_agreement",
                            "tweet1_affect_explanation", "tweet1_affect",
                        ],
                    }
                }
            ],
            function_call={"name": "get_affective_polarization"},
        )    
        res_1, res_2, res_3, res_4, res_5, res_6 = '', '', '', '', '', ''    
        pattern_res_1 = re.compile(r'tweet1_stance_explanation', re.IGNORECASE)
        pattern_res_2 = re.compile(r'tweet1_stance', re.IGNORECASE)
        pattern_res_3 = re.compile(r'tweets_agreement_explanation', re.IGNORECASE)
        pattern_res_4 = re.compile(r'tweets_agreement', re.IGNORECASE)
        pattern_res_5 = re.compile(r'tweet1_affect_explanation', re.IGNORECASE)
        pattern_res_6 = re.compile(r'tweet1_affect', re.IGNORECASE)    
        try:
            result = model.invoke("Tweet1: " + tweet1 + " \n\n\n\n Tweet2: " + tweet2)
            tool_calls = result.to_json()['kwargs']['tool_calls']
            for tc in tool_calls:
                if tc['name'] == 'get_affective_polarization':
                    args = tc['args']
                    for key, value in args.items():
                        if pattern_res_1.search(key):
                            res_1 = value
                        elif pattern_res_2.search(key) and 'explanation' not in key:
                            res_2 = value
                        elif pattern_res_3.search(key):
                            res_3 = value
                        elif pattern_res_4.search(key) and 'explanation' not in key:
                            res_4 = value
                        elif pattern_res_5.search(key):
                            res_5 = value
                        elif pattern_res_6.search(key) and 'explanation' not in key:
                            res_6 = value
                    break
        except Exception as e1:
            try:
                error_message = str(e1)
                if 'tool_input' in error_message:
                    json_start_index = error_message.find('{')
                    json_end_index = error_message.rfind('}') + 1
                    json_str = error_message[json_start_index:json_end_index]
                    parsed_json = json.loads(json_str)
                    args = parsed_json.get("tool_input", {})
                    for key, value in args.items():
                        if pattern_res_1.search(key):
                            res_1 = value
                        elif pattern_res_2.search(key) and 'explanation' not in key:
                            res_2 = value
                        elif pattern_res_3.search(key):
                            res_3 = value
                        elif pattern_res_4.search(key) and 'explanation' not in key:
                            res_4 = value
                        elif pattern_res_5.search(key):
                            res_5 = value
                        elif pattern_res_6.search(key) and 'explanation' not in key:
                            res_6 = value            
                else:
                    raise Exception('tool_input not in error message!')
            except Exception as e2:
                if "don't have direct access to external" in str(e1) or "don't have direct access to external" in str(e2) or \
                "need to analyze the content" in str(e1) or "need to analyze the content" in str(e2) or \
                "need more information" in str(e1) or "need more information" in str(e2) or \
                "need more context" in str(e1) or "need more context" in str(e2) or \
                "Expecting ',' delimiter" in str(e1) or "Expecting ',' delimiter" in str(e2) or \
                "Error parsing JSON: Invalid" in str(e1) or "Error parsing JSON: Invalid" in str(e2) or \
                "Error parsing JSON: Extra data" in str(e1) or "Error parsing JSON: Extra data" in str(e2) or \
                "not sure how they relate to the topic" in str(e1) or "not sure how they relate to the topic" in str(e2) or \
                "don't have enough information" in str(e1) or "don't have enough information" in str(e2) or \
                "cannot access external" in str(e1) or "cannot access external" in str(e2) or \
                "don't have the capability to access external" in str(e1) or "don't have the capability to access external" in str(e2) or \
                "don't seem to be related to" in str(e1) or "don't seem to be related to" in str(e2):
                    pass
                else:
                    add_log(f'Error in model.invoke(): {e1}\nError parsing JSON: {e2}')                
        ures_2 = "don't know"
        if res_2 == 'pro':
            ures_2 = 'anti'
        if res_2 == 'anti':
            ures_2 = 'pro'
        response = {
            'tweet1_stance_explanation': res_1,
            'tweet1_stance': ures_2,        
            'tweets_agreement_explanation': res_3,
            'tweets_agreement': res_4,        
            'tweet1_affect_explanation': res_5,
            'tweet1_affect': res_6,                
        }                        
    elif tweet1:
        model = model.bind(
            functions=[
                {
                    "name": "get_affective_polarization",
                    "description": (
                        "Analyze the content of the provided tweets to assess their stance and emotional tone with respect to the topic: gun access. " +  
                        "Provide detailed classifications for stance of the tweets, agreement between tweets, and affective polarization (emotional negativity towards opposing views). "
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tweet1_stance_explanation": {
                                "type": "string",
                                "description": (
                                    "Provide a brief explanation of tweet1's stance on the topic: gun access. " + 
                                    "If tweet1 supports or in favor of gun access, then classify it's stance as pro. " + 
                                    "If tweet1 opposes or against of gun access, then classify it's stance as anti. " + 
                                    "Also explain the reasoning behind the classification of the stance. " + 
                                    "If the stance is not clear, label the stance as don't know."
                                )
                            },
                            "tweet1_stance": {
                                "type": "string",
                                "description": (
                                    "Classify tweet1's stance on the topic: gun access. " + 
                                    "Possible values: pro (if supports or in favor of gun access), anti (if opposes or against gun access), don't know (if it is not clear whether the tweet supports or opposes gun access). This classification should be based on the explanation provided in 'tweet1_stance_explanation'. "
                                ),
                                "enum": topic_categories
                            },
                            "tweet1_affect_explanation": {
                                "type": "string",
                                "description": (
                                    "Explain whether tweet1 contains affective polarization i.e. deeply negative emotions or attitudes specifically towards people who hold opposing views on the topic: gun access. The focus is on emotional negativity beyond the stance itself."
                                )
                            },
                            "tweet1_affect": {
                                "type": "string",
                                "description": (
                                    "Classify tweet1's affective polarization, i.e., emotional negativity specifically towards opposing views on the topic: gun access. " + 
                                    "Possible values: yes (contains affective polarization), no (doesn't contain affective polarization), don't know (uncertain about affective polarization)."
                                ),
                                "enum": ["yes", "no", "don't know"]
                            },
                        },
                        "required": [
                            "tweet1_stance_explanation", "tweet1_stance", 
                            "tweet1_affect_explanation", "tweet1_affect",
                        ],
                    }
                }
            ],
            function_call={"name": "get_affective_polarization"},
        )    
        res_1, res_2, res_3, res_4, res_5, res_6 = '', '', '', '', '', ''    
        pattern_res_1 = re.compile(r'tweet1_stance_explanation', re.IGNORECASE)
        pattern_res_2 = re.compile(r'tweet1_stance', re.IGNORECASE)
        pattern_res_5 = re.compile(r'tweet1_affect_explanation', re.IGNORECASE)
        pattern_res_6 = re.compile(r'tweet1_affect', re.IGNORECASE)    
        try:
            result = model.invoke("Tweet1: " + tweet1)
            tool_calls = result.to_json()['kwargs']['tool_calls']
            for tc in tool_calls:
                if tc['name'] == 'get_affective_polarization':
                    args = tc['args']
                    for key, value in args.items():
                        if pattern_res_1.search(key):
                            res_1 = value
                        elif pattern_res_2.search(key) and 'explanation' not in key:
                            res_2 = value
                        elif pattern_res_5.search(key):
                            res_5 = value
                        elif pattern_res_6.search(key) and 'explanation' not in key:
                            res_6 = value
                    break
        except Exception as e1:
            try:
                error_message = str(e1)
                if 'tool_input' in error_message:
                    json_start_index = error_message.find('{')
                    json_end_index = error_message.rfind('}') + 1
                    json_str = error_message[json_start_index:json_end_index]
                    parsed_json = json.loads(json_str)
                    args = parsed_json.get("tool_input", {})
                    for key, value in args.items():
                        if pattern_res_1.search(key):
                            res_1 = value
                        elif pattern_res_2.search(key) and 'explanation' not in key:
                            res_2 = value
                        elif pattern_res_5.search(key):
                            res_5 = value
                        elif pattern_res_6.search(key) and 'explanation' not in key:
                            res_6 = value            
                else:
                    raise Exception('tool_input not in error message!')
            except Exception as e2:
                if "don't have direct access to external" in str(e1) or "don't have direct access to external" in str(e2) or \
                "need to analyze the content" in str(e1) or "need to analyze the content" in str(e2) or \
                "need more information" in str(e1) or "need more information" in str(e2) or \
                "need more context" in str(e1) or "need more context" in str(e2) or \
                "Expecting ',' delimiter" in str(e1) or "Expecting ',' delimiter" in str(e2) or \
                "Error parsing JSON: Invalid" in str(e1) or "Error parsing JSON: Invalid" in str(e2) or \
                "Error parsing JSON: Extra data" in str(e1) or "Error parsing JSON: Extra data" in str(e2) or \
                "not sure how they relate to the topic" in str(e1) or "not sure how they relate to the topic" in str(e2) or \
                "don't have enough information" in str(e1) or "don't have enough information" in str(e2) or \
                "cannot access external" in str(e1) or "cannot access external" in str(e2) or \
                "don't have the capability to access external" in str(e1) or "don't have the capability to access external" in str(e2) or \
                "don't seem to be related to" in str(e1) or "don't seem to be related to" in str(e2):
                    pass
                else:
                    add_log(f'Error in model.invoke(): {e1}\nError parsing JSON: {e2}')                
        ures_2 = "don't know"
        if res_2 == 'pro':
            ures_2 = 'anti'
        if res_2 == 'anti':
            ures_2 = 'pro'
        response = {
            'tweet1_stance_explanation': res_1,
            'tweet1_stance': ures_2,        
            'tweets_agreement_explanation': res_3,
            'tweets_agreement': res_4,        
            'tweet1_affect_explanation': res_5,
            'tweet1_affect': res_6,                
        }                        
    else:
        pass
    return response
        

def generate_info_SAA(df, llm, topic, topic_categories):
    df["Tweet Stance"] = ""
    df["Tweet Affect"] = ""
    df["Tweets Agreement"] = ""

    df["Tweet Stance Explanation"] = ""
    df["Tweet Affect Explanation"] = ""
    df["Tweets Agreement Explanation"] = ""

    for i in range(df.shape[0]):
        id_tweet = df.index[i]
        txt_tweet = df.loc[id_tweet]["Tweet Text"]
        id_parent_tweet = df.loc[id_tweet]["Parent Tweet ID"]

        if pd.api.types.is_numeric_dtype(type(id_parent_tweet)):
            is_nan = np.isnan(id_parent_tweet)
        else:
            is_nan = False

        if id_parent_tweet and not is_nan and id_parent_tweet != "Root Author":
            try:
                id_parent_tweet = int(id_parent_tweet)                
                
                txt_parent_tweet = df.loc[id_parent_tweet]["Tweet Text"]

                result =  get_response_llm(txt_tweet, txt_parent_tweet, topic, topic_categories, llm, i) 

                df.at[id_tweet, "Tweet Stance"] = result["tweet1_stance"]
                df.at[id_tweet, "Tweet Affect"] = result["tweet1_affect"]
                df.at[id_tweet, "Tweets Agreement"] = result["tweets_agreement"]

                df.at[id_tweet, "Tweet Stance Explanation"] = result["tweet1_stance_explanation"]
                df.at[id_tweet, "Tweet Affect Explanation"] = result["tweet1_affect_explanation"]
                df.at[id_tweet, "Tweets Agreement Explanation"] = result["tweets_agreement_explanation"]   

            except Exception as e:
                add_log(f'Error in SAA: {e}')                

                result =  get_response_llm(txt_tweet, '', topic, topic_categories, llm, i) 

                df.at[id_tweet, "Tweet Stance"] = result["tweet1_stance"]
                df.at[id_tweet, "Tweet Affect"] = result["tweet1_affect"]

                df.at[id_tweet, "Tweet Stance Explanation"] = result["tweet1_stance_explanation"]
                df.at[id_tweet, "Tweet Affect Explanation"] = result["tweet1_affect_explanation"]
        else:

            result =  get_response_llm(txt_tweet, '', topic, topic_categories, llm, i) 

            df.at[id_tweet, "Tweet Stance"] = result["tweet1_stance"]
            df.at[id_tweet, "Tweet Affect"] = result["tweet1_affect"]

            df.at[id_tweet, "Tweet Stance Explanation"] = result["tweet1_stance_explanation"]
            df.at[id_tweet, "Tweet Affect Explanation"] = result["tweet1_affect_explanation"]
        
    return df

def classify_interaction(stance_tweet, stance_parent_tweet, affect_tweet, affect_parent_tweet, tweets_agreement):
    # Function to classify the interaction
    # Generate categories for pairs of tweets
    #
    # Tweet Stance |  Parent Stance | Tweet Affect | Parent Affect | Agreement | Class | Description 
    #      same stance              |         yes anywhere         |     yes   |   10  | very high danger - eco chamber like no interaction with the oposite stance but negative emotions towards the oposite stance        
    #      same stance              |         yes anywhere         |     no    |    8  | high danger - disagreement on the same stance is a good sign but still negative emotions towards the oposite stance
    #      same stance              |              no              |     yes   |    6  | medium danger - eco chamber like no interaction with the oposite stance        
    #      same stance              |              no              |     no    |    2  | low danger - disagreement on the same stance is very good 
    #      opposite stance          |         yes anywhere         |     no    |    8  | high danger - disagreement opposite stance but with negative emotions (still a bit better that there is interaction)
    #      opposite stance          |         yes anywhere         |     yes   |    6  | medium danger - agreeemnt opposite stances is good but with negative emotions
    #      opposite stance          |              no              |     no    |    4  | less danger - interaction with no negative emotions
    #      opposite stance          |              no              |     yes   |    0  | no danger - interaction, no negative emotions, and agreement

    #
    # TODO: further refinement - understand the implications on the replies
    #
    # if the Parent Affect = yes and Tweet Affect = no but Agreement then this is more dangerous than
    # if the Parent Affect = no and Tweet Affect = yes but Agreement 
    
    
    same_stance = stance_tweet == stance_parent_tweet
    any_affect_yes = affect_tweet == 'yes' or affect_parent_tweet == 'yes'

    # Correct classification logic based on the detailed table
    if same_stance:
        if any_affect_yes:
            if tweets_agreement == 'yes':
                return 10  # very high danger - eco chamber like no interaction with the oposite stance but negative emotions towards the oposite stance    
            else:
                return 8  # high danger - disagreement on the same stance is a good sign but still negative emotions towards the oposite stance
        else:
            if tweets_agreement == 'yes':
                return 6  # medium danger - eco chamber like no interaction with the oposite stance    
            else:
                return 2  # low danger - disagreement on the same stance is very good 
    else:  # Opposite stance
        if any_affect_yes:
            if tweets_agreement == 'yes':
                return  6 # medium danger - agreeemnt opposite stances is good but with negative emotions
            else:
                return 8  # high danger - disagreement opposite stance but with negative emotions (still a bit better that there is interaction)
        else:
            if tweets_agreement == 'yes':
                return 0  # no danger - interaction, no negative emotions, and agreement
            else:
                return 4  # less danger - interaction with no negative emotions

def generate_info_IC(df):
    df['Interaction Class'] = None

    for i in range(df.shape[0]):
        id_tweet = df.index[i]
        txt_tweet = df.loc[id_tweet]["Tweet Text"]
        id_parent_tweet = df.loc[id_tweet]["Parent Tweet ID"]
        stance_tweet = df.loc[id_tweet]["Tweet Stance"]
        affect_tweet = df.loc[id_tweet]["Tweet Affect"]
        tweets_agreement = df.loc[id_tweet]["Tweets Agreement"]

        if pd.api.types.is_numeric_dtype(type(id_parent_tweet)):
            is_nan = np.isnan(id_parent_tweet)
        else:
            is_nan = False

        if id_parent_tweet and not is_nan and id_parent_tweet != "Root Author":
            try:
                id_parent_tweet = int(id_parent_tweet)                

                txt_parent_tweet = df.loc[id_parent_tweet]["Tweet Text"]
                stance_parent_tweet = df.loc[id_parent_tweet]["Tweet Stance"]
                affect_parent_tweet = df.loc[id_parent_tweet]["Tweet Affect"]
                interaction_class = classify_interaction(stance_tweet, stance_parent_tweet, affect_tweet, affect_parent_tweet, tweets_agreement)
                df.at[id_tweet, "Interaction Class"] = interaction_class        
            except Exception as e:
                add_log(f'Error in IC: {e}')                                
                continue
    return df

def add_log(message, level='info'):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"Time: {current_time} - {message}"
    
    if level == 'info':
        logging.info(log_message)
    elif level == 'error':
        logging.error(log_message)
    else:
        logging.debug(log_message)  
        

if __name__=='__main__':
    topic = "gun access"
    topic_categories = ["pro", "anti", "don't know"]
    directory = "/curdir/datasets/V2test"
    pattern = re.compile(r'^.*_\d{4}-\d{2}-\d{2}.*\.csv$')
    files = sorted([file for file in os.listdir(directory) if pattern.match(file)])
    print(len(files))
    event_gc = {
        'shooting_texas': ("2022-05-24", "2022-05-24"),
        'shooting_illinois': ("2022-07-04", "2022-07-04"),
        'shooting_MIV': ("2022-06-07", "2022-06-07"),
        'shooting_colorado': ("2022-11-19", "2022-11-20"),
    }
    pattern2 = re.compile(r'^.*_(\d{4}-\d{2}-\d{2}).*\.csv$')    
    valid_files = []
    for event_name, (start_date_str, end_date_str) in event_gc.items():
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
        date_range_start = start_date - timedelta(days=3)
        date_range_end = end_date + timedelta(days=3)
        for file in files:
            match = pattern2.match(file)
            if match:
                file_date_str = match.group(1)
                file_date = datetime.strptime(file_date_str, "%Y-%m-%d")
                if date_range_start <= file_date <= date_range_end:
                    valid_files.append(file)
    print(len(files))    
    llm = OllamaFunctions(model="llama3.1:70b", temperature=0.0)
    logging.basicConfig(filename='/curdir/loggerGC.txt', level=logging.INFO)
    directory_out = "/curdir/outputs/all_gunControlV2" 

    file_index = 0
    for sample_no in range(len(files)): 
        file_path = os.path.join(directory, files[file_index])
        output_file_path = os.path.join(directory_out, os.path.basename(file_path))
        while os.path.exists(output_file_path):
            file_index +=1
            if file_index == len(files):
                break
            file_path = os.path.join(directory, files[file_index])
            output_file_path = os.path.join(directory_out, os.path.basename(file_path))
            
        if file_index == len(files): 
            break
            
        add_log(f'{file_index+1}/{len(files)}: {file_path} - START')    
        df = pd.read_csv(file_path, index_col="Tweet ID", dtype={'Parent Tweet ID': 'object'})  
        print(df.shape)
        df = generate_info_SAA(df, llm, topic, topic_categories)
        df = generate_info_IC(df) 
        df2 = df[df['Interaction Class'].notna() | (df["Parent Tweet ID"] == 'Root Author')]
        df2.to_csv(output_file_path, encoding='utf-8-sig', errors='replace')
        
        add_log(f'{file_index+1}/{len(files)}: {file_path} - END')