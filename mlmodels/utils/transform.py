# transform original set to training set for disambiguity model
# "total sales from CUSTOMER_VALUE_LABEL" ---> "total sales from <REPLACE>", "O O O CUSTOMER_VALUE_LABLE"
import pandas as pd
"""
example: transform_file(original_file, destination_file)
original_file: NP training Corpus with columns [eng_query, sql]
destination file format
total sales from <REPLACE>, O O O CUSTOMER_VALUE_LABLE
"""

prefix = "s_"

def transform_col1(text):
    token_list = text.split()
    column1 = ' '.join(["<REPLACE>" if 'value_label' in token.lower() or 'duckling' in token.lower()
                        else token for token in token_list])
    return column1

def clean_num(token):
    if "duckling_time" in token.lower():
        return prefix + "duckling_time"
    if "duckling_amount" in token.lower():
        return prefix + "duckling_amount"
    else:
        return (prefix + token).replace("'", "")

def transform_col2(text):
    token_list=text.split()
    column2 = ' '.join(["o" if not ('value_label' in token.lower() or 'duckling' in token.lower())
                        else clean_num(token) for token in token_list])
    return column2

def transform_file(original_file, converted_file):
    try:
        data = pd.read_csv(original_file, header=0, index_col=None)
        new_datafram = pd.DataFrame()
        new_datafram["input_label"] = data["eng_query"].apply(transform_col1)
        new_datafram["output_label"] = data["eng_query"].apply(transform_col2)
        new_datafram.to_csv(converted_file, index=False)

    except Exception as ex:
        raise Exception("Transform files has an error of {}".format(str(ex)))
