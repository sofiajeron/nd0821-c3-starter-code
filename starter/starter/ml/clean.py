def clean_column_names(data):
    data.columns = data.columns.str.replace(' ', '')
    return data
