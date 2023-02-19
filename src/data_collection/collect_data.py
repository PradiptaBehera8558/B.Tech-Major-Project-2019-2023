def return_dataframe(path, file_type, **kwargs):
    """
    Return the dataframe as per source like flat file types
    If multiple data source included switch to Oops

    Args:
        path: path of the data source
        file_type: Type of file like CSV, excel etc 
    """

    if file_type == 'xlsx':
        pass
    if file_type == '.csv':
        pass
    else:
        print("file type not supported!")
    return df