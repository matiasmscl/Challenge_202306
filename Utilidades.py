import pandas

def Quitarlista(Dataframe,Lista):
    to_drop=[]
    for elemento0 in Lista:
        for elemento1 in Dataframe.columns:
            if elemento0 in elemento1:
                to_drop=to_drop+[elemento1]
    Dataframe=Dataframe.drop(columns=to_drop)
    print(to_drop)
    return(Dataframe)


def LlenarTabla(Tabla0,Tabla1):
    Tabla1=Tabla1.copy().reset_index().drop(columns=['index'])
    Tabla0=Tabla0.copy().reset_index().drop(columns=['index'])
    DB_copy=Tabla1.copy()
    to_drop=['index']
    for col1 in Tabla1.columns:
        cuenta=0
        for col0 in Tabla0.columns:
            if (col0 in col1) and ((col0+'_') not in col1):
                for fil0 in Tabla0.index.tolist():
                    fil1=fil0+len(Tabla1)
                    DB_copy.loc[fil1,col1]=Tabla0.loc[fil0,col0]
                cuenta=cuenta+1
        if cuenta==0:
            to_drop=to_drop+[col1]
    DB_copy=DB_copy.copy().reset_index().drop(columns=to_drop)
    return(DB_copy)



def convert_to_numeric(value):
    if isinstance(value, int):
        return value
    elif isinstance(value, float):
        return int(value) if value.is_integer() else value
    elif isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            try:
                float_value = float(value)
                return int(float_value) if float_value.is_integer(
                    ) else float_value
            except ValueError:
                return value
    else:
        return value

def limpiezabasica(dataframe):
    aux0=set(dataframe.copy().columns.tolist())
    to_drop=[]
    for col in dataframe.columns:
        dataframe[col]=dataframe[col].apply(convert_to_numeric)
        try:
            if len(dataframe[col].unique())<2:
                to_drop=to_drop+[col]
                print('Drop no variacion:'+col)
                print(dataframe[col].unique())
                aux0=aux0-set([col])
        except:
            print('Excepcion en:')
            print(col)
    dataframe=dataframe.drop(columns=to_drop)
    dataframe=dataframe.drop_duplicates()
    dataframe=dataframe.T.drop_duplicates().T
    aux1=set(dataframe.copy().columns.tolist())
    print('Drop col redundante:')
    print(list(aux0-aux1))
    for col in dataframe.columns:
        dataframe[col]=dataframe[col].apply(convert_to_numeric)
    return dataframe


def limpiarBD(dataframe, limite=50, exceptions=None):
    dataframe=dataframe.drop_duplicates()
    dataframe=dataframe.T.drop_duplicates().T
    return(limpiarBD1(dataframe, limite=limite, exceptions=exceptions))
        
def limpiarBD1(dataframe, limite=150, exceptions=None):
    dataframe=dataframe.copy()
    if exceptions is None:
        exceptions = ['Fecha-I','Fecha-O']
    # Extract exception columns that exist in the DataFrame
    existing_exceptions = [col for col in exceptions if col in dataframe.columns]
    exception_columns = dataframe[existing_exceptions].copy()
    dataframe.drop(columns=existing_exceptions, inplace=True)
    # Convert to numeric
    for col in dataframe.columns:
        dataframe[col]=dataframe[col].apply(convert_to_numeric)
    for col in exception_columns.columns:
        exception_columns[col]=exception_columns[col].apply(convert_to_numeric)
    # Drop columns with only one unique value
    to_drop = dataframe.columns[dataframe.nunique() <= 1]
    dataframe.drop(columns=to_drop, inplace=True)
    # Get columns that are not datetime
    columns_notimes = dataframe.select_dtypes(exclude=['datetime']).columns.tolist()
    # Create dummy variables
    for col in columns_notimes:
        dataframe[col]=dataframe[col].fillna(0)
        my_list = dataframe[col].unique()
        is_binary = all(x in [0, 1] for x in my_list)
        if len(my_list) < limite and not(is_binary):
            if len(dataframe[col].unique())>19:
                print('más de 19 categorias')
                print(col)
            df_dummies = pandas.get_dummies(dataframe[col], prefix=col, drop_first=False, dtype='int8')
            df_dummies.columns = df_dummies.columns.str.strip().str.replace('[\s:.;,@]+', '_', regex=True)
            dataframe = pandas.concat([dataframe, df_dummies], axis=1)
            dataframe.drop(columns=[col], inplace=True)
    # Concatenate exception columns back if they exist
    if not exception_columns.empty:
        result_dataframe = pandas.concat([dataframe, exception_columns], axis=1)
    else:
        result_dataframe = dataframe
    return result_dataframe
