import pandas as pd
import rns_model
import numpy as np
from matplotlib import pyplot as plt


def write_StarDataFrame_to_csv_file(StarDataFrame, Filename):
    try:
        StarDataFrame.to_csv(Filename, index=False)
    except FileNotFoundError:
        print("We could not find the right dirctory")
        return 1
    except PermissionError:
        print("We did not have the permission to write into the file")
        return 1
    print("Data successfully written onto", Filename)
    return 0


# Reads a Dataframe with the right size (Should always be the right size here),
# and adds it to your already consisting DataFrame
def read_StarDataFrame_from_csv_file(StarDataFrame, Filename):
    DataFrame = pd.DataFrame()
    try:
        DataFrame = pd.read_csv(Filename)
    except FileNotFoundError:
        print("We could not find the right dirctory")
        return 1
    except PermissionError:
        print("We did not have the permission to write into the file")
        return 1
    print("Data successfully copied from", Filename)
    if StarDataFrame.empty:
        return DataFrame
    else:
        StarDataFrame = StarDataFrame.concat([StarDataFrame, DataFrame])
    return StarDataFrame

#I got eos data from Crhistian, but they have to many lines. I have to get rid of some of them
def clean_eos_data(Filename):
    # I have to make the eos files smaller. The limit is 200
    # Schritt 1: Datei zum Lesen öffnen und alle Zeilen in eine Liste lesen
    with open(Filename, 'r') as data:
        rows = data.readlines()
        Size = int(rows[0])
        if Size <= 200:
            return 0
        ratio = np.ceil(Size/200)

    # Schritt 2: Zeilen entfernen, die den Text "löschen" enthalten
    new_rows = []
    for i in np.arange(1,Size,ratio):
        new_rows.append(rows[int(i)])
    new_rows = [str(len(new_rows)) + "\n"] + new_rows

    # Schritt 3: Datei zum Schreiben öffnen und die modifizierte Liste zurück in die Datei schreiben
    with open(Filename, 'w') as datei:
        datei.writelines(str(lines) for lines in new_rows)

def plotStarData(sdf):
    if 'color' not in sdf.columns:
        sdf['color'] = pd.Categorical(sdf['eos']).codes
    sdf.plot.scatter(x='rho_c', y='M', c='color', colormap='viridis', legend=False)
    plt.show()

