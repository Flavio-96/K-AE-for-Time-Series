from tsfresh import extract_relevant_features, extract_features
import utilFeatExtr as util
import pandas as pd
import sys

# CALL EXAMPLE
# python featureExtractionTsfresh.py ../data/UCRArchive_2018/ECG200/ECG200_TEST.tsv ECG200.pkl

if __name__ == '__main__': 

    # Path of the file target
    filePath = sys.argv[1]
    # File name for results from tsfresh activity
    resultName = sys.argv[2]

	# Series ti restituisce anche le classi di appartenenza perché vi servono se volete estrarre
	# le features rilevanti
    listOut,series = util.adaptTimeSeries(filePath)
    
    print(series)
    # Questa è la funzione che vi estrae quelle interessanti
    features_filtered_direct = extract_relevant_features(listOut,series, column_id='id', column_sort='time')

    # Questa è la funzione che vi estrae tutte le features
    features_filtered_direct = extract_features(listOut,column_id='id', column_sort='time')
    print(len(features_filtered_direct))
    
    # Questa funzione consente di salvare le features che avete estratto, senza doverle riestrarre di nuovo
    # Occhio all'estensione perché se no jastemmate pur francese, esperienza personale.
    features_filtered_direct.to_pickle("../tsfreshResults/"+resultName)

    # Questa funzione invece vi consente di estrarre le features dal pickle creato
    # features_filtered_direct = pd.read_pickle("<filename>.pkl")