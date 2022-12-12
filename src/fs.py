from feature_selector import FeatureSelector
import pandas as pd

df = pd.read_parquet("df_input.parquet")


LABEL_COLUMN_NAME = "voto"
WANTED_COLUMNS = df.columns.to_list()

WANTED_COLUMNS.remove("voto")
WANTED_COLUMNS.remove("peso_investimento_partidario")
WANTED_COLUMNS.remove("doacoes_id_parlamentar")

WANTED_COLUMNS.remove("peso_indice_peso_politico")


fs = FeatureSelector(df, WANTED_COLUMNS, LABEL_COLUMN_NAME, 10000)

r = fs.select_features(WANTED_COLUMNS)
