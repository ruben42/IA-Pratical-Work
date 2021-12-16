#Implementar um f(iltro de spam usando o algoritmo Naive Bayes
import pandas as pd
data_dir = r'/Users/halone/Desktop/UT/IA/Projeto'

full_path = data_dir + r'//spam.csv'
read = pd.read_csv(full_path, encoding='latin1')

print(read)