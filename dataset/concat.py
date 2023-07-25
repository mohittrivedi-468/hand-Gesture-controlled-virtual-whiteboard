import pandas as pd 

df1 = pd.read_csv('D:/virtual whiteboard/dataset/draw.csv')
df2 = pd.read_csv('D:/virtual whiteboard/dataset/erase.csv')
df3 = pd.read_csv('D:/virtual whiteboard/dataset/none.csv')

df4 = pd.concat([df1, df2])
df5 = pd.concat([df4, df3])
df5.to_csv("D:/virtual whiteboard/dataset/dataset.csv")