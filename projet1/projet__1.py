import pandas as pd
import numpy as np
import math as m
import matplotlib.pyplot as plt
import itertools
from itertools import combinations,permutations
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import simpledialog
from colorama import Fore, Style
import streamlit as st
file_path = 'Dataset3.csv'
df = pd.read_csv(file_path, delimiter=',',dtype={"Soil":str})
supp_min=0.15

def nombre_transaction(df):
    print("Nombre de transaction est :",df.shape[0])
    return df.shape[0]

def visualiser():
    d_f = pd.read_csv(file_path, delimiter=',',dtype={"Soil":str})
    List = ['Temperature', 'Humidity', 'Rainfall']
    for col in List:
        d_f[col] = d_f[col].str.replace(',', '.', regex=True).astype(float)
    return d_f, d_f.head()

def nombre_Item(df):
    print("Nombre d' Item est :", df.shape[1])
    return df.shape[1]

def delete(df, supp_min, Nbr_transaction):
    for column_name in df.columns:
       if (df[column_name].sum()/Nbr_transaction) < supp_min:
           df = df.drop(column_name, axis=1)

    return df

def combinaison(data,df):
    list = []
    clone = {}
    columns = data.columns.tolist()
    for e, i in enumerate(columns[:-1], start=1):
        for j in columns[e:]:
            P = j.split("-")
            o=i.split("-")
            for k in range(len(P)):
                if P[k] not in o:
                    str=(i+"-"+P[k])
                    elements = str.split('-')
                    combinations = itertools.permutations(elements)
                    found = False
                    for combo in combinations:
                        combo_str = '-'.join(combo)
                        if any(combo_str in item for item in clone):
                            found = True
                            break 
                    if not found:
                        list.append(str)
                        clone[str] = data[i] & df[P[k]] 
                        
    clone = pd.DataFrame.from_dict(clone, orient='index')
    clone = clone.T
    data = pd.concat([data, clone], axis=1)
    
    data = data.drop(columns, axis=1)
    return  data

def Apriori(data, supp_min, Nbr_transaction,choix):
    data=pretraitement(data,choix)
    df=data
    L=[]
    data = delete(data, supp_min, Nbr_transaction)
    L.append(data.columns)
    while(not data.empty):
        data=combinaison(data,df)
        data = delete(data, supp_min, Nbr_transaction)
        if not data.empty:
            L.append(data.columns)
        
    L = [itemset.tolist() for itemset in L] 
    F_Patterns = [item for sublist in L for item in sublist]
    print(len(F_Patterns))
    return F_Patterns

def confiance(rules, Nbr_transaction):
    L=[]
    for i in range(len(rules)):
        sup_AuB = ((df[rules[i][0]] == 1) & (df[rules[i][1]] == 1)).sum()
        suppA = (df[rules[i][0]]==1).sum()
        d = sup_AuB/suppA
        L.append(d)

    print(L)
   
def dicreminisation_frequance(df, arg,att):
    num_elt=len(df)//arg
    sorted_values = df[att].sort_values().reset_index(drop=True)
    list=[]
    print(sorted_values.max())
    for i in range(arg):
        t=(sorted_values[i+num_elt*i],sorted_values[num_elt*(i+1)-1])
        list.append(t)
    list2=[]
    for i in range(len(list)-1):
        z=(list[i][0],list[i+1][0])
        list2.append(z)
    list2.append(list[-1])
    list=list2   
    for index, row in df.iterrows():
        value = row[att]
        for i, (inff, supp) in enumerate(list,start=1):
            if inff <= value <= supp:
                df.at[index, att] = att+str(i)
                break
    return df

def dicreminisation(df, att):
    att2=att+"2"
    for column in df.columns:
        st.write(column)
    num_intervals = int(1+(10/3)*m.log10(df[att].count()))
    interval_width = (df[att].max()-df[att].min())/int(num_intervals)
    min=df[att].min()
    intervals = [(i * interval_width+min, (i + 1) * interval_width+min)for i in range(num_intervals)]
    for index, row in df.iterrows():
        value = row[att]
        for i, (inff, supp) in enumerate(intervals,start=1):
            if inff <= value <= supp:
                df.at[index, att2] = att+str(i)
                break
    df = df.drop([att], axis=1)
    if att=="P":
        st.write(df[att2].dtype)
        st.write(df[att].dtype)
    if df[att2].dtype=="object":
        df[att] = df[att2].astype(str)
    else:
        df[att] = df[att2]
    df = df.drop([att2], axis=1)
    return df

def dicreminisation2(df,att,N_Classe=8):
    data=df[att]
    discretized_data = pd.cut(data, bins=N_Classe, labels=[att+str(i+1) for i in range(N_Classe)], include_lowest=True)
    df[att]=discretized_data
    return df

def pretraitement(df,choix):
    # df = df.drop(["Temperature", "Humidity","Rainfall"], axis=1)
    list=["Temperature", "Humidity","Rainfall"]
    list.remove(choix)
    df = df.drop(list, axis=1)
    df = df[[choix, 'Soil', 'Crop', 'Fertilizer']]
    list=[]
    for i in range(df[choix].nunique(dropna=True)):
        list.append(f"{i+1}")
    list=list + df['Fertilizer'].dropna().unique().tolist()+df['Crop'].dropna().unique().tolist()+df['Soil'].dropna().unique().tolist()
    data = pd.DataFrame(False, columns=list, index=range(df.shape[0]))
    for i in range(df.shape[0]):
        vals = df.loc[i]
        for k in range(len(vals)):
            nom_cat = str(vals[k])
            data.loc[i, nom_cat] = True
                   
    return data
       
def calcul_Corelation(df, item1, item2,item3):
    condition = df[item1].eq(1).all(axis=1)
    support_combined=condition.mean()
    condition2 = df[item2].eq(1).all(axis=1)
    support_item1=condition2.mean()
    condition3 = df[item3].eq(1).all(axis=1)
    support_item2=condition3.mean()
    lift = support_combined / (support_item1 * support_item2)
    return lift

def calcul_Confiance(df, item1, item2,):
    condition = df[item1].eq(1).all(axis=1)
    condition2 = df[item2].eq(1).all(axis=1)
    support_combined=condition.mean()
    support_item1=condition2.mean()
    confidence = support_combined / support_item1
    return confidence

def calcul_Levrage(df, item1, item2,item3):
    condition = df[item1].eq(1).all(axis=1)
    support_combined=condition.mean()
    condition2 = df[item2].eq(1).all(axis=1)
    support_item1=condition2.mean()
    condition3 = df[item3].eq(1).all(axis=1)
    support_item2=condition3.mean()
    Levrage=support_combined-(support_item1 * support_item2)
    
    return Levrage

def calculate_cosine_similarity(data, item1, item2):
    item1_vector = data[item1].values.reshape(1, -1)
    item2_vector = data[item2].values.reshape(1, -1)
    cosine_sim = cosine_similarity(item1_vector, item2_vector)
    return cosine_sim[0][0]

def calcul_conviction(df, item2, item3):
    support_combined = ((df[item2] == True).all(axis=1) & (df[item3] == False).all(axis=1)).sum()
    support_combined=support_combined/df.shape[0]
    condition_item1 = df[item2].eq(1).all(axis=1)
    support_item1 = condition_item1.mean()
    condition_item2 = df[item3].eq(1).all(axis=1)
    support_item2 = condition_item2.mean()
    conviction_value = (support_item1 * (1 - support_item2)) / (support_combined)
    return conviction_value

def calcul_zhang(df, item1, item2,item3):
    condition = df[item1].eq(1).all(axis=1)
    support_combined=condition.mean()
    condition2 = df[item2].eq(1).all(axis=1)
    support_item1=condition2.mean()
    condition3 = df[item3].eq(1).all(axis=1)
    support_item2=condition3.mean()
    x=support_combined-(support_item1 * support_item2)
    zhang=x/max(x,support_item1*support_item2-support_combined)
     
    return zhang

def Get_Rules(data,F_Patterns,seuil,choix,sort="Corelation"):
    rule=set()
    df=pretraitement(data,choix)
    colonnes = ["Regle d'association", "Corelation", "Confiance","Conviction","Levrage","Zhang"]
    Rules = pd.DataFrame(columns=colonnes)
    List_regles=[]
    combinations_finale=[]
    for i in range(len(F_Patterns)):
        elt = F_Patterns[i].split("-")
        if(len(elt)>1): 
            combinations_list = []
            for r in range(1, len(elt) + 1):
                for combo in combinations(elt, r):
                    remaining = [e for e in elt if e not in combo]
                    if len(remaining) > 0:
                        combinations_list.append((list(combo), remaining))
                        combinations_finale.append((list(combo), remaining))
            for paire in combinations_list:
                x=[]
                x.extend(paire[0])
                x.extend(paire[1])
                y=paire[0]
                z=paire[1]
                confiance=calcul_Confiance(df,x,y)
                corelation=calcul_Corelation(df,x,y,z)
                Levrage=calcul_Levrage(df,x,y,z)
                conviction=calcul_conviction(df,y,z)
                zhang=calcul_zhang(df,x,y,z)
                # sim=calculate_cosine_similarity(df,y,z)
                name=f"Si {paire[0]} Alors {paire[1]}."
                Rules.loc[len(Rules)] = [name,corelation,confiance,conviction,Levrage,zhang]
    Rules = Rules[Rules['Confiance'] >= seuil]                
    return Rules, combinations_finale

def predict(DATA,choix,rules,arr1,arr2,arr3,arr4):
    data = rules
    df=pretraitement(DATA,choix)
    user_inputt =arr2+arr3+arr4+arr1
    resullts = [tup for tup in data if len(set(user_inputt).intersection(tup[0])) >= 2]
    colonnes = ["Regle d'association","Confiance"]
    Rules = pd.DataFrame(columns=colonnes)
    for paire in resullts:
        x=[]
        x.extend(paire[0])
        x.extend(paire[1])
        y=paire[0]
        z=paire[1]
        confiance=calcul_Confiance(df,x,y)
        name=f"Si {paire[0]} Alors {paire[1]}."
        Rules.loc[len(Rules)] = [name,confiance]
    Rules = Rules.sort_values(by='Confiance',ascending=False)
    return Rules
      
    