from app import app
from collections import OrderedDict
import json
import pickle
from pprint import pprint, pformat

from flask import Flask, render_template, request, jsonify, session, redirect
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from io import StringIO
import csv
from flask import make_response

from flask_wtf import Form
from wtforms import StringField, SubmitField

from db_setup import init_db, db_session, engine
from models import top_by_cat2, purchases
import sqlalchemy
from sqlalchemy.orm import sessionmaker

init_db()



def filter_by_order_count(df):
    df=df[df['commande']!='suggestion']
    customer_num_orders = df.groupby(['client', 'commande']).size().reset_index().groupby(['client']).size()

    # how many products each customer ordered
    #customer_num_products = df.groupby(['client', 'code_article']).size().reset_index().groupby(['client']).size()

    #min_products, max_products = 0, 20
    min_orders, max_orders = 1, 4

    # customers_minmax_products = customer_num_products[(customer_num_products >= min_products) &
    #                                                   (customer_num_products <= max_products)].index
    customers_minmax_orders = customer_num_orders[(customer_num_orders >= min_orders) &
                                                  (customer_num_orders <= max_orders)].index

    customers = set(customers_minmax_orders) #& set(customers_minmax_products)

    return df[df['client'].isin(customers)]




class Recommender(object):

    def __init__(self):
        qry = db_session.query(purchases)
        df = pd.read_sql(qry.statement, qry.session.bind)                                     

        df = filter_by_order_count(df)

        # add per-customer counts for category2 and model_id
        df['qty']=df.groupby(['client'])['model_id'].transform('size')
        ndf=df.groupby(['client','model_id'])['qty'].sum().reset_index()
        self.df = df
        self.ndf=ndf

        self.client_ids = list(np.sort(ndf.client.unique()))

        items = ndf.pivot(index = 'model_id', columns = 'client', values = 'qty').fillna(0)
        self.items=items
        item_rows=csr_matrix(items.values)
        self.item_rows=item_rows

        model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
        model_knn.fit(item_rows)
        self.model_knn = model_knn

    def random_client_id(self):
        return np.random.choice(self.client_ids)

    def get_history(self, client_id):
        return self.df[self.df['client'] == client_id]

    def get_model_ids(self, model_ids):
        columns = [
            'CAT1',
            'category1',
            'famille_intitule',

            'category2',
            'sous_famille_intitule',

            'modele_intitule',
            'model_id',
            #'code_article'
        ]

        fdf = self.df[self.df['model_id'].isin(model_ids)][columns].drop_duplicates()
        fdf['order'] = fdf.apply(lambda row: model_ids.index(row['model_id']), axis=1)
        fdf.sort_values('order', inplace=True)
        # grouped = self.df.groupby(['CAT1', 'model_id'])['model_id'].agg({"code_count": len}).sort_values("code_count", ascending=False).reset_index()
        # grouped = grouped.groupby('CAT1').head(5).reset_index().sort_values('CAT1', ascending=True)
        # grouped = grouped.groupby(['CAT1'])['model_id'].apply(list).reset_index()
        # new = pd.merge(fdf, grouped, on = 'CAT1', how='left')
        return fdf

    def update_history(self, client_id, model_id):
        #model_id1=1
        self.df=self.df[['famille_intitule','sous_famille_intitule','modele_intitule','CAT1','category1','category2','model_id','code_article','client','date_commande_client','qty','commande']]

        lookup=self.df[['famille_intitule','sous_famille_intitule','modele_intitule','CAT1','category1','category2','model_id','code_article']]
        lookup.drop_duplicates(inplace=True)

        #today=pd.to_datetime('today')

        today=pd.to_datetime('today').strftime('%Y-%m-%d')

        model_id_lst=[]
        if model_id:
            model_id_lst.append(model_id)
        # if model_id1:
        #     model_id_lst.append(model_id1)
        # if model_id2:
        #     model_id_lst.append(model_id2)

        temp=lookup[lookup['model_id'].isin(model_id_lst)]
        temp=temp.values.tolist()
        for i in range(len(model_id_lst)):
        	try:
        		temp[i].extend([client_id, today, 50, '2018'])
        	except IndexError:
        		print ('Model ID does not exist in database')
        #temp[0].extend([client_id, today, 50, 'suggestion'])
        #temp[1].extend([client_id, today, 50, 'suggestion'])
        tempdf=pd.DataFrame(temp, columns = list(self.df))
        self.df = self.df.append(tempdf)
        self.df= self.df.drop_duplicates()

        return self.df

    def update_history2(self, client_id, model_id):
        #model_id1=1
        self.df=self.df[['famille_intitule','sous_famille_intitule','modele_intitule','CAT1','category1','category2','model_id','client','date_commande_client','qty','commande']]

        lookup=self.df[['famille_intitule','sous_famille_intitule','modele_intitule','CAT1','category1','category2','model_id',]]
        lookup.drop_duplicates(inplace=True)

        #today=pd.to_datetime('today')

        today=pd.to_datetime('today').strftime('%Y-%m-%d')

        model_id_lst=[]
        if model_id:
            model_id_lst.append(model_id)
        # if model_id1:
        #     model_id_lst.append(model_id1)
        # if model_id2:
        #     model_id_lst.append(model_id2)

        temp=lookup[lookup['model_id'].isin(model_id_lst)]
        temp=temp.values.tolist()
        for i in range(len(model_id_lst)):
        	try:
        		temp[i].extend([client_id, today, 50, 'suggestion'])
        	except IndexError:
        		print ('Model ID does not exist in database')
        #temp[0].extend([client_id, today, 50, 'suggestion'])
        #temp[1].extend([client_id, today, 50, 'suggestion'])
        tempdf=pd.DataFrame(temp, columns = list(self.df))
        tempdf.drop_duplicates(inplace=True)
        self.tempdf=tempdf
        # self.df = self.df.append(tempdf)
        

        return self.tempdf


    def get_list(self):

    	grouped = self.df.groupby(['CAT1', 'model_id'])['model_id'].agg(
		    {"code_count": len}).sort_values("code_count", ascending=False).reset_index()
    	grouped = grouped.groupby('CAT1').head(5).reset_index().sort_values('CAT1', ascending=True)
    	#grouped = grouped.groupby('CAT1')['model_id'].apply(list)

    	return grouped

    # def get_csv(self):
    # 	self.df.dropna(inplace=True)
    # 	return self.df

    def recommend(self, client_id):

        recommendations = {}
        temp=self.ndf[self.ndf.client==str(client_id)]
        self.temp=temp
        client_items=temp.sort_values('qty')['model_id']
        self.client_items=client_items

        # find nearest neighbours for category2
        for val in client_items.values:
            distances, indices = self.model_knn.kneighbors(
                self.items.loc[str(val), :].values.reshape(1, -1), n_neighbors = 10)

        nearest_item_list = [
            self.items.index[indices.flatten()[i]]
            for i in range(0, len(distances.flatten()))
        ]

        grouped = (self.df[self.df['model_id'].isin(nearest_item_list)]
                    .groupby(['category1','model_id'])
                    .agg({'qty':sum}))
        #grouped = grouped[~grouped.model_id.isin(client_items.values)]
        grouped = grouped['qty'].groupby(level=0, group_keys=False)
        grouped = list(grouped.nlargest(1)[0:5].items())
        grouped.sort(key=lambda tup: tup[1], reverse=True)


        recommendations.update({
            'category1': grouped
        })


        recommendations.update({
            'model_id': grouped
        })
        return recommendations






recommender = Recommender()

@app.route('/test', methods=['GET', 'POST'])
def make_test():
    qry = db_session.query(top_by_cat2)
    df = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df['modele_intitule'].unique():
        temp=df[df['modele_intitule']==x]['model_id'].unique()
    #print(x)
        list2.append((x, temp))
    df2=df[['CAT1','modele_intitule']].drop_duplicates()
    df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df.sort_values(by=['famille_intitule'])
    df4=df4.groupby(['CAT1','famille_intitule'])['modele_intitule'].apply(list).reset_index()
    
    list1 = ['A']#, 'A', 'Dining Room', 'Bedroom', 'O']

    

    #results = [item[0] for item in qry.all()]
    return render_template('test2.html', df=df, list1=list1, list2=list2, df4=df4, df2=df2, df3=df3)

@app.route('/', methods=['GET','POST'])

def index():
	return render_template('index.html')
	#, client_id=session['client_id'])

# @app.route('/random_client')
# def random_user():
# 	#list_dropdown = recommender.get_list()
# 	client_id = recommender.random_client_id()
# 	recommendations = recommender.recommend(client_id)
# 	recommended_model_ids = [m for (c, m), cnt in recommendations['model_id']]
# 	df_reco_models = recommender.get_model_ids(recommended_model_ids)
# 	if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
# 		df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
# 	else:
# 		df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]

# 	df_history = recommender.get_history(client_id)
# 	columns= [
#         ('date_commande_client', 'ORDER DATE'),
#         ('famille_intitule', 'FAMILY'),
#         ('sous_famille_intitule', 'SUB-FAMILY'),
#         ('modele_intitule', 'MODEL')]
# 	return render_template('index_reco7.html',
#         client_id=client_id,
#         history_df=df_history,
#         history_columns=columns,
#         recommendations=pformat(recommendations),
#         recomm_df=df_reco_models.groupby('CAT1').head(1)
#         #list_dropdown = list_dropdown
#         )

@app.route('/random_client', methods=["GET","POST"])
def random(): 
    qry = db_session.query(top_by_cat2)
    df = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df['modele_intitule'].unique():
        temp=df[df['modele_intitule']==x]['model_id'].unique()
        list2.append((x, temp))
    df2=df[['CAT1','modele_intitule']].drop_duplicates()
    df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df.sort_values(by=['famille_intitule'])
    df4=df4.groupby(['CAT1','famille_intitule'])['modele_intitule'].apply(list).reset_index()
    
    list1 = ['A']
    client_id = recommender.random_client_id()
    recommendations = recommender.recommend(session['client_id'])
    recommended_model_ids = [m for (c, m), cnt in recommendations['model_id']]
    df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    df_history = recommender.get_history(session['client_id'])
    columns= [
        ('date_commande_client', 'ORDER DATE'),
        ('famille_intitule', 'FAMILY'),
        ('sous_famille_intitule', 'SUB-FAMILY'),
        ('modele_intitule', 'MODEL'),
        ('model_id', 'MODEL ID')
    ]

    return render_template('reco8.html',
        client_id=session['client_id'],
        history_df=df_history,
        history_columns=columns,
        recommendations=pformat(recommendations),
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df=df, list1=list1, list2=list2, df4=df4, df2=df2, df3=df3
        )



@app.route('/suggestion', methods=["GET","POST"])
def suggestion(): 
    qry = db_session.query(top_by_cat2)
    df = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df['modele_intitule'].unique():
        temp=df[df['modele_intitule']==x]['model_id'].unique()
        list2.append((x, temp))
    df2=df[['CAT1','modele_intitule']].drop_duplicates()
    df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df.sort_values(by=['famille_intitule'])
    df4=df4.groupby(['CAT1','famille_intitule'])['modele_intitule'].apply(list).reset_index()
    
    list1 = ['A']
    session['client_id']=request.args['query']
    recommendations = recommender.recommend(session['client_id'])
    recommended_model_ids = [m for (c, m), cnt in recommendations['model_id']]
    df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    df_history = recommender.get_history(session['client_id'])
    columns= [
        ('date_commande_client', 'ORDER DATE'),
        ('famille_intitule', 'FAMILY'),
        ('sous_famille_intitule', 'SUB-FAMILY'),
        ('modele_intitule', 'MODEL'),
        ('model_id', 'MODEL ID')
    ]

    return render_template('index_reco6.html',
        client_id=session['client_id'],
        history_df=df_history,
        history_columns=columns,
        recommendations=pformat(recommendations),
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df=df, list1=list1, list2=list2, df4=df4, df2=df2, df3=df3
        )

@app.route('/suggest_other', methods=["GET","POST"])
def suggest_other(): 
    qry = db_session.query(top_by_cat2)
    df = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df['modele_intitule'].unique():
        temp=df[df['modele_intitule']==x]['model_id'].unique()
        list2.append((x, temp))
    df2=df[['CAT1','modele_intitule']].drop_duplicates()
    df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df.sort_values(by=['famille_intitule'])
    df4=df4.groupby(['CAT1','famille_intitule'])['modele_intitule'].apply(list).reset_index()
    
    list1 = ['A']
    session['client_id']=request.args['query']
    recommendations = recommender.recommend(session['client_id'])
    recommended_model_ids = [m for (c, m), cnt in recommendations['model_id']]
    df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    df_history = recommender.get_history(session['client_id'])
    columns= [
        ('date_commande_client', 'ORDER DATE'),
        ('famille_intitule', 'FAMILY'),
        ('sous_famille_intitule', 'SUB-FAMILY'),
        ('modele_intitule', 'MODEL'),
        ('model_id', 'MODEL ID')
    ]

    return render_template('index_reco7.html',
        client_id=session['client_id'],
        history_df=df_history,
        history_columns=columns,
        recommendations=pformat(recommendations),
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df=df, list1=list1, list2=list2, df4=df4, df2=df2, df3=df3
        )



# @app.route('/db_add',methods=['GET','POST'])
# def get_data():
   
#     model_id=request.args['item']
#     client_id = request.args['query']

#     df=recommender.update_history2(client_id, model_id)
#     listToWrite = df.to_dict(orient='records')

#     Session = sessionmaker(bind=engine)
#     session = Session()
#     metadata = sqlalchemy.schema.MetaData(bind=engine,reflect=True)
#     table = sqlalchemy.Table("inbox_table", metadata, autoload=True)

# # Inser the dataframe into the database in one bulk
#     conn = engine.connect()
#     conn.execute(table.insert(), listToWrite)

# # Commit the changes
#     session.commit()

# # Close the session
#     session.close()
#     return render_template("index.html")

@app.route('/db_add2',methods=['GET','POST'])
def get_data2():
    model_id=request.args['item']
    client_id = request.args['query']

    df=recommender.update_history2(client_id, model_id)
    listToWrite = df.to_dict(orient='records')

    Session = sessionmaker(bind=engine)
    session = Session()
    metadata = sqlalchemy.schema.MetaData(bind=engine,reflect=True)
    table = sqlalchemy.Table("inbox_table", metadata, autoload=True)

# Inser the dataframe into the database in one bulk
    conn = engine.connect()
    conn.execute(table.insert(), listToWrite)

# Commit the changes
    session.commit()

# Close the session
    session.close()

#get the other stuff to repopulate the page

    qry = db_session.query(top_by_cat2)
    df = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df['modele_intitule'].unique():
        temp=df[df['modele_intitule']==x]['model_id'].unique()
        list2.append((x, temp))
    df2=df[['CAT1','modele_intitule']].drop_duplicates()
    df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df.sort_values(by=['famille_intitule'])
    df4=df4.groupby(['CAT1','famille_intitule'])['modele_intitule'].apply(list).reset_index()
    
    list1 = ['A']
    #client_id = request.args['query']
    recommendations = recommender.recommend(client_id)
    recommended_model_ids = [m for (c, m), cnt in recommendations['model_id']]
    df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    df_history = recommender.get_history(client_id)
    columns= [
        ('date_commande_client', 'ORDER DATE'),
        ('famille_intitule', 'FAMILY'),
        ('sous_famille_intitule', 'SUB-FAMILY'),
        ('modele_intitule', 'MODEL'),
        ('model_id', 'MODEL ID')
    ]

    return render_template('reco8.html',
        client_id=client_id,
        history_df=df_history,
        history_columns=columns,
        recommendations=pformat(recommendations),
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df=df, list1=list1, list2=list2, df4=df4, df2=df2, df3=df3
        )



@app.route('/db_add3',methods=['GET','POST'])
def get_data3():
   
    model_id=request.args['item']
    client_id = request.args['query']

    df=recommender.update_history2(client_id, model_id)
    listToWrite = df.to_dict(orient='records')

    Session = sessionmaker(bind=engine)
    session = Session()
    metadata = sqlalchemy.schema.MetaData(bind=engine,reflect=True)
    table = sqlalchemy.Table("inbox_table", metadata, autoload=True)

# Inser the dataframe into the database in one bulk
    conn = engine.connect()
    conn.execute(table.insert(), listToWrite)

# Commit the changes
    session.commit()

# Close the session
    session.close()

#get the other stuff to repopulate the page

    qry = db_session.query(top_by_cat2)
    df = pd.read_sql(qry.statement, qry.session.bind)
    list2=[]
    for x in df['modele_intitule'].unique():
        temp=df[df['modele_intitule']==x]['model_id'].unique()
        list2.append((x, temp))
    df2=df[['CAT1','modele_intitule']].drop_duplicates()
    df3=df[['CAT1','sous_famille_intitule','modele_intitule']].drop_duplicates()
    df4=df.sort_values(by=['famille_intitule'])
    df4=df4.groupby(['CAT1','famille_intitule'])['modele_intitule'].apply(list).reset_index()
    
    list1 = ['A']
    #client_id = request.args['query']
    recommendations = recommender.recommend(client_id)
    recommended_model_ids = [m for (c, m), cnt in recommendations['model_id']]
    df_reco_models = recommender.get_model_ids(recommended_model_ids)
    if len(df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]['CAT1'].unique()) == 3:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B'])]
    else:
        df_reco_models=df_reco_models[df_reco_models['category1'].isin(['L','D','B', 'A'])]
    df_history = recommender.get_history(client_id)
    columns= [
        ('date_commande_client', 'ORDER DATE'),
        ('famille_intitule', 'FAMILY'),
        ('sous_famille_intitule', 'SUB-FAMILY'),
        ('modele_intitule', 'MODEL'),
        ('model_id', 'MODEL ID')
    ]

    return render_template('reco8.html',
        client_id=client_id,
        history_df=df_history,
        history_columns=columns,
        recommendations=pformat(recommendations),
        recomm_df=df_reco_models.groupby('CAT1').head(1),
        df=df, list1=list1, list2=list2, df4=df4, df2=df2, df3=df3
        )







if __name__ == '__main__':
    app.run(debug=True)