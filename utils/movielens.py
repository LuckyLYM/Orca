from time import time
import os 
from datetime import datetime
import torch
import pickle
import torch.nn as nn
import sys
from collections import defaultdict
from numba import jit
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import coo_matrix
import os
import pickle
import sys



def data_profile(edges):

    # * Get all the edges happens in a time interval
	idx = edges['idx']
	time_column=idx[:,ECOLS.time]
	pairs = idx[:,[ECOLS.source, ECOLS.target]]

	unique_time=torch.unique(time_column)
	n_snapshot=len(unique_time)
	n_edges=time_column.shape[0]

	unique_nodes=torch.unique(pairs)
	n_nodes=len(unique_nodes)
	print(f'timestamps {unique_time}')
	print(f'#snapshots {n_snapshot} #edges {n_edges} #nodes {n_nodes}')


	for time in unique_time:
		subset = time_column == time
		pairs = idx[subset][:,[ECOLS.source, ECOLS.target]]
		n_edges=pairs.shape[0]
		unique_nodes=torch.unique(pairs)
		n_nodes=len(unique_nodes)
		print(f'time {time} #edges {n_edges} #nodes {n_nodes}')

	sys.exit(1)



'''
movielens_args:
  folder: ./data/movielens/ml-20
  edges_file: ratings.csv
  nodes_file: movies.csv
  aggr_time: 7 #number of days
'''

class Movielens_Dataset():
	def __init__(self,args):
		folder='./ml-20'
		edges_file='ratings.csv'
		nodes_file='movies.csv'


		cached_file= os.path.join(folder,'movielens')



		self.n_user=138493
		self.n_item=27278
		self.num_nodes=self.n_user+self.n_item

		if os.path.exists(cached_file):

			f=open(cached_file,'rb')
			data=pickle.load(f)
			f.close()

			self.edges=data['edges']
			self.max_time=data['max_time']
			self.min_time=data['min_time']
			self.num_classes=data['num_classes']
			self.num_nodes=data['num_nodes']
			self.feats_per_node = data['feats_per_node']
			self.nodes_feats = data['nodes_feats']
			print('load pickle file')
		else:
			
			# -------- better set feature per node to 172...
			# id_dict is the mapping for movieid
			feats,id_dict=self.load_nodes(nodes_file,folder,172)
			edges=self.load_edges(edges_file,folder,id_dict)

			time_colum=edges[:,2]
			mask = time_colum >= 180
			edges=edges[mask]
			# !filter out the first snapshot here

			max_time_before= edges[:,2].max()
			edges[:,2] = u.aggregate_by_time(edges[:,2],args.movielens_args.aggr_time)
			max_time = edges[:,2].max()
			num_nodes= feats.shape[0]

			# max_before 7385  max_after 82
			print(f'max_before {max_time_before}  max_after {max_time}')
			#edges = self.make_undirected(edges)  # 40m edges after making undirected
			# TODO: we make it undirected when building the adj matrix...


			sp_indices = edges[:,:3].t()
			sp_values = edges[:,3]

			sp_edges = torch.sparse.LongTensor(sp_indices,sp_values,torch.Size([num_nodes,num_nodes,max_time+1])).coalesce()
			labels = sp_edges._values()

			new_labels = torch.zeros(labels.size(0),dtype=torch.long)
			new_labels[labels>3] = 1	# like
			new_labels[labels<=3] = 0	# dis like


			# TODO: change it to a two-classification task >=3 and below 3

			# * print(labels[0].dtype)  # int64
			# * shape [n,4]

			indices_labels = torch.cat([sp_edges._indices().t(),new_labels.view(-1,1)],dim=1)
			weight = torch.ones(edges.size(0),dtype=torch.long)
			self.edges = {'idx': indices_labels, 'vals': weight}
			self.num_classes = 2
			self.feats_per_node = feats.size(1)
			self.num_nodes = num_nodes
			self.nodes_feats = feats
			self.max_time = max_time
			self.min_time = 0


			# edges format idx: (user,item,time,lable) val: weight 1

			f=open(cached_file,'wb')
			data=dict()
			data['edges']=self.edges
			data['max_time']=self.max_time
			data['min_time']=self.min_time
			data['num_classes']=self.num_classes
			data['num_nodes']=self.num_nodes
			data['feats_per_node']=self.feats_per_node
			data['nodes_feats']=self.nodes_feats
			print('load pickle file')

			pickle.dump(data,f)
			f.close()
			print('file saved! ',cached_file)
			




	











	def load_edges(self,edges_file,folder,id_dict):


		cached_file= os.path.join(folder,'edges')

		if os.path.exists(cached_file):
			f=open(cached_file,'rb')
			edges=pickle.load(f)
			f.close()
		else:
			

			# file format: userId,movieId,rating,timestamp
			edges = []
			file = edges_file
			file = os.path.join(folder,file)

			with open(file) as file:
				file = file.read().splitlines()
				
			cols = u.Namespace({'user': 0,
								'item': 1,
								'rating': 2,
								'time': 3})

			for index,line in enumerate(file[1:]):
				if index%10000==0:
					print(f'line {index}')

				fields = line.split(',')
				user = int(fields[cols.user])-1
				item = int(fields[cols.item])

				# TODO: convert it to a binary classification
				item= id_dict[item]+138493	
				rating = int(float(fields[cols.rating])*2)  # 0.5-5 => 1-10
				rating = (rating+1)//2	# 1-5


				time = int(fields[cols.time])
				edges.append([user,item,time,rating])

				# make it undirected here
				# edges.append([user,item,time,rating])

			edges = torch.LongTensor(edges)
			
			# TODO: fix the time aggregation here. besides, we need to sample some subsets
			edges[:,2] = u.aggregate_by_time(edges[:,2],3600*24)

			p=torch.argsort(edges[:,2])
			edges=edges[p]

			n_user = torch.max(edges[:,0])
			n_item = torch.max(edges[:,1])
			max_time=torch.max(edges[:,2])
			labels=torch.unique(edges[:,3])


			f=open(cached_file,'wb')
			pickle.dump(edges,f)
			f.close()
			print('file saved! ',cached_file)

			print(f'n_user {n_user} n_item {n_item} max_time {max_time} labels {labels}')
		return edges
	






	# 19964833 edges in total...
	def load_nodes(self,nodes_file,folder,feats_per_node):

		cached_file= os.path.join(folder,'nodes')
		if os.path.exists(cached_file):
			f=open(cached_file,'rb')
			data=pickle.load(f)
			f.close()
			feat=data['features']
			id_dict=data['map']
		else:
			feat = []
			id_dict = {}
			movie_counter =0
			genres_dict ={}
			genres_counter = 0
			genre_list=[]
			file = nodes_file
			file = os.path.join(folder,file)

			# -------- read file
			# file: movies.csv
			# format: movieid, title, genres
			with open(file) as file:
				file = file.read().splitlines()
				
			for index,line in enumerate(file[1:]):
				if index%10000==0:
					print(f'line {index}')
				fields = line.split(',')
				genres = fields[-1].split('|')
				movie_id = int(fields[0])
				id_dict[movie_id]=movie_counter
				movie_counter+=1
				l=[]
				for genre in genres:
					if genre not in genres_dict.keys():
						genre_id=genres_counter
						genres_dict[genre]=genre_id
						genres_counter+=1
					genre_id=genres_dict[genre]
					l.append(genre_id)
				genre_list.append(l)

			# process user embeds
			#user_embeds=torch.empty(138493,20)
			#user_embeds=nn.init.xavier_normal_(user_embeds, gain=1.414)


			# -------- genere embedding for movies
			item_embeds=torch.zeros(27278,20)
			for index,l in enumerate(genre_list):
				for id in l:
					item_embeds[index,id]=1

			# -------- empty embedding for users
			user_embeds=torch.empty(138493,20)
			feat=torch.cat((user_embeds,item_embeds),0)

			# -------- one-hot embedding for both users and movies
			num_nodes=27278+138493
			one_hot=u.one_hot(num_nodes,feats_per_node-20)
			feat=torch.cat((feat,one_hot),1)

			h=feat.size(0)
			w=feat.size(1)

			print(f'n_movies {len(genre_list)}  n_genres {genres_counter}  h {h}  w {w}')

			data={}
			data['features']=feat
			data['map']=id_dict

			f=open(cached_file,'wb')
			pickle.dump(data,f)
			f.close()
			print('file saved! ',cached_file)

		return feat,id_dict

















































# ----------------------------- New Code Version ----------------------------


@jit(nopython=True)
def filter_by_bitmap_numba(user,user_bitmap, item, item_bitmap):
	user_mask = np.array([index for index,value in enumerate(user) if user_bitmap[value]!=0])
	item_mask = np.array([index for index,value in enumerate(item) if item_bitmap[value]!=0])
	#ind=np.intersect1d(user_mask,item_mask)
	return user_mask,item_mask


@jit(nopython=True)
def filter_by_bitmap_numba2(user,user_bitmap):
	user_mask = np.array([index for index,value in enumerate(user) if user_bitmap[value]!=0])
	return user_mask


# file format: userId,movieId,rating,timestamp
def sample_edges():

	core=10
	cached_file= os.path.join('ml-20m','edges_'+str(core)+'core')
	f=open(cached_file,'rb')
	d=pickle.load(f)
	f.close()

	user = d['user']
	item = d['item']
	rating = d['rating']
	time = d['time']


	# ---------- get the sampled users here...
	unique_user=np.unique(user)
	print(len(unique_user))
	user_size=np.max(unique_user)+1
	user_bitmap=np.zeros(user_size)

	n_user=30000
	np.random.shuffle(unique_user)
	sampled_user=unique_user[:n_user]
	user_bitmap[sampled_user]=1

	# ---------- get the sampled items here...
	unique_item=np.unique(item)
	print(len(unique_item))
	item_size=np.max(unique_item)+1
	item_bitmap=np.zeros(item_size)

	n_item=10000
	np.random.shuffle(unique_item)
	sampled_item=unique_item[:n_item]
	print(sampled_item)
	item_bitmap[sampled_item]=1

	user_mask,item_mask = filter_by_bitmap_numba(user,user_bitmap, item, item_bitmap)
	ind=np.intersect1d(user_mask,item_mask)	

	print(f'sampled edges {len(ind)}')
	#print('file saved! ',cached_file)

	# 2833165 edge till now

	# we can sample the latest 500k edges
	n_edge=500000
	ind=ind[-n_edge:]
	sampled_time=time[ind]

	maxt=np.max(sampled_time)
	mint=np.min(sampled_time)
	
	print(f'min {mint}  max {maxt}  duration {(maxt-mint)/60/60/24}')
	print(len(ind))



def sample_edges_by_time():

	core=10
	cached_file= os.path.join('ml-20m','edges_'+str(core)+'core')


	f=open(cached_file,'rb')
	d=pickle.load(f)
	f.close()

	user = d['user']
	item = d['item']
	rating = d['rating']
	time = d['time']


	p=np.argsort(time)
	user = user[p]
	item = item[p]
	rating = rating[p]
	time = time[p]

	k=500000
	sampled_time=time[-k:]
	sampled_user=user[-k:]
	sampled_item=item[-k:]
	sampled_rating=rating[-k:]

	d=dict()
	d['user'] = sampled_user
	d['item'] = sampled_item
	d['rating'] = sampled_rating
	d['time'] = sampled_time

	maxt=np.max(sampled_time)
	mint=np.min(sampled_time)
	unique_user=np.unique(sampled_user)
	unique_item=np.unique(sampled_item)
	
	print(f'min {mint}  max {maxt}  duration {(maxt-mint)/60/60/24}')
	print(f'user {len(unique_user)}  item {len(unique_item)}')

	cached_file= os.path.join('ml-20m','edges_'+str(core)+'core_500k')
	f=open(cached_file,'wb')
	pickle.dump(d,f)
	f.close()
	print('file saved')


# file format: userId,movieId,rating,timestamp
def filter_edges():

	core=10
	cached_file= os.path.join('ml-20m','edges_'+str(core)+'core')
	#f=open(cached_file,'rb')


	if os.path.exists(cached_file):

		f=open(cached_file,'rb')
		edges=pickle.load(f)
		f.close()

	else:
		
		edges = []
		user = []
		item = []
		rat = []
		t = []

		
		file = os.path.join('ml-20m','ratings.csv')
		with open(file) as file:
			file = file.read().splitlines()
			
		user_counter = defaultdict(lambda: 0)
		item_counter = defaultdict(lambda: 0)

		for index,line in enumerate(file[1:]):
			if index%100000==0:
				print(f'line {index}')
	
			fields = line.split(',')
			userid = int(fields[0])
			itemid = int(fields[1])
			user_counter[userid]+=1
			item_counter[itemid]+=1

			rating = int(float(fields[2]))
			time = int(fields[3])

			user.append(userid)
			item.append(itemid)
			rat.append(rating)
			t.append(time)

		user=np.array(user)
		item=np.array(item)
		rat=np.array(rat)
		t=np.array(t)

		#edges = torch.LongTensor(edges)

		core_user=[]
		core_item=[]
		user_size=max(user_counter.keys())+1
		item_size=max(item_counter.keys())+1

		# start counter here
		for key,value in user_counter.items():
			if value>=core:
				core_user.append(key)

		for key,value in item_counter.items():
			if value>=core:
				core_item.append(key)


		user_bitmap=np.zeros(user_size)
		item_bitmap=np.zeros(item_size)
		user_bitmap[core_user]=1
		item_bitmap[core_item]=1

		print(f'core_user {len(core_user)}, core_item {len(core_item)}')


		user_mask,item_mask = filter_by_bitmap_numba(user,user_bitmap, item, item_bitmap)
		ind=np.intersect1d(user_mask,item_mask)

		user=user[ind]
		item=item[ind]
		rat=rat[ind]
		t=t[ind]
		
		d=dict()
		d['user']=user
		d['item']=item
		d['rating']=rat
		d['time']=t

		print(f'filtered edge {len(ind)}')
		f=open(cached_file,'wb')
		pickle.dump(d,f)
		f.close()

		print(f'{len(user)}  {len(item)}  {len(rat)}  {len(t)}')

		print('file saved! ',cached_file)










def one_hot_embed(n,dimension):
    data=np.ones(n)
    ind=np.arange(n)
    X=coo_matrix((data,(ind,ind)))
    svd = TruncatedSVD(n_components=dimension, n_iter=7, random_state=42)
    one_hot=svd.fit_transform(X)
    one_hot=torch.from_numpy(one_hot).float()
    return one_hot




def process_data():
	folder='./ml-20m/'
	suffix='10core_500k'
	edges_file='edges_'+suffix
	nodes_file='movies.csv'
	cached_file= 'movielens_'+suffix
	feature_dim=172


	# ------------ load edges here -------------
	print('load edges')
	f=open(folder+edges_file,'rb')
	d=pickle.load(f)
	f.close()
	user = d['user']
	item = d['item']
	rating = d['rating']
	time = d['time']
	unique_user=np.unique(user)
	unique_item=np.unique(item)
	n_user=len(unique_user)
	n_item=len(unique_item)

	print(f'unique_user {n_user} unique_item {n_item}')


	# ------------ load nodes here ------------
	print('load nodes')
	id_dict = {}
	movie_counter =0
	genres_dict ={}
	genres_counter = 0
	genre_list=[]

	file = nodes_file
	file = os.path.join(folder,file)
	with open(file) as file:
		file = file.read().splitlines()
		
	for index,line in enumerate(file[1:]):
		if index%10000==0:
			print(f'line {index}')
		fields = line.split(',')
		genres = fields[-1].split('|')
		movie_id = int(fields[0])

		# check existence of movies
		if movie_id in unique_item:
			id_dict[movie_id]=movie_counter
			movie_counter+=1
			l=[]
			for genre in genres:
				if genre not in genres_dict.keys():
					genre_id=genres_counter
					genres_dict[genre]=genre_id
					genres_counter+=1
				genre_id=genres_dict[genre]
				l.append(genre_id)
			genre_list.append(l)


	# ----------- process embeddings here ----------
	print('start embedding process')
	feat = []

	n_genre=len(genres_dict.keys())
	item_embeds=torch.zeros(n_item,n_genre)
	for index,l in enumerate(genre_list):
		for id in l:
			item_embeds[index,id]=1

	user_embeds=torch.empty(n_user,n_genre)
	feat=torch.cat((user_embeds,item_embeds),0)

	# one-hot embedding for both users and movies
	num_nodes=n_item+n_user
	one_hot=one_hot_embed(num_nodes,feature_dim-n_genre)
	feat=torch.cat((feat,one_hot),1)
	h=feat.size(0)
	w=feat.size(1)
	print(f'n_movies {len(genre_list)}  n_genres {genres_counter}  h {h}  w {w}')


	# ---------- remapping node id ----------
	print('remap node id')
	userid_dict = {}
	user_id=0

	print(len(user))
	for index,u in enumerate(user):
		if u not in userid_dict.keys():
			userid_dict[u]=user_id
			new_id=user_id
			user_id+=1
		else:
			new_id=userid_dict[u]
		user[index]=new_id

	print(len(item))
	for index, i in enumerate(item):
		item[index]=id_dict[i]



	# ---------- remapping label -----------
	new_rating = np.zeros(len(rating))
	new_rating[rating>3] = 1	# like
	new_rating[rating<=3] = 0	# dis like


	# ---------- remapping time ------------
	min_time=np.min(time)
	time=time-min_time

	d={}
	d['user'] = user
	d['item'] = item
	d['rating'] = new_rating
	d['time'] = time
	d['feature'] = feat.numpy()

	f=open(cached_file,'wb')
	pickle.dump(d,f)
	f.close()

	print(d)
	print('file saved! ',cached_file)








if __name__ == "__main__":
	#filter_edges()
	#sample_edges_by_time()
	process_data()


