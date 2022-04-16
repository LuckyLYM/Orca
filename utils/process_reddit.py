from time import time
import os 
from datetime import datetime
import numpy as np
import sys


# we filter out the nodes without raw features
def load():

	folder = './'
	title_edges_file = 'soc-redditHyperlinks-title.tsv'
	body_edges_file = 'soc-redditHyperlinks-body.tsv'
	nodes_file = 'web-redditEmbeddings-subreddits.csv'


	# load node file
	file = nodes_file
	file = os.path.join(folder,file)
	with open(file) as file:
		file = file.read().splitlines()
	ids_str_to_int = {}
	id_counter = 0
	feats = []

	# remap node id: mapping from strs to real node ids
	print('make dictionary')
	for line in file:
		line = line.split(',')
		nd_id = line[0]
		if nd_id not in ids_str_to_int.keys():
			ids_str_to_int[nd_id] = id_counter
			id_counter += 1
			nd_feats = [float(r) for r in line[1:]]
			feats.append(nd_feats)
		else:
			print('duplicate id', nd_id)
			raise Exception('duplicate_id')



	print('load edges')
	edges = []
	not_found = 0
	edges_tmp, not_found_tmp = load_edges_from_file(title_edges_file,folder,ids_str_to_int)
	edges.extend(edges_tmp)
	not_found += not_found_tmp

	# * load edges in bodies
	edges_tmp, not_found_tmp = load_edges_from_file(body_edges_file,folder,ids_str_to_int)
	edges.extend(edges_tmp)
	not_found += not_found_tmp
	print(f'edges {len(edges)}')



	edges=np.array(edges)
	time=edges[:,2].astype(float) 
	min_time=np.min(time)
	ind=np.argsort(time)
	edges=edges[ind]
	edges[:,2]=edges[:,2].astype(float)-min_time
	print('start writing file')
	file=os.path.join(folder,'large_reddit')
	f=open(file,'w')
	f.write('src, trg, timestamp, label, comma_separated_list_of_features\n')
	print(f'min time {min_time}')
	for index,edge in enumerate(edges):
		if index<=10:
			print(f'{edge[0]},{edge[1]},{edge[2]},{edge[3]}')
		f.write(f'{edge[0]},{edge[1]},{edge[2]},{edge[3]},{edge[4]}\n')
	f.close()




# in the revised version, we don't add inverse edges
def load_edges_from_file(edges_file,folder,ids_str_to_int):
	edges = []
	not_found = 0
	file = edges_file
	file = os.path.join(folder,file)
	with open(file) as file:
		file = file.read().splitlines()


	#base_time = datetime.strptime("19800101", '%Y%m%d')
	base_time=datetime.strptime('1980-01-01 0:0:0', '%Y-%m-%d %H:%M:%S')
	#print(f'base time {base_time}')
	src_list = []
	tg_list= []
	time_list = []
	label_lsit = []
	feature_list = []
	for index,line in enumerate(file[1:]):
		fields = line.split('\t')
		sr = fields[0]
		tg = fields[1]
		if sr in ids_str_to_int.keys() and tg in ids_str_to_int.keys():
			sr = ids_str_to_int[sr]
			tg = ids_str_to_int[tg]
			time = fields[3]
			time = datetime.strptime(time,'%Y-%m-%d %H:%M:%S')
			delta= time - base_time
			time = delta.seconds + delta.days*3600*24
			#print(f'delta time {time}')
			label = int(fields[4])
			feature= fields[5]
			'''
			src_list.append(sr)
			tg_list.append(tg)
			time_list.append(time)
			label_lsit.append(label)
			feature_list.append(feature)
			'''
			edges.append([sr,tg,time,label,feature])
			'''
			if index==10:
				sys.exit(1)
				print([sr,tg,time,label,feature])
				print(f'feature length {len(feature)}')
			'''
		else:
			not_found+=1
	return edges,not_found

load()
























