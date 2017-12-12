# # find the set of users who ranked at least k = 20 businesses
# explore the user data set
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import json
from sklearn.metrics.pairwise import euclidean_distances
import heapq as hq

user = pd.read_csv("search/user.csv")
business = pd.read_csv("search/business.csv")
review = pd.read_csv("search/review.csv")
tips = pd.read_csv("search/tip.csv")
# Subseting for users who have written atleast 20 reviews
user_20 =user[user['review_count'].between(50, 100, inclusive=True)]

user_20_userids=user_20['user_id'].values
review_user=review[review['user_id'].isin(user_20_userids)]

# get the attributes of businesses rated by these users
review_user_20= review_user[['user_id','review_id','business_id','stars','date']]
review_user_20.rename(columns= {'stars':'userStars'},inplace= True)
# get the business attributes for these businesses
review_business = pd.merge(review_user_20, business, on= 'business_id', how= 'left')

# Replace all the null values with False
review_business.fillna(value= False,inplace=True)

bool_dtypes =list( review_business.dtypes[review_business.dtypes == 'bool'].index)
	
business_features = review_business[bool_dtypes].astype(int)

# Create a subset of all businesses 
dcols = [col for col in review_business.columns if 'attributes' in col]
## select features that are not boolean and re-treat them
non_bool_dtypes = [ col for col in dcols if col not in bool_dtypes]

#replace the incorrectly replaced Falss values with the values that are present in the column
review_business['attributes.AgesAllowed'].replace(False, 'allages', inplace= True)
review_business['attributes.AgesAllowed'].replace('False', 'allages', inplace= True)
review_business['attributes.Alcohol'].replace(False, 'none', inplace= True)
review_business['attributes.NoiseLevel'].replace(False, 'none', inplace= True)
review_business['attributes.RestaurantsAttire'].replace(False, 'nocode', inplace= True)
review_business['attributes.BYOBCorkage'].replace('nocode' , 'none', inplace= True)
review_business['attributes.WiFi'].replace(False , 'no', inplace= True)
review_business['attributes.Smoking'].replace(False, 'unknown',inplace= True)

# now merge the boolean and the non boolean data types
business_features2 = pd.concat([review_business['business_id'],business_features, review_business[non_bool_dtypes]],axis=1)
business_features2.drop_duplicates(inplace= True)
# wirte the file 
business_features2.to_csv('search\features_final')


user_20_sample = user_20.sample(int(0.02*len(user_20)))
user_20_userids_sample = user_20_sample.user_id.values
review_business_s = review_business[review_business['user_id'].isin(user_20_userids_sample)]
# split test and training
msk = np.random.rand(len(review_business_s)) < 0.8
train , test = review_business_s[msk], review_business_s[~msk]

# subset using the business_ids
train_business_ids = business_features2[business_features2.business_id.isin(train.business_id.values)]


attributes1 = business_features2.set_index('business_id').loc[ :, business_features2.columns.difference(['business_id','attributes.AgesAllowed','attributes.Alcohol','attributes.NoiseLevel','attributes.RestaurantsAttire','attributes.BYOBCorkage','attributes.WiFi','attributes.Smoking'])]
dummied_features = pd.get_dummies( business_features2.set_index('business_id')[['attributes.AgesAllowed','attributes.Alcohol','attributes.NoiseLevel','attributes.RestaurantsAttire','attributes.BYOBCorkage','attributes.WiFi','attributes.Smoking']]).astype(int)

similarity_input = pd.concat([attributes1,dummied_features],axis=1)

# do the same operations t data to train

attributes1 = train_business_ids.set_index('business_id').loc[ :, train_business_ids.columns.difference(['business_id','attributes.AgesAllowed','attributes.Alcohol','attributes.NoiseLevel','attributes.RestaurantsAttire','attributes.BYOBCorkage','attributes.WiFi','attributes.Smoking'])]
dummied_features = pd.get_dummies( train_business_ids.set_index('business_id')[['attributes.AgesAllowed','attributes.Alcohol','attributes.NoiseLevel','attributes.RestaurantsAttire','attributes.BYOBCorkage','attributes.WiFi','attributes.Smoking']]).astype(int)

similarity_input_train = pd.concat([attributes1,dummied_features],axis=1)

# store the train and test files
train.to_csv('search/train')
test.to_csv('search/test')
similarity_input.to_csv('search/similarity_input')
similarity_input_train.to_csv('search/similarity_input_train')
user_20_sample.to_csv('search/user_20_sample')

# Use KD Tree from sklearn

kdt = KDTree(np.asarray(similarity_input), leaf_size=30, metric='euclidean')
split1 , split2 ,split3, split4, split5 = np.array_split(similarity_input_train, 5)
distance, indices = kdt.query(split1, k=100 , return_distance=True)
distance_t, indices_t = kdt.query(split2, k=100, return_distance=True)
distance = np.vstack((distance,distance_t))
indices = np.vstack((indices,indices_t))

distance_t, indices_t = kdt.query(split3, k=100, return_distance=True)
distance = np.vstack((distance,distance_t))
indices = np.vstack((indices,indices_t))

distance_t, indices_t = kdt.query(split4, k=100, return_distance=True)
distance = np.vstack((distance,distance_t))
indices = np.vstack((indices,indices_t))

distance_t, indices_t = kdt.query(split5, k=100, return_distance=True)
distance = np.vstack((distance,distance_t))
indices = np.vstack((indices,indices_t))

# get indices of the sampled data to map the indices back
index_map = business_features2.index
business_features_subset = train_business_ids.loc[index_map]
user_business_subset= pd.merge(review_business_s, business_features_subset, on= 'business_id', how= 'inner')

train_business_ids['similarIndicesEuclidean']= indices.tolist()

# map it to the user.

train = pd.merge(train, train_business_ids[['business_id','similarIndicesEuclidean']] , on = 'business_id', how='inner')

train.to_csv('search/trainModel')
# for recommendations at a user_id level.
train['similarIndicesEuclidean']= train['similarIndicesEuclidean'].astype(str)
user_reco = train.groupby('user_id')['similarIndicesEuclidean'].apply(lambda x: "[%s]" % ','.join(x))

flat_list=[]
# for for each user id get the unique list of business
for rec_list in user_reco.values:
	flat_list.append(list(set([item for sublist in json.loads(rec_list) for item in sublist])))

# map indexes to business_ids

buss_list = [ np.asarray(business_features2['business_id'])[lst].tolist() for lst in flat_list ]



# create a disctionary for faster computation
train_dict = {}
for user in user_reco.index.values:
	train_dict[user]= train[train['user_id']==user]['business_id'].values

feature_dict = similarity_input.T.to_dict('list')
i=0
final_list={}
for user in user_reco.index.values:
	rated_business = train_dict[user]
	# remove businesses_alrady rated by user in the train set
	temp= list(set(buss_list[i]).difference(set(rated_business))) 
	U = np.asarray([feature_dict[key] for key in rated_business])
	priorityq = []
	for ind in temp:
		x = np.asarray(feature_dict[ind])
		rank = sum(euclidean_distances(U, x))
		hq.heappush(priorityq,(rank[0],ind))
	# get the top ranked item
	if len(temp)>=50:
		final_list[user]= hq.nsmallest(50,priorityq)
	else:
		final_list[user]= hq.nsmallest(len(temp),priorityq)
	i+=1
	
# results
results = {}
for user in final_list:
	results[user]= map(lambda x: x[1] , final_list[user])
# retain only users present in train
test_sub = test[test['user_id'].isin(user_reco.index.values)]
test_sub_dict={}
for user in test_sub.user_id.unique():
		test_sub_dict[user] = test_sub[test_sub['user_id']==user]['business_id'].values
# check the numberof hits
user_hits= {}
for user in test_sub_dict:
	user_hits[user] = len(set(test_sub_dict[user]).intersection(set(results[user])))

print  sum(user_hits.values())

# find the percentage of new businesses and old businesses
reco_buss = [item   for val in test_sub_dict.values() for item in val]

test_buss_attr = business[business['business_id'].isin(reco_buss)][['business_id','review_count']]

test_buss_attr['new_business']=0
test_buss_attr['old_business']=0

# defining old business as ones that got more than 120 reviews 
test_buss_attr['new_business'][business['review_count']<=120]= 1
test_buss_attr['old_business'][business['review_count']>120]= 1

test_buss_attr.drop_duplicates(inplace= True)

# Calculating the % of new businesses and old businesses recommended
print "% of new business recommeded", sum(test_buss_attr['new_business']) * 100.0 /len(test_buss_attr)
print "% of old businesses recommeded", sum(test_buss_attr['old_business']) * 100.0 /len(test_buss_attr)

#Coverage of a top n recommender

unique_buss = set(reco_buss)

coverage = len(unique_buss) *100.0 /len(business_features2)
