import time as timer
import random
import pandas as pd
import numpy as np
import torch 
import itertools
from torch.utils.data import DataLoader, Dataset

tt = torch.tensor
#--------------------------------------------------------------------------------------
def bpr_loss_func(pos, neg):
    # Return average loss
    return -torch.log(torch.sigmoid(pos - neg)).sum() # / pos.shape[0]
#--------------------------------------------------------------------------------------
class myDataset(Dataset):
    """
    Attributes: 
        dct : dictionary with attributes ['member_attr', 'item_attr', 'feature_offsets']

    Constructor: 
        filenm (string) : file to process
        nrows (int) : number of rows to process. All rows if None. 
    """
    def __init__(self, filenm, nrows=None):
        super(Dataset, self).__init__()
        if nrows == None:
            df = pd.read_csv(filenm)
        else: 
            df = pd.read_csv(filenm, nrows=nrows)
        # keep just a few attributes
        #cols = ["MEMBER_ID", "D", "age_at_flight", "booking_mo", "booking_dowk", "flight_mo", "flight_dowk"]

        # Remove flights with identical MEMBER_ID and D

        df = df.drop_duplicates(['MEMBER_ID', 'D'])
        
        cols = ["MEMBER_ID", "D", "age_at_flight"]
        
        # Member Attributes
        # 1. age_at_flight, which should be changed to the average age
        # 2. number of miles 
        # 3. gender
        # 4. average number of trips per year
        #
        # Destination
        # 1. Temperature
        # 
        # Cross-attributes
        # 1. booking month and day of week
        # 2. flight month and day of week randomly
        #
        #
        # Initially, only consider member and destination attributes (if any)
        
        # When estimating rankings, choose booking month and week randomly. 
        
        df = df[cols]
        df['age_at_flight'] = df.groupby('MEMBER_ID').transform('mean')
        df['age'] = pd.cut(df.age_at_flight, bins=[0, 30, 50, 70, 120])
        df = df.drop('age_at_flight', axis=1)
        
        # Now, change all columns to ordinal numbers
        df = df.astype('category')
        D = df['D']
        M = df['MEMBER_ID']
        df = df.apply(lambda x: x.cat.codes)
        
        df['dest'] = D
        df['member_id'] = M
        
        self.unique_D = set(df['D'])
        
        # Map from ordinal destination codes and the destinations (3 <-> 'HAV')
        # Destination ordinal numbers cannot be compared between files
        # destinations: 
        pair = df[['D', 'dest']]
        p = pair.apply(tuple, axis=1).unique()
        self.destination_dict = dict(sorted(p, key=lambda x: x[0]))
        #self.destination_dict = dict(p)
        self.idx2dest = self.destination_dict;
        self.dest2idx = dict([(v, k) for k,v in self.idx2dest.items()])
        
        # Ideally, idx2member should return member and its attributes
        pair1 = df[['MEMBER_ID', 'member_id', 'age']]   # include attributes <<<<<<<<<<
        p1 = pair1.apply(tuple, axis=1).unique()
        self.member_dict = sorted(p1, key=lambda x: x[0]) # idx -> member
        self.member_dict = {item[0]:item[1:] for item in self.member_dict}
        self.idx2member = self.member_dict
        self.member2idx = {v[0] : (k,v[1:]) for k,v in self.idx2member.items()}
        
        # Generate negative samples. These will be destinations not chosen by the user, but chosen
        # by at least one other user. 
        
        # Destinations by user
        self.D_set = df.groupby('MEMBER_ID').agg({'D':set}).iloc[:, -1].to_frame('E')  # keeps column name 'D'
        
        df = df.drop(['dest','member_id'], axis=1)
        
        field_dims = list(df.nunique())
        field_dims = tt(field_dims, dtype=torch.int)
        
        self.field_dims = field_dims
        self.df = df
        
        self.member_attr = [0,2]
        self.dest_attr = [1]
        
        dct['member_attr'] = self.member_attr
        dct['dest_attr']   = self.dest_attr
        dct['field_dims']  = self.field_dims
        dct['D_set']       = self.D_set
        dct['dest2idx']    = self.dest2idx
        dct['idx2dest']    = self.idx2dest
        dct['member2idx']  = self.member2idx
        dct['idx2member']  = self.idx2member
        self.dct = dct
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx,:]
        member_id = row['MEMBER_ID']
        # starting with all destinations, calculate set of destinations not flown to by this member
        diff_set = self.unique_D - self.D_set.loc[member_id, :].values[0]
        # Pick a random element from the resulting set
        neg = random.choice(list(diff_set))  # negative sample
        return tt(row, dtype=torch.int), tt([neg]), tt([1.])
    
    def __len__(self):
        return self.df.shape[0]

#-----------------------------------------------------------------------
def getData(file, shuffle=True, batch_size=32, nrows=None):
    """
    Arguments:
    ---------
    file: file to process
    batch_size: default: 32
    nrows: default: None  (read all rows)
        Number of rows to read. 
    shuffle: default: True.
        Whether to shuffle the data each iteration or not.

    Return:
        dct: dictionary 
            dataset ('train_data') and data_loader iterator ('train_iter')
    ------
    """
    train_file = file
    train_data = myDataset(train_file, nrows=nrows)
    train_iter = DataLoader(train_data, batch_size = batch_size, shuffle=True)
    dct = {"train_data": train_data, "train_iter": train_iter}
    return dct

#----------------------------------------------------------------------------
class AccuracyDataset(Dataset):
    """
    Given a Dataframe, and a list of destinations, create a virtual dataset
    That contains all cities for each row.
    Constructor Arguments: 
    ---------------------
        dataset: dataframe with data
        destinations: list of all destinations. Order is immaterial. 
        
    Assume the member list is [7, 2, 5], and the city list is [9,2,8]. 
    Create an effective database: 
         Member   City
         7         9
         7         2
         7         8
         2         9
         2         2
         2         8
         8         9
         8         2
         8         8
         
    """
    def __init__(self, dataset, destinations):
        super(Dataset, self).__init__()
        self.dataset = dataset
        self.df = dataset.df
        n_dest    = len(destinations)  # 75  # I ASSUME LESS RELIABLE
        #self.n_members = self.df.shape[0]
        self.n_members = np.array(list(dataset.dct['idx2member'].keys())).max() + 1  
        self.n_dest = np.array(list(dataset.dct['idx2dest'].keys())).max() + 1  # 76: 0 -> 75 inclusive

        print("dest keys: ", dataset.dct['idx2dest'].keys())  # 76 keys: 0 -> 75
        print("destinations: ", destinations)  # highest is 74
        print("nb members: ", self.n_members)
        #print("member: ", self.n_members, n_members)
        print("dest: ", self.n_dest, n_dest)

        assert self.n_dest == n_dest, f"Error in n_dest (self.n_dest:{self.n_dest}, n_dest:{n_dest})"
        #print("n_dest: ", self.n_dest)
        # vars = df.groupby(['MEMBER_ID','D'])['age'].mean()
        self.member_count = 0
        self.city_count = 0
        
        
    def __len__(self):
        #print("len: ",  self.n_members * self.n_dest)
        return self.n_members * self.n_dest
    
    def __getitem__(self, idx):
        #print("==== idx: ", idx)
        dest = idx % self.n_dest
        member = (idx - dest) // self.n_dest
        #if member > 2215: # member_max = 2216
            #print("member: ", member)
        member_attr = self.dataset.idx2member[member]
        # I need the age for this member
        return member, dest, 2  # same age for everybody: 2 (age bracket, probably 30-50)

#----------------------------------------------------------------------------

def setup_trainer(net, gdct):
    lr = gdct['lr']
    wd = gdct['wd']
    optimizer_method = gdct['optim']
    print("optimiz_meth: ", optimizer_method)

    if optimizer_method == 'adamW':
        print("adamW")
        optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_method == 'adam':
        print("adam")
        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_method == 'sgd':
        print("SGD")
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
    else:
        print("optimizer not supported")
        raise "Error"

    #loss_func = nn.BCELoss()
    loss_func = bpr_loss_func
    return optimizer, loss_func
#----------------------------------------------------------------------------
def train_epoch_new(model, optimizer, data_loader, criterion, device, log_interval=100):
    #"""
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    #"""

    # Model is already on the device
    model.train()
    total_loss = 0
    # tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=5.0, nrows=5)
    losses = []
    count = 0

    #"""
    start.record()  # 500 ms (data_loader takes 1/2)
    #"""
    for i, fields in enumerate(data_loader):
        #start.record()  # 2 ms
        pos_float, neg_float, pos_emb, neg_emb = fields  # member_id, D, attr

        # Target not on the device
        # data transferred to the device in myDataset class

        # Max value of pos and neg_emb (36467)

        ypos = model(pos_emb)   # <<<< ERROR
        yneg = model(neg_emb)

        # neg and pos are actually destinations + their attributes
        # So I should have a fields_pos and a fields_neg. It is easy if
        # there are no attributes for the destination. What is not clear is how
        # to handl the cross-attributes (i.e., date of flight, time of flight).
        # If there are only member and flight attributes, it is not an issue.
        # HOW TO HANDLE THIS??? For this, look into sequence-aware recommenders and
        # check out how they do it.

        # apply BCR loss/criterion
        loss = criterion(ypos, yneg)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1

        # NO IDEA WHAT THIS DOES
        # if (i + 1) % log_interval == 0:
        #     tk0.set_postfix(loss=total_loss / log_interval)
        #     total_loss = 0
        #if i % 10 == 0: 
            #print(f"total_loss (sum over batch): {total_loss}, count: {count}")
            #pass
        total_loss = total_loss / data_loader.dataset.df_.shape[0]

                                  # 48 ms per batch (size 4*1024) (entire function). Time in both model calls: 0.33 ms. 
                                  # Cost proportional to batch size. Logical. 
        #end.record()
    #"""
    end.record()
    torch.cuda.synchronize()  # 12 ms per batch (size 1024) (entire function). Time in both model calls: 0.33 ms. 
    print("elapased time per epoch: ", start.elapsed_time(end))
    #"""



    return(total_loss)

#-----------------------------------------------------------------------
#----------------------------------------------------------------------------
def train_epoch(model, optimizer, data_loader, criterion, device, log_interval=100):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()


    #print("INSIDE train_epoch")
    # Model is already on the device
    model.train()
    total_loss = 0
    # tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=5.0, nrows=5)
    losses = []
    dct = {}
    count = 0
    # attr_D and attr_negD: avg_yr_L, avg_yr_h, LAT_DEC, LON_DEC, HEIGHT
    # attr_user: age_departure, gender
    for i, fields in enumerate(data_loader):
        batch_size = data_loader.batch_size
        user_attr = fields[0]  # member_id, D, attr

        pos_attr = fields[1]   # member_id, D, attr(D)
        neg_attr = fields[2]   # member_id, D, negD, attr(negD)
        pos_attr = pos_attr[:, 1:]  
        neg_attr = neg_attr[:, 2:]
        target = fields[3]

        # Target is always 1
        # user_attr, pos_attr, neg_attr, target = user_attr.to(device), pos_attr.to(device), neg_attr.to(device), target.to(device)

        member_id = user_attr[:, 0].unsqueeze(1) 
        D = pos_attr[:, 0].unsqueeze(1)
        negD = neg_attr[:, 0].unsqueeze(1)
        #print("D,negD,member_id: ", D.shape, negD.shape, member_id.shape)
        user_attr = user_attr[:, 2:] # skip MEMBER_ID, D
        pos_attr = pos_attr[:, 1:]  
        neg_attr = neg_attr[:, 1:]

        target = target.unsqueeze(1)

        #print("user, pos, neg, target: ", user_attr.shape, pos_attr.shape, neg_attr.shape, target.shape)


        # let us not worry about efficiency
        #print(user_attr.device, pos_attr.device, neg_attr.device, target.device)
        fields_pos = torch.cat([member_id, D, user_attr, pos_attr], axis=1)
        fields_neg = torch.cat([member_id, negD, user_attr, neg_attr], axis=1)

        # ISSUE: fields_pos are all floats. However, the fields slatted for an embeddings should be ints, 
        # and only these fields should be fed to the network. 
         
        # Extract all the integer fields
        fields_pos_emb = torch.cat([fields_pos[:,0:2], fields_pos[:,3:4]], axis=1)
        fields_neg_emb = torch.cat([fields_neg[:,0:2], fields_neg[:,3:4]], axis=1)
        fields_pos_float = torch.cat([fields_pos[:,2:3], fields_pos[:,4:]], axis=1)
        fields_neg_float = torch.cat([fields_neg[:,2:3], fields_neg[:,4:]], axis=1)

        # make all the embedding fields into integers
        #print("pos_emb: ", fields_pos_emb[0,:])
        #print("neg_emb: ", fields_neg_emb[0,:])
        fields_pos_emb = fields_pos_emb.type(torch.int32)
        fields_neg_emb = fields_neg_emb.type(torch.int32)
        #print("pos_emb: ", fields_pos_emb[0,:])
        #print("neg_emb: ", fields_neg_emb[0,:])

        #print(fields_neg.shape, fields_pos.shape)
        #print("pos: ", pos_attr[0,:])
        #print("neg: ", neg_attr[0,:])
        #print("user: ", user_attr[0,:])
        #print("member_id: ", member_id[0,:])
        #print("fields_pos: ", fields_pos[0,:])
        #print("fields_neg: ", fields_neg[0,:])

        # Create the dims field


        # Fields: Member, Dest, age, gender, yr_l, yr_h, lat, long, altitude


        # Deterministic model. The base FM has arguments in any order
        #start.record()
        ypos = model(fields_pos_emb)   # <<<< ERROR
        yneg = model(fields_neg_emb)
        #end.record()

        # yneg = model(fields, neg)
        # apply BCR loss/criterion

        # neg and pos are actually destinations + their attributes
        # So I should have a fields_pos and a fields_neg. It is easy if
        # there are no attributes for the destination. What is not clear is how
        # to handl the cross-attributes (i.e., date of flight, time of flight).
        # If there are only member and flight attributes, it is not an issue.
        # HOW TO HANDLE THIS??? For this, look into sequence-aware recommenders and
        # check out how they do it.

        loss = criterion(ypos, yneg)

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += 1
        # NO IDEA WHAT THIS DOES
        # if (i + 1) % log_interval == 0:
        #     tk0.set_postfix(loss=total_loss / log_interval)
        #     total_loss = 0
        if i % 10 == 0: 
            #print(f"total_loss (sum over batch): {total_loss}, count: {count}")
            pass
        total_loss = total_loss / data_loader.dataset.df_.shape[0]

        end.record()
        torch.cuda.synchronize()  # 12 ms per batch (size 1024) (entire function). Time in both model calls: 0.33 ms. 
                                  # 48 ms per batch (size 4*1024) (entire function). Time in both model calls: 0.33 ms. 
                                  # Cost proportional to batch size. Logical. 
        print("elapased time in train_epoch: ", start.elapsed_time(end))


    return(total_loss)

#-----------------------------------------------------------------------
def test_accuracy(model, data_loader, device):
    model.eval()
    fields_ = []
    predicts = []
    count = 0
    for i, pos_emb in enumerate(data_loader):

        with torch.no_grad():
            # Ignore float attributes for now
            y = model(pos_emb)   # <<<<< ERROR in forward

        predicts.append(y)
        fields_.append(pos_emb)

    predicts = torch.cat(predicts, axis=0).to('cpu')
    fields_  = torch.cat(fields_, axis=0).to('cpu')

    return fields_, predicts

#--------------------------------------------------------------------------
def compute_hits_1(model, loader, dct, members, pairs, topN, valid_dest_sets):
    fields, predicts = test_accuracy(model, loader, dct.device)  # cost: 3.8 sec
    pairs['pred'] = predicts
    res1 = pairs.groupby('MEMBER_ID').agg({'D':list, 'pred':list})
    res1['argsort'] = res1['pred'].apply(lambda x: np.argsort(x)[::-1])
    # return  # 5.4
    res2 = extract_topN(res1, topn=topN)

    v_df = valid_dest_sets.drop('MEMBER_ID', axis=1).reset_index()
    r_df = res2.reset_index()
    df = v_df.merge(r_df, how='inner', on='MEMBER_ID')

    def intersect(row):
        return min(len(row['D'].intersection(row['D1'])), 1)

    df['hits'] = df.apply(intersect, axis=1)
    hit_rate =  df['hits'].sum() / len(members)
    return hit_rate, res1

#--------------------------------------------------------------------------
def extract_topN(res1, topn):
    # Perform the topN ranking
    def argsortcolD(row):
        return np.array(row['D'])[row['argsort'][0:topn]]

    def argsortcolPred(row):
        return np.array(row['pred'])[row['argsort'][0:topn]]

    res1['D1'] = res1.apply(argsortcolD, axis=1)
    res1['pred1'] = res1.apply(argsortcolPred, axis=1)
    res2 = res1[['D1', 'pred1']]
    return res2

#--------------------------------------------------------------------------
def flatten(lst_of_lists):
    # Flatten a list of lists
    return list(itertools.chain(*lst_of_lists))

#--------------------------------------------------------------------------
def hetero_MD_df(members, valid_dest_sets):
    #Mlist = members
    #valid_dest_sets["lists"] = valid_dest_sets["neg_set"].apply(lambda x: list(x))
    #print(f"hetero: valid_dest_sets: {valid_dest_sets.shape}: \n{valid_dest_sets.list{5}")

    # members in the valid_dest_sets list
    Mlist = list(valid_dest_sets.MEMBER_ID.values)
    #print("Mlist: ", Mlist[0:5])

    DD = list( valid_dest_sets["neg_set"].apply(lambda x: list(x)).values )
    #print("hetero: DD: \n", len(DD), DD[0:5])
    MM = [ [Mlist[i]] * len(DD[i]) for i in range(len(Mlist)) ]
    #print("hetero: MM: \n", len(MM), MM[0:5])
    df_flat = pd.DataFrame({'MM': flatten(MM), 'DD': flatten(DD)})
    return df_flat

#--------------------------------------------------------------------------
def compute_hits_2(members, train_dest_sets, valid_dest_sets):
    hits = 0
    nb_members = len(members)
    print("compute_hits, nb_members: ", nb_members)

    for member in members:
        try:
            td = train_dest_sets.loc[member]  # train destinations (set)
        except:
            nb_members -= 1  # Member not found in training set. This should not happen. 
            continue
        set_valid = valid_dest_sets.loc[member]
        vd = set_valid['D']    # validation destinations (set)
        rp = res2.loc[member, 'D1'] # ranked prediction (ndarray)
        rk = []
        for d in rp:
            if d in td['D']: 
                continue
            rk.append(d)
            if len(rk) == topN:
                break

        if len(rk) != topN:
            print("len(rk): ", len(rk))
        if len(rk) < topN:
            print("==> member: ", member)
            print("    vd = ", vd, "    (validation dest)")
            print("    td= ", td.values, "  (training dest)")
            print("    rk= ", rk[0:10], "   (ranked dest)")
            print("    rp= ", rp, "    (trimmed ranked dest)")
            print("    len(vd): ", len(vd), "   , len(td): ", len(td))     
        if vd.intersection(rk):
            hits += 1

#--------------------------------------------------------------------------
def compute_hits_2_faster(members, train_dest_sets, valid_dest_sets, res2):
    hits = 0
    nb_members = len(members)
    print("compute_hits_faster, nb_members: ", nb_members)

    # Step 1: Only work with members
    td_df = train_dest_sets.loc[members]
    set_valid_df = valid_dest_sets.loc[members]
    print("td_df: ", td_df.shape, td_df.columns)  # D, MEMBER_ID
    print("set_valid_df: ", set_valid_df.shape, set_valid_df.columns) # D, MEMBER_ID
    vd_df = set_valid_df['D']
    rp_df = res2.loc[members, ['D1']]  # ['D1'] so that rp_df is a dataframe
    print("rp_df: ", rp_df.shape)
    print(rp_df.head())

    for member in members:
        # continue  # 6.4
        """
        try:
            td = train_dest_sets.loc[member]  # train destinations (set)
        except:
            nb_members -= 1  # Member not found in training set. This should not happen. 
            continue
        set_valid = valid_dest_sets.loc[member]
        vd = set_valid['D']    # validation destinations (set)
        """
        #continue  # 6.11
        rk = []
        rp = res2.loc[member, 'D1'] # ranked prediction (ndarray), list of D
        print("rp: ", rp)

        # td['D'] list of destinations by this member in training set
        # rp: ranked destinations for this member 

        # rp: list of ranked destination recommendations for member
        # td: list of destinations by member in training set

        for d in rp:
            if d in td['D']: 
                continue
            rk.append(d)
            if len(rk) == topN:
                break
        continue  # 8.76

        # If the next two conditionals are triggered, it means that the number of new destinations
        # is either 
        #     1) not equal to the number requested 
        #     2) less than the number requested 
        if len(rk) != topN:
            print("len(rk): ", len(rk))
        if len(rk) < topN:
            print("==> member: ", member)
            print("    vd = ", vd, "    (validation dest)")
            print("    td= ", td.values, "  (training dest)")
            print("    rk= ", rk[0:10], "   (ranked dest)")
            print("    rp= ", rp, "    (trimmed ranked dest)")
            print("    len(vd): ", len(vd), "   , len(td): ", len(td))     
        if vd.intersection(rk):
            hits += 1

#--------------------------------------------------------------------------
class PairDataset():
    """
    Data already on device
    """
    def __init__(self, dct, pairs, data_valid):
        """
        Arguments: 
        ---------
        data_valid: (DataFrame)
            All members of data_valid appear in the training set
        """
        self.pairs = pairs
        self.data = data_valid
        self.members = self.data.drop_duplicates('MEMBER_ID')[['MEMBER_ID', 'GENDER']]
        self.dest = dct.dataset_train.data[['D']].drop_duplicates('D')
        self.newp = self.members.merge(self.dest, how="cross")
        self.newp = self.newp[['MEMBER_ID', 'D', 'GENDER']]

        # Drop from pairs all members not in self.data
        self.newp = tt(self.newp.values).to(dct.device)
    def __len__(self):
        return self.newp.shape[0]
    def __getitem__(self, idx):
        return self.newp[idx,:]

#--------------------------------------------------------------------------
def recommender(model, dct, topN=5, keep_nb_members=None):
    """
    Compute the hit_rate to evaluate destinations

    Arguments: 
    ---------
    model
    interactions_dct: dictionary
    topN: [5]
        number of elements to keep
    keep_members: [None]
        number of members to consider in validation

    Return
    ------
    pair: list of pairs (MEMBER_ID, D) used to validate

    """
    # Only keep users in the training set
    # user_attrib is build from the validation dataset with only users in the training set. 
    user_attrib = restrict_member_attrib(dct)

    print("user_attrib: ", user_attrib.shape) # 55073 rows left (out of 72380 in validation set)
    print("data_valid unique members: ", dct.data_valid.MEMBER_ID.nunique())  # 26235
    print("user_attrib unique members: ", user_attrib.MEMBER_ID.nunique())  # 16669

    data_train = dct['data_train']
    #data_valid = dct['data_valid']
    data_valid = user_attrib   # (members not in training set have been removed) (16669)

    print("data_valid: ", data_valid.shape)

    all_data = dct['df_members']
    all_dest= np.asarray(sorted(list(all_data.D.unique())))

    #return # 20 ms

    # Translate all_dest from destinations with 'PTY' into an integer
    # To be consistent with labeling during training phase
    dest_idx = [dct.dest2idx[dest] for dest in all_dest]
    all_dest = dest_idx  # (86) total destinations at Copa
    print("all_dest len: ", len(all_dest))  # 86

    # Restrict all_dest to the destinations in the training set
    # since those are the datasets the model is trained on
    all_dest = data_train['D'].unique()
    print("nb of unique dest in the training set: ", all_dest.shape)  # 74
    print("nb of unique dest in the validation set: ", data_valid['D'].unique().shape)  # 76

    train_dest_sets = data_train.groupby('MEMBER_ID').agg({'D':set})
    valid_dest_sets = data_valid.groupby('MEMBER_ID').agg({'D':set})

    # Remove from valid_dest_sets all users not in the training set

    dct['train_dest_sets'] = train_dest_sets
    dct['valid_dest_sets'] = valid_dest_sets

    # For each member in the validation set, estimate the scores for all destinations. 
    members = data_valid.MEMBER_ID.unique()  # ndarray of members

    # Only keep members in the training set (avoid cold-start problem)
    members_training_set = set(data_train.MEMBER_ID.unique())
    members_valid_set    = set(members)
    members = np.array(list(members_valid_set.intersection(members_training_set)))

    # Restrict valid data set to the members in the training set
    data_valid_nocold = data_valid.set_index('MEMBER_ID', drop='True').loc[members].reset_index()

    data_valid = data_valid_nocold.copy()  # Does not overwrite dct.data_valid

    # Access MEMBER_ID via index or column
    valid_dest_sets['MEMBER_ID'] = valid_dest_sets.index
    train_dest_sets['MEMBER_ID'] = train_dest_sets.index

    # Keep a small number of members

    if keep_nb_members != None:
        members = members[0:keep_nb_members]

    pairs = []

    # How to create 
    for i, member in enumerate(members):
        for dest in all_dest:
            pairs.append((member, dest))

    pairs = pd.DataFrame(pairs, columns=['MEMBER_ID','D'])  # (26235,2)
    print("pairs.shape: ", pairs.shape) # 2M records. For each member, 80 destinations

    pair_dataset = PairDataset(dct, pairs, data_valid)
    loader = DataLoader(pair_dataset, batch_size=4*4096, shuffle=False) # False for debugging

    # STEP 1: fix the hit rate without filtering
    results = compute_hits_1(model, loader, dct, members, pairs, topN, valid_dest_sets)
    hr_not_filtered, res1 = results
    print("hit rate (without previous filter) = ", hr_not_filtered)  # 0.36 (Should be higher). 
    #------------------------------------------------------------------------------------------------------

    # Remove training D from the destinations to check
    # For each member, construct a list or set of D to check. 
    # Construct a list of (MEMBER_ID, D, GENDER) to check against. 
    # How to construct this list efficiently? In Julia or C++, I could use loops. 

    #def remove_trai

    #return # 5.8s

    # res2 has columns 'D1' and 'pred1': 
    # 'D1' predicted destinations
    # 'scores' predicted scores in ranked drder

#  First issue: even when running 500 epochs, the scores are a mixture of neg and positive. WHY? 
#  I wonder if this is true with the training data? 

    # Calculate hit rate assuming a hit if "at least" one of topN predictions is in the 
    # flights actually flown in the validation set. 
    # Keep all recommended flights even if one or more D are found in the training set. 

    # Do not filter out hits in the training set

    #====================================================================================================

    # Filter out hits in the training set
    # Inefficient implementation
    # Calculate the number of hits when filtered
    # Calculate hit rate assuming a hit if "at least" one of topN predictions is in the 
    # flights actually flown in the validation set. 
    # Remove from the recommended flights, any flights found in the training set. 

    print("res1.columns: ", res1.columns)
    # return
    res2 = extract_topN(res1, topn=8*topN)

    print("train_dest_sets: ", train_dest_sets.shape, train_dest_sets.columns)  # (18578,2) (D, MEMBER_ID) (WHY THIS SHAPE?)

    # Storing lists in Dataframe columns is not a good idea. 

    # Add a new column with the destinations to check in the recommender (all destinations minus the ones in the training set)
    #print("train_dest_sets: ", train_dest_sets)

    # Collect the members in the validation set
    trai = train_dest_sets[train_dest_sets.MEMBER_ID.isin(valid_dest_sets.MEMBER_ID)]  # (16669, 2)
    #print(f"valid_dest_sets: {valid_dest_sets.shape}\n")  # (26235, 2)
    #print("nb destinations in training set: ", all_dest)

    #print(f"trai (members in validation set): {trai.shape}\n", trai)
    all_dest_set = set(all_dest)
    trai['neg_set'] = trai['D'].map(lambda x: all_dest_set.difference(x))
    #print(f"trai (members in validation set): {trai.shape}\n", trai)


    # compute pairs when each member has a different number of destinations
    df_flat = hetero_MD_df(members, trai)
    print(f"df_flat.shape: {df_flat.shape}\n", df_flat.head(5))

    return # 6.8
    raise "ERROR"

    #----------------------

    compute_hits_2_faster(members, train_dest_sets, valid_dest_sets, res2)

    # I should also print how many are correct out of the number of actual trips. TO DO.  MYHouse 
    # Identical to the hit rate built into rankfm! That is great!
    print("no filter, nb_members: ", nb_members)
    hr_filtered = hits/nb_members
    print("hit rate (with previous filter) = ", hr_filtered)

    return pairs, hr_notfiltered, hr_filtered
#---------------------------------------------------------------------------------------------------
def restrict_member_attrib(dct):
    """
    Restrict user_features in dct['data_valid'to members in data_train. The user_features will not 
    change between runs. 

    Return: 
    -------
    user_attrib: DataFrame
        The unique members match the unique members of data_train, a requirement of rankfm. 
    """
    print(dct.keys())
    data_train = dct['data_train']
    user_attrib = dct['data_valid']
    user_attrib = user_attrib[user_attrib['MEMBER_ID'].isin(data_train.MEMBER_ID)]
    return user_attrib

#----------------------------------------------------------------------------------------------
