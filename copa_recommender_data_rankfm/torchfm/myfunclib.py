import random
import pandas as pd
import numpy as np
import torch 
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
    def __init__(self, filenm, nrows='all'):
        super(Dataset, self).__init__()
        if nrows == 'all':
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
        # print(df['age_at_flight'])         # <<<<<
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
        #print("p1: ", p1)
        #print("gordon")
        self.member_dict = sorted(p1, key=lambda x: x[0]) # idx -> member
        #print("fan")
        #print(self.member_dict)
        self.member_dict = {item[0]:item[1:] for item in self.member_dict}
        #print(self.member_dict)
        #print("fan")
        #self.member_dict = dict(sorted(p1, key=lambda x: x[0])) # idx -> member
        #self.member_dict = dict(p1)
        self.idx2member = self.member_dict
        #print("idx2member[0]: ", self.idx2member[0])
        #print("idx2member.items(): ", self.idx2member.items())
        self.member2idx = {v[0] : (k,v[1:]) for k,v in self.idx2member.items()}
        #print("self.member2idx: ", self.member2idx)
        
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
        
        dct = {}
        dct['member_attr'] = self.member_attr
        dct['dest_attr'] = self.dest_attr
        dct['field_dims'] = self.field_dims
        dct['D_set'] = self.D_set
        dct['dest2idx'] = self.dest2idx
        dct['idx2dest'] = self.idx2dest
        dct['member2idx'] = self.member2idx
        dct['idx2member'] = self.idx2member
        self.dct = dct
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx,:]
        member_id = row['MEMBER_ID']
        diff_set = self.unique_D - self.D_set.loc[member_id, :].values[0]
        # Pick a random element from the set
        neg = random.choice(list(diff_set))  # negative sample
        return tt(row, dtype=torch.int), tt([neg]), tt([1.])
    
    def __len__(self):
        return self.df.shape[0]

#-----------------------------------------------------------------------
def getData(file, shuffle=True, batch_size=32, nrows='all'):
    """
    Arguments:
    ---------
    file: file to process
    batch_size: default: 32
    nrows: default: 'all'  (read all rows)
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
        print("len: ",  self.n_members * self.n_dest)
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
def train_epoch(model, optimizer, data_loader, criterion, device, log_interval=100):
    # Model is already on the device
    model.train()
    total_loss = 0
    # tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=5.0, nrows=5)
    losses = []
    dct = {}
    count = 0
    #for i, (fields, neg, target) in enumerate(tk0):
    for i, (fields, neg, target) in enumerate(data_loader):
        # More efficient to collect tensors together on CPU and send them all at once
        fields, neg, target = fields.to(device), neg.to(device), target.to(device)

        # The 2nd field is the destination

        fields_pos = fields
        a1 = fields[:,0].unsqueeze(1)  # (B,1)
        a2 = neg[:,0].unsqueeze(1)  # (B, 1)
        a3 = fields[:, 2:]   # (B, n)
        fields_neg = torch.cat((a1,a2,a3), 1)

        # Deterministic model
        ypos = model(fields_pos)
        yneg = model(fields_neg)

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
        total_loss = total_loss / data_loader.dataset.df.shape[0]
    return(total_loss)

#-----------------------------------------------------------------------
def test_accuracy(model, data_loader, device):
    model.eval()
    targets  = [torch.empty([]).unsqueeze(0)] # 1D tensor
    #print(targets[0].shape, targets[0])
    predicts = [torch.empty([]).unsqueeze(0)] #data_loader.batch_size])]
    fields_ = [torch.empty([1,3])]
    #print("before loop: len(predicts): ", len(predicts))
    count = 0
    predicts = [torch.empty([]).unsqueeze(0)] #data_loader.batch_size])]
    #print("data_loader: ", data_loader, data_loader.batch_size)
    for i, (d0, d1,age)  in enumerate(data_loader):
        #print("d0,d1,d2: ", d0, d1, age)
        #print("d0: ", d0.max())
        d0 = d0.unsqueeze(1)
        d1 = d1.unsqueeze(1)
        d2 = age.unsqueeze(1)
        d0, d1, d2 = d0.to(device), d1.to(device), d2.to(device)
        #print(d0.shape, d1.shape, d2.shape)
        fields = torch.cat([d0,d1,d2], axis=1)

        #print("before model")
        # Change model to take
        with torch.no_grad():
            y = model(fields)

        count = count + 1
        if count % 10 == 0:
            print(count, y.shape)

        # targets.extend(target.tolist())
        predicts.append(y)
        fields_.append(fields)
        #print("len(predicts): ", len(predicts))
        #print("y shape: ", y.shape)
        #print("y: ", y)

    predicts = torch.cat(predicts, axis=0)
    fields_  = torch.cat(fields_, axis=0)
    print("fields_: ", fields_.shape, predicts.shape)
    print(type(predicts))


    predicts = predicts.to('cpu')
    fields_ = fields_.to('cpu')

    # Create a single dataframe
    # return roc_auc_score(targets, predicts)

    return fields_, predicts

#--------------------------------------------------------------------------
