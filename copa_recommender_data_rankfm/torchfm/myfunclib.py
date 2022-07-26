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
def train_epoch(model, optimizer, data_loader, criterion, device, log_interval=100):
    print("INSIDE train_epoch")
    # Model is already on the device
    model.train()
    total_loss = 0
    # tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=5.0, nrows=5)
    losses = []
    dct = {}
    count = 0
    #for i, (fields, neg, target) in enumerate(data_loader):
    print("before for")
    print("data_loader: ", data_loader)
    for i, fields in enumerate(data_loader):
        #print("inside for")   # NOT PRINTING. WHY?
        print("i= ", i)
        #print("len fields= ", len(fields))
        batch_size = data_loader.batch_size
        user_attr = fields[0]  # member_id, D, attr
        pos_attr = fields[1]   # member_id, D, attr(D)
        neg_attr = fields[2]   # member_id, D, negD, attr(negD)
        # Remove
        #print("1 pos_attr: \n", pos_attr[0,:])
        #print("1 neg_attr: \n", neg_attr[0,:])
        pos_attr = pos_attr[:, 1:]  # <<< ERROR
        neg_attr = neg_attr[:, 2:]
        #print("1 pos_attr: \n", pos_attr[0,:])
        #print("1 neg_attr: \n", neg_attr[0,:])
        target = fields[3]
        print("2 user_attr: \n", user_attr[0,:])
        # I need to remove the second column from user_attr. Alternatively, just leave it, 
        # and modify input to the NN FM model.
        print("2 pos_attr: \n", pos_attr[0,:])
        print("neg_attr: \n", neg_attr[0,:])
        print("target: \n", target[0])
        print("after for, fields, user, pos, neg, target shapes: ", 
            len(fields), user_attr.shape, pos_attr.shape, neg_attr.shape, target.shape) # (B,3), (B,1), (B,1)
        # More efficient to collect tensors together on CPU and send them all at once

        # Target is always 1
        user_attr, pos_attr, neg_attr, target = user_attr.to(device), pos_attr.to(device), neg_attr.to(device), target.to(device)

        # The 2nd field is the destination
        #fields_pos = fields
        #a1 = fields[:,0].unsqueeze(1)  # (B,1)  # Why choose the 0th field? 
        #a2 = neg[:,0].unsqueeze(1)  # (B,1)
        #a3 = fields[:, 2:]   # (B, n)  (up to n attributes). fields[:,0] and fields[:1] are users and items
        #print("a1,a2,a3 shapes after unsqueeze: ", a1.shape, a2.shape, a3.shape)  # (4096,1), (4096,1)
        #fields_neg = torch.cat((a1,a2,a3), 1)
        #print("field_neg shape: ", fields_neg.shape)   # (4096,3)

        target = target.unsqueeze(1)

        # let us not worry about efficiency
        print(user_attr.device, pos_attr.device, neg_attr.device, target.device)
        fields_pos = torch.cat([user_attr, pos_attr, target], axis=1)
        fields_neg = torch.cat([user_attr, neg_attr, target], axis=1)

        print(fields_neg.shape, fields_pos.shape)
        print("pos: ", fields_pos[0,:])
        print("neg: ", fields_neg[0,:])

        # Deterministic model
        ypos = model(fields_pos)   # <<<< ERROR
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
    fields_ = []
    predicts = []
    count = 0
    # (d0, d1, d2): fields, neg, target
    for i, (d0, d1, age)  in enumerate(data_loader):
        fields, neg, target = d0.to(device), d1.to(device), age.to(device)
        # d1 is negative element
        # Why is age third element of the data_loader?
        fields_pos = fields   # (B,3)

        with torch.no_grad():
            y = model(fields)

        predicts.append(y)
        fields_.append(fields)

    predicts = torch.cat(predicts[1:], axis=0)
    fields_  = torch.cat(fields_[1:], axis=0)

    predicts = predicts.to('cpu')
    fields_ = fields_.to('cpu')

    # Create a single dataframe
    #return roc_auc_score(targets, predicts)  # ERROR IN THIS ROUTINE

    return fields_, predicts

#--------------------------------------------------------------------------
