All work on recommender systems occurs under copa_recommender_systems/ folder. 

Scripts: 

===============================================================

# members.ipynb: 
Read in the initial ACTIVITY file and do the following: 
- read dates and transform them to timestamps (in seconds)
- only keep rows where OPERATING_COMPANY == 'CM'
- only keep rows where HOLDING_OD == TRUE_OD
- only keep columns deemed useful

These operations ensure that only destinations managed by Copa are considered.

===============================================================

===> analyze_pnr.ipynb <===

Read the fie: "reduced_df.csv", 990,144 records. 
Produce the file: "members_d.csv" with columns: MEMBER_ID, D, FLIGHT_DATE, family_size


# Reduce rows to one per family
* Starting point: A file which only contains the last flight segment of the TRUE_OD itinerary.
    * This was identified by equating the destination of TRUE_OD and the destination of SEGMENT_OD
    * I only keep flights where the country of origin corresponds to the country of the country of the origin

# Remove flights within country:  shape oes from 990144 => 910561

# Analyze membership, families, etc
* There is already a 1-1 correspondance between PNR and member ids
* Rows with undefined PNR have been removed (about 50k rows)

## Remove all records where one PNR has multiple MEMBER_IDs
* At least until we understand this case better

### How many TICKET_NUMBERs for each PNR?
* I would expect one per person if several people are covered. 
* How many MEMBER_IDs per PNR? I would expect 1

### (PNR, FLIGHT#, FLIGHT_date) should be unique
* different ticket numbers corresponding to a reservation 
  made by a single member that are the same flight should be counted as 1. 
* The number of different tickets would be the family size (TODO)

### One record per family
nb records left 865,446
There are 50,793 members. 

## 50793 member ids, 862409 PNRs. 

### OUTPUT FILE ("member_d.csv")
with columns: MEMBER_ID, D, FLIGHT_DATE, family_size

===================================================

Using Surprise Library
surprise_SVD_Copa.ipynb 

Read in the file member_d.csv as a Pandas dataframe. Use Surprise library in Python to do a matrix
factorization. 

===================================================
analyze_PNR.ipynb 
produces the file  "activity_reduced_before_attributes.csv"
that will be used as a starting point to compute attributes to be used in Factorizaiton Machine
