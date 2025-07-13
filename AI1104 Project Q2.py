import numpy as np  #declaring the necessary functions
import matplotlib.pyplot as plt

'''defining to manipulate changes in the mapping dictionary and centroids
    and controlling the iterations'''
def assign_clusters(data_points,C_1,C_2,mappings) : 
    number=0    #number for furthur use 
    '''Initialised coordinates to assign clusters of same centroids'''
    B_1=np.array([0,0]).astype(np.float64)
    B_2=np.array([0,0]).astype(np.float64)
    length_1=0  #length of the cluster of data points certain 
    length_2=0  

    for data in data_points:    
        x=mappings[data]    #to conclude the number of points updated
        if(np.linalg.norm(data-C_1)<np.linalg.norm(data-C_2)):  #To check the point's Euclidean distance from
            mappings[data]=1                                    #each of the two centroids assigned
            B_1+=data   #to sum up the data points of the same centroid
            length_1+=1 #to conclude the average 

        else:
            mappings[data]=2  #declaring the cluster with the condition above
            B_2+=data 
            length_2+=1

        if(x!=mappings[data]) : 
            number+=1   #increasing number based on number of points undated
    C_1=B_1/length_1    #updating the centroid values
    C_2=B_2/length_2
    return number

#Question 2.1 starts

theta_arr=np.linspace(0,np.pi,500)  #generating 500 eqidistant values between 0 and pi
np.random.seed(123) #random number generator

semicircle1_x = np.cos(theta_arr)   #calculating the x co-ordinates with theta 
noice_x=np.random.normal(loc=0,scale=0.1,size=500) #create the disturbances to the x co-ordinates
semicircle1_x = semicircle1_x + noice_x #calculating the x co-ordinates with the disturbances

semicircle1_y = np.sin(theta_arr)   #calculating the y co-ordinates with the theta
noice_y=np.random.normal(loc=0,scale=0.1,size=500)  #create the disturbances to the y co-ordinates
semicircle1_y = semicircle1_y + noice_y  #calculating y co-ordinates with the disturbances

semicircle2_x = 1+np.cos(theta_arr)     #calculating the x co-ordinates with theta
semicircle2_y = -np.sin(theta_arr)      #calculating the y co-ordinates with the theta

semicircle2_x = semicircle2_x + noice_x #adding disturbances
semicircle2_y = semicircle2_y + noice_y   #adding disturbances

#plotting the data sets with no distinguishion between those two semi cirles
plt.scatter(x=semicircle1_x, y=semicircle1_y,color='blue')  
plt.scatter(x=semicircle2_x, y=semicircle2_y,color='blue')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title("Original Dataset")
plt.grid(True)
plt.savefig("Ori_dataset.png")  #to save the picture
plt.show()

d1 = [(x,y) for x,y in zip(semicircle1_x,semicircle1_y)]    #unzipping the data and satoring the coordinates
d2 = [(x,y) for x,y in zip(semicircle2_x,semicircle2_y)]
data_points = d1 + d2   #storing in one data set
#Question 2.1 ends

#Question 2.2 starts
a1,a2=(-2+5*np.random.random(size=2)).astype(np.float64)    #randomly choosing the x of centroids
b1,b2=(-2+4*np.random.random(size=2)).astype(np.float64)    #randomly choosing the y of centroids

C_1=[a1,b1]    #creating the centroids
C_2=[a2,b2]

C_1=np.array(C_1)   #converting to array to manipulate changes in it 
C_2=np.array(C_2)

mappings={} #to store the data and clustering
for data in data_points:
    mappings[data]=np.random.choice(range(1,3)) #to randomly store cluster where the data belongs

num_points_updated=0    #the number of points whose cluster was updated during an iteration.
before_number=1 #for customization in the upcoming while loop
max_iters=10000 #stop the iteration when it reaches 10000 iterations
current_iters=0 #count the number of iterations


    
while(max_iters>current_iters and before_number!=num_points_updated) :  #condition based on the given instructions
    before_number=num_points_updated    #for the check condition
    num_points_updated=assign_clusters(data_points,C_1,C_2,mappings)    #implementing the defined functions
    current_iters+=1    #continuosly increasing the iterations

'''declaring lists to plot along the x , y coordinates for the refraimed clusters'''
x_1=[]
y_1=[]
x_2=[]
y_2=[]

'''data[0]= x co-ordinate and data[1]= y co-ordinate'''

for data,clust in mappings.items(): #to open up the items in mappings
    if(clust==1):   #condition to check
        x_1.append(data[0]) #collection of x coordinates of 1st cluster
        y_1.append(data[1]) #collection of y coordinates of 1st cluster
    else:      
        x_2.append(data[0]) #2nd cluster's x coordinates
        y_2.append(data[1]) #2nd cluster's y coordinates

#plotting the scatter points of both clusters in one plot
plt.scatter(x=x_2, y=y_2,color='black',label='Cluster 1') #first cluster
plt.scatter(x=x_1, y=y_1,color='green',label='Cluster 2')  #second cluster
plt.xlabel('X position')
plt.ylabel('Y position')
plt.title("K-Means Clusters")   #this graph gives the two clusters separated by colors 
plt.legend()
plt.grid(True)
plt.savefig("k_means_clusters.png") 
plt.show()

'''we finally obtained the k mean clusterings of the two clusters from the original dataset'''
#Question 2.2 ends