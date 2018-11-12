'''
ifn.py
version 0.1.3

Traffic Assignment based on Ideal Flow Network

@Author: Kardi Teknomo
Date: November 12, 2018

Last Changes:
+ Class
+ network color

'''
import numpy as np
import math
import os
import sys
import IdealFlowNetwork as ifn
import display_network as dn

class IFN():
    def __init__(self, scenario):
        self.readScenario(scenario)
        self.mLinkInsertMaxSpeedCapacity()   # add max speed and capacity
        C=self.mLink2WeightedAdjacency(self.mLink,fieldNo=6) # capacity
        U=self.mLink2WeightedAdjacency(self.mLink,fieldNo=5) # max Speed
        L=self.mLink2WeightedAdjacency(self.mLink,fieldNo=3) # link distance
        S=ifn.capacity2stochastic(C)               # Markov stochastic
       
        # first try at arbitrary kappa=100
        pi=ifn.steadyStateMC(S,kappa=100)          # node values
        F=ifn.idealFlow(S,pi)                      # ideal flow
        G=self.HadamardDivision(F,C)               # congestion
        maxCongestion=np.max(G)

        if self.calibrationBasis=="flow":
            # calibrate with new kappa to reach totalFlow
            kappa=totalFlow
        else: # calibrationBasis=="congestion"
            # calibrate with new kappa to reach max congestion level
            kappa=100*float(self.maxAllowableCongestion)/maxCongestion # total flow

        # compute ideal flow and congestion
        pi=ifn.steadyStateMC(S,kappa)                         # node values
        F=ifn.idealFlow(S,pi)                                 # scaled ideal flow
        G=self.HadamardDivision(F,C)                               # congestion
        maxCongestion=np.max(G)

        # compute link performances
        self.mLink=self.addFlow2mLink(self.mLink,F) # fieldNo=7 flow
        self.mLink=self.addFlow2mLink(self.mLink,G) # fieldNo=8 congestion level
        self.mLink=self.computeLinkPerformance(self.mLink,self.travelTimeModel,self.cloudNode) # fieldNo=9 to 11
    
        # save output mLink
        mR,mC=self.mLink.shape
        fmt="%d,%d,%d,%0.3f,%d,%0.3f,%d,%0.3f,%0.3f,%0.3f,%0.3f,%0.3f"
        header="LinkNo,Node1,Node2,Dist,Lanes,MaxSpeed,Capacity,Flow,Congestion,Speed,TravelTime,Delay"
        mLink2=self.mLink.T
        with open(self.folder+self.scenarioName+'.csv', 'w') as fh:
            for j in range(mC):
                col=mLink2[j,:]
                if j==0:
                    np.savetxt(fh, col.reshape(1, -1), fmt=fmt,header=header,delimiter=',')
                else:
                    np.savetxt(fh, col.reshape(1, -1), fmt=fmt,delimiter=',')

        # network performance
        avgSpeed=np.nanmean(self.mLink[9,:])
        avgTravelTime=np.nanmean(self.mLink[10,:])
        avgDelay=np.nanmean(self.mLink[11,:])
        avgDist=np.nanmean(self.mLink[3,:])

        # save network performance
        with open(self.folder+self.scenarioName+'.net', 'w') as fh:
            fh.write("totalFlow="+str(kappa)+"\n")              # in pcu/hour
            fh.write("maxCongestion="+str(maxCongestion)+"\n")  
            fh.write("avgSpeed="+str(avgSpeed)+"\n")            # in km/hour
            fh.write("avgTravelTime="+str(avgTravelTime)+"\n")  # in hour
            fh.write("avgDelay="+str(avgDelay)+"\n")            # in hour
            fh.write("avgDist="+str(avgDist)+"\n")              # in meter

        # report
        print(self.scenarioName)
        print("Network performance:")
        print("\tTotal Flow = ", round(kappa,2)," pcu/hour")
        print("\tMax Congestion = ",round(maxCongestion,4))
        print("\tAvg Link Speed =",round(avgSpeed,4)," km/hour")
        print("\tAvg Link Travel Time = ",round(1000*60*avgTravelTime/avgDist,4)," min/km")
        print("\tAvg Link Delay = ",round(1000*3600*avgDelay/avgDist,4), " seconds/km")
        print("Basis:")
        print("\tAvg Link Distance = ",round(avgDist,4), " m/link")
        print("\tAvg Link Travel Time = ",round(3600*avgTravelTime,4)," seconds/link")
        print("\tAvg Link Delay = ",round(3600*avgDelay,4), " seconds/link")
        arrThreshold=[0.8,0.9,1]
        plt=dn.display_network(self.mLink.T,8,self.mNode.T,arrThreshold,9) # display congestion
        plt.show()


    def readScenario(self,scenario):
        '''
        parse scenario, read node file and link file
        return node matrix and link matrix
        the fields are in rows
        '''
        # initialize the default values
        self.travelTimeModel=None
        self.maxAllowableCongestion=1
        self.totalFlow=None
        self.calibrationBasis=None
        self.cloudNode=None
        self.capacityBasis=None
        
        # read scenario
        self.folder=os.path.dirname(scenario)
        if self.folder!="":
            self.folder=self.folder+"\\"
        lines=open(scenario,"r").read().splitlines()

        # parsing scenario
        for item in lines:
            (lhs,rhs)=item.split('=')
            if lhs=='ScenarioName':
                self.scenarioName=rhs
            if lhs=='Node':
                self.mNode=self.readCSVFileSkipOneRow(self.folder+rhs)
                self.mNode=self.mNode.T
            if lhs=="Link":
                self.mLink=self.readCSVFileSkipOneRow(self.folder+rhs)
                self.mLink=self.mLink.T
            if lhs=='maxAllowableCongestion':
                self.maxAllowableCongestion=rhs
            if lhs=='travelTimeModel':
                self.travelTimeModel=rhs
            if lhs=='totalFlow':
                self.totalFlow=float(rhs)
            if lhs=='calibrationBasis':
                self.calibrationBasis=rhs
            if lhs=='cloudNode':
                self.cloudNode=rhs
            if lhs=='capacityBasis':
                self.capacityBasis=rhs


    def readCSVFileSkipOneRow(self,fileName):
        return np.loadtxt(open(fileName, "r"), delimiter=",", skiprows=1)


    def HadamardDivision(self,A,B):
        '''
        return A./B with agreement 0/0=0
        '''
        B[B==0]=np.inf        
        return np.divide(A,B)


    def mLinkInsertMaxSpeedCapacity(self):
        '''
        assume order field in mLink :=
               0    1     2     3    4      5       6
            LinkNo,Node1,Node2,Dist,Lanes,MaxSpeed,Capacity
        return mLink after insertion of maxSpeed and Capacity fields

        if capacityBasis='width' then field[4]=road width
        if capacityBasis='lanes' then field[4]=Lanes
        '''
        mR,mC=self.mLink.shape
        if mR<6: # if maxSpeed is missing
            arrSpeed=[]
            for j in range(mC):
                if self.capacityBasis=='width':
                    roadWidth=self.mLink[4,j]
                    maxSpeed=20+15*(roadWidth/3-1)
                else:
                    numLane=self.mLink[4,j]
                    maxSpeed=20+15*(numLane-1)
                arrSpeed.append(maxSpeed)
            self.mLink=np.vstack([self.mLink,arrSpeed])
        if mR<7: # if capacity is missing
            arrCap=[]
            for j in range(mC):
                if self.capacityBasis=='width':
                    roadWidth=self.mLink[4,j]  # in meter per direction
                    capacity=500*roadWidth # in pcu/hour
                else:
                    numLane=self.mLink[4,j]     # number of lanes per direction
                    capacity=1500*numLane  # in pcu/hour
                arrCap.append(capacity)
            self.mLink=np.vstack([self.mLink,arrCap])
        return self.mLink


    def addFlow2mLink(self,mLink,F):
        '''
        return mLink with additional row about F
        matrix F size is n by n
        '''
        mR,mC=mLink.shape
        
        arrF=[]
        for j in range(mC):
            r=int(mLink[1,j])-1
            c=int(mLink[2,j])-1
            v=F[r,c]
            arrF.append(v)
        mLink=np.vstack([mLink,arrF])
        return mLink


    def computeLinkPerformance(self,mLink,travelTimeModel=None,cloudNode=None):
        '''
        return mLink with additional link performance
        
        '''
        mR,mC=mLink.shape
        arrSpeed=[]
        arrTravelTime=[]
        arrDelay=[]
        for j in range(mC):
            node1=int(mLink[1,j])
            node2=int(mLink[2,j])
            if cloudNode is not None and (cloudNode==str(node1) or cloudNode==str(node2)):
                speed=np.nan            # v
                travelTime=np.nan       # t
                minTravelTime=np.nan    # t0
                delay=np.nan            # delta
            else:
                maxSpeed=mLink[5,j]     # u in km/hour
                dist=mLink[3,j]/1000    # d in km; mLink[3,j] in meter
                congestion=mLink[8,j]   # g

                if travelTimeModel=='Greenshield':
                    # based on greenshield
                    speed=maxSpeed/2*(1+math.sqrt(1-congestion)) # v in km/hour
                    travelTime=dist/speed       # t   in hour
                    minTravelTime=dist/maxSpeed # t0  in hour
                    delay=travelTime-minTravelTime # delta in hour
                else:
                    # based on BPR (by default)
                    minTravelTime=dist/maxSpeed    # t0  in hour
                    travelTime=minTravelTime*(1+0.15*congestion**4) # t in hour
                    speed=dist/travelTime          # v  in km/hour
                    delay=travelTime-minTravelTime # delta in hour
                
            arrSpeed.append(speed)
            arrTravelTime.append(travelTime)
            arrDelay.append(delay)
        mLink=np.vstack([mLink,arrSpeed])      # fieldNo=9 link speed
        mLink=np.vstack([mLink,arrTravelTime]) # fieldNo=10 link travel time
        mLink=np.vstack([mLink,arrDelay])      # fieldNo=11 link delay
        return mLink
        


    def mLink2Adjacency(self,mLink):
        '''
        assume order field in mLink :=
               0    1     2     3    4      5       6
            LinkNo,Node1,Node2,Dist,Lanes,MaxSpeed,Capacity
        fields of mLinks are in rows
        return adjacency matrix
        '''
        # get unique node IDs from second and third fields of mLink
        nodeIds=np.union1d(mLink[1,:],mLink[2,:])
        n=np.prod(nodeIds.shape)
        A=np.zeros((n,n))
        # fill up with 1 when there is a link
        coord=zip(mLink[1,:],mLink[2,:])
        for item in coord:
            (r,c)=item
            A[int(r)-1,int(c)-1]=1
        return A

    
    def mLink2WeightedAdjacency(self,mLink,fieldNo=6):
        '''
        assume order field in mLink :=
               0    1     2     3    4      5       6
            LinkNo,Node1,Node2,Dist,Lanes,MaxSpeed,Capacity
        fields of mLinks are in rows
        return capacity matrix (by default)
        but depending on the fieldNo, it can also return Dist,Lanes,MaxSpeed
        '''
        # get unique node IDs from second and third fields of mLink
        nodeIds=np.union1d(mLink[1,:],mLink[2,:])
        n=np.prod(nodeIds.shape)
        A=np.zeros((n,n))
        # fill up with 1 when there is a link
        coord=zip(mLink[1,:],mLink[2,:],mLink[fieldNo,:])
        for item in coord:
            (r,c,k)=item
            
            A[int(r)-1,int(c)-1]=k
        return A


if __name__ == '__main__':
    if len(sys.argv)>1:
        scenario = sys.argv[1]
        if scenario=="":
            print("to use: input the scenario file (including the folder name)")
    else:
        scenario='C:\Data\simplestScenario\scenario2.txt'
    IFN(scenario)
    
