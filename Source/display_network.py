import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.drawing.nx_agraph import to_agraph
import os
os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

def readCSVFileSkipOneRow(fileName):
    return np.loadtxt(open(fileName, "r"), delimiter=",", skiprows=1)


def display_network(mLink,fieldNo,mNode=None,arrThreshold=None,ignoreFieldNo=None):
    mR,mC=mLink.shape
    G = nx.DiGraph()
    arr=mLink[:,fieldNo]
    m=max(arr)

    if mNode is None:
        nodeIds=np.union1d(mLink[:,1],mLink[:,2])
        mRn=len(nodeIds)
    else:
        mRn,mCn=mNode.shape
        for r in range(mRn):
            nodeID=int(mNode[r,0])
            x=mNode[r,1]
            y=mNode[r,2]
            G.add_node(nodeID,pos=(x,y))
    
    for j in range(mR):
        node1=int(mLink[j,1])
        node2=int(mLink[j,2])
        weight=mLink[j,fieldNo]#/m
        if ignoreFieldNo is not None:
            ignoreField=mLink[j,ignoreFieldNo]
        else:
            ignoreField=-1
        if not np.isnan(ignoreField):            
            G.add_edge(node1, node2, weight=weight)

    if mNode is None:
        pos = nx.spring_layout(G)  # if node position is set automatically
    else:
        pos = nx.get_node_attributes(G,'pos') # if node position are given

    # nodes
    nodes=nx.draw_networkx_nodes(G, pos, node_color ='w', node_size=200)
    nodes.set_edgecolor('b')

    # edges
    all_weights=[] # edge thickness
    for (node1,node2,data) in G.edges(data=True):
        all_weights.append(data['weight']) 
    unique_weights = list(set(all_weights))
    for weight in unique_weights: # draw one by one per weight
        weighted_edges = [(node1,node2) for (node1,node2,edge_attr) in G.edges(data=True) if edge_attr['weight']==weight]
        width = 5*weight*mRn/sum(all_weights)
        if arrThreshold is not None:
            if weight<arrThreshold[0]:
                color='green'
            elif weight<arrThreshold[1]:
                color='yellow'
            else:
                color='red'
        else:
            color='black'
##        print(weight,color,arrThreshold)
        nx.draw_networkx_edges(G,pos,edgelist=weighted_edges,width=width,edge_color=color)
    
    # labels
    nx.draw_networkx_labels(G, pos, font_size=6, font_family='sans-serif')
    edge_labels=dict([((u,v,),round(d['weight'],2))for u,v,d in G.edges(data=True)])
    nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,font_size=6, font_family='sans-serif')
    plt.axis('off')
    return plt

if __name__ == '__main__':
    fieldNo=4
    arrThreshold=[1.5,2.5,3]
    mLink=readCSVFileSkipOneRow("Link.txt")
    mNode=readCSVFileSkipOneRow("Node.txt")
    ignoreFieldNo=1
##    mNode=None
    plt=display_network(mLink,fieldNo,mNode,arrThreshold)
    plt.show()
    
