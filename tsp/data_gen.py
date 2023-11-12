import numpy as np
import pandas as pd
import math

def build_coordinates(coords):
    coords_string_list = [f'({i}): ({coords.iloc[i]["xs"]},{coords.iloc[i]["ys"]})' for i in range(len(coords))]
    return ', '.join(coords_string_list)

def tsp_data_gen(dim,ubound):
    i = range(0,dim)
    xs = np.random.randint((-ubound), (ubound+1), dim)
    ys = np.random.randint((-ubound), (ubound+1), dim)
    coords = pd.DataFrame({'is':i, 'xs': xs, 'ys': ys})

    return (coords, build_coordinates(coords))

def coords_dist(pt1, pt2):
    return math.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)


def tsp_ans(coords):
    pts = [(coords.iloc[i]['xs'],coords.iloc[i]['ys']) for i in range(0,len(coords))]

    # store total number of waypoints
    n = len(pts)

    # create waypoint distance matrix to store distances
    dist_matrix = []
    for i in range(0,n):
        for j in range(i,n):
            if i==0:
                if i==j:
                    dist_matrix.append([0])
                else:
                    d = coords_dist(pts[i], pts[j])
                    dist_matrix[i].append(d)
                    dist_matrix.append([d])
            else:
                if i==j:
                    dist_matrix[i].append(0)
                else:
                    d = coords_dist(pts[i], pts[j])
                    dist_matrix[i].append(d)
                    dist_matrix[j].append(d)


    # gist of tsp algorithm: build up sequences of possible distances from home node to j-nodes, while storing the minimal distance for each sequence at each stage

    # create matrix to store possible distances at each stage
    tsp_stages = [{}]

    # fill in bottom stage of tsp algorithm (distance from home node to all other nodes)
    for j in range(1,n):
        tsp_stages[0][str([j])] = ([j],dist_matrix[j][0])

    # fill in the rest of the stages for tsp algorithm
    for i in range(1,n-1):
        # add new stage
        tsp_stages.append({})

        # populate possibilites for new stage
        # iterate through all possible non-home nodes (these j-nodes are the ending nodes of the sequences for this stage)
        for j in range(1,n):
            # iterate through all the sequences from the previous stage
            for key in tsp_stages[i-1].keys():
                # add j-nodes to any sequence that does not already contain them
                if j not in tsp_stages[i-1][key][0]:
                    # troubleshooting: (print("j: " + str(j) + ", key: " + str(key) + ", l: " + str(tsp_stages[i-1][key][0]) + ", d: " + str(tsp_stages[i-1][key][1])))
                    # store the last node of sequence from previous stage (prev-node)
                    prev = tsp_stages[i-1][key][0][-1]
                    # troubleshooting: (print("prev: " + str(prev)))
                    # sort the sequence from previous stage so we can eliminate non-minimal paths later
                    l = tsp_stages[i-1][key][0][:]
                    lsort = l[:]
                    lsort.sort()
                    # add j-node to sorted sequence from previous stage (now j-node is last node of sequence)
                    l.append(j)
                    lsort.append(j)
                    # store new sequence as a string (to be used as a dictionary key)
                    keyl = str(lsort)
                    # calculate the new distance by adding the distance for the previous sequence to the distance between the j-node and prev-node
                    d = tsp_stages[i-1][key][1] + dist_matrix[j][prev]
                    # troubleshooting: (print("keyl: " + str(keyl) + ", l: " + str(l) + ", d: " + str(d)))
                    # add sequence and distance to dictionary if the sequence isn't already in the dictionary or if the sequence has minimal distance compared to sequences already in the dictionary
                    if keyl not in tsp_stages[i].keys() or tsp_stages[i][keyl][1] > d:
                        tsp_stages[i][keyl] = (l,d)


    # take all sequences back to home node

    # initialize variables to store path distance and sequence of locations
    path_dist = 0
    path_list = []

    # take all sequences from the final stage
    for key in tsp_stages[n-2].keys():
        # store the last node of each sequence (prev-node)
        prev = tsp_stages[n-2][key][0][-1]
        # find the total path distance by the adding path distance of each sequence to the distance between home node and prev-node
        d = tsp_stages[n-2][key][1] + dist_matrix[0][prev]
        # store the minimal total path distance
        if path_dist == 0 or d < path_dist:
            path_dist = d
            # store the sequence that resulted in the minimal total path distance
            l = tsp_stages[n-2][key][0][:]
            l.append(0)
            l.reverse()
            l.append(0)
            path_list = l
    
    # return the minimal total path distance and the sequence that resulted in the minimal total path distance
    return (path_dist, path_list)