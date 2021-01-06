"""
this file generate a list to indicate which group this voxel is in
"""
import numpy as np
import hdf5storage
import random

def generate_group_indexes(coordinates, voxel_num, ridius, method):
    """
    method: 0: original; 1: reverse; 2: random;
    """
    group_idx_vec = np.zeros(voxel_num)
    group_idx = 1
    p1 = np.zeros(3)
    p2 = np.zeros(3)

    if method == 0:
        iteration_list = range(voxel_num)
    elif method == 1:
        iteration_list = range(voxel_num - 1, -1, -1)
    elif method == 2:
        iteration_list = list(range(voxel_num))
        iteration_list = random.shuffle(iteration_list)

    for i in iteration_list:
        print(i)
        p1[0] = coordinates[i, 0]
        p1[1] = coordinates[i, 1]
        p1[2] = coordinates[i, 2]
        if group_idx_vec[i] == 0:
            group_idx_vec[i] = group_idx
            for j in range(voxel_num):
                if group_idx_vec[j] == 0:
                    p2[0] = coordinates[j, 0]
                    p2[1] = coordinates[j, 1]
                    p2[2] = coordinates[j, 2]
                    dist = np.linalg.norm(p1 - p2)
                    if dist < ridius:
                        group_idx_vec[j] = group_idx
            group_idx += 1
    return group_idx_vec


def generate_group_indexes_furthest(coordinates, voxel_num, ridius):
    """
    Next group is furthest from previous group
    """
    group_idx_vec = np.zeros(voxel_num)
    group_idx = 1
    p1 = np.zeros(3)
    p2 = np.zeros(3)

    i = 0
    while True:
        print(i)
        # get the group with point[i]
        p1[0] = coordinates[i, 0]
        p1[1] = coordinates[i, 1]
        p1[2] = coordinates[i, 2]
        if group_idx_vec[i] == 0:
            group_idx_vec[i] = group_idx
            for j in range(voxel_num):
                if group_idx_vec[j] == 0:
                    p2[0] = coordinates[j, 0]
                    p2[1] = coordinates[j, 1]
                    p2[2] = coordinates[j, 2]
                    dist = np.linalg.norm(p1 - p2)
                    if dist < ridius:
                        group_idx_vec[j] = group_idx
            group_idx += 1
        # find next point which is furthest from point[i]
        if np.min(group_idx_vec) == 0:
            dist_i = np.linalg.norm(p1 - coordinates, axis=1)
            sorted_index = np.argsort(dist_i)[::-1]
            for i in sorted_index:
                if group_idx_vec[i] == 0:
                    break
        else:
            break

    return group_idx_vec

def generate_group_indexes_furthest_toall(coordinates, voxel_num, ridius):
    """
    Next group should be furthest from all existing groups
    """
    group_idx_vec = np.zeros(voxel_num)
    group_idx = 1
    p1 = np.zeros(3)
    p2 = np.zeros(3)
    center_set = set()

    i = 0
    while True:
        print(i)
        center_set.add(i)
        # get the group with point[i]
        p1[0] = coordinates[i, 0]
        p1[1] = coordinates[i, 1]
        p1[2] = coordinates[i, 2]
        if group_idx_vec[i] == 0:
            group_idx_vec[i] = group_idx
            for j in range(voxel_num):
                if group_idx_vec[j] == 0:
                    p2[0] = coordinates[j, 0]
                    p2[1] = coordinates[j, 1]
                    p2[2] = coordinates[j, 2]
                    dist = np.linalg.norm(p1 - p2)
                    if dist < ridius:
                        group_idx_vec[j] = group_idx
            group_idx += 1
        # find next point which is furthest from point[i]
        if np.min(group_idx_vec) == 0:
            dist_i = dist_func(center_set, coordinates)
            sorted_index = np.argsort(dist_i)[::-1]
            for i in sorted_index:
                if group_idx_vec[i] == 0:
                    break
        else:
            break

    return group_idx_vec


def dist_func(center_set, coordinates):
    p1 = np.zeros(3)
    distance_sum = np.zeros(coordinates.shape[0])
    for i in list(center_set):
        p1[0] = coordinates[i, 0]
        p1[1] = coordinates[i, 1]
        p1[2] = coordinates[i, 2]
        distance_sum += np.linalg.norm(p1 - coordinates, axis=1)
    return distance_sum


def main():
    voxel_num = 429655
    ridius = 12
    grouping_method = 4  # 0: original; 1: reverse; 2: random; 3: furthest
    print('r = {}, method = {}'.format(ridius, grouping_method))

    # read coordinates
    coordinates = hdf5storage.loadmat('mni_mask.mat')
    coordinates = coordinates['mni_mask']
    if grouping_method == 0 or grouping_method == 1 or grouping_method == 2:
        group_idx_vec = generate_group_indexes(coordinates, voxel_num, ridius, grouping_method)
    elif grouping_method == 3:
        group_idx_vec = generate_group_indexes_furthest(coordinates, voxel_num, ridius)
    elif grouping_method == 4:
        group_idx_vec = generate_group_indexes_furthest_toall(coordinates, voxel_num, ridius)
    print(np.max(group_idx_vec))
    np.save('group_idx_m_{}_r_{}.npy'.format(grouping_method, ridius), group_idx_vec)

if __name__ == '__main__':
    main()