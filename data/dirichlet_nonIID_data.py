import math
import hashlib
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

#《Federated Learning on Non-IID Data Silos: An Experimental Study》
#按Dirichlet分布划分Non-IID数据集：https://zhuanlan.zhihu.com/p/468992765
from data.util.custom_tensor_dataset import CustomTensorDataset

def compute_hash(data_point):
    # 将Tensor转换为numpy数组，然后计算其哈希值
    np_array = data_point.numpy()  # 将Tensor转换为numpy数组
    return hashlib.sha256(np_array.tobytes()).hexdigest()


def dirichlet_split_noniid(train_labels, alpha, n_clients, seed, overlap_ratio = 0.30): #设置每个客户端之间的重复率
    '''
    参数为alpha的Dirichlet分布将数据索引划分为n_clients个子集
    狄利克雷分布相关函数
    '''
    np.random.seed(seed)
    train_labels=train_labels.clone().detach()
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)  # 第一个参数是list，是n_clients个alpha
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]
    # 记录每个K个类别对应的样本下标

    # 选择一部分数据用于人为添加重复
    overlap_data_idcs = []
    for c in class_idcs:
        overlap_size = int(len(c) * overlap_ratio)
        overlap_data_idcs.extend(c[:overlap_size])
    overlap_data_idcs = np.array(overlap_data_idcs).flatten()  # 确保重叠数据索引是一维数组

    client_idcs = [[] for _ in range(n_clients)]  # 替换成DataFrame
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            # enumerate在字典上是枚举、列举的意思
            # client_idcs[i] += [idcs]
            client_idcs[i].extend(idcs)

    # 数据划分完成后，将重复数据按比率分配给客户端
    for i in range(n_clients):
        overlap_per_client = np.random.choice(overlap_data_idcs, size=len(overlap_data_idcs) // n_clients, replace=False)
        client_idcs[i] = np.concatenate((client_idcs[i], overlap_per_client))

    # client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    #修改
    client_idcs = [np.unique(idcs).flatten() for idcs in client_idcs]  # 移除重复的索引并确保是一维数组

    """这里返回的是一个二维list，每个二级list装了对应下标的client分配到的数据的索引"""
    # print(client_idcs)
    return client_idcs

def create_Non_iid_subsamples_dirichlet(n_clients, alpha, seed, train_data):
    """
    使用狄利克雷分布划分数据集
    x是数据，y是标签
    @Author:LingXinPeng
    """
    if train_data.data.ndim==4:  #默认这个是cifar10,下面的transforms参数来源于getdata时候的参数
        transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]
        )

    else:  #这个是mnist和fmnist数据
        train_data.data = torch.unsqueeze(train_data.data, 3)  #升维为NHWC，默认1通道。这边注意我们不需要转换维度，CustomTensorDataset包装后，后面会自动转换维度
        transform = torchvision.transforms.ToTensor()

    # 这里返回的是一个二维list，每个二级list装了对应下标的client分配到的数据的索引
    train_labels = train_data.targets.clone().detach()   #得到全部样本的标签

    client_idcs = dirichlet_split_noniid(train_labels, alpha, n_clients, seed)
    clients_data_list=[]

    #展示数据分配完成后，客户端数据的标签分布情况
    classes = train_data.classes
    n_classes = len(classes)
    # labels = np.concatenate(
    #     [np.array(train_data.targets), np.array(test_data.targets)], axis=0)
    labels = np.array(train_data.targets)
    plt.figure(figsize=(12, 8))
    label_distribution = [[] for _ in range(n_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, n_clients + 1.5, 1),
             label=classes, rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client %d" %
                                      c_id for c_id in range(n_clients)])
    plt.xlabel("Client ID")
    plt.ylabel("Number of samples")
    plt.legend()
    plt.title("Display Label Distribution on Different Clients")
    plt.show()

    for i in range(n_clients):
        indices = np.sort(client_idcs[i])
        indices=torch.tensor(indices)
        image = torch.index_select(train_data.data.clone().detach(), 0, indices)
        targets=torch.index_select(train_labels,0,indices)
        # 收集客户端数据的哈希值
        client_hashes = [compute_hash(data_point) for data_point in image]
        # print(client_hashes)
        data_info=CustomTensorDataset((image,targets), transform, client_hashes)
        clients_data_list.append(data_info)

    return clients_data_list

# 比较客户端间的数据重复程度
def compare_client_data(clients_data_list):
    all_hashes = [set(client_data.hashes) for client_data in clients_data_list if hasattr(client_data,'hashes')]
    for i, client_hashes in enumerate(all_hashes):
        for j in range(i+1, len(all_hashes)):
            duplicates = client_hashes.intersection(all_hashes[j])
            print(f"客户端 {i} 和 {j} 之间的重复数据数量: {len(duplicates)}")


def fed_dataset_NonIID_Dirichlet(train_data, n_clients, alpha, seed,q):
    """
    按Dirichlet分布划分Non-IID数据集，来源：https://zhuanlan.zhihu.com/p/468992765
    x是样本，y是标签
    :return:
    """
    #调用create_Non_iid_subsamples_dirichlet拿到每个客户端的训练样本字典
    clients_data_list = create_Non_iid_subsamples_dirichlet(n_clients, alpha, seed,train_data)
    # 要把每个客户端的权重也返回去，后面做加权平均用
    number_of_data_on_each_clients = [len(clients_data_list[i]) for i in range(len(clients_data_list))]
    total_data_length = sum(number_of_data_on_each_clients)
    weight_of_each_clients = [x / total_data_length for x in number_of_data_on_each_clients]  #根据每个客户端的数据量确定每个客户端的权重

    print("··········y_trian_dict:每个类别在训练集中的样本数量···········")
    for i in range(len(clients_data_list)):
        print(f"客户端:{i}, 数量:{len(clients_data_list[i])}")
        lst = []
        for data, target,_ in clients_data_list[i]:
            #print("target:",target)
            lst.append(target.item())

        print("该客户端的数据集的不同类别的数量(标签0-9)：")
        for i in range(10):     #0-9是标签，这个需要根据不同的数据集来打印，mnist和fashionmnist是只有0-9的标签
            print(lst.count(i), end=' ')
        #print(len(client_data_dict[key].dataset.targets))
        print()
    print("··········weight_of_each_clients：每个客户端的权重···········")
    print(weight_of_each_clients) #权重打印
    print()

    batch_size_of_each_clients=[ math.floor(len(clients_data_list[i]) * q) for i in range(len(clients_data_list))]

    #调用函数比较客户端之间的数据重复程度
    compare_client_data(clients_data_list)

    return clients_data_list, weight_of_each_clients,batch_size_of_each_clients