from FL.fl_utils.center_average_model_with_weights import set_averaged_weights_as_main_model_weights
from FL.fl_utils.local_clients_train_process import local_clients_train_process_without_dp_one_batch
from FL.fl_utils.send_main_model_to_clients import send_main_model_to_clients
from data.dirichlet_nonIID_data import fed_dataset_NonIID_Dirichlet
from FL.fl_utils.optimizier_and_model_distribution import create_model_optimizer_criterion_dict
from data.get_data import get_data
from model.CNN import CNN
from train_and_validation.validation import validation
import torch

import time
import psutil
import matplotlib.pyplot as plt


# 获取当前进程ID，然后创建一个psutil.Process对象以监控当前进程
current_process = psutil.Process()
cpu_cores = psutil.cpu_count(logical=True)  # 使用逻辑核心数，或者改为logical=False使用物理核心数


# 显示监控数据
def display_monitoring_data(training_time, cpu_usage, memory_usage_bytes):
    # 将内存使用量从字节转换为MB
    memory_usage_mb = [usage / (1024 ** 2) for usage in memory_usage_bytes]

    # 打印监控数据
    print("训练时间 (s): ", training_time)
    print("CPU使用率 (%): ", cpu_usage)
    print("内存使用率 (MB): ", memory_usage_mb)

    # 绘制训练时间图表
    plt.figure(figsize=(10, 4))
    plt.plot(training_time, label='Training Time (seconds)')
    plt.xlabel('Iterations')
    plt.ylabel('Time (seconds)')
    plt.title('Training Time per Iteration')
    plt.legend()
    plt.show()

    # 绘制CPU使用率图表
    plt.figure(figsize=(10, 4))
    plt.plot(cpu_usage, label='CPU Usage (%)')
    plt.xlabel('Iterations')
    plt.ylabel('Usage (%)')
    plt.title('CPU Usage per Iteration')
    plt.legend()
    plt.show()

    # 绘制内存使用率图表，单位为MB
    plt.figure(figsize=(10, 4))
    plt.plot(memory_usage_mb, label='Memory Usage (MB)')
    plt.xlabel('Iterations')
    plt.ylabel('Usage (MB)')
    plt.title('Memory Usage per Iteration')
    plt.legend()
    plt.show()


def remove_duplicate_data(client_data_list):
    # 存储数据点的哈希值
    hash_set = set()

    # 用于存储去重后的数据
    deduplicated_data_list = []

    # 遍历客户端数据集,去除各个客户端之间的重复数据
    for client_data in client_data_list:
        deduplicated_data = []
        for data_point, target, data_hash in client_data:
            # 检查哈希值是否已存在
            if data_hash not in hash_set:
                # 如果哈希值不存在，则添加数据点到去重后的数据集中
                deduplicated_data.append((data_point, target, data_hash))
                # 将哈希值添加到集合中
                hash_set.add(data_hash)
        # 将去重后的数据添加到列表中
        deduplicated_data_list.append(deduplicated_data)

    return deduplicated_data_list


def plot_data_comparison(before_deduplication, after_deduplication):
    # 客户端数量
    num_clients = len(before_deduplication)

    # 可视化数据对比
    plt.figure(figsize=(10, 6))
    for i in range(num_clients):
        plt.bar(i - 0.2, len(before_deduplication[i]), width=0.4, label=f"Client {i} Before Deduplication")
        plt.bar(i + 0.2, len(after_deduplication[i]), width=0.4, label=f"Client {i} After Deduplication")

    plt.xlabel('Client ID')
    plt.ylabel('Number of Data Points')
    plt.title('Data Comparison Before and After Deduplication')
    plt.xticks(range(num_clients), [f'Client {i}' for i in range(num_clients)])
    plt.legend()
    plt.show()

def plot_accuracy_and_loss(test_accuracy_record,test_loss_record):
    print("测试精度: ",test_accuracy_record)
    print("损失函数值: ",test_loss_record)

    # 绘制测试精度曲线
    plt.figure(figsize=(10, 6))
    plt.plot(test_accuracy_record, label='Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy Curve')
    plt.legend()
    plt.show()

    # 绘制损失函数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(test_loss_record, label='Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Test Loss Curve')
    plt.legend()
    plt.show()

def fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q,model,deduplicate):
    #客户端的样本分配
    clients_data_list, weight_of_each_clients,batch_size_of_each_clients =fed_dataset_NonIID_Dirichlet(train_data,number_of_clients,alpha,seed,q)
    #clients_data_list, weight_of_each_clients,batch_size_of_each_clients =pathological_split_noniid(train_data,number_of_clients,alpha,seed,q)
    # iid样本
    # clients_data_list = split_iid(train_data,number_of_clients)

    deduplicated_data_list = remove_duplicate_data(clients_data_list)
    if deduplicate:
        plot_data_comparison(clients_data_list, deduplicated_data_list)  # 对比去重前后的数据差异
        clients_data_list = deduplicated_data_list
        print("去重后训练")

    # 各个客户端的model,optimizer,criterion的分配
    clients_model_list, clients_optimizer_list, clients_criterion_list = create_model_optimizer_criterion_dict(number_of_clients, learning_rate,model)
    # 初始化中心模型,本质上是用来接收客户端的模型并加权平均进行更新的一个变量
    center_model = model

    # 预热调用，准备CPU使用率的测量
    current_process.cpu_percent()
    # 新增：初始化性能监控数据结构
    training_time = []
    cpu_usage = []
    memory_usage = []
    # gpu_usage = []  # 如果使用GPU

    test_dl = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False)

    print("-------------------联邦学习整体流程开始-------------------")
    test_accuracy_record=[]
    test_loss_record=[]
    max_accuracy = 0  # 保存历史最高测试精度
    # max_accuracy_iter = 0  # 保存历史最高测试精度时的迭代次数

    for i in range(iters):
        # 记录开始时间
        start_time = time.time()
        print("现在进行和中心方的第{:3.0f}轮联邦训练".format(i+1))
        # 1 中心方广播参数给各个客户端
        clients_model_list = send_main_model_to_clients(center_model, clients_model_list)
        # 2本地梯度下降
        local_clients_train_process_without_dp_one_batch(number_of_clients,clients_data_list,clients_model_list,clients_criterion_list,clients_optimizer_list,numEpoch,q)
        # 3 客户端上传参数到中心方进行加权平均并更新中心方参数(根据客户端数量加权平均)
        center_model = set_averaged_weights_as_main_model_weights(center_model,clients_model_list,weight_of_each_clients)
        # center_model = set_averaged_weights_as_main_model_weights_fully_averaged(center_model,clients_model_list)

        # 查看中心方模型效果，测试精度
        test_loss, test_accuracy = validation(center_model, test_dl)
        print("Iteration", str(i + 1), ": main_model accuracy on all test data: {:7.4f}".format(test_accuracy))

        test_loss_record.append(test_loss)
        test_accuracy_record.append(test_accuracy)
        max_accuracy = max(test_accuracy_record)

        # if test_accuracy > max_accuracy:
        #     max_accuracy = test_accuracy
        #     max_accuracy_iter = i + 1
        #
        # # 如果连续5次测试精度都没有提高，则停止训练
        # if i - max_accuracy_iter >= 5:
        #     print("连续5次测试精度都没有提高，停止训练。")
        #     break

        # if test_accuracy >= 96:
        #     print("达到96%测试精度，停止训练。")
        #     break

        # 记录资源使用情况
        cpu_usage.append(current_process.cpu_percent() / cpu_cores)  # 记录当前进程的CPU使用率
        memory_usage.append(current_process.memory_info().rss)  # 记录当前进程的内存使用量
        end_time = time.time()
        training_time.append(end_time - start_time)

    plot_accuracy_and_loss(test_accuracy_record,test_loss_record)

    # 保存训练记录
    record = [iters, numEpoch, test_loss_record, test_accuracy_record, training_time, cpu_usage, memory_usage]
    print("max_accuracy: ",max_accuracy)
    # 性能监控
    display_monitoring_data(training_time, cpu_usage, memory_usage)


if __name__=="__main__":
    train_data, test_data = get_data('mnist', augment=False) #mnist
    model = CNN()
    deduplicate = False
    batch_size=64
    learning_rate = 0.1
    numEpoch = 5       #客户端本地下降次数
    number_of_clients=5
    momentum=0.9
    alpha=0.05 #狄立克雷的异质参数
    seed=1   #随机种子
    q_for_batch_size=0.1  #基于该数据采样率组建每个客户端的batchsize
    iters = 500
    iters=int(iters/numEpoch)
    fed_avg(train_data,test_data,number_of_clients,learning_rate,momentum,numEpoch,iters,alpha,seed,q_for_batch_size,model,deduplicate)

