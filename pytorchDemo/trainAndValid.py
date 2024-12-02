import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
from torch.utils.data import random_split
from model import MyNN

epoch = 100
validation_share=0.01
batchsize=64
lr=0.01
train_data = torchvision.datasets.CIFAR10(root="dataset", train=True,
                                          transform=torchvision.transforms.ToTensor(),download=True)
test_data = torchvision.datasets.CIFAR10(root="dataset", train=False,
                                          transform=torchvision.transforms.ToTensor(),download=True)
train_size=int(0.8*len(train_data))
val_size=len(train_data)-train_size
test_size=len(test_data)
train_data,val_data= random_split(train_data,[train_size,val_size])
train_dataLoader = DataLoader(train_data, batch_size=batchsize)
val_dataLoader=DataLoader(val_data,batch_size=batchsize)
test_dataLoader = DataLoader(test_data, batch_size=batchsize)
myNN = MyNN()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(myNN.parameters(), lr=lr)
writer = SummaryWriter("logs")
print(f"---------train_data.size:{len(train_data)}-----val_data.size:{len(val_data)}-----test_data.size:{len(test_data)}---------")
def train():
    startTime = time.time()
    total_train_step=0
    total_val_step = 0
    for i in range(epoch):
        print(f"---------------{i + 1}th train has started---------------")
        myNN.train()  # 将模型设置为训练模式
        for data in train_dataLoader:
            imgs, targets = data
            outputs = myNN(imgs)  # 获取模型输出
            loss = loss_fn(outputs, targets)  # 计算损失
            optimizer.zero_grad()  # 将梯度置为0 pytorch默认会将梯度累加
            loss.backward()  # 反向传播 ：也即计算Cost_function 对于每个参数的偏导
            optimizer.step()  # 梯度下降
            total_train_step += 1
            if total_train_step % 100 == 0:
                print(f"训练次数{total_train_step}时，损失值为：{loss}")
                endTime = time.time()
                print(f"经过时间：{endTime - startTime}")
                writer.add_scalar("训练集损失 :", loss.item(), total_train_step)
        # 验证步骤

        total_accuracy = 0
        myNN.eval()  # 将模型设置为验证模式
        total_valid_loss=0
        with torch.no_grad():  # 用于在其作用域内禁用梯度计算。所有在这个上下文管理器内的操作都不会计算梯度，也不会存储梯度信息。
            for data in test_dataLoader:
                imgs, targets = data
                outputs = myNN(imgs)  # 获取模型输出
                loss = loss_fn(outputs, targets)  # 计算损失
                total_valid_loss += loss.item()
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy

        print(f"验证集整体loss：{total_valid_loss}")
        print(f'验证集正确率：{total_accuracy / 1.0 / val_size}')
        writer.add_scalar("验证集整体正确率", total_accuracy / 1.0 / val_size, total_val_step)
        writer.add_scalar("验证集整体loss", total_valid_loss, total_val_step)
        total_val_step += 1
def test():
    total_test_loss = 0
    total_accuracy = 0

    myNN.eval()  # 将模型设置为验证模式
    with torch.no_grad():  # 用于在其作用域内禁用梯度计算。所有在这个上下文管理器内的操作都不会计算梯度，也不会存储梯度信息。
        for data in test_dataLoader:
            imgs, targets = data
            outputs = myNN(imgs)  # 获取模型输出
            loss = loss_fn(outputs, targets)  # 计算损失
            total_test_loss += loss.item()
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy

    print(f"测试集整体loss：{total_test_loss}")
    print(f'测试集正确率：{total_accuracy / 1.0 / test_size}')
    # writer.add_scalar("测试集整体正确率", total_accuracy / 1.0 / test_size, total_test_step)
    # writer.add_scalar("测试集整体loss", total_test_loss, total_test_step)

def model_to_onnx():
    myNN.eval()
    x=torch.rand(size=(1,3,32,32),dtype=torch.float32)
    torch.onnx.export(myNN,x,f='myNN.onnx',input_names=['input'],output_names=['output'],opset_version=11)


state_dict = torch.load('models/myNN.pth')

myNN.load_state_dict(state_dict)
train()

torch.save(myNN.state_dict(),"models/myNN.pth") #保存模型参数 官方推荐
test()
writer.close()
model_to_onnx()