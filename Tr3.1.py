import torch
import torch.nn as nn
import torch.nn.functional as Func
import torch.optim as opt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets,transforms
import os
from PIL import Image
import numpy as np
import shutil
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

#Document 卷积神经网络是一种特征提取器，LSTM也是，全连接层的每一个神经元都会与上一层的所有神经元相连，其可以获取全局信息，称为全局神经元，相对的卷积神经网络中的像素点则是局部神经元
#Document 使用torch.nn对应的是一个类，使用torch.nn.functional对应的是一个函数。此处的最大区别在于是否需要人为设计参数，例如nn.conv2d、nn.Linear()等都是需要添加参数的，所以要使用torch.nn，而max_pool2d()不需要添加参数，故使用Func
#Document 使用Dropout与L1、L2等正则化手段可以提升算法的鲁棒性与泛化性，而使用池化层可以提升神经网络的平移、旋转和拉伸不变性(也是泛化性)
#超参数
img_dir = "C:\\Users\\谭诚凯\\Desktop\\数据集\\data_v3.0\\images"
label_dir = "C:\\Users\\谭诚凯\\Desktop\\数据集\\data_v3.0\\labels"
Device = torch.device("cuda")
Batch_Size = 128
Epoch_Lim = 200
Lr = 0.001
Test_Accuracy_List = []
#超参数


plt.ion()
class YoloDataset(Dataset):
    def __init__(self,img_dir,label_dir,transform = None,Is_train = True,augment = None):
        self.img_dir = img_dir
        self.augment = augment,
        self.label_dir = label_dir
        self.transform = transform
        self.Is_train = Is_train
        set_dir = "train" if Is_train == True else "val"
        self.image_file = [image for image in os.listdir(os.path.join(img_dir,set_dir)) if image.endswith(".png")]
        self.label_file = [label for label in os.listdir(os.path.join(label_dir,set_dir)) if label.endswith(".txt")]
        self.image_with_label = list(zip(self.image_file,self.label_file))

    def __len__(self):
        return len(self.image_with_label)  

    def polygon_label_parser(self,label_path = ""):
        if label_path.endswith(".txt"):
            with open(label_path,"r",encoding="utf-8") as f:
                data = f.read().strip().split()
                class_id = int(data[0])
                return class_id
        else:
            pass

    def __getitem__(self,idx):
        image_file,label_file = self.image_with_label[idx]

        image_path = os.path.join(self.img_dir,"train" if self.Is_train else "val",image_file)
        label_path = os.path.join(self.label_dir,"train" if self.Is_train else "val",label_file)

        img = Image.open(image_path).convert("L")
        label = self.polygon_label_parser(label_path=label_path)
        if self.transform:
            img = self.transform(img)
        return img,label
#方法定义区
Train_Accuracy_show = 0
def Train_module(module,train_data,device,optimizer,epoch):
    global Train_Accuracy_show
    module.train()
    Accuracy_train = 0
    sum_num = 0
    for batch_index,(data,label) in enumerate(train_data):
        data,label = data.to(device),label.to(device)
        optimizer.zero_grad()
        output = module(data)
        Pred = output.max(1,keepdim = True)[1]
        Accuracy_train += Pred.eq(label.view_as(Pred)).sum().item()
        sum_num += data.size(0)
        loss_ce = Func.cross_entropy(output,label)
        # loss_norm_L1 = sum(p.abs().sum() for p in module.parameters())
        loss = loss_ce 
        loss.backward()
        optimizer.step()
    Accuracy_train /= sum_num
    Train_Accuracy_show = 100*Accuracy_train
    print("epoch is {},Accuracy_train is {:6f}".format(epoch,100*Accuracy_train))
    del data, label, output, loss
    torch.cuda.empty_cache()
def Get_Module_Parameters(Pth_path):
    module_pth_path = Pth_path
    module.load_state_dict(torch.load(module_pth_path,map_location=Device))



def yolo_collate_fn(batch):
    # 分离图像和多边形列表
    images, labels = zip(*batch)
    labels_tensor = torch.tensor(labels)
    # 将图像转换为张量（如果它们还不是张量的话）
    # 注意：这里假设self.transform已经将图像转换为张量，如果没有，则需要添加转换逻辑
    images_tensor = torch.stack(images,dim=0)
    # 保持多边形列表作为列表的列表
    # 注意：这里不对多边形进行任何转换或堆叠，因为它们长度不同
    # 你可能需要在后续的数据处理或模型输入阶段对多边形进行特殊处理
    
    # 返回图像张量和一个多边形列表的列表
    return images_tensor,labels_tensor


Test_Accuracy_Show = 0
Avg_Loss = 0
def test_module(module,test_data,device):
    global Test_Accuracy_Show
    global Avg_Loss
    module.eval()
    Accuracy_test = 0
    Test_Loss = 0
    with torch.no_grad():
        for data,label in test_data:
            data,label = data.to(device),label.to(device)
            out_put = module(data)
            Pred = out_put.max(1,keepdim = True)[1]
            Test_Loss  += Func.cross_entropy(out_put,label).item()
            Accuracy_test += Pred.eq(label.view_as(Pred)).sum().item()
        Test_Loss /= len(test_loader.dataset)
        Avg_Loss = Test_Loss
        Avg_Correct_rate = 100*Accuracy_test / len(test_loader.dataset)
        Test_Accuracy_Show = Avg_Correct_rate
        Test_Accuracy_List.append(Avg_Correct_rate)
        print("Avg_loss is {:6f},Accuracy_test is {:6f}".format(Test_Loss,Avg_Correct_rate))
#方法定义区
#定义模型
class Module(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入参数为（1,227,227）
        self.Conv_First = nn.Conv2d(1,64,kernel_size=11,stride=4,padding=2)
        self.Conv_Second = nn.Conv2d(10,20,3)
        self.Fc_First = nn.Linear(20*124*124,1000)
        self.Dropout_First = nn.Dropout(p=0.25)
        self.Fc_Second = nn.Linear(1000,14)
    def forward(self,x):
        x = self.Conv_First(x) #输出参数为(64,57,57)
        x = Func.relu(x) #激活函数不影响输出的尺寸
        x = Func.max_pool2d(x,2,2) #输出参数为(10,126,126)
        x = self.Conv_Second(x) #输出参数为(20,124,124)
        x = Func.relu(x) #激活函数不影响输出的尺寸
        x = x.view(x.size(0),-1) #强制展平.此处x.size(0)是指的批次数，即batchsize，-1则是pytorch自动计算尺寸的指令
        x = self.Fc_First(x) #输出参数为500
        x = Func.relu(x) #激活函数不影响输出值
        x = self.Dropout_First(x)
        x = self.Fc_Second(x) #输出参数为10
        output = Func.softmax(x,dim=1)
        return output
#定义模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(1, 48, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第二层卷积
            nn.Conv2d(48, 128, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第三层卷积
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第四层卷积
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第五层卷积
            nn.Conv2d(192, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 6 * 6)  # 展平
        x = self.classifier(x)
        return x

class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ZFNet(nn.Module):
    def __init__(self, num_classes):
        super(ZFNet, self).__init__()
        self.features = nn.Sequential(
            # 第一个卷积层
            nn.Conv2d(1, 96, kernel_size=9, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(96, 128, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(128),  # 添加批量归一化层，这是ZFNet相对于AlexNet的一个改进
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第二个卷积层
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2, groups=2, bias=False),  # 使用分组卷积
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 第三个卷积层（这里保持与AlexNet相似的结构，但使用ReLU）
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 第四个卷积层
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            # 第五个卷积层（ZFNet中通常没有这个层，但为了与某些描述保持一致，这里可以添加一个可选的卷积层）
            # 注意：这个层是可选的，并且可能需要根据具体任务进行调整
            # nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(256),
            # nn.ReLU(inplace=True),
            # 如果添加了上面的卷积层，则可能需要相应地调整全连接层的输入大小
            # 并且可能需要添加额外的MaxPool2d层来减少特征图的尺寸
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            # 如果没有添加额外的卷积层，则通常会有一个全局平均池化层或适应性平均池化层
            # 这里我们使用适应性平均池化层来确保输出尺寸为6x6（或根据需要调整）
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(384 * 6 * 6, 2048),  # 注意这里的输入特征数量需要根据前面的层进行调整
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
 
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)  # 将特征图展平为一维向量
        x = self.classifier(x)
        return x  # 返回logits

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride = stride
 
    def forward(self, x):
        identity = x
 
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
 
        out = self.conv2(out)
        out = self.bn2(out)
 
        if self.downsample is not None:
            identity = self.downsample(x)
 
        out += identity
        out = self.relu(out)
 
        return out
 
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64
        # 修改第一层卷积以接受单通道输入
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        # 使用全局平均池化替换原始的平均池化和全连接层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 对于BasicBlock，expansion为1
 
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
 
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
 
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
 
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
 
        return x
 
def ResNet18_SingleChannel(num_classes=14):  # 你可以根据需要更改类别数
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

pipeline = transforms.Compose([
    transforms.Resize((227,227)),
    # transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5), 
    transforms.ToTensor(),
    transforms.Normalize((0.369012),(0.482585))
])

train_data = datasets.MNIST(root="./data",train=True,transform=pipeline,download=True)
test_data = datasets.MNIST(root="./data",train=False,transform=pipeline,download=True)
train_data_yolo = YoloDataset(img_dir=img_dir,label_dir=label_dir,transform=pipeline,Is_train=True)
test_data_yolo = YoloDataset(img_dir=img_dir,label_dir=label_dir,transform=pipeline,Is_train=False)
train_loader = DataLoader(dataset=train_data_yolo,batch_size=Batch_Size,shuffle=True,collate_fn=yolo_collate_fn)
test_loader = DataLoader(dataset=test_data_yolo,batch_size=Batch_Size,shuffle=True,collate_fn=yolo_collate_fn)
# Pth_PATH = r"D:\Deep_Learn\mnist_final_result\98.0773206532975.pth"
module = AlexNet(num_classes=14).to(Device)
# state_dict = torch.load(Pth_PATH)
# module.load_state_dict(state_dict=state_dict)
# module.eval()
# Predicted_Images = "C:\\Users\\谭诚凯\\Desktop\\武汉\\武汉市建筑物数据\\WH_data"
# #预测
# def Pred_Translater(Pred):
#     if Pred == 0:
#         return "矩形"
#     if Pred == 1:
#         return "直线形"
#     if Pred == 2:
#         return "三角形"
#     if Pred == 3:
#         return "梯形"
#     if Pred == 4:
#         return "圆形"
#     if Pred == 5:
#         return "三叉形"
#     if Pred == 6:
#         return "廾形"
#     if Pred == 7:
#         return "C形"
#     if Pred == 8:
#         return "T形"
#     if Pred == 9:
#         return "U形"
#     if Pred == 10:
#         return "L形"
#     if Pred == 11:
#         return "十字形"
#     if Pred == 12:
#         return "Z形"
#     if Pred == 13:
#         return "H形"
# for filename in os.listdir(Predicted_Images):
#     if filename.endswith('.png'):
#         # 加载图像
#         image_path = os.path.join(Predicted_Images, filename)
#         image = Image.open(image_path).convert('L')  # 确保图像是 RGB 格式
#         # 预处理图像
#         image_tensor = pipeline(image)
#         image_tensor = image_tensor.to(Device)
#         image_tensor = image_tensor.unsqueeze(0)  # 增加批次维度
        
#         # 进行预测
#         with torch.no_grad():
#             prediction = module(image_tensor)
        
#         # 处理预测结果（根据你的任务进行）
#         # 假设你的模型输出是 logits，你可能需要应用 softmax 或 argmax
#         # 这里只是一个示例，你需要根据你的实际情况调整
#             _,probs = torch.max(prediction.data,1)
#             predicted_class = torch.argmax(prediction, dim=1).item()
        
#         # 打印或保存结果
#             print(f'Filename: {filename}, Predicted class: {Pred_Translater(predicted_class)}, Probabilities: {probs}')
        # 如果你想保存结果到文件，可以使用以下代码（根据你的需求调整）
        # with open(f'predictions_{filename}.txt', 'w') as f:
        #     f.write(f'Predicted class: {predicted_class}\nProbabilities: {probs}\n')
 
# 注意：上面的代码假设你的模型输出是分类任务的 logits。
# 如果你的任务是回归或其他类型的任务，你需要相应地调整预测结果的处理方式。



#预测
optimizer = opt.Adam(module.parameters(),lr = 0.0001,weight_decay=0.00005)

#定义图像参数
Epoch_List = []
Train_Accuracy_List = []
Test_Accuracy_List_Use = []
Train_Accuracy_Lines = []
Test_Accuracy_Lines = []
Avg_Loss_List = []
Avg_Loss_Lines = []
#定义图像参数
fig1,axe1 = plt.subplots()
fig2,axe2 = plt.subplots()


for epoch in range(1,Epoch_Lim+1):
    Epoch_List.append(epoch)
    Train_module(module,train_loader,optimizer=optimizer,epoch=epoch,device=Device)
    Train_Accuracy_List.append(Train_Accuracy_show)
    test_module(module,test_loader,Device)
    Test_Accuracy_List_Use.append(Test_Accuracy_Show)
    Avg_Loss_List.append(Avg_Loss)
    try:
        Train_Accuracy_Lines.remove(Train_Accuracy_Lines[-1])
        Test_Accuracy_Lines.remove(Test_Accuracy_Lines[-1])
        Avg_Loss_Lines.remove(Avg_Loss_Lines[-1])
    except Exception:
        pass
    Train_Accuracy_Lines = axe1.plot(Epoch_List,Train_Accuracy_List,linestyle = "dashed",c = "r",lw = 2)
    Test_Accuracy_Lines = axe1.plot(Epoch_List,Test_Accuracy_List_Use,c = "b",lw = 2)
    Avg_Loss_Lines = axe2.plot(Epoch_List,Avg_Loss_List,c = "g",lw = 2)

    axe1.set_xlabel("Epoch")
    axe1.set_ylabel("Accuracy")
    axe1.set_title("Train and Test Accuracy")
    axe1.legend(["Train_Accuracy","Test_Accuracy"],loc = "lower right")

    axe2.set_xlabel("Epoch")
    axe2.set_ylabel("Avg_Loss")
    axe2.set_title("Avg_Loss")
    axe2.legend(["Avg_Loss"],loc = "upper right")
    if epoch == Epoch_Lim:
        fig1.savefig(r"C:\Users\谭诚凯\Desktop\论文结果图\acc_result_1.png")
        fig2.savefig(r"C:\Users\谭诚凯\Desktop\论文结果图\loss_result_1.png")
        print("Fine!")
    plt.show()
    plt.pause(0.1)
    Save_path = os.path.join(r"D:\Deep_Learn\mnist_result",str(Test_Accuracy_Show)+".pth")
    torch.save(module.state_dict(),Save_path)
print("最大测试准确度：{:6f}".format(max(Test_Accuracy_List)))
folder_path = r"D:\Deep_Learn\mnist_result\\"
all_pths = os.listdir(folder_path)
for pth_accuracy in all_pths:
    pth_path = os.path.join(folder_path,pth_accuracy)
    Saved_path = pth_path.split("\\")[-1].split(".")[0] + "." + pth_path.split("\\")[-1].split(".")[1]
    if str(Saved_path)  != str(max(Test_Accuracy_List)):
        os.remove(pth_path)
    else:
        pass
        print("Saved!")

#修正transform.normalize()后，Accuracy = 77.38498789346247 Type:Alexnet
#修正transforms.RandomHorizontalFlip(p=0.5)后，Accuracy = 77.62711864406779  Type:Alexnet
#调整lr=0.0001后效果明显，Accuracy = 78.8861985472155 Type:Alexnet
#使用ZFnet后，Accuracy = 79.90314769975787 Type：ZFnet
#使用ZFnet后，将Relu改为Relu6，Accuracy = 80.58111380145279 Type：ZFnet

#参数调整策略：不改变，Accuracy = 82.70783847980998
#参数调整策略：添加一层全连接层，无效 Accuracy = 81.61520190023752
#参数调整策略：不改变层全连接层层数，减半全连接层神经元数量，中立 Accuracy = 82.51781472684085
#参数调整策略：修正Normalize参数，中立 Accuracy = 82.13776722090262
#参数调整策略 data_v2.3 有效,Accuracy = 93.61702127659575
#参数调整策略 nn.AdaptiveAvgPool2d((10, 10))，中立 Accuracy = 93.43971631205673
#参数调整策略 data_v2.4 有效,Accuracy = 93.8501515807709
#参数调整策略 data_v2.5 中立,Accuracy = 93.8301515807709
#参数调整策略 降低lr与提高batch，有效，Accuracy = 93.93676916414032
#参数调整策略 依据数据修正mean与std，有效，Accuracy = 94.06669553919446
#参数调整策略 增加randomvertical(p = 0.25), 有效，Accuracy = 94.1966219142486
#参数调整策略 将randomvertical(p = 0.5)与randomhorizional(p = 0.5), 有效，Accuracy = 94.2399307059333
#参数调整策略 将所有激活函数修正为Relu而不是Relu6，有效，Accuracy = 94.45647466435686，此时batchsize = 128，epoch = 200
#参数调整策略 提高batchsize = 256，无效，Accuracy =  94.2399307059333，此时batchsize = 256，epoch = 100
#参数调整策略 降低weight_decay = 0.00005,有效， Accuracy = 94.62970983109571
#参数调整策略 添加RandomRotation(),无效
#参数调整策略 使用data_v3.0,有效，Accuracy = 96.96092619392185