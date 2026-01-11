'''
Author: ViolinSolo
Date: 2025-10-31 12:31:32
LastEditTime: 2025-10-31 12:40:16
LastEditors: ViolinSolo
Description: 
FilePath: /ProjectBrainBaseline/test/test_gnn.py
'''
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class GNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        print("输入节点特征形状:", x.shape)
        print("输入边索引形状:", edge_index.shape)
        
        # 图卷积层 + 激活函数
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        print("经过第一层卷积后的节点特征形状:", x.shape)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        
        print("经过第三层卷积后的节点特征形状:", x.shape)

        # 图池化（对于图分类任务）
        x = global_mean_pool(x, batch)
        print("池化后的图特征形状:", x.shape)
        
        # 分类器
        x = self.classifier(x)
        
        return F.log_softmax(x, dim=1)

def create_mock_graph(num_nodes=10, feature_dim=16, num_classes=3):
    """创建模拟的图数据"""
    # 节点特征
    x = torch.randn(num_nodes, feature_dim)
    
    # 创建随机边（确保图是连通的）
    edge_index = []
    for i in range(num_nodes):
        # 每个节点连接到1-3个随机邻居
        num_edges = np.random.randint(1, 4)
        neighbors = np.random.choice(
            [j for j in range(num_nodes) if j != i], 
            min(num_edges, num_nodes-1), 
            replace=False
        )
        for neighbor in neighbors:
            edge_index.append([i, neighbor])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 随机标签
    y = torch.tensor([np.random.randint(0, num_classes)], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes)

def create_mock_node_classification_graph(num_nodes=20, feature_dim=16, num_classes=3):
    """创建节点分类的模拟图数据"""
    # 节点特征
    x = torch.randn(num_nodes, feature_dim)
    
    # 创建随机边
    edge_index = []
    for i in range(num_nodes):
        num_edges = np.random.randint(1, 4)
        neighbors = np.random.choice(
            [j for j in range(num_nodes) if j != i], 
            min(num_edges, num_nodes-1), 
            replace=False
        )
        for neighbor in neighbors:
            edge_index.append([i, neighbor])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # 每个节点都有一个标签（节点分类任务）
    y = torch.tensor([np.random.randint(0, num_classes) for _ in range(num_nodes)], dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index, y=y)

class NodeGNN(torch.nn.Module):
    """节点分类的GNN模型"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NodeGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(0.5)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        print("=== 节点分类模型前向传播 ===")
        print("输入节点特征形状:", x.shape)
        print("输入边索引形状:", edge_index.shape)
        
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        print("经过第一层卷积后的节点特征形状:", x.shape)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        print("经过第二层卷积后的节点特征形状:", x.shape)

        x = self.conv3(x, edge_index)

        print("经过第三层卷积后的节点特征形状:", x.shape)

        # 节点分类，不需要图池化
        x = self.classifier(x)

        print("经过分类器后的节点特征形状:", x.shape)
        
        return F.log_softmax(x, dim=1)

def train_and_evaluate():
    """训练和评估模型"""
    print("=== 图分类任务 ===")
    
    # 创建模拟数据集
    graphs = []
    for _ in range(50):  # 50个图
        graphs.append(create_mock_graph(
            num_nodes=np.random.randint(8, 15),
            feature_dim=16,
            num_classes=3
        ))
    
    # 创建数据加载器
    loader = DataLoader(graphs, batch_size=8, shuffle=True)
    
    # 初始化模型
    model = GNN(input_dim=16, hidden_dim=32, output_dim=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # 训练循环
    model.train()
    losses = []
    for epoch in range(100):
        total_loss = 0
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = F.nll_loss(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {avg_loss:.4f}')
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('图分类训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    print("\n=== 节点分类任务 ===")
    
    # 节点分类任务
    node_data = create_mock_node_classification_graph(num_nodes=30, feature_dim=16, num_classes=3)
    
    node_model = NodeGNN(input_dim=16, hidden_dim=32, output_dim=3)
    node_optimizer = torch.optim.Adam(node_model.parameters(), lr=0.01, weight_decay=5e-4)
    
    node_losses = []
    node_model.train()
    for epoch in range(100):
        node_optimizer.zero_grad()
        out = node_model(node_data)
        loss = F.nll_loss(out, node_data.y)
        loss.backward()
        node_optimizer.step()
        node_losses.append(loss.item())
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')
    
    # 节点分类评估
    node_model.eval()
    with torch.no_grad():
        pred = node_model(node_data).argmax(dim=1)
        accuracy = (pred == node_data.y).sum().item() / node_data.y.size(0)
        print(f'\n节点分类准确率: {accuracy:.4f}')
    
    plt.subplot(1, 2, 2)
    plt.plot(node_losses)
    plt.title('节点分类训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.tight_layout()
    plt.show()
    
    # 打印模型结构信息
    print(f"\n=== 模型信息 ===")
    print(f"图分类模型参数量: {sum(p.numel() for p in model.parameters())}")
    print(f"节点分类模型参数量: {sum(p.numel() for p in node_model.parameters())}")
    
    # 打印一些示例数据信息
    sample_graph = graphs[0]
    print(f"\n=== 数据示例 ===")
    print(f"节点数: {sample_graph.num_nodes}")
    print(f"边数: {sample_graph.edge_index.shape[1]}")
    print(f"节点特征维度: {sample_graph.x.shape[1]}")
    print(f"图标签: {sample_graph.y.item()}")

if __name__ == "__main__":
    train_and_evaluate()