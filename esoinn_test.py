import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs
from sklearn.metrics.pairwise import euclidean_distances
import copy

# ==========================================
# 1. 轻量级 ESOINN 核心 (用于提取动态原型)
# ==========================================
class SimpleESOINN:
    """
    这是一个简化的 SOINN 实现，专注于 Idea 1 的核心：
    '自适应地学习能够覆盖数据分布的拓扑节点（原型）'
    """
    def __init__(self, age_max=50, iter_threshold=0.5):
        self.nodes = [] # 存储原型向量
        self.win_counts = [] # 记录胜利次数(用于去噪)
        self.age_max = age_max # 简化版：这里仅用于控制何时触发去噪
        self.similarity_threshold = iter_threshold # 初始阈值

    def _get_distance(self, x, y):
        return np.linalg.norm(x - y)

    def fit(self, X):
        # 增量学习过程
        for i, x in enumerate(X):
            if len(self.nodes) < 2:
                self.nodes.append(x)
                self.win_counts.append(1)
                continue

            # 1. 寻找最近的两个节点 (Winner s1, Second Winner s2)
            dists = [self._get_distance(x, n) for n in self.nodes]
            sorted_indices = np.argsort(dists)
            s1_idx = sorted_indices[0]
            s2_idx = sorted_indices[1]
            dist_s1 = dists[s1_idx]
            dist_s2 = dists[s2_idx]

            # 2. 计算自适应阈值 (这里简化为 s1 到 s2 的距离)
            # 在完整 ESOINN 中，T 计算更复杂，这里为了演示简化
            threshold = self._get_distance(self.nodes[s1_idx], self.nodes[s2_idx])
            
            # 如果是密集区域，阈值会很小；稀疏区域，阈值会很大

            # 3. 核心判断：是新模式吗？
            # 如果输入点 x 离最近的节点都很远，说明需要新的原型
            if dist_s1 > threshold or dist_s1 > self.similarity_threshold:
                self.nodes.append(x)
                self.win_counts.append(1)
            else:
                # 4. 如果是旧模式，更新赢家节点的位置 (类似于 K-Means 或 SOM)
                # 移动步长：1/win_count (越来越稳)
                learning_rate = 1.0 / (self.win_counts[s1_idx] + 1)
                self.nodes[s1_idx] += learning_rate * (x - self.nodes[s1_idx])
                self.win_counts[s1_idx] += 1
        
        # 5. (可选) 去噪：Adjusted-SOINN 的思想
        # 移除那些极少“赢”的节点，它们可能是噪声
        if len(self.nodes) > 5:
            avg_win = np.mean(self.win_counts)
            # 简单的过滤逻辑：保留赢过一定次数的节点
            valid_indices = [i for i, w in enumerate(self.win_counts) if w >= avg_win * 0.1]
            self.nodes = [self.nodes[i] for i in valid_indices]
            self.win_counts = [self.win_counts[i] for i in valid_indices]

        return np.array(self.nodes)

# ==========================================
# 2. 你的方法：动态多原型分类器 (Dynamic Multi-Prototype Classifier)
# ==========================================
class SOINNClassifier:
    def __init__(self):
        self.prototypes = {} # 字典：{class_label: [node1, node2, ...]}

    def fit(self, X, y):
        unique_classes = np.unique(y)
        for c in unique_classes:
            # 提取属于当前类的数据
            X_c = X[y == c]
            
            # 对该类初始化一个 SOINN
            # 注意：实际使用中，feature维度很高，iter_threshold需要根据backbone输出调整
            soinn = SimpleESOINN(iter_threshold=1.0) 
            
            # 训练并获取该类的拓扑节点
            nodes = soinn.fit(X_c)
            
            # 存储这些节点作为该类的“多原型”
            self.prototypes[c] = nodes
            print(f"Class {c}: Generated {len(nodes)} prototypes.")

    def predict(self, X):
        y_pred = []
        for x in X:
            min_dist = float('inf')
            best_class = -1
            
            # 遍历所有类的所有原型，寻找最近的那一个 (Nearest Prototype Strategy)
            for c, nodes in self.prototypes.items():
                # 计算样本 x 到该类所有原型的距离
                dists = np.linalg.norm(nodes - x, axis=1)
                # 找到该类中离 x 最近的原型距离
                d_c = np.min(dists)
                
                if d_c < min_dist:
                    min_dist = d_c
                    best_class = c
            y_pred.append(best_class)
        return np.array(y_pred)

# ==========================================
# 3. 传统的 NCM 分类器 (用于对比)
# ==========================================
class NCMClassifier:
    def __init__(self):
        self.means = {}

    def fit(self, X, y):
        unique_classes = np.unique(y)
        for c in unique_classes:
            self.means[c] = np.mean(X[y == c], axis=0)

    def predict(self, X):
        y_pred = []
        for x in X:
            min_dist = float('inf')
            best_class = -1
            for c, center in self.means.items():
                d = np.linalg.norm(x - center)
                if d < min_dist:
                    min_dist = d
                    best_class = c
            y_pred.append(best_class)
        return np.array(y_pred)

# ==========================================
# 4. 主程序与可视化
# ==========================================

# A. 生成非凸数据 (Moons) - NCM 的噩梦
X, y = make_moons(n_samples=400, noise=0.1, random_state=42)

# B. 训练模型
print("--- Training NCM ---")
ncm = NCMClassifier()
ncm.fit(X, y)

print("\n--- Training SOINN-Classifier ---")
soinn_clf = SOINNClassifier()
soinn_clf.fit(X, y)

# C. 可视化边界函数
def plot_decision_boundary(clf, X, y, title, ax):
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    
    # 预测网格中每个点的类别
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘图
    ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors='k', cmap=plt.cm.coolwarm)
    ax.set_title(title)
    
    # 画出原型 (如果是 SOINN)
    if isinstance(clf, SOINNClassifier):
        for c, nodes in clf.prototypes.items():
            ax.scatter(nodes[:, 0], nodes[:, 1], s=200, marker='*', c='yellow', edgecolors='black', label=f'Prototypes C{c}')

    # 画出均值 (如果是 NCM)
    if isinstance(clf, NCMClassifier):
        for c, mean in clf.means.items():
            ax.scatter(mean[0], mean[1], s=200, marker='X', c='yellow', edgecolors='black', label=f'Mean C{c}')

# D. 绘图对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

plot_decision_boundary(ncm, X, y, "NCM Classifier (Single Prototype)", ax1)
plot_decision_boundary(soinn_clf, X, y, "SOINN Classifier (Dynamic Multi-Prototypes)", ax2)

plt.tight_layout()
plt.show()