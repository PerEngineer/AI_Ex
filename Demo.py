"""
实验一：感知器线性分类器的设计实现
Experiment 1: Design and Implementation of Perceptron Linear Classifier
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class PerceptronClassifier:
    """感知器多类分类器"""
    
    def __init__(self, num_classes, learning_rate=1.0, max_iter=1000):
        """
        初始化感知器分类器
        
        参数:
        num_classes: 类别数量
        learning_rate: 学习率（步长）
        max_iter: 最大迭代次数
        """
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = None  # 权重矩阵 [num_classes, feature_dim+1]
        
    def normalize_samples(self, X):
        """
        样本规范化处理（增广向量）
        将原始特征向量转换为增广向量 [x1, x2, ..., xd, 1]
        
        参数:
        X: 输入特征矩阵 [n_samples, feature_dim]
        
        返回:
        normalized_X: 规范化后的增广向量 [n_samples, feature_dim+1]
        """
        n_samples, feature_dim = X.shape
        # 添加常数项1，形成增广向量
        normalized_X = np.hstack([X, np.ones((n_samples, 1))])
        return normalized_X
    
    def fit(self, X, y):
        """
        训练感知器分类器（多类判别）
        
        参数:
        X: 训练特征 [n_samples, feature_dim]
        y: 训练标签 [n_samples]，标签值从0到num_classes-1
        """
        # 规范化样本（增广向量）
        X_normalized = self.normalize_samples(X)
        n_samples, feature_dim = X_normalized.shape
        
        # 初始化权重矩阵：每一类一个权向量
        # 使用零初始化或小的随机值
        self.weights = np.zeros((self.num_classes, feature_dim))
        
        # 使用单样本修正法训练
        converged = False
        iteration = 0
        
        while not converged and iteration < self.max_iter:
            iteration += 1
            misclassified = 0
            
            # 遍历所有样本
            for j in range(n_samples):
                sample = X_normalized[j]  # 当前样本（增广向量）
                true_label = y[j]  # 真实标签
                
                # 计算所有类的判别函数值
                scores = self.weights @ sample  # [num_classes]
                
                # 检查是否错分：w_i^T * y_j <= w_t^T * y_j for any t != i
                predicted_label = np.argmax(scores)
                
                if predicted_label != true_label:
                    misclassified += 1
                    # 更新权重：
                    # w_i(k+1) = w_i(k) + y_j  (正确类)
                    # w_t(k+1) = w_t(k) - y_j (竞争类)
                    self.weights[true_label] += self.learning_rate * sample
                    self.weights[predicted_label] -= self.learning_rate * sample
            
            # 如果没有错分样本，算法收敛
            if misclassified == 0:
                converged = True
                print(f"算法在第 {iteration} 次迭代后收敛")
                break
            
            if iteration % 100 == 0:
                print(f"迭代 {iteration}: 错分样本数 = {misclassified}")
        
        if not converged:
            print(f"警告: 算法在 {self.max_iter} 次迭代后未收敛")
        
        return self
    
    def predict(self, X):
        """
        预测样本类别
        
        参数:
        X: 测试特征 [n_samples, feature_dim]
        
        返回:
        predictions: 预测类别 [n_samples]
        """
        X_normalized = self.normalize_samples(X)
        scores = X_normalized @ self.weights.T  # [n_samples, num_classes]
        predictions = np.argmax(scores, axis=1)
        return predictions
    
    def score(self, X, y):
        """
        计算分类准确率
        
        参数:
        X: 测试特征
        y: 真实标签
        
        返回:
        accuracy: 准确率
        """
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy


def generate_gaussian_data(num_classes, samples_per_class, feature_dim=2, 
                          mean_range=(-5, 5), std_range=(0.5, 1.5), seed=None, 
                          ensure_separable=True):
    """
    使用高斯模型生成多类数据
    
    参数:
    num_classes: 类别数量（N>5）
    samples_per_class: 每类样本数
    feature_dim: 特征维度（2或3）
    mean_range: 均值范围
    std_range: 标准差范围
    seed: 随机种子
    ensure_separable: 是否确保数据线性可分（通过分散均值位置）
    
    返回:
    X: 特征矩阵 [total_samples, feature_dim]
    y: 标签向量 [total_samples]
    """
    if seed is not None:
        np.random.seed(seed)
    
    X_list = []
    y_list = []
    
    if ensure_separable and feature_dim == 2:
        # 对于2D数据，将类别分布在圆形或网格上，增加线性可分的可能性
        angle_step = 2 * np.pi / num_classes
        radius = 8.0
        
        for class_idx in range(num_classes):
            # 在圆周上均匀分布类别中心
            angle = class_idx * angle_step
            mean = np.array([
                radius * np.cos(angle),
                radius * np.sin(angle)
            ])
            
            # 使用较小的标准差，使类别更紧凑
            std_val = np.random.uniform(0.8, 1.2)
            cov = np.eye(feature_dim) * std_val
            
            # 生成高斯分布样本
            class_samples = np.random.multivariate_normal(mean, cov, samples_per_class)
            X_list.append(class_samples)
            y_list.append(np.full(samples_per_class, class_idx))
    else:
        # 原始方法：随机生成
        for class_idx in range(num_classes):
            # 为每个类别随机生成均值和协方差矩阵
            mean = np.random.uniform(mean_range[0], mean_range[1], feature_dim)
            
            # 生成随机正定协方差矩阵
            A = np.random.randn(feature_dim, feature_dim)
            cov = A.T @ A + np.eye(feature_dim) * np.random.uniform(std_range[0], std_range[1])
            
            # 生成高斯分布样本
            class_samples = np.random.multivariate_normal(mean, cov, samples_per_class)
            
            X_list.append(class_samples)
            y_list.append(np.full(samples_per_class, class_idx))
    
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    # 打乱数据顺序
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def plot_decision_boundaries_2d(X, y, classifier, num_classes, title="决策边界"):
    """
    绘制2D数据的决策边界
    
    参数:
    X: 特征矩阵 [n_samples, 2]
    y: 标签向量
    classifier: 训练好的分类器
    num_classes: 类别数量
    title: 图表标题
    """
    # 创建网格
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 预测网格点的类别
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = classifier.predict(grid_points)
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界和区域
    plt.figure(figsize=(12, 10))
    cmap = ListedColormap(plt.cm.tab10.colors[:num_classes])
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, levels=num_classes-1)
    
    # 绘制样本点
    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    for class_idx in range(num_classes):
        mask = y == class_idx
        plt.scatter(X[mask, 0], X[mask, 1], 
                   c=[colors[class_idx]], label=f'类别 {class_idx+1}',
                   edgecolors='black', s=50, alpha=0.7)
    
    plt.xlabel('特征 1', fontsize=12)
    plt.ylabel('特征 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def main():
    """主实验流程"""
    print("=" * 60)
    print("实验一：感知器线性分类器的设计实现")
    print("=" * 60)
    
    # ========== 1. 数据生成及规范化处理 ==========
    print("\n【步骤1】数据生成及规范化处理")
    print("-" * 60)
    
    num_classes = 6  # N > 5
    samples_per_class = 50
    feature_dim = 2  # 2D数据便于可视化
    
    print(f"生成 {num_classes} 类数据，每类 {samples_per_class} 个样本，维度: {feature_dim}D")
    
    # 生成训练数据（确保线性可分）
    X_train, y_train = generate_gaussian_data(
        num_classes=num_classes,
        samples_per_class=samples_per_class,
        feature_dim=feature_dim,
        seed=42,
        ensure_separable=True
    )
    
    print(f"训练数据形状: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"类别分布: {np.bincount(y_train)}")
    
    # ========== 2. 训练感知器分类器 ==========
    print("\n【步骤2】训练感知器多类分类器")
    print("-" * 60)
    
    classifier = PerceptronClassifier(
        num_classes=num_classes,
        learning_rate=1.0,
        max_iter=1000
    )
    
    classifier.fit(X_train, y_train)
    print(f"训练完成！权重矩阵形状: {classifier.weights.shape}")
    
    # 计算训练准确率
    train_accuracy = classifier.score(X_train, y_train)
    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    
    # ========== 3. 绘制决策边界 ==========
    print("\n【步骤3】绘制决策边界")
    print("-" * 60)
    
    if feature_dim == 2:
        plot_decision_boundaries_2d(
            X_train, y_train, classifier, num_classes,
            title="感知器多类分类 - 决策边界"
        )
        plt.savefig('perceptron_decision_boundary.png', dpi=300, bbox_inches='tight')
        print("决策边界图已保存为: perceptron_decision_boundary.png")
        plt.show()
    
    # ========== 4. 生成测试数据并分类 ==========
    print("\n【步骤4】生成测试数据并进行分类判别")
    print("-" * 60)
    
    # 生成测试数据（使用相同的分布策略）
    X_test, y_test = generate_gaussian_data(
        num_classes=num_classes,
        samples_per_class=30,  # 每类30个测试样本
        feature_dim=feature_dim,
        seed=123,
        ensure_separable=True
    )
    
    print(f"测试数据形状: X_test {X_test.shape}, y_test {y_test.shape}")
    
    # 预测测试数据
    y_pred = classifier.predict(X_test)
    
    # 计算测试准确率
    test_accuracy = classifier.score(X_test, y_test)
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # 显示分类结果统计
    print("\n分类结果统计:")
    for class_idx in range(num_classes):
        mask = y_test == class_idx
        if np.any(mask):
            correct = np.sum(y_pred[mask] == class_idx)
            total = np.sum(mask)
            print(f"  类别 {class_idx+1}: {correct}/{total} 正确 ({correct/total*100:.1f}%)")
    
    # 绘制测试结果（如果是2D）
    if feature_dim == 2:
        plt.figure(figsize=(12, 5))
        
        # 真实标签
        plt.subplot(1, 2, 1)
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
        for class_idx in range(num_classes):
            mask = y_test == class_idx
            plt.scatter(X_test[mask, 0], X_test[mask, 1],
                       c=[colors[class_idx]], label=f'类别 {class_idx+1}',
                       edgecolors='black', s=50, alpha=0.7)
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title('测试数据 - 真实标签')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 预测标签
        plt.subplot(1, 2, 2)
        for class_idx in range(num_classes):
            mask = y_pred == class_idx
            if np.any(mask):
                plt.scatter(X_test[mask, 0], X_test[mask, 1],
                           c=[colors[class_idx]], label=f'类别 {class_idx+1}',
                           edgecolors='black', s=50, alpha=0.7)
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.title('测试数据 - 预测标签')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
        print("测试结果图已保存为: test_results.png")
        plt.show()
    
    print("\n" + "=" * 60)
    print("实验完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
