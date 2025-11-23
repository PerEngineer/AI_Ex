"""
实验一 选做内容：基于ORL人脸库的多类人脸识别
使用LDA降维 + 感知器最大值判决准则 (Scikit-learn实现)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import Perceptron # <--- 导入Scikit-learn的感知器
import seaborn as sns
import warnings

# --- 1. 环境与配置 ---

# 尝试设置中文字体，如果失败则回退
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    warnings.warn("中文字体设置失败，将使用默认字体。")

# --- 2. 数据加载与处理函数 ---

def load_orl_faces(dataset_path='att_faces'):
    """
    加载ORL人脸数据集并进行预处理
    """
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"数据集路径 '{dataset_path}' 不存在。\n"
                              f"请从 https://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html 下载并解压。")

    images = []
    labels = []
    
    person_dirs = sorted([d for d in os.listdir(dataset_path) 
                         if os.path.isdir(os.path.join(dataset_path, d)) and d.startswith('s')])
    
    for person_idx, person_dir in enumerate(person_dirs):
        person_path = os.path.join(dataset_path, person_dir)
        image_files = sorted([f for f in os.listdir(person_path) if f.endswith('.pgm')])
        
        for img_file in image_files:
            img_path = os.path.join(person_path, img_file)
            img = Image.open(img_path)
            # 转换为浮点型并归一化到 [0, 1] 区间
            img_array = np.array(img, dtype=np.float64).flatten() / 255.0
            images.append(img_array)
            labels.append(person_idx)
    
    return np.array(images), np.array(labels)


def apply_lda(X_train, y_train, X_test, n_components=None):
    """
    使用LDA进行降维
    """
    num_classes = len(np.unique(y_train))
    if n_components is None:
        n_components = min(num_classes - 1, X_train.shape[1])
    
    print(f"原始特征维度: {X_train.shape[1]}")
    print(f"LDA降维后维度: {n_components}")
    
    # 备注：当特征维度远大于样本数时（p >> n），类内散度矩阵是奇异的。
    # scikit-learn的LDA实现中，solver='svd'可以很好地处理这种情况。
    lda = LDA(n_components=n_components, solver='svd')
    X_train_lda = lda.fit_transform(X_train, y_train)
    X_test_lda = lda.transform(X_test)
    
    return X_train_lda, X_test_lda, lda


# --- 3. 可视化函数 (与上一版相同，无需修改) ---

def visualize_faces(images, labels, title="人脸样本", n_samples_per_class=3):
    unique_labels = np.unique(labels)
    n_classes = min(len(unique_labels), 10)
    fig, axes = plt.subplots(n_classes, n_samples_per_class, 
                            figsize=(n_samples_per_class * 1.5, n_classes * 1.8))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    if n_classes == 1: axes = axes.reshape(1, -1)
    if n_samples_per_class == 1: axes = axes.reshape(-1, 1)
    img_height, img_width = 112, 92
    for i, label in enumerate(unique_labels[:n_classes]):
        class_images = images[labels == label]
        for j in range(n_samples_per_class):
            ax = axes[i, j]
            if j < len(class_images):
                img = class_images[j].reshape(img_height, img_width)
                ax.imshow(img, cmap='gray')
                if j == 0:
                    ax.set_title(f'类别 {label+1}', fontsize=10, loc='left', pad=2)
            ax.axis('off')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def plot_classification_results(X_test, y_test, y_pred, title="分类结果"):
    dim = X_test.shape[1]
    if dim not in [2, 3]:
        print(f"特征维度为 {dim}，无法绘制散点图（仅支持2D或3D）。")
        return None
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    unique_labels = np.unique(np.concatenate((y_test, y_pred)))
    colors = plt.cm.get_cmap('tab20', len(unique_labels))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d' if dim == 3 else None)
    for i, label in enumerate(unique_labels):
        mask = y_test == label
        if dim == 2:
            ax1.scatter(X_test[mask, 0], X_test[mask, 1], color=colors(i), 
                        label=f'类别 {label+1}', edgecolors='k', s=50, alpha=0.8)
        else:
            ax1.scatter(X_test[mask, 0], X_test[mask, 1], X_test[mask, 2], 
                        color=colors(i), label=f'类别 {label+1}', s=50, alpha=0.8)
    ax1.set_title('测试数据 - 真实标签')
    ax1.set_xlabel('LDA特征 1'); ax1.set_ylabel('LDA特征 2')
    if dim == 3: ax1.set_zlabel('LDA特征 3')
    ax1.grid(True, alpha=0.3)
    ax2 = fig.add_subplot(1, 2, 2, projection='3d' if dim == 3 else None)
    for i, label in enumerate(unique_labels):
        mask = y_pred == label
        if dim == 2:
            ax2.scatter(X_test[mask, 0], X_test[mask, 1], color=colors(i), 
                        label=f'类别 {label+1}', edgecolors='k', s=50, alpha=0.8)
        else:
            ax2.scatter(X_test[mask, 0], X_test[mask, 1], X_test[mask, 2], 
                        color=colors(i), label=f'类别 {label+1}', s=50, alpha=0.8)
    ax2.set_title('测试数据 - 预测标签')
    ax2.set_xlabel('LDA特征 1'); ax2.set_ylabel('LDA特征 2')
    if dim == 3: ax2.set_zlabel('LDA特征 3')
    ax2.grid(True, alpha=0.3)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(1.0, 0.9), loc='upper left', title="类别")
    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    return fig

def plot_confusion_matrix(y_test, y_pred, num_classes):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=range(1, num_classes + 1), 
                yticklabels=range(1, num_classes + 1))
    ax.set_xlabel('预测类别', fontsize=12)
    ax.set_ylabel('真实类别', fontsize=12)
    ax.set_title('混淆矩阵', fontsize=16, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    return fig


# --- 4. 主实验流程 ---

def main():
    """主实验流程"""
    print("=" * 70)
    print("实验一 选做内容：基于ORL人脸库的多类人脸识别 (Scikit-learn实现)")
    print("使用LDA降维 + 感知器最大值判决准则")
    print("=" * 70)
    
    # ========== 1. 加载ORL人脸数据集 ==========
    print("\n【步骤1】加载并预处理ORL人脸数据集")
    print("-" * 70)
    try:
        images, labels = load_orl_faces('att_faces')
    except FileNotFoundError as e:
        print(e)
        return
    num_classes = len(np.unique(labels))
    print(f"加载完成！总样本数: {len(images)}, 类别数: {num_classes}")
    
    fig1 = visualize_faces(images, labels, title="ORL人脸库样本")
    plt.savefig('face_samples.png', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # ========== 2. 划分训练集和测试集 ==========
    print("\n【步骤2】划分训练集和测试集 (采用分层抽样)")
    print("-" * 70)
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    print(f"训练集: {X_train.shape[0]} 个样本, 测试集: {X_test.shape[0]} 个样本")
    
    # ========== 3. LDA降维 ==========
    print("\n【步骤3】使用LDA进行数据降维")
    print("-" * 70)
    # 降维到 num_classes - 1 维可以最大化类别可分性
    n_components_lda = num_classes - 1 
    X_train_lda, X_test_lda, lda_model = apply_lda(X_train, y_train, X_test, n_components=n_components_lda)
    
    # ========== 4. 训练感知器分类器 ==========
    print("\n【步骤4】训练Scikit-learn感知器分类器")
    print("-" * 70)
    # eta0: 学习率
    # max_iter: 最大迭代次数(轮数)
    # random_state: 保证结果可复现
    classifier = Perceptron(max_iter=2000, eta0=0.1, random_state=42, tol=1e-4)
    classifier.fit(X_train_lda, y_train)
    
    # ========== 5. 测试与评估 ==========
    print("\n【步骤5】测试分类性能并进行评估")
    print("-" * 70)
    y_train_pred = classifier.predict(X_train_lda)
    y_test_pred = classifier.predict(X_test_lda)
    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"训练集准确率: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"测试集准确率: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # ========== 6. 可视化结果 ==========
    print("\n【步骤6】可视化实验结果")
    print("-" * 70)
    
    fig_cm = plot_confusion_matrix(y_test, y_test_pred, num_classes)
    fig_cm.savefig('face_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("混淆矩阵图已保存为: face_confusion_matrix.png")
    plt.close(fig_cm)
    
    # 如果希望可视化散点图，可以重新运行LDA并降到2维或3维
    print("\n为进行散点图可视化，将数据降至3维...")
    X_train_lda_3d, X_test_lda_3d, _ = apply_lda(X_train, y_train, X_test, n_components=3)
    # 使用相同的模型在新维度上重新训练和预测
    classifier_3d = Perceptron(max_iter=2000, eta0=0.1, random_state=42, tol=1e-4).fit(X_train_lda_3d, y_train)
    y_test_pred_3d = classifier_3d.predict(X_test_lda_3d)
    
    fig_scatter = plot_classification_results(X_test_lda_3d, y_test, y_test_pred_3d, 
                                              title="LDA降维至3D后的分类结果")
    if fig_scatter:
        fig_scatter.savefig('face_classification_3d.png', dpi=300, bbox_inches='tight')
        print("3D分类结果图已保存为: face_classification_3d.png")
        plt.close(fig_scatter)

    print("\n" + "=" * 70)
    print("实验完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()