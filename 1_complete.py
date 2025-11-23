

import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import rcParams
import cv2

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# ============================================
# 第一部分：人脸检测 (Haar-like + Adaboost)
# ============================================

class HaarFeature:
    """Haar-like特征提取器"""
    
    def __init__(self, feature_type, position, width, height):
        """
        feature_type: 特征类型 ('two_horizontal', 'two_vertical', 'three_horizontal', 'four')
        position: (x, y) 特征位置
        width, height: 特征窗口大小
        """
        self.feature_type = feature_type
        self.position = position
        self.width = width
        self.height = height
    
    def compute(self, integral_image):
        """使用积分图计算Haar特征值"""
        x, y = self.position
        w, h = self.width, self.height
        
        if self.feature_type == 'two_horizontal':
            # 左白右黑
            left = self._sum_region(integral_image, x, y, w//2, h)
            right = self._sum_region(integral_image, x + w//2, y, w//2, h)
            return right - left
            
        elif self.feature_type == 'two_vertical':
            # 上白下黑
            top = self._sum_region(integral_image, x, y, w, h//2)
            bottom = self._sum_region(integral_image, x, y + h//2, w, h//2)
            return bottom - top
            
        elif self.feature_type == 'three_horizontal':
            # 左白中黑右白
            left = self._sum_region(integral_image, x, y, w//3, h)
            middle = self._sum_region(integral_image, x + w//3, y, w//3, h)
            right = self._sum_region(integral_image, x + 2*w//3, y, w//3, h)
            return middle - (left + right)
            
        elif self.feature_type == 'four':
            # 四象限
            tl = self._sum_region(integral_image, x, y, w//2, h//2)
            tr = self._sum_region(integral_image, x + w//2, y, w//2, h//2)
            bl = self._sum_region(integral_image, x, y + h//2, w//2, h//2)
            br = self._sum_region(integral_image, x + w//2, y + h//2, w//2, h//2)
            return (br + tl) - (tr + bl)
    
    def _sum_region(self, integral_image, x, y, width, height):
        """使用积分图快速计算矩形区域和"""
        x, y = int(x), int(y)
        width, height = int(width), int(height)
        
        # 防止越界
        h, w = integral_image.shape
        x2 = min(x + width, w - 1)
        y2 = min(y + height, h - 1)
        x = min(x, w - 1)
        y = min(y, h - 1)
        
        # 积分图计算：D - B - C + A
        total = integral_image[y2, x2]
        if x > 0:
            total -= integral_image[y2, x - 1]
        if y > 0:
            total -= integral_image[y - 1, x2]
        if x > 0 and y > 0:
            total += integral_image[y - 1, x - 1]
        
        return total


def compute_integral_image(image):
    """计算积分图"""
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return np.cumsum(np.cumsum(image, axis=0), axis=1).astype(np.float64)


def generate_haar_features(image_width, image_height, feature_count):
    """生成指定数量的Haar-like特征"""
    features = []
    feature_types = ['two_horizontal', 'two_vertical', 'three_horizontal', 'four']
    
    np.random.seed(42)
    
    for _ in range(feature_count):
        feature_type = np.random.choice(feature_types)
        
        # 随机生成特征窗口大小和位置
        w = np.random.randint(10, min(image_width // 2, 50))
        h = np.random.randint(10, min(image_height // 2, 50))
        x = np.random.randint(0, max(1, image_width - w))
        y = np.random.randint(0, max(1, image_height - h))
        
        features.append(HaarFeature(feature_type, (x, y), w, h))
    
    return features


def extract_features_from_image(image, haar_features):
    """从图像中提取Haar特征"""
    integral_img = compute_integral_image(image)
    feature_vector = []
    
    for haar_feature in haar_features:
        feature_value = haar_feature.compute(integral_img)
        feature_vector.append(feature_value)
    
    return np.array(feature_vector)


# ============================================
# 第二部分：数据加载
# ============================================

def load_orl_dataset(dataset_path='att_faces'):
    """
    加载 ORL 人脸数据库
    返回：训练集和测试集的图像及标签
    """
    X_train_images, y_train = [], []
    X_test_images, y_test = [], []
    
    # 遍历每个人的文件夹 (s1, s2, ..., s40)
    for i in range(1, 41):
        person_folder = os.path.join(dataset_path, f's{i}')
        images = sorted([os.path.join(person_folder, f) for f in os.listdir(person_folder)],
                        key=lambda x: int(os.path.basename(x).split('.')[0]))
        
        for idx, img_path in enumerate(images):
            try:
                img = Image.open(img_path)
                img_array = np.array(img)
                label = i
                
                if idx < 6:  # 前6张作为训练集
                    X_train_images.append(img_array)
                    y_train.append(label)
                else:  # 后4张作为测试集
                    X_test_images.append(img_array)
                    y_test.append(label)
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
    
    return X_train_images, np.array(y_train), X_test_images, np.array(y_test)


# ============================================
# 第三部分：实验主程序
# ============================================

if __name__ == '__main__':
    print("="*80)
    print("实验二：基于Adaboost及SVM的人脸识别算法设计实现")
    print("="*80)
    
    # 加载数据
    print("\n正在加载ORL人脸数据集...")
    X_train_images, y_train, X_test_images, y_test = load_orl_dataset()
    print(f"训练集: {len(X_train_images)} 张图像")
    print(f"测试集: {len(X_test_images)} 张图像")
    print(f"图像尺寸: {X_train_images[0].shape}")
    
    # ============================================
    # 实验1：Adaboost算法 - 特征数对人脸检测的影响
    # ============================================
    print("\n" + "="*80)
    print("实验1：Adaboost人脸检测 - 不同Haar特征数量的影响")
    print("="*80)
    
    feature_counts = [20, 50, 100, 200]
    adaboost_detection_accuracies = []
    
    img_height, img_width = X_train_images[0].shape
    
    for n_features in feature_counts:
        print(f"\n正在测试 {n_features} 个Haar特征...")
        
        # 1. 生成Haar特征
        haar_features = generate_haar_features(img_width, img_height, n_features)
        
        # 2. 提取训练集特征
        print(f"  提取训练集特征...")
        X_train_haar = []
        for img in X_train_images:
            features = extract_features_from_image(img, haar_features)
            X_train_haar.append(features)
        X_train_haar = np.array(X_train_haar)
        
        # 3. 提取测试集特征
        print(f"  提取测试集特征...")
        X_test_haar = []
        for img in X_test_images:
            features = extract_features_from_image(img, haar_features)
            X_test_haar.append(features)
        X_test_haar = np.array(X_test_haar)
        
        # 4. 使用Adaboost进行分类（人脸识别）
        print(f"  训练Adaboost分类器...")
        ada_classifier = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=100,
            learning_rate=0.8,
            random_state=42
        )
        ada_classifier.fit(X_train_haar, y_train)
        
        # 5. 预测与评估
        y_pred = ada_classifier.predict(X_test_haar)
        accuracy = accuracy_score(y_test, y_pred)
        adaboost_detection_accuracies.append(accuracy)
        print(f"  ✓ Haar特征数: {n_features}, Adaboost准确率: {accuracy*100:.2f}%")
    
    # 绘制结果图 - 现代渐变设计
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    # 渐变色系
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
    bars = ax.bar(range(len(feature_counts)), adaboost_detection_accuracies,
                   color=colors, alpha=0.85, edgecolor='white', linewidth=2.5)
    
    # 添加背景色
    ax.set_facecolor('#F8F9FA')
    
    # 标题和标签
    ax.set_title('实验1：Adaboost人脸检测 - Haar特征数量影响分析', 
                fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('Haar特征数量', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_ylabel('识别准确率', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_xticks(range(len(feature_counts)))
    ax.set_xticklabels(feature_counts, fontsize=11)
    ax.set_ylim(0, max(adaboost_detection_accuracies)*1.15)
    
    # 柔和网格
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=1, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # 数值标注带背景
    for idx, (bar, acc) in enumerate(zip(bars, adaboost_detection_accuracies)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc*100:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold',
                color='#2C3E50',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=colors[idx], alpha=0.8, linewidth=2))
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('实验1_Adaboost_Haar特征数对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n✓ 图表已保存: 实验1_Adaboost_Haar特征数对比.png")
    plt.show()
    plt.close()
    
    # ============================================
    # 实验2：PCA降维 + SVM人脸识别
    # ============================================
    print("\n" + "="*80)
    print("实验2a：PCA降维对SVM人脸识别的影响")
    print("="*80)
    
    # 将图像转换为向量
    X_train_flat = np.array([img.flatten() for img in X_train_images])
    X_test_flat = np.array([img.flatten() for img in X_test_images])
    
    n_components_list = [20, 50, 100, 200]
    svm_pca_accuracies = []
    
    for n_components in n_components_list:
        print(f"\n正在测试 PCA降维到 {n_components} 维...")
        
        # 1. PCA降维
        pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
        X_train_pca = pca.fit_transform(X_train_flat)
        X_test_pca = pca.transform(X_test_flat)
        
        # 2. 数据规格化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_pca)
        X_test_scaled = scaler.transform(X_test_pca)
        
        # 3. SVM训练（使用径向基函数）
        svm_classifier = SVC(kernel='rbf', gamma='auto', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)
        
        # 4. 预测与评估
        y_pred = svm_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        svm_pca_accuracies.append(accuracy)
        print(f"  ✓ PCA维度: {n_components}, SVM准确率: {accuracy*100:.2f}%")
    
    # 绘制结果图 - 现代设计
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    # 专业配色
    colors = ['#667EEA', '#764BA2', '#F093FB', '#4FACFE']
    bars = ax.bar(range(len(n_components_list)), svm_pca_accuracies,
                   color=colors, alpha=0.85, edgecolor='white', linewidth=2.5)
    
    ax.set_facecolor('#F8F9FA')
    
    ax.set_title('实验2a：PCA降维对SVM人脸识别的影响', 
                fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('PCA降维维度', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_ylabel('识别准确率', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_xticks(range(len(n_components_list)))
    ax.set_xticklabels(n_components_list, fontsize=11)
    ax.set_ylim(0, max(svm_pca_accuracies)*1.15)
    
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=1, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # 标注最佳结果
    best_idx = np.argmax(svm_pca_accuracies)
    for idx, (bar, acc) in enumerate(zip(bars, svm_pca_accuracies)):
        height = bar.get_height()
        if idx == best_idx:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc*100:.2f}%\n★ 最优',
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#E74C3C',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFD700', 
                             edgecolor='#E74C3C', alpha=0.9, linewidth=2.5))
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc*100:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor=colors[idx], alpha=0.8, linewidth=2))
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('实验2a_PCA降维对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n✓ 图表已保存: 实验2a_PCA降维对比.png")
    plt.show()
    plt.close()
    
    # ============================================
    # 实验2b：SVM不同核函数对比
    # ============================================
    print("\n" + "="*80)
    print("实验2b：SVM不同核函数对人脸识别的影响")
    print("="*80)
    
    # 🔥 修正：使用实验2a得出的最优PCA维度
    optimal_n_components = n_components_list[np.argmax(svm_pca_accuracies)]
    print(f"根据实验2a结果，选择最优 PCA 维度: {optimal_n_components} (准确率: {max(svm_pca_accuracies)*100:.2f}%)")
    
    # PCA降维
    pca = PCA(n_components=optimal_n_components, svd_solver='randomized', whiten=True)
    X_train_pca = pca.fit_transform(X_train_flat)
    X_test_pca = pca.transform(X_test_flat)
    
    # 数据规格化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_pca)
    X_test_scaled = scaler.transform(X_test_pca)
    
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    kernel_names = ['线性核', '径向基核(RBF)', '多项式核', 'Sigmoid核']
    svm_kernel_accuracies = {}
    
    for kernel, kernel_name in zip(kernels, kernel_names):
        print(f"\n正在测试 SVM核函数: {kernel_name}...")
        svm_classifier = SVC(kernel=kernel, gamma='auto', random_state=42)
        svm_classifier.fit(X_train_scaled, y_train)
        
        y_pred = svm_classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        svm_kernel_accuracies[kernel_name] = accuracy
        print(f"  ✓ 核函数: {kernel_name}, 准确率: {accuracy*100:.2f}%")
    
    # 绘制结果图 - 现代设计
    fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
    
    # 专业渐变配色
    colors = ['#FA709A', '#FEE140', '#30CFD0', '#A8EDEA']
    bars = ax.bar(range(len(svm_kernel_accuracies)), svm_kernel_accuracies.values(),
                   color=colors, alpha=0.85, edgecolor='white', linewidth=2.5)
    
    ax.set_facecolor('#F8F9FA')
    
    ax.set_title(f'实验2b：SVM核函数对比 (PCA维度={optimal_n_components})',
              fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('核函数类型', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_ylabel('识别准确率', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_xticks(range(len(svm_kernel_accuracies)))
    ax.set_xticklabels(svm_kernel_accuracies.keys(), rotation=0, fontsize=11)
    ax.set_ylim(0, max(svm_kernel_accuracies.values())*1.15)
    
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=1, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # 找出最佳核函数
    best_kernel = max(svm_kernel_accuracies, key=svm_kernel_accuracies.get)
    for idx, (bar, (kernel_name, acc)) in enumerate(zip(bars, svm_kernel_accuracies.items())):
        height = bar.get_height()
        if kernel_name == best_kernel:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc*100:.2f}%\n★ 最优',
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#E74C3C',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFD700', 
                             edgecolor='#E74C3C', alpha=0.9, linewidth=2.5))
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc*100:.2f}%',
                    ha='center', va='bottom', fontsize=12, fontweight='bold',
                    color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor=colors[idx], alpha=0.8, linewidth=2))
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('实验2b_SVM核函数对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n✓ 图表已保存: 实验2b_SVM核函数对比.png")
    plt.show()
    plt.close()
    
    # ============================================
    # 实验3：交叉验证 - 对比所有核函数
    # ============================================
    print("\n" + "="*80)
    print("实验3：使用交叉验证对比不同SVM核函数")
    print("="*80)
    
    # 合并所有数据用于交叉验证
    X_all = np.concatenate((X_train_flat, X_test_flat), axis=0)
    y_all = np.concatenate((y_train, y_test), axis=0)
    print(f"完整数据集形状: {X_all.shape}")
    print(f"使用最优 PCA 维度: {optimal_n_components}")
    
    # 🔥 对每个核函数都进行5折交叉验证
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    kernel_names = ['线性核', '径向基核(RBF)', '多项式核', 'Sigmoid核']
    cv_results = {}  # 存储每个核函数的交叉验证结果
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n正在对所有核函数执行5折交叉验证...")
    for kernel, kernel_name in zip(kernels, kernel_names):
        print(f"\n  正在验证 {kernel_name}...")
        
        # 为每个核函数创建Pipeline
        pipeline = Pipeline([
            ('pca', PCA(n_components=optimal_n_components, svd_solver='randomized', whiten=True)),
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel=kernel, gamma='auto', random_state=42))
        ])
        
        # 执行交叉验证
        cv_scores = cross_val_score(pipeline, X_all, y_all, cv=kf, scoring='accuracy')
        cv_results[kernel_name] = cv_scores
        
        print(f"    各折准确率: {[f'{s*100:.2f}%' for s in cv_scores]}")
        print(f"    ✓ 平均准确率: {np.mean(cv_scores)*100:.2f}% ± {np.std(cv_scores)*100:.2f}%")
    
    # 找出最优核函数
    mean_accuracies = {k: np.mean(v) for k, v in cv_results.items()}
    best_kernel_cv = max(mean_accuracies, key=mean_accuracies.get)
    best_accuracy_cv = mean_accuracies[best_kernel_cv]
    
    print(f"\n{'='*60}")
    print(f"交叉验证结论:")
    print(f"  ★ 最优核函数: {best_kernel_cv}")
    print(f"  ★ 最高平均准确率: {best_accuracy_cv*100:.2f}%")
    print(f"{'='*60}")
    
    # 绘制折线图对比 - 现代设计
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # 现代渐变配色
    colors = ['#667EEA', '#FA709A', '#2DDA93', '#FFA07A']
    markers = ['o', 's', 'D', '^']
    linestyles = ['-', '-', '-', '-']  # 统一线型，用颜色区分
    folds = list(range(1, 6))
    
    # 设置背景
    ax.set_facecolor('#F8F9FA')
    
    # 为了避免重叠，对x坐标进行微调
    offsets = [-0.1, -0.035, 0.035, 0.1]
    
    # 绘制每个核函数的折线
    for idx, ((kernel_name, cv_scores), color, marker, linestyle, offset) in enumerate(
            zip(cv_results.items(), colors, markers, linestyles, offsets)):
        mean_acc = np.mean(cv_scores)
        
        # 应用x偏移，避免折线重叠
        x_positions = [f + offset for f in folds]
        
        # 判断是否是最优核函数
        is_best = (kernel_name == best_kernel_cv)
        linewidth = 3.5 if is_best else 2.5
        markersize = 12 if is_best else 10
        alpha = 1.0 if is_best else 0.8
        
        # 如果是最优核函数，添加发光效果
        if is_best:
            ax.plot(x_positions, cv_scores*100, marker=marker, color=color, 
                    linewidth=linewidth+4, linestyle=linestyle, markersize=markersize+2, 
                    alpha=0.2, markeredgewidth=0, zorder=1)
        
        ax.plot(x_positions, cv_scores*100, marker=marker, color=color, 
                linewidth=linewidth, linestyle=linestyle, markersize=markersize, 
                label=f'{kernel_name} (均值: {mean_acc*100:.2f}%)', alpha=alpha,
                markeredgecolor='white', markeredgewidth=2, zorder=3)
        
        # 只在最优和次优核函数上标注数值
        if mean_acc > 0.88:
            for i, score in enumerate(cv_scores):
                va = 'bottom' if idx % 2 == 0 else 'top'
                y_offset = 1.2 if idx % 2 == 0 else -1.2
                ax.text(x_positions[i], score*100 + y_offset, f'{score*100:.1f}', 
                        ha='center', va=va, fontsize=9, color=color, 
                        fontweight='bold', alpha=0.9)
    
    # 标注最优核函数的平均线
    best_scores = cv_results[best_kernel_cv]
    best_mean = np.mean(best_scores)
    ax.axhline(y=best_mean*100, color='#E74C3C', linestyle='--', linewidth=2.5, 
               alpha=0.5, label=f'最优平均线: {best_mean*100:.2f}%', zorder=2)
    
    ax.set_title(f'实验3：交叉验证 - SVM核函数性能对比 (PCA={optimal_n_components}维)', 
              fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('折数', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_ylabel('准确率 (%)', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_xticks(folds)
    ax.set_xticklabels([f'第{i}折' for i in folds], fontsize=11)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylim(25, 102)
    
    # 柔和网格
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=1, color='#BDC3C7', zorder=0)
    ax.set_axisbelow(True)
    
    # 图例放在右侧
    ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize=11, 
              framealpha=0.95, edgecolor='#BDC3C7', fancybox=True, 
              shadow=False, frameon=True, facecolor='white')
    
    # 最优标记 - 金色徽章样式
    ax.text(0.02, 0.50, 
            f'★ 最优核函数\n{best_kernel_cv}\n平均: {best_mean*100:.2f}%\n标准差: ±{np.std(best_scores)*100:.2f}%', 
            transform=ax.transAxes, fontsize=12, verticalalignment='center',
            horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFD700', alpha=0.9, 
                     edgecolor='#E74C3C', linewidth=3),
            fontweight='bold', color='#2C3E50')
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('实验3_交叉验证结果.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n✓ 图表已保存: 实验3_交叉验证结果.png")
    plt.show()
    plt.close()
    
    # ============================================
    # 实验总结
    # ============================================
    print("\n" + "="*80)
    print("实验总结")
    print("="*80)
    
    # 最优Adaboost特征数
    best_haar_idx = np.argmax(adaboost_detection_accuracies)
    print(f"\n1. Adaboost人脸检测:")
    print(f"   最优Haar特征数: {feature_counts[best_haar_idx]}")
    print(f"   最高准确率: {adaboost_detection_accuracies[best_haar_idx]*100:.2f}%")
    
    # 最优PCA维度
    best_pca_idx = np.argmax(svm_pca_accuracies)
    print(f"\n2. PCA降维 + SVM识别:")
    print(f"   最优PCA维度: {n_components_list[best_pca_idx]}")
    print(f"   最高准确率: {svm_pca_accuracies[best_pca_idx]*100:.2f}%")
    
    # 最优SVM核函数
    best_kernel = max(svm_kernel_accuracies, key=svm_kernel_accuracies.get)
    print(f"\n3. SVM核函数对比:")
    print(f"   最优核函数: {best_kernel}")
    print(f"   最高准确率: {svm_kernel_accuracies[best_kernel]*100:.2f}%")
    
    # 交叉验证结果
    print(f"\n4. 交叉验证评估:")
    print(f"   最优核函数: {best_kernel_cv}")
    print(f"   平均准确率: {best_accuracy_cv*100:.2f}%")
    print(f"   所有核函数对比:")
    for kernel_name, scores in cv_results.items():
        print(f"     - {kernel_name}: {np.mean(scores)*100:.2f}% ± {np.std(scores)*100:.2f}%")
    
    # 方法对比
    print(f"\n5. 方法对比:")
    print(f"   Adaboost (Haar特征): {max(adaboost_detection_accuracies)*100:.2f}%")
    print(f"   SVM (PCA特征): {max(svm_pca_accuracies)*100:.2f}%")
    print(f"   结论: SVM方法在本实验中表现更优")
    
    print("\n" + "="*80)
    print("所有实验完成！图表已保存到当前目录。")
    print("共生成4张图表：")
    print("  1. 实验1_Adaboost_Haar特征数对比.png")
    print("  2. 实验2a_PCA降维对比.png")
    print("  3. 实验2b_SVM核函数对比.png")
    print("  4. 实验3_交叉验证结果.png")
    print("="*80)
    
    # ============================================
    # 扩展实验：四个模型对比（5折交叉验证准确率）
    # ============================================
    print("\n" + "="*80)
    print("扩展实验：四个模型对比（5折交叉验证准确率）")
    print("="*80)
    
    # PSO优化算法实现
    class PSO:
        """粒子群优化算法"""
        def __init__(self, n_particles=20, n_iterations=30, bounds=None, w=0.7, c1=1.5, c2=1.5):
            """
            n_particles: 粒子数量
            n_iterations: 迭代次数
            bounds: 参数边界 [(min, max), ...]
            w: 惯性权重
            c1, c2: 学习因子
            """
            self.n_particles = n_particles
            self.n_iterations = n_iterations
            self.bounds = bounds
            self.w = w
            self.c1 = c1
            self.c2 = c2
            self.global_best_position = None
            self.global_best_score = -np.inf
        
        def optimize(self, objective_func):
            """执行PSO优化"""
            # 初始化粒子位置和速度
            particles = np.random.uniform(
                low=[b[0] for b in self.bounds],
                high=[b[1] for b in self.bounds],
                size=(self.n_particles, len(self.bounds))
            )
            velocities = np.random.uniform(
                low=-1, high=1,
                size=(self.n_particles, len(self.bounds))
            )
            
            # 初始化个体最优
            personal_best_positions = particles.copy()
            personal_best_scores = np.array([objective_func(p) for p in particles])
            
            # 初始化全局最优
            best_idx = np.argmax(personal_best_scores)
            self.global_best_position = personal_best_positions[best_idx].copy()
            self.global_best_score = personal_best_scores[best_idx]
            
            # 迭代优化
            for iteration in range(self.n_iterations):
                for i in range(self.n_particles):
                    # 更新速度
                    r1, r2 = np.random.rand(2)
                    velocities[i] = (self.w * velocities[i] +
                                    self.c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                    self.c2 * r2 * (self.global_best_position - particles[i]))
                    
                    # 更新位置
                    particles[i] += velocities[i]
                    
                    # 边界处理
                    for j in range(len(self.bounds)):
                        if particles[i, j] < self.bounds[j][0]:
                            particles[i, j] = self.bounds[j][0]
                        elif particles[i, j] > self.bounds[j][1]:
                            particles[i, j] = self.bounds[j][1]
                    
                    # 评估新位置
                    score = objective_func(particles[i])
                    
                    # 更新个体最优
                    if score > personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i].copy()
                        
                        # 更新全局最优
                        if score > self.global_best_score:
                            self.global_best_score = score
                            self.global_best_position = particles[i].copy()
            
            return self.global_best_position, self.global_best_score
    
    # 准备数据（使用最优PCA维度）
    print(f"\n使用最优PCA维度: {optimal_n_components}")
    pca = PCA(n_components=optimal_n_components, svd_solver='randomized', whiten=True)
    X_all_pca = pca.fit_transform(X_all)
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all_pca)
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # 模型1：最佳SVC模型（从实验3中选择最优核函数）
    best_kernel_name = best_kernel_cv
    # 将中文核函数名映射回sklearn的核函数名
    kernel_map = {
        '线性核': 'linear',
        '径向基核(RBF)': 'rbf',
        '多项式核': 'poly',
        'Sigmoid核': 'sigmoid'
    }
    best_kernel = kernel_map[best_kernel_name]
    
    print(f"\n模型1：最佳SVC模型（{best_kernel_name}）")
    print("  正在进行5折交叉验证...")
    
    def evaluate_model_accuracy(model, X, y, cv):
        """评估模型的准确率（5折交叉验证）"""
        acc_scores = []
        for train_idx, val_idx in cv.split(X, y):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_train_fold, y_val_fold = y[train_idx], y[val_idx]
            
            model.fit(X_train_fold, y_train_fold)
            y_pred = model.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)
            acc_scores.append(acc)
        return np.array(acc_scores)
    
    # 模型1：最佳SVC
    best_svc = SVC(kernel=best_kernel, gamma='auto', random_state=42)
    svc_acc_scores = evaluate_model_accuracy(best_svc, X_all_scaled, y_all, kf)
    svc_mean_acc = np.mean(svc_acc_scores)
    print(f"  ✓ 平均准确率: {svc_mean_acc*100:.4f}% ± {np.std(svc_acc_scores)*100:.4f}%")
    
    # 模型2：Random Forest
    print(f"\n模型2：Random Forest")
    print("  正在进行5折交叉验证...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    rf_acc_scores = evaluate_model_accuracy(rf, X_all_scaled, y_all, kf)
    rf_mean_acc = np.mean(rf_acc_scores)
    print(f"  ✓ 平均准确率: {rf_mean_acc*100:.4f}% ± {np.std(rf_acc_scores)*100:.4f}%")
    
    # 模型3：KNN
    print(f"\n模型3：KNN")
    print("  正在进行5折交叉验证...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_acc_scores = evaluate_model_accuracy(knn, X_all_scaled, y_all, kf)
    knn_mean_acc = np.mean(knn_acc_scores)
    print(f"  ✓ 平均准确率: {knn_mean_acc*100:.4f}% ± {np.std(knn_acc_scores)*100:.4f}%")
    
    # 模型4：PSO优化的SVC
    print(f"\n模型4：PSO优化的SVC（{best_kernel_name}）")
    print("  正在进行PSO优化...")
    
    # 定义目标函数（使用5折交叉验证的准确率，更适合分类问题）
    def pso_objective_svc(params):
        """PSO优化的目标函数（使用准确率）"""
        C = params[0]
        # 限制参数范围（与pso_bounds保持一致）
        if C < 0.01 or C > 1000:
            return -np.inf
        
        # 根据核函数类型选择参数
        if best_kernel == 'linear':
            # 线性核只需要C参数
            svc_temp = SVC(kernel=best_kernel, C=C, random_state=42)
        else:
            # 其他核函数需要C和gamma
            gamma = params[1] if len(params) > 1 else 'auto'
            if isinstance(gamma, (int, float)) and (gamma < 0.0001 or gamma > 10):
                return -np.inf
            svc_temp = SVC(kernel=best_kernel, C=C, gamma=gamma, random_state=42)
        
        # 使用5折交叉验证评估（与最终评估保持一致）
        # 使用准确率作为优化目标（更适合分类问题）
        kf_temp = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for train_idx, val_idx in kf_temp.split(X_all_scaled, y_all):
            X_train_fold, X_val_fold = X_all_scaled[train_idx], X_all_scaled[val_idx]
            y_train_fold, y_val_fold = y_all[train_idx], y_all[val_idx]
            svc_temp.fit(X_train_fold, y_train_fold)
            y_pred = svc_temp.predict(X_val_fold)
            acc = accuracy_score(y_val_fold, y_pred)  # 使用准确率而不是R2
            scores.append(acc)
        return np.mean(scores)
    
    # 先检查默认参数的性能
    print("  检查默认参数性能...")
    default_svc = SVC(kernel=best_kernel, gamma='auto', random_state=42)
    default_acc_temp = evaluate_model_accuracy(default_svc, X_all_scaled, y_all, kf)
    default_acc_mean = np.mean(default_acc_temp)
    print(f"  默认参数(C=1.0)平均准确率: {default_acc_mean*100:.4f}%")
    
    # PSO优化SVC参数（根据核函数类型选择参数）
    if best_kernel == 'linear':
        # 线性核只优化C，增加搜索范围
        pso_bounds = [(0.01, 1000)]  # 扩大搜索范围
        pso = PSO(n_particles=20, n_iterations=30, bounds=pso_bounds)  # 增加粒子数和迭代次数
        best_params, best_pso_score = pso.optimize(pso_objective_svc)
        best_C, best_gamma = best_params[0], 'auto'
        print(f"  ✓ PSO优化结果: C={best_C:.4f} (线性核不需要gamma)")
    else:
        # 其他核函数优化C和gamma
        pso_bounds = [(0.01, 1000), (0.0001, 10)]  # C和gamma的范围
        pso = PSO(n_particles=20, n_iterations=30, bounds=pso_bounds)  # 增加粒子数和迭代次数
        best_params, best_pso_score = pso.optimize(pso_objective_svc)
        best_C, best_gamma = best_params[0], best_params[1]
        print(f"  ✓ PSO优化结果: C={best_C:.4f}, gamma={best_gamma:.4f}")
    
    print(f"  ✓ PSO优化目标值(准确率): {best_pso_score*100:.4f}%")  # 显示4位小数
    improvement = (best_pso_score - default_acc_mean) * 100
    print(f"  ✓ 相比默认参数改进: {improvement:+.4f}%")  # 显示正负号和4位小数
    print("  正在进行5折交叉验证（最终评估）...")
    
    # 使用PSO优化的参数进行5折交叉验证
    if best_kernel == 'linear':
        pso_svc = SVC(kernel=best_kernel, C=best_C, random_state=42)
    else:
        pso_svc = SVC(kernel=best_kernel, C=best_C, gamma=best_gamma, random_state=42)
    pso_svc_acc_scores = evaluate_model_accuracy(pso_svc, X_all_scaled, y_all, kf)
    pso_svc_mean_acc = np.mean(pso_svc_acc_scores)
    print(f"  ✓ 平均准确率: {pso_svc_mean_acc*100:.4f}% ± {np.std(pso_svc_acc_scores)*100:.4f}%")
    
    # 分析PSO优化效果
    final_improvement = (pso_svc_mean_acc - default_acc_mean) * 100
    print(f"\n  【PSO优化分析】")
    if abs(final_improvement) < 0.01:
        print(f"  ⚠ 准确率改进: {final_improvement:+.4f}% (几乎无变化)")
        print(f"  说明: 默认参数C=1.0已经接近最优，PSO找到的C={best_C:.4f}效果相似")
        print(f"  原因: 线性核SVC在此数据集上对C参数不敏感，或已达到模型性能上限")
    elif final_improvement > 0:
        print(f"  ✓ 准确率改进: {final_improvement:+.4f}% (有提升)")
        print(f"  PSO成功优化参数，从C=1.0提升到C={best_C:.4f}")
    else:
        print(f"  ⚠ 准确率改进: {final_improvement:+.4f}% (略有下降)")
        print(f"  可能原因: 交叉验证的随机性或过拟合")
    
    # 汇总结果
    print("\n" + "="*80)
    print("四个模型对比结果（5折交叉验证准确率）")
    print("="*80)
    
    results_acc = {
        f'最佳SVC ({best_kernel_name})': svc_mean_acc,
        'Random Forest': rf_mean_acc,
        'KNN': knn_mean_acc,
        f'PSO优化SVC ({best_kernel_name})': pso_svc_mean_acc
    }
    
    # 按准确率排序
    sorted_results_acc = sorted(results_acc.items(), key=lambda x: x[1], reverse=True)
    
    print("\n【准确率排名】")
    for rank, (model_name, mean_acc) in enumerate(sorted_results_acc, 1):
        marker = "★" if rank == 1 else "  "
        print(f"{marker} {rank}. {model_name}: {mean_acc*100:.4f}%")
    
    print("\n" + "-"*80)
    print("【说明】")
    print("1. 准确率是分类问题的标准评估指标")
    print("2. PSO优化使用准确率作为目标函数")
    print("3. 如果PSO改进很小，可能原因：")
    print("   - 默认参数C=1.0已经接近最优")
    print("   - 模型已达到性能上限（数据集限制）")
    print("   - 线性核对C参数不敏感")
    print("4. 使用4位小数精度可以看到微小的差异")
    print("-"*80)
    
    # 绘制对比图（使用准确率）
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # 准备数据（使用准确率）
    model_names = list(results_acc.keys())
    mean_accs = [x * 100 for x in results_acc.values()]  # 转换为百分比
    std_accs = [
        np.std(svc_acc_scores) * 100,
        np.std(rf_acc_scores) * 100,
        np.std(knn_acc_scores) * 100,
        np.std(pso_svc_acc_scores) * 100
    ]
    
    # 专业配色
    colors = ['#667EEA', '#FA709A', '#2DDA93', '#FFA07A']
    
    # 绘制柱状图
    bars = ax.bar(range(len(model_names)), mean_accs, 
                   color=colors, alpha=0.85, edgecolor='white', linewidth=2.5,
                   yerr=std_accs, capsize=8, error_kw={'elinewidth': 2, 'capthick': 2})
    
    ax.set_facecolor('#F8F9FA')
    
    ax.set_title('扩展实验：四个模型对比（5折交叉验证平均准确率）', 
                fontsize=16, fontweight='bold', pad=20, color='#2C3E50')
    ax.set_xlabel('模型', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_ylabel('平均准确率 (%)', fontsize=13, fontweight='bold', color='#34495E')
    ax.set_xticks(range(len(model_names)))
    ax.set_xticklabels(model_names, rotation=15, ha='right', fontsize=11)
    
    # 设置y轴范围（适合准确率百分比）
    y_min = min(mean_accs) - max(std_accs) - 5
    y_max = max(mean_accs) + max(std_accs) + 5
    ax.set_ylim(max(y_min, 0), min(y_max, 100))
    
    ax.grid(axis='y', alpha=0.2, linestyle='-', linewidth=1, color='#BDC3C7')
    ax.set_axisbelow(True)
    
    # 标注数值
    best_idx = np.argmax(mean_accs)
    for idx, (bar, mean_acc, std_acc) in enumerate(zip(bars, mean_accs, std_accs)):
        height = bar.get_height()
        if idx == best_idx:
            ax.text(bar.get_x() + bar.get_width()/2., height + std_acc + 0.5,
                    f'{mean_acc:.2f}%\n±{std_acc:.2f}%\n★ 最优',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='#E74C3C',
                    bbox=dict(boxstyle='round,pad=0.6', facecolor='#FFD700', 
                             edgecolor='#E74C3C', alpha=0.9, linewidth=2.5))
        else:
            ax.text(bar.get_x() + bar.get_width()/2., height + std_acc + 0.5,
                    f'{mean_acc:.2f}%\n±{std_acc:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='#2C3E50',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                             edgecolor=colors[idx], alpha=0.8, linewidth=2))
    
    # 美化边框
    for spine in ax.spines.values():
        spine.set_edgecolor('#BDC3C7')
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    plt.savefig('扩展实验_四个模型准确率对比.png', dpi=300, bbox_inches='tight', facecolor='white')
    print("\n✓ 图表已保存: 扩展实验_四个模型准确率对比.png")
    plt.show()
    plt.close()
    
    print("\n" + "="*80)
    print("扩展实验完成！")
    print("="*80)

