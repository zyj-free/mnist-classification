import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
import time
import os
from models.init import get_model,get_available_models

# 设置日志
def setup_logging(model_name):
    """设置日志记录"""
    if not os.path.exists('logs'):
        os.makedirs('logs')

    log_filename = f'logs/{model_name}_{time.strftime("%Y%m%d_%H%M%S")}.log'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def train_model(model, train_loader, test_loader, device, model_name, num_epochs=10):
    """训练模型"""
    logger = setup_logging(model_name)

    # 优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 训练记录
    train_losses = []
    test_accuracies = []
    best_accuracy = 0.0

    logger.info(f"开始训练模型: {model_name}")
    logger.info(f"设备: {device}")
    logger.info(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if batch_idx % 100 == 0:
                logger.info(f'训练 Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                            f'({100. * batch_idx / len(train_loader):.0f}%)]\t损失: {loss.item():.6f}')

        # 计算平均训练损失
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 评估阶段
        test_accuracy = evaluate_model(model, test_loader, device)
        test_accuracies.append(test_accuracy)

        logger.info(f'Epoch {epoch}: 平均训练损失 = {avg_train_loss:.4f}, 测试准确率 = {test_accuracy:.2f}%')

        # 保存最佳模型
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            if not os.path.exists('checkpoints'):
                os.makedirs('checkpoints')
            torch.save(model.state_dict(), f'checkpoints/{model_name}_best.pth')
            logger.info(f'新的最佳准确率: {test_accuracy:.2f}%, 模型已保存')

    logger.info(f'训练完成! 最佳测试准确率: {best_accuracy:.2f}%')

    # 保存训练记录
    results = {
        'train_losses': train_losses,
        'test_accuracies': test_accuracies,
        'best_accuracy': best_accuracy
    }
    torch.save(results, f'logs/{model_name}_results.pth')

    return best_accuracy


def evaluate_model(model, test_loader, device):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


def main():
    # 参数设置
    model_name = "resnet34"  # 可以改为 "resnet18" 或 "resnet34"
    batch_size = 64
    num_epochs = 10

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    # 创建模型
    print(f"可用模型: {get_available_models()}")
    print(f"创建模型: {model_name}")

    model = get_model(model_name)
    model = model.to(device)

    # 开始训练
    best_accuracy = train_model(model, train_loader, test_loader, device, model_name, num_epochs)
    print(f"\n最终结果 - 模型: {model_name}, 最佳准确率: {best_accuracy:.2f}%")


if __name__ == '__main__':
    main()