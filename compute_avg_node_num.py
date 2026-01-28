import re
import sys
from pathlib import Path

def extract_node_counts_from_log(log_file_path):
    """
    从日志文件中提取HC-SOINN的节点数量
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        tuple: (节点数量列表, 类别总数)
    """
    node_counts = []
    total_classes = 0
    
    with open(log_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取节点数量：匹配 "soinn_refined=数字"
            match = re.search(r'soinn_refined=(\d+)', line)
            if match:
                node_count = int(match.group(1))
                node_counts.append(node_count)
            
            # 尝试从最后任务中提取总类别数
            # 匹配 "Task X, Epoch" 或 "total classes: X"
            class_match = re.search(r'total classes:\s*(\d+)', line)
            if class_match:
                total_classes = max(total_classes, int(class_match.group(1)))
    
    # 如果没有找到total classes，使用节点数量作为类别数
    if total_classes == 0:
        total_classes = len(node_counts)
    
    return node_counts, total_classes

def compute_avg_nodes_per_class(log_file_path):
    """
    计算HC-SOINN的平均节点数（每个类别的平均节点数）
    
    Args:
        log_file_path: 日志文件路径
        
    Returns:
        dict: 包含统计信息的字典
    """
    node_counts, total_classes = extract_node_counts_from_log(log_file_path)
    
    if len(node_counts) == 0:
        return {
            'error': 'No node counts found in log file',
            'total_classes': total_classes,
            'node_counts': []
        }
    
    total_nodes = sum(node_counts)
    avg_nodes = total_nodes / len(node_counts)
    avg_nodes_per_class = total_nodes / total_classes if total_classes > 0 else avg_nodes
    
    return {
        'log_file': str(log_file_path),
        'total_classes': total_classes,
        'classes_with_nodes': len(node_counts),
        'total_nodes': total_nodes,
        'avg_nodes_per_class_with_data': avg_nodes,  # 有数据的类的平均节点数
        'avg_nodes_per_class': avg_nodes_per_class,  # 所有类的平均节点数（除以总类别数）
        'min_nodes': min(node_counts),
        'max_nodes': max(node_counts),
        'node_counts': node_counts
    }

def main():
    """主函数：支持命令行参数或交互式输入"""
    if len(sys.argv) > 1:
        log_file_path = Path(sys.argv[1])
    else:
        # 交互式输入
        log_file_input = input("请输入日志文件路径: ").strip()
        log_file_path = Path(log_file_input)
    
    if not log_file_path.exists():
        print(f"错误: 文件不存在: {log_file_path}")
        sys.exit(1)
    
    print(f"正在分析日志文件: {log_file_path}")
    print("-" * 60)
    
    result = compute_avg_nodes_per_class(log_file_path)
    
    if 'error' in result:
        print(f"错误: {result['error']}")
        sys.exit(1)
    
    print(f"日志文件: {result['log_file']}")
    print(f"总类别数: {result['total_classes']}")
    print(f"有节点数据的类别数: {result['classes_with_nodes']}")
    print(f"节点总数: {result['total_nodes']}")
    print(f"平均节点数（有数据的类）: {result['avg_nodes_per_class_with_data']:.2f}")
    print(f"平均节点数（除以总类别数）: {result['avg_nodes_per_class']:.2f}")
    print(f"最小节点数: {result['min_nodes']}")
    print(f"最大节点数: {result['max_nodes']}")
    print("-" * 60)
    print(f"\n每个类别的节点数: {result['node_counts']}")

if __name__ == '__main__':
    main()
