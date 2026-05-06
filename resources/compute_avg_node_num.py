import re
import sys
from pathlib import Path

def extract_node_counts_from_log(log_file_path):
    """Handle extract node counts from log."""
    node_counts = []
    total_classes = 0
    
    encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1', 'cp1252']
    file_content = None
    
    for encoding in encodings:
        try:
            with open(log_file_path, 'r', encoding=encoding, errors='ignore') as f:
                file_content = f.readlines()
            break
        except (UnicodeDecodeError, LookupError):
            continue
    
    if file_content is None:
        with open(log_file_path, 'rb') as f:
            file_content = f.read().decode('utf-8', errors='ignore').splitlines()
    
    for line in file_content:
        match = re.search(r'soinn_refined=(\d+)', line)
        if match:
            node_count = int(match.group(1))
            node_counts.append(node_count)
        
        class_match = re.search(r'total classes:\s*(\d+)', line)
        if class_match:
            total_classes = max(total_classes, int(class_match.group(1)))
    
    if total_classes == 0:
        total_classes = len(node_counts)
    
    return node_counts, total_classes

def compute_avg_nodes_per_class(log_file_path):
    """Handle compute avg nodes per class."""
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
        'avg_nodes_per_class_with_data': avg_nodes,
        'avg_nodes_per_class': avg_nodes_per_class,
        'min_nodes': min(node_counts),
        'max_nodes': max(node_counts),
        'node_counts': node_counts
    }

def main():
    """Handle main."""
    if len(sys.argv) > 1:
        log_file_path = Path(sys.argv[1])
    else:
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

def summarize_hc_soinn_from_classifier(hc_soinn):
    """Handle summarize hc soinn from classifier."""
    from utils.hc_soinn_node_stats import summarize_hc_soinn_classifier
    return summarize_hc_soinn_classifier(hc_soinn)


if __name__ == '__main__':
    main()
