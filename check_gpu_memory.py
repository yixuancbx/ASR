"""
检查GPU内存使用情况
在训练前运行此脚本以检查GPU状态
"""
import torch
import gc

def check_gpu_memory():
    """检查GPU内存状态"""
    if not torch.cuda.is_available():
        print("CUDA不可用，无法检查GPU内存")
        return
    
    print("=" * 60)
    print("GPU内存检查")
    print("=" * 60)
    
    # 获取GPU信息
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")
    
    for i in range(gpu_count):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  总内存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        
        # 清理缓存
        torch.cuda.empty_cache()
        gc.collect()
        
        # 当前内存使用
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(i) / 1024**3
        
        print(f"  已分配: {allocated:.2f} GB")
        print(f"  已保留: {reserved:.2f} GB")
        print(f"  峰值分配: {max_allocated:.2f} GB")
        print(f"  可用: {(torch.cuda.get_device_properties(i).total_memory / 1024**3) - reserved:.2f} GB")
    
    print("\n" + "=" * 60)
    print("建议:")
    print("1. 如果已分配内存较大，请关闭其他使用GPU的程序")
    print("2. 如果内存不足，考虑进一步减少batch_size")
    print("3. 运行: torch.cuda.empty_cache() 清理缓存")
    print("=" * 60)

def clear_gpu_cache():
    """清理GPU缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()
        print("GPU缓存已清理")
    else:
        print("CUDA不可用")

if __name__ == '__main__':
    check_gpu_memory()
    print("\n清理GPU缓存...")
    clear_gpu_cache()
    print("\n重新检查...")
    check_gpu_memory()





