"""
测试音频后端是否可用
"""
import torchaudio

def test_backend():
    """测试torchaudio后端"""
    print("测试torchaudio后端...")
    print(f"torchaudio版本: {torchaudio.__version__}")
    
    # 检查可用的后端
    try:
        import soundfile
        print("✓ soundfile已安装")
    except ImportError:
        print("✗ soundfile未安装")
    
    try:
        import sox
        print("✓ sox已安装")
    except ImportError:
        print("✗ sox未安装")
    
    # 检查torchaudio后端
    print(f"\n可用后端: {torchaudio.list_audio_backends()}")
    
    # 尝试设置后端
    try:
        torchaudio.set_audio_backend("soundfile")
        print("✓ 成功设置soundfile后端")
    except Exception as e:
        print(f"✗ 设置soundfile后端失败: {e}")
        try:
            torchaudio.set_audio_backend("sox")
            print("✓ 成功设置sox后端")
        except Exception as err:
            print(f"✗ 设置sox后端失败: {err}")

if __name__ == '__main__':
    test_backend()





