#!/usr/bin/env python3
"""
Test script to verify PaddlePaddle installation
"""

def test_paddle():
    try:
        import paddle
        print(f"PaddlePaddle version: {paddle.__version__}")
        print(f"Available devices: {paddle.get_device()}")
        
        # Test basic functionality
        x = paddle.to_tensor([1.0, 2.0, 3.0])
        y = x * 2
        print(f"Basic tensor operation result: {y}")
        print("PaddlePaddle is working correctly!")
        return True
    except Exception as e:
        print(f"Error importing or using PaddlePaddle: {e}")
        return False

if __name__ == "__main__":
    test_paddle()