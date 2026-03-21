# 开发文档

## 线下测试环境
基于uv管理环境，请使用uv pip安装任何依赖
CUDA 12.8 H800 * 8

```bash
uv venv --python=3.14
uv pip install -r requirements.txt
```

## 线下测试数据
```bash
hf download TAAC2026/data_sample_1000 --cache-dir ./data --type dataset
```

## 线上运行环境
TODO

## 线上训练数据
TODO