#!/usr/bin/env python
# -*- coding: utf-8 -*-

files = [
    'pages/chapter2.tex',
    'pages/chapter3.tex',
    'pages/chapter4.tex',
    'pages/chapter5.tex',
    'pages/chapter6.tex'
]

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 替换所有ASCII双引号为中文引号
    # 开引号和闭引号都替换为中文引号
    new_content = content.replace('"', '"').replace('"', '"')

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"已处理: {filepath}")

print("所有文件处理完成")
