#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""分析论文引用情况"""

import re
import glob

# 读取所有tex文件中的引用
cited = set()
for tex_file in glob.glob('pages/*.tex'):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
        # 提取所有\cite{}中的内容
        matches = re.findall(r'\\cite\{([^}]+)\}', content)
        for match in matches:
            # 分割多个引用
            refs = [r.strip() for r in match.split(',')]
            cited.update(refs)

# 读取bib文件中的所有条目
all_refs = {}
with open('ref/references.bib', 'r', encoding='utf-8') as f:
    content = f.read()
    # 提取每个条目的key和类型
    entries = re.findall(r'@(\w+)\{([^,]+),', content)
    for entry_type, key in entries:
        all_refs[key] = entry_type

# 找出未引用的文献
uncited = sorted(set(all_refs.keys()) - cited)

print(f"总文献数: {len(all_refs)}")
print(f"已引用: {len(cited)}")
print(f"未引用: {len(uncited)}")
print(f"\n已引用的文献 ({len(cited)}篇):")
for ref in sorted(cited):
    if ref in all_refs:
        print(f"  [CITED] {ref} ({all_refs[ref]})")
    else:
        print(f"  [ERROR] {ref} (NOT FOUND IN BIB!)")

print(f"\n未引用的文献 ({len(uncited)}篇):")
for ref in uncited:
    print(f"  [UNCITED] {ref} ({all_refs[ref]})")
