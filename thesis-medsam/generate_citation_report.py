#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成引用增加报告"""

import re
import glob

# 读取所有tex文件中的引用
cited = set()
for tex_file in glob.glob('pages/*.tex'):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
        matches = re.findall(r'\\cite\{([^}]+)\}', content)
        for match in matches:
            refs = [r.strip() for r in match.split(',')]
            cited.update(refs)

# 读取bib文件中的所有条目
all_refs = {}
with open('ref/references.bib', 'r', encoding='utf-8') as f:
    content = f.read()
    entries = re.findall(r'@(\w+)\{([^,]+),', content)
    for entry_type, key in entries:
        all_refs[key] = entry_type

# 新增的12篇文献
new_refs = [
    'cicek20163dunet',
    'roy2023mednext',
    'zheng2021setr',
    'xie2021segformer',
    'ravi2024sam2',
    'wang2023sammed3d',
    'sudre2017generalised',
    'salehi2017tversky',
    'berman2018lovasz',
    'yeung2022unified',
    'chen2022adaptformer',
    'jia2022vpt'
]

print("=" * 60)
print("参考文献增加报告")
print("=" * 60)
print(f"\n总文献数: {len(all_refs)}")
print(f"已引用: {len(cited)}")
print(f"未引用: {len(all_refs) - len(cited)}")

print(f"\n新增的12篇引用:")
for i, ref in enumerate(new_refs, 1):
    status = "[OK]" if ref in cited else "[MISSING]"
    ref_type = all_refs.get(ref, "unknown")
    print(f"  {i:2d}. {status} {ref} ({ref_type})")

print(f"\n引用分布:")
for tex_file in sorted(glob.glob('pages/*.tex')):
    with open(tex_file, 'r', encoding='utf-8') as f:
        content = f.read()
        file_refs = set()
        matches = re.findall(r'\\cite\{([^}]+)\}', content)
        for match in matches:
            refs = [r.strip() for r in match.split(',')]
            file_refs.update(refs)

        # 统计新增引用
        new_in_file = [r for r in new_refs if r in file_refs]
        if new_in_file:
            print(f"  {tex_file}: {len(file_refs)} total, {len(new_in_file)} new")
            for ref in new_in_file:
                print(f"    + {ref}")

print("\n" + "=" * 60)
