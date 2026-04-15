from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLIND_PDF = ROOT / "基于平衡损失与局部适配的医学图像分割.pdf"
FORMAL_PDF = ROOT / "基于平衡损失与局部适配的医学图像分割_正式版.pdf"
TMP_DIR = ROOT / "tmp" / "variant_verify"


def extract_text(pdf_path: Path) -> str:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    stem = "blind" if pdf_path == BLIND_PDF else "formal"
    text_path = TMP_DIR / f"{stem}.txt"
    subprocess.run(
        ["pdftotext", str(pdf_path), str(text_path)],
        check=True,
        cwd=ROOT,
    )
    return text_path.read_text(encoding="utf-8", errors="ignore")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    require(BLIND_PDF.exists(), f"缺少盲审版 PDF: {BLIND_PDF.name}")
    require(FORMAL_PDF.exists(), f"缺少正式版 PDF: {FORMAL_PDF.name}")

    blind_text = extract_text(BLIND_PDF)
    formal_text = extract_text(FORMAL_PDF)
    blind_compact = "".join(blind_text.split())
    formal_compact = "".join(formal_text.split())

    require("学生类型：专硕" in blind_compact, "盲审版缺少“学生类型：专硕”提示页")
    require("学生类型：专硕" in formal_compact, "正式版缺少“学生类型：专硕”提示页")
    require(
        "1绪论11.1研究背景与意义" in blind_compact,
        "盲审版缺少目录",
    )
    require(
        "1绪论11.1研究背景与意义" in formal_compact,
        "正式版缺少目录",
    )

    for forbidden in (
        "2024282210310",
        "周贤玮",
        "陈刚",
        "致谢",
        "国家网络安全学院",
        "个人简历",
        "研究生姓名",
        "校内导师姓名",
        "学号",
        "2001 年 5 月 28 日出生于湖南省湘西土家族苗族自治州吉首市。",
        "2019 年 9 月考入南方科技大学计算机科学与技术专业，2024 年 6 月本科毕业并获得工学学士学位。",
        "2024 年 9 月考入武汉大学国家网络安全学院攻读硕士学位至今。",
    ):
        require(forbidden not in blind_compact, f"盲审版仍包含不应出现的信息: {forbidden}")

    for expected in ("陈刚", "致谢"):
        require(expected in formal_compact, f"正式版缺少应保留的信息: {expected}")

    require(
        ("周贤玮" in formal_compact)
        or ("研究生姓名：周" in formal_compact and "贤玮号" in formal_compact),
        "正式版缺少作者信息",
    )
    require(
        "2001年5月28日出生于湖南省湘西土家族苗族自治州吉首市。" in formal_compact,
        "正式版缺少个人简历内容",
    )

    print("论文正式版/盲审版校验通过")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as exc:
        print(f"校验失败: {exc}", file=sys.stderr)
        raise SystemExit(1)
