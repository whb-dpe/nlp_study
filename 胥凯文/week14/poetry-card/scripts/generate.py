#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诗词卡片生成脚本 - 优化版
用法: python generate.py <诗人名> [输出目录]
"""
import json, sys
from pathlib import Path

def load_poet_data(skill_dir, name):
    p = skill_dir / "data" / f"{name}.json"
    if not p.exists():
        print(f"错误: 未找到 {name} 的数据文件\n路径: {p}")
        sys.exit(1)
    with open(p, encoding='utf-8') as f:
        return json.load(f)

def render_tags(tags):
    return ''.join(f'<span class="tag">{t}</span>' for t in tags)

def render_classic_lines(lines):
    parts = []
    for item in lines:
        if isinstance(item, str):
            text, source, expl = item, '', ''
        else:
            text = item.get('text', '')
            source = item.get('source', '')
            expl = item.get('explanation', '')
        s = f'<div class="line-source">{source}</div>' if source else ''
        e = f'<div class="line-explanation">{expl}</div>' if expl else ''
        parts.append(f'<div class="line-item"><div class="line-text">{text}</div>{s}{e}</div>')
    return ''.join(parts)

def render_poems(poems):
    parts = []
    for p in poems:
        trans = p.get('translation', p.get('annotation', ''))
        appr = p.get('appreciation', '')
        t = f'<div class="poem-annotation"><h4>📝 译文</h4><p>{trans}</p></div>' if trans else ''
        a = f'<div class="poem-appreciation"><h4>🎨 赏析</h4><p>{appr}</p></div>' if appr else ''
        parts.append(f'<div class="poem"><div class="poem-title">{p["title"]}</div><div class="poem-content">{p["content"]}</div>{t}{a}</div>')
    return ''.join(parts)

def generate_html(skill_dir, name, out_dir=None):
    with open(skill_dir / "assets" / "template.html", encoding='utf-8') as f:
        tpl = f.read()
    data = load_poet_data(skill_dir, name)
    career = data.get('career', '')
    career_html = f'<div class="career"><div class="career-title">生平经历</div><div class="career-content">{career}</div></div>' if career else ''
    repl = {
        '{{POET_NAME}}': data['name'],
        '{{POET_TITLE}}': data.get('title', ''),
        '{{DYNASTY}}': data.get('dynasty', ''),
        '{{BIO}}': data.get('bio', ''),
        '{{CAREER}}': career_html,
        '{{STYLE_TAGS}}': render_tags(data.get('style_tags', [])),
        '{{CLASSIC_LINES}}': render_classic_lines(data.get('classic_lines', data.get('famous_lines', []))),
        '{{POEMS}}': render_poems(data.get('poems', [])),
    }
    for k, v in repl.items():
        tpl = tpl.replace(k, v)
    out = Path(out_dir) if out_dir else Path.cwd()
    out_file = out / f"诗词卡-{name}.html"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(tpl)
    print(f"✅ 诗词卡片已生成: {out_file}")
    return out_file

def main():
    if len(sys.argv) < 2:
        print("用法: python generate.py <诗人名> [输出目录]")
        sys.exit(1)
    skill_dir = Path(__file__).parent.parent
    generate_html(skill_dir, sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)

if __name__ == '__main__':
    main()
