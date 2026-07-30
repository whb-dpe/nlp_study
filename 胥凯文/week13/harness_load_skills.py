"""
DeepSeek Function Calling · 【严格延迟读取版】按需渐进式 Skill Harness
================================================================
启动阶段：
  - 只读 SKILL.md 的 YAML frontmatter（name + 简短 description），整份文件不全读
  - Function Tool 的 description 只包含简短功能描述，不附加触发词/流程
  - 不碰 scripts/、不碰 data/
触发阶段（DeepSeek 发起 tool_call 之后）：
  - 才从磁盘读取 SKILL.md 全文（含执行步骤、数据格式、注意事项）送入后续 AI 上下文
  - 才扫描 scripts / data 文件、加载现有样本、校验完整性、注册 harness 钩子并执行
"""
import os
import re
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Optional


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY","")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"

SKILLS_DIR = Path(__file__).parent / "skills"


def safe_decode_bytes(b: bytes) -> str:
    if b is None:
        return ""
    for enc in ("utf-8", "gbk", "mbcs", "cp936", "latin-1"):
        try:
            return b.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return b.decode("utf-8", errors="replace")


def log_stage(stage_num, total_stages, title, detail=""):
    bar = f"[{'=' * stage_num}{' ' * (total_stages - stage_num)}]"
    print(f"\n  {'─'*68}")
    print(f"  ▶ 阶段 {stage_num}/{total_stages} {bar}  {title}")
    if detail:
        print(f"      {detail}")
    print(f"  {'─'*68}")


def log_step(msg, level=0):
    indent = "      " + "  " * level
    print(f"{indent}• {msg}")


def _requests():
    try:
        import requests
        return requests
    except ImportError:
        print("  ⚠️  请先 pip install requests")
        return None


def deepseek_chat(messages, tools=None, temperature=0.7):
    reqs = _requests()
    if reqs is None:
        return None
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "messages": messages,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        t0 = time.time()
        resp = reqs.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers, json=payload, timeout=120,
        )
        if resp.status_code != 200:
            print(f"  ⚠️  DeepSeek {resp.status_code}: {resp.text[:300]}")
            return None
        data = resp.json()
        data["_elapsed"] = round(time.time() - t0, 2)
        return data
    except Exception as e:
        print(f"  ⚠️  API 异常: {type(e).__name__}: {e}")
        return None


def deepseek_stream_text(messages, tools=None, temperature=0.7):
    reqs = _requests()
    if reqs is None:
        return None
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": DEEPSEEK_MODEL,
        "temperature": temperature,
        "messages": messages,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    try:
        collected = []
        print("  🤖 DeepSeek: ", end="", flush=True)
        with reqs.post(
            f"{DEEPSEEK_BASE_URL}/chat/completions",
            headers=headers, json=payload, timeout=120, stream=True,
        ) as resp:
            if resp.status_code != 200:
                print(f"\n  ⚠️  {resp.status_code}: {resp.text[:200]}")
                return None
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    d = json.loads(chunk)
                    c = d["choices"][0]["delta"]
                    if "tool_calls" in c and c["tool_calls"]:
                        print(" …(检测到 Tool Call，转为非流式处理)")
                        return "__TOOL_CALL__"
                    tok = c.get("content", "")
                    if tok:
                        collected.append(tok)
                        print(tok, end="", flush=True)
                except Exception:
                    pass
        print()
        return "".join(collected) if collected else ""
    except Exception as e:
        print(f"\n  ⚠️  流式异常: {type(e).__name__}: {e}")
        return None


def read_frontmatter_only(md_path: Path):
    """
    【延迟读取关键实现】
    只逐行读取 SKILL.md 直到 YAML frontmatter 闭合，提取 name / description。
    文件指针不会继续向后读取正文。description 尽量截断为简短版本。
    """
    if not md_path.exists():
        return None
    name = ""
    description = ""
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            first = f.readline()
            if not first or first.strip() != "---":
                return None
            in_desc = False
            desc_lines = []
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                stripped = line.strip()
                # frontmatter 结束，立刻停止读取，不再触碰正文
                if stripped == "---":
                    break
                if in_desc:
                    if line.startswith(" ") or stripped == "":
                        if stripped:
                            desc_lines.append(stripped)
                    else:
                        f.seek(pos)  # 回退，后续触发时再读
                        break
                elif stripped.startswith("name:"):
                    name = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                elif stripped.startswith("description:"):
                    right = stripped.split(":", 1)[1].strip()
                    if right.startswith(">-") or right.startswith(">") or right.startswith("|-") or right.startswith("|"):
                        in_desc = True
                    else:
                        description = right.strip().strip('"').strip("'")
    except Exception:
        return None
    if not name:
        return None
    if in_desc:
        description = " ".join(desc_lines).strip()
    if len(description) > 160:
        description = description[:157].rstrip() + "..."
    return {"name": name, "description": description}


def build_function_tool_strict(name: str, short_desc: str) -> dict:
    """【严格版】Tool Schema description 只保留功能简述，不附触发词/步骤。"""
    return {
        "type": "function",
        "function": {
            "name": f"skill__{name.replace('-', '_')}",
            "description": short_desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "word": {
                        "type": "string",
                        "description": "目标英语单词（小写），如: crazy, resilient, thrill, meticulous",
                    },
                    "output_filename": {
                        "type": "string",
                        "description": "可选，自定义输出 HTML 文件名/路径，默认为 <word>.html",
                    },
                },
                "required": ["word"],
                "additionalProperties": False,
            },
        },
    }


def startup_lazy_discover():
    """
    启动时严格只做：
      1. 枚举 skills/* 目录
      2. 只读每个 SKILL.md 的 YAML frontmatter → name + 短 description
      3. 构建严格简短的 Function Tool Schema
    绝不读取 SKILL.md 正文，绝不扫描 scripts / data。
    """
    log_stage(1, 1, "启动 · 严格延迟模式：仅读取 SKILL.md frontmatter (name + 短描述)",
              "SKILL.md 正文 / 脚本文件 / 数据文件 → 均在 Tool Call 触发时才加载")
    registry = {}
    if not SKILLS_DIR.exists():
        log_step(f"❌ Skills 目录不存在: {SKILLS_DIR}")
        return registry, []
    candidates = [p for p in sorted(SKILLS_DIR.iterdir())
                  if p.is_dir() and (p / "SKILL.md").exists()]
    log_step(f"发现 Skill 候选目录: {len(candidates)} 个")
    tools = []
    for d in candidates:
        md_path = d / "SKILL.md"
        meta_stub = read_frontmatter_only(md_path)
        if meta_stub is None:
            log_step(f"⚠️  跳过无法读取 frontmatter 的 Skill: {d.name}", level=1)
            continue
        name = meta_stub["name"] or d.name
        short_desc = meta_stub["description"]
        registry[name] = {
            "name": name,
            "description": short_desc,
            "dir": str(d),
            "_md_path": str(md_path),
            "_raw_md_loaded": False,
            "loaded": False,
            "harness_ready": False,
        }
        tool_def = build_function_tool_strict(name, short_desc)
        tools.append(tool_def)
        log_step(f"✅ 已注册 Skill 签名 → 「{name}」", level=1)
        log_step(f"   Function Name : {tool_def['function']['name']}", level=2)
        log_step(f"   Description   : {short_desc[:100]}{'…' if len(short_desc)>100 else ''}", level=2)
        log_step(f"   ⚙️  触发时才加载: SKILL.md正文 / scripts/ / data/", level=2)
    log_step(f"启动完成：注册 Skill {len(registry)} 个，Function Tools {len(tools)} 个（严格延迟模式）")
    return registry, tools


def full_load_skill_md(meta: dict) -> Optional[str]:
    """首次触发 Skill 时才从磁盘读取整份 SKILL.md。"""
    if meta.get("_raw_md"):
        return meta["_raw_md"]
    md_path = Path(meta["_md_path"])
    if not md_path.exists():
        return None
    try:
        text = md_path.read_text(encoding="utf-8")
    except Exception:
        return None
    meta["_raw_md"] = text
    meta["_raw_md_loaded"] = True
    triggers = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("- "):
            quoted = re.findall(r'"([^"]+)"', s)
            if quoted:
                triggers.extend(quoted)
            elif any(k in s for k in ("闪卡", "单词卡", "flash card", "flashcard")):
                c = s.lstrip("- ").strip().strip('"').strip("'")
                if c:
                    triggers.append(c)
    meta["triggers"] = triggers
    return text


def scan_skill_files(skill_dir: Path) -> dict:
    files = {"md": None, "scripts": [], "data": [], "others": []}
    for root, _, filenames in os.walk(skill_dir):
        rel_root = Path(root).relative_to(skill_dir)
        for fn in sorted(filenames):
            rel = (rel_root / fn).as_posix()
            if fn == "SKILL.md":
                files["md"] = rel
            elif fn.endswith(".py"):
                files["scripts"].append(rel)
            elif fn.endswith(".json"):
                files["data"].append(rel)
            else:
                files["others"].append(rel)
    return files


def load_data_samples(skill_dir: Path, data_rel: list) -> list:
    loaded = []
    for rel in data_rel:
        fp = skill_dir / rel
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            size = fp.stat().st_size
            preview = data.get("word", list(data.keys())[:3])
            loaded.append({"path": rel, "size": size, "preview": str(preview)})
            log_step(f"读入 {rel} ({size} B) → word={preview}", level=2)
        except Exception as e:
            log_step(f"⚠️  读取 {rel} 失败: {e}", level=2)
    return loaded


def ai_enrich_metadata(raw_md: str) -> dict:
    sys_prompt = (
        "你是 Skill 元数据提取器。根据 SKILL.md 内容仅输出 JSON："
        '{"steps":["步骤名",...],"data_schema":{"字段名":"说明"},'
        '"expected_artifacts":["路径模式"],"input_rule":"用户输入需要包含什么信息"}'
    )
    resp = deepseek_chat(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": raw_md}],
        temperature=0.1,
    )
    if not resp:
        return {}
    raw = resp["choices"][0]["message"]["content"]
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        return json.loads(raw[s:e]) if 0 <= s < e else {}
    except Exception:
        return {}


def ensure_data_json(skill_dir: Path, word: str, raw_md: str) -> Optional[Path]:
    data_dir = skill_dir / "data"
    data_dir.mkdir(exist_ok=True)
    target = data_dir / f"{word}.json"
    if target.exists():
        log_step(f"数据文件已存在: {target.name}，直接复用", level=2)
        return target
    log_step(f"未找到 {word}.json，调用 DeepSeek 按 SKILL.md 规范生成 ...", level=2)
    sys_prompt = (
        "你是英语单词学习数据编写者。严格按 SKILL.md 格式，只输出 JSON："
        "word / phonetic / pos / definition / "
        "examples(恰好3条，每条 {en, zh}) / synonyms(4-6 个)。"
    )
    resp = deepseek_chat(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": f"目标单词：{word}\n\nSKILL.md 参考：\n{raw_md}"}],
        temperature=0.3,
    )
    if not resp:
        log_step("⚠️  API 无响应，生成失败", level=2)
        return None
    raw = resp["choices"][0]["message"]["content"]
    try:
        s, e = raw.find("{"), raw.rfind("}") + 1
        obj = json.loads(raw[s:e]) if 0 <= s < e else None
        if not obj:
            raise ValueError("JSON 非法")
        target.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        log_step(f"已写入 {target.name} ({target.stat().st_size} B)", level=2)
        return target
    except Exception as ex:
        log_step(f"⚠️  写入 JSON 失败: {ex}", level=2)
        return None


def run_entry_script(entry: str, data_path: Path, extra_args=None):
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    argv = [sys.executable, entry, str(data_path)]
    if extra_args:
        argv.extend(extra_args)
    try:
        result = subprocess.run(
            argv, capture_output=True,
            cwd=str(Path(__file__).parent),
            timeout=60, env=env,
        )
        return (result.returncode,
                safe_decode_bytes(result.stdout),
                safe_decode_bytes(result.stderr))
    except Exception as e:
        return (-1, "", f"{type(e).__name__}: {e}")


def progressive_load_and_run(registry: dict, skill_name: str, tool_args: dict) -> str:
    """
    DeepSeek 发起 tool_call 之后才真正执行：
      0) 从磁盘完整读取 SKILL.md 全文（此前只读了 frontmatter）★关键延迟点
      1) AI 基于全文补全元数据
      2) 扫描脚本/数据文件
      3) 载入已有数据样本
      4) 校验完整性
      5) 注册 harness 钩子
      → 准备数据 → 执行脚本 → 返回 tool_result
    """
    meta = registry.get(skill_name)
    if meta is None:
        return f"[错误] 未知 Skill: {skill_name}"
    skill_dir = Path(meta["dir"])
    word = str(tool_args.get("word", "")).strip().lower()
    extra_args = []
    custom_output = tool_args.get("output_filename")
    if custom_output:
        extra_args = ["-o", str(custom_output)]

    print(f"\n  ⚡ [DeepSeek Function Call] 触发 Skill: {skill_name}")
    print(f"     Tool Call 参数 → word={word!r}  output_filename={custom_output!r}")

    if not meta.get("harness_ready"):
        total = 5

        # ★ 阶段 0：此刻才读取 SKILL.md 全文
        log_stage(0, total, f"加载 Skill「{skill_name}」· 【首次】从磁盘读取 SKILL.md 全文",
                  "会话开始前仅读取 frontmatter(name+描述)，此刻才把正文载入 AI 可用上下文")
        raw_md = full_load_skill_md(meta)
        if raw_md:
            lines = raw_md.splitlines()
            log_step(f"读取 SKILL.md 成功 → {len(raw_md.encode('utf-8'))} B / {len(lines)} 行")
            log_step(f"识别触发话术 {len(meta.get('triggers',[]))} 条（用于后续校验参考）", level=1)
        else:
            log_step("⚠️  SKILL.md 全文读取失败，后续步骤继续但可能效果下降")

        # 1/5
        log_stage(1, total, f"加载 Skill「{skill_name}」· AI 基于全文补全元数据",
                  "steps / data_schema / artifacts / input_rule")
        enriched = ai_enrich_metadata(raw_md or "")
        for k, v in enriched.items():
            meta.setdefault(k, v)
        log_step(f"执行步骤  : {meta.get('steps', [])}")
        log_step(f"数据 schema: {list(meta.get('data_schema', {}).keys())}")
        log_step(f"输入规则  : {meta.get('input_rule', '(默认规则)')}")

        # 2/5
        log_stage(2, total, f"加载 Skill「{skill_name}」· 扫描内部文件 scripts/ data/")
        files = scan_skill_files(skill_dir)
        meta["files"] = files
        log_step(f"SKILL.md : {files['md']}")
        log_step(f"脚本数   : {len(files['scripts'])} → {', '.join(files['scripts']) or '(无)'}")
        log_step(f"数据数   : {len(files['data'])} → {', '.join(files['data']) or '(无)'}")
        if files["others"]:
            log_step(f"其他文件 : {', '.join(files['others'])}")

        # 3/5
        log_stage(3, total, f"加载 Skill「{skill_name}」· 加载现有数据样本 JSON")
        samples = load_data_samples(skill_dir, files["data"])
        meta["data_samples"] = samples
        log_step(f"已载入数据样本: {len(samples)} 个")

        # 4/5
        log_stage(4, total, f"加载 Skill「{skill_name}」· 校验完整性 & 一致性")
        issues = []
        if not files["scripts"]:
            issues.append("缺少 .py 执行脚本")
        else:
            sp = skill_dir / files["scripts"][0]
            try:
                c = sp.read_text(encoding="utf-8")
                if "argparse" not in c and "sys.argv" not in c:
                    issues.append("脚本未提供 CLI (无 argparse/sys.argv)")
            except Exception as e:
                issues.append(f"读取脚本失败: {e}")
        if not meta.get("triggers") or len(meta["triggers"]) < 1:
            issues.append("SKILL.md 中未找到触发话术示例")
        if issues:
            for i in issues:
                log_step(f"⚠️  {i}")
        else:
            log_step("✅ 校验通过")
        meta["issues"] = issues
        meta["valid"] = len(issues) == 0

        # 5/5
        log_stage(5, total, f"加载 Skill「{skill_name}」· 注册 Harness 钩子 → 可执行",
                  "渐进式加载完成，后续命中直接执行，不再重复加载")
        entry_script = str(skill_dir / files["scripts"][0]) if files["scripts"] else None
        meta["harness"] = {
            "entry_script": entry_script,
            "data_dir": str(skill_dir / "data"),
            "status": "READY" if meta["valid"] else "READY_WITH_WARNINGS",
            "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        meta["loaded"] = True
        meta["harness_ready"] = True
        registry[skill_name] = meta
        log_step(f"✅ Skill「{skill_name}」→ {meta['harness']['status']}")
        log_step(f"   入口脚本 : {entry_script}")
        log_step(f"   数据目录 : {meta['harness']['data_dir']}")
    else:
        print(f"  ⚡ [命中缓存] Skill「{skill_name}」已加载过，跳过渐进加载，直接执行")

    # ── 执行 Skill ──
    print(f"\n  {'═'*68}")
    print(f"  🚀 执行 Skill：{skill_name}  word={word!r}")
    print(f"  {'═'*68}")

    if not word:
        return "[失败] 缺少必需参数 word（目标英语单词），请在 function call 参数中提供。"
    entry = meta["harness"]["entry_script"]
    if not entry or not Path(entry).exists():
        return "[失败] 入口脚本不存在，无法执行 Skill。"

    log_step("[1/3] 准备数据文件 <word>.json（无则 AI 生成）")
    raw_md = meta.get("_raw_md") or full_load_skill_md(meta) or ""
    data_path = ensure_data_json(skill_dir, word, raw_md)
    if data_path is None:
        return f"[失败] 无法准备 {word}.json 数据文件"

    log_step(f"[2/3] 调用入口脚本 → python {Path(entry).name} {data_path.name}"
             + (f" {' '.join(extra_args)}" if extra_args else ""))
    code, out, err = run_entry_script(entry, data_path, extra_args)
    if out.strip():
        for ln in out.strip().splitlines():
            log_step(f"[stdout] {ln}", level=1)
    if err.strip():
        for ln in err.strip().splitlines():
            log_step(f"[stderr] {ln}", level=1)
    if code != 0:
        return f"[失败] 脚本退出码 {code}。stderr 摘要: {err[:500]}"

    log_step("[3/3] 定位输出产物 HTML")
    artifact = None
    if custom_output:
        cop = Path(custom_output)
        if cop.exists():
            artifact = cop
        elif Path(Path(__file__).parent, custom_output).exists():
            artifact = Path(Path(__file__).parent, custom_output)
    default_out = Path(__file__).parent / f"{data_path.stem}.html"
    if artifact is None and default_out.exists():
        artifact = default_out

    if artifact is None:
        return "[成功] 脚本执行成功，但未能定位最终 HTML 产物文件。"
    kb = artifact.stat().st_size / 1024
    log_step(f"✅ 产物已生成 → {artifact.resolve()}  ({kb:.2f} KB)")
    return (
        f"[成功] 单词闪卡已生成。\n"
        f"  • 目标单词 : {word}\n"
        f"  • 绝对路径 : {artifact.resolve()}\n"
        f"  • 文件大小 : {kb:.2f} KB\n"
        f"  • 下一步   : 请用户在浏览器中打开该 HTML 即可查看精美的单词闪卡页面。"
    )


def function_name_to_skill_name(fn_name: str, registry: dict) -> Optional[str]:
    if fn_name.startswith("skill__"):
        cand = fn_name[len("skill__"):].replace("_", "-")
        if cand in registry:
            return cand
    for k in registry:
        if fn_name == f"skill__{k.replace('-', '_')}":
            return k
    return None


def main_loop():
    banner = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║  Skill Harness · DeepSeek FC · 【严格延迟读取】按需渐进式加载            ║
╠══════════════════════════════════════════════════════════════════════════╣
║  模型  : {DEEPSEEK_MODEL:<60}║
║  启动  : 仅读取每个 SKILL.md 的 YAML frontmatter                         ║
║          (name + 简短 description，≤160 字)，截断后续读取               ║
║          不触碰 SKILL.md 正文 / scripts/ / data/                        ║
║  触发  : DeepSeek 发起 tool_call 后                                      ║
║          → Harness 才从磁盘读取 SKILL.md 全文 (打印加载过程)             ║
║          → 扫描 scripts/data、校验、注册 harness、执行脚本               ║
║          → tool_result 回传 → DeepSeek 组织最终回答                     ║
║  命令  : /skills  已注册签名 /loaded 已加载 /tools Tool Schema /quit    ║
╚══════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)
    registry, tools = startup_lazy_discover()
    if not registry:
        print("  ⚠️  没有可用 Skill，退出。")
        return
    sys_ctx = (
        "你是一位带 Skill 系统的智能助手。系统给你注册了若干 function tools，"
        "当用户需求匹配对应 Skill 能力时（如生成单词闪卡、学习卡），请发起 "
        "function call 调用对应的 skill；不需要在对话中先向用户复述整个工具列表。"
        "工具执行的返回结果会以 tool role 的消息回传给你，请你基于其内容向用户"
        "生成清晰的最终回答，并告知在浏览器打开 HTML 文件。不匹配 Skill 时自然回答即可。"
    )
    messages = [{"role": "system", "content": sys_ctx}]

    while True:
        try:
            user_text = input("\n  👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 再见！")
            break
        if not user_text:
            continue
        if user_text.lower() in ("/quit", "exit", "q"):
            print("  👋 再见！")
            break
        if user_text == "/skills":
            print(f"\n  📖 已注册 Skill 签名 ({len(registry)}，均为『待触发加载』状态):")
            for n, m in registry.items():
                tag = "✅已加载" if m.get("harness_ready") else "⏳仅name+描述"
                fn = f"skill__{n.replace('-','_')}"
                loaded_tag = f", SKILL.md正文={'已载入' if m.get('_raw_md_loaded') else '未载入'}"
                print(f"    • {n:<15} [{tag}{loaded_tag}]  fn={fn}")
            continue
        if user_text == "/loaded":
            L = [(n, m) for n, m in registry.items() if m.get("harness_ready")]
            print(f"\n  🚀 已完成渐进加载的 Skill: {len(L)}/{len(registry)}")
            for n, m in L:
                h = m.get("harness", {})
                print(f"    • {n}  状态={h.get('status')}  入口={h.get('entry_script','')}")
            continue
        if user_text == "/tools":
            print(f"\n  🛠️  当前 Function Tools（仅 name + 短描述）:")
            print(json.dumps(tools, ensure_ascii=False, indent=4))
            continue

        messages.append({"role": "user", "content": user_text})
        reply = deepseek_stream_text(messages, tools=tools, temperature=0.7)
        if reply is None:
            print("  ⚠️  DeepSeek 无响应")
            messages.pop()
            continue
        if reply != "__TOOL_CALL__":
            messages.append({"role": "assistant", "content": reply})
            continue

        max_turns, final_text = 5, None
        for _ in range(max_turns):
            resp = deepseek_chat(messages, tools=tools, temperature=0.7)
            if resp is None:
                break
            msg = resp["choices"][0]["message"]
            elapsed = resp.get("_elapsed", 0)
            if "tool_calls" not in msg or not msg["tool_calls"]:
                final_text = msg.get("content") or ""
                if final_text:
                    print(f"  🤖 DeepSeek: {final_text}")
                    messages.append({"role": "assistant", "content": final_text})
                break
            tc_list = msg["tool_calls"]
            print(f"  🛠️  DeepSeek 请求 Function Call × {len(tc_list)}  (耗时 {elapsed}s)")
            assistant_msg = {"role": "assistant", "content": msg.get("content"), "tool_calls": []}
            for tc in tc_list:
                assistant_msg["tool_calls"].append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]},
                })
            messages.append(assistant_msg)
            for tc in tc_list:
                tc_id = tc["id"]
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except Exception:
                    fn_args = {}
                skill_name = function_name_to_skill_name(fn_name, registry)
                if skill_name is None:
                    tool_result = f"[错误] 未知的 function name: {fn_name}"
                else:
                    tool_result = progressive_load_and_run(registry, skill_name, fn_args)
                print(f"  ↩️  回传 tool_result 给 DeepSeek  (call_id={tc_id})")
                messages.append({"role": "tool", "tool_call_id": tc_id, "content": tool_result})

        if final_text is None:
            last = deepseek_chat(messages, tools=tools, temperature=0.7)
            if last:
                fm = last["choices"][0]["message"]
                txt = fm.get("content") or ""
                if txt:
                    print(f"  🤖 DeepSeek: {txt}")
                    messages.append({"role": "assistant", "content": txt})

        if len(messages) > 40:
            messages = [messages[0]] + messages[-38:]


if __name__ == "__main__":
    main_loop()