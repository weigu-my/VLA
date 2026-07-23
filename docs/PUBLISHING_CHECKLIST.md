# 公开发布检查清单

本文档用于将项目材料上传到个人 GitHub 前的自查。默认目标是建立一个新的、干净的公开仓库，而不是直接推送当前工作目录及其历史。

## 建议提交的内容

- `README.md`
- `docs/PROJECT_MEMORY.md`
- `docs/PUBLISHING_CHECKLIST.md`
- `openvla_plan.md`
- `pi0_plan.md`，完成账号和绝对路径脱敏后
- `openvla_resume.md`
- `results/*.csv`
- `results/*.json`
- 自己编写且已完成路径参数化的 benchmark 脚本

## 禁止提交的内容

- 公司私有仓库、patch、配置和内部 Git 历史
- 公司数据集名称、挂载路径、样本和统计量
- 模型权重、adapter、optimizer state 和 checkpoint
- W&B API key、`.netrc`、`.env`、SSH key 和云平台凭据
- 原始 W&B `output.log`、requirements 快照和 metadata
- 带人员姓名、内网域名、远程用户名或机器 IP 的日志
- rollout 视频中可能包含的内部场景或设备画面
- 未确认许可证允许再分发的第三方源码、模型和数据

## 发布前命令

先检查当前仓库来源，避免把公司仓库历史推到个人远端：

```bash
git remote -v
git status --short
git submodule status
```

扫描常见敏感信息：

```bash
rg -n -i \
  'api[_-]?key|secret|token|password|BEGIN .*PRIVATE KEY|gitlab\.|/mnt/|/home/[^/]+|ssh-rsa' \
  README.md docs results *.md *.py *.sh
```

检查大文件：

```bash
find . -type f -size +20M \
  -not -path './.git/*' \
  -printf '%s %p\n' | sort -nr
```

只暂存经过确认的文件：

```bash
git add README.md .gitignore docs results
git diff --cached --stat
git diff --cached
```

不要在当前目录直接执行 `git add .`。这里包含多个第三方仓库、模型、数据、虚拟环境和私有源码目录。

## 推荐发布方式

1. 在 GitHub 新建空仓库。
2. 在另一个空目录初始化 Git。
3. 只复制上方“建议提交”的个人材料。
4. 再运行敏感信息和大文件扫描。
5. 添加许可证，许可证只覆盖自己编写的内容。
6. 提交后在 GitHub 网页再次检查文件列表和历史。

## 凭据处理

W&B 登录成功通常意味着 key 保存在用户主目录的 `.netrc` 中，它不会因为复制几份 Markdown 自动进入仓库。但如果凭据曾经进入 Git 历史，仅从最新提交删除文件是不够的，需要重写历史并立即轮换 key。
