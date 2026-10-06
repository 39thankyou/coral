"""修改下方配置后，在项目根目录直接运行：

    python ExpDrivAerNet/auto_run_multi_commands.py

COMMANDS 中的命令按顺序各执行一次，当前命令退出后才启动下一条。
命令不经过 shell，不支持管道、重定向、&& 或后台运行符 &。
子进程的输出直接显示在终端；任何命令失败，脚本最终退出码为 1。
"""

import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time


def log(message):
    print("[{}] {}".format(time.strftime("%Y-%m-%d %H:%M:%S"), message), flush=True)


def stop_process(process):
    """中断时清理当前任务；POSIX 下同时通知任务派生的子进程。"""

    def send(sig):
        try:
            if os.name == "posix":
                os.killpg(process.pid, sig)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass

    send(signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        if os.name == "posix":
            send(signal.SIGKILL)
        elif process.poll() is None:
            process.kill()
        process.wait()


# ==================== 直接修改这里的配置 ====================
# 默认使用运行本脚本的 Python，也可以填写目标环境的解释器路径：
# PYTHON_EXECUTABLE = "~/miniconda3/envs/pt271cu128/bin/python"
PYTHON_EXECUTABLE = "~/miniconda3/envs/marble/bin/python"

# 默认是项目根目录，与启动本脚本时所在的目录无关。
WORKING_DIRECTORY = "~/project/coral"

# 按列表顺序执行，每条命令只执行一次。可以增加、删除或修改任意一条。
# 以 python / python3 / Python 解释器路径开头时，统一使用上面配置的解释器。
# 也支持省略 python 的 .py 脚本、-m 模块、-c 代码，以及其他可执行命令。
mode = "author"
COMMANDS = [
    f"python run_adaptor/run_pipe.py --stage all --mode {mode}",
    f"python run_adaptor/run_cylinder_flow.py --stage all --mode {mode}",
    f"python run_adaptor/run_airfoil_flow.py --stage all --mode {mode}",
    f"python run_adaptor/run_shallow_water.py --stage all --mode {mode}",
]

POLL_INTERVAL = 10.0  # 每隔多少秒输出一次当前命令仍在运行的状态
STOP_ON_ERROR = False  # True：失败后停止；False：失败后继续下一条
# ==========================================================


def prepare_commands():
    """启动前检查所有配置并解析命令，避免执行到中途才发现格式错误。"""
    executable = shutil.which(os.path.expanduser(PYTHON_EXECUTABLE))
    if executable is None:
        raise ValueError("找不到可执行的 Python 解释器：{}".format(PYTHON_EXECUTABLE))
    # 不使用 resolve()，保留虚拟环境解释器的符号链接路径。
    executable = os.path.abspath(executable)
    cwd = Path(WORKING_DIRECTORY).expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError("工作目录不存在或不是目录：{}".format(cwd))
    if not math.isfinite(POLL_INTERVAL) or POLL_INTERVAL <= 0:
        raise ValueError("POLL_INTERVAL 必须是大于 0 的有限数值")
    if not isinstance(COMMANDS, list) or not COMMANDS:
        raise ValueError("COMMANDS 必须是非空的命令字符串列表")
    commands = []
    for index, command in enumerate(COMMANDS, start=1):
        if not isinstance(command, str):
            raise ValueError("第 {} 条命令必须是字符串".format(index))
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            raise ValueError("第 {} 条命令解析失败：{}".format(index, exc)) from exc
        if not parts:
            raise ValueError("第 {} 条命令不能为空".format(index))
        if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?(?:\.exe)?", Path(parts[0]).name):
            parts = [executable, "-u"] + parts[1:]
            if len(parts) == 2:
                raise ValueError(
                    "第 {} 条命令缺少脚本路径、-m 模块或 -c 代码".format(index)
                )
        elif parts[0].endswith(".py") or parts[0] in ("-m", "-c"):
            parts = [executable, "-u"] + parts
        commands.append(parts)
    return executable, cwd, commands


def main():
    try:
        executable, cwd, commands = prepare_commands()
    except (ValueError, TypeError) as exc:
        log("配置错误：{}".format(exc))
        return 1

    env = os.environ.copy()
    env["PATH"] = os.path.dirname(executable) + os.pathsep + env.get("PATH", "")
    env["PYTHONUNBUFFERED"] = "1"
    succeeded = failed = 0
    process = None
    total = len(commands)
    log("工作目录：{}；计划顺序执行 {} 条命令".format(cwd, total))
    try:
        for index, command in enumerate(commands, start=1):
            started = time.monotonic()
            log(
                "第 {}/{} 条命令开始：{}".format(
                    index, total, " ".join(shlex.quote(part) for part in command)
                )
            )
            try:
                process = subprocess.Popen(
                    command,
                    cwd=str(cwd),
                    env=env,
                    start_new_session=(os.name == "posix"),
                )
            except OSError as exc:
                failed += 1
                log("第 {}/{} 条命令无法启动：{}".format(index, total, exc))
                if STOP_ON_ERROR:
                    break
                continue

            while True:
                try:
                    returncode = process.wait(timeout=POLL_INTERVAL)
                    break
                except subprocess.TimeoutExpired:
                    log(
                        "第 {}/{} 条命令仍在运行，已耗时 {:.1f} 秒".format(
                            index, total, time.monotonic() - started
                        )
                    )
            process = None
            if returncode == 0:
                succeeded += 1
            else:
                failed += 1
            log(
                "第 {}/{} 条命令结束，退出码 {}，耗时 {:.1f} 秒".format(
                    index, total, returncode, time.monotonic() - started
                )
            )
            if returncode != 0 and STOP_ON_ERROR:
                break
    except KeyboardInterrupt:
        log("收到中断，停止当前命令和后续命令")
        if process is not None:
            stop_process(process)
        return 130

    log(
        "运行完成：成功 {} 条，失败 {} 条，未运行 {} 条".format(
            succeeded, failed, total - succeeded - failed
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
