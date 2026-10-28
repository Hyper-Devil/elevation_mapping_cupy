# RELLIS emcupy 环境

这些脚本是工作区激活、编译和验收的唯一维护入口。源码须位于
`WORKSPACE/src/elevation_mapping_cupy`；容器统一将 WORKSPACE 挂载为
`/home/root/catkin_emcp`。优先复用匹配的现有容器，不覆盖已有测试输出。

## 使用已验收的镜像

镜像 `emcupy:blackwell-20261026` 的精确 image ID、原始基础镜像和版本记录在
[environment.json](environment.json)。该镜像是本机制作的快照，没有声明已经发布到
公共镜像仓库；换机器时须先通过 `docker save` / `docker load` 搬运，或按下节重建。
`create_container.sh` 校验 image ID，不会将同名但不同内容的镜像默认为已验证环境。
已有容器的 image ID、工作区挂载和 GPU 配置匹配时直接复用；已停止时启动它，
不再要求另开容器。配置不匹配时报告错误，由调用者检查。

本机日常编译、插件测试和 RELLIS GPU 制集统一使用 `emcupy`。2026-10-28
已将原 `rellis-emcupy-20261027` 改名为 `emcupy`，保留其 `/work`、`/builder`、
`/dataset` 和 `/home/root/catkin_emcp` 挂载；删除了仅挂载工作区的重复容器。
SLAM/物理响应继续使用 `rellis-builder-20261027`，因为当前 GPU 镜像缺少其
GeographicLib/GTSAM 动态库。容器清理不删除镜像、宿主工作区或验证记录。

日常使用：

```bash
bash tools/rellis_runtime/create_container.sh /mnt/nvme1n1/whd/emcp
bash tools/rellis_runtime/enter_container.sh
```

只有确实需要不同工作区或镜像进行独立验证时，才显式指定其他容器名：

```bash
EMCP_CONTAINER=emcupy-repro bash tools/rellis_runtime/create_container.sh /path/to/new/workspace
EMCP_CONTAINER=emcupy-repro bash tools/rellis_runtime/enter_container.sh
# 容器内：
bash src/elevation_mapping_cupy/tools/rellis_runtime/build_emcupy.sh
```

编译使用独立 `blackwell` profile 和 `build_blackwell` / `devel_blackwell`，Python
前缀默认 `/opt/emcupy-py310`，并行度为 4 个编译线程、2 个包。不要提交工作区构建目录。
`EMCP_WORKSPACE` 和 `EMCP_PYTHON_PREFIX` 可显式覆盖；C++ 嵌入 Python 必须用同一前缀重编译。

## 镜像重建与依赖锁定

```bash
docker build -t emcupy:rellis-rebuilt -f tools/rellis_runtime/Dockerfile tools/rellis_runtime
docker image inspect emcupy:rellis-rebuilt --format '{{.Id}}'
```

Dockerfile 固定原 ROS 基础镜像 digest；`conda-linux-64.lock` 保存 53 个包的实际 URL
和 MD5，`requirements-pip.lock` 保存 60 个 pip 包的版本（含 PyTorch cu128 索引）。
不使用包含构建机 `file://` 路径的原始 pip freeze，也不通过 pip 重装系统 ROS 包。
新构建的 image ID 与原快照可以不同，应完整验收后设置 `EMCP_IMAGE` 和 `EMCP_IMAGE_ID`
再创建容器。记录重建得到的 ID；没有测试过的重建镜像不能沿用原镜像的验收结论。
重建依赖外部包仓库可用性；原镜像归档是离线恢复的方式，Git 不保存镜像二进制。

## 真实传感器验收

测试夹具和输出位于源码仓库之外。可以从已有 foundation 重新提取，而不是依赖某个
已删除的旧数据目录。提取容器需要将 foundation 只读挂载，例如 `/foundation/00002`：

```bash
source src/elevation_mapping_cupy/tools/rellis_runtime/activate_emcupy.sh
python src/elevation_mapping_cupy/tools/rellis_runtime/prepare_fixtures.py \
  --foundation /foundation/00002 --output /home/root/catkin_emcp/validation/new_fixtures
bash src/elevation_mapping_cupy/tools/rellis_runtime/run_validation.sh \
  /home/root/catkin_emcp/validation/new_fixtures \
  /home/root/catkin_emcp/validation/runs/new_check
```

默认执行 CUDA/Torch、坡度/粗糙度/语义融合、真实 Ouster 到 GPU 栅格，以及 RELLIS
插件专项测试；随后通过 C++ ROS 节点检查真实点云、RGB/CameraInfo 和 GridMap 输出。
使用独立 ROS master（`EMCP_ROS_PORT` 默认 11641），检查端口占用，退出只清理自己
启动的进程组。姿态仅近似静止初始坐标，测试通过不代表 SLAM 精度或语义真值已验证。

原工作区根的同名脚本只转发至这里；原 `validation/` 日志和旧夹具作为历史证据保留，
不提交到 Git。完整制集标准见本机
[RELLIS 制作标准](/mnt/nvme1n1/whd/traversability_dataset_builder/docs/rellis_pipeline.md)。
