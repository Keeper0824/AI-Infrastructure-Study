# WSL 实习练习工作约定

- 使用中文交流。
- 用户新建的下载、源码、脚本、模型缓存、实验报告和环境清单，默认放在 Windows 挂载盘 `/mnt/` 下，便于 Windows 直接查看。
- 本项目根目录：`/mnt/e/Visual Studio code/2026_code/vllm`，对应 `E:\Visual Studio code\2026_code\vllm`。
- 默认下载目录：`/mnt/d/s_dis/Downloads`，对应 `D:\s_dis\Downloads`。
- 系统安装目录与现有 Miniconda 环境保持其必要的 Linux 路径；不要擅自迁移已安装环境。
- 使用现有 Conda 环境 `ai-infra`，不新建第二个项目环境。
- 保留 `nanovllm/` 已有内容；环境验收脚本放 `scripts/`，采集结果和环境记录放 `results/`。
- 在 NTFS 挂载目录测得的文件 I/O 和编译耗时，不直接当作原生 Linux 文件系统性能。
- GPU 检查只做小规模正确性验证；未经用户要求，不启动长时间压测、不停止量化系统和串流服务。
