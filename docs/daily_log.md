# Daily Log

## 2026-05-01

**工作内容**：
- 明确 FEM 与 PINN 实验问题
- 创建项目结构与 README

**遇到问题**：
- 公式在 Markdown 中换行渲染问题，需要确保运算符不单独成行

**解决方案/思路**：
- 保证公式宽度，运算符放行尾
- 初始化 docs 文件夹与模板文件

---

## 2026-05-02

**工作内容**：
- FEM 网格生成模块开发
- 组装质量矩阵和刚度矩阵
- 隐式 Euler 时间推进实现
- Case 1 完整 FEM 流水线验证（L2 误差 5.86e-04）

---

## 2026-05-07

**工作内容**：
- 实现 PINN 正问题求解器（model、loss、sampling、train）
- 修复 FEM solver 边界条件处理：从 apply_dirichlet 移入 solver，在组合矩阵 A=M+dt*K 上施加 BC
- 修复 matplotlib 负号显示：改用 Microsoft YaHei 字体，调整 rcParams 设置顺序
- 修复 ic_loss 未使用的 alpha 参数、边界采样余数分配、推理循环重复创建 tensor
- 添加梯度裁剪 clip_grad_norm
- 重构可视化：2×3 对比图（解析解/数值解/误差分布 + 时间曲线/切面/误差切面）
- 实现验证体系：随机配点验证 + 网格误差回调 + 损失分量曲线
- 训练变量重命名加 train_ 前缀
- 实现时间泛化验证（val_time_ratio=0.6，训练 t∈[0,0.3]，外推 t∈(0.3,0.5]）
- 实现 Early Stop（基于验证损失）和最优模型保存（best_model.pt）
- 统一 colorbar，添加监测点标记和训练/外推分界线
- 更新所有文档（README、CLAUDE.md、experiments.md、daily_log.md）

**实验结果**：
- FEM Case 1：L2 误差 5.86e-04
- PINN Case 1（时间泛化）：训练域 L2≈0.9%，外推域 L2≈1.8%
