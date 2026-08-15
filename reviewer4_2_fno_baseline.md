# Reviewer 4, Comment 2：与 FNO、DeepONet、PINN 的比较

## 审稿意见与处理思路

审稿人建议进一步与 FNO、DeepONet 或 PINN 等方法比较。当前修订补充了与二维 FNO 的直接比较。所有 FNO 模型均使用 MSE loss 训练，并且采用与 U-Net 相同的位移输入、模量输出、数据划分、固定测试样本、噪声 realization 和评价指标。DeepONet 和 PINN 需要重新定义 branch/trunk 输入、PDE residual、collocation 点和优化预算，因此本轮不在缺少统一实验定义的情况下强行加入。

---

# 可直接用于回复审稿人的英文文本（精简版）

> **Response:** We thank the reviewer for suggesting comparisons with additional neural-operator and physics-based methods. In the revised manuscript, we added a direct comparison with a two-dimensional Fourier Neural Operator (FNO). All evaluated FNO configurations were trained using the mean-squared error (MSE) loss and used the same displacement input, modulus target, data split, fixed test samples, noise realizations, and error metrics as the U-Net models.
>
> On the MIX test set, the FNO configuration with width 64, 16 × 16 retained Fourier modes, and four Fourier layers obtained a mean relative \(L_1\) error of 0.00759 for clean inputs, compared with 0.00522, 0.00374, and 0.00397 for MSE-M, LM-M, and GM-M, respectively. At 10% input noise, its error increased to 0.14173, whereas the corresponding U-Net errors were 0.03021, 0.02420, and 0.02362. The Appendix now reports the architecture and computational information of four FNO configurations, the complete test statistics of the best-performing FNO architecture under six input-noise levels, and representative FNO prediction and relative-error fields. DeepONet and PINN were not added because a fair comparison would require a different branch/trunk or PDE-collocation formulation and a separately matched optimization budget; we state this scope explicitly rather than presenting an under-specified comparison.

---

# 建议放入论文附录的英文内容

## Appendix X. Comparison with Fourier neural operators

> We implemented two-dimensional Fourier Neural Operators (FNOs) in PyTorch using `torch.fft`. All FNO configurations were trained using the mean-squared error (MSE) loss. Each model receives the same two-component displacement field, \(\mathbf{x}\in\mathbb{R}^{N\times2\times40\times40}\), as the U-Net models and predicts the same modulus field, \(\hat{\mathbf{y}}\in\mathbb{R}^{N\times40\times40}\). No additional normalization was applied to the input or target. A two-dimensional coordinate grid normalized to \([0,1]\) is appended internally as part of the FNO architecture and does not introduce additional measured information.
>
> The FNOs were optimized with Adam using an initial learning rate of \(3\times10^{-4}\), a batch size of 32, a maximum of 1,500 epochs, a learning-rate patience of 10 epochs, an early-stopping patience of 25 epochs, and seed 42. Table X1 summarizes the four evaluated configurations. Complex-valued spectral weights are counted as two real-valued trainable scalars. The validation mean and standard deviation are calculated from the sample-wise relative \(L_1\) errors over the 4,000 clean MIX validation samples. Training times are the recorded durations of the corresponding runs and are included as implementation records rather than hardware-independent benchmarks.

**Table X1.** Architecture, computational size, validation statistics, and recorded training time of the FNO configurations. All configurations use four Fourier layers and are trained using the MSE loss. The mean and standard deviation are calculated from the sample-wise relative \(L_1\) errors over the 4,000 clean MIX validation samples.

| FNO configuration | Width | Fourier modes | Layers | Trainable parameters | MIX validation mean | MIX validation std | Training time (s) |
|:---|---:|:---:|---:|---:|---:|---:|---:|
| `w21–m8×8–l4` | 21 | 8 × 8 | 4 | 454,189 | 0.010581 | 0.034127 | 1,411 |
| `w32–m12×12–l4` | 32 | 12 × 12 | 4 | 2,365,025 | 0.008453 | 0.034429 | 1,020 |
| `w32–m16×16–l4` | 32 | 16 × 16 | 4 | 4,200,033 | 0.006972 | 0.028753 | 2,117 |
| `w64–m16×16–l4` | 64 | 16 × 16 | 4 | 16,798,913 | 0.004781 | 0.016281 | 4,353 |

> The best-performing architecture in Table X1, `w64–m16×16–l4`, was evaluated on the same fixed BIL, EXP, GRF, and MIX test samples at input-noise levels of 0%, 2%, 4%, 6%, 8%, and 10%. Table X2 reports its test performance. Each mean and standard deviation is calculated over the sample-wise relative \(L_1\) errors in the corresponding fixed test subset; the standard deviation therefore describes variation among test samples for the trained model.

**Table X2.** Statistical prediction results of the best-performing MSE-trained FNO architecture (`w64–m16×16–l4`) on the test datasets under different input-noise levels. Each entry reports the mean and standard deviation of the sample-wise relative \(L_1\) error.

| Noise level (%) | BIL mean | BIL std | EXP mean | EXP std | GRF mean | GRF std | MIX mean | MIX std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0026 | 0.0017 | 0.0020 | 0.0017 | 0.0175 | 0.1120 | 0.0076 | 0.0809 |
| 2 | 0.0205 | 0.0032 | 0.0197 | 0.0032 | 0.0314 | 0.1184 | 0.0241 | 0.0853 |
| 4 | 0.0428 | 0.0075 | 0.0412 | 0.0076 | 0.0526 | 0.1341 | 0.0459 | 0.0964 |
| 6 | 0.0693 | 0.0143 | 0.0663 | 0.0142 | 0.0793 | 0.1605 | 0.0721 | 0.1143 |
| 8 | 0.1009 | 0.0242 | 0.0963 | 0.0237 | 0.1125 | 0.1992 | 0.1038 | 0.1404 |
| 10 | 0.1383 | 0.0375 | 0.1319 | 0.0363 | 0.1531 | 0.2532 | 0.1417 | 0.1766 |

> Figures X1 and X2 show the FNO prediction and pointwise relative-error fields for BIL sample 600, EXP sample 200, and GRF sample 410. The column headings show the dataset names without sample indices, while rows (a)–(f) correspond to 0%, 2%, 4%, 6%, 8%, and 10% input noise. The same noise realization used in the quantitative comparison is retained for each sample and noise level.

![FNO prediction fields arranged by noise level and sample case](results/force_load/asm_unet_comparison/GN/fno_sample_grid/fno_prediction_noise_by_sample_GN.png)

> **Fig. X1.** Predicted modulus fields obtained with the MSE-trained FNO (`w64–m16×16–l4`). Columns show BIL sample 600, EXP sample 200, and GRF sample 410; rows (a)–(f) show input-noise levels of 0%, 2%, 4%, 6%, 8%, and 10%, respectively. Each panel has its own colorbar, while identical color limits are used for all noise levels of the same sample case.

![FNO relative-error fields arranged by noise level and sample case](results/force_load/asm_unet_comparison/GN/fno_sample_grid/fno_error_noise_by_sample_GN.png)

> **Fig. X2.** Pointwise relative-error fields corresponding to Fig. X1. Rows (a)–(f) represent input-noise levels of 0%, 2%, 4%, 6%, 8%, and 10%, respectively. The value shown in each panel is the sample-wise relative \(L_1\) error, and a common 0%–10% error scale is used for all panels.

> The reported results characterize the four MSE-trained FNO configurations evaluated here. They should not be interpreted as an exhaustive assessment of the FNO family because the number of Fourier layers was fixed, all runs used one training seed, and no noise augmentation was applied. Noise-aware validation and training-time noise augmentation may improve robustness but are outside the scope of the current revision.

---

# 复现脚本与结果来源

FNO-only 总图由以下脚本读取已保存的逐噪声 `.npz` 文件生成，不重新运行模型推理：

```powershell
python igfe_unet/script/plot_fno_sample_grid.py
```

脚本默认使用 `bil:600 exp:200 grf:410` 三列和 `0 2 4 6 8 10` 六行，也可以通过 `--cases`、`--noise-levels`、`--comparison-root` 和 `--output-dir` 修改。

```text
igfe_unet/script/plot_fno_sample_grid.py
results_fno/force_load/architecture_sweep/validation_selection.csv
results_fno/force_load/architecture_sweep/all_experiments.csv
trained_models_fno/force_load/arch/*/all_samples_test/
results/force_load/asm_unet_comparison/GN/fno_sample_grid/fno_prediction_noise_by_sample_GN.png
results/force_load/asm_unet_comparison/GN/fno_sample_grid/fno_error_noise_by_sample_GN.png
```
