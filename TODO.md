# TODO 列表

2021-05-20

此项 TODO 列表是在程序大体完成后写的。由于程序基本可以正确使用了，所以并非所有的 TODO 都一定要完成。

## 1. 服务器部署

- [ ] 1-1 需要再验证一下大体系、小内存情况下的数值导数
- [x] 1-2 需要检查内存是否控制过于严格或产生溢出 (特别是开壳层) **结论：似乎没有严重溢出，但有些过剩。以后还需要微调。**
- [ ] 1-3 **未完全解决** 确定 cProfile 的工作流程，并确定各函数调用时间与打印方式
- [x] 1-4 确定较为自动的脚本，使得对于同一输入卡，服务器与本地可以轻松地分别执行大分子与小分子计算
- [ ] 1-5 确定是否能写队列脚本，是否可以用 Gaussian 输入卡作为 CLI
  
## 2 效率测试与效率提升

### 2.1 I/O

- [ ] 2.1-1 粗略评估较大体系下，硬盘算法与非硬盘算法的时间差距
- [ ] 2.1-2 评估分批算法与非分批算法的效率差距
- [x] 2.1-3 评估 PT2 激发张量一般情况下的硬盘读盘速度，确定 overhead 对效率的影响 (针对 Stoychev, Neese JCTC 2018, 10.1021/acs.jctc.8b00624)
- [ ] 2.1-4 确定并行读盘是否比串行读盘更快，以及确定合适的读盘并发线程数量，和硬盘理论读写速度
- [ ] 2.1-5 异步读盘与写盘 (参考 `pyscf.lib.call_in_background`，以及 `pyscf.ao2mo.outcore.general` 对该函数的使用)，并确定程序效率是否确实提升、正确性能否保证
- [ ] 2.1-6 如果并行读盘更快，且可以进行异步读盘与写盘，那么确定两者的平行是否会冲突

### 2.2 NumPy 与 TBLIS 效率

- [x] 2.2-1 评估 `pyscf.lib.transpose`，`mkl_?omatcopy`, numba 这几种做法的效率比较；**结论：竟然是 PySCF 的最快！……人傻了**
- [x] 2.2-2 确定如何使用张量转置，以及张量转置的合适并发线程数量；**结论：不控制并发数量；效率有一定提升。但还是有一些串行代码 (inplace add/multi) 可以优化；只是多半要用 C 底层来写了，不能用 numpy 或者 numexpr/numba**
- [x] 2.2-3 确定程序后门，即预留普通的 NumPy 转置与 C 的转置
- [ ] 2.2-4 一个程序 TODO：对于 `lib.einsum("Auv, suv -> A")` 的情况为何失败
- [ ] 2.2-5 对每个调用 einsum 的部分，在假设向量是连续的情况下，作效率测试；并确定类似于 `Ppq, pq -> P` 的问题是否先重定义形状再作 dot 会更快 **后来的评述：`Ppq, pq -> P` 在有 K 积分时不是决速步，先不考虑**
- [ ] 2.2-6 万一 tblis 用户没有安装，要考虑对于 `np.einsum(optimize=True)` 也会很慢的情况，要找到优化这种代码的方法；不要让 `np.einsum` 不能用 

### 2.3 其它程序效率

- [ ] 2.3-1 全面地了解 ORCA 在基组、RI、存盘、内存不同情况下的效率表现
- [ ] 2.3-2 重新作对于 Gaussian, ORCA, Q-Chem, Molpro, Psi4, TurboMole, MPQC 作默认算法下的效率测评

## 3 用户与开发者文档
  
- [ ] 3-1 确定程序的 API 文档风格与外观，自动化编写 module 的 API 文档
- [ ] 3-2 为了避免较短的程序与太长的文档，确定一种分离文档与程序的方式
- [ ] 3-3 用户文档需要参考 PySCF 更新过的文档 https://pyscf.org/user/mp.html
- [ ] 3-4 开发者文档需要写一份 Jupyter 的、简化的程序与公式直接对应的版本
- [ ] 3-5 开发者文档需要整理一下，程序的每个变量的生命周期 (何时生成，何时使用，是否多余等等)
- [ ] 3-6 统一变量名称的定义 (特别是与 ri 和 jkfit 有关的部分)

## 4 算法、替换函数与其它程序功能

- [ ] 4-1 重新确定所有函数是否有更简单的 PySCF 对应 (例子是一阶梯度的 JK 与 GGA 核算过程)
- [ ] 4-2 确定 RIJONX, RIJCOSX 是否可能实现
- [ ] 4-3 如果 RIJONX 实现难度不高，那么需要确认 1) RIJONX 在 CP-KS 过程的效率 2) 在 JK 梯度求取过程中的效率 3) 是否存在积分优化 (譬如对于长链分子 [pyscf/pyscf/#924](https://github.com/pyscf/pyscf/issues/924))
- [ ] 4-4 在梯度导数中，确定一种避免直接对 $J_{PQ} = (P|r^{-1}|Q)$ 作求逆的做法 (似乎 `pyscf.df.hessian.rhf._gen_jk` 可以做到？)
- [ ] 4-5 确定 Frozen Core 近似的做法
- [ ] 4-6 确定一阶梯度中，格点导数要如何写

## 5 细节的功能与程序风格

- [ ] 5-1 对于一阶梯度，需要与 PySCF 现有 API 接口作对接，实现类似于 `as_scanner` 或 `optimizer`
- [ ] 5-2 细致地评估一下程序的 restart 功能是否完善
- [ ] 5-3 确定一下程序重新执行的时候，是否可能出现结果变化的情况
- [ ] 5-4 写一个确定所需要硬盘空间大小的判断；如果硬盘空间远不如需要的内存空间，那么就走全内存的流程
- [ ] 5-5 确定并统一 Typing 的做法，避免大多数 IDE 提醒
- [ ] 5-6 确定如何写输出文件与调试输出
- [ ] 5-7 `calc_batch_size` 函数要能固定 batch 大小，用作调试；对于大分子，需要调试输出 batch 的信息
- [ ] 5-8 指定特定的原子列表计算梯度，而非所有原子 (与 PySCF 的大多数梯度函数能对应起来)
- [ ] 5-9 再确定有没有复杂函数能从类定义中提取出来

## 6 能量泛函的功能

- [ ] 6-1 溶剂化模型
- [ ] 6-2 穷举目前现有的泛函 (包括 DSD 型、XYG 型、ωB97 型、RSH 型)
- [ ] 6-3 作功能上的扩充 (包括 LT-SOS 型、RPA 型、D3 矫正型)
- [ ] 6-4 写一个 Conventional 情况下的能量接口，这个不应该很难
- [ ] 6-5 对周期性方法作实验性程序计算
- [ ] 6-6 确定 oniom 的能量是否还算容易计算；甚至是否允许 XO

## 7 程序部署

- [ ] 7-1 参考文档作程序部署的准备 https://pyscf.org/contributing.html#how-to-make-an-extension
- [ ] 7-2 寻找方法作 CI 与 CodeCov
- [ ] 7-3 重新确定测试文件要如何写
- [ ] 7-4 提交部署前，需要写完整的程序功能与局限列表 (但需要将这里的 TODO 尽可能完成再说)

## 8 具体程序的评述

- [x] 8-1 `get_gradient_jk`：并行效率 37/40，无需更改
- [ ] 8-2 `get_gradient_gga`：并行效率存在问题，但似乎是内存 bandwidth 控制，难以修改代码
- [ ] 8-3 `get_cderi_mo` 与 `get_cpks_eri`：这些涉及到是否允许 async 读写盘；但目前似乎无法判断程序效。
          甚至感到使用 async 之后程序效率更低；可能需要询问专家了。
- [ ] 8-4 `Ax0_Core_HF` 与 `Ax0_cpks_HF` 的效率在小体系体现不出问题，但大体系需要关心。
          还是解决不了异步的问题。