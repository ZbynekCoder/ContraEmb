考虑如下的对比学习优化问题，设$E:\mathcal{B} \to \mathbb{R}^n,T:\mathbb{R}^n \to \mathbb{R}^n$为可学习算子，$\lambda$为固定常数，记$z_q = [E(q), \lambda T(E(q))],z_d = [E(d),E(d)]$，对于任一$q \in \mathcal{B}$，其给出$\mathcal{B}$的一个划分$\mathcal{B} = \{q\} \cup \mathcal{B}^+ \cup \mathcal{B}^\dagger \cup \mathcal{B}^-$。设对应元素分别为$d^+,d^\dagger,d^-$，设计对比学习loss function使其满足

1. $d(E(q),E(d^+))$减小，$d(T(E(q)),E(d^+))$减小
2. $d(E(q),E(d^\dagger))$减小，$d(T(E(q)),E(d^\dagger))$增大
3. $d(E(q),E(d^-))$增大
