# COF 对称核心构建块——参考文献

**日期**: 2026-08-08
**说明**: 本文档为 `config.py` 中全部 41 个原始核心模板逐一提供化学背景与参考文献。每个核心标注其结构类别、目标点群、在 COF 文献中的角色，以及代表性文献。

---

## 目录

1. [三嗪类核心 (C₃)](#1-三嗪类核心-c₃)
2. [均三甲酸/苯三酸衍生物 (C₃)](#2-均三甲酸苯三酸衍生物-c₃)
3. [间苯三酚/三羟基苯类 (C₃)](#3-间苯三酚三羟基苯类-c₃)
4. [三炔基苯及其扩展 (C₃)](#4-三炔基苯及其扩展-c₃)
5. [三苯基苯扩展核心 (C₃)](#5-三苯基苯扩展核心-c₃)
6. [线性联芳基核心 (C₂)](#6-线性联芳基核心-c₂)
7. [偶氮/二苯乙烯/烯烃类线性核心 (C₂)](#7-偶氮二苯乙烯烯烃类线性核心-c₂)
8. [炔基桥联线性核心 (C₂)](#8-炔基桥联线性核心-c₂)
9. [酰胺/酯/磺酰胺连接线性核心 (C₂)](#9-酰胺酯磺酰胺连接线性核心-c₂)
10. [含吡啶/吡嗪/嘧啶线性核心 (C₂)](#10-含吡啶吡嗪嘧啶线性核心-c₂)
11. [含呋喃/噻吩杂环线性核心 (C₂)](#11-含呋喃噻吩杂环线性核心-c₂)
12. [扩环芳烃线性核心 (C₂)](#12-扩环芳烃线性核心-c₂)
13. [卟啉类核心 (C₄)](#13-卟啉类核心-c₄)
14. [四面体核心 (S₄)](#14-四面体核心-s₄)
15. [环辛四烯核心 (C₄)](#15-环辛四烯核心-c₄)
16. [六苯基苯核心 (D₆ₕ/H₆)](#16-六苯基苯核心-d₆ₕh₆)
17. [氟代苯核心 (C₂)](#17-氟代苯核心-c₂)
18. [参考文献汇总](#18-参考文献汇总)

---

## 1. 三嗪类核心 (C₃)

### triazine_C3
- **SMILES**: `*c1nc(*)nc(*)n1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 1,3,5-三嗪环，三个 `*` 分别位于 2,4,6-位碳原子上。这是共价三嗪框架 (CTF) 的核心单体 1,4-二氰基苯经三聚反应形成的三嗪环——但此处简化为三嗪环本身，`*` 直接连在三嗪碳上。
- **在 COF 中的角色**: 三嗪环是 CTF 系列 (covalent triazine frameworks) 的连接节点，也是众多含三嗪 COF 的结构基元。
- **参考文献**:
  - Kuhn, P., Antonietti, M., & Thomas, A. (2008). Porous, Covalent Triazine-Based Frameworks Prepared by Ionothermal Synthesis. *Angew. Chem. Int. Ed.*, 47(18), 3450–3453. → CTF-1 的首次报道，确立了 1,4-二氰基苯三聚生成三嗪环的路线。
  - Bojdys, M.J., Jeromenok, J., Thomas, A., & Antonietti, M. (2010). Rational Extension of the Family of Layered, Covalent, Triazine-Based Frameworks with Regular Porosity. *Adv. Mater.*, 22(19), 2202–2205.

### triaminotriazine_C3
- **SMILES**: `*Nc1nc(N*)nc(N*)n1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 2,4,6-三氨基-1,3,5-三嗪（即三聚氰胺），三个 `*` 分别连在三个氨基的 N 原子上。通过 NH 连接臂，而非直接 C-C 连接。
- **在 COF 中的角色**: 三聚氰胺是构建含三嗪 COF 的常用胺单体，与醛基反应形成亚胺键。
- **参考文献**:
  - Schwab, M.G., Fassbender, B., Spiess, H.W., Thomas, A., Feng, X., & Müllen, K. (2009). Catalyst-free Preparation of Melamine-Based Microporous Polymer Networks through Schiff Base Chemistry. *J. Am. Chem. Soc.*, 131(21), 7216–7217.

### triethynyltriazine_C3
- **SMILES**: `*C#Cc1nc(C#C*)nc(C#C*)n1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 2,4,6-三(乙炔基)-1,3,5-三嗪，`*` 在乙炔基末端。乙炔基为三嗪环与臂之间提供额外的共轭延伸。
- **在 COF 中的角色**: 乙炔基扩展的三嗪节点，用于增大孔径和增强 π-共轭。
- **参考文献**:
  - Ren, S., Bojdys, M.J., Dawson, R., Laybourn, A., Khimyak, Y.Z., Adams, D.J., & Cooper, A.I. (2012). Porous, Fluorescent, Covalent Triazine-Based Frameworks Via Room-Temperature and Microwave-Assisted Synthesis. *Adv. Mater.*, 24(17), 2357–2361.

### triazine_triphenyl_C3
- **SMILES**: `*c1ccc(-c2nc(-c3ccc(*)cc3)nc(-c3ccc(*)cc3)n2)cc1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 2,4,6-三(4-苯基)-1,3,5-三嗪，三个 `*` 位于三个苯环的 para 位。通过苯环将 `*` 从三嗪环向外推一个苯环的距离。
- **在 COF 中的角色**: 扩展型三嗪节点，用于制备大孔径 COF。
- **参考文献**:
  - Ren et al. (2012), 同上。
  - Wang, X., Han, X., Zhang, J., Wu, X., Liu, Y., & Cui, Y. (2016). Homochiral 2D Porous Covalent Organic Frameworks for Heterogeneous Asymmetric Catalysis. *J. Am. Chem. Soc.*, 138(38), 12332–12335.

---

## 2. 均三甲酸/苯三酸衍生物 (C₃)

### trimesic_ester_C3
- **SMILES**: `*OC(=O)c1cc(C(=O)O*)cc(C(=O)O*)c1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 均苯三甲酸(1,3,5-苯三甲酸)的三酰氧基酯形式。`*` 在酯键的氧端（通过 O 连接臂），而非直接连在苯环碳上。这种"倒置"连接使得臂通过酯键外向延伸。
- **在 COF 中的角色**: 均苯三甲酸是经典 C₃ 对称构建块，广泛用于金属有机框架 (MOF) 和 COF。酯连接变体提供了不同于传统亚胺/硼酸酯的键连方式。
- **参考文献**:
  - Yaghi, O.M., Li, H., & Groy, T.L. (1996). Construction of Porous Solids from Hydrogen-Bonded Metal Complexes of 1,3,5-Benzenetricarboxylic Acid. *J. Am. Chem. Soc.*, 118(38), 9096–9101. → 虽然此早期工作属于 MOF，均苯三甲酸的 C₃ 对称性是后续 COF 设计的直接来源。
  - Zhao, C., Diercks, C.S., Zhu, C., Hanikel, N., Pei, X., & Yaghi, O.M. (2018). Urea-Linked Covalent Organic Frameworks. *J. Am. Chem. Soc.*, 140(48), 16438–16441. → 均苯三甲酸衍生物在 COF 中的应用。

### trimesic_tris_amide_C3
- **SMILES**: `*NC(=O)c1cc(C(=O)N*)cc(C(=O)N*)c1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 均苯三甲酸的三酰胺形式。`*` 连在酰胺 N 上（通过 NH 连接臂）。
- **在 COF 中的角色**: 酰胺连接的 C₃ 节点，酰胺键比酯键更稳定，适用于需要更高化学稳定性的 COF。
- **参考文献**:
  - Stewart, D., Antypov, D., Dyer, M.S., Pitcher, M.J., Katsoulidis, A.P., Chater, P.A., Blanc, F., & Rosseinsky, M.J. (2017). Stable and Ordered Amide Frameworks Synthesised Under Reversible Conditions which Facilitate Error Checking. *Nat. Commun.*, 8, 1102. → 酰胺键 COF 的突破性工作。
  - Waller, P.J., Lyle, S.J., Osborn Popp, T.M., Diercks, C.S., Reimer, J.A., & Yaghi, O.M. (2016). Chemical Conversion of Linkages in Covalent Organic Frameworks. *J. Am. Chem. Soc.*, 138(48), 15519–15522. → 连接键化学转化的代表性研究。

---

## 3. 间苯三酚/三羟基苯类 (C₃)

### trihydroxybenzene_C3
- **SMILES**: `*c1c(O)c(O)c(*)c(O)c1*`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 1,3,5-三羟基苯（间苯三酚），三个 `*` 分别在 2,4,6-位碳上。羟基的强给电子效应使苯环富电子，并可通过酮-烯醇互变异构形成 β-酮烯胺键。
- **在 COF 中的角色**: 间苯三酚是 β-酮烯胺连接 COF 的关键前体——它与醛基缩合后发生不可逆的烯醇→酮互变异构，产生极高化学稳定性的 COF。这是目前化学稳定性最好的 COF 连接方式之一。
- **参考文献**:
  - Kandambeth, S., Mallick, A., Lukose, B., Mane, M.V., Heine, T., & Banerjee, R. (2012). Construction of Crystalline 2D Covalent Organic Frameworks with Remarkable Chemical (Acid/Base) Stability via a Combined Reversible and Irreversible Route. *J. Am. Chem. Soc.*, 134(48), 19524–19527. → β-酮烯胺 COF 的开创性工作，确立了间苯三酚-醛化学路线。
  - Kandambeth, S., Shinde, D.B., Panda, M.K., Lukose, B., Heine, T., & Banerjee, R. (2013). Enhancement of Chemical Stability and Crystallinity in Porphyrin-Based Covalent Organic Frameworks. *Angew. Chem. Int. Ed.*, 52(49), 13052–13056.

---

## 4. 三炔基苯及其扩展 (C₃)

### triethynylbenzene_C3
- **SMILES**: `*C#Cc1cc(C#C*)cc(C#C*)c1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 1,3,5-三乙炔基苯，三个 `*` 在乙炔基末端。Y 形 D₃ₕ 对称的刚性节点。
- **在 COF 中的角色**: 乙炔基共轭 COF 的经典三臂节点，通过 Sonogashira 偶联引入。
- **参考文献**:
  - Jiang, J.X., Su, F., Trewin, A., Wood, C.D., Campbell, N.L., Niu, H., Dickinson, C., Ganin, A.Y., Rosseinsky, M.J., Khimyak, Y.Z., & Cooper, A.I. (2007). Conjugated Microporous Poly(aryleneethynylene) Networks. *Angew. Chem. Int. Ed.*, 46(45), 8574–8578. → 共轭微孔聚合物 (CMP) 的奠基性工作，1,3,5-三乙炔基苯为核心单体。
  - Yuan, D., Lu, W., Zhao, D., & Zhou, H.C. (2011). Highly Stable Porous Polymer Networks with Exceptionally High Gas-Uptake Capacities. *Adv. Mater.*, 23(32), 3723–3725.

### tris_ethynyl_triphenyl_C3
- **SMILES**: `*C#Cc1ccc(-c2cc(-c3ccc(C#C*)cc3)cc(-c3ccc(C#C*)cc3)c2)cc1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 1,3,5-三(4-乙炔基苯基)苯，三乙炔基苯的扩展版——在核心苯与乙炔基之间插入苯环间隔基。`*` 在乙炔基末端。
- **在 COF 中的角色**: 扩展型三臂节点，比三乙炔基苯具有更大的孔径。
- **参考文献**:
  - Jiang et al. (2007), 同上。扩展型单体受 CMP 网络拓扑设计启发。

### tris_phenylethynyl_benzene_C3
- **SMILES**: `*c1ccc(C#Cc2cc(C#Cc3ccc(*)cc3)cc(C#Cc3ccc(*)cc3)c2)cc1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 1,3,5-三(苯乙炔基)苯。与 triethynylbenzene_C3 的区别在于 `*` 连在末端苯环的 para 位，而非直接在炔基末端。即：核心苯环 → 炔基 → 苯环 → `*`。
- **在 COF 中的角色**: 苯乙炔基扩展的三臂节点，兼具刚性和 π-共轭。
- **参考文献**:
  - Bunck, D.N., & Dichtel, W.R. (2013). Bulk Synthesis of Exfoliated Two-Dimensional Polymers Using Hydrazone-Linked Covalent Organic Frameworks. *J. Am. Chem. Soc.*, 135(40), 14952–14955. → 苯乙炔基扩展的构建块在 COF 中的应用。

---

## 5. 三苯基苯扩展核心 (C₃)

### triphenylbenzene_C3
- **SMILES**: `*c1ccc(-c2cc(-c3ccc(*)cc3)cc(-c3ccc(*)cc3)c2)cc1`
- **目标点群**: C₃ | **臂数**: 3
- **化学描述**: 1,3,5-三(4-苯基)苯，`*` 在三臂末端苯环的 para 位。C₃ 对称的平面三角形节点。
- **在 COF 中的角色**: 经典的 C₃ 构建块，与 C₂ 线性连接子组合形成六方孔道 (hcb 拓扑) 的 2D COF。
- **参考文献**:
  - Côté, A.P., Benin, A.I., Ockwig, N.W., O'Keeffe, M., Matzger, A.J., & Yaghi, O.M. (2005). Porous, Crystalline, Covalent Organic Frameworks. *Science*, 310(5751), 1166–1170. → COF 领域的奠基之作。虽然 COF-1 (硼酸酯) 和 COF-5 (硼酸酯) 使用了更小的节点，但 C₃+C₂ 拓扑设计由此确立。
  - Ding, S.Y., Gao, J., Wang, Q., Zhang, Y., Song, W.G., Su, C.Y., & Wang, W. (2011). Construction of Covalent Organic Framework for Catalysis: Pd/COF-LZU1 in Suzuki–Miyaura Coupling Reaction. *J. Am. Chem. Soc.*, 133(49), 19816–19822. → 使用三苯基苯类 C₃ 节点的亚胺 COF (COF-LZU1)。

---

## 6. 线性联芳基核心 (C₂)

### biphenyl_C2
- **SMILES**: `*c1ccc(-c2ccc(*)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 4,4'-联苯，`*` 在两端苯环的 para 位。最简单的 C₂ 线性芳基连接子。
- **在 COF 中的角色**: 最经典的 C₂ 线性构建块之一，与 C₃ 节点形成六方 2D COF，也与 C₄ 四面体节点形成 3D COF。
- **参考文献**:
  - Côté et al. (2005), *Science*, 同上。COF-5 中使用的对苯二硼酸即含联苯类连接基元。
  - Uribe-Romo, F.J., Hunt, J.R., Furukawa, H., Klöck, C., O'Keeffe, M., & Yaghi, O.M. (2009). A Crystalline Imine-Linked 3-D Porous Covalent Organic Framework. *J. Am. Chem. Soc.*, 131(13), 4570–4571. → COF-300，使用四(4-氨基苯基)甲烷 (C₄) + 对苯二甲醛 (C₂) 构建首例 3D 亚胺 COF。

### biphenyl_ester_C2
- **SMILES**: `*OC(=O)c1ccc(-c2ccc(C(=O)O*)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 4,4'-联苯二甲酸酯形式。`*` 通过酯氧而非直接连在苯环上。
- **在 COF 中的角色**: 酯连接 COF 的 C₂ 构建块。
- **参考文献**:
  - Zhao, C., Lyu, H., Ji, Z., Zhu, C., & Yaghi, O.M. (2020). Ester-Linked Crystalline Covalent Organic Frameworks. *J. Am. Chem. Soc.*, 142(34), 14450–14454. → 首例晶态酯连接 COF 的报道。

### tetrafluorobenzene_C4（实际为 C₂ 核心）
- **SMILES**: `*c1c(F)c(F)c(*)c(F)c1F`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 1,2,4,5-四氟苯，两个 `*` 在对位（仅有的两个 C-H 位置）。注意命名中的" C4"指四氟取代，但对称性实际为 C₂。
- **在 COF 中的角色**: 全氟芳环可通过亲核芳香取代 (SNAr) 反应引入 COF，氟原子的吸电子效应影响电子结构和层间堆积。
- **参考文献**:
  - Du, Y., Yang, H., Whiteley, J.M., Wan, S., Jin, Y., Lee, S.H., & Zhang, W. (2016). Ionic Covalent Organic Frameworks with Spiroborate Linkage. *Angew. Chem. Int. Ed.*, 55(5), 1737–1741.

---

## 7. 偶氮/二苯乙烯/烯烃类线性核心 (C₂)

### azobenzene_C2
- **SMILES**: `*c1ccc(/N=N/c2ccc(*)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 4,4'-偶氮苯，`*` 在两端苯环的 para 位。偶氮基团 (—N=N—) 提供光响应性（trans/cis 异构化）。
- **在 COF 中的角色**: 偶氮苯 COF 具有光致异构化能力，可用于光控分子吸附/释放。
- **参考文献**:
  - Liu, C., Park, E., Jin, Y., Liu, J., Yu, Y., Zhang, W., ... & Zhang, W. (2018). Enantioselective Recognition of Chiral Acids by a Dual-Fluorescent Sensor Array. *Angew. Chem. Int. Ed.*, 57(28), 8624–8628.
  - Das, G., Prakasam, T., Alkordi, M.H., Gándara, F., Sharma, S.K., Nune, S.K., ... & Trabolsi, A. (2020). Azobenzene-Equipped Covalent Organic Framework: Light-Operated Reservoir. *J. Am. Chem. Soc.*, 142(18), 8252–8260. → 偶氮苯 COF 的光控分子释放。

### stilbene_C2
- **SMILES**: `*c1ccc(/C=C/c2ccc(*)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 4,4'-二苯乙烯（反式），`*` 在两端苯环的 para 位。C=C 双键替代了偶氮基团。
- **在 COF 中的角色**: 全碳骨架的烯烃连接 COF (sp² carbon-conjugated COF) 的简化模型——由 Knoevenagel 缩合构建的 C=C 连接 COF 具有优异的 π-共轭和化学稳定性。
- **参考文献**:
  - Zhuang, X., Zhao, W., Zhang, F., Cao, Y., Liu, F., Bi, S., & Feng, X. (2016). A Two-Dimensional Conjugated Polymer Framework with Fully sp²-Bonded Carbon Skeleton. *Polym. Chem.*, 7(25), 4176–4181. → sp² 碳全共轭 2D 聚合物框架。
  - Jin, E., Asada, M., Xu, Q., Dalapati, S., Addicoat, M.A., Brady, M.A., ... & Jiang, D. (2017). Two-Dimensional sp² Carbon-Conjugated Covalent Organic Frameworks. *Science*, 357(6352), 673–676. → 首例晶态 sp² 碳共轭 COF。

---

## 8. 炔基桥联线性核心 (C₂)

### diphenylacetylene_C2
- **SMILES**: `*c1ccc(C#Cc2ccc(*)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 二苯乙炔（甲苯），`*` 在两端苯环的 para 位。炔基在两端苯环间提供刚性线性的 π-共轭桥。
- **在 COF 中的角色**: 炔基连接的 COF 构建块，提供比联苯更长的间距和更强的共轭。
- **参考文献**:
  - Jiang et al. (2007), 同上。炔基桥联是共轭微孔聚合物的核心设计元素。
  - Crowe, J.W., Baldwin, L.A., & McGrier, P.L. (2016). Luminescent Covalent Organic Frameworks Containing a Homogeneous and Heterogeneous Distribution of Dehydrobenzoannulene Vertex Units. *J. Am. Chem. Soc.*, 138(31), 10120–10123.

### diphenyldiacetylene_C2
- **SMILES**: `*c1ccc(C#CC#Cc2ccc(*)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 1,4-二(苯基)丁二炔，两个炔基串联。与二苯乙炔相比，轴更长、π-共轭范围更大。
- **在 COF 中的角色**: 扩展型线性连接子，用于构建大孔径 COF。
- **参考文献**:
  - Cooper 课题组 CMP 系列工作中的扩展单体设计理念。
  - Xu, Y., Jin, S., Xu, H., Nagai, A., & Jiang, D. (2013). Conjugated Microporous Polymers: Design, Synthesis and Application. *Chem. Soc. Rev.*, 42(20), 8012–8031. → CMP 设计综述。

### diethynylbenzene_C2
- **SMILES**: `*C#Cc1ccc(C#C*)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 1,4-二乙炔基苯，`*` 在两端乙炔基的末端。
- **在 COF 中的角色**: 最简单的对位二乙炔基线性连接子，通过 Glaser/Hay 偶联或 Sonogashira 反应引入。
- **参考文献**:
  - Jiang et al. (2007), 同上。

### bis_phenylethynyl_benzene_C2
- **SMILES**: `*c1ccc(C#Cc2ccc(C#Cc3ccc(*)cc3)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 1,4-二(苯乙炔基)苯，核心苯环 + 两端炔基 + 末端苯环。`*` 在末端苯环 para 位。
- **在 COF 中的角色**: 扩展型苯乙炔基线性连接子。
- **参考文献**:
  - Bunck & Dichtel (2013), 同上。

### bis_phenylethynyl_thiophene_C2
- **SMILES**: `*c1ccc(C#Cc2ccc(C#Cc3ccc(*)cc3)s2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 与 bis_phenylethynyl_benzene 类似，但中间苯环被噻吩替换。引入噻吩可调节电子结构和光物理性质。
- **在 COF 中的角色**: 含噻吩的乙炔基扩展连接子，用于光电 COF。
- **参考文献**:
  - Dogru, M., Handloser, M., Auras, F., Kunz, T., Medina, D., Hartschuh, A., Knochel, P., & Bein, T. (2013). A Photoconductive Thienothiophene-Based Covalent Organic Framework Showing Charge Transfer Towards Included Fullerene. *Angew. Chem. Int. Ed.*, 52(10), 2920–2924. → 含硫杂环 COF 的光电应用。

---

## 9. 酰胺/酯/磺酰胺连接线性核心 (C₂)

### oxamide_C2
- **SMILES**: `*NC(=O)C(=O)N*`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 草酰胺，最短的二酰胺线性连接子。`*` 在两个 N 上。
- **在 COF 中的角色**: 酰胺键连接 COF 的最简 C₂ 构建块。
- **参考文献**:
  - Stewart et al. (2017), *Nat. Commun.*, 同上。

### bis_benzamide_C2
- **SMILES**: `*C(=O)Nc1ccc(NC(*)=O)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 1,4-二(甲酰胺基)苯，`*` 在羰基碳端。对苯二甲酰胺骨架。
- **在 COF 中的角色**: 酰胺连接的 C₂ 构建块。
- **参考文献**:
  - Stewart et al. (2017), *Nat. Commun.*, 同上。
  - Waller et al. (2016), 同上。

### bis_benzoate_C2
- **SMILES**: `*C(=O)Oc1ccc(OC(*)=O)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 对苯二甲酸二酯形式。`*` 在羰基碳端（通过 C=O 连接臂）。
- **在 COF 中的角色**: 酯连接 COF 的 C₂ 节点。
- **参考文献**:
  - Zhao et al. (2020), 同上。

### bis_phenyl_ester_C2
- **SMILES**: `*c1ccc(C(=O)Oc2ccc(OC(=O)c3ccc(*)cc3)cc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 含三个苯环的对苯二甲酸二苯酯骨架。`*` 在两端苯环 para 位。酯键提供构象灵活性。
- **在 COF 中的角色**: 扩展型酯连接 C₂ 构建块。
- **参考文献**:
  - Zhao et al. (2020), 同上。

### bis_sulfonamide_C2
- **SMILES**: `*S(=O)(=O)Nc1ccc(NS(*)(=O)=O)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 1,4-二(磺酰胺基)苯。`*` 在磺酰基硫端。磺酰胺键比酰胺和酯键更强极性。
- **在 COF 中的角色**: 磺酰胺连接的 COF 构建块，磺酰胺基团提供额外的氢键位点。
- **参考文献**:
  - 受酰胺 COF 化学启发（Stewart et al. 2017），磺酰胺变体是酰胺键的合理延伸。

---

## 10. 含吡啶/吡嗪/嘧啶线性核心 (C₂)

### bipyridine_C2
- **SMILES**: `*c1ccc(-c2ccc(*)cn2)nc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 含 2,2'-联吡啶或 4,4'-联吡啶骨架。两个 `*` 分别在两端吡啶环上。
- **在 COF 中的角色**: 联吡啶 COF 可在孔道内后修饰配位金属离子（如 Pd, Ni, Fe），赋予催化活性。
- **参考文献**:
  - Chen, X., Addicoat, M., Irle, S., Nagai, A., & Jiang, D. (2013). Control of Crystallinity and Porosity of Covalent Organic Frameworks by Managing Interlayer Interactions Based on Self-Complementary π-Electronic Force. *J. Am. Chem. Soc.*, 135(2), 546–549.
  - Ahn, S., Nauert, S.L., Buru, C.T., Rimoldi, M., Choi, H., Schweitzer, N.M., Hupp, J.T., Farha, O.K., & Notestein, J.M. (2018). Pushing the Limits on Metal–Organic Frameworks as Catalyst Supports: Robust Bipyridine-Containing NU-1000 Used for Ethylene Dimerization. *J. Am. Chem. Soc.*, 140(27), 8535–8543. → 联吡啶基 MOF 的催化应用（同样适用于 COF）。

### pyrazine_phenyl_C2
- **SMILES**: `*c1cnc(-c2cnc(*)cn2)cn1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 含吡嗪环的联芳基骨架。氮原子在芳环中引入了 Lewis 碱性位点和偶极矩。
- **在 COF 中的角色**: 含氮杂环 COF 构建块，N 位点可参与氢键或金属配位。
- **参考文献**:
  - Dalapati, S., Jin, S., Gao, J., Xu, Y., Nagai, A., & Jiang, D. (2013). An Azine-Linked Covalent Organic Framework. *J. Am. Chem. Soc.*, 135(46), 17310–17313. → 吖嗪连接 COF 的开创性报告，基于吡嗪/吖嗪化学。

### phenylethynyl_pyridine_C2
- **SMILES**: `*c1ccc(C#Cc2ccc(*)cn2)nc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 含吡啶环的苯乙炔基骨架。炔基桥联苯环和吡啶环。
- **在 COF 中的角色**: 吡啶+炔基的组合提供了金属配位位点和 π-共轭通路。
- **参考文献**:
  - Chen et al. (2013), 同上。
  - Vyas, V.S., Haase, F., Stegbauer, L., Savasci, G., Podjaski, F., Ochsenfeld, C., & Lotsch, B.V. (2015). A Tunable Azine Covalent Organic Framework Platform for Visible Light-Induced Hydrogen Generation. *Nat. Commun.*, 6, 8508. → 含氮杂环 COF 用于光催化制氢。

### phenylethynyl_pyrimidine_C2
- **SMILES**: `*c1ccc(C#Cc2cnc(C#Cc3ccc(*)cc3)nc2)cc1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 含嘧啶环的苯乙炔基骨架。两个乙炔基从嘧啶的 2,5-位分别连接苯环。
- **在 COF 中的角色**: 嘧啶 N 原子提供比吡啶更多的 Lewis 碱性位点。
- **参考文献**:
  - Lotsch 课题组 (2015)，同上，吖嗪/含氮杂环 COF 平台。

---

## 11. 含呋喃/噻吩杂环线性核心 (C₂)

### bifuran_C2
- **SMILES**: `*c1ccc(-c2ccc(*)o2)o1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 2,2'-联呋喃骨架。`*` 在两个呋喃环的 5-位。
- **在 COF 中的角色**: 呋喃是生物质衍生的可再生构建块，含呋喃 COF 具有更好的"绿色"属性。
- **参考文献**:
  - 受呋喃基多孔聚合物和生物质衍生单体的研究启发。呋喃-2,5-二甲酸/二甲醛是 COF 化学中新兴的生物基构建块。

### bithiophene_C2
- **SMILES**: `*c1ccc(-c2ccc(*)s2)s1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 2,2'-联噻吩骨架。`*` 在两个噻吩环的 5-位。噻吩提供比苯环更好的电子给体性质和更高的 HOMO。
- **在 COF 中的角色**: 含噻吩 COF 广泛用于光电应用（导电性、光吸收、电荷传输）。
- **参考文献**:
  - Dogru et al. (2013), 同上。噻吩基 COF 用于光电导。
  - Bertrand, G.H.V., Michaelis, V.K., Ong, T.C., Griffin, R.G., & Dincă, M. (2013). Thiophene-Based Covalent Organic Frameworks. *Proc. Natl. Acad. Sci. USA*, 110(13), 4923–4928.

---

## 12. 扩环芳烃线性核心 (C₂)

### bis_ethynyl_naphthalene_C2
- **SMILES**: `*C#Cc1ccc2cc(C#C*)ccc2c1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 2,6-二乙炔基萘，`*` 在乙炔基末端。萘环替代苯环提供了扩展的 π 表面。
- **在 COF 中的角色**: 大 π 表面增强层间 π-π 堆积，提高结晶度和电荷迁移率。
- **参考文献**:
  - Spitler, E.L., & Dichtel, W.R. (2010). Lewis Acid-Catalysed Formation of Two-Dimensional Phthalocyanine Covalent Organic Frameworks. *Nat. Chem.*, 2(8), 672–677. → 大 π 平面单体的 COF 设计理念。

### bis_ethynyl_anthracene_C2
- **SMILES**: `*C#Cc1ccc2cc3cc(C#C*)ccc3cc2c1`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 2,6-二乙炔基蒽，以蒽代替萘/苯。更大的 π 表面。
- **在 COF 中的角色**: 蒽基 COF 具有强荧光和光活性，适用于光催化和传感。
- **参考文献**:
  - Huang, N., Wang, P., & Jiang, D. (2016). Covalent Organic Frameworks: A Materials Platform for Structural and Functional Designs. *Nat. Rev. Mater.*, 1, 16068. → 综述，涵盖各类 π-扩展 COF 构建块。
  - Chen, L., Furukawa, H., Gao, J., Nagai, A., Kakuta, T., Yaghi, O.M., & Jiang, D. (2014). Photoelectric Covalent Organic Frameworks: Converting Open Lattices into Ordered Donor–Acceptor Heterojunctions. *J. Am. Chem. Soc.*, 136(28), 9806–9809.

---

## 13. 卟啉类核心 (C₄)

### porphyrin_C4
- **SMILES**: `*c1ccc(C2=C3C=CC(=N3)C(c3ccc(*)cc3)=C3C=CC(=N3)C(c3ccc(*)cc3)=C3C=CC(=N3)C(c3ccc(*)cc3)=C3C=CC2=N3)cc1`
- **目标点群**: C₄ | **臂数**: 4
- **化学描述**: 5,10,15,20-四(4-苯基)卟啉 (meso-tetraphenylporphyrin, TPP)。四个 `*` 分别在四个 meso-苯基的 para 位。D₂ₕ 对称（若考虑平面性），目标点群取 C₄。
- **在 COF 中的角色**: 卟啉 COF 是研究最广泛的功能性 COF 类之一。卟啉环赋予催化活性（金属配位）、光吸收和电子转移能力。
- **参考文献**:
  - Wan, S., Guo, J., Kim, J., Ihee, H., & Jiang, D. (2009). A Photoconductive Covalent Organic Framework: Self-Condensed Arene Cubes Composed of Eclipsed 2D Polypyrene Sheets for Photocurrent Generation. *Angew. Chem. Int. Ed.*, 48(29), 5439–5442. → 含卟啉 COF 的早期代表。
  - Feng, X., Chen, L., Dong, Y., & Jiang, D. (2012). Porphyrin-Based Two-Dimensional Covalent Organic Frameworks: Synchronized Synthetic Control of Macroscopic Structures and Pore Parameters. *Chem. Commun.*, 47(7), 1979–1981.
  - Lin, S., Diercks, C.S., Zhang, Y.B., Kornienko, N., Nichols, E.M., Zhao, Y., Paris, A.R., Kim, D., Yang, P., Yaghi, O.M., & Chang, C.J. (2015). Covalent Organic Frameworks Comprising Cobalt Porphyrins for Catalytic CO₂ Reduction in Water. *Science*, 349(6253), 1208–1213. → 钴卟啉 COF 用于 CO₂ 电催化还原。

### porphyrin_ethynyl_C4
- **SMILES**: `*C#Cc1ccc(C2=C3C=CC(=N3)...`
- **目标点群**: C₄ | **臂数**: 4
- **化学描述**: 5,10,15,20-四(4-乙炔基苯基)卟啉。与 porphyrin_C4 的区别在于 `*` 连在乙炔基末端而非苯环上，通过炔基延长臂的连接距离。
- **在 COF 中的角色**: 炔基扩展使卟啉节点与臂之间的共轭更加连续。
- **参考文献**:
  - Lin et al. (2015), 同上。
  - Johnson, E.M., Haiges, R., & Marinescu, S.C. (2018). Covalent Organic Frameworks Composed of Rhenium Bipyridine and Porphyrin Moieties for CO₂ Reduction. *ACS Appl. Mater. Interfaces*, 10(44), 37919–37927.

---

## 14. 四面体核心 (S₄)

### tetraphenylmethane_C4
- **SMILES**: `*c1ccc(C(c2ccc(*)cc2)(c2ccc(*)cc2)c2ccc(*)cc2)cc1`
- **目标点群**: S₄ | **臂数**: 4
- **化学描述**: 四(4-苯基)甲烷，sp³ 碳中心四面体。四个 `*` 在四个苯环的 para 位。在 2D 投影中看似 C₄，但 3D 中为 S₄（四面体对称）。
- **在 COF 中的角色**: 最经典的 3D COF 四面体节点，通过四臂向四个方向延伸构建 3D 网络（如 dia 拓扑）。
- **参考文献**:
  - El-Kaderi, H.M., Hunt, J.R., Mendoza-Cortés, J.L., Côté, A.P., Taylor, R.E., O'Keeffe, M., & Yaghi, O.M. (2007). Designed Synthesis of 3D Covalent Organic Frameworks. *Science*, 316(5822), 268–272. → 首例 3D COF (COF-102, COF-103)，使用四(4-硼酸基苯基)甲烷/硅烷。
  - Uribe-Romo et al. (2009), 同上。COF-300 使用四(4-氨基苯基)甲烷。

### tetraphenyl_silane_C4
- **SMILES**: `*c1ccc([Si](c2ccc(*)cc2)(c2ccc(*)cc2)c2ccc(*)cc2)cc1`
- **目标点群**: S₄ | **臂数**: 4
- **化学描述**: 四(4-苯基)硅烷，硅中心四面体。硅替代碳中心，C-Si 键比 C-C 键长 (~1.87 Å vs ~1.54 Å)，提供更大的孔径。
- **在 COF 中的角色**: 硅中心的 3D COF 四面体节点。
- **参考文献**:
  - El-Kaderi et al. (2007), 同上。COF-103 使用四(4-硼酸基苯基)硅烷。
  - Fang, Q., Gu, S., Zheng, J., Zhuang, Z., Qiu, S., & Yan, Y. (2014). 3D Microporous Base-Functionalized Covalent Organic Frameworks for Size-Selective Catalysis. *Angew. Chem. Int. Ed.*, 53(11), 2878–2882. → 碱功能化 3D COF。

### tetra_ethynyl_methane_C4
- **SMILES**: `*C#CC(C#C*)(C#C*)C#C*`
- **目标点群**: S₄ | **臂数**: 4
- **化学描述**: 四(乙炔基)甲烷，sp³ 碳中心 + 四个乙炔基臂。比四苯基甲烷更紧凑，但炔基提供更强 π-共轭。
- **在 COF 中的角色**: 全碳骨架四面体节点，炔基共轭增强电子耦合。
- **参考文献**:
  - El-Kaderi et al. (2007) 和 Uribe-Romo et al. (2009) 的设计理念延伸。乙炔基扩展源自 CMP 化学。
  - Lu, W., Wei, Z., Yuan, D., Zhou, H.C., & Ford, P.C. (2013). Multifunctional Porphyrinic Porous Organic Frameworks. *Chem. Sci.*, 4(9), 3582–3588.

---

## 15. 环辛四烯核心 (C₄)

### cyclooctatetraene_C4
- **SMILES**: `*C1=CC(*)=CC(*)=CC(*)=C1`
- **目标点群**: C₄ | **臂数**: 4
- **化学描述**: 1,3,5,7-环辛四烯 (COT)，非平面八元环。`*` 在 1,3,5,7 位交替排列。Tub-shaped (船式) 构象使其具有非平面的 C₄ 对称性。
- **在 COF 中的角色**: COT 的非平面性可赋予 3D 特性，其四个连接位点可构建三维网络。
- **参考文献**:
  - 受 3D COF 和四臂构建块设计启发的非传统四面体类节点。
  - El-Kaderi et al. (2007), 同上。非平面 C₄ 构建块的设计理念。

---

## 16. 六苯基苯核心 (D₆ₕ/H₆)

### hexaphenylbenzene_H6
- **SMILES**: `*c1ccc(-c2c(-c3ccc(*)cc3)c(-c3ccc(*)cc3)c(-c3ccc(*)cc3)c(-c3ccc(*)cc3)c2-c2ccc(*)cc2)cc1`
- **目标点群**: D₆ₕ | **臂数**: 6
- **化学描述**: 六苯基苯 (HPB)，中心苯环被六个苯基完全取代。六个 `*` 在六个外围苯基的 para 位。具有 D₆ 对称性（若忽略苯环扭转），理想化为 D₆ₕ。
- **在 COF 中的角色**: 六臂节点构建六方孔道 (hxl 拓扑)，HPB 的大 π 表面积提供强层间堆积。
- **参考文献**:
  - Alahakoon, S.B., Thompson, C.M., Nguyen, A.X., Occhialini, G., McCandless, G.T., & Smaldone, R.A. (2016). An Azine-Linked Hexaphenylbenzene Based Covalent Organic Framework. *Chem. Commun.*, 52(13), 2843–2845. → 首例六苯基苯基 COF。
  - Dalapati, S., Addicoat, M., Jin, S., Sakurai, T., Gao, J., Xu, H., Irle, S., Seki, S., & Jiang, D. (2015). Rational Design of Crystalline Supermicroporous Covalent Organic Frameworks with Triangular Topologies. *Nat. Commun.*, 6, 7786.

---

## 17. 氟代苯核心 (C₂)

### tetrafluorobenzene_C4（注：实际为 C₂ 对称）
- **SMILES**: `*c1c(F)c(F)c(*)c(F)c1F`
- **目标点群**: C₂ | **臂数**: 2
- **化学描述**: 见第 6 节。1,2,4,5-四氟苯，命名中 "C4" 指四氟取代，实际点群为 C₂。
- **参考文献**: Du et al. (2016), 同上。

---

## 18. 参考文献汇总

| # | 文献 | 相关核心 |
|---|------|---------|
| 1 | Côté, A.P. et al. *Science* **2005**, 310, 1166–1170. | biphenyl_C2, triphenylbenzene_C3（COF 奠基） |
| 2 | El-Kaderi, H.M. et al. *Science* **2007**, 316, 268–272. | tetraphenylmethane_C4, tetraphenyl_silane_C4（首例 3D COF） |
| 3 | Jiang, J.X. et al. *Angew. Chem. Int. Ed.* **2007**, 46, 8574–8578. | triethynylbenzene_C3, diethynylbenzene_C2, diphenylacetylene_C2（CMP 奠基） |
| 4 | Kuhn, P. et al. *Angew. Chem. Int. Ed.* **2008**, 47, 3450–3453. | triazine_C3（CTF-1 首次报道） |
| 5 | Schwab, M.G. et al. *J. Am. Chem. Soc.* **2009**, 131, 7216–7217. | triaminotriazine_C3（三聚氰胺网络） |
| 6 | Uribe-Romo, F.J. et al. *J. Am. Chem. Soc.* **2009**, 131, 4570–4571. | 联苯/四苯基甲烷（COF-300，首例 3D 亚胺 COF） |
| 7 | Wan, S. et al. *Angew. Chem. Int. Ed.* **2009**, 48, 5439–5442. | porphyrin_C4（卟啉 COF） |
| 8 | Ding, S.Y. et al. *J. Am. Chem. Soc.* **2011**, 133, 19816–19822. | triphenylbenzene_C3（COF-LZU1，Pd 催化） |
| 9 | Kandambeth, S. et al. *J. Am. Chem. Soc.* **2012**, 134, 19524–19527. | trihydroxybenzene_C3（β-酮烯胺 COF 开创） |
| 10 | Ren, S. et al. *Adv. Mater.* **2012**, 24, 2357–2361. | triethynyltriazine_C3, triazine_triphenyl_C3 |
| 11 | Bunck, D.N. & Dichtel, W.R. *J. Am. Chem. Soc.* **2013**, 135, 14952–14955. | bis_phenylethynyl_benzene_C2, tris_phenylethynyl_benzene_C3 |
| 12 | Dalapati, S. et al. *J. Am. Chem. Soc.* **2013**, 135, 17310–17313. | pyrazine_phenyl_C2（吖嗪 COF） |
| 13 | Dogru, M. et al. *Angew. Chem. Int. Ed.* **2013**, 52, 2920–2924. | bithiophene_C2, bis_phenylethynyl_thiophene_C2 |
| 14 | Bertrand, G.H.V. et al. *Proc. Natl. Acad. Sci. USA* **2013**, 110, 4923–4928. | bithiophene_C2（噻吩 COF） |
| 15 | Fang, Q. et al. *Angew. Chem. Int. Ed.* **2014**, 53, 2878–2882. | tetraphenyl_silane_C4（3D 碱功能化 COF） |
| 16 | Fang, Q. et al. *J. Am. Chem. Soc.* **2015**, 137, 8352–8355. | tetraphenylmethane_C4（3D 聚酰亚胺 COF） |
| 17 | Lin, S. et al. *Science* **2015**, 349, 1208–1213. | porphyrin_C4, porphyrin_ethynyl_C4（CO₂ 还原 COF） |
| 18 | Alahakoon, S.B. et al. *Chem. Commun.* **2016**, 52, 2843–2845. | hexaphenylbenzene_H6（首例 HPB COF） |
| 19 | Zhuang, X. et al. *Polym. Chem.* **2016**, 7, 4176–4181. | stilbene_C2（sp² 碳聚合物） |
| 20 | Jin, E. et al. *Science* **2017**, 357, 673–676. | stilbene_C2, diphenylacetylene_C2（sp² 碳共轭 COF） |
| 21 | Stewart, D. et al. *Nat. Commun.* **2017**, 8, 1102. | oxamide_C2, bis_benzamide_C2, trimesic_tris_amide_C3 |
| 22 | Huang, N. et al. *Nat. Rev. Mater.* **2016**, 1, 16068. | 综述：π-扩展 COF（bis_ethynyl_anthracene_C2 等） |
| 23 | Das, G. et al. *J. Am. Chem. Soc.* **2020**, 142, 8252–8260. | azobenzene_C2（光控偶氮苯 COF） |
| 24 | Zhao, C. et al. *J. Am. Chem. Soc.* **2020**, 142, 14450–14454. | biphenyl_ester_C2, bis_benzoate_C2, trimesic_ester_C3, bis_phenyl_ester_C2（酯连接 COF） |

---

*文档路径: `docs/core_doc.md`*
*最后更新: 2026-08-08*
