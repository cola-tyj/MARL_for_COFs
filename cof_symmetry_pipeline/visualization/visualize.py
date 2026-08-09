#!/usr/bin/env python3
"""
COF 对称分子数据集 — 可视化分析脚本

生成 8 张图，覆盖数据集的全维度统计与分子结构展示。
输出路径: cof_symmetry_pipeline/visualization/

用法:
    python cof_symmetry_pipeline/visualization/visualize.py
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from collections import Counter

# ── 全局样式 ──────────────────────────────────────────────────────────
# 配置中文字体 —— 使用 arphic uming (简体中文, 字形完整)
import matplotlib.font_manager as fm

# 策略：删除 matplotlib 字体缓存 → 添加中文字体 → 设为默认 sans-serif
zh_font_path = '/usr/share/fonts/truetype/arphic/uming.ttc'
if os.path.exists(zh_font_path):
    fm.fontManager.addfont(zh_font_path)
    zh_prop = fm.FontProperties(fname=zh_font_path)
    zh_family = zh_prop.get_name()
    plt.rcParams.update({
        'font.sans-serif': [zh_family, 'DejaVu Sans', 'Arial'],
        'font.family': 'sans-serif',
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
        'axes.unicode_minus': False,
    })
    print(f'中文字体: {zh_family}')
else:
    plt.rcParams.update({
        'font.size': 11, 'axes.titlesize': 13, 'axes.labelsize': 11,
        'figure.dpi': 150, 'savefig.bbox': 'tight', 'savefig.dpi': 150,
    })
    print('未找到中文字体，使用默认字体')

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
CSV_2D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'augmented_dataset.csv')
CSV_3D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'cof_dataset_3d_pass.csv')
XYZ_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output', 'xyz')

# ── 加载数据 ──────────────────────────────────────────────────────────
df_2d = pd.read_csv(CSV_2D)                                                       # 2D 数据集
df_3d = pd.read_csv(CSV_3D)                                                       # 3D 验证数据集
# 统一列名: cof_dataset_3d_pass.csv 使用标准大写列名
df_3d = df_3d.rename(columns={
    'smiles': 'SMILES', 'target_pg': 'Target_PG', 'core': 'Core', 'arm': 'Arm',
    'final_pg': 'Point_Group', 'energy': 'Energy_kcal_mol', 'status': 'Status',
    'xyz_path': 'XYZ_File_Path',
})

pass_df = df_3d.copy()                                                             # 3D 数据 = 全部 PASS
unique_df = pass_df.drop_duplicates(subset='SMILES')                               # 去重独特分子

print(f'数据加载: 2D={len(df_2d)} 分子, 3D PASS={len(pass_df)}, 独特={len(unique_df)}')
print(f'核心数: {df_2d["Core"].nunique()}, 臂数: {df_2d["Arm"].nunique()}')


# ══════════════════════════════════════════════════════════════════════
# Fig 1: 数据集全景仪表盘 (3×2)
# ══════════════════════════════════════════════════════════════════════
def fig1_overview():
    fig = plt.figure(figsize=(20, 13))
    gs = GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.35)

    # (0,0) 点群分布柱状图
    ax = fig.add_subplot(gs[0, 0])
    pg = pass_df['Point_Group'].value_counts()
    colors_pg = plt.cm.tab20(np.linspace(0, 1, len(pg)))
    bars = ax.bar(range(len(pg)), pg.values, color=colors_pg, edgecolor='white', linewidth=0.5)
    ax.set_xticks(range(len(pg)))
    ax.set_xticklabels(pg.index, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('分子数', fontsize=11)
    ax.set_title(f'点群分布 ({len(pg)} 种)', fontsize=13)
    for b, v in zip(bars, pg.values):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 1, str(v), ha='center', fontsize=7)

    # (0,1) 对称族饼图
    ax = fig.add_subplot(gs[0, 1])
    families = {'C2 族': 0, 'C3 族': 0, 'C4/S4 族': 0, 'D6h/C6 族': 0}
    for p in pass_df['Point_Group']:
        if any(x in p for x in ['C2', 'D2']): families['C2 族'] += 1
        elif any(x in p for x in ['C3', 'D3']): families['C3 族'] += 1
        elif any(x in p for x in ['C4', 'S4', 'D4']): families['C4/S4 族'] += 1
        elif any(x in p for x in ['C6', 'D6']): families['D6h/C6 族'] += 1
    fam_colors = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    wedges, texts, autotexts = ax.pie(
        families.values(), labels=families.keys(), autopct='%1.1f%%',
        colors=fam_colors, startangle=90, explode=(0, 0, 0.05, 0.1)
    )
    ax.set_title('对称族分布', fontsize=13)

    # (0,2) 原子数分布直方图
    ax = fig.add_subplot(gs[0, 2])
    n_atoms = []
    for smi in unique_df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        if mol: n_atoms.append(mol.GetNumAtoms())
    ax.hist(n_atoms, bins=35, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(n_atoms), color='red', linestyle='--', linewidth=2,
               label=f'均值: {np.mean(n_atoms):.0f}')
    ax.axvline(np.median(n_atoms), color='orange', linestyle=':', linewidth=2,
               label=f'中位数: {np.median(n_atoms):.0f}')
    ax.set_xlabel('原子数', fontsize=11)
    ax.set_ylabel('分子数', fontsize=11)
    ax.set_title(f'原子数分布 ({len(n_atoms)} 个独特分子)', fontsize=13)
    ax.legend(fontsize=9)

    # (1,0) 核心产出横向柱状图
    ax = fig.add_subplot(gs[1, 0])
    cu = unique_df.groupby('Core').size().sort_values()
    ct = pass_df.groupby('Core').size()
    colors_core = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(cu)))
    ax.barh(range(len(cu)), cu.values, color=colors_core, edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(cu)))
    ax.set_yticklabels([f'{c[:28]}' for c in cu.index], fontsize=7)
    ax.set_xlabel('独特分子数', fontsize=11)
    ax.set_title(f'各核心产出 (共 {len(cu)} 个)', fontsize=13)

    # (1,1) 臂产出 TOP 25
    ax = fig.add_subplot(gs[1, 1])
    au = unique_df.groupby('Arm').size().sort_values(ascending=True)
    top_arms = au.iloc[-25:] if len(au) > 25 else au
    # 按官能团类型着色
    fg_colors = []
    for a in top_arms.index:
        if 'CC' in a and 'direct' in a: fg_colors.append('#E74C3C')      # 乙炔桥
        elif 'direct' in a: fg_colors.append('#3498DB')                    # 直接挂载
        elif '_phph' in a: fg_colors.append('#95A5A6')                     # 联苯
        elif '_ph' in a: fg_colors.append('#2ECC71')                       # 苯基
        else: fg_colors.append('#F39C12')
    ax.barh(range(len(top_arms)), top_arms.values, color=fg_colors, edgecolor='white', linewidth=0.3)
    ax.set_yticks(range(len(top_arms)))
    ax.set_yticklabels([f'{a[:22]}' for a in top_arms.index], fontsize=6.5)
    ax.set_xlabel('独特分子数', fontsize=11)
    ax.set_title(f'臂产出 TOP {len(top_arms)} (共 {len(au)} 种)', fontsize=13)

    # (1,2) 能量/原子直方图
    ax = fig.add_subplot(gs[1, 2])
    epa = []
    for __, r in unique_df.iterrows():
        mol = Chem.MolFromSmiles(r['SMILES'])
        if mol and not np.isnan(r['Energy_kcal_mol']):
            epa.append(r['Energy_kcal_mol'] / mol.GetNumAtoms())
    ax.hist(epa, bins=40, color='mediumseagreen', edgecolor='white', alpha=0.85)
    ax.axvline(np.mean(epa), color='red', linestyle='--', linewidth=2,
               label=f'均值: {np.mean(epa):.2f} kcal/mol/atom')
    ax.set_xlabel('MMFF 能量 / 原子 (kcal/mol)', fontsize=11)
    ax.set_ylabel('分子数', fontsize=11)
    ax.set_title(f'能量/原子分布', fontsize=13)
    ax.legend(fontsize=9)

    fig.suptitle('COF 对称分子数据集 — 全景仪表盘', fontsize=16, fontweight='bold', y=0.99)
    fig.savefig(f'{OUT}/01_overview.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  ✓ 01_overview.png')


# ══════════════════════════════════════════════════════════════════════
# Fig 2: 核心 × 臂 热力图
# ══════════════════════════════════════════════════════════════════════
def fig2_heatmap():
    fig, ax = plt.subplots(figsize=(22, 10))
    hm = pass_df.pivot_table(index='Core', columns='Arm', values='SMILES', aggfunc='count').fillna(0)
    # 排序：总和最大者在前
    row_order = hm.sum(axis=1).sort_values(ascending=False).index
    col_order = hm.sum(axis=0).sort_values(ascending=False).index
    hm2 = hm.loc[row_order, col_order]

    im = ax.imshow(hm2.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)
    ax.set_title(f'核心 × 臂 热力图 ({len(row_order)}核心 × {len(col_order)}臂)', fontsize=14)
    plt.colorbar(im, ax=ax, shrink=0.8, label='分子数')
    fig.tight_layout()
    fig.savefig(f'{OUT}/02_heatmap.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  ✓ 02_heatmap.png')


# ══════════════════════════════════════════════════════════════════════
# Fig 3: 每核心点群堆叠百分比
# ══════════════════════════════════════════════════════════════════════
def fig3_pg_per_core():
    fig, ax = plt.subplots(figsize=(14, 8))
    cp = pass_df.groupby(['Core', 'Point_Group']).size().unstack(fill_value=0)
    cp_pct = cp.div(cp.sum(axis=1), axis=0) * 100
    cp_pct.plot(kind='barh', stacked=True, ax=ax, colormap='tab20', edgecolor='white', linewidth=0.3)
    ax.set_xlabel('占比 (%)', fontsize=11)
    ax.set_title('各核心点群组成 (堆叠百分比)', fontsize=13)
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, title='点群')
    fig.tight_layout()
    fig.savefig(f'{OUT}/03_pg_per_core.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  ✓ 03_pg_per_core.png')


# ══════════════════════════════════════════════════════════════════════
# Fig 4: 代表性分子 2D 结构 (按点群)
# ══════════════════════════════════════════════════════════════════════
def fig4_structures():
    targets = ['C2', 'C2h', 'C2v', 'C3', 'C3h', 'C3v', 'D2', 'D2d',
               'D2h', 'D3', 'D3d', 'D3h', 'S4', 'D6h']
    available = [t for t in targets if t in set(pass_df['Point_Group'])]
    n_show = min(len(available), 16)
    n_cols = 4
    n_rows = math.ceil(n_show / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i, tg in enumerate(available[:n_show]):
        ax = axes[i]
        sub = pass_df[pass_df['Point_Group'] == tg]
        if len(sub) == 0:
            ax.axis('off'); continue
        # 取中间样品（避免边缘案例）
        row = sub.iloc[len(sub) // 2]
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            # 突出显示对称轴信息
            n_atoms = mol.GetNumAtoms()
            core_short = row['Core'][:30]
            arm_short = row['Arm'][:20]
            img = Draw.MolToImage(mol, size=(380, 280))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(f'{tg}  |  {n_atoms} atoms\n{core_short}\n{arm_short}', fontsize=7.5)
        else:
            ax.axis('off')

    for j in range(n_show, len(axes)):
        axes[j].axis('off')

    fig.suptitle('代表性 COF 分子结构 (按点群分组)', fontsize=15, fontweight='bold', y=0.98)
    fig.tight_layout()
    fig.savefig(f'{OUT}/04_structures.png', bbox_inches='tight', dpi=120)
    plt.close()
    print('  ✓ 04_structures.png')


# ══════════════════════════════════════════════════════════════════════
# Fig 5: 核心生产力散点图 (独特分子 vs 总 PASS, 气泡=点群种类)
# ══════════════════════════════════════════════════════════════════════
def fig5_productivity():
    fig, ax = plt.subplots(figsize=(14, 8))
    x = unique_df.groupby('Core').size()
    y = pass_df.groupby('Core').size()
    pg_uniq = pass_df.groupby('Core')['Point_Group'].nunique()

    scatter = ax.scatter(x, y, s=pg_uniq * 60, c=pg_uniq, cmap='viridis',
                         alpha=0.8, edgecolors='black', linewidth=0.5)

    # 标注核心名
    for core in x.index:
        ax.annotate(
            core[:25],
            (x[core], y[core]),
            textcoords='offset points', xytext=(5, 5),
            fontsize=6.5, alpha=0.8,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6)
        )

    ax.set_xlabel('独特分子数 (Unique SMILES)', fontsize=11)
    ax.set_ylabel('总 PASS 条目', fontsize=11)
    ax.set_title('核心生产力: 独特性 vs 产量 (气泡大小 = 点群多样性)', fontsize=13)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('点群种类数', fontsize=10)
    fig.tight_layout()
    fig.savefig(f'{OUT}/05_productivity.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  ✓ 05_productivity.png')


# ══════════════════════════════════════════════════════════════════════
# Fig 6: 臂连接子类型 vs 通过率对比
# ══════════════════════════════════════════════════════════════════════
def fig6_linker_comparison():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左: 按连接子类型统计
    ax = axes[0]
    linker_map = {}
    for arm in unique_df['Arm'].unique():
        if 'CC' in arm and 'direct' in arm: t = '乙炔桥 (*-C≡C-FG)'
        elif 'O' in arm and 'direct' in arm and arm.startswith('O'): t = '氧桥 (*-O-FG)'
        elif 'NH' in arm and 'direct' in arm and arm.startswith('NH'): t = '氮桥 (*-NH-FG)'
        elif '_phph' in arm: t = '联苯基 (*-Ph-Ph-FG)'
        elif '_ph' in arm: t = '苯基 (*-Ph-FG)'
        elif '_direct' in arm or 'direct' in arm: t = '直接挂载 (*-FG)'
        else: t = '其他'
        linker_map[arm] = t

    unique_df_copy = unique_df.copy()
    unique_df_copy['linker_type'] = unique_df_copy['Arm'].map(linker_map)
    linker_counts = unique_df_copy.groupby('linker_type').size().sort_values(ascending=True)
    linker_colors = {'直接挂载 (*-FG)': '#3498DB', '苯基 (*-Ph-FG)': '#2ECC71',
                     '联苯基 (*-Ph-Ph-FG)': '#95A5A6', '乙炔桥 (*-C≡C-FG)': '#E74C3C',
                     '氧桥 (*-O-FG)': '#F39C12', '氮桥 (*-NH-FG)': '#9B59B6'}
    colors = [linker_colors.get(t, '#888888') for t in linker_counts.index]
    ax.barh(range(len(linker_counts)), linker_counts.values, color=colors, edgecolor='white')
    ax.set_yticks(range(len(linker_counts)))
    ax.set_yticklabels(linker_counts.index, fontsize=9)
    ax.set_xlabel('独特分子数', fontsize=11)
    ax.set_title('连接子类型产出', fontsize=13)

    # 右: 官能团族统计
    ax = axes[1]
    fg_families = {
        '卤素 (-F/Cl/Br/I)': ['Cl_direct','Br_direct','I_direct','CF3_direct','Cl_ph','CF3_ph'],
        '醛/酮': ['CHO_direct','CHO_ph','CHO_phph','acetyl_direct','acetyl_ph','CCCHO_direct','CCacetyl_direct'],
        '胺/酰胺': ['NH2_direct','NH2_ph','NH2_phph','CONH2_direct','NHCHO_direct','NHacetyl_direct','CCNH2_direct'],
        '氰/异氰': ['CN_direct','CN_ph','CN_phph','NCO_direct','NCS_direct','NHCN_direct','CCCN_direct','OCN_direct'],
        '羧/酯': ['COOH_direct','COOH_ph','COOH_phph','COOMe_direct','CCCOOH_direct','CCCOOMe_direct'],
        '硝基/叠氮': ['NO2_direct','NO2_ph','NO2_phph','N3_direct','N3_ph','CCNO2_direct'],
        '羟/醚': ['OH_direct','OH_ph','OH_phph','OMe_direct','OMe_ph','CCOH_direct'],
        '硫系': ['SH_direct','SO3H_direct','SO2NH2_direct','SO2Cl_direct','SCN_direct'],
        '炔/烯': ['ethynyl_direct','vinyl_direct','CCH_ph','CCH_phph'],
        '硼/磷/硅': ['BOH2_direct','BOH2_ph','Bpin_direct','PO3H2_direct','POCl2_direct','SiMe3_direct'],
        '酰氯': ['COCl_direct'],
        '甲基': ['methyl_direct','methyl_ph'],
    }
    fg_out = {}
    for fam, arms in fg_families.items():
        fg_out[fam] = len(unique_df[unique_df['Arm'].isin(arms)])
    sorted_fg = sorted(fg_out.items(), key=lambda x: -x[1])
    labels = [x[0] for x in sorted_fg]
    vals = [x[1] for x in sorted_fg]
    fg_colors2 = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    ax.barh(range(len(labels)), vals, color=fg_colors2, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel('独特分子数', fontsize=11)
    ax.set_title('官能团族产出', fontsize=13)

    fig.suptitle('臂类型分析: 连接子 vs 官能团', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(f'{OUT}/06_linker_fg_analysis.png', bbox_inches='tight', dpi=150)
    plt.close()
    print('  ✓ 06_linker_fg_analysis.png')


# ══════════════════════════════════════════════════════════════════════
# Fig 7: 分子结构大网格 (所有核心 × 最短臂的代表)
# ══════════════════════════════════════════════════════════════════════
def fig7_core_gallery():
    # 每个核心选一个代表分子（优先最短臂）
    representatives = []
    for core in sorted(unique_df['Core'].unique()):
        core_mols = unique_df[unique_df['Core'] == core]
        # 优先选直接挂载臂
        direct = core_mols[core_mols['Arm'].str.contains('direct', na=False)]
        if len(direct) > 0:
            # 选原子数最小的（最简结构）
            best_idx = None
            best_n = 9999
            for idx, r in direct.iterrows():
                mol = Chem.MolFromSmiles(r['SMILES'])
                if mol and mol.GetNumAtoms() < best_n:
                    best_n = mol.GetNumAtoms()
                    best_idx = idx
            row = direct.loc[best_idx] if best_idx else direct.iloc[0]
        else:
            row = core_mols.iloc[0]
        representatives.append(row)

    n = len(representatives)
    n_cols = 5
    n_rows = math.ceil(n / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, 4.5 * n_rows))
    axes = axes.flatten()

    for i, row in enumerate(representatives[:len(axes)]):
        ax = axes[i]
        mol = Chem.MolFromSmiles(row['SMILES'])
        if mol:
            img = Draw.MolToImage(mol, size=(380, 280))
            ax.imshow(img)
            ax.axis('off')
            pg_val = row['Point_Group']
            core_val = row['Core'][:32]
            ax.set_title(f'{pg_val} | {core_val}', fontsize=7)
        else:
            ax.axis('off')

    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'COF 对称核心全景图 ({n} 个核心 × 最佳臂)', fontsize=15, fontweight='bold', y=0.99)
    fig.tight_layout()
    fig.savefig(f'{OUT}/07_core_gallery.png', bbox_inches='tight', dpi=120)
    plt.close()
    print('  ✓ 07_core_gallery.png')


# ══════════════════════════════════════════════════════════════════════
# Fig 8: 数据集统计摘要卡
# ══════════════════════════════════════════════════════════════════════
def fig8_summary_card():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis('off')

    n_atoms_list = []
    for smi in unique_df['SMILES']:
        mol = Chem.MolFromSmiles(smi)
        if mol: n_atoms_list.append(mol.GetNumAtoms())

    epa_list = []
    for __, r in unique_df.iterrows():
        mol = Chem.MolFromSmiles(r['SMILES'])
        if mol and not np.isnan(r['Energy_kcal_mol']):
            epa_list.append(r['Energy_kcal_mol'] / mol.GetNumAtoms())

    # 点群族分布
    c2_fam = sum(1 for p in unique_df['Point_Group'] if any(x in p for x in ['C2','D2']))
    c3_fam = sum(1 for p in unique_df['Point_Group'] if any(x in p for x in ['C3','D3']))
    c4_fam = sum(1 for p in unique_df['Point_Group'] if any(x in p for x in ['C4','S4','D4']))
    c6_fam = sum(1 for p in unique_df['Point_Group'] if any(x in p for x in ['C6','D6']))

    # 连接子分布
    direct_arms = sum(1 for a in unique_df['Arm'] if 'direct' in a and 'CC' not in a and not a.startswith('O') and not a.startswith('NH'))
    cc_arms = sum(1 for a in unique_df['Arm'] if 'CC' in a)
    ph_arms = sum(1 for a in unique_df['Arm'] if '_ph' in a and '_phph' not in a)
    phph_arms = sum(1 for a in unique_df['Arm'] if '_phph' in a)
    o_nh_arms = sum(1 for a in unique_df['Arm'] if a.startswith('O') or a.startswith('NH'))

    n_cores_3d = unique_df['Core'].nunique()
    n_arms_3d = unique_df['Arm'].nunique()
    n_xyz = pass_df['XYZ_File_Path'].nunique() if 'XYZ_File_Path' in pass_df.columns else 0

    summary = f"""
    ╔══════════════════════════════════════════════╗
    ║     COF 对称分子数据集 — 统计摘要            ║
    ╠══════════════════════════════════════════════╣
    ║                                              ║
    ║  [数据规模]                                  ║
    ║     ├─ 2D 组合分子       :  {len(df_2d):>5}             ║
    ║     ├─ 3D PASS 分子      :  {len(pass_df):>5}             ║
    ║     ├─ 独特分子 (3D)     :  {len(unique_df):>5}             ║
    ║     └─ XYZ 坐标文件      :  {n_xyz:>5}             ║
    ║                                              ║
    ║  [模板库规模]                                ║
    ║     ├─ 原始核心模板      :    41             ║
    ║     ├─ 层级核心模板      :   414             ║
    ║     ├─ pyCOFBuilder核心  :    56             ║
    ║     ├─ 臂模板            :    82             ║
    ║     ├─ 通过3D验证的核心  :  {n_cores_3d:>5}             ║
    ║     └─ 通过3D验证的臂    :  {n_arms_3d:>5}             ║
    ║                                              ║
    ║  [化学多样性]                                ║
    ║     ├─ 唯一点群类型      :  {unique_df['Point_Group'].nunique():>5}             ║
    ║     ├─ C2 族             :  {c2_fam:>5} ({100*c2_fam/len(unique_df):.0f}%)        ║
    ║     ├─ C3 族             :  {c3_fam:>5} ({100*c3_fam/len(unique_df):.0f}%)        ║
    ║     ├─ C4/S4 族          :  {c4_fam:>5} ({100*c4_fam/len(unique_df):.0f}%)        ║
    ║     └─ D6h/C6 族        :  {c6_fam:>5} ({100*c6_fam/len(unique_df):.0f}%)        ║
    ║                                              ║
    ║  [物理性质]                                  ║
    ║     ├─ 平均原子数        :  {np.mean(n_atoms_list):>5.0f}             ║
    ║     ├─ 原子数范围        :  {min(n_atoms_list):>4} - {max(n_atoms_list):>4}       ║
    ║     ├─ 平均能量          :  {np.mean(epa_list):>5.2f} kcal/mol/atom ║
    ║     └─ 力场后端          :  MMFF94 (RDKit)  ║
    ║                                              ║
    ║  [连接子分布]                                ║
    ║     ├─ 直接挂载 (*-FG)   :  {direct_arms:>5}             ║
    ║     ├─ 乙炔桥 (*-CC-FG)   :  {cc_arms:>5}             ║
    ║     ├─ 苯基 (*-Ph-FG)    :  {ph_arms:>5}             ║
    ║     ├─ 联苯 (*-Ph-Ph-FG) :  {phph_arms:>5}             ║
    ║     └─ 杂原子桥 (O/NH)   :  {o_nh_arms:>5}             ║
    ║                                              ║
    ║  [通过率]                                    ║
    ║     3D PASS / 2D total = {len(pass_df)}/{len(df_2d)}                ║
    ║     通过率 = {100*len(pass_df)/len(df_2d):.1f}%                       ║
    ║                                              ║
    ╚══════════════════════════════════════════════╝
    """

    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=10,
            fontfamily=zh_family, verticalalignment='center',
            horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a1a2e',
                      edgecolor='#16213e', alpha=0.95),
            color='#e0e0e0')

    fig.suptitle('数据集统计摘要', fontsize=14, fontweight='bold', color='#333333')
    fig.tight_layout()
    fig.savefig(f'{OUT}/08_summary_card.png', bbox_inches='tight', dpi=150,
                facecolor='white')
    plt.close()
    print('  ✓ 08_summary_card.png')


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print(f'\n生成可视化图表到 {OUT}/\n')
    fig1_overview()
    fig2_heatmap()
    fig3_pg_per_core()
    fig4_structures()
    fig5_productivity()
    fig6_linker_comparison()
    fig7_core_gallery()
    fig8_summary_card()
    print(f'\n全部完成! 8 张图已保存到 {OUT}/')
