"""训练数据包的稳定 schema。"""

from rdkit import Chem

SCHEMA_VERSION = "1.0"

# 索引从 0 开始；顺序与项目既有数据表示保持一致。
ATOM_SYMBOLS = (
    "H", "C", "N", "O", "F", "B", "P", "S", "Cl", "Br", "I", "Si", "Sn"
)
ATOM_TO_INDEX = {symbol: index for index, symbol in enumerate(ATOM_SYMBOLS)}

# 0 预留给“无键”，NPZ 的边表只保存 1--4。
BOND_TYPE_NAMES = ("none", "single", "double", "triple", "aromatic")
BOND_TYPE_TO_INDEX = {
    Chem.BondType.SINGLE: 1,
    Chem.BondType.DOUBLE: 2,
    Chem.BondType.TRIPLE: 3,
    Chem.BondType.AROMATIC: 4,
}


def atomic_numbers() -> tuple[int, ...]:
    table = Chem.GetPeriodicTable()
    return tuple(table.GetAtomicNumber(symbol) for symbol in ATOM_SYMBOLS)
