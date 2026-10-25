import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1117_block_000 : List Nat := [1201, 1327, 1429]
set_option maxRecDepth 2048 in
theorem smallRow1117_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1117 = smallRow1117_block_000 := by decide

def smallRow1117_block_001 : List Nat := [2269, 2341, 2731, 2797, 2851, 3067, 3229, 3343, 3511, 3637, 3823]
set_option maxRecDepth 2048 in
theorem smallRow1117_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1117 = smallRow1117_block_001 := by decide

def smallRow1117_block_002 : List Nat := [4357, 4363, 4561, 4993, 5077, 5167, 5323, 5869, 6133]
set_option maxRecDepth 2048 in
theorem smallRow1117_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1117 = smallRow1117_block_002 := by decide

def smallRow1117_block_003 : List Nat := [6211, 6637, 6679, 7591]
set_option maxRecDepth 2048 in
theorem smallRow1117_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1117 = smallRow1117_block_003 := by decide

def smallRow1117_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1117_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1117 = smallRow1117_block_004 := by decide

def smallRow1117_small_left : List Nat := smallRow1117_block_000 ++ smallRow1117_block_001
theorem smallRow1117_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1117 = smallRow1117_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1117 ++ upperNeighbors PrimePairVertices.tree_block_001 1117 = _
  rw [smallRow1117_block_000_checked, smallRow1117_block_001_checked]
  rfl

def smallRow1117_small_right_right : List Nat := smallRow1117_block_003 ++ smallRow1117_block_004
theorem smallRow1117_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1117 = smallRow1117_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1117 ++ upperNeighbors PrimePairVertices.tree_block_004 1117 = _
  rw [smallRow1117_block_003_checked, smallRow1117_block_004_checked]
  rfl

def smallRow1117_small_right : List Nat := smallRow1117_block_002 ++ smallRow1117_small_right_right
theorem smallRow1117_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1117 = smallRow1117_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1117 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1117 = _
  rw [smallRow1117_block_002_checked, smallRow1117_small_right_right_checked]
  rfl

def smallRow1117_small : List Nat := smallRow1117_small_left ++ smallRow1117_small_right
theorem smallRow1117_small_checked : upperNeighbors PrimePairVertices.tree_small 1117 = smallRow1117_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1117 ++ upperNeighbors PrimePairVertices.tree_small_right 1117 = _
  rw [smallRow1117_small_left_checked, smallRow1117_small_right_checked]
  rfl

def smallRow1117 : List Nat := smallRow1117_small
theorem smallRow1117_checked : upperNeighbors PrimePairVertices.smallTree 1117 = smallRow1117 := smallRow1117_small_checked

def smallRow1123_block_000 : List Nat := [1201, 1399, 1453, 1657, 1783, 2029]
set_option maxRecDepth 2048 in
theorem smallRow1123_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1123 = smallRow1123_block_000 := by decide

def smallRow1123_block_001 : List Nat := [2203, 2281, 3373, 3433, 3919]
set_option maxRecDepth 2048 in
theorem smallRow1123_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1123 = smallRow1123_block_001 := by decide

def smallRow1123_block_002 : List Nat := [4423, 5281, 5347, 5527, 5641, 5737, 6037]
set_option maxRecDepth 2048 in
theorem smallRow1123_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1123 = smallRow1123_block_002 := by decide

def smallRow1123_block_003 : List Nat := [7489]
set_option maxRecDepth 2048 in
theorem smallRow1123_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1123 = smallRow1123_block_003 := by decide

def smallRow1123_block_004 : List Nat := [8221]
set_option maxRecDepth 2048 in
theorem smallRow1123_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1123 = smallRow1123_block_004 := by decide

def smallRow1123_small_left : List Nat := smallRow1123_block_000 ++ smallRow1123_block_001
theorem smallRow1123_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1123 = smallRow1123_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1123 ++ upperNeighbors PrimePairVertices.tree_block_001 1123 = _
  rw [smallRow1123_block_000_checked, smallRow1123_block_001_checked]
  rfl

def smallRow1123_small_right_right : List Nat := smallRow1123_block_003 ++ smallRow1123_block_004
theorem smallRow1123_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1123 = smallRow1123_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1123 ++ upperNeighbors PrimePairVertices.tree_block_004 1123 = _
  rw [smallRow1123_block_003_checked, smallRow1123_block_004_checked]
  rfl

def smallRow1123_small_right : List Nat := smallRow1123_block_002 ++ smallRow1123_small_right_right
theorem smallRow1123_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1123 = smallRow1123_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1123 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1123 = _
  rw [smallRow1123_block_002_checked, smallRow1123_small_right_right_checked]
  rfl

def smallRow1123_small : List Nat := smallRow1123_small_left ++ smallRow1123_small_right
theorem smallRow1123_small_checked : upperNeighbors PrimePairVertices.tree_small 1123 = smallRow1123_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1123 ++ upperNeighbors PrimePairVertices.tree_small_right 1123 = _
  rw [smallRow1123_small_left_checked, smallRow1123_small_right_checked]
  rfl

def smallRow1123 : List Nat := smallRow1123_small
theorem smallRow1123_checked : upperNeighbors PrimePairVertices.smallTree 1123 = smallRow1123 := smallRow1123_small_checked

def smallRow1129_block_000 : List Nat := [1741, 1999]
set_option maxRecDepth 2048 in
theorem smallRow1129_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1129 = smallRow1129_block_000 := by decide

def smallRow1129_block_001 : List Nat := [2143, 2269, 2473, 2557, 2647, 2767, 3001]
set_option maxRecDepth 2048 in
theorem smallRow1129_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1129 = smallRow1129_block_001 := by decide

def smallRow1129_block_002 : List Nat := [4561, 5107, 5923]
set_option maxRecDepth 2048 in
theorem smallRow1129_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1129 = smallRow1129_block_002 := by decide

def smallRow1129_block_003 : List Nat := [6421, 6451, 6829, 6871, 7057, 7417, 7603, 7621]
set_option maxRecDepth 2048 in
theorem smallRow1129_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1129 = smallRow1129_block_003 := by decide

def smallRow1129_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow1129_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1129 = smallRow1129_block_004 := by decide

def smallRow1129_small_left : List Nat := smallRow1129_block_000 ++ smallRow1129_block_001
theorem smallRow1129_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1129 = smallRow1129_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1129 ++ upperNeighbors PrimePairVertices.tree_block_001 1129 = _
  rw [smallRow1129_block_000_checked, smallRow1129_block_001_checked]
  rfl

def smallRow1129_small_right_right : List Nat := smallRow1129_block_003 ++ smallRow1129_block_004
theorem smallRow1129_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1129 = smallRow1129_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1129 ++ upperNeighbors PrimePairVertices.tree_block_004 1129 = _
  rw [smallRow1129_block_003_checked, smallRow1129_block_004_checked]
  rfl

def smallRow1129_small_right : List Nat := smallRow1129_block_002 ++ smallRow1129_small_right_right
theorem smallRow1129_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1129 = smallRow1129_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1129 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1129 = _
  rw [smallRow1129_block_002_checked, smallRow1129_small_right_right_checked]
  rfl

def smallRow1129_small : List Nat := smallRow1129_small_left ++ smallRow1129_small_right
theorem smallRow1129_small_checked : upperNeighbors PrimePairVertices.tree_small 1129 = smallRow1129_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1129 ++ upperNeighbors PrimePairVertices.tree_small_right 1129 = _
  rw [smallRow1129_small_left_checked, smallRow1129_small_right_checked]
  rfl

def smallRow1129 : List Nat := smallRow1129_small
theorem smallRow1129_checked : upperNeighbors PrimePairVertices.smallTree 1129 = smallRow1129 := smallRow1129_small_checked

def smallRow1151_block_000 : List Nat := [1217, 1277, 1361, 1433, 1481, 1559, 1607, 1823]
set_option maxRecDepth 2048 in
theorem smallRow1151_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1151 = smallRow1151_block_000 := by decide

def smallRow1151_block_001 : List Nat := [2399, 2789, 3371, 3527, 4001]
set_option maxRecDepth 2048 in
theorem smallRow1151_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1151 = smallRow1151_block_001 := by decide

def smallRow1151_block_002 : List Nat := [4127, 4337, 4871, 5333, 5417, 5897, 6101]
set_option maxRecDepth 2048 in
theorem smallRow1151_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1151 = smallRow1151_block_002 := by decide

def smallRow1151_block_003 : List Nat := [6473, 6521, 6863, 7019, 7577]
set_option maxRecDepth 2048 in
theorem smallRow1151_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1151 = smallRow1151_block_003 := by decide

def smallRow1151_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1151_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1151 = smallRow1151_block_004 := by decide

def smallRow1151_small_left : List Nat := smallRow1151_block_000 ++ smallRow1151_block_001
theorem smallRow1151_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1151 = smallRow1151_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1151 ++ upperNeighbors PrimePairVertices.tree_block_001 1151 = _
  rw [smallRow1151_block_000_checked, smallRow1151_block_001_checked]
  rfl

def smallRow1151_small_right_right : List Nat := smallRow1151_block_003 ++ smallRow1151_block_004
theorem smallRow1151_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1151 = smallRow1151_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1151 ++ upperNeighbors PrimePairVertices.tree_block_004 1151 = _
  rw [smallRow1151_block_003_checked, smallRow1151_block_004_checked]
  rfl

def smallRow1151_small_right : List Nat := smallRow1151_block_002 ++ smallRow1151_small_right_right
theorem smallRow1151_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1151 = smallRow1151_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1151 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1151 = _
  rw [smallRow1151_block_002_checked, smallRow1151_small_right_right_checked]
  rfl

def smallRow1151_small : List Nat := smallRow1151_small_left ++ smallRow1151_small_right
theorem smallRow1151_small_checked : upperNeighbors PrimePairVertices.tree_small 1151 = smallRow1151_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1151 ++ upperNeighbors PrimePairVertices.tree_small_right 1151 = _
  rw [smallRow1151_small_left_checked, smallRow1151_small_right_checked]
  rfl

def smallRow1151 : List Nat := smallRow1151_small
theorem smallRow1151_checked : upperNeighbors PrimePairVertices.smallTree 1151 = smallRow1151 := smallRow1151_small_checked

def smallRow1153_block_000 : List Nat := [1231, 1543, 1699, 1759, 1801]
set_option maxRecDepth 2048 in
theorem smallRow1153_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1153 = smallRow1153_block_000 := by decide

def smallRow1153_block_001 : List Nat := [2383, 2473, 2803, 3163, 3253, 3673, 3853, 3919, 3943, 4093]
set_option maxRecDepth 2048 in
theorem smallRow1153_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1153 = smallRow1153_block_001 := by decide

def smallRow1153_block_002 : List Nat := [4423, 4597, 4951, 5179, 5437, 6037, 6067]
set_option maxRecDepth 2048 in
theorem smallRow1153_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1153 = smallRow1153_block_002 := by decide

def smallRow1153_block_003 : List Nat := [6361, 6673, 6691, 7243, 7507, 7573, 7681, 7741, 7993, 8017]
set_option maxRecDepth 2048 in
theorem smallRow1153_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1153 = smallRow1153_block_003 := by decide

def smallRow1153_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1153_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1153 = smallRow1153_block_004 := by decide

def smallRow1153_small_left : List Nat := smallRow1153_block_000 ++ smallRow1153_block_001
theorem smallRow1153_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1153 = smallRow1153_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1153 ++ upperNeighbors PrimePairVertices.tree_block_001 1153 = _
  rw [smallRow1153_block_000_checked, smallRow1153_block_001_checked]
  rfl

def smallRow1153_small_right_right : List Nat := smallRow1153_block_003 ++ smallRow1153_block_004
theorem smallRow1153_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1153 = smallRow1153_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1153 ++ upperNeighbors PrimePairVertices.tree_block_004 1153 = _
  rw [smallRow1153_block_003_checked, smallRow1153_block_004_checked]
  rfl

def smallRow1153_small_right : List Nat := smallRow1153_block_002 ++ smallRow1153_small_right_right
theorem smallRow1153_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1153 = smallRow1153_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1153 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1153 = _
  rw [smallRow1153_block_002_checked, smallRow1153_small_right_right_checked]
  rfl

def smallRow1153_small : List Nat := smallRow1153_small_left ++ smallRow1153_small_right
theorem smallRow1153_small_checked : upperNeighbors PrimePairVertices.tree_small 1153 = smallRow1153_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1153 ++ upperNeighbors PrimePairVertices.tree_small_right 1153 = _
  rw [smallRow1153_small_left_checked, smallRow1153_small_right_checked]
  rfl

def smallRow1153 : List Nat := smallRow1153_small
theorem smallRow1153_checked : upperNeighbors PrimePairVertices.smallTree 1153 = smallRow1153 := smallRow1153_small_checked

def smallRow1163_block_000 : List Nat := [1733, 1877]
set_option maxRecDepth 2048 in
theorem smallRow1163_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1163 = smallRow1163_block_000 := by decide

def smallRow1163_block_001 : List Nat := [2111, 2417, 2447, 2927]
set_option maxRecDepth 2048 in
theorem smallRow1163_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1163 = smallRow1163_block_001 := by decide

def smallRow1163_block_002 : List Nat := [4547, 4673, 4967, 5639, 5711, 5849]
set_option maxRecDepth 2048 in
theorem smallRow1163_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1163 = smallRow1163_block_002 := by decide

def smallRow1163_block_003 : List Nat := [6197, 6803, 6899, 7211, 7349, 7853, 7919]
set_option maxRecDepth 2048 in
theorem smallRow1163_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1163 = smallRow1163_block_003 := by decide

def smallRow1163_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow1163_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1163 = smallRow1163_block_004 := by decide

def smallRow1163_small_left : List Nat := smallRow1163_block_000 ++ smallRow1163_block_001
theorem smallRow1163_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1163 = smallRow1163_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1163 ++ upperNeighbors PrimePairVertices.tree_block_001 1163 = _
  rw [smallRow1163_block_000_checked, smallRow1163_block_001_checked]
  rfl

def smallRow1163_small_right_right : List Nat := smallRow1163_block_003 ++ smallRow1163_block_004
theorem smallRow1163_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1163 = smallRow1163_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1163 ++ upperNeighbors PrimePairVertices.tree_block_004 1163 = _
  rw [smallRow1163_block_003_checked, smallRow1163_block_004_checked]
  rfl

def smallRow1163_small_right : List Nat := smallRow1163_block_002 ++ smallRow1163_small_right_right
theorem smallRow1163_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1163 = smallRow1163_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1163 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1163 = _
  rw [smallRow1163_block_002_checked, smallRow1163_small_right_right_checked]
  rfl

def smallRow1163_small : List Nat := smallRow1163_small_left ++ smallRow1163_small_right
theorem smallRow1163_small_checked : upperNeighbors PrimePairVertices.tree_small 1163 = smallRow1163_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1163 ++ upperNeighbors PrimePairVertices.tree_small_right 1163 = _
  rw [smallRow1163_small_left_checked, smallRow1163_small_right_checked]
  rfl

def smallRow1163 : List Nat := smallRow1163_small
theorem smallRow1163_checked : upperNeighbors PrimePairVertices.smallTree 1163 = smallRow1163 := smallRow1163_small_checked

def smallRow1171_block_000 : List Nat := [1303, 1699, 2011]
set_option maxRecDepth 2048 in
theorem smallRow1171_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1171 = smallRow1171_block_000 := by decide

def smallRow1171_block_001 : List Nat := [2053, 2137, 2161, 3607]
set_option maxRecDepth 2048 in
theorem smallRow1171_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1171 = smallRow1171_block_001 := by decide

def smallRow1171_block_002 : List Nat := [4663, 4933, 4951, 5881, 6007]
set_option maxRecDepth 2048 in
theorem smallRow1171_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1171 = smallRow1171_block_002 := by decide

def smallRow1171_block_003 : List Nat := [6337, 7873, 7963]
set_option maxRecDepth 2048 in
theorem smallRow1171_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1171 = smallRow1171_block_003 := by decide

def smallRow1171_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1171_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1171 = smallRow1171_block_004 := by decide

def smallRow1171_small_left : List Nat := smallRow1171_block_000 ++ smallRow1171_block_001
theorem smallRow1171_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1171 = smallRow1171_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1171 ++ upperNeighbors PrimePairVertices.tree_block_001 1171 = _
  rw [smallRow1171_block_000_checked, smallRow1171_block_001_checked]
  rfl

def smallRow1171_small_right_right : List Nat := smallRow1171_block_003 ++ smallRow1171_block_004
theorem smallRow1171_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1171 = smallRow1171_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1171 ++ upperNeighbors PrimePairVertices.tree_block_004 1171 = _
  rw [smallRow1171_block_003_checked, smallRow1171_block_004_checked]
  rfl

def smallRow1171_small_right : List Nat := smallRow1171_block_002 ++ smallRow1171_small_right_right
theorem smallRow1171_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1171 = smallRow1171_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1171 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1171 = _
  rw [smallRow1171_block_002_checked, smallRow1171_small_right_right_checked]
  rfl

def smallRow1171_small : List Nat := smallRow1171_small_left ++ smallRow1171_small_right
theorem smallRow1171_small_checked : upperNeighbors PrimePairVertices.tree_small 1171 = smallRow1171_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1171 ++ upperNeighbors PrimePairVertices.tree_small_right 1171 = _
  rw [smallRow1171_small_left_checked, smallRow1171_small_right_checked]
  rfl

def smallRow1171 : List Nat := smallRow1171_small
theorem smallRow1171_checked : upperNeighbors PrimePairVertices.smallTree 1171 = smallRow1171 := smallRow1171_small_checked

def smallRow1181_block_000 : List Nat := [1361, 1367, 1427, 1451, 1601, 1847]
set_option maxRecDepth 2048 in
theorem smallRow1181_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1181 = smallRow1181_block_000 := by decide

def smallRow1181_block_001 : List Nat := [2621, 2687, 2729, 3083, 3209, 3251, 3797, 4013]
set_option maxRecDepth 2048 in
theorem smallRow1181_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1181 = smallRow1181_block_001 := by decide

def smallRow1181_block_002 : List Nat := [4889, 4931, 5171, 5501, 5717]
set_option maxRecDepth 2048 in
theorem smallRow1181_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1181 = smallRow1181_block_002 := by decide

def smallRow1181_block_003 : List Nat := [6449, 6719, 6911, 6977, 7433, 7577, 8069]
set_option maxRecDepth 2048 in
theorem smallRow1181_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1181 = smallRow1181_block_003 := by decide

def smallRow1181_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow1181_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1181 = smallRow1181_block_004 := by decide

def smallRow1181_small_left : List Nat := smallRow1181_block_000 ++ smallRow1181_block_001
theorem smallRow1181_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1181 = smallRow1181_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1181 ++ upperNeighbors PrimePairVertices.tree_block_001 1181 = _
  rw [smallRow1181_block_000_checked, smallRow1181_block_001_checked]
  rfl

def smallRow1181_small_right_right : List Nat := smallRow1181_block_003 ++ smallRow1181_block_004
theorem smallRow1181_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1181 = smallRow1181_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1181 ++ upperNeighbors PrimePairVertices.tree_block_004 1181 = _
  rw [smallRow1181_block_003_checked, smallRow1181_block_004_checked]
  rfl

def smallRow1181_small_right : List Nat := smallRow1181_block_002 ++ smallRow1181_small_right_right
theorem smallRow1181_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1181 = smallRow1181_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1181 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1181 = _
  rw [smallRow1181_block_002_checked, smallRow1181_small_right_right_checked]
  rfl

def smallRow1181_small : List Nat := smallRow1181_small_left ++ smallRow1181_small_right
theorem smallRow1181_small_checked : upperNeighbors PrimePairVertices.tree_small 1181 = smallRow1181_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1181 ++ upperNeighbors PrimePairVertices.tree_small_right 1181 = _
  rw [smallRow1181_small_left_checked, smallRow1181_small_right_checked]
  rfl

def smallRow1181 : List Nat := smallRow1181_small
theorem smallRow1181_checked : upperNeighbors PrimePairVertices.smallTree 1181 = smallRow1181 := smallRow1181_small_checked

end ProjectEulerSolutions.PrimePairCache
