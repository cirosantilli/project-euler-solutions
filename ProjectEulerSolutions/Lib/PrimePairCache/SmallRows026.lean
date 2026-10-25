import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1297_block_000 : List Nat := [1549]
set_option maxRecDepth 2048 in
theorem smallRow1297_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1297 = smallRow1297_block_000 := by decide

def smallRow1297_block_001 : List Nat := [2293, 2671, 3529, 3547]
set_option maxRecDepth 2048 in
theorem smallRow1297_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1297 = smallRow1297_block_001 := by decide

def smallRow1297_block_002 : List Nat := [4813, 5821]
set_option maxRecDepth 2048 in
theorem smallRow1297_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1297 = smallRow1297_block_002 := by decide

def smallRow1297_block_003 : List Nat := [6781, 7603]
set_option maxRecDepth 2048 in
theorem smallRow1297_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1297 = smallRow1297_block_003 := by decide

def smallRow1297_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1297_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1297 = smallRow1297_block_004 := by decide

def smallRow1297_small_left : List Nat := smallRow1297_block_000 ++ smallRow1297_block_001
theorem smallRow1297_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1297 = smallRow1297_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1297 ++ upperNeighbors PrimePairVertices.tree_block_001 1297 = _
  rw [smallRow1297_block_000_checked, smallRow1297_block_001_checked]
  rfl

def smallRow1297_small_right_right : List Nat := smallRow1297_block_003 ++ smallRow1297_block_004
theorem smallRow1297_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1297 = smallRow1297_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1297 ++ upperNeighbors PrimePairVertices.tree_block_004 1297 = _
  rw [smallRow1297_block_003_checked, smallRow1297_block_004_checked]
  rfl

def smallRow1297_small_right : List Nat := smallRow1297_block_002 ++ smallRow1297_small_right_right
theorem smallRow1297_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1297 = smallRow1297_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1297 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1297 = _
  rw [smallRow1297_block_002_checked, smallRow1297_small_right_right_checked]
  rfl

def smallRow1297_small : List Nat := smallRow1297_small_left ++ smallRow1297_small_right
theorem smallRow1297_small_checked : upperNeighbors PrimePairVertices.tree_small 1297 = smallRow1297_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1297 ++ upperNeighbors PrimePairVertices.tree_small_right 1297 = _
  rw [smallRow1297_small_left_checked, smallRow1297_small_right_checked]
  rfl

def smallRow1297 : List Nat := smallRow1297_small
theorem smallRow1297_checked : upperNeighbors PrimePairVertices.smallTree 1297 = smallRow1297 := smallRow1297_small_checked

def smallRow1301_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1301_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1301 = smallRow1301_block_000 := by decide

def smallRow1301_block_001 : List Nat := [2141, 2393, 2579, 2927, 2969, 2999]
set_option maxRecDepth 2048 in
theorem smallRow1301_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1301 = smallRow1301_block_001 := by decide

def smallRow1301_block_002 : List Nat := [4583, 4973, 5003, 5279, 5531, 5669, 6011]
set_option maxRecDepth 2048 in
theorem smallRow1301_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1301 = smallRow1301_block_002 := by decide

def smallRow1301_block_003 : List Nat := [7127, 7283, 7481, 7643, 7757]
set_option maxRecDepth 2048 in
theorem smallRow1301_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1301 = smallRow1301_block_003 := by decide

def smallRow1301_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1301_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1301 = smallRow1301_block_004 := by decide

def smallRow1301_small_left : List Nat := smallRow1301_block_000 ++ smallRow1301_block_001
theorem smallRow1301_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1301 = smallRow1301_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1301 ++ upperNeighbors PrimePairVertices.tree_block_001 1301 = _
  rw [smallRow1301_block_000_checked, smallRow1301_block_001_checked]
  rfl

def smallRow1301_small_right_right : List Nat := smallRow1301_block_003 ++ smallRow1301_block_004
theorem smallRow1301_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1301 = smallRow1301_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1301 ++ upperNeighbors PrimePairVertices.tree_block_004 1301 = _
  rw [smallRow1301_block_003_checked, smallRow1301_block_004_checked]
  rfl

def smallRow1301_small_right : List Nat := smallRow1301_block_002 ++ smallRow1301_small_right_right
theorem smallRow1301_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1301 = smallRow1301_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1301 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1301 = _
  rw [smallRow1301_block_002_checked, smallRow1301_small_right_right_checked]
  rfl

def smallRow1301_small : List Nat := smallRow1301_small_left ++ smallRow1301_small_right
theorem smallRow1301_small_checked : upperNeighbors PrimePairVertices.tree_small 1301 = smallRow1301_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1301 ++ upperNeighbors PrimePairVertices.tree_small_right 1301 = _
  rw [smallRow1301_small_left_checked, smallRow1301_small_right_checked]
  rfl

def smallRow1301 : List Nat := smallRow1301_small
theorem smallRow1301_checked : upperNeighbors PrimePairVertices.smallTree 1301 = smallRow1301 := smallRow1301_small_checked

def smallRow1303_block_000 : List Nat := [1723]
set_option maxRecDepth 2048 in
theorem smallRow1303_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1303 = smallRow1303_block_000 := by decide

def smallRow1303_block_001 : List Nat := [2143, 2857, 2953, 3109, 3469]
set_option maxRecDepth 2048 in
theorem smallRow1303_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1303 = smallRow1303_block_001 := by decide

def smallRow1303_block_002 : List Nat := [4363, 4519, 4561, 4759, 4993, 5023, 5119, 5443, 6043]
set_option maxRecDepth 2048 in
theorem smallRow1303_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1303 = smallRow1303_block_002 := by decide

def smallRow1303_block_003 : List Nat := [6871, 7027, 7207, 7309, 7333, 8101]
set_option maxRecDepth 2048 in
theorem smallRow1303_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1303 = smallRow1303_block_003 := by decide

def smallRow1303_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1303_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1303 = smallRow1303_block_004 := by decide

def smallRow1303_small_left : List Nat := smallRow1303_block_000 ++ smallRow1303_block_001
theorem smallRow1303_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1303 = smallRow1303_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1303 ++ upperNeighbors PrimePairVertices.tree_block_001 1303 = _
  rw [smallRow1303_block_000_checked, smallRow1303_block_001_checked]
  rfl

def smallRow1303_small_right_right : List Nat := smallRow1303_block_003 ++ smallRow1303_block_004
theorem smallRow1303_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1303 = smallRow1303_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1303 ++ upperNeighbors PrimePairVertices.tree_block_004 1303 = _
  rw [smallRow1303_block_003_checked, smallRow1303_block_004_checked]
  rfl

def smallRow1303_small_right : List Nat := smallRow1303_block_002 ++ smallRow1303_small_right_right
theorem smallRow1303_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1303 = smallRow1303_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1303 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1303 = _
  rw [smallRow1303_block_002_checked, smallRow1303_small_right_right_checked]
  rfl

def smallRow1303_small : List Nat := smallRow1303_small_left ++ smallRow1303_small_right
theorem smallRow1303_small_checked : upperNeighbors PrimePairVertices.tree_small 1303 = smallRow1303_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1303 ++ upperNeighbors PrimePairVertices.tree_small_right 1303 = _
  rw [smallRow1303_small_left_checked, smallRow1303_small_right_checked]
  rfl

def smallRow1303 : List Nat := smallRow1303_small
theorem smallRow1303_checked : upperNeighbors PrimePairVertices.smallTree 1303 = smallRow1303 := smallRow1303_small_checked

def smallRow1307_block_000 : List Nat := [1787, 1979]
set_option maxRecDepth 2048 in
theorem smallRow1307_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1307 = smallRow1307_block_000 := by decide

def smallRow1307_block_001 : List Nat := [2357, 2663, 2837, 3023, 3491, 3701, 3719, 3989, 4079]
set_option maxRecDepth 2048 in
theorem smallRow1307_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1307 = smallRow1307_block_001 := by decide

def smallRow1307_block_002 : List Nat := [4133, 4583, 5861, 5927, 5939, 6053]
set_option maxRecDepth 2048 in
theorem smallRow1307_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1307 = smallRow1307_block_002 := by decide

def smallRow1307_block_003 : List Nat := [6863, 7499, 7541, 7583, 7901, 8171]
set_option maxRecDepth 2048 in
theorem smallRow1307_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1307 = smallRow1307_block_003 := by decide

def smallRow1307_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1307_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1307 = smallRow1307_block_004 := by decide

def smallRow1307_small_left : List Nat := smallRow1307_block_000 ++ smallRow1307_block_001
theorem smallRow1307_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1307 = smallRow1307_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1307 ++ upperNeighbors PrimePairVertices.tree_block_001 1307 = _
  rw [smallRow1307_block_000_checked, smallRow1307_block_001_checked]
  rfl

def smallRow1307_small_right_right : List Nat := smallRow1307_block_003 ++ smallRow1307_block_004
theorem smallRow1307_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1307 = smallRow1307_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1307 ++ upperNeighbors PrimePairVertices.tree_block_004 1307 = _
  rw [smallRow1307_block_003_checked, smallRow1307_block_004_checked]
  rfl

def smallRow1307_small_right : List Nat := smallRow1307_block_002 ++ smallRow1307_small_right_right
theorem smallRow1307_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1307 = smallRow1307_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1307 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1307 = _
  rw [smallRow1307_block_002_checked, smallRow1307_small_right_right_checked]
  rfl

def smallRow1307_small : List Nat := smallRow1307_small_left ++ smallRow1307_small_right
theorem smallRow1307_small_checked : upperNeighbors PrimePairVertices.tree_small 1307 = smallRow1307_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1307 ++ upperNeighbors PrimePairVertices.tree_small_right 1307 = _
  rw [smallRow1307_small_left_checked, smallRow1307_small_right_checked]
  rfl

def smallRow1307 : List Nat := smallRow1307_small
theorem smallRow1307_checked : upperNeighbors PrimePairVertices.smallTree 1307 = smallRow1307 := smallRow1307_small_checked

def smallRow1319_block_000 : List Nat := [1361, 1613, 1637, 1901, 1979]
set_option maxRecDepth 2048 in
theorem smallRow1319_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1319 = smallRow1319_block_000 := by decide

def smallRow1319_block_001 : List Nat := [2243, 2789, 3407, 3533, 3623, 3863, 4007]
set_option maxRecDepth 2048 in
theorem smallRow1319_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1319 = smallRow1319_block_001 := by decide

def smallRow1319_block_002 : List Nat := [4133, 4253, 4283, 5843, 5981]
set_option maxRecDepth 2048 in
theorem smallRow1319_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1319 = smallRow1319_block_002 := by decide

def smallRow1319_block_003 : List Nat := [6971, 7649, 8039]
set_option maxRecDepth 2048 in
theorem smallRow1319_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1319 = smallRow1319_block_003 := by decide

def smallRow1319_block_004 : List Nat := [8363]
set_option maxRecDepth 2048 in
theorem smallRow1319_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1319 = smallRow1319_block_004 := by decide

def smallRow1319_small_left : List Nat := smallRow1319_block_000 ++ smallRow1319_block_001
theorem smallRow1319_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1319 = smallRow1319_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1319 ++ upperNeighbors PrimePairVertices.tree_block_001 1319 = _
  rw [smallRow1319_block_000_checked, smallRow1319_block_001_checked]
  rfl

def smallRow1319_small_right_right : List Nat := smallRow1319_block_003 ++ smallRow1319_block_004
theorem smallRow1319_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1319 = smallRow1319_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1319 ++ upperNeighbors PrimePairVertices.tree_block_004 1319 = _
  rw [smallRow1319_block_003_checked, smallRow1319_block_004_checked]
  rfl

def smallRow1319_small_right : List Nat := smallRow1319_block_002 ++ smallRow1319_small_right_right
theorem smallRow1319_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1319 = smallRow1319_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1319 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1319 = _
  rw [smallRow1319_block_002_checked, smallRow1319_small_right_right_checked]
  rfl

def smallRow1319_small : List Nat := smallRow1319_small_left ++ smallRow1319_small_right
theorem smallRow1319_small_checked : upperNeighbors PrimePairVertices.tree_small 1319 = smallRow1319_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1319 ++ upperNeighbors PrimePairVertices.tree_small_right 1319 = _
  rw [smallRow1319_small_left_checked, smallRow1319_small_right_checked]
  rfl

def smallRow1319 : List Nat := smallRow1319_small
theorem smallRow1319_checked : upperNeighbors PrimePairVertices.smallTree 1319 = smallRow1319 := smallRow1319_small_checked

def smallRow1321_block_000 : List Nat := [1381]
set_option maxRecDepth 2048 in
theorem smallRow1321_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1321 = smallRow1321_block_000 := by decide

def smallRow1321_block_001 : List Nat := [2767, 2803, 2917, 3001]
set_option maxRecDepth 2048 in
theorem smallRow1321_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1321 = smallRow1321_block_001 := by decide

def smallRow1321_block_002 : List Nat := [4261, 4273, 4447, 4507, 5581]
set_option maxRecDepth 2048 in
theorem smallRow1321_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1321 = smallRow1321_block_002 := by decide

def smallRow1321_block_003 : List Nat := [6607, 7537, 7759]
set_option maxRecDepth 2048 in
theorem smallRow1321_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1321 = smallRow1321_block_003 := by decide

def smallRow1321_block_004 : List Nat := [8263]
set_option maxRecDepth 2048 in
theorem smallRow1321_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1321 = smallRow1321_block_004 := by decide

def smallRow1321_small_left : List Nat := smallRow1321_block_000 ++ smallRow1321_block_001
theorem smallRow1321_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1321 = smallRow1321_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1321 ++ upperNeighbors PrimePairVertices.tree_block_001 1321 = _
  rw [smallRow1321_block_000_checked, smallRow1321_block_001_checked]
  rfl

def smallRow1321_small_right_right : List Nat := smallRow1321_block_003 ++ smallRow1321_block_004
theorem smallRow1321_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1321 = smallRow1321_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1321 ++ upperNeighbors PrimePairVertices.tree_block_004 1321 = _
  rw [smallRow1321_block_003_checked, smallRow1321_block_004_checked]
  rfl

def smallRow1321_small_right : List Nat := smallRow1321_block_002 ++ smallRow1321_small_right_right
theorem smallRow1321_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1321 = smallRow1321_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1321 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1321 = _
  rw [smallRow1321_block_002_checked, smallRow1321_small_right_right_checked]
  rfl

def smallRow1321_small : List Nat := smallRow1321_small_left ++ smallRow1321_small_right
theorem smallRow1321_small_checked : upperNeighbors PrimePairVertices.tree_small 1321 = smallRow1321_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1321 ++ upperNeighbors PrimePairVertices.tree_small_right 1321 = _
  rw [smallRow1321_small_left_checked, smallRow1321_small_right_checked]
  rfl

def smallRow1321 : List Nat := smallRow1321_small
theorem smallRow1321_checked : upperNeighbors PrimePairVertices.smallTree 1321 = smallRow1321 := smallRow1321_small_checked

def smallRow1327_block_000 : List Nat := [1429, 1471, 1723]
set_option maxRecDepth 2048 in
theorem smallRow1327_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1327 = smallRow1327_block_000 := by decide

def smallRow1327_block_001 : List Nat := [2293, 2551, 3109, 3361, 3739, 3769, 3853]
set_option maxRecDepth 2048 in
theorem smallRow1327_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1327 = smallRow1327_block_001 := by decide

def smallRow1327_block_002 : List Nat := [4357, 4621, 5701]
set_option maxRecDepth 2048 in
theorem smallRow1327_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1327 = smallRow1327_block_002 := by decide

def smallRow1327_block_003 : List Nat := [6163, 6217, 6301, 6619, 7129, 7351, 7717, 7963]
set_option maxRecDepth 2048 in
theorem smallRow1327_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1327 = smallRow1327_block_003 := by decide

def smallRow1327_block_004 : List Nat := [8263]
set_option maxRecDepth 2048 in
theorem smallRow1327_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1327 = smallRow1327_block_004 := by decide

def smallRow1327_small_left : List Nat := smallRow1327_block_000 ++ smallRow1327_block_001
theorem smallRow1327_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1327 = smallRow1327_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1327 ++ upperNeighbors PrimePairVertices.tree_block_001 1327 = _
  rw [smallRow1327_block_000_checked, smallRow1327_block_001_checked]
  rfl

def smallRow1327_small_right_right : List Nat := smallRow1327_block_003 ++ smallRow1327_block_004
theorem smallRow1327_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1327 = smallRow1327_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1327 ++ upperNeighbors PrimePairVertices.tree_block_004 1327 = _
  rw [smallRow1327_block_003_checked, smallRow1327_block_004_checked]
  rfl

def smallRow1327_small_right : List Nat := smallRow1327_block_002 ++ smallRow1327_small_right_right
theorem smallRow1327_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1327 = smallRow1327_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1327 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1327 = _
  rw [smallRow1327_block_002_checked, smallRow1327_small_right_right_checked]
  rfl

def smallRow1327_small : List Nat := smallRow1327_small_left ++ smallRow1327_small_right
theorem smallRow1327_small_checked : upperNeighbors PrimePairVertices.tree_small 1327 = smallRow1327_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1327 ++ upperNeighbors PrimePairVertices.tree_small_right 1327 = _
  rw [smallRow1327_small_left_checked, smallRow1327_small_right_checked]
  rfl

def smallRow1327 : List Nat := smallRow1327_small
theorem smallRow1327_checked : upperNeighbors PrimePairVertices.smallTree 1327 = smallRow1327 := smallRow1327_small_checked

def smallRow1361_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1361_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1361 = smallRow1361_block_000 := by decide

def smallRow1361_block_001 : List Nat := [2141, 3461, 3581, 3701, 3917, 3947]
set_option maxRecDepth 2048 in
theorem smallRow1361_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1361 = smallRow1361_block_001 := by decide

def smallRow1361_block_002 : List Nat := [4793, 5099, 5669, 6047]
set_option maxRecDepth 2048 in
theorem smallRow1361_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1361 = smallRow1361_block_002 := by decide

def smallRow1361_block_003 : List Nat := [6599, 6737, 6899, 6947, 7103, 7607]
set_option maxRecDepth 2048 in
theorem smallRow1361_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1361 = smallRow1361_block_003 := by decide

def smallRow1361_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1361_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1361 = smallRow1361_block_004 := by decide

def smallRow1361_small_left : List Nat := smallRow1361_block_000 ++ smallRow1361_block_001
theorem smallRow1361_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1361 = smallRow1361_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1361 ++ upperNeighbors PrimePairVertices.tree_block_001 1361 = _
  rw [smallRow1361_block_000_checked, smallRow1361_block_001_checked]
  rfl

def smallRow1361_small_right_right : List Nat := smallRow1361_block_003 ++ smallRow1361_block_004
theorem smallRow1361_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1361 = smallRow1361_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1361 ++ upperNeighbors PrimePairVertices.tree_block_004 1361 = _
  rw [smallRow1361_block_003_checked, smallRow1361_block_004_checked]
  rfl

def smallRow1361_small_right : List Nat := smallRow1361_block_002 ++ smallRow1361_small_right_right
theorem smallRow1361_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1361 = smallRow1361_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1361 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1361 = _
  rw [smallRow1361_block_002_checked, smallRow1361_small_right_right_checked]
  rfl

def smallRow1361_small : List Nat := smallRow1361_small_left ++ smallRow1361_small_right
theorem smallRow1361_small_checked : upperNeighbors PrimePairVertices.tree_small 1361 = smallRow1361_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1361 ++ upperNeighbors PrimePairVertices.tree_small_right 1361 = _
  rw [smallRow1361_small_left_checked, smallRow1361_small_right_checked]
  rfl

def smallRow1361 : List Nat := smallRow1361_small
theorem smallRow1361_checked : upperNeighbors PrimePairVertices.smallTree 1361 = smallRow1361 := smallRow1361_small_checked

end ProjectEulerSolutions.PrimePairCache
