import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1237_block_000 : List Nat := [1543, 1549, 1657, 1741, 1753]
set_option maxRecDepth 2048 in
theorem smallRow1237_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1237 = smallRow1237_block_000 := by decide

def smallRow1237_block_001 : List Nat := [2341, 3019, 3187, 3253, 3307, 4027]
set_option maxRecDepth 2048 in
theorem smallRow1237_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1237 = smallRow1237_block_001 := by decide

def smallRow1237_block_002 : List Nat := [4993, 5557, 5743]
set_option maxRecDepth 2048 in
theorem smallRow1237_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1237 = smallRow1237_block_002 := by decide

def smallRow1237_block_003 : List Nat := [6271, 6571, 6673, 7789]
set_option maxRecDepth 2048 in
theorem smallRow1237_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1237 = smallRow1237_block_003 := by decide

def smallRow1237_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1237_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1237 = smallRow1237_block_004 := by decide

def smallRow1237_small_left : List Nat := smallRow1237_block_000 ++ smallRow1237_block_001
theorem smallRow1237_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1237 = smallRow1237_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1237 ++ upperNeighbors PrimePairVertices.tree_block_001 1237 = _
  rw [smallRow1237_block_000_checked, smallRow1237_block_001_checked]
  rfl

def smallRow1237_small_right_right : List Nat := smallRow1237_block_003 ++ smallRow1237_block_004
theorem smallRow1237_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1237 = smallRow1237_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1237 ++ upperNeighbors PrimePairVertices.tree_block_004 1237 = _
  rw [smallRow1237_block_003_checked, smallRow1237_block_004_checked]
  rfl

def smallRow1237_small_right : List Nat := smallRow1237_block_002 ++ smallRow1237_small_right_right
theorem smallRow1237_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1237 = smallRow1237_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1237 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1237 = _
  rw [smallRow1237_block_002_checked, smallRow1237_small_right_right_checked]
  rfl

def smallRow1237_small : List Nat := smallRow1237_small_left ++ smallRow1237_small_right
theorem smallRow1237_small_checked : upperNeighbors PrimePairVertices.tree_small 1237 = smallRow1237_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1237 ++ upperNeighbors PrimePairVertices.tree_small_right 1237 = _
  rw [smallRow1237_small_left_checked, smallRow1237_small_right_checked]
  rfl

def smallRow1237 : List Nat := smallRow1237_small
theorem smallRow1237_checked : upperNeighbors PrimePairVertices.smallTree 1237 = smallRow1237 := smallRow1237_small_checked

def smallRow1249_block_000 : List Nat := [1867]
set_option maxRecDepth 2048 in
theorem smallRow1249_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1249 = smallRow1249_block_000 := by decide

def smallRow1249_block_001 : List Nat := [2131, 2377, 2551, 3391, 3727]
set_option maxRecDepth 2048 in
theorem smallRow1249_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1249 = smallRow1249_block_001 := by decide

def smallRow1249_block_002 : List Nat := [4441, 5563, 5569, 5851, 6079]
set_option maxRecDepth 2048 in
theorem smallRow1249_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1249 = smallRow1249_block_002 := by decide

def smallRow1249_block_003 : List Nat := [6151, 6871, 6949, 7369, 7621]
set_option maxRecDepth 2048 in
theorem smallRow1249_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1249 = smallRow1249_block_003 := by decide

def smallRow1249_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1249_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1249 = smallRow1249_block_004 := by decide

def smallRow1249_small_left : List Nat := smallRow1249_block_000 ++ smallRow1249_block_001
theorem smallRow1249_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1249 = smallRow1249_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1249 ++ upperNeighbors PrimePairVertices.tree_block_001 1249 = _
  rw [smallRow1249_block_000_checked, smallRow1249_block_001_checked]
  rfl

def smallRow1249_small_right_right : List Nat := smallRow1249_block_003 ++ smallRow1249_block_004
theorem smallRow1249_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1249 = smallRow1249_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1249 ++ upperNeighbors PrimePairVertices.tree_block_004 1249 = _
  rw [smallRow1249_block_003_checked, smallRow1249_block_004_checked]
  rfl

def smallRow1249_small_right : List Nat := smallRow1249_block_002 ++ smallRow1249_small_right_right
theorem smallRow1249_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1249 = smallRow1249_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1249 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1249 = _
  rw [smallRow1249_block_002_checked, smallRow1249_small_right_right_checked]
  rfl

def smallRow1249_small : List Nat := smallRow1249_small_left ++ smallRow1249_small_right
theorem smallRow1249_small_checked : upperNeighbors PrimePairVertices.tree_small 1249 = smallRow1249_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1249 ++ upperNeighbors PrimePairVertices.tree_small_right 1249 = _
  rw [smallRow1249_small_left_checked, smallRow1249_small_right_checked]
  rfl

def smallRow1249 : List Nat := smallRow1249_small
theorem smallRow1249_checked : upperNeighbors PrimePairVertices.smallTree 1249 = smallRow1249 := smallRow1249_small_checked

def smallRow1259_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1259_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1259 = smallRow1259_block_000 := by decide

def smallRow1259_block_001 : List Nat := [2213, 2273, 2297, 2357, 2609, 2633, 2819, 3989]
set_option maxRecDepth 2048 in
theorem smallRow1259_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1259 = smallRow1259_block_001 := by decide

def smallRow1259_block_002 : List Nat := [5039, 5507]
set_option maxRecDepth 2048 in
theorem smallRow1259_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1259 = smallRow1259_block_002 := by decide

def smallRow1259_block_003 : List Nat := [6599, 6959, 7283]
set_option maxRecDepth 2048 in
theorem smallRow1259_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1259 = smallRow1259_block_003 := by decide

def smallRow1259_block_004 : List Nat := [8291]
set_option maxRecDepth 2048 in
theorem smallRow1259_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1259 = smallRow1259_block_004 := by decide

def smallRow1259_small_left : List Nat := smallRow1259_block_000 ++ smallRow1259_block_001
theorem smallRow1259_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1259 = smallRow1259_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1259 ++ upperNeighbors PrimePairVertices.tree_block_001 1259 = _
  rw [smallRow1259_block_000_checked, smallRow1259_block_001_checked]
  rfl

def smallRow1259_small_right_right : List Nat := smallRow1259_block_003 ++ smallRow1259_block_004
theorem smallRow1259_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1259 = smallRow1259_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1259 ++ upperNeighbors PrimePairVertices.tree_block_004 1259 = _
  rw [smallRow1259_block_003_checked, smallRow1259_block_004_checked]
  rfl

def smallRow1259_small_right : List Nat := smallRow1259_block_002 ++ smallRow1259_small_right_right
theorem smallRow1259_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1259 = smallRow1259_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1259 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1259 = _
  rw [smallRow1259_block_002_checked, smallRow1259_small_right_right_checked]
  rfl

def smallRow1259_small : List Nat := smallRow1259_small_left ++ smallRow1259_small_right
theorem smallRow1259_small_checked : upperNeighbors PrimePairVertices.tree_small 1259 = smallRow1259_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1259 ++ upperNeighbors PrimePairVertices.tree_small_right 1259 = _
  rw [smallRow1259_small_left_checked, smallRow1259_small_right_checked]
  rfl

def smallRow1259 : List Nat := smallRow1259_small
theorem smallRow1259_checked : upperNeighbors PrimePairVertices.smallTree 1259 = smallRow1259 := smallRow1259_small_checked

def smallRow1277_block_000 : List Nat := [1511, 1523]
set_option maxRecDepth 2048 in
theorem smallRow1277_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1277 = smallRow1277_block_000 := by decide

def smallRow1277_block_001 : List Nat := [3359, 4049]
set_option maxRecDepth 2048 in
theorem smallRow1277_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1277 = smallRow1277_block_001 := by decide

def smallRow1277_block_002 : List Nat := [4457, 4787, 4889, 4967, 5003, 6011, 6053]
set_option maxRecDepth 2048 in
theorem smallRow1277_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1277 = smallRow1277_block_002 := by decide

def smallRow1277_block_003 : List Nat := [6173, 6353, 6761, 7433, 7529, 7559, 7577, 7703, 8147]
set_option maxRecDepth 2048 in
theorem smallRow1277_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1277 = smallRow1277_block_003 := by decide

def smallRow1277_block_004 : List Nat := [8291, 8369]
set_option maxRecDepth 2048 in
theorem smallRow1277_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1277 = smallRow1277_block_004 := by decide

def smallRow1277_small_left : List Nat := smallRow1277_block_000 ++ smallRow1277_block_001
theorem smallRow1277_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1277 = smallRow1277_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1277 ++ upperNeighbors PrimePairVertices.tree_block_001 1277 = _
  rw [smallRow1277_block_000_checked, smallRow1277_block_001_checked]
  rfl

def smallRow1277_small_right_right : List Nat := smallRow1277_block_003 ++ smallRow1277_block_004
theorem smallRow1277_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1277 = smallRow1277_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1277 ++ upperNeighbors PrimePairVertices.tree_block_004 1277 = _
  rw [smallRow1277_block_003_checked, smallRow1277_block_004_checked]
  rfl

def smallRow1277_small_right : List Nat := smallRow1277_block_002 ++ smallRow1277_small_right_right
theorem smallRow1277_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1277 = smallRow1277_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1277 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1277 = _
  rw [smallRow1277_block_002_checked, smallRow1277_small_right_right_checked]
  rfl

def smallRow1277_small : List Nat := smallRow1277_small_left ++ smallRow1277_small_right
theorem smallRow1277_small_checked : upperNeighbors PrimePairVertices.tree_small 1277 = smallRow1277_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1277 ++ upperNeighbors PrimePairVertices.tree_small_right 1277 = _
  rw [smallRow1277_small_left_checked, smallRow1277_small_right_checked]
  rfl

def smallRow1277 : List Nat := smallRow1277_small
theorem smallRow1277_checked : upperNeighbors PrimePairVertices.smallTree 1277 = smallRow1277 := smallRow1277_small_checked

def smallRow1279_block_000 : List Nat := [1297, 1399, 1459]
set_option maxRecDepth 2048 in
theorem smallRow1279_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1279 = smallRow1279_block_000 := by decide

def smallRow1279_block_001 : List Nat := [2179, 2389, 2803, 3463, 3559, 3691]
set_option maxRecDepth 2048 in
theorem smallRow1279_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1279 = smallRow1279_block_001 := by decide

def smallRow1279_block_002 : List Nat := [5059, 5179, 5437, 5479, 6037]
set_option maxRecDepth 2048 in
theorem smallRow1279_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1279 = smallRow1279_block_002 := by decide

def smallRow1279_block_003 : List Nat := [6271, 6607, 6781, 7069, 8101]
set_option maxRecDepth 2048 in
theorem smallRow1279_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1279 = smallRow1279_block_003 := by decide

def smallRow1279_block_004 : List Nat := [8293]
set_option maxRecDepth 2048 in
theorem smallRow1279_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1279 = smallRow1279_block_004 := by decide

def smallRow1279_small_left : List Nat := smallRow1279_block_000 ++ smallRow1279_block_001
theorem smallRow1279_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1279 = smallRow1279_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1279 ++ upperNeighbors PrimePairVertices.tree_block_001 1279 = _
  rw [smallRow1279_block_000_checked, smallRow1279_block_001_checked]
  rfl

def smallRow1279_small_right_right : List Nat := smallRow1279_block_003 ++ smallRow1279_block_004
theorem smallRow1279_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1279 = smallRow1279_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1279 ++ upperNeighbors PrimePairVertices.tree_block_004 1279 = _
  rw [smallRow1279_block_003_checked, smallRow1279_block_004_checked]
  rfl

def smallRow1279_small_right : List Nat := smallRow1279_block_002 ++ smallRow1279_small_right_right
theorem smallRow1279_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1279 = smallRow1279_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1279 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1279 = _
  rw [smallRow1279_block_002_checked, smallRow1279_small_right_right_checked]
  rfl

def smallRow1279_small : List Nat := smallRow1279_small_left ++ smallRow1279_small_right
theorem smallRow1279_small_checked : upperNeighbors PrimePairVertices.tree_small 1279 = smallRow1279_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1279 ++ upperNeighbors PrimePairVertices.tree_small_right 1279 = _
  rw [smallRow1279_small_left_checked, smallRow1279_small_right_checked]
  rfl

def smallRow1279 : List Nat := smallRow1279_small
theorem smallRow1279_checked : upperNeighbors PrimePairVertices.smallTree 1279 = smallRow1279 := smallRow1279_small_checked

def smallRow1283_block_000 : List Nat := [1619, 1667, 1913]
set_option maxRecDepth 2048 in
theorem smallRow1283_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1283 = smallRow1283_block_000 := by decide

def smallRow1283_block_001 : List Nat := [2111, 2423, 2741, 2801, 3593, 3617, 3719]
set_option maxRecDepth 2048 in
theorem smallRow1283_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1283 = smallRow1283_block_001 := by decide

def smallRow1283_block_002 : List Nat := [4127, 4397, 5021, 5441, 5807]
set_option maxRecDepth 2048 in
theorem smallRow1283_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1283 = smallRow1283_block_002 := by decide

def smallRow1283_block_003 : List Nat := [6221, 6287, 6827, 6869, 7001, 7247, 7757, 7949]
set_option maxRecDepth 2048 in
theorem smallRow1283_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1283 = smallRow1283_block_003 := by decide

def smallRow1283_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow1283_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1283 = smallRow1283_block_004 := by decide

def smallRow1283_small_left : List Nat := smallRow1283_block_000 ++ smallRow1283_block_001
theorem smallRow1283_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1283 = smallRow1283_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1283 ++ upperNeighbors PrimePairVertices.tree_block_001 1283 = _
  rw [smallRow1283_block_000_checked, smallRow1283_block_001_checked]
  rfl

def smallRow1283_small_right_right : List Nat := smallRow1283_block_003 ++ smallRow1283_block_004
theorem smallRow1283_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1283 = smallRow1283_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1283 ++ upperNeighbors PrimePairVertices.tree_block_004 1283 = _
  rw [smallRow1283_block_003_checked, smallRow1283_block_004_checked]
  rfl

def smallRow1283_small_right : List Nat := smallRow1283_block_002 ++ smallRow1283_small_right_right
theorem smallRow1283_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1283 = smallRow1283_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1283 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1283 = _
  rw [smallRow1283_block_002_checked, smallRow1283_small_right_right_checked]
  rfl

def smallRow1283_small : List Nat := smallRow1283_small_left ++ smallRow1283_small_right
theorem smallRow1283_small_checked : upperNeighbors PrimePairVertices.tree_small 1283 = smallRow1283_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1283 ++ upperNeighbors PrimePairVertices.tree_small_right 1283 = _
  rw [smallRow1283_small_left_checked, smallRow1283_small_right_checked]
  rfl

def smallRow1283 : List Nat := smallRow1283_small
theorem smallRow1283_checked : upperNeighbors PrimePairVertices.smallTree 1283 = smallRow1283 := smallRow1283_small_checked

def smallRow1289_block_000 : List Nat := [1409, 1721, 2027, 2039]
set_option maxRecDepth 2048 in
theorem smallRow1289_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1289 = smallRow1289_block_000 := by decide

def smallRow1289_block_001 : List Nat := [2237, 2801, 3767, 3833]
set_option maxRecDepth 2048 in
theorem smallRow1289_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1289 = smallRow1289_block_001 := by decide

def smallRow1289_block_002 : List Nat := [4139, 4733]
set_option maxRecDepth 2048 in
theorem smallRow1289_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1289 = smallRow1289_block_002 := by decide

def smallRow1289_block_003 : List Nat := [6311, 6449, 7541, 7853, 7919]
set_option maxRecDepth 2048 in
theorem smallRow1289_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1289 = smallRow1289_block_003 := by decide

def smallRow1289_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1289_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1289 = smallRow1289_block_004 := by decide

def smallRow1289_small_left : List Nat := smallRow1289_block_000 ++ smallRow1289_block_001
theorem smallRow1289_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1289 = smallRow1289_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1289 ++ upperNeighbors PrimePairVertices.tree_block_001 1289 = _
  rw [smallRow1289_block_000_checked, smallRow1289_block_001_checked]
  rfl

def smallRow1289_small_right_right : List Nat := smallRow1289_block_003 ++ smallRow1289_block_004
theorem smallRow1289_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1289 = smallRow1289_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1289 ++ upperNeighbors PrimePairVertices.tree_block_004 1289 = _
  rw [smallRow1289_block_003_checked, smallRow1289_block_004_checked]
  rfl

def smallRow1289_small_right : List Nat := smallRow1289_block_002 ++ smallRow1289_small_right_right
theorem smallRow1289_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1289 = smallRow1289_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1289 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1289 = _
  rw [smallRow1289_block_002_checked, smallRow1289_small_right_right_checked]
  rfl

def smallRow1289_small : List Nat := smallRow1289_small_left ++ smallRow1289_small_right
theorem smallRow1289_small_checked : upperNeighbors PrimePairVertices.tree_small 1289 = smallRow1289_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1289 ++ upperNeighbors PrimePairVertices.tree_small_right 1289 = _
  rw [smallRow1289_small_left_checked, smallRow1289_small_right_checked]
  rfl

def smallRow1289 : List Nat := smallRow1289_small
theorem smallRow1289_checked : upperNeighbors PrimePairVertices.smallTree 1289 = smallRow1289 := smallRow1289_small_checked

def smallRow1291_block_000 : List Nat := [1777]
set_option maxRecDepth 2048 in
theorem smallRow1291_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1291 = smallRow1291_block_000 := by decide

def smallRow1291_block_001 : List Nat := [2281, 2293, 3457, 3793]
set_option maxRecDepth 2048 in
theorem smallRow1291_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1291 = smallRow1291_block_001 := by decide

def smallRow1291_block_002 : List Nat := [4273, 4513, 4903, 5059, 5557]
set_option maxRecDepth 2048 in
theorem smallRow1291_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1291 = smallRow1291_block_002 := by decide

def smallRow1291_block_003 : List Nat := [6733, 6907, 7741]
set_option maxRecDepth 2048 in
theorem smallRow1291_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1291 = smallRow1291_block_003 := by decide

def smallRow1291_block_004 : List Nat := [8209, 8293]
set_option maxRecDepth 2048 in
theorem smallRow1291_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1291 = smallRow1291_block_004 := by decide

def smallRow1291_small_left : List Nat := smallRow1291_block_000 ++ smallRow1291_block_001
theorem smallRow1291_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1291 = smallRow1291_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1291 ++ upperNeighbors PrimePairVertices.tree_block_001 1291 = _
  rw [smallRow1291_block_000_checked, smallRow1291_block_001_checked]
  rfl

def smallRow1291_small_right_right : List Nat := smallRow1291_block_003 ++ smallRow1291_block_004
theorem smallRow1291_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1291 = smallRow1291_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1291 ++ upperNeighbors PrimePairVertices.tree_block_004 1291 = _
  rw [smallRow1291_block_003_checked, smallRow1291_block_004_checked]
  rfl

def smallRow1291_small_right : List Nat := smallRow1291_block_002 ++ smallRow1291_small_right_right
theorem smallRow1291_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1291 = smallRow1291_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1291 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1291 = _
  rw [smallRow1291_block_002_checked, smallRow1291_small_right_right_checked]
  rfl

def smallRow1291_small : List Nat := smallRow1291_small_left ++ smallRow1291_small_right
theorem smallRow1291_small_checked : upperNeighbors PrimePairVertices.tree_small 1291 = smallRow1291_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1291 ++ upperNeighbors PrimePairVertices.tree_small_right 1291 = _
  rw [smallRow1291_small_left_checked, smallRow1291_small_right_checked]
  rfl

def smallRow1291 : List Nat := smallRow1291_small
theorem smallRow1291_checked : upperNeighbors PrimePairVertices.smallTree 1291 = smallRow1291 := smallRow1291_small_checked

end ProjectEulerSolutions.PrimePairCache
