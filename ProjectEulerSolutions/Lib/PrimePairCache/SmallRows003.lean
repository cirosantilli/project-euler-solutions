import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow103_block_000 : List Nat := [307, 421, 997, 1483, 1669, 1759, 1831, 1999]
set_option maxRecDepth 2048 in
theorem smallRow103_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 103 = smallRow103_block_000 := by decide

def smallRow103_block_001 : List Nat := [2341, 2467, 2617, 2683, 2803, 3061, 4027]
set_option maxRecDepth 2048 in
theorem smallRow103_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 103 = smallRow103_block_001 := by decide

def smallRow103_block_002 : List Nat := [4339, 4357, 4567, 5413, 5527, 5869]
set_option maxRecDepth 2048 in
theorem smallRow103_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 103 = smallRow103_block_002 := by decide

def smallRow103_block_003 : List Nat := [6229, 6247, 6661, 7489, 7879]
set_option maxRecDepth 2048 in
theorem smallRow103_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 103 = smallRow103_block_003 := by decide

def smallRow103_block_004 : List Nat := [8269, 8329]
set_option maxRecDepth 2048 in
theorem smallRow103_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 103 = smallRow103_block_004 := by decide

def smallRow103_small_left : List Nat := smallRow103_block_000 ++ smallRow103_block_001
theorem smallRow103_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 103 = smallRow103_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 103 ++ upperNeighbors PrimePairVertices.tree_block_001 103 = _
  rw [smallRow103_block_000_checked, smallRow103_block_001_checked]
  rfl

def smallRow103_small_right_right : List Nat := smallRow103_block_003 ++ smallRow103_block_004
theorem smallRow103_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 103 = smallRow103_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 103 ++ upperNeighbors PrimePairVertices.tree_block_004 103 = _
  rw [smallRow103_block_003_checked, smallRow103_block_004_checked]
  rfl

def smallRow103_small_right : List Nat := smallRow103_block_002 ++ smallRow103_small_right_right
theorem smallRow103_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 103 = smallRow103_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 103 ++ upperNeighbors PrimePairVertices.tree_small_right_right 103 = _
  rw [smallRow103_block_002_checked, smallRow103_small_right_right_checked]
  rfl

def smallRow103_small : List Nat := smallRow103_small_left ++ smallRow103_small_right
theorem smallRow103_small_checked : upperNeighbors PrimePairVertices.tree_small 103 = smallRow103_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 103 ++ upperNeighbors PrimePairVertices.tree_small_right 103 = _
  rw [smallRow103_small_left_checked, smallRow103_small_right_checked]
  rfl

def smallRow103 : List Nat := smallRow103_small
theorem smallRow103_checked : upperNeighbors PrimePairVertices.smallTree 103 = smallRow103 := smallRow103_small_checked

def smallRow107_block_000 : List Nat := [449, 857, 1061, 1193, 1439, 1601, 1907]
set_option maxRecDepth 2048 in
theorem smallRow107_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 107 = smallRow107_block_000 := by decide

def smallRow107_block_001 : List Nat := [2129, 2339, 2381, 2633, 2843, 2957, 2999, 3221]
set_option maxRecDepth 2048 in
theorem smallRow107_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 107 = smallRow107_block_001 := by decide

def smallRow107_block_002 : List Nat := [4289, 4523, 4691, 4919, 4973, 5279, 5507, 5531, 5897]
set_option maxRecDepth 2048 in
theorem smallRow107_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 107 = smallRow107_block_002 := by decide

def smallRow107_block_003 : List Nat := [6359, 6563, 6917, 7607]
set_option maxRecDepth 2048 in
theorem smallRow107_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 107 = smallRow107_block_003 := by decide

def smallRow107_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow107_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 107 = smallRow107_block_004 := by decide

def smallRow107_small_left : List Nat := smallRow107_block_000 ++ smallRow107_block_001
theorem smallRow107_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 107 = smallRow107_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 107 ++ upperNeighbors PrimePairVertices.tree_block_001 107 = _
  rw [smallRow107_block_000_checked, smallRow107_block_001_checked]
  rfl

def smallRow107_small_right_right : List Nat := smallRow107_block_003 ++ smallRow107_block_004
theorem smallRow107_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 107 = smallRow107_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 107 ++ upperNeighbors PrimePairVertices.tree_block_004 107 = _
  rw [smallRow107_block_003_checked, smallRow107_block_004_checked]
  rfl

def smallRow107_small_right : List Nat := smallRow107_block_002 ++ smallRow107_small_right_right
theorem smallRow107_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 107 = smallRow107_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 107 ++ upperNeighbors PrimePairVertices.tree_small_right_right 107 = _
  rw [smallRow107_block_002_checked, smallRow107_small_right_right_checked]
  rfl

def smallRow107_small : List Nat := smallRow107_small_left ++ smallRow107_small_right
theorem smallRow107_small_checked : upperNeighbors PrimePairVertices.tree_small 107 = smallRow107_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 107 ++ upperNeighbors PrimePairVertices.tree_small_right 107 = _
  rw [smallRow107_small_left_checked, smallRow107_small_right_checked]
  rfl

def smallRow107 : List Nat := smallRow107_small
theorem smallRow107_checked : upperNeighbors PrimePairVertices.smallTree 107 = smallRow107 := smallRow107_small_checked

def smallRow109_block_000 : List Nat := [139, 199, 313, 661, 673, 859, 883, 919, 1063, 1381, 1399, 1459, 1609, 1777]
set_option maxRecDepth 2048 in
theorem smallRow109_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 109 = smallRow109_block_000 := by decide

def smallRow109_block_001 : List Nat := [2389, 2803, 3331, 3529, 3943]
set_option maxRecDepth 2048 in
theorem smallRow109_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 109 = smallRow109_block_001 := by decide

def smallRow109_block_002 : List Nat := [4129, 4441, 4567, 4759, 4801, 4957, 5503, 5791]
set_option maxRecDepth 2048 in
theorem smallRow109_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 109 = smallRow109_block_002 := by decide

def smallRow109_block_003 : List Nat := [6673, 7297, 7321, 7351, 7933]
set_option maxRecDepth 2048 in
theorem smallRow109_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 109 = smallRow109_block_003 := by decide

def smallRow109_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow109_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 109 = smallRow109_block_004 := by decide

def smallRow109_small_left : List Nat := smallRow109_block_000 ++ smallRow109_block_001
theorem smallRow109_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 109 = smallRow109_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 109 ++ upperNeighbors PrimePairVertices.tree_block_001 109 = _
  rw [smallRow109_block_000_checked, smallRow109_block_001_checked]
  rfl

def smallRow109_small_right_right : List Nat := smallRow109_block_003 ++ smallRow109_block_004
theorem smallRow109_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 109 = smallRow109_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 109 ++ upperNeighbors PrimePairVertices.tree_block_004 109 = _
  rw [smallRow109_block_003_checked, smallRow109_block_004_checked]
  rfl

def smallRow109_small_right : List Nat := smallRow109_block_002 ++ smallRow109_small_right_right
theorem smallRow109_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 109 = smallRow109_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 109 ++ upperNeighbors PrimePairVertices.tree_small_right_right 109 = _
  rw [smallRow109_block_002_checked, smallRow109_small_right_right_checked]
  rfl

def smallRow109_small : List Nat := smallRow109_small_left ++ smallRow109_small_right
theorem smallRow109_small_checked : upperNeighbors PrimePairVertices.tree_small 109 = smallRow109_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 109 ++ upperNeighbors PrimePairVertices.tree_small_right 109 = _
  rw [smallRow109_small_left_checked, smallRow109_small_right_checked]
  rfl

def smallRow109 : List Nat := smallRow109_small
theorem smallRow109_checked : upperNeighbors PrimePairVertices.smallTree 109 = smallRow109 := smallRow109_small_checked

def smallRow113_block_000 : List Nat := [131, 149, 167, 227, 233, 383, 647, 761, 983, 1217, 1259, 1973]
set_option maxRecDepth 2048 in
theorem smallRow113_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 113 = smallRow113_block_000 := by decide

def smallRow113_block_001 : List Nat := [2309, 2861, 3191, 3359, 3533, 3659, 3947]
set_option maxRecDepth 2048 in
theorem smallRow113_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 113 = smallRow113_block_001 := by decide

def smallRow113_block_002 : List Nat := [4421, 4517, 5333, 5711]
set_option maxRecDepth 2048 in
theorem smallRow113_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 113 = smallRow113_block_002 := by decide

def smallRow113_block_003 : List Nat := [6389, 6659, 6719, 6869, 7457]
set_option maxRecDepth 2048 in
theorem smallRow113_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 113 = smallRow113_block_003 := by decide

def smallRow113_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow113_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 113 = smallRow113_block_004 := by decide

def smallRow113_small_left : List Nat := smallRow113_block_000 ++ smallRow113_block_001
theorem smallRow113_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 113 = smallRow113_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 113 ++ upperNeighbors PrimePairVertices.tree_block_001 113 = _
  rw [smallRow113_block_000_checked, smallRow113_block_001_checked]
  rfl

def smallRow113_small_right_right : List Nat := smallRow113_block_003 ++ smallRow113_block_004
theorem smallRow113_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 113 = smallRow113_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 113 ++ upperNeighbors PrimePairVertices.tree_block_004 113 = _
  rw [smallRow113_block_003_checked, smallRow113_block_004_checked]
  rfl

def smallRow113_small_right : List Nat := smallRow113_block_002 ++ smallRow113_small_right_right
theorem smallRow113_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 113 = smallRow113_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 113 ++ upperNeighbors PrimePairVertices.tree_small_right_right 113 = _
  rw [smallRow113_block_002_checked, smallRow113_small_right_right_checked]
  rfl

def smallRow113_small : List Nat := smallRow113_small_left ++ smallRow113_small_right
theorem smallRow113_small_checked : upperNeighbors PrimePairVertices.tree_small 113 = smallRow113_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 113 ++ upperNeighbors PrimePairVertices.tree_small_right 113 = _
  rw [smallRow113_small_left_checked, smallRow113_small_right_checked]
  rfl

def smallRow113 : List Nat := smallRow113_small
theorem smallRow113_checked : upperNeighbors PrimePairVertices.smallTree 113 = smallRow113 := smallRow113_small_checked

def smallRow127_block_000 : List Nat := [157, 163, 241, 271, 331, 373, 601, 607, 733, 1033, 1069, 1129, 1987]
set_option maxRecDepth 2048 in
theorem smallRow127_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 127 = smallRow127_block_000 := by decide

def smallRow127_block_001 : List Nat := [2281, 2749, 3001, 3343, 3541, 3673]
set_option maxRecDepth 2048 in
theorem smallRow127_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 127 = smallRow127_block_001 := by decide

def smallRow127_block_002 : List Nat := [4363, 4423, 4759, 5179, 5563, 5683, 5749, 5779]
set_option maxRecDepth 2048 in
theorem smallRow127_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 127 = smallRow127_block_002 := by decide

def smallRow127_block_003 : List Nat := [6271, 6361, 6481, 6571, 6733, 6871, 6949, 7369, 7723, 7753, 7993]
set_option maxRecDepth 2048 in
theorem smallRow127_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 127 = smallRow127_block_003 := by decide

def smallRow127_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow127_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 127 = smallRow127_block_004 := by decide

def smallRow127_small_left : List Nat := smallRow127_block_000 ++ smallRow127_block_001
theorem smallRow127_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 127 = smallRow127_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 127 ++ upperNeighbors PrimePairVertices.tree_block_001 127 = _
  rw [smallRow127_block_000_checked, smallRow127_block_001_checked]
  rfl

def smallRow127_small_right_right : List Nat := smallRow127_block_003 ++ smallRow127_block_004
theorem smallRow127_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 127 = smallRow127_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 127 ++ upperNeighbors PrimePairVertices.tree_block_004 127 = _
  rw [smallRow127_block_003_checked, smallRow127_block_004_checked]
  rfl

def smallRow127_small_right : List Nat := smallRow127_block_002 ++ smallRow127_small_right_right
theorem smallRow127_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 127 = smallRow127_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 127 ++ upperNeighbors PrimePairVertices.tree_small_right_right 127 = _
  rw [smallRow127_block_002_checked, smallRow127_small_right_right_checked]
  rfl

def smallRow127_small : List Nat := smallRow127_small_left ++ smallRow127_small_right
theorem smallRow127_small_checked : upperNeighbors PrimePairVertices.tree_small 127 = smallRow127_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 127 ++ upperNeighbors PrimePairVertices.tree_small_right 127 = _
  rw [smallRow127_small_left_checked, smallRow127_small_right_checked]
  rfl

def smallRow127 : List Nat := smallRow127_small
theorem smallRow127_checked : upperNeighbors PrimePairVertices.smallTree 127 = smallRow127 := smallRow127_small_checked

def smallRow131_block_000 : List Nat := [449, 479, 617, 641, 743, 797, 839, 941, 1217, 1229, 1523, 1901]
set_option maxRecDepth 2048 in
theorem smallRow131_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 131 = smallRow131_block_000 := by decide

def smallRow131_block_001 : List Nat := [2411, 2963, 3083, 3467, 3881, 3929]
set_option maxRecDepth 2048 in
theorem smallRow131_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 131 = smallRow131_block_001 := by decide

def smallRow131_block_002 : List Nat := [4127, 4133, 4217, 4259, 4283, 4673, 5231, 5399, 5441, 5519, 5651]
set_option maxRecDepth 2048 in
theorem smallRow131_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 131 = smallRow131_block_002 := by decide

def smallRow131_block_003 : List Nat := [6299, 6779, 6971, 7013, 7487, 7703, 7727]
set_option maxRecDepth 2048 in
theorem smallRow131_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 131 = smallRow131_block_003 := by decide

def smallRow131_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow131_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 131 = smallRow131_block_004 := by decide

def smallRow131_small_left : List Nat := smallRow131_block_000 ++ smallRow131_block_001
theorem smallRow131_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 131 = smallRow131_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 131 ++ upperNeighbors PrimePairVertices.tree_block_001 131 = _
  rw [smallRow131_block_000_checked, smallRow131_block_001_checked]
  rfl

def smallRow131_small_right_right : List Nat := smallRow131_block_003 ++ smallRow131_block_004
theorem smallRow131_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 131 = smallRow131_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 131 ++ upperNeighbors PrimePairVertices.tree_block_004 131 = _
  rw [smallRow131_block_003_checked, smallRow131_block_004_checked]
  rfl

def smallRow131_small_right : List Nat := smallRow131_block_002 ++ smallRow131_small_right_right
theorem smallRow131_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 131 = smallRow131_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 131 ++ upperNeighbors PrimePairVertices.tree_small_right_right 131 = _
  rw [smallRow131_block_002_checked, smallRow131_small_right_right_checked]
  rfl

def smallRow131_small : List Nat := smallRow131_small_left ++ smallRow131_small_right
theorem smallRow131_small_checked : upperNeighbors PrimePairVertices.tree_small 131 = smallRow131_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 131 ++ upperNeighbors PrimePairVertices.tree_small_right 131 = _
  rw [smallRow131_small_left_checked, smallRow131_small_right_checked]
  rfl

def smallRow131 : List Nat := smallRow131_small
theorem smallRow131_checked : upperNeighbors PrimePairVertices.smallTree 131 = smallRow131 := smallRow131_small_checked

def smallRow137_block_000 : List Nat := [191, 197, 239, 353, 359, 491, 587, 659, 743, 947, 1031]
set_option maxRecDepth 2048 in
theorem smallRow137_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 137 = smallRow137_block_000 := by decide

def smallRow137_block_001 : List Nat := [2621, 2963, 3137, 3167, 3191, 3677, 3989]
set_option maxRecDepth 2048 in
theorem smallRow137_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 137 = smallRow137_block_001 := by decide

def smallRow137_block_002 : List Nat := [4481, 4721, 5237, 5981]
set_option maxRecDepth 2048 in
theorem smallRow137_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 137 = smallRow137_block_002 := by decide

def smallRow137_block_003 : List Nat := [6173, 6197, 6359, 6491, 6971, 7043, 7121, 7127, 7457, 7499, 7517, 7577, 7643, 7757, 8147]
set_option maxRecDepth 2048 in
theorem smallRow137_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 137 = smallRow137_block_003 := by decide

def smallRow137_block_004 : List Nat := [8219, 8231]
set_option maxRecDepth 2048 in
theorem smallRow137_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 137 = smallRow137_block_004 := by decide

def smallRow137_small_left : List Nat := smallRow137_block_000 ++ smallRow137_block_001
theorem smallRow137_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 137 = smallRow137_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 137 ++ upperNeighbors PrimePairVertices.tree_block_001 137 = _
  rw [smallRow137_block_000_checked, smallRow137_block_001_checked]
  rfl

def smallRow137_small_right_right : List Nat := smallRow137_block_003 ++ smallRow137_block_004
theorem smallRow137_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 137 = smallRow137_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 137 ++ upperNeighbors PrimePairVertices.tree_block_004 137 = _
  rw [smallRow137_block_003_checked, smallRow137_block_004_checked]
  rfl

def smallRow137_small_right : List Nat := smallRow137_block_002 ++ smallRow137_small_right_right
theorem smallRow137_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 137 = smallRow137_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 137 ++ upperNeighbors PrimePairVertices.tree_small_right_right 137 = _
  rw [smallRow137_block_002_checked, smallRow137_small_right_right_checked]
  rfl

def smallRow137_small : List Nat := smallRow137_small_left ++ smallRow137_small_right
theorem smallRow137_small_checked : upperNeighbors PrimePairVertices.tree_small 137 = smallRow137_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 137 ++ upperNeighbors PrimePairVertices.tree_small_right 137 = _
  rw [smallRow137_small_left_checked, smallRow137_small_right_checked]
  rfl

def smallRow137 : List Nat := smallRow137_small
theorem smallRow137_checked : upperNeighbors PrimePairVertices.smallTree 137 = smallRow137 := smallRow137_small_checked

def smallRow139_block_000 : List Nat := [367, 457, 547, 619, 661, 709, 787, 907, 967, 1051, 1447, 1549, 1597]
set_option maxRecDepth 2048 in
theorem smallRow139_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 139 = smallRow139_block_000 := by decide

def smallRow139_block_001 : List Nat := [2803, 3229, 3253, 3313, 3697, 3739]
set_option maxRecDepth 2048 in
theorem smallRow139_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 139 = smallRow139_block_001 := by decide

def smallRow139_block_002 : List Nat := [4177, 5059, 5167, 5323, 5743, 5869]
set_option maxRecDepth 2048 in
theorem smallRow139_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 139 = smallRow139_block_002 := by decide

def smallRow139_block_003 : List Nat := [6679, 6841, 6949, 7057, 7177, 7219, 7933, 8011, 8053]
set_option maxRecDepth 2048 in
theorem smallRow139_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 139 = smallRow139_block_003 := by decide

def smallRow139_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow139_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 139 = smallRow139_block_004 := by decide

def smallRow139_small_left : List Nat := smallRow139_block_000 ++ smallRow139_block_001
theorem smallRow139_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 139 = smallRow139_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 139 ++ upperNeighbors PrimePairVertices.tree_block_001 139 = _
  rw [smallRow139_block_000_checked, smallRow139_block_001_checked]
  rfl

def smallRow139_small_right_right : List Nat := smallRow139_block_003 ++ smallRow139_block_004
theorem smallRow139_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 139 = smallRow139_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 139 ++ upperNeighbors PrimePairVertices.tree_block_004 139 = _
  rw [smallRow139_block_003_checked, smallRow139_block_004_checked]
  rfl

def smallRow139_small_right : List Nat := smallRow139_block_002 ++ smallRow139_small_right_right
theorem smallRow139_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 139 = smallRow139_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 139 ++ upperNeighbors PrimePairVertices.tree_small_right_right 139 = _
  rw [smallRow139_block_002_checked, smallRow139_small_right_right_checked]
  rfl

def smallRow139_small : List Nat := smallRow139_small_left ++ smallRow139_small_right
theorem smallRow139_small_checked : upperNeighbors PrimePairVertices.tree_small 139 = smallRow139_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 139 ++ upperNeighbors PrimePairVertices.tree_small_right 139 = _
  rw [smallRow139_small_left_checked, smallRow139_small_right_checked]
  rfl

def smallRow139 : List Nat := smallRow139_small
theorem smallRow139_checked : upperNeighbors PrimePairVertices.smallTree 139 = smallRow139 := smallRow139_small_checked

end ProjectEulerSolutions.PrimePairCache
