import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1019_block_000 : List Nat := [1439]
set_option maxRecDepth 2048 in
theorem smallRow1019_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1019 = smallRow1019_block_000 := by decide

def smallRow1019_block_001 : List Nat := [2549, 2621, 2957, 2999, 3221, 3767, 3797, 3929, 4001, 4049]
set_option maxRecDepth 2048 in
theorem smallRow1019_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1019 = smallRow1019_block_001 := by decide

def smallRow1019_block_002 : List Nat := [4337, 4517, 4931, 5477, 5483, 5741]
set_option maxRecDepth 2048 in
theorem smallRow1019_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1019 = smallRow1019_block_002 := by decide

def smallRow1019_block_003 : List Nat := [6947, 7193, 7673, 7907, 7949]
set_option maxRecDepth 2048 in
theorem smallRow1019_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1019 = smallRow1019_block_003 := by decide

def smallRow1019_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1019_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1019 = smallRow1019_block_004 := by decide

def smallRow1019_small_left : List Nat := smallRow1019_block_000 ++ smallRow1019_block_001
theorem smallRow1019_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1019 = smallRow1019_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1019 ++ upperNeighbors PrimePairVertices.tree_block_001 1019 = _
  rw [smallRow1019_block_000_checked, smallRow1019_block_001_checked]
  rfl

def smallRow1019_small_right_right : List Nat := smallRow1019_block_003 ++ smallRow1019_block_004
theorem smallRow1019_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1019 = smallRow1019_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1019 ++ upperNeighbors PrimePairVertices.tree_block_004 1019 = _
  rw [smallRow1019_block_003_checked, smallRow1019_block_004_checked]
  rfl

def smallRow1019_small_right : List Nat := smallRow1019_block_002 ++ smallRow1019_small_right_right
theorem smallRow1019_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1019 = smallRow1019_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1019 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1019 = _
  rw [smallRow1019_block_002_checked, smallRow1019_small_right_right_checked]
  rfl

def smallRow1019_small : List Nat := smallRow1019_small_left ++ smallRow1019_small_right
theorem smallRow1019_small_checked : upperNeighbors PrimePairVertices.tree_small 1019 = smallRow1019_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1019 ++ upperNeighbors PrimePairVertices.tree_small_right 1019 = _
  rw [smallRow1019_small_left_checked, smallRow1019_small_right_checked]
  rfl

def smallRow1019 : List Nat := smallRow1019_small
theorem smallRow1019_checked : upperNeighbors PrimePairVertices.smallTree 1019 = smallRow1019 := smallRow1019_small_checked

def smallRow1021_block_000 : List Nat := [1699]
set_option maxRecDepth 2048 in
theorem smallRow1021_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1021 = smallRow1021_block_000 := by decide

def smallRow1021_block_001 : List Nat := [2539, 2749, 3529, 3559, 3571, 3769]
set_option maxRecDepth 2048 in
theorem smallRow1021_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1021 = smallRow1021_block_001 := by decide

def smallRow1021_block_002 : List Nat := [4597, 5281, 5701, 6079]
set_option maxRecDepth 2048 in
theorem smallRow1021_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1021 = smallRow1021_block_002 := by decide

def smallRow1021_block_003 : List Nat := [6247, 6373, 6397, 6691, 7213, 7873]
set_option maxRecDepth 2048 in
theorem smallRow1021_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1021 = smallRow1021_block_003 := by decide

def smallRow1021_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1021_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1021 = smallRow1021_block_004 := by decide

def smallRow1021_small_left : List Nat := smallRow1021_block_000 ++ smallRow1021_block_001
theorem smallRow1021_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1021 = smallRow1021_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1021 ++ upperNeighbors PrimePairVertices.tree_block_001 1021 = _
  rw [smallRow1021_block_000_checked, smallRow1021_block_001_checked]
  rfl

def smallRow1021_small_right_right : List Nat := smallRow1021_block_003 ++ smallRow1021_block_004
theorem smallRow1021_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1021 = smallRow1021_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1021 ++ upperNeighbors PrimePairVertices.tree_block_004 1021 = _
  rw [smallRow1021_block_003_checked, smallRow1021_block_004_checked]
  rfl

def smallRow1021_small_right : List Nat := smallRow1021_block_002 ++ smallRow1021_small_right_right
theorem smallRow1021_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1021 = smallRow1021_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1021 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1021 = _
  rw [smallRow1021_block_002_checked, smallRow1021_small_right_right_checked]
  rfl

def smallRow1021_small : List Nat := smallRow1021_small_left ++ smallRow1021_small_right
theorem smallRow1021_small_checked : upperNeighbors PrimePairVertices.tree_small 1021 = smallRow1021_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1021 ++ upperNeighbors PrimePairVertices.tree_small_right 1021 = _
  rw [smallRow1021_small_left_checked, smallRow1021_small_right_checked]
  rfl

def smallRow1021 : List Nat := smallRow1021_small
theorem smallRow1021_checked : upperNeighbors PrimePairVertices.smallTree 1021 = smallRow1021 := smallRow1021_small_checked

def smallRow1031_block_000 : List Nat := [1481, 1601, 1709, 1811, 1913, 1997]
set_option maxRecDepth 2048 in
theorem smallRow1031_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1031 = smallRow1031_block_000 := by decide

def smallRow1031_block_001 : List Nat := [2207, 2531, 2927, 3533]
set_option maxRecDepth 2048 in
theorem smallRow1031_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1031 = smallRow1031_block_001 := by decide

def smallRow1031_block_002 : List Nat := [4253, 4643, 5003, 5237]
set_option maxRecDepth 2048 in
theorem smallRow1031_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1031 = smallRow1031_block_002 := by decide

def smallRow1031_block_003 : List Nat := [6899, 7193, 7253, 7529]
set_option maxRecDepth 2048 in
theorem smallRow1031_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1031 = smallRow1031_block_003 := by decide

def smallRow1031_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1031_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1031 = smallRow1031_block_004 := by decide

def smallRow1031_small_left : List Nat := smallRow1031_block_000 ++ smallRow1031_block_001
theorem smallRow1031_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1031 = smallRow1031_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1031 ++ upperNeighbors PrimePairVertices.tree_block_001 1031 = _
  rw [smallRow1031_block_000_checked, smallRow1031_block_001_checked]
  rfl

def smallRow1031_small_right_right : List Nat := smallRow1031_block_003 ++ smallRow1031_block_004
theorem smallRow1031_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1031 = smallRow1031_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1031 ++ upperNeighbors PrimePairVertices.tree_block_004 1031 = _
  rw [smallRow1031_block_003_checked, smallRow1031_block_004_checked]
  rfl

def smallRow1031_small_right : List Nat := smallRow1031_block_002 ++ smallRow1031_small_right_right
theorem smallRow1031_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1031 = smallRow1031_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1031 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1031 = _
  rw [smallRow1031_block_002_checked, smallRow1031_small_right_right_checked]
  rfl

def smallRow1031_small : List Nat := smallRow1031_small_left ++ smallRow1031_small_right
theorem smallRow1031_small_checked : upperNeighbors PrimePairVertices.tree_small 1031 = smallRow1031_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1031 ++ upperNeighbors PrimePairVertices.tree_small_right 1031 = _
  rw [smallRow1031_small_left_checked, smallRow1031_small_right_checked]
  rfl

def smallRow1031 : List Nat := smallRow1031_small
theorem smallRow1031_checked : upperNeighbors PrimePairVertices.smallTree 1031 = smallRow1031 := smallRow1031_small_checked

def smallRow1033_block_000 : List Nat := [1249, 1543, 1663, 1759]
set_option maxRecDepth 2048 in
theorem smallRow1033_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1033 = smallRow1033_block_000 := by decide

def smallRow1033_block_001 : List Nat := [2131, 3067, 3529]
set_option maxRecDepth 2048 in
theorem smallRow1033_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1033 = smallRow1033_block_001 := by decide

def smallRow1033_block_002 : List Nat := [4231, 5233, 5869, 6091]
set_option maxRecDepth 2048 in
theorem smallRow1033_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1033 = smallRow1033_block_002 := by decide

def smallRow1033_block_003 : List Nat := [6301, 6421, 6451, 7417, 8089, 8179]
set_option maxRecDepth 2048 in
theorem smallRow1033_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1033 = smallRow1033_block_003 := by decide

def smallRow1033_block_004 : List Nat := [8269, 8353]
set_option maxRecDepth 2048 in
theorem smallRow1033_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1033 = smallRow1033_block_004 := by decide

def smallRow1033_small_left : List Nat := smallRow1033_block_000 ++ smallRow1033_block_001
theorem smallRow1033_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1033 = smallRow1033_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1033 ++ upperNeighbors PrimePairVertices.tree_block_001 1033 = _
  rw [smallRow1033_block_000_checked, smallRow1033_block_001_checked]
  rfl

def smallRow1033_small_right_right : List Nat := smallRow1033_block_003 ++ smallRow1033_block_004
theorem smallRow1033_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1033 = smallRow1033_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1033 ++ upperNeighbors PrimePairVertices.tree_block_004 1033 = _
  rw [smallRow1033_block_003_checked, smallRow1033_block_004_checked]
  rfl

def smallRow1033_small_right : List Nat := smallRow1033_block_002 ++ smallRow1033_small_right_right
theorem smallRow1033_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1033 = smallRow1033_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1033 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1033 = _
  rw [smallRow1033_block_002_checked, smallRow1033_small_right_right_checked]
  rfl

def smallRow1033_small : List Nat := smallRow1033_small_left ++ smallRow1033_small_right
theorem smallRow1033_small_checked : upperNeighbors PrimePairVertices.tree_small 1033 = smallRow1033_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1033 ++ upperNeighbors PrimePairVertices.tree_small_right 1033 = _
  rw [smallRow1033_small_left_checked, smallRow1033_small_right_checked]
  rfl

def smallRow1033 : List Nat := smallRow1033_small
theorem smallRow1033_checked : upperNeighbors PrimePairVertices.smallTree 1033 = smallRow1033 := smallRow1033_small_checked

def smallRow1039_block_000 : List Nat := [1279, 1663, 1999]
set_option maxRecDepth 2048 in
theorem smallRow1039_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1039 = smallRow1039_block_000 := by decide

def smallRow1039_block_001 : List Nat := [2467, 2659, 2917, 3457, 3637, 3967]
set_option maxRecDepth 2048 in
theorem smallRow1039_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1039 = smallRow1039_block_001 := by decide

def smallRow1039_block_002 : List Nat := [4297, 4723, 5233, 5821, 5851, 6073]
set_option maxRecDepth 2048 in
theorem smallRow1039_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1039 = smallRow1039_block_002 := by decide

def smallRow1039_block_003 : List Nat := [6709, 7993, 8161]
set_option maxRecDepth 2048 in
theorem smallRow1039_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1039 = smallRow1039_block_003 := by decide

def smallRow1039_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1039_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1039 = smallRow1039_block_004 := by decide

def smallRow1039_small_left : List Nat := smallRow1039_block_000 ++ smallRow1039_block_001
theorem smallRow1039_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1039 = smallRow1039_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1039 ++ upperNeighbors PrimePairVertices.tree_block_001 1039 = _
  rw [smallRow1039_block_000_checked, smallRow1039_block_001_checked]
  rfl

def smallRow1039_small_right_right : List Nat := smallRow1039_block_003 ++ smallRow1039_block_004
theorem smallRow1039_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1039 = smallRow1039_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1039 ++ upperNeighbors PrimePairVertices.tree_block_004 1039 = _
  rw [smallRow1039_block_003_checked, smallRow1039_block_004_checked]
  rfl

def smallRow1039_small_right : List Nat := smallRow1039_block_002 ++ smallRow1039_small_right_right
theorem smallRow1039_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1039 = smallRow1039_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1039 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1039 = _
  rw [smallRow1039_block_002_checked, smallRow1039_small_right_right_checked]
  rfl

def smallRow1039_small : List Nat := smallRow1039_small_left ++ smallRow1039_small_right
theorem smallRow1039_small_checked : upperNeighbors PrimePairVertices.tree_small 1039 = smallRow1039_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1039 ++ upperNeighbors PrimePairVertices.tree_small_right 1039 = _
  rw [smallRow1039_small_left_checked, smallRow1039_small_right_checked]
  rfl

def smallRow1039 : List Nat := smallRow1039_small
theorem smallRow1039_checked : upperNeighbors PrimePairVertices.smallTree 1039 = smallRow1039 := smallRow1039_small_checked

def smallRow1049_block_000 : List Nat := [1109, 1181, 1847]
set_option maxRecDepth 2048 in
theorem smallRow1049_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1049 = smallRow1049_block_000 := by decide

def smallRow1049_block_001 : List Nat := [2087, 2297, 2633, 3449, 3617, 3701, 3851]
set_option maxRecDepth 2048 in
theorem smallRow1049_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1049 = smallRow1049_block_001 := by decide

def smallRow1049_block_002 : List Nat := [4493, 4787, 5297, 5393, 5897]
set_option maxRecDepth 2048 in
theorem smallRow1049_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1049 = smallRow1049_block_002 := by decide

def smallRow1049_block_003 : List Nat := [6203, 6971, 7013, 7229, 7523, 7559, 7577, 7607, 7841]
set_option maxRecDepth 2048 in
theorem smallRow1049_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1049 = smallRow1049_block_003 := by decide

def smallRow1049_block_004 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallRow1049_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1049 = smallRow1049_block_004 := by decide

def smallRow1049_small_left : List Nat := smallRow1049_block_000 ++ smallRow1049_block_001
theorem smallRow1049_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1049 = smallRow1049_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1049 ++ upperNeighbors PrimePairVertices.tree_block_001 1049 = _
  rw [smallRow1049_block_000_checked, smallRow1049_block_001_checked]
  rfl

def smallRow1049_small_right_right : List Nat := smallRow1049_block_003 ++ smallRow1049_block_004
theorem smallRow1049_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1049 = smallRow1049_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1049 ++ upperNeighbors PrimePairVertices.tree_block_004 1049 = _
  rw [smallRow1049_block_003_checked, smallRow1049_block_004_checked]
  rfl

def smallRow1049_small_right : List Nat := smallRow1049_block_002 ++ smallRow1049_small_right_right
theorem smallRow1049_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1049 = smallRow1049_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1049 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1049 = _
  rw [smallRow1049_block_002_checked, smallRow1049_small_right_right_checked]
  rfl

def smallRow1049_small : List Nat := smallRow1049_small_left ++ smallRow1049_small_right
theorem smallRow1049_small_checked : upperNeighbors PrimePairVertices.tree_small 1049 = smallRow1049_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1049 ++ upperNeighbors PrimePairVertices.tree_small_right 1049 = _
  rw [smallRow1049_small_left_checked, smallRow1049_small_right_checked]
  rfl

def smallRow1049 : List Nat := smallRow1049_small
theorem smallRow1049_checked : upperNeighbors PrimePairVertices.smallTree 1049 = smallRow1049 := smallRow1049_small_checked

def smallRow1051_block_000 : List Nat := [2017]
set_option maxRecDepth 2048 in
theorem smallRow1051_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1051 = smallRow1051_block_000 := by decide

def smallRow1051_block_001 : List Nat := [2221, 2347, 2389, 2617, 2713, 3067, 3271, 3331, 3583, 3709, 3739, 4003, 4057]
set_option maxRecDepth 2048 in
theorem smallRow1051_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1051 = smallRow1051_block_001 := by decide

def smallRow1051_block_002 : List Nat := [4507, 4729, 4813, 4909, 5443, 5839]
set_option maxRecDepth 2048 in
theorem smallRow1051_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1051 = smallRow1051_block_002 := by decide

def smallRow1051_block_003 : List Nat := [6301, 6607, 7867]
set_option maxRecDepth 2048 in
theorem smallRow1051_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1051 = smallRow1051_block_003 := by decide

def smallRow1051_block_004 : List Nat := [8311, 8353]
set_option maxRecDepth 2048 in
theorem smallRow1051_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1051 = smallRow1051_block_004 := by decide

def smallRow1051_small_left : List Nat := smallRow1051_block_000 ++ smallRow1051_block_001
theorem smallRow1051_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1051 = smallRow1051_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1051 ++ upperNeighbors PrimePairVertices.tree_block_001 1051 = _
  rw [smallRow1051_block_000_checked, smallRow1051_block_001_checked]
  rfl

def smallRow1051_small_right_right : List Nat := smallRow1051_block_003 ++ smallRow1051_block_004
theorem smallRow1051_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1051 = smallRow1051_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1051 ++ upperNeighbors PrimePairVertices.tree_block_004 1051 = _
  rw [smallRow1051_block_003_checked, smallRow1051_block_004_checked]
  rfl

def smallRow1051_small_right : List Nat := smallRow1051_block_002 ++ smallRow1051_small_right_right
theorem smallRow1051_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1051 = smallRow1051_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1051 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1051 = _
  rw [smallRow1051_block_002_checked, smallRow1051_small_right_right_checked]
  rfl

def smallRow1051_small : List Nat := smallRow1051_small_left ++ smallRow1051_small_right
theorem smallRow1051_small_checked : upperNeighbors PrimePairVertices.tree_small 1051 = smallRow1051_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1051 ++ upperNeighbors PrimePairVertices.tree_small_right 1051 = _
  rw [smallRow1051_small_left_checked, smallRow1051_small_right_checked]
  rfl

def smallRow1051 : List Nat := smallRow1051_small
theorem smallRow1051_checked : upperNeighbors PrimePairVertices.smallTree 1051 = smallRow1051 := smallRow1051_small_checked

def smallRow1061_block_000 : List Nat := [1151, 1229, 1493, 1619, 2027]
set_option maxRecDepth 2048 in
theorem smallRow1061_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1061 = smallRow1061_block_000 := by decide

def smallRow1061_block_001 : List Nat := [2213, 2711, 3041, 3167, 3257, 3767]
set_option maxRecDepth 2048 in
theorem smallRow1061_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1061 = smallRow1061_block_001 := by decide

def smallRow1061_block_002 : List Nat := [4139, 4391, 4463, 4649, 4733, 4973, 5147, 5477]
set_option maxRecDepth 2048 in
theorem smallRow1061_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1061 = smallRow1061_block_002 := by decide

def smallRow1061_block_003 : List Nat := [6197, 6329, 7589, 8117]
set_option maxRecDepth 2048 in
theorem smallRow1061_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1061 = smallRow1061_block_003 := by decide

def smallRow1061_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1061_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1061 = smallRow1061_block_004 := by decide

def smallRow1061_small_left : List Nat := smallRow1061_block_000 ++ smallRow1061_block_001
theorem smallRow1061_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1061 = smallRow1061_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1061 ++ upperNeighbors PrimePairVertices.tree_block_001 1061 = _
  rw [smallRow1061_block_000_checked, smallRow1061_block_001_checked]
  rfl

def smallRow1061_small_right_right : List Nat := smallRow1061_block_003 ++ smallRow1061_block_004
theorem smallRow1061_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1061 = smallRow1061_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1061 ++ upperNeighbors PrimePairVertices.tree_block_004 1061 = _
  rw [smallRow1061_block_003_checked, smallRow1061_block_004_checked]
  rfl

def smallRow1061_small_right : List Nat := smallRow1061_block_002 ++ smallRow1061_small_right_right
theorem smallRow1061_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1061 = smallRow1061_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1061 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1061 = _
  rw [smallRow1061_block_002_checked, smallRow1061_small_right_right_checked]
  rfl

def smallRow1061_small : List Nat := smallRow1061_small_left ++ smallRow1061_small_right
theorem smallRow1061_small_checked : upperNeighbors PrimePairVertices.tree_small 1061 = smallRow1061_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1061 ++ upperNeighbors PrimePairVertices.tree_small_right 1061 = _
  rw [smallRow1061_small_left_checked, smallRow1061_small_right_checked]
  rfl

def smallRow1061 : List Nat := smallRow1061_small
theorem smallRow1061_checked : upperNeighbors PrimePairVertices.smallTree 1061 = smallRow1061 := smallRow1061_small_checked

end ProjectEulerSolutions.PrimePairCache
