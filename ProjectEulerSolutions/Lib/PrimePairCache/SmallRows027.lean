import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1367_block_000 : List Nat := [1733, 2027]
set_option maxRecDepth 2048 in
theorem smallRow1367_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1367 = smallRow1367_block_000 := by decide

def smallRow1367_block_001 : List Nat := [2297, 2591, 2843, 2963, 3221, 3467, 3929, 4079]
set_option maxRecDepth 2048 in
theorem smallRow1367_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1367 = smallRow1367_block_001 := by decide

def smallRow1367_block_002 : List Nat := [4373, 4691, 4733, 4877, 5471]
set_option maxRecDepth 2048 in
theorem smallRow1367_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1367 = smallRow1367_block_002 := by decide

def smallRow1367_block_003 : List Nat := [6323, 6353, 6563, 7043, 7229, 7919, 7949]
set_option maxRecDepth 2048 in
theorem smallRow1367_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1367 = smallRow1367_block_003 := by decide

def smallRow1367_block_004 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallRow1367_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1367 = smallRow1367_block_004 := by decide

def smallRow1367_small_left : List Nat := smallRow1367_block_000 ++ smallRow1367_block_001
theorem smallRow1367_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1367 = smallRow1367_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1367 ++ upperNeighbors PrimePairVertices.tree_block_001 1367 = _
  rw [smallRow1367_block_000_checked, smallRow1367_block_001_checked]
  rfl

def smallRow1367_small_right_right : List Nat := smallRow1367_block_003 ++ smallRow1367_block_004
theorem smallRow1367_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1367 = smallRow1367_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1367 ++ upperNeighbors PrimePairVertices.tree_block_004 1367 = _
  rw [smallRow1367_block_003_checked, smallRow1367_block_004_checked]
  rfl

def smallRow1367_small_right : List Nat := smallRow1367_block_002 ++ smallRow1367_small_right_right
theorem smallRow1367_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1367 = smallRow1367_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1367 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1367 = _
  rw [smallRow1367_block_002_checked, smallRow1367_small_right_right_checked]
  rfl

def smallRow1367_small : List Nat := smallRow1367_small_left ++ smallRow1367_small_right
theorem smallRow1367_small_checked : upperNeighbors PrimePairVertices.tree_small 1367 = smallRow1367_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1367 ++ upperNeighbors PrimePairVertices.tree_small_right 1367 = _
  rw [smallRow1367_small_left_checked, smallRow1367_small_right_checked]
  rfl

def smallRow1367 : List Nat := smallRow1367_small
theorem smallRow1367_checked : upperNeighbors PrimePairVertices.smallTree 1367 = smallRow1367 := smallRow1367_small_checked

def smallRow1373_block_000 : List Nat := [1439, 1451, 1499, 1511, 1637, 1901]
set_option maxRecDepth 2048 in
theorem smallRow1373_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1373 = smallRow1373_block_000 := by decide

def smallRow1373_block_001 : List Nat := [2423, 2531, 2591, 3023, 3539]
set_option maxRecDepth 2048 in
theorem smallRow1373_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1373 = smallRow1373_block_001 := by decide

def smallRow1373_block_002 : List Nat := [4493, 5333, 6143]
set_option maxRecDepth 2048 in
theorem smallRow1373_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1373 = smallRow1373_block_002 := by decide

def smallRow1373_block_003 : List Nat := [6491, 6803, 7541, 7727, 7907, 7937]
set_option maxRecDepth 2048 in
theorem smallRow1373_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1373 = smallRow1373_block_003 := by decide

def smallRow1373_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1373_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1373 = smallRow1373_block_004 := by decide

def smallRow1373_small_left : List Nat := smallRow1373_block_000 ++ smallRow1373_block_001
theorem smallRow1373_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1373 = smallRow1373_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1373 ++ upperNeighbors PrimePairVertices.tree_block_001 1373 = _
  rw [smallRow1373_block_000_checked, smallRow1373_block_001_checked]
  rfl

def smallRow1373_small_right_right : List Nat := smallRow1373_block_003 ++ smallRow1373_block_004
theorem smallRow1373_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1373 = smallRow1373_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1373 ++ upperNeighbors PrimePairVertices.tree_block_004 1373 = _
  rw [smallRow1373_block_003_checked, smallRow1373_block_004_checked]
  rfl

def smallRow1373_small_right : List Nat := smallRow1373_block_002 ++ smallRow1373_small_right_right
theorem smallRow1373_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1373 = smallRow1373_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1373 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1373 = _
  rw [smallRow1373_block_002_checked, smallRow1373_small_right_right_checked]
  rfl

def smallRow1373_small : List Nat := smallRow1373_small_left ++ smallRow1373_small_right
theorem smallRow1373_small_checked : upperNeighbors PrimePairVertices.tree_small 1373 = smallRow1373_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1373 ++ upperNeighbors PrimePairVertices.tree_small_right 1373 = _
  rw [smallRow1373_small_left_checked, smallRow1373_small_right_checked]
  rfl

def smallRow1373 : List Nat := smallRow1373_small
theorem smallRow1373_checked : upperNeighbors PrimePairVertices.smallTree 1373 = smallRow1373 := smallRow1373_small_checked

def smallRow1381_block_000 : List Nat := [1657, 1789]
set_option maxRecDepth 2048 in
theorem smallRow1381_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1381 = smallRow1381_block_000 := by decide

def smallRow1381_block_001 : List Nat := [2683, 2767, 3169, 4027]
set_option maxRecDepth 2048 in
theorem smallRow1381_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1381 = smallRow1381_block_001 := by decide

def smallRow1381_block_002 : List Nat := [4111, 5233]
set_option maxRecDepth 2048 in
theorem smallRow1381_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1381 = smallRow1381_block_002 := by decide

def smallRow1381_block_003 : List Nat := [6571, 6823, 7933, 8017]
set_option maxRecDepth 2048 in
theorem smallRow1381_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1381 = smallRow1381_block_003 := by decide

def smallRow1381_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1381_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1381 = smallRow1381_block_004 := by decide

def smallRow1381_small_left : List Nat := smallRow1381_block_000 ++ smallRow1381_block_001
theorem smallRow1381_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1381 = smallRow1381_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1381 ++ upperNeighbors PrimePairVertices.tree_block_001 1381 = _
  rw [smallRow1381_block_000_checked, smallRow1381_block_001_checked]
  rfl

def smallRow1381_small_right_right : List Nat := smallRow1381_block_003 ++ smallRow1381_block_004
theorem smallRow1381_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1381 = smallRow1381_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1381 ++ upperNeighbors PrimePairVertices.tree_block_004 1381 = _
  rw [smallRow1381_block_003_checked, smallRow1381_block_004_checked]
  rfl

def smallRow1381_small_right : List Nat := smallRow1381_block_002 ++ smallRow1381_small_right_right
theorem smallRow1381_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1381 = smallRow1381_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1381 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1381 = _
  rw [smallRow1381_block_002_checked, smallRow1381_small_right_right_checked]
  rfl

def smallRow1381_small : List Nat := smallRow1381_small_left ++ smallRow1381_small_right
theorem smallRow1381_small_checked : upperNeighbors PrimePairVertices.tree_small 1381 = smallRow1381_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1381 ++ upperNeighbors PrimePairVertices.tree_small_right 1381 = _
  rw [smallRow1381_small_left_checked, smallRow1381_small_right_checked]
  rfl

def smallRow1381 : List Nat := smallRow1381_small
theorem smallRow1381_checked : upperNeighbors PrimePairVertices.smallTree 1381 = smallRow1381 := smallRow1381_small_checked

def smallRow1399_block_000 : List Nat := [1489, 1741]
set_option maxRecDepth 2048 in
theorem smallRow1399_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1399 = smallRow1399_block_000 := by decide

def smallRow1399_block_001 : List Nat := [2131, 2551, 3433]
set_option maxRecDepth 2048 in
theorem smallRow1399_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1399 = smallRow1399_block_001 := by decide

def smallRow1399_block_002 : List Nat := [4261, 4339, 4801, 4987, 5881, 6133]
set_option maxRecDepth 2048 in
theorem smallRow1399_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1399 = smallRow1399_block_002 := by decide

def smallRow1399_block_003 : List Nat := [6481, 6529, 6793, 7759]
set_option maxRecDepth 2048 in
theorem smallRow1399_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1399 = smallRow1399_block_003 := by decide

def smallRow1399_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1399_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1399 = smallRow1399_block_004 := by decide

def smallRow1399_small_left : List Nat := smallRow1399_block_000 ++ smallRow1399_block_001
theorem smallRow1399_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1399 = smallRow1399_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1399 ++ upperNeighbors PrimePairVertices.tree_block_001 1399 = _
  rw [smallRow1399_block_000_checked, smallRow1399_block_001_checked]
  rfl

def smallRow1399_small_right_right : List Nat := smallRow1399_block_003 ++ smallRow1399_block_004
theorem smallRow1399_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1399 = smallRow1399_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1399 ++ upperNeighbors PrimePairVertices.tree_block_004 1399 = _
  rw [smallRow1399_block_003_checked, smallRow1399_block_004_checked]
  rfl

def smallRow1399_small_right : List Nat := smallRow1399_block_002 ++ smallRow1399_small_right_right
theorem smallRow1399_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1399 = smallRow1399_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1399 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1399 = _
  rw [smallRow1399_block_002_checked, smallRow1399_small_right_right_checked]
  rfl

def smallRow1399_small : List Nat := smallRow1399_small_left ++ smallRow1399_small_right
theorem smallRow1399_small_checked : upperNeighbors PrimePairVertices.tree_small 1399 = smallRow1399_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1399 ++ upperNeighbors PrimePairVertices.tree_small_right 1399 = _
  rw [smallRow1399_small_left_checked, smallRow1399_small_right_checked]
  rfl

def smallRow1399 : List Nat := smallRow1399_small
theorem smallRow1399_checked : upperNeighbors PrimePairVertices.smallTree 1399 = smallRow1399 := smallRow1399_small_checked

def smallRow1409_block_000 : List Nat := [1499]
set_option maxRecDepth 2048 in
theorem smallRow1409_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1409 = smallRow1409_block_000 := by decide

def smallRow1409_block_001 : List Nat := [3329, 4013]
set_option maxRecDepth 2048 in
theorem smallRow1409_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1409 = smallRow1409_block_001 := by decide

def smallRow1409_block_002 : List Nat := [4517, 4643, 5231, 5387, 6029]
set_option maxRecDepth 2048 in
theorem smallRow1409_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1409 = smallRow1409_block_002 := by decide

def smallRow1409_block_003 : List Nat := [6521, 6857, 6983, 7247, 7817, 7907]
set_option maxRecDepth 2048 in
theorem smallRow1409_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1409 = smallRow1409_block_003 := by decide

def smallRow1409_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1409_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1409 = smallRow1409_block_004 := by decide

def smallRow1409_small_left : List Nat := smallRow1409_block_000 ++ smallRow1409_block_001
theorem smallRow1409_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1409 = smallRow1409_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1409 ++ upperNeighbors PrimePairVertices.tree_block_001 1409 = _
  rw [smallRow1409_block_000_checked, smallRow1409_block_001_checked]
  rfl

def smallRow1409_small_right_right : List Nat := smallRow1409_block_003 ++ smallRow1409_block_004
theorem smallRow1409_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1409 = smallRow1409_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1409 ++ upperNeighbors PrimePairVertices.tree_block_004 1409 = _
  rw [smallRow1409_block_003_checked, smallRow1409_block_004_checked]
  rfl

def smallRow1409_small_right : List Nat := smallRow1409_block_002 ++ smallRow1409_small_right_right
theorem smallRow1409_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1409 = smallRow1409_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1409 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1409 = _
  rw [smallRow1409_block_002_checked, smallRow1409_small_right_right_checked]
  rfl

def smallRow1409_small : List Nat := smallRow1409_small_left ++ smallRow1409_small_right
theorem smallRow1409_small_checked : upperNeighbors PrimePairVertices.tree_small 1409 = smallRow1409_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1409 ++ upperNeighbors PrimePairVertices.tree_small_right 1409 = _
  rw [smallRow1409_small_left_checked, smallRow1409_small_right_checked]
  rfl

def smallRow1409 : List Nat := smallRow1409_small
theorem smallRow1409_checked : upperNeighbors PrimePairVertices.smallTree 1409 = smallRow1409 := smallRow1409_small_checked

def smallRow1423_block_000 : List Nat := [1699, 1999, 2017]
set_option maxRecDepth 2048 in
theorem smallRow1423_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1423 = smallRow1423_block_000 := by decide

def smallRow1423_block_001 : List Nat := [2137, 2287, 2389, 2797, 3187, 3253, 4027, 4057]
set_option maxRecDepth 2048 in
theorem smallRow1423_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1423 = smallRow1423_block_001 := by decide

def smallRow1423_block_002 : List Nat := [5521, 5791, 5821]
set_option maxRecDepth 2048 in
theorem smallRow1423_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1423 = smallRow1423_block_002 := by decide

def smallRow1423_block_003 : List Nat := [6337, 6367, 6427, 8017]
set_option maxRecDepth 2048 in
theorem smallRow1423_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1423 = smallRow1423_block_003 := by decide

def smallRow1423_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1423_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1423 = smallRow1423_block_004 := by decide

def smallRow1423_small_left : List Nat := smallRow1423_block_000 ++ smallRow1423_block_001
theorem smallRow1423_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1423 = smallRow1423_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1423 ++ upperNeighbors PrimePairVertices.tree_block_001 1423 = _
  rw [smallRow1423_block_000_checked, smallRow1423_block_001_checked]
  rfl

def smallRow1423_small_right_right : List Nat := smallRow1423_block_003 ++ smallRow1423_block_004
theorem smallRow1423_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1423 = smallRow1423_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1423 ++ upperNeighbors PrimePairVertices.tree_block_004 1423 = _
  rw [smallRow1423_block_003_checked, smallRow1423_block_004_checked]
  rfl

def smallRow1423_small_right : List Nat := smallRow1423_block_002 ++ smallRow1423_small_right_right
theorem smallRow1423_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1423 = smallRow1423_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1423 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1423 = _
  rw [smallRow1423_block_002_checked, smallRow1423_small_right_right_checked]
  rfl

def smallRow1423_small : List Nat := smallRow1423_small_left ++ smallRow1423_small_right
theorem smallRow1423_small_checked : upperNeighbors PrimePairVertices.tree_small 1423 = smallRow1423_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1423 ++ upperNeighbors PrimePairVertices.tree_small_right 1423 = _
  rw [smallRow1423_small_left_checked, smallRow1423_small_right_checked]
  rfl

def smallRow1423 : List Nat := smallRow1423_small
theorem smallRow1423_checked : upperNeighbors PrimePairVertices.smallTree 1423 = smallRow1423 := smallRow1423_small_checked

def smallRow1427_block_000 : List Nat := [1931]
set_option maxRecDepth 2048 in
theorem smallRow1427_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1427 = smallRow1427_block_000 := by decide

def smallRow1427_block_001 : List Nat := [2339, 2777, 3011, 3359, 3767, 3989, 4019]
set_option maxRecDepth 2048 in
theorem smallRow1427_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1427 = smallRow1427_block_001 := by decide

def smallRow1427_block_002 : List Nat := [4157, 4451, 5297, 5477, 5501, 5843]
set_option maxRecDepth 2048 in
theorem smallRow1427_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1427 = smallRow1427_block_002 := by decide

def smallRow1427_block_003 : List Nat := [7451, 7937, 8111]
set_option maxRecDepth 2048 in
theorem smallRow1427_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1427 = smallRow1427_block_003 := by decide

def smallRow1427_block_004 : List Nat := [8273]
set_option maxRecDepth 2048 in
theorem smallRow1427_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1427 = smallRow1427_block_004 := by decide

def smallRow1427_small_left : List Nat := smallRow1427_block_000 ++ smallRow1427_block_001
theorem smallRow1427_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1427 = smallRow1427_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1427 ++ upperNeighbors PrimePairVertices.tree_block_001 1427 = _
  rw [smallRow1427_block_000_checked, smallRow1427_block_001_checked]
  rfl

def smallRow1427_small_right_right : List Nat := smallRow1427_block_003 ++ smallRow1427_block_004
theorem smallRow1427_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1427 = smallRow1427_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1427 ++ upperNeighbors PrimePairVertices.tree_block_004 1427 = _
  rw [smallRow1427_block_003_checked, smallRow1427_block_004_checked]
  rfl

def smallRow1427_small_right : List Nat := smallRow1427_block_002 ++ smallRow1427_small_right_right
theorem smallRow1427_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1427 = smallRow1427_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1427 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1427 = _
  rw [smallRow1427_block_002_checked, smallRow1427_small_right_right_checked]
  rfl

def smallRow1427_small : List Nat := smallRow1427_small_left ++ smallRow1427_small_right
theorem smallRow1427_small_checked : upperNeighbors PrimePairVertices.tree_small 1427 = smallRow1427_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1427 ++ upperNeighbors PrimePairVertices.tree_small_right 1427 = _
  rw [smallRow1427_small_left_checked, smallRow1427_small_right_checked]
  rfl

def smallRow1427 : List Nat := smallRow1427_small
theorem smallRow1427_checked : upperNeighbors PrimePairVertices.smallTree 1427 = smallRow1427 := smallRow1427_small_checked

def smallRow1429_block_000 : List Nat := [1723, 1831]
set_option maxRecDepth 2048 in
theorem smallRow1429_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1429 = smallRow1429_block_000 := by decide

def smallRow1429_block_001 : List Nat := [2221, 3313, 3691, 3739, 3919]
set_option maxRecDepth 2048 in
theorem smallRow1429_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1429 = smallRow1429_block_001 := by decide

def smallRow1429_block_002 : List Nat := [4957]
set_option maxRecDepth 2048 in
theorem smallRow1429_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1429 = smallRow1429_block_002 := by decide

def smallRow1429_block_003 : List Nat := [6211, 6343, 6427, 6553, 6619, 6967, 7351, 7477, 7963, 8101]
set_option maxRecDepth 2048 in
theorem smallRow1429_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1429 = smallRow1429_block_003 := by decide

def smallRow1429_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1429_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1429 = smallRow1429_block_004 := by decide

def smallRow1429_small_left : List Nat := smallRow1429_block_000 ++ smallRow1429_block_001
theorem smallRow1429_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1429 = smallRow1429_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1429 ++ upperNeighbors PrimePairVertices.tree_block_001 1429 = _
  rw [smallRow1429_block_000_checked, smallRow1429_block_001_checked]
  rfl

def smallRow1429_small_right_right : List Nat := smallRow1429_block_003 ++ smallRow1429_block_004
theorem smallRow1429_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1429 = smallRow1429_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1429 ++ upperNeighbors PrimePairVertices.tree_block_004 1429 = _
  rw [smallRow1429_block_003_checked, smallRow1429_block_004_checked]
  rfl

def smallRow1429_small_right : List Nat := smallRow1429_block_002 ++ smallRow1429_small_right_right
theorem smallRow1429_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1429 = smallRow1429_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1429 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1429 = _
  rw [smallRow1429_block_002_checked, smallRow1429_small_right_right_checked]
  rfl

def smallRow1429_small : List Nat := smallRow1429_small_left ++ smallRow1429_small_right
theorem smallRow1429_small_checked : upperNeighbors PrimePairVertices.tree_small 1429 = smallRow1429_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1429 ++ upperNeighbors PrimePairVertices.tree_small_right 1429 = _
  rw [smallRow1429_small_left_checked, smallRow1429_small_right_checked]
  rfl

def smallRow1429 : List Nat := smallRow1429_small
theorem smallRow1429_checked : upperNeighbors PrimePairVertices.smallTree 1429 = smallRow1429 := smallRow1429_small_checked

end ProjectEulerSolutions.PrimePairCache
