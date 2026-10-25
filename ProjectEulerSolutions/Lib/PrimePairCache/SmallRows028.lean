import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1433_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1433_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1433 = smallRow1433_block_000 := by decide

def smallRow1433_block_001 : List Nat := [2663, 2957, 2999, 3089, 3323]
set_option maxRecDepth 2048 in
theorem smallRow1433_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1433 = smallRow1433_block_001 := by decide

def smallRow1433_block_002 : List Nat := [4157, 4283, 4637, 4643, 5393, 5501, 5903, 6029]
set_option maxRecDepth 2048 in
theorem smallRow1433_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1433 = smallRow1433_block_002 := by decide

def smallRow1433_block_003 : List Nat := [6299, 6737, 6761, 7607, 7793, 7937]
set_option maxRecDepth 2048 in
theorem smallRow1433_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1433 = smallRow1433_block_003 := by decide

def smallRow1433_block_004 : List Nat := [8297]
set_option maxRecDepth 2048 in
theorem smallRow1433_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1433 = smallRow1433_block_004 := by decide

def smallRow1433_small_left : List Nat := smallRow1433_block_000 ++ smallRow1433_block_001
theorem smallRow1433_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1433 = smallRow1433_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1433 ++ upperNeighbors PrimePairVertices.tree_block_001 1433 = _
  rw [smallRow1433_block_000_checked, smallRow1433_block_001_checked]
  rfl

def smallRow1433_small_right_right : List Nat := smallRow1433_block_003 ++ smallRow1433_block_004
theorem smallRow1433_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1433 = smallRow1433_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1433 ++ upperNeighbors PrimePairVertices.tree_block_004 1433 = _
  rw [smallRow1433_block_003_checked, smallRow1433_block_004_checked]
  rfl

def smallRow1433_small_right : List Nat := smallRow1433_block_002 ++ smallRow1433_small_right_right
theorem smallRow1433_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1433 = smallRow1433_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1433 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1433 = _
  rw [smallRow1433_block_002_checked, smallRow1433_small_right_right_checked]
  rfl

def smallRow1433_small : List Nat := smallRow1433_small_left ++ smallRow1433_small_right
theorem smallRow1433_small_checked : upperNeighbors PrimePairVertices.tree_small 1433 = smallRow1433_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1433 ++ upperNeighbors PrimePairVertices.tree_small_right 1433 = _
  rw [smallRow1433_small_left_checked, smallRow1433_small_right_checked]
  rfl

def smallRow1433 : List Nat := smallRow1433_small
theorem smallRow1433_checked : upperNeighbors PrimePairVertices.smallTree 1433 = smallRow1433 := smallRow1433_small_checked

def smallRow1439_block_000 : List Nat := [2003]
set_option maxRecDepth 2048 in
theorem smallRow1439_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1439 = smallRow1439_block_000 := by decide

def smallRow1439_block_001 : List Nat := [2243, 2579, 2837, 3413, 3719, 3833, 4007, 4091]
set_option maxRecDepth 2048 in
theorem smallRow1439_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1439 = smallRow1439_block_001 := by decide

def smallRow1439_block_002 : List Nat := [4271, 4337, 4463, 4931, 5021, 5231, 5261, 5273, 5279, 5657, 5783]
set_option maxRecDepth 2048 in
theorem smallRow1439_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1439 = smallRow1439_block_002 := by decide

def smallRow1439_block_003 : List Nat := [6581, 6659, 6869, 6983, 7127, 7583, 8039, 8081]
set_option maxRecDepth 2048 in
theorem smallRow1439_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1439 = smallRow1439_block_003 := by decide

def smallRow1439_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1439_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1439 = smallRow1439_block_004 := by decide

def smallRow1439_small_left : List Nat := smallRow1439_block_000 ++ smallRow1439_block_001
theorem smallRow1439_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1439 = smallRow1439_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1439 ++ upperNeighbors PrimePairVertices.tree_block_001 1439 = _
  rw [smallRow1439_block_000_checked, smallRow1439_block_001_checked]
  rfl

def smallRow1439_small_right_right : List Nat := smallRow1439_block_003 ++ smallRow1439_block_004
theorem smallRow1439_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1439 = smallRow1439_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1439 ++ upperNeighbors PrimePairVertices.tree_block_004 1439 = _
  rw [smallRow1439_block_003_checked, smallRow1439_block_004_checked]
  rfl

def smallRow1439_small_right : List Nat := smallRow1439_block_002 ++ smallRow1439_small_right_right
theorem smallRow1439_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1439 = smallRow1439_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1439 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1439 = _
  rw [smallRow1439_block_002_checked, smallRow1439_small_right_right_checked]
  rfl

def smallRow1439_small : List Nat := smallRow1439_small_left ++ smallRow1439_small_right
theorem smallRow1439_small_checked : upperNeighbors PrimePairVertices.tree_small 1439 = smallRow1439_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1439 ++ upperNeighbors PrimePairVertices.tree_small_right 1439 = _
  rw [smallRow1439_small_left_checked, smallRow1439_small_right_checked]
  rfl

def smallRow1439 : List Nat := smallRow1439_small
theorem smallRow1439_checked : upperNeighbors PrimePairVertices.smallTree 1439 = smallRow1439 := smallRow1439_small_checked

def smallRow1447_block_000 : List Nat := [1543, 1549, 1753, 1993, 2011]
set_option maxRecDepth 2048 in
theorem smallRow1447_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1447 = smallRow1447_block_000 := by decide

def smallRow1447_block_001 : List Nat := [2347, 2719, 2803, 2833, 3229, 3271, 3313, 3709, 3769, 4051, 4093]
set_option maxRecDepth 2048 in
theorem smallRow1447_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1447 = smallRow1447_block_001 := by decide

def smallRow1447_block_002 : List Nat := [4441, 5641, 5683, 5689]
set_option maxRecDepth 2048 in
theorem smallRow1447_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1447 = smallRow1447_block_002 := by decide

def smallRow1447_block_003 : List Nat := [6949, 6967, 6991, 7759, 8059]
set_option maxRecDepth 2048 in
theorem smallRow1447_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1447 = smallRow1447_block_003 := by decide

def smallRow1447_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1447_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1447 = smallRow1447_block_004 := by decide

def smallRow1447_small_left : List Nat := smallRow1447_block_000 ++ smallRow1447_block_001
theorem smallRow1447_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1447 = smallRow1447_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1447 ++ upperNeighbors PrimePairVertices.tree_block_001 1447 = _
  rw [smallRow1447_block_000_checked, smallRow1447_block_001_checked]
  rfl

def smallRow1447_small_right_right : List Nat := smallRow1447_block_003 ++ smallRow1447_block_004
theorem smallRow1447_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1447 = smallRow1447_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1447 ++ upperNeighbors PrimePairVertices.tree_block_004 1447 = _
  rw [smallRow1447_block_003_checked, smallRow1447_block_004_checked]
  rfl

def smallRow1447_small_right : List Nat := smallRow1447_block_002 ++ smallRow1447_small_right_right
theorem smallRow1447_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1447 = smallRow1447_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1447 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1447 = _
  rw [smallRow1447_block_002_checked, smallRow1447_small_right_right_checked]
  rfl

def smallRow1447_small : List Nat := smallRow1447_small_left ++ smallRow1447_small_right
theorem smallRow1447_small_checked : upperNeighbors PrimePairVertices.tree_small 1447 = smallRow1447_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1447 ++ upperNeighbors PrimePairVertices.tree_small_right 1447 = _
  rw [smallRow1447_small_left_checked, smallRow1447_small_right_checked]
  rfl

def smallRow1447 : List Nat := smallRow1447_small
theorem smallRow1447_checked : upperNeighbors PrimePairVertices.smallTree 1447 = smallRow1447 := smallRow1447_small_checked

def smallRow1451_block_000 : List Nat := [1481, 2039]
set_option maxRecDepth 2048 in
theorem smallRow1451_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1451 = smallRow1451_block_000 := by decide

def smallRow1451_block_001 : List Nat := [2699, 3413, 3557, 3761]
set_option maxRecDepth 2048 in
theorem smallRow1451_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1451 = smallRow1451_block_001 := by decide

def smallRow1451_block_002 : List Nat := [4373, 4457, 5147]
set_option maxRecDepth 2048 in
theorem smallRow1451_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1451 = smallRow1451_block_002 := by decide

def smallRow1451_block_003 : List Nat := [6581, 6659, 6869, 7001, 7193, 7499, 7823, 7883, 7907]
set_option maxRecDepth 2048 in
theorem smallRow1451_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1451 = smallRow1451_block_003 := by decide

def smallRow1451_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1451_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1451 = smallRow1451_block_004 := by decide

def smallRow1451_small_left : List Nat := smallRow1451_block_000 ++ smallRow1451_block_001
theorem smallRow1451_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1451 = smallRow1451_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1451 ++ upperNeighbors PrimePairVertices.tree_block_001 1451 = _
  rw [smallRow1451_block_000_checked, smallRow1451_block_001_checked]
  rfl

def smallRow1451_small_right_right : List Nat := smallRow1451_block_003 ++ smallRow1451_block_004
theorem smallRow1451_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1451 = smallRow1451_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1451 ++ upperNeighbors PrimePairVertices.tree_block_004 1451 = _
  rw [smallRow1451_block_003_checked, smallRow1451_block_004_checked]
  rfl

def smallRow1451_small_right : List Nat := smallRow1451_block_002 ++ smallRow1451_small_right_right
theorem smallRow1451_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1451 = smallRow1451_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1451 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1451 = _
  rw [smallRow1451_block_002_checked, smallRow1451_small_right_right_checked]
  rfl

def smallRow1451_small : List Nat := smallRow1451_small_left ++ smallRow1451_small_right
theorem smallRow1451_small_checked : upperNeighbors PrimePairVertices.tree_small 1451 = smallRow1451_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1451 ++ upperNeighbors PrimePairVertices.tree_small_right 1451 = _
  rw [smallRow1451_small_left_checked, smallRow1451_small_right_checked]
  rfl

def smallRow1451 : List Nat := smallRow1451_small
theorem smallRow1451_checked : upperNeighbors PrimePairVertices.smallTree 1451 = smallRow1451 := smallRow1451_small_checked

def smallRow1453_block_000 : List Nat := [1669, 1879]
set_option maxRecDepth 2048 in
theorem smallRow1453_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1453 = smallRow1453_block_000 := by decide

def smallRow1453_block_001 : List Nat := [2437, 2593, 2797, 3049, 3361]
set_option maxRecDepth 2048 in
theorem smallRow1453_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1453 = smallRow1453_block_001 := by decide

def smallRow1453_block_002 : List Nat := [4483, 4651, 4783, 5113, 5281, 5581, 5743, 5923]
set_option maxRecDepth 2048 in
theorem smallRow1453_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1453 = smallRow1453_block_002 := by decide

def smallRow1453_block_003 : List Nat := [6763, 7129, 7417, 7927, 7933, 8089]
set_option maxRecDepth 2048 in
theorem smallRow1453_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1453 = smallRow1453_block_003 := by decide

def smallRow1453_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1453_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1453 = smallRow1453_block_004 := by decide

def smallRow1453_small_left : List Nat := smallRow1453_block_000 ++ smallRow1453_block_001
theorem smallRow1453_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1453 = smallRow1453_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1453 ++ upperNeighbors PrimePairVertices.tree_block_001 1453 = _
  rw [smallRow1453_block_000_checked, smallRow1453_block_001_checked]
  rfl

def smallRow1453_small_right_right : List Nat := smallRow1453_block_003 ++ smallRow1453_block_004
theorem smallRow1453_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1453 = smallRow1453_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1453 ++ upperNeighbors PrimePairVertices.tree_block_004 1453 = _
  rw [smallRow1453_block_003_checked, smallRow1453_block_004_checked]
  rfl

def smallRow1453_small_right : List Nat := smallRow1453_block_002 ++ smallRow1453_small_right_right
theorem smallRow1453_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1453 = smallRow1453_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1453 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1453 = _
  rw [smallRow1453_block_002_checked, smallRow1453_small_right_right_checked]
  rfl

def smallRow1453_small : List Nat := smallRow1453_small_left ++ smallRow1453_small_right
theorem smallRow1453_small_checked : upperNeighbors PrimePairVertices.tree_small 1453 = smallRow1453_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1453 ++ upperNeighbors PrimePairVertices.tree_small_right 1453 = _
  rw [smallRow1453_small_left_checked, smallRow1453_small_right_checked]
  rfl

def smallRow1453 : List Nat := smallRow1453_small
theorem smallRow1453_checked : upperNeighbors PrimePairVertices.smallTree 1453 = smallRow1453 := smallRow1453_small_checked

def smallRow1459_block_000 : List Nat := [1567, 1657, 1861, 1873]
set_option maxRecDepth 2048 in
theorem smallRow1459_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1459 = smallRow1459_block_000 := by decide

def smallRow1459_block_001 : List Nat := [2089, 3511, 3673]
set_option maxRecDepth 2048 in
theorem smallRow1459_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1459 = smallRow1459_block_001 := by decide

def smallRow1459_block_002 : List Nat := [4813, 5011, 5233, 5437, 6079]
set_option maxRecDepth 2048 in
theorem smallRow1459_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1459 = smallRow1459_block_002 := by decide

def smallRow1459_block_003 : List Nat := [6271, 6991, 7027, 7717, 7963]
set_option maxRecDepth 2048 in
theorem smallRow1459_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1459 = smallRow1459_block_003 := by decide

def smallRow1459_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1459_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1459 = smallRow1459_block_004 := by decide

def smallRow1459_small_left : List Nat := smallRow1459_block_000 ++ smallRow1459_block_001
theorem smallRow1459_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1459 = smallRow1459_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1459 ++ upperNeighbors PrimePairVertices.tree_block_001 1459 = _
  rw [smallRow1459_block_000_checked, smallRow1459_block_001_checked]
  rfl

def smallRow1459_small_right_right : List Nat := smallRow1459_block_003 ++ smallRow1459_block_004
theorem smallRow1459_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1459 = smallRow1459_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1459 ++ upperNeighbors PrimePairVertices.tree_block_004 1459 = _
  rw [smallRow1459_block_003_checked, smallRow1459_block_004_checked]
  rfl

def smallRow1459_small_right : List Nat := smallRow1459_block_002 ++ smallRow1459_small_right_right
theorem smallRow1459_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1459 = smallRow1459_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1459 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1459 = _
  rw [smallRow1459_block_002_checked, smallRow1459_small_right_right_checked]
  rfl

def smallRow1459_small : List Nat := smallRow1459_small_left ++ smallRow1459_small_right
theorem smallRow1459_small_checked : upperNeighbors PrimePairVertices.tree_small 1459 = smallRow1459_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1459 ++ upperNeighbors PrimePairVertices.tree_small_right 1459 = _
  rw [smallRow1459_small_left_checked, smallRow1459_small_right_checked]
  rfl

def smallRow1459 : List Nat := smallRow1459_small
theorem smallRow1459_checked : upperNeighbors PrimePairVertices.smallTree 1459 = smallRow1459 := smallRow1459_small_checked

def smallRow1471_block_000 : List Nat := [1597]
set_option maxRecDepth 2048 in
theorem smallRow1471_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1471 = smallRow1471_block_000 := by decide

def smallRow1471_block_001 : List Nat := [2221, 2311, 2389, 2521, 3541, 3571, 3613, 3931]
set_option maxRecDepth 2048 in
theorem smallRow1471_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1471 = smallRow1471_block_001 := by decide

def smallRow1471_block_002 : List Nat := [4159, 4561, 4951, 5179, 5419, 6073, 6133]
set_option maxRecDepth 2048 in
theorem smallRow1471_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1471 = smallRow1471_block_002 := by decide

def smallRow1471_block_003 : List Nat := [6301, 7069, 7207, 7321, 7933]
set_option maxRecDepth 2048 in
theorem smallRow1471_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1471 = smallRow1471_block_003 := by decide

def smallRow1471_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1471_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1471 = smallRow1471_block_004 := by decide

def smallRow1471_small_left : List Nat := smallRow1471_block_000 ++ smallRow1471_block_001
theorem smallRow1471_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1471 = smallRow1471_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1471 ++ upperNeighbors PrimePairVertices.tree_block_001 1471 = _
  rw [smallRow1471_block_000_checked, smallRow1471_block_001_checked]
  rfl

def smallRow1471_small_right_right : List Nat := smallRow1471_block_003 ++ smallRow1471_block_004
theorem smallRow1471_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1471 = smallRow1471_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1471 ++ upperNeighbors PrimePairVertices.tree_block_004 1471 = _
  rw [smallRow1471_block_003_checked, smallRow1471_block_004_checked]
  rfl

def smallRow1471_small_right : List Nat := smallRow1471_block_002 ++ smallRow1471_small_right_right
theorem smallRow1471_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1471 = smallRow1471_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1471 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1471 = _
  rw [smallRow1471_block_002_checked, smallRow1471_small_right_right_checked]
  rfl

def smallRow1471_small : List Nat := smallRow1471_small_left ++ smallRow1471_small_right
theorem smallRow1471_small_checked : upperNeighbors PrimePairVertices.tree_small 1471 = smallRow1471_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1471 ++ upperNeighbors PrimePairVertices.tree_small_right 1471 = _
  rw [smallRow1471_small_left_checked, smallRow1471_small_right_checked]
  rfl

def smallRow1471 : List Nat := smallRow1471_small
theorem smallRow1471_checked : upperNeighbors PrimePairVertices.smallTree 1471 = smallRow1471 := smallRow1471_small_checked

def smallRow1481_block_000 : List Nat := [1733]
set_option maxRecDepth 2048 in
theorem smallRow1481_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1481 = smallRow1481_block_000 := by decide

def smallRow1481_block_001 : List Nat := [2663, 2879, 3089, 3167, 3803, 3881]
set_option maxRecDepth 2048 in
theorem smallRow1481_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1481 = smallRow1481_block_001 := by decide

def smallRow1481_block_002 : List Nat := [4127, 4211, 5189, 5639, 5981]
set_option maxRecDepth 2048 in
theorem smallRow1481_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1481 = smallRow1481_block_002 := by decide

def smallRow1481_block_003 : List Nat := [6569, 6869, 7529, 7703, 7823]
set_option maxRecDepth 2048 in
theorem smallRow1481_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1481 = smallRow1481_block_003 := by decide

def smallRow1481_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1481_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1481 = smallRow1481_block_004 := by decide

def smallRow1481_small_left : List Nat := smallRow1481_block_000 ++ smallRow1481_block_001
theorem smallRow1481_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1481 = smallRow1481_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1481 ++ upperNeighbors PrimePairVertices.tree_block_001 1481 = _
  rw [smallRow1481_block_000_checked, smallRow1481_block_001_checked]
  rfl

def smallRow1481_small_right_right : List Nat := smallRow1481_block_003 ++ smallRow1481_block_004
theorem smallRow1481_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1481 = smallRow1481_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1481 ++ upperNeighbors PrimePairVertices.tree_block_004 1481 = _
  rw [smallRow1481_block_003_checked, smallRow1481_block_004_checked]
  rfl

def smallRow1481_small_right : List Nat := smallRow1481_block_002 ++ smallRow1481_small_right_right
theorem smallRow1481_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1481 = smallRow1481_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1481 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1481 = _
  rw [smallRow1481_block_002_checked, smallRow1481_small_right_right_checked]
  rfl

def smallRow1481_small : List Nat := smallRow1481_small_left ++ smallRow1481_small_right
theorem smallRow1481_small_checked : upperNeighbors PrimePairVertices.tree_small 1481 = smallRow1481_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1481 ++ upperNeighbors PrimePairVertices.tree_small_right 1481 = _
  rw [smallRow1481_small_left_checked, smallRow1481_small_right_checked]
  rfl

def smallRow1481 : List Nat := smallRow1481_small
theorem smallRow1481_checked : upperNeighbors PrimePairVertices.smallTree 1481 = smallRow1481 := smallRow1481_small_checked

end ProjectEulerSolutions.PrimePairCache
