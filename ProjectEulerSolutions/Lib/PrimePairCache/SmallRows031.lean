import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1597_block_000 : List Nat := [1777, 1873]
set_option maxRecDepth 2048 in
theorem smallRow1597_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1597 = smallRow1597_block_000 := by decide

def smallRow1597_block_001 : List Nat := [2113, 2251, 2617, 2713, 2731, 3121, 3163, 3301, 3469, 3691, 3823]
set_option maxRecDepth 2048 in
theorem smallRow1597_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1597 = smallRow1597_block_001 := by decide

def smallRow1597_block_002 : List Nat := [4201, 4363, 4657, 5503]
set_option maxRecDepth 2048 in
theorem smallRow1597_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1597 = smallRow1597_block_002 := by decide

def smallRow1597_block_003 : List Nat := [7333, 7573, 7699]
set_option maxRecDepth 2048 in
theorem smallRow1597_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1597 = smallRow1597_block_003 := by decide

def smallRow1597_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow1597_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1597 = smallRow1597_block_004 := by decide

def smallRow1597_small_left : List Nat := smallRow1597_block_000 ++ smallRow1597_block_001
theorem smallRow1597_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1597 = smallRow1597_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1597 ++ upperNeighbors PrimePairVertices.tree_block_001 1597 = _
  rw [smallRow1597_block_000_checked, smallRow1597_block_001_checked]
  rfl

def smallRow1597_small_right_right : List Nat := smallRow1597_block_003 ++ smallRow1597_block_004
theorem smallRow1597_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1597 = smallRow1597_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1597 ++ upperNeighbors PrimePairVertices.tree_block_004 1597 = _
  rw [smallRow1597_block_003_checked, smallRow1597_block_004_checked]
  rfl

def smallRow1597_small_right : List Nat := smallRow1597_block_002 ++ smallRow1597_small_right_right
theorem smallRow1597_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1597 = smallRow1597_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1597 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1597 = _
  rw [smallRow1597_block_002_checked, smallRow1597_small_right_right_checked]
  rfl

def smallRow1597_small : List Nat := smallRow1597_small_left ++ smallRow1597_small_right
theorem smallRow1597_small_checked : upperNeighbors PrimePairVertices.tree_small 1597 = smallRow1597_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1597 ++ upperNeighbors PrimePairVertices.tree_small_right 1597 = _
  rw [smallRow1597_small_left_checked, smallRow1597_small_right_checked]
  rfl

def smallRow1597 : List Nat := smallRow1597_small
theorem smallRow1597_checked : upperNeighbors PrimePairVertices.smallTree 1597 = smallRow1597 := smallRow1597_small_checked

def smallRow1601_block_000 : List Nat := [1811, 1823, 1847]
set_option maxRecDepth 2048 in
theorem smallRow1601_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1601 = smallRow1601_block_000 := by decide

def smallRow1601_block_001 : List Nat := [2063, 2207, 2879, 3533]
set_option maxRecDepth 2048 in
theorem smallRow1601_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1601 = smallRow1601_block_001 := by decide

def smallRow1601_block_002 : List Nat := [4637, 6029, 6113]
set_option maxRecDepth 2048 in
theorem smallRow1601_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1601 = smallRow1601_block_002 := by decide

def smallRow1601_block_003 : List Nat := [6359, 6701, 7229, 7307]
set_option maxRecDepth 2048 in
theorem smallRow1601_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1601 = smallRow1601_block_003 := by decide

def smallRow1601_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1601_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1601 = smallRow1601_block_004 := by decide

def smallRow1601_small_left : List Nat := smallRow1601_block_000 ++ smallRow1601_block_001
theorem smallRow1601_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1601 = smallRow1601_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1601 ++ upperNeighbors PrimePairVertices.tree_block_001 1601 = _
  rw [smallRow1601_block_000_checked, smallRow1601_block_001_checked]
  rfl

def smallRow1601_small_right_right : List Nat := smallRow1601_block_003 ++ smallRow1601_block_004
theorem smallRow1601_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1601 = smallRow1601_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1601 ++ upperNeighbors PrimePairVertices.tree_block_004 1601 = _
  rw [smallRow1601_block_003_checked, smallRow1601_block_004_checked]
  rfl

def smallRow1601_small_right : List Nat := smallRow1601_block_002 ++ smallRow1601_small_right_right
theorem smallRow1601_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1601 = smallRow1601_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1601 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1601 = _
  rw [smallRow1601_block_002_checked, smallRow1601_small_right_right_checked]
  rfl

def smallRow1601_small : List Nat := smallRow1601_small_left ++ smallRow1601_small_right
theorem smallRow1601_small_checked : upperNeighbors PrimePairVertices.tree_small 1601 = smallRow1601_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1601 ++ upperNeighbors PrimePairVertices.tree_small_right 1601 = _
  rw [smallRow1601_small_left_checked, smallRow1601_small_right_checked]
  rfl

def smallRow1601 : List Nat := smallRow1601_small
theorem smallRow1601_checked : upperNeighbors PrimePairVertices.smallTree 1601 = smallRow1601 := smallRow1601_small_checked

def smallRow1607_block_000 : List Nat := [1901]
set_option maxRecDepth 2048 in
theorem smallRow1607_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1607 = smallRow1607_block_000 := by decide

def smallRow1607_block_001 : List Nat := [2549, 3203, 3767]
set_option maxRecDepth 2048 in
theorem smallRow1607_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1607 = smallRow1607_block_001 := by decide

def smallRow1607_block_002 : List Nat := [4271, 4481, 4637, 5009, 5099, 5351, 5471, 5849, 5903]
set_option maxRecDepth 2048 in
theorem smallRow1607_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1607 = smallRow1607_block_002 := by decide

def smallRow1607_block_003 : List Nat := [7109, 7541, 7673, 7877, 8171]
set_option maxRecDepth 2048 in
theorem smallRow1607_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1607 = smallRow1607_block_003 := by decide

def smallRow1607_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1607_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1607 = smallRow1607_block_004 := by decide

def smallRow1607_small_left : List Nat := smallRow1607_block_000 ++ smallRow1607_block_001
theorem smallRow1607_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1607 = smallRow1607_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1607 ++ upperNeighbors PrimePairVertices.tree_block_001 1607 = _
  rw [smallRow1607_block_000_checked, smallRow1607_block_001_checked]
  rfl

def smallRow1607_small_right_right : List Nat := smallRow1607_block_003 ++ smallRow1607_block_004
theorem smallRow1607_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1607 = smallRow1607_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1607 ++ upperNeighbors PrimePairVertices.tree_block_004 1607 = _
  rw [smallRow1607_block_003_checked, smallRow1607_block_004_checked]
  rfl

def smallRow1607_small_right : List Nat := smallRow1607_block_002 ++ smallRow1607_small_right_right
theorem smallRow1607_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1607 = smallRow1607_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1607 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1607 = _
  rw [smallRow1607_block_002_checked, smallRow1607_small_right_right_checked]
  rfl

def smallRow1607_small : List Nat := smallRow1607_small_left ++ smallRow1607_small_right
theorem smallRow1607_small_checked : upperNeighbors PrimePairVertices.tree_small 1607 = smallRow1607_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1607 ++ upperNeighbors PrimePairVertices.tree_small_right 1607 = _
  rw [smallRow1607_small_left_checked, smallRow1607_small_right_checked]
  rfl

def smallRow1607 : List Nat := smallRow1607_small
theorem smallRow1607_checked : upperNeighbors PrimePairVertices.smallTree 1607 = smallRow1607 := smallRow1607_small_checked

def smallRow1609_block_000 : List Nat := [1783, 2017]
set_option maxRecDepth 2048 in
theorem smallRow1609_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1609 = smallRow1609_block_000 := by decide

def smallRow1609_block_001 : List Nat := [2161, 2467, 2803, 2833, 3613, 3673, 3697, 3769, 3793, 3967]
set_option maxRecDepth 2048 in
theorem smallRow1609_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1609 = smallRow1609_block_001 := by decide

def smallRow1609_block_002 : List Nat := [4129, 4201, 4621, 4723, 5011, 5179, 5647]
set_option maxRecDepth 2048 in
theorem smallRow1609_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1609 = smallRow1609_block_002 := by decide

def smallRow1609_block_003 : List Nat := [6991, 7243, 7603]
set_option maxRecDepth 2048 in
theorem smallRow1609_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1609 = smallRow1609_block_003 := by decide

def smallRow1609_block_004 : List Nat := [8287]
set_option maxRecDepth 2048 in
theorem smallRow1609_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1609 = smallRow1609_block_004 := by decide

def smallRow1609_small_left : List Nat := smallRow1609_block_000 ++ smallRow1609_block_001
theorem smallRow1609_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1609 = smallRow1609_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1609 ++ upperNeighbors PrimePairVertices.tree_block_001 1609 = _
  rw [smallRow1609_block_000_checked, smallRow1609_block_001_checked]
  rfl

def smallRow1609_small_right_right : List Nat := smallRow1609_block_003 ++ smallRow1609_block_004
theorem smallRow1609_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1609 = smallRow1609_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1609 ++ upperNeighbors PrimePairVertices.tree_block_004 1609 = _
  rw [smallRow1609_block_003_checked, smallRow1609_block_004_checked]
  rfl

def smallRow1609_small_right : List Nat := smallRow1609_block_002 ++ smallRow1609_small_right_right
theorem smallRow1609_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1609 = smallRow1609_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1609 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1609 = _
  rw [smallRow1609_block_002_checked, smallRow1609_small_right_right_checked]
  rfl

def smallRow1609_small : List Nat := smallRow1609_small_left ++ smallRow1609_small_right
theorem smallRow1609_small_checked : upperNeighbors PrimePairVertices.tree_small 1609 = smallRow1609_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1609 ++ upperNeighbors PrimePairVertices.tree_small_right 1609 = _
  rw [smallRow1609_small_left_checked, smallRow1609_small_right_checked]
  rfl

def smallRow1609 : List Nat := smallRow1609_small
theorem smallRow1609_checked : upperNeighbors PrimePairVertices.smallTree 1609 = smallRow1609 := smallRow1609_small_checked

def smallRow1613_block_000 : List Nat := [1637, 1949, 1979]
set_option maxRecDepth 2048 in
theorem smallRow1613_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1613 = smallRow1613_block_000 := by decide

def smallRow1613_block_001 : List Nat := [2477, 2693, 3023, 3779]
set_option maxRecDepth 2048 in
theorem smallRow1613_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1613 = smallRow1613_block_001 := by decide

def smallRow1613_block_002 : List Nat := [4133, 4673, 4967, 5297, 5813]
set_option maxRecDepth 2048 in
theorem smallRow1613_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1613 = smallRow1613_block_002 := by decide

def smallRow1613_block_003 : List Nat := [6311, 6899, 6971, 7229, 7481, 7487, 7853]
set_option maxRecDepth 2048 in
theorem smallRow1613_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1613 = smallRow1613_block_003 := by decide

def smallRow1613_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1613_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1613 = smallRow1613_block_004 := by decide

def smallRow1613_small_left : List Nat := smallRow1613_block_000 ++ smallRow1613_block_001
theorem smallRow1613_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1613 = smallRow1613_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1613 ++ upperNeighbors PrimePairVertices.tree_block_001 1613 = _
  rw [smallRow1613_block_000_checked, smallRow1613_block_001_checked]
  rfl

def smallRow1613_small_right_right : List Nat := smallRow1613_block_003 ++ smallRow1613_block_004
theorem smallRow1613_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1613 = smallRow1613_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1613 ++ upperNeighbors PrimePairVertices.tree_block_004 1613 = _
  rw [smallRow1613_block_003_checked, smallRow1613_block_004_checked]
  rfl

def smallRow1613_small_right : List Nat := smallRow1613_block_002 ++ smallRow1613_small_right_right
theorem smallRow1613_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1613 = smallRow1613_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1613 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1613 = _
  rw [smallRow1613_block_002_checked, smallRow1613_small_right_right_checked]
  rfl

def smallRow1613_small : List Nat := smallRow1613_small_left ++ smallRow1613_small_right
theorem smallRow1613_small_checked : upperNeighbors PrimePairVertices.tree_small 1613 = smallRow1613_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1613 ++ upperNeighbors PrimePairVertices.tree_small_right 1613 = _
  rw [smallRow1613_small_left_checked, smallRow1613_small_right_checked]
  rfl

def smallRow1613 : List Nat := smallRow1613_small
theorem smallRow1613_checked : upperNeighbors PrimePairVertices.smallTree 1613 = smallRow1613 := smallRow1613_small_checked

def smallRow1619_block_000 : List Nat := [2003]
set_option maxRecDepth 2048 in
theorem smallRow1619_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1619 = smallRow1619_block_000 := by decide

def smallRow1619_block_001 : List Nat := [2213, 2339, 2531, 2777, 3119, 3257, 3407, 3677]
set_option maxRecDepth 2048 in
theorem smallRow1619_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1619 = smallRow1619_block_001 := by decide

def smallRow1619_block_002 : List Nat := [4127, 4547, 4817, 4943, 5279, 5573, 5801, 5903, 5987, 6143]
set_option maxRecDepth 2048 in
theorem smallRow1619_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1619 = smallRow1619_block_002 := by decide

def smallRow1619_block_003 : List Nat := [6197, 6701, 7331, 7949]
set_option maxRecDepth 2048 in
theorem smallRow1619_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1619 = smallRow1619_block_003 := by decide

def smallRow1619_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1619_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1619 = smallRow1619_block_004 := by decide

def smallRow1619_small_left : List Nat := smallRow1619_block_000 ++ smallRow1619_block_001
theorem smallRow1619_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1619 = smallRow1619_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1619 ++ upperNeighbors PrimePairVertices.tree_block_001 1619 = _
  rw [smallRow1619_block_000_checked, smallRow1619_block_001_checked]
  rfl

def smallRow1619_small_right_right : List Nat := smallRow1619_block_003 ++ smallRow1619_block_004
theorem smallRow1619_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1619 = smallRow1619_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1619 ++ upperNeighbors PrimePairVertices.tree_block_004 1619 = _
  rw [smallRow1619_block_003_checked, smallRow1619_block_004_checked]
  rfl

def smallRow1619_small_right : List Nat := smallRow1619_block_002 ++ smallRow1619_small_right_right
theorem smallRow1619_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1619 = smallRow1619_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1619 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1619 = _
  rw [smallRow1619_block_002_checked, smallRow1619_small_right_right_checked]
  rfl

def smallRow1619_small : List Nat := smallRow1619_small_left ++ smallRow1619_small_right
theorem smallRow1619_small_checked : upperNeighbors PrimePairVertices.tree_small 1619 = smallRow1619_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1619 ++ upperNeighbors PrimePairVertices.tree_small_right 1619 = _
  rw [smallRow1619_small_left_checked, smallRow1619_small_right_checked]
  rfl

def smallRow1619 : List Nat := smallRow1619_small
theorem smallRow1619_checked : upperNeighbors PrimePairVertices.smallTree 1619 = smallRow1619 := smallRow1619_small_checked

def smallRow1621_block_000 : List Nat := [1801]
set_option maxRecDepth 2048 in
theorem smallRow1621_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1621 = smallRow1621_block_000 := by decide

def smallRow1621_block_001 : List Nat := [2083, 2437, 2767, 2803, 2851, 3187, 3313, 3823, 3943, 4021]
set_option maxRecDepth 2048 in
theorem smallRow1621_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1621 = smallRow1621_block_001 := by decide

def smallRow1621_block_002 : List Nat := [4153, 4441, 5443, 5503, 5527, 6037, 6073]
set_option maxRecDepth 2048 in
theorem smallRow1621_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1621 = smallRow1621_block_002 := by decide

def smallRow1621_block_003 : List Nat := [6247, 7477, 7603, 7639, 7687, 7933]
set_option maxRecDepth 2048 in
theorem smallRow1621_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1621 = smallRow1621_block_003 := by decide

def smallRow1621_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1621_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1621 = smallRow1621_block_004 := by decide

def smallRow1621_small_left : List Nat := smallRow1621_block_000 ++ smallRow1621_block_001
theorem smallRow1621_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1621 = smallRow1621_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1621 ++ upperNeighbors PrimePairVertices.tree_block_001 1621 = _
  rw [smallRow1621_block_000_checked, smallRow1621_block_001_checked]
  rfl

def smallRow1621_small_right_right : List Nat := smallRow1621_block_003 ++ smallRow1621_block_004
theorem smallRow1621_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1621 = smallRow1621_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1621 ++ upperNeighbors PrimePairVertices.tree_block_004 1621 = _
  rw [smallRow1621_block_003_checked, smallRow1621_block_004_checked]
  rfl

def smallRow1621_small_right : List Nat := smallRow1621_block_002 ++ smallRow1621_small_right_right
theorem smallRow1621_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1621 = smallRow1621_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1621 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1621 = _
  rw [smallRow1621_block_002_checked, smallRow1621_small_right_right_checked]
  rfl

def smallRow1621_small : List Nat := smallRow1621_small_left ++ smallRow1621_small_right
theorem smallRow1621_small_checked : upperNeighbors PrimePairVertices.tree_small 1621 = smallRow1621_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1621 ++ upperNeighbors PrimePairVertices.tree_small_right 1621 = _
  rw [smallRow1621_small_left_checked, smallRow1621_small_right_checked]
  rfl

def smallRow1621 : List Nat := smallRow1621_small
theorem smallRow1621_checked : upperNeighbors PrimePairVertices.smallTree 1621 = smallRow1621 := smallRow1621_small_checked

def smallRow1627_block_000 : List Nat := [1753]
set_option maxRecDepth 2048 in
theorem smallRow1627_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1627 = smallRow1627_block_000 := by decide

def smallRow1627_block_001 : List Nat := [2917, 3217, 3463, 3637, 3769]
set_option maxRecDepth 2048 in
theorem smallRow1627_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1627 = smallRow1627_block_001 := by decide

def smallRow1627_block_002 : List Nat := [4591, 5011, 5179, 5821]
set_option maxRecDepth 2048 in
theorem smallRow1627_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1627 = smallRow1627_block_002 := by decide

def smallRow1627_block_003 : List Nat := [6277, 6961, 7159, 7507, 7741]
set_option maxRecDepth 2048 in
theorem smallRow1627_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1627 = smallRow1627_block_003 := by decide

def smallRow1627_block_004 : List Nat := [8287]
set_option maxRecDepth 2048 in
theorem smallRow1627_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1627 = smallRow1627_block_004 := by decide

def smallRow1627_small_left : List Nat := smallRow1627_block_000 ++ smallRow1627_block_001
theorem smallRow1627_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1627 = smallRow1627_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1627 ++ upperNeighbors PrimePairVertices.tree_block_001 1627 = _
  rw [smallRow1627_block_000_checked, smallRow1627_block_001_checked]
  rfl

def smallRow1627_small_right_right : List Nat := smallRow1627_block_003 ++ smallRow1627_block_004
theorem smallRow1627_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1627 = smallRow1627_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1627 ++ upperNeighbors PrimePairVertices.tree_block_004 1627 = _
  rw [smallRow1627_block_003_checked, smallRow1627_block_004_checked]
  rfl

def smallRow1627_small_right : List Nat := smallRow1627_block_002 ++ smallRow1627_small_right_right
theorem smallRow1627_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1627 = smallRow1627_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1627 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1627 = _
  rw [smallRow1627_block_002_checked, smallRow1627_small_right_right_checked]
  rfl

def smallRow1627_small : List Nat := smallRow1627_small_left ++ smallRow1627_small_right
theorem smallRow1627_small_checked : upperNeighbors PrimePairVertices.tree_small 1627 = smallRow1627_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1627 ++ upperNeighbors PrimePairVertices.tree_small_right 1627 = _
  rw [smallRow1627_small_left_checked, smallRow1627_small_right_checked]
  rfl

def smallRow1627 : List Nat := smallRow1627_small
theorem smallRow1627_checked : upperNeighbors PrimePairVertices.smallTree 1627 = smallRow1627 := smallRow1627_small_checked

end ProjectEulerSolutions.PrimePairCache
