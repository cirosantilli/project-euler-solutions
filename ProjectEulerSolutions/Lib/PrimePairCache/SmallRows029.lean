import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1483_block_000 : List Nat := [1867]
set_option maxRecDepth 2048 in
theorem smallRow1483_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1483 = smallRow1483_block_000 := by decide

def smallRow1483_block_001 : List Nat := [2269, 2281, 2341, 2593, 2791, 3457]
set_option maxRecDepth 2048 in
theorem smallRow1483_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1483 = smallRow1483_block_001 := by decide

def smallRow1483_block_002 : List Nat := [4357, 5683, 5869]
set_option maxRecDepth 2048 in
theorem smallRow1483_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1483 = smallRow1483_block_002 := by decide

def smallRow1483_block_003 : List Nat := [6163, 6961, 7027, 7159, 7393, 7591, 7789, 7873, 8191]
set_option maxRecDepth 2048 in
theorem smallRow1483_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1483 = smallRow1483_block_003 := by decide

def smallRow1483_block_004 : List Nat := [8209]
set_option maxRecDepth 2048 in
theorem smallRow1483_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1483 = smallRow1483_block_004 := by decide

def smallRow1483_small_left : List Nat := smallRow1483_block_000 ++ smallRow1483_block_001
theorem smallRow1483_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1483 = smallRow1483_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1483 ++ upperNeighbors PrimePairVertices.tree_block_001 1483 = _
  rw [smallRow1483_block_000_checked, smallRow1483_block_001_checked]
  rfl

def smallRow1483_small_right_right : List Nat := smallRow1483_block_003 ++ smallRow1483_block_004
theorem smallRow1483_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1483 = smallRow1483_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1483 ++ upperNeighbors PrimePairVertices.tree_block_004 1483 = _
  rw [smallRow1483_block_003_checked, smallRow1483_block_004_checked]
  rfl

def smallRow1483_small_right : List Nat := smallRow1483_block_002 ++ smallRow1483_small_right_right
theorem smallRow1483_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1483 = smallRow1483_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1483 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1483 = _
  rw [smallRow1483_block_002_checked, smallRow1483_small_right_right_checked]
  rfl

def smallRow1483_small : List Nat := smallRow1483_small_left ++ smallRow1483_small_right
theorem smallRow1483_small_checked : upperNeighbors PrimePairVertices.tree_small 1483 = smallRow1483_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1483 ++ upperNeighbors PrimePairVertices.tree_small_right 1483 = _
  rw [smallRow1483_small_left_checked, smallRow1483_small_right_checked]
  rfl

def smallRow1483 : List Nat := smallRow1483_small
theorem smallRow1483_checked : upperNeighbors PrimePairVertices.smallTree 1483 = smallRow1483 := smallRow1483_small_checked

def smallRow1487_block_000 : List Nat := [1553, 1601, 1907, 1979]
set_option maxRecDepth 2048 in
theorem smallRow1487_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1487 = smallRow1487_block_000 := by decide

def smallRow1487_block_001 : List Nat := [2267, 2729, 3659, 3671]
set_option maxRecDepth 2048 in
theorem smallRow1487_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1487 = smallRow1487_block_001 := by decide

def smallRow1487_block_002 : List Nat := [5717, 5801, 5843]
set_option maxRecDepth 2048 in
theorem smallRow1487_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1487 = smallRow1487_block_002 := by decide

def smallRow1487_block_003 : List Nat := [6473, 6761, 7013, 7727]
set_option maxRecDepth 2048 in
theorem smallRow1487_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1487 = smallRow1487_block_003 := by decide

def smallRow1487_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1487_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1487 = smallRow1487_block_004 := by decide

def smallRow1487_small_left : List Nat := smallRow1487_block_000 ++ smallRow1487_block_001
theorem smallRow1487_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1487 = smallRow1487_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1487 ++ upperNeighbors PrimePairVertices.tree_block_001 1487 = _
  rw [smallRow1487_block_000_checked, smallRow1487_block_001_checked]
  rfl

def smallRow1487_small_right_right : List Nat := smallRow1487_block_003 ++ smallRow1487_block_004
theorem smallRow1487_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1487 = smallRow1487_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1487 ++ upperNeighbors PrimePairVertices.tree_block_004 1487 = _
  rw [smallRow1487_block_003_checked, smallRow1487_block_004_checked]
  rfl

def smallRow1487_small_right : List Nat := smallRow1487_block_002 ++ smallRow1487_small_right_right
theorem smallRow1487_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1487 = smallRow1487_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1487 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1487 = _
  rw [smallRow1487_block_002_checked, smallRow1487_small_right_right_checked]
  rfl

def smallRow1487_small : List Nat := smallRow1487_small_left ++ smallRow1487_small_right
theorem smallRow1487_small_checked : upperNeighbors PrimePairVertices.tree_small 1487 = smallRow1487_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1487 ++ upperNeighbors PrimePairVertices.tree_small_right 1487 = _
  rw [smallRow1487_small_left_checked, smallRow1487_small_right_checked]
  rfl

def smallRow1487 : List Nat := smallRow1487_small
theorem smallRow1487_checked : upperNeighbors PrimePairVertices.smallTree 1487 = smallRow1487 := smallRow1487_small_checked

def smallRow1489_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1489_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1489 = smallRow1489_block_000 := by decide

def smallRow1489_block_001 : List Nat := [3691]
set_option maxRecDepth 2048 in
theorem smallRow1489_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1489 = smallRow1489_block_001 := by decide

def smallRow1489_block_002 : List Nat := [4597, 4801, 5119, 5557, 5857]
set_option maxRecDepth 2048 in
theorem smallRow1489_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1489 = smallRow1489_block_002 := by decide

def smallRow1489_block_003 : List Nat := [6547, 6883]
set_option maxRecDepth 2048 in
theorem smallRow1489_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1489 = smallRow1489_block_003 := by decide

def smallRow1489_block_004 : List Nat := [8263, 8329]
set_option maxRecDepth 2048 in
theorem smallRow1489_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1489 = smallRow1489_block_004 := by decide

def smallRow1489_small_left : List Nat := smallRow1489_block_000 ++ smallRow1489_block_001
theorem smallRow1489_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1489 = smallRow1489_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1489 ++ upperNeighbors PrimePairVertices.tree_block_001 1489 = _
  rw [smallRow1489_block_000_checked, smallRow1489_block_001_checked]
  rfl

def smallRow1489_small_right_right : List Nat := smallRow1489_block_003 ++ smallRow1489_block_004
theorem smallRow1489_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1489 = smallRow1489_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1489 ++ upperNeighbors PrimePairVertices.tree_block_004 1489 = _
  rw [smallRow1489_block_003_checked, smallRow1489_block_004_checked]
  rfl

def smallRow1489_small_right : List Nat := smallRow1489_block_002 ++ smallRow1489_small_right_right
theorem smallRow1489_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1489 = smallRow1489_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1489 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1489 = _
  rw [smallRow1489_block_002_checked, smallRow1489_small_right_right_checked]
  rfl

def smallRow1489_small : List Nat := smallRow1489_small_left ++ smallRow1489_small_right
theorem smallRow1489_small_checked : upperNeighbors PrimePairVertices.tree_small 1489 = smallRow1489_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1489 ++ upperNeighbors PrimePairVertices.tree_small_right 1489 = _
  rw [smallRow1489_small_left_checked, smallRow1489_small_right_checked]
  rfl

def smallRow1489 : List Nat := smallRow1489_small
theorem smallRow1489_checked : upperNeighbors PrimePairVertices.smallTree 1489 = smallRow1489 := smallRow1489_small_checked

def smallRow1493_block_000 : List Nat := [1523]
set_option maxRecDepth 2048 in
theorem smallRow1493_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1493 = smallRow1493_block_000 := by decide

def smallRow1493_block_001 : List Nat := [2339, 2441, 2609, 3329, 3623]
set_option maxRecDepth 2048 in
theorem smallRow1493_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1493 = smallRow1493_block_001 := by decide

def smallRow1493_block_002 : List Nat := [4649, 4943, 5021, 5189, 5279, 5651, 5867]
set_option maxRecDepth 2048 in
theorem smallRow1493_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1493 = smallRow1493_block_002 := by decide

def smallRow1493_block_003 : List Nat := [6599, 7229, 7457, 7541, 7607]
set_option maxRecDepth 2048 in
theorem smallRow1493_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1493 = smallRow1493_block_003 := by decide

def smallRow1493_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1493_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1493 = smallRow1493_block_004 := by decide

def smallRow1493_small_left : List Nat := smallRow1493_block_000 ++ smallRow1493_block_001
theorem smallRow1493_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1493 = smallRow1493_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1493 ++ upperNeighbors PrimePairVertices.tree_block_001 1493 = _
  rw [smallRow1493_block_000_checked, smallRow1493_block_001_checked]
  rfl

def smallRow1493_small_right_right : List Nat := smallRow1493_block_003 ++ smallRow1493_block_004
theorem smallRow1493_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1493 = smallRow1493_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1493 ++ upperNeighbors PrimePairVertices.tree_block_004 1493 = _
  rw [smallRow1493_block_003_checked, smallRow1493_block_004_checked]
  rfl

def smallRow1493_small_right : List Nat := smallRow1493_block_002 ++ smallRow1493_small_right_right
theorem smallRow1493_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1493 = smallRow1493_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1493 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1493 = _
  rw [smallRow1493_block_002_checked, smallRow1493_small_right_right_checked]
  rfl

def smallRow1493_small : List Nat := smallRow1493_small_left ++ smallRow1493_small_right
theorem smallRow1493_small_checked : upperNeighbors PrimePairVertices.tree_small 1493 = smallRow1493_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1493 ++ upperNeighbors PrimePairVertices.tree_small_right 1493 = _
  rw [smallRow1493_small_left_checked, smallRow1493_small_right_checked]
  rfl

def smallRow1493 : List Nat := smallRow1493_small
theorem smallRow1493_checked : upperNeighbors PrimePairVertices.smallTree 1493 = smallRow1493 := smallRow1493_small_checked

def smallRow1499_block_000 : List Nat := [1619]
set_option maxRecDepth 2048 in
theorem smallRow1499_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1499 = smallRow1499_block_000 := by decide

def smallRow1499_block_001 : List Nat := [2129, 2309, 2543, 2633, 2801, 2897, 3359, 3947, 4013]
set_option maxRecDepth 2048 in
theorem smallRow1499_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1499 = smallRow1499_block_001 := by decide

def smallRow1499_block_002 : List Nat := [4253]
set_option maxRecDepth 2048 in
theorem smallRow1499_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1499 = smallRow1499_block_002 := by decide

def smallRow1499_block_003 : List Nat := [6323, 6521, 6563, 7121, 7331, 7589, 8171]
set_option maxRecDepth 2048 in
theorem smallRow1499_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1499 = smallRow1499_block_003 := by decide

def smallRow1499_block_004 : List Nat := [8273]
set_option maxRecDepth 2048 in
theorem smallRow1499_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1499 = smallRow1499_block_004 := by decide

def smallRow1499_small_left : List Nat := smallRow1499_block_000 ++ smallRow1499_block_001
theorem smallRow1499_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1499 = smallRow1499_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1499 ++ upperNeighbors PrimePairVertices.tree_block_001 1499 = _
  rw [smallRow1499_block_000_checked, smallRow1499_block_001_checked]
  rfl

def smallRow1499_small_right_right : List Nat := smallRow1499_block_003 ++ smallRow1499_block_004
theorem smallRow1499_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1499 = smallRow1499_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1499 ++ upperNeighbors PrimePairVertices.tree_block_004 1499 = _
  rw [smallRow1499_block_003_checked, smallRow1499_block_004_checked]
  rfl

def smallRow1499_small_right : List Nat := smallRow1499_block_002 ++ smallRow1499_small_right_right
theorem smallRow1499_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1499 = smallRow1499_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1499 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1499 = _
  rw [smallRow1499_block_002_checked, smallRow1499_small_right_right_checked]
  rfl

def smallRow1499_small : List Nat := smallRow1499_small_left ++ smallRow1499_small_right
theorem smallRow1499_small_checked : upperNeighbors PrimePairVertices.tree_small 1499 = smallRow1499_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1499 ++ upperNeighbors PrimePairVertices.tree_small_right 1499 = _
  rw [smallRow1499_small_left_checked, smallRow1499_small_right_checked]
  rfl

def smallRow1499 : List Nat := smallRow1499_small
theorem smallRow1499_checked : upperNeighbors PrimePairVertices.smallTree 1499 = smallRow1499 := smallRow1499_small_checked

def smallRow1511_block_000 : List Nat := [1571]
set_option maxRecDepth 2048 in
theorem smallRow1511_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1511 = smallRow1511_block_000 := by decide

def smallRow1511_block_001 : List Nat := [2129, 2267, 3011, 3389]
set_option maxRecDepth 2048 in
theorem smallRow1511_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1511 = smallRow1511_block_001 := by decide

def smallRow1511_block_002 : List Nat := [4373, 4889, 5039, 5099, 5843, 5981]
set_option maxRecDepth 2048 in
theorem smallRow1511_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1511 = smallRow1511_block_002 := by decide

def smallRow1511_block_003 : List Nat := [7043, 7853]
set_option maxRecDepth 2048 in
theorem smallRow1511_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1511 = smallRow1511_block_003 := by decide

def smallRow1511_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1511_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1511 = smallRow1511_block_004 := by decide

def smallRow1511_small_left : List Nat := smallRow1511_block_000 ++ smallRow1511_block_001
theorem smallRow1511_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1511 = smallRow1511_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1511 ++ upperNeighbors PrimePairVertices.tree_block_001 1511 = _
  rw [smallRow1511_block_000_checked, smallRow1511_block_001_checked]
  rfl

def smallRow1511_small_right_right : List Nat := smallRow1511_block_003 ++ smallRow1511_block_004
theorem smallRow1511_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1511 = smallRow1511_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1511 ++ upperNeighbors PrimePairVertices.tree_block_004 1511 = _
  rw [smallRow1511_block_003_checked, smallRow1511_block_004_checked]
  rfl

def smallRow1511_small_right : List Nat := smallRow1511_block_002 ++ smallRow1511_small_right_right
theorem smallRow1511_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1511 = smallRow1511_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1511 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1511 = _
  rw [smallRow1511_block_002_checked, smallRow1511_small_right_right_checked]
  rfl

def smallRow1511_small : List Nat := smallRow1511_small_left ++ smallRow1511_small_right
theorem smallRow1511_small_checked : upperNeighbors PrimePairVertices.tree_small 1511 = smallRow1511_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1511 ++ upperNeighbors PrimePairVertices.tree_small_right 1511 = _
  rw [smallRow1511_small_left_checked, smallRow1511_small_right_checked]
  rfl

def smallRow1511 : List Nat := smallRow1511_small
theorem smallRow1511_checked : upperNeighbors PrimePairVertices.smallTree 1511 = smallRow1511 := smallRow1511_small_checked

def smallRow1523_block_000 : List Nat := [1913]
set_option maxRecDepth 2048 in
theorem smallRow1523_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1523 = smallRow1523_block_000 := by decide

def smallRow1523_block_001 : List Nat := [2111, 2447, 2579, 2909, 2957, 2999, 3203, 3299, 3461, 3761, 3833, 3929, 4013]
set_option maxRecDepth 2048 in
theorem smallRow1523_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1523 = smallRow1523_block_001 := by decide

def smallRow1523_block_002 : List Nat := [4139, 4253, 4517, 4679, 5279, 5303, 5309, 5483, 5573, 5813]
set_option maxRecDepth 2048 in
theorem smallRow1523_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1523 = smallRow1523_block_002 := by decide

def smallRow1523_block_003 : List Nat := [6311, 6827, 7043, 7583, 7907]
set_option maxRecDepth 2048 in
theorem smallRow1523_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1523 = smallRow1523_block_003 := by decide

def smallRow1523_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1523_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1523 = smallRow1523_block_004 := by decide

def smallRow1523_small_left : List Nat := smallRow1523_block_000 ++ smallRow1523_block_001
theorem smallRow1523_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1523 = smallRow1523_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1523 ++ upperNeighbors PrimePairVertices.tree_block_001 1523 = _
  rw [smallRow1523_block_000_checked, smallRow1523_block_001_checked]
  rfl

def smallRow1523_small_right_right : List Nat := smallRow1523_block_003 ++ smallRow1523_block_004
theorem smallRow1523_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1523 = smallRow1523_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1523 ++ upperNeighbors PrimePairVertices.tree_block_004 1523 = _
  rw [smallRow1523_block_003_checked, smallRow1523_block_004_checked]
  rfl

def smallRow1523_small_right : List Nat := smallRow1523_block_002 ++ smallRow1523_small_right_right
theorem smallRow1523_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1523 = smallRow1523_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1523 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1523 = _
  rw [smallRow1523_block_002_checked, smallRow1523_small_right_right_checked]
  rfl

def smallRow1523_small : List Nat := smallRow1523_small_left ++ smallRow1523_small_right
theorem smallRow1523_small_checked : upperNeighbors PrimePairVertices.tree_small 1523 = smallRow1523_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1523 ++ upperNeighbors PrimePairVertices.tree_small_right 1523 = _
  rw [smallRow1523_small_left_checked, smallRow1523_small_right_checked]
  rfl

def smallRow1523 : List Nat := smallRow1523_small
theorem smallRow1523_checked : upperNeighbors PrimePairVertices.smallTree 1523 = smallRow1523 := smallRow1523_small_checked

def smallRow1531_block_000 : List Nat := [1987]
set_option maxRecDepth 2048 in
theorem smallRow1531_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1531 = smallRow1531_block_000 := by decide

def smallRow1531_block_001 : List Nat := [2467, 2677, 3169]
set_option maxRecDepth 2048 in
theorem smallRow1531_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1531 = smallRow1531_block_001 := by decide

def smallRow1531_block_002 : List Nat := [4153, 4177, 5059, 5119, 5281, 5647]
set_option maxRecDepth 2048 in
theorem smallRow1531_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1531 = smallRow1531_block_002 := by decide

def smallRow1531_block_003 : List Nat := [6211, 6361, 6421, 7213, 7537, 7681, 7717, 7789, 7927]
set_option maxRecDepth 2048 in
theorem smallRow1531_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1531 = smallRow1531_block_003 := by decide

def smallRow1531_block_004 : List Nat := [8311]
set_option maxRecDepth 2048 in
theorem smallRow1531_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1531 = smallRow1531_block_004 := by decide

def smallRow1531_small_left : List Nat := smallRow1531_block_000 ++ smallRow1531_block_001
theorem smallRow1531_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1531 = smallRow1531_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1531 ++ upperNeighbors PrimePairVertices.tree_block_001 1531 = _
  rw [smallRow1531_block_000_checked, smallRow1531_block_001_checked]
  rfl

def smallRow1531_small_right_right : List Nat := smallRow1531_block_003 ++ smallRow1531_block_004
theorem smallRow1531_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1531 = smallRow1531_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1531 ++ upperNeighbors PrimePairVertices.tree_block_004 1531 = _
  rw [smallRow1531_block_003_checked, smallRow1531_block_004_checked]
  rfl

def smallRow1531_small_right : List Nat := smallRow1531_block_002 ++ smallRow1531_small_right_right
theorem smallRow1531_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1531 = smallRow1531_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1531 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1531 = _
  rw [smallRow1531_block_002_checked, smallRow1531_small_right_right_checked]
  rfl

def smallRow1531_small : List Nat := smallRow1531_small_left ++ smallRow1531_small_right
theorem smallRow1531_small_checked : upperNeighbors PrimePairVertices.tree_small 1531 = smallRow1531_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1531 ++ upperNeighbors PrimePairVertices.tree_small_right 1531 = _
  rw [smallRow1531_small_left_checked, smallRow1531_small_right_checked]
  rfl

def smallRow1531 : List Nat := smallRow1531_small
theorem smallRow1531_checked : upperNeighbors PrimePairVertices.smallTree 1531 = smallRow1531 := smallRow1531_small_checked

end ProjectEulerSolutions.PrimePairCache
