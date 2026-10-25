import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1543_block_000 : List Nat := [1579, 1693, 1777]
set_option maxRecDepth 2048 in
theorem smallRow1543_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1543 = smallRow1543_block_000 := by decide

def smallRow1543_block_001 : List Nat := [3217, 3463, 3511, 3931, 3967, 4093]
set_option maxRecDepth 2048 in
theorem smallRow1543_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1543 = smallRow1543_block_001 := by decide

def smallRow1543_block_002 : List Nat := [4129, 4273, 4423, 5443, 5689, 5857]
set_option maxRecDepth 2048 in
theorem smallRow1543_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1543 = smallRow1543_block_002 := by decide

def smallRow1543_block_003 : List Nat := [6451, 6571, 6823, 7159, 7879, 8053]
set_option maxRecDepth 2048 in
theorem smallRow1543_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1543 = smallRow1543_block_003 := by decide

def smallRow1543_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1543_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1543 = smallRow1543_block_004 := by decide

def smallRow1543_small_left : List Nat := smallRow1543_block_000 ++ smallRow1543_block_001
theorem smallRow1543_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1543 = smallRow1543_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1543 ++ upperNeighbors PrimePairVertices.tree_block_001 1543 = _
  rw [smallRow1543_block_000_checked, smallRow1543_block_001_checked]
  rfl

def smallRow1543_small_right_right : List Nat := smallRow1543_block_003 ++ smallRow1543_block_004
theorem smallRow1543_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1543 = smallRow1543_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1543 ++ upperNeighbors PrimePairVertices.tree_block_004 1543 = _
  rw [smallRow1543_block_003_checked, smallRow1543_block_004_checked]
  rfl

def smallRow1543_small_right : List Nat := smallRow1543_block_002 ++ smallRow1543_small_right_right
theorem smallRow1543_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1543 = smallRow1543_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1543 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1543 = _
  rw [smallRow1543_block_002_checked, smallRow1543_small_right_right_checked]
  rfl

def smallRow1543_small : List Nat := smallRow1543_small_left ++ smallRow1543_small_right
theorem smallRow1543_small_checked : upperNeighbors PrimePairVertices.tree_small 1543 = smallRow1543_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1543 ++ upperNeighbors PrimePairVertices.tree_small_right 1543 = _
  rw [smallRow1543_small_left_checked, smallRow1543_small_right_checked]
  rfl

def smallRow1543 : List Nat := smallRow1543_small
theorem smallRow1543_checked : upperNeighbors PrimePairVertices.smallTree 1543 = smallRow1543 := smallRow1543_small_checked

def smallRow1549_block_000 : List Nat := [1933]
set_option maxRecDepth 2048 in
theorem smallRow1549_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1549 = smallRow1549_block_000 := by decide

def smallRow1549_block_001 : List Nat := [2143, 2221, 2311, 2707, 2833, 3019, 3253, 3463, 3529, 3637, 3733, 3823, 4057]
set_option maxRecDepth 2048 in
theorem smallRow1549_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1549 = smallRow1549_block_001 := by decide

def smallRow1549_block_002 : List Nat := [4561, 4783, 4789, 5059, 5209, 5479, 5881]
set_option maxRecDepth 2048 in
theorem smallRow1549_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1549 = smallRow1549_block_002 := by decide

def smallRow1549_block_003 : List Nat := [6361, 6577, 6997, 7333, 7393]
set_option maxRecDepth 2048 in
theorem smallRow1549_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1549 = smallRow1549_block_003 := by decide

def smallRow1549_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1549_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1549 = smallRow1549_block_004 := by decide

def smallRow1549_small_left : List Nat := smallRow1549_block_000 ++ smallRow1549_block_001
theorem smallRow1549_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1549 = smallRow1549_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1549 ++ upperNeighbors PrimePairVertices.tree_block_001 1549 = _
  rw [smallRow1549_block_000_checked, smallRow1549_block_001_checked]
  rfl

def smallRow1549_small_right_right : List Nat := smallRow1549_block_003 ++ smallRow1549_block_004
theorem smallRow1549_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1549 = smallRow1549_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1549 ++ upperNeighbors PrimePairVertices.tree_block_004 1549 = _
  rw [smallRow1549_block_003_checked, smallRow1549_block_004_checked]
  rfl

def smallRow1549_small_right : List Nat := smallRow1549_block_002 ++ smallRow1549_small_right_right
theorem smallRow1549_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1549 = smallRow1549_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1549 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1549 = _
  rw [smallRow1549_block_002_checked, smallRow1549_small_right_right_checked]
  rfl

def smallRow1549_small : List Nat := smallRow1549_small_left ++ smallRow1549_small_right
theorem smallRow1549_small_checked : upperNeighbors PrimePairVertices.tree_small 1549 = smallRow1549_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1549 ++ upperNeighbors PrimePairVertices.tree_small_right 1549 = _
  rw [smallRow1549_small_left_checked, smallRow1549_small_right_checked]
  rfl

def smallRow1549 : List Nat := smallRow1549_small
theorem smallRow1549_checked : upperNeighbors PrimePairVertices.smallTree 1549 = smallRow1549 := smallRow1549_small_checked

def smallRow1553_block_000 : List Nat := [1583, 1613, 1697, 1811]
set_option maxRecDepth 2048 in
theorem smallRow1553_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1553 = smallRow1553_block_000 := by decide

def smallRow1553_block_001 : List Nat := [2087, 2393, 2549, 2687, 4049]
set_option maxRecDepth 2048 in
theorem smallRow1553_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1553 = smallRow1553_block_001 := by decide

def smallRow1553_block_002 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1553_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1553 = smallRow1553_block_002 := by decide

def smallRow1553_block_003 : List Nat := [6203, 6221, 6947, 7013, 7211, 7757, 7883, 8111, 8147]
set_option maxRecDepth 2048 in
theorem smallRow1553_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1553 = smallRow1553_block_003 := by decide

def smallRow1553_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1553_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1553 = smallRow1553_block_004 := by decide

def smallRow1553_small_left : List Nat := smallRow1553_block_000 ++ smallRow1553_block_001
theorem smallRow1553_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1553 = smallRow1553_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1553 ++ upperNeighbors PrimePairVertices.tree_block_001 1553 = _
  rw [smallRow1553_block_000_checked, smallRow1553_block_001_checked]
  rfl

def smallRow1553_small_right_right : List Nat := smallRow1553_block_003 ++ smallRow1553_block_004
theorem smallRow1553_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1553 = smallRow1553_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1553 ++ upperNeighbors PrimePairVertices.tree_block_004 1553 = _
  rw [smallRow1553_block_003_checked, smallRow1553_block_004_checked]
  rfl

def smallRow1553_small_right : List Nat := smallRow1553_block_002 ++ smallRow1553_small_right_right
theorem smallRow1553_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1553 = smallRow1553_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1553 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1553 = _
  rw [smallRow1553_block_002_checked, smallRow1553_small_right_right_checked]
  rfl

def smallRow1553_small : List Nat := smallRow1553_small_left ++ smallRow1553_small_right
theorem smallRow1553_small_checked : upperNeighbors PrimePairVertices.tree_small 1553 = smallRow1553_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1553 ++ upperNeighbors PrimePairVertices.tree_small_right 1553 = _
  rw [smallRow1553_small_left_checked, smallRow1553_small_right_checked]
  rfl

def smallRow1553 : List Nat := smallRow1553_small
theorem smallRow1553_checked : upperNeighbors PrimePairVertices.smallTree 1553 = smallRow1553 := smallRow1553_small_checked

def smallRow1559_block_000 : List Nat := [1721, 1787]
set_option maxRecDepth 2048 in
theorem smallRow1559_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1559 = smallRow1559_block_000 := by decide

def smallRow1559_block_001 : List Nat := [2477, 2693, 3191, 3593, 3911]
set_option maxRecDepth 2048 in
theorem smallRow1559_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1559 = smallRow1559_block_001 := by decide

def smallRow1559_block_002 : List Nat := [5441, 5507, 5861, 5897]
set_option maxRecDepth 2048 in
theorem smallRow1559_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1559 = smallRow1559_block_002 := by decide

def smallRow1559_block_003 : List Nat := [6299, 7079, 7559, 8171]
set_option maxRecDepth 2048 in
theorem smallRow1559_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1559 = smallRow1559_block_003 := by decide

def smallRow1559_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow1559_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1559 = smallRow1559_block_004 := by decide

def smallRow1559_small_left : List Nat := smallRow1559_block_000 ++ smallRow1559_block_001
theorem smallRow1559_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1559 = smallRow1559_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1559 ++ upperNeighbors PrimePairVertices.tree_block_001 1559 = _
  rw [smallRow1559_block_000_checked, smallRow1559_block_001_checked]
  rfl

def smallRow1559_small_right_right : List Nat := smallRow1559_block_003 ++ smallRow1559_block_004
theorem smallRow1559_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1559 = smallRow1559_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1559 ++ upperNeighbors PrimePairVertices.tree_block_004 1559 = _
  rw [smallRow1559_block_003_checked, smallRow1559_block_004_checked]
  rfl

def smallRow1559_small_right : List Nat := smallRow1559_block_002 ++ smallRow1559_small_right_right
theorem smallRow1559_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1559 = smallRow1559_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1559 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1559 = _
  rw [smallRow1559_block_002_checked, smallRow1559_small_right_right_checked]
  rfl

def smallRow1559_small : List Nat := smallRow1559_small_left ++ smallRow1559_small_right
theorem smallRow1559_small_checked : upperNeighbors PrimePairVertices.tree_small 1559 = smallRow1559_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1559 ++ upperNeighbors PrimePairVertices.tree_small_right 1559 = _
  rw [smallRow1559_small_left_checked, smallRow1559_small_right_checked]
  rfl

def smallRow1559 : List Nat := smallRow1559_small
theorem smallRow1559_checked : upperNeighbors PrimePairVertices.smallTree 1559 = smallRow1559 := smallRow1559_small_checked

def smallRow1567_block_000 : List Nat := [1879]
set_option maxRecDepth 2048 in
theorem smallRow1567_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1567 = smallRow1567_block_000 := by decide

def smallRow1567_block_001 : List Nat := [2161, 2311, 2539, 2551, 2593, 3919]
set_option maxRecDepth 2048 in
theorem smallRow1567_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1567 = smallRow1567_block_001 := by decide

def smallRow1567_block_002 : List Nat := [4441, 4507, 4567, 4597, 5431, 5641]
set_option maxRecDepth 2048 in
theorem smallRow1567_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1567 = smallRow1567_block_002 := by decide

def smallRow1567_block_003 : List Nat := [6277, 6481, 7027, 7759]
set_option maxRecDepth 2048 in
theorem smallRow1567_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1567 = smallRow1567_block_003 := by decide

def smallRow1567_block_004 : List Nat := [8293]
set_option maxRecDepth 2048 in
theorem smallRow1567_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1567 = smallRow1567_block_004 := by decide

def smallRow1567_small_left : List Nat := smallRow1567_block_000 ++ smallRow1567_block_001
theorem smallRow1567_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1567 = smallRow1567_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1567 ++ upperNeighbors PrimePairVertices.tree_block_001 1567 = _
  rw [smallRow1567_block_000_checked, smallRow1567_block_001_checked]
  rfl

def smallRow1567_small_right_right : List Nat := smallRow1567_block_003 ++ smallRow1567_block_004
theorem smallRow1567_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1567 = smallRow1567_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1567 ++ upperNeighbors PrimePairVertices.tree_block_004 1567 = _
  rw [smallRow1567_block_003_checked, smallRow1567_block_004_checked]
  rfl

def smallRow1567_small_right : List Nat := smallRow1567_block_002 ++ smallRow1567_small_right_right
theorem smallRow1567_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1567 = smallRow1567_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1567 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1567 = _
  rw [smallRow1567_block_002_checked, smallRow1567_small_right_right_checked]
  rfl

def smallRow1567_small : List Nat := smallRow1567_small_left ++ smallRow1567_small_right
theorem smallRow1567_small_checked : upperNeighbors PrimePairVertices.tree_small 1567 = smallRow1567_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1567 ++ upperNeighbors PrimePairVertices.tree_small_right 1567 = _
  rw [smallRow1567_small_left_checked, smallRow1567_small_right_checked]
  rfl

def smallRow1567 : List Nat := smallRow1567_small
theorem smallRow1567_checked : upperNeighbors PrimePairVertices.smallTree 1567 = smallRow1567 := smallRow1567_small_checked

def smallRow1571_block_000 : List Nat := [1607, 1907]
set_option maxRecDepth 2048 in
theorem smallRow1571_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1571 = smallRow1571_block_000 := by decide

def smallRow1571_block_001 : List Nat := [2399, 2819]
set_option maxRecDepth 2048 in
theorem smallRow1571_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1571 = smallRow1571_block_001 := by decide

def smallRow1571_block_002 : List Nat := [4133, 4157, 4547, 5333, 5879, 5981]
set_option maxRecDepth 2048 in
theorem smallRow1571_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1571 = smallRow1571_block_002 := by decide

def smallRow1571_block_003 : List Nat := [6563, 6983, 7529, 7907]
set_option maxRecDepth 2048 in
theorem smallRow1571_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1571 = smallRow1571_block_003 := by decide

def smallRow1571_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1571_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1571 = smallRow1571_block_004 := by decide

def smallRow1571_small_left : List Nat := smallRow1571_block_000 ++ smallRow1571_block_001
theorem smallRow1571_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1571 = smallRow1571_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1571 ++ upperNeighbors PrimePairVertices.tree_block_001 1571 = _
  rw [smallRow1571_block_000_checked, smallRow1571_block_001_checked]
  rfl

def smallRow1571_small_right_right : List Nat := smallRow1571_block_003 ++ smallRow1571_block_004
theorem smallRow1571_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1571 = smallRow1571_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1571 ++ upperNeighbors PrimePairVertices.tree_block_004 1571 = _
  rw [smallRow1571_block_003_checked, smallRow1571_block_004_checked]
  rfl

def smallRow1571_small_right : List Nat := smallRow1571_block_002 ++ smallRow1571_small_right_right
theorem smallRow1571_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1571 = smallRow1571_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1571 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1571 = _
  rw [smallRow1571_block_002_checked, smallRow1571_small_right_right_checked]
  rfl

def smallRow1571_small : List Nat := smallRow1571_small_left ++ smallRow1571_small_right
theorem smallRow1571_small_checked : upperNeighbors PrimePairVertices.tree_small 1571 = smallRow1571_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1571 ++ upperNeighbors PrimePairVertices.tree_small_right 1571 = _
  rw [smallRow1571_small_left_checked, smallRow1571_small_right_checked]
  rfl

def smallRow1571 : List Nat := smallRow1571_small
theorem smallRow1571_checked : upperNeighbors PrimePairVertices.smallTree 1571 = smallRow1571 := smallRow1571_small_checked

def smallRow1579_block_000 : List Nat := [1627, 1669]
set_option maxRecDepth 2048 in
theorem smallRow1579_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1579 = smallRow1579_block_000 := by decide

def smallRow1579_block_001 : List Nat := [3301, 3907]
set_option maxRecDepth 2048 in
theorem smallRow1579_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1579 = smallRow1579_block_001 := by decide

def smallRow1579_block_002 : List Nat := [4783, 6073]
set_option maxRecDepth 2048 in
theorem smallRow1579_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1579 = smallRow1579_block_002 := by decide

def smallRow1579_block_003 : List Nat := [6301, 7477, 7723, 8059]
set_option maxRecDepth 2048 in
theorem smallRow1579_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1579 = smallRow1579_block_003 := by decide

def smallRow1579_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1579_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1579 = smallRow1579_block_004 := by decide

def smallRow1579_small_left : List Nat := smallRow1579_block_000 ++ smallRow1579_block_001
theorem smallRow1579_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1579 = smallRow1579_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1579 ++ upperNeighbors PrimePairVertices.tree_block_001 1579 = _
  rw [smallRow1579_block_000_checked, smallRow1579_block_001_checked]
  rfl

def smallRow1579_small_right_right : List Nat := smallRow1579_block_003 ++ smallRow1579_block_004
theorem smallRow1579_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1579 = smallRow1579_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1579 ++ upperNeighbors PrimePairVertices.tree_block_004 1579 = _
  rw [smallRow1579_block_003_checked, smallRow1579_block_004_checked]
  rfl

def smallRow1579_small_right : List Nat := smallRow1579_block_002 ++ smallRow1579_small_right_right
theorem smallRow1579_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1579 = smallRow1579_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1579 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1579 = _
  rw [smallRow1579_block_002_checked, smallRow1579_small_right_right_checked]
  rfl

def smallRow1579_small : List Nat := smallRow1579_small_left ++ smallRow1579_small_right
theorem smallRow1579_small_checked : upperNeighbors PrimePairVertices.tree_small 1579 = smallRow1579_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1579 ++ upperNeighbors PrimePairVertices.tree_small_right 1579 = _
  rw [smallRow1579_small_left_checked, smallRow1579_small_right_checked]
  rfl

def smallRow1579 : List Nat := smallRow1579_small
theorem smallRow1579_checked : upperNeighbors PrimePairVertices.smallTree 1579 = smallRow1579 := smallRow1579_small_checked

def smallRow1583_block_000 : List Nat := [1619, 1637, 1877, 1973]
set_option maxRecDepth 2048 in
theorem smallRow1583_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1583 = smallRow1583_block_000 := by decide

def smallRow1583_block_001 : List Nat := [2447, 3347, 3557, 3929, 4001]
set_option maxRecDepth 2048 in
theorem smallRow1583_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1583 = smallRow1583_block_001 := by decide

def smallRow1583_block_002 : List Nat := [4229, 4283, 4451, 4799, 5009, 5333, 5861, 5879, 6113, 6143]
set_option maxRecDepth 2048 in
theorem smallRow1583_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1583 = smallRow1583_block_002 := by decide

def smallRow1583_block_003 : List Nat := [6287, 6551, 6659, 6803, 6827, 6869, 7043, 7079, 7643, 7949, 8171]
set_option maxRecDepth 2048 in
theorem smallRow1583_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1583 = smallRow1583_block_003 := by decide

def smallRow1583_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow1583_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1583 = smallRow1583_block_004 := by decide

def smallRow1583_small_left : List Nat := smallRow1583_block_000 ++ smallRow1583_block_001
theorem smallRow1583_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1583 = smallRow1583_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1583 ++ upperNeighbors PrimePairVertices.tree_block_001 1583 = _
  rw [smallRow1583_block_000_checked, smallRow1583_block_001_checked]
  rfl

def smallRow1583_small_right_right : List Nat := smallRow1583_block_003 ++ smallRow1583_block_004
theorem smallRow1583_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1583 = smallRow1583_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1583 ++ upperNeighbors PrimePairVertices.tree_block_004 1583 = _
  rw [smallRow1583_block_003_checked, smallRow1583_block_004_checked]
  rfl

def smallRow1583_small_right : List Nat := smallRow1583_block_002 ++ smallRow1583_small_right_right
theorem smallRow1583_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1583 = smallRow1583_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1583 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1583 = _
  rw [smallRow1583_block_002_checked, smallRow1583_small_right_right_checked]
  rfl

def smallRow1583_small : List Nat := smallRow1583_small_left ++ smallRow1583_small_right
theorem smallRow1583_small_checked : upperNeighbors PrimePairVertices.tree_small 1583 = smallRow1583_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1583 ++ upperNeighbors PrimePairVertices.tree_small_right 1583 = _
  rw [smallRow1583_small_left_checked, smallRow1583_small_right_checked]
  rfl

def smallRow1583 : List Nat := smallRow1583_small
theorem smallRow1583_checked : upperNeighbors PrimePairVertices.smallTree 1583 = smallRow1583 := smallRow1583_small_checked

end ProjectEulerSolutions.PrimePairCache
