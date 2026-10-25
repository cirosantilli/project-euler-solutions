import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow587_block_000 : List Nat := [617, 677, 1163, 1223, 1571]
set_option maxRecDepth 2048 in
theorem smallRow587_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 587 = smallRow587_block_000 := by decide

def smallRow587_block_001 : List Nat := [2357, 2549, 2903, 3011, 3719, 3767, 3851]
set_option maxRecDepth 2048 in
theorem smallRow587_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 587 = smallRow587_block_001 := by decide

def smallRow587_block_002 : List Nat := [4133, 4229, 4397, 4793, 5021, 5087, 6011]
set_option maxRecDepth 2048 in
theorem smallRow587_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 587 = smallRow587_block_002 := by decide

def smallRow587_block_003 : List Nat := [6323, 6911, 6977, 7127, 7247, 7481, 7523, 7829, 8069, 8123]
set_option maxRecDepth 2048 in
theorem smallRow587_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 587 = smallRow587_block_003 := by decide

def smallRow587_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow587_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 587 = smallRow587_block_004 := by decide

def smallRow587_small_left : List Nat := smallRow587_block_000 ++ smallRow587_block_001
theorem smallRow587_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 587 = smallRow587_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 587 ++ upperNeighbors PrimePairVertices.tree_block_001 587 = _
  rw [smallRow587_block_000_checked, smallRow587_block_001_checked]
  rfl

def smallRow587_small_right_right : List Nat := smallRow587_block_003 ++ smallRow587_block_004
theorem smallRow587_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 587 = smallRow587_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 587 ++ upperNeighbors PrimePairVertices.tree_block_004 587 = _
  rw [smallRow587_block_003_checked, smallRow587_block_004_checked]
  rfl

def smallRow587_small_right : List Nat := smallRow587_block_002 ++ smallRow587_small_right_right
theorem smallRow587_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 587 = smallRow587_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 587 ++ upperNeighbors PrimePairVertices.tree_small_right_right 587 = _
  rw [smallRow587_block_002_checked, smallRow587_small_right_right_checked]
  rfl

def smallRow587_small : List Nat := smallRow587_small_left ++ smallRow587_small_right
theorem smallRow587_small_checked : upperNeighbors PrimePairVertices.tree_small 587 = smallRow587_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 587 ++ upperNeighbors PrimePairVertices.tree_small_right 587 = _
  rw [smallRow587_small_left_checked, smallRow587_small_right_checked]
  rfl

def smallRow587 : List Nat := smallRow587_small
theorem smallRow587_checked : upperNeighbors PrimePairVertices.smallTree 587 = smallRow587 := smallRow587_small_checked

def smallRow593_block_000 : List Nat := [647, 863, 977]
set_option maxRecDepth 2048 in
theorem smallRow593_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 593 = smallRow593_block_000 := by decide

def smallRow593_block_001 : List Nat := [2141, 2411, 2417, 3203, 3251, 3461, 3467, 3539, 3581, 4001]
set_option maxRecDepth 2048 in
theorem smallRow593_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 593 = smallRow593_block_001 := by decide

def smallRow593_block_002 : List Nat := [4133, 4211, 5393, 5399, 5639, 5861, 5879, 5987]
set_option maxRecDepth 2048 in
theorem smallRow593_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 593 = smallRow593_block_002 := by decide

def smallRow593_block_003 : List Nat := [6653, 6857, 7433, 7559, 7577, 7949, 8123]
set_option maxRecDepth 2048 in
theorem smallRow593_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 593 = smallRow593_block_003 := by decide

def smallRow593_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow593_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 593 = smallRow593_block_004 := by decide

def smallRow593_small_left : List Nat := smallRow593_block_000 ++ smallRow593_block_001
theorem smallRow593_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 593 = smallRow593_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 593 ++ upperNeighbors PrimePairVertices.tree_block_001 593 = _
  rw [smallRow593_block_000_checked, smallRow593_block_001_checked]
  rfl

def smallRow593_small_right_right : List Nat := smallRow593_block_003 ++ smallRow593_block_004
theorem smallRow593_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 593 = smallRow593_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 593 ++ upperNeighbors PrimePairVertices.tree_block_004 593 = _
  rw [smallRow593_block_003_checked, smallRow593_block_004_checked]
  rfl

def smallRow593_small_right : List Nat := smallRow593_block_002 ++ smallRow593_small_right_right
theorem smallRow593_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 593 = smallRow593_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 593 ++ upperNeighbors PrimePairVertices.tree_small_right_right 593 = _
  rw [smallRow593_block_002_checked, smallRow593_small_right_right_checked]
  rfl

def smallRow593_small : List Nat := smallRow593_small_left ++ smallRow593_small_right
theorem smallRow593_small_checked : upperNeighbors PrimePairVertices.tree_small 593 = smallRow593_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 593 ++ upperNeighbors PrimePairVertices.tree_small_right 593 = _
  rw [smallRow593_small_left_checked, smallRow593_small_right_checked]
  rfl

def smallRow593 : List Nat := smallRow593_small
theorem smallRow593_checked : upperNeighbors PrimePairVertices.smallTree 593 = smallRow593 := smallRow593_small_checked

def smallRow599_block_000 : List Nat := [719, 941, 1709, 1823, 1907]
set_option maxRecDepth 2048 in
theorem smallRow599_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 599 = smallRow599_block_000 := by decide

def smallRow599_block_001 : List Nat := [3023, 3557, 3671, 3761, 4091]
set_option maxRecDepth 2048 in
theorem smallRow599_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 599 = smallRow599_block_001 := by decide

def smallRow599_block_002 : List Nat := [5081, 5309, 5399, 5573, 6047]
set_option maxRecDepth 2048 in
theorem smallRow599_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 599 = smallRow599_block_002 := by decide

def smallRow599_block_003 : List Nat := [7517, 7841, 7877, 7883, 8039]
set_option maxRecDepth 2048 in
theorem smallRow599_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 599 = smallRow599_block_003 := by decide

def smallRow599_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow599_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 599 = smallRow599_block_004 := by decide

def smallRow599_small_left : List Nat := smallRow599_block_000 ++ smallRow599_block_001
theorem smallRow599_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 599 = smallRow599_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 599 ++ upperNeighbors PrimePairVertices.tree_block_001 599 = _
  rw [smallRow599_block_000_checked, smallRow599_block_001_checked]
  rfl

def smallRow599_small_right_right : List Nat := smallRow599_block_003 ++ smallRow599_block_004
theorem smallRow599_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 599 = smallRow599_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 599 ++ upperNeighbors PrimePairVertices.tree_block_004 599 = _
  rw [smallRow599_block_003_checked, smallRow599_block_004_checked]
  rfl

def smallRow599_small_right : List Nat := smallRow599_block_002 ++ smallRow599_small_right_right
theorem smallRow599_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 599 = smallRow599_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 599 ++ upperNeighbors PrimePairVertices.tree_small_right_right 599 = _
  rw [smallRow599_block_002_checked, smallRow599_small_right_right_checked]
  rfl

def smallRow599_small : List Nat := smallRow599_small_left ++ smallRow599_small_right
theorem smallRow599_small_checked : upperNeighbors PrimePairVertices.tree_small 599 = smallRow599_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 599 ++ upperNeighbors PrimePairVertices.tree_small_right 599 = _
  rw [smallRow599_small_left_checked, smallRow599_small_right_checked]
  rfl

def smallRow599 : List Nat := smallRow599_small
theorem smallRow599_checked : upperNeighbors PrimePairVertices.smallTree 599 = smallRow599 := smallRow599_small_checked

def smallRow601_block_000 : List Nat := [823, 1171, 1321, 1597, 1627, 1867, 1879, 2011]
set_option maxRecDepth 2048 in
theorem smallRow601_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 601 = smallRow601_block_000 := by decide

def smallRow601_block_001 : List Nat := [2689, 2719, 3187]
set_option maxRecDepth 2048 in
theorem smallRow601_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 601 = smallRow601_block_001 := by decide

def smallRow601_block_002 : List Nat := [4159, 4663, 4909, 5197]
set_option maxRecDepth 2048 in
theorem smallRow601_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 601 = smallRow601_block_002 := by decide

def smallRow601_block_003 : List Nat := [7213, 7411, 7603, 7639]
set_option maxRecDepth 2048 in
theorem smallRow601_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 601 = smallRow601_block_003 := by decide

def smallRow601_block_004 : List Nat := [8317, 8377]
set_option maxRecDepth 2048 in
theorem smallRow601_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 601 = smallRow601_block_004 := by decide

def smallRow601_small_left : List Nat := smallRow601_block_000 ++ smallRow601_block_001
theorem smallRow601_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 601 = smallRow601_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 601 ++ upperNeighbors PrimePairVertices.tree_block_001 601 = _
  rw [smallRow601_block_000_checked, smallRow601_block_001_checked]
  rfl

def smallRow601_small_right_right : List Nat := smallRow601_block_003 ++ smallRow601_block_004
theorem smallRow601_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 601 = smallRow601_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 601 ++ upperNeighbors PrimePairVertices.tree_block_004 601 = _
  rw [smallRow601_block_003_checked, smallRow601_block_004_checked]
  rfl

def smallRow601_small_right : List Nat := smallRow601_block_002 ++ smallRow601_small_right_right
theorem smallRow601_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 601 = smallRow601_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 601 ++ upperNeighbors PrimePairVertices.tree_small_right_right 601 = _
  rw [smallRow601_block_002_checked, smallRow601_small_right_right_checked]
  rfl

def smallRow601_small : List Nat := smallRow601_small_left ++ smallRow601_small_right
theorem smallRow601_small_checked : upperNeighbors PrimePairVertices.tree_small 601 = smallRow601_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 601 ++ upperNeighbors PrimePairVertices.tree_small_right 601 = _
  rw [smallRow601_small_left_checked, smallRow601_small_right_checked]
  rfl

def smallRow601 : List Nat := smallRow601_small
theorem smallRow601_checked : upperNeighbors PrimePairVertices.smallTree 601 = smallRow601 := smallRow601_small_checked

def smallRow607_block_000 : List Nat := [619, 967, 991, 1297, 1453, 1693, 1999]
set_option maxRecDepth 2048 in
theorem smallRow607_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 607 = smallRow607_block_000 := by decide

def smallRow607_block_001 : List Nat := [2467, 2887, 3037, 3733, 3889]
set_option maxRecDepth 2048 in
theorem smallRow607_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 607 = smallRow607_block_001 := by decide

def smallRow607_block_002 : List Nat := [4363, 4447, 4483, 4561, 4591, 4603, 5077, 5107, 6121]
set_option maxRecDepth 2048 in
theorem smallRow607_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 607 = smallRow607_block_002 := by decide

def smallRow607_block_003 : List Nat := [6337, 6673, 6709, 6883, 7039, 7573, 7639]
set_option maxRecDepth 2048 in
theorem smallRow607_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 607 = smallRow607_block_003 := by decide

def smallRow607_block_004 : List Nat := [8287, 8353]
set_option maxRecDepth 2048 in
theorem smallRow607_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 607 = smallRow607_block_004 := by decide

def smallRow607_small_left : List Nat := smallRow607_block_000 ++ smallRow607_block_001
theorem smallRow607_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 607 = smallRow607_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 607 ++ upperNeighbors PrimePairVertices.tree_block_001 607 = _
  rw [smallRow607_block_000_checked, smallRow607_block_001_checked]
  rfl

def smallRow607_small_right_right : List Nat := smallRow607_block_003 ++ smallRow607_block_004
theorem smallRow607_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 607 = smallRow607_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 607 ++ upperNeighbors PrimePairVertices.tree_block_004 607 = _
  rw [smallRow607_block_003_checked, smallRow607_block_004_checked]
  rfl

def smallRow607_small_right : List Nat := smallRow607_block_002 ++ smallRow607_small_right_right
theorem smallRow607_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 607 = smallRow607_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 607 ++ upperNeighbors PrimePairVertices.tree_small_right_right 607 = _
  rw [smallRow607_block_002_checked, smallRow607_small_right_right_checked]
  rfl

def smallRow607_small : List Nat := smallRow607_small_left ++ smallRow607_small_right
theorem smallRow607_small_checked : upperNeighbors PrimePairVertices.tree_small 607 = smallRow607_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 607 ++ upperNeighbors PrimePairVertices.tree_small_right 607 = _
  rw [smallRow607_small_left_checked, smallRow607_small_right_checked]
  rfl

def smallRow607 : List Nat := smallRow607_small
theorem smallRow607_checked : upperNeighbors PrimePairVertices.smallTree 607 = smallRow607 := smallRow607_small_checked

def smallRow613_block_000 : List Nat := [661, 673, 829, 883, 1303, 1753]
set_option maxRecDepth 2048 in
theorem smallRow613_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 613 = smallRow613_block_000 := by decide

def smallRow613_block_001 : List Nat := [2089, 2851, 3037, 3643, 3769, 3943, 4021]
set_option maxRecDepth 2048 in
theorem smallRow613_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 613 = smallRow613_block_001 := by decide

def smallRow613_block_002 : List Nat := [4129, 4153, 4339, 4363, 4801, 5581, 5743]
set_option maxRecDepth 2048 in
theorem smallRow613_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 613 = smallRow613_block_002 := by decide

def smallRow613_block_003 : List Nat := [6379, 6547, 6673, 6679, 6733, 6967, 7069, 7177, 8191]
set_option maxRecDepth 2048 in
theorem smallRow613_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 613 = smallRow613_block_003 := by decide

def smallRow613_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow613_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 613 = smallRow613_block_004 := by decide

def smallRow613_small_left : List Nat := smallRow613_block_000 ++ smallRow613_block_001
theorem smallRow613_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 613 = smallRow613_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 613 ++ upperNeighbors PrimePairVertices.tree_block_001 613 = _
  rw [smallRow613_block_000_checked, smallRow613_block_001_checked]
  rfl

def smallRow613_small_right_right : List Nat := smallRow613_block_003 ++ smallRow613_block_004
theorem smallRow613_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 613 = smallRow613_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 613 ++ upperNeighbors PrimePairVertices.tree_block_004 613 = _
  rw [smallRow613_block_003_checked, smallRow613_block_004_checked]
  rfl

def smallRow613_small_right : List Nat := smallRow613_block_002 ++ smallRow613_small_right_right
theorem smallRow613_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 613 = smallRow613_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 613 ++ upperNeighbors PrimePairVertices.tree_small_right_right 613 = _
  rw [smallRow613_block_002_checked, smallRow613_small_right_right_checked]
  rfl

def smallRow613_small : List Nat := smallRow613_small_left ++ smallRow613_small_right
theorem smallRow613_small_checked : upperNeighbors PrimePairVertices.tree_small 613 = smallRow613_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 613 ++ upperNeighbors PrimePairVertices.tree_small_right 613 = _
  rw [smallRow613_small_left_checked, smallRow613_small_right_checked]
  rfl

def smallRow613 : List Nat := smallRow613_small
theorem smallRow613_checked : upperNeighbors PrimePairVertices.smallTree 613 = smallRow613 := smallRow613_small_checked

def smallRow617_block_000 : List Nat := [647, 887, 983, 1091, 1301, 1427]
set_option maxRecDepth 2048 in
theorem smallRow617_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 617 = smallRow617_block_000 := by decide

def smallRow617_block_001 : List Nat := [2063, 2657, 2741, 2909, 3593, 4001]
set_option maxRecDepth 2048 in
theorem smallRow617_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 617 = smallRow617_block_001 := by decide

def smallRow617_block_002 : List Nat := [4253, 4391, 5261, 5333, 5693, 6011]
set_option maxRecDepth 2048 in
theorem smallRow617_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 617 = smallRow617_block_002 := by decide

def smallRow617_block_003 : List Nat := [6323, 7229, 7541, 8087]
set_option maxRecDepth 2048 in
theorem smallRow617_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 617 = smallRow617_block_003 := by decide

def smallRow617_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow617_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 617 = smallRow617_block_004 := by decide

def smallRow617_small_left : List Nat := smallRow617_block_000 ++ smallRow617_block_001
theorem smallRow617_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 617 = smallRow617_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 617 ++ upperNeighbors PrimePairVertices.tree_block_001 617 = _
  rw [smallRow617_block_000_checked, smallRow617_block_001_checked]
  rfl

def smallRow617_small_right_right : List Nat := smallRow617_block_003 ++ smallRow617_block_004
theorem smallRow617_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 617 = smallRow617_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 617 ++ upperNeighbors PrimePairVertices.tree_block_004 617 = _
  rw [smallRow617_block_003_checked, smallRow617_block_004_checked]
  rfl

def smallRow617_small_right : List Nat := smallRow617_block_002 ++ smallRow617_small_right_right
theorem smallRow617_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 617 = smallRow617_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 617 ++ upperNeighbors PrimePairVertices.tree_small_right_right 617 = _
  rw [smallRow617_block_002_checked, smallRow617_small_right_right_checked]
  rfl

def smallRow617_small : List Nat := smallRow617_small_left ++ smallRow617_small_right
theorem smallRow617_small_checked : upperNeighbors PrimePairVertices.tree_small 617 = smallRow617_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 617 ++ upperNeighbors PrimePairVertices.tree_small_right 617 = _
  rw [smallRow617_small_left_checked, smallRow617_small_right_checked]
  rfl

def smallRow617 : List Nat := smallRow617_small
theorem smallRow617_checked : upperNeighbors PrimePairVertices.smallTree 617 = smallRow617 := smallRow617_small_checked

def smallRow619_block_000 : List Nat := [811, 1051, 1579, 1597, 1747]
set_option maxRecDepth 2048 in
theorem smallRow619_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 619 = smallRow619_block_000 := by decide

def smallRow619_block_001 : List Nat := [2671, 2707, 3571, 3793]
set_option maxRecDepth 2048 in
theorem smallRow619_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 619 = smallRow619_block_001 := by decide

def smallRow619_block_002 : List Nat := [4621, 5419]
set_option maxRecDepth 2048 in
theorem smallRow619_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 619 = smallRow619_block_002 := by decide

def smallRow619_block_003 : List Nat := [6991, 7207, 7309, 7459, 7687, 7753]
set_option maxRecDepth 2048 in
theorem smallRow619_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 619 = smallRow619_block_003 := by decide

def smallRow619_block_004 : List Nat := [8377]
set_option maxRecDepth 2048 in
theorem smallRow619_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 619 = smallRow619_block_004 := by decide

def smallRow619_small_left : List Nat := smallRow619_block_000 ++ smallRow619_block_001
theorem smallRow619_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 619 = smallRow619_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 619 ++ upperNeighbors PrimePairVertices.tree_block_001 619 = _
  rw [smallRow619_block_000_checked, smallRow619_block_001_checked]
  rfl

def smallRow619_small_right_right : List Nat := smallRow619_block_003 ++ smallRow619_block_004
theorem smallRow619_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 619 = smallRow619_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 619 ++ upperNeighbors PrimePairVertices.tree_block_004 619 = _
  rw [smallRow619_block_003_checked, smallRow619_block_004_checked]
  rfl

def smallRow619_small_right : List Nat := smallRow619_block_002 ++ smallRow619_small_right_right
theorem smallRow619_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 619 = smallRow619_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 619 ++ upperNeighbors PrimePairVertices.tree_small_right_right 619 = _
  rw [smallRow619_block_002_checked, smallRow619_small_right_right_checked]
  rfl

def smallRow619_small : List Nat := smallRow619_small_left ++ smallRow619_small_right
theorem smallRow619_small_checked : upperNeighbors PrimePairVertices.tree_small 619 = smallRow619_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 619 ++ upperNeighbors PrimePairVertices.tree_small_right 619 = _
  rw [smallRow619_small_left_checked, smallRow619_small_right_checked]
  rfl

def smallRow619 : List Nat := smallRow619_small
theorem smallRow619_checked : upperNeighbors PrimePairVertices.smallTree 619 = smallRow619 := smallRow619_small_checked

end ProjectEulerSolutions.PrimePairCache
