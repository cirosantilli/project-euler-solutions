import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow631_block_000 : List Nat := [739, 751, 1297, 1483]
set_option maxRecDepth 2048 in
theorem smallRow631_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 631 = smallRow631_block_000 := by decide

def smallRow631_block_001 : List Nat := [2221, 3253, 3457, 3709, 3889, 4051, 4057]
set_option maxRecDepth 2048 in
theorem smallRow631_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 631 = smallRow631_block_001 := by decide

def smallRow631_block_002 : List Nat := [4801, 4933, 5479, 5527, 5557, 5737]
set_option maxRecDepth 2048 in
theorem smallRow631_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 631 = smallRow631_block_002 := by decide

def smallRow631_block_003 : List Nat := [6199, 7243, 7681, 7873, 8161]
set_option maxRecDepth 2048 in
theorem smallRow631_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 631 = smallRow631_block_003 := by decide

def smallRow631_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow631_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 631 = smallRow631_block_004 := by decide

def smallRow631_small_left : List Nat := smallRow631_block_000 ++ smallRow631_block_001
theorem smallRow631_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 631 = smallRow631_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 631 ++ upperNeighbors PrimePairVertices.tree_block_001 631 = _
  rw [smallRow631_block_000_checked, smallRow631_block_001_checked]
  rfl

def smallRow631_small_right_right : List Nat := smallRow631_block_003 ++ smallRow631_block_004
theorem smallRow631_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 631 = smallRow631_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 631 ++ upperNeighbors PrimePairVertices.tree_block_004 631 = _
  rw [smallRow631_block_003_checked, smallRow631_block_004_checked]
  rfl

def smallRow631_small_right : List Nat := smallRow631_block_002 ++ smallRow631_small_right_right
theorem smallRow631_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 631 = smallRow631_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 631 ++ upperNeighbors PrimePairVertices.tree_small_right_right 631 = _
  rw [smallRow631_block_002_checked, smallRow631_small_right_right_checked]
  rfl

def smallRow631_small : List Nat := smallRow631_small_left ++ smallRow631_small_right
theorem smallRow631_small_checked : upperNeighbors PrimePairVertices.tree_small 631 = smallRow631_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 631 ++ upperNeighbors PrimePairVertices.tree_small_right 631 = _
  rw [smallRow631_small_left_checked, smallRow631_small_right_checked]
  rfl

def smallRow631 : List Nat := smallRow631_small
theorem smallRow631_checked : upperNeighbors PrimePairVertices.smallTree 631 = smallRow631 := smallRow631_small_checked

def smallRow641_block_000 : List Nat := [821, 863, 881, 929]
set_option maxRecDepth 2048 in
theorem smallRow641_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 641 = smallRow641_block_000 := by decide

def smallRow641_block_001 : List Nat := [2543, 2801, 2957, 4013]
set_option maxRecDepth 2048 in
theorem smallRow641_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 641 = smallRow641_block_001 := by decide

def smallRow641_block_002 : List Nat := [4787, 4937, 5237, 5441, 5693]
set_option maxRecDepth 2048 in
theorem smallRow641_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 641 = smallRow641_block_002 := by decide

def smallRow641_block_003 : List Nat := [6197, 6203, 6653, 7247, 7457]
set_option maxRecDepth 2048 in
theorem smallRow641_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 641 = smallRow641_block_003 := by decide

def smallRow641_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow641_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 641 = smallRow641_block_004 := by decide

def smallRow641_small_left : List Nat := smallRow641_block_000 ++ smallRow641_block_001
theorem smallRow641_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 641 = smallRow641_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 641 ++ upperNeighbors PrimePairVertices.tree_block_001 641 = _
  rw [smallRow641_block_000_checked, smallRow641_block_001_checked]
  rfl

def smallRow641_small_right_right : List Nat := smallRow641_block_003 ++ smallRow641_block_004
theorem smallRow641_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 641 = smallRow641_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 641 ++ upperNeighbors PrimePairVertices.tree_block_004 641 = _
  rw [smallRow641_block_003_checked, smallRow641_block_004_checked]
  rfl

def smallRow641_small_right : List Nat := smallRow641_block_002 ++ smallRow641_small_right_right
theorem smallRow641_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 641 = smallRow641_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 641 ++ upperNeighbors PrimePairVertices.tree_small_right_right 641 = _
  rw [smallRow641_block_002_checked, smallRow641_small_right_right_checked]
  rfl

def smallRow641_small : List Nat := smallRow641_small_left ++ smallRow641_small_right
theorem smallRow641_small_checked : upperNeighbors PrimePairVertices.tree_small 641 = smallRow641_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 641 ++ upperNeighbors PrimePairVertices.tree_small_right 641 = _
  rw [smallRow641_small_left_checked, smallRow641_small_right_checked]
  rfl

def smallRow641 : List Nat := smallRow641_small
theorem smallRow641_checked : upperNeighbors PrimePairVertices.smallTree 641 = smallRow641 := smallRow641_small_checked

def smallRow643_block_000 : List Nat := [751, 991, 1213, 1993, 2017]
set_option maxRecDepth 2048 in
theorem smallRow643_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 643 = smallRow643_block_000 := by decide

def smallRow643_block_001 : List Nat := [2053, 2383, 2917]
set_option maxRecDepth 2048 in
theorem smallRow643_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 643 = smallRow643_block_001 := by decide

def smallRow643_block_002 : List Nat := [4111, 5119, 5413, 5563, 6007, 6037]
set_option maxRecDepth 2048 in
theorem smallRow643_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 643 = smallRow643_block_002 := by decide

def smallRow643_block_003 : List Nat := [7393, 7477, 8059, 8191]
set_option maxRecDepth 2048 in
theorem smallRow643_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 643 = smallRow643_block_003 := by decide

def smallRow643_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow643_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 643 = smallRow643_block_004 := by decide

def smallRow643_small_left : List Nat := smallRow643_block_000 ++ smallRow643_block_001
theorem smallRow643_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 643 = smallRow643_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 643 ++ upperNeighbors PrimePairVertices.tree_block_001 643 = _
  rw [smallRow643_block_000_checked, smallRow643_block_001_checked]
  rfl

def smallRow643_small_right_right : List Nat := smallRow643_block_003 ++ smallRow643_block_004
theorem smallRow643_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 643 = smallRow643_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 643 ++ upperNeighbors PrimePairVertices.tree_block_004 643 = _
  rw [smallRow643_block_003_checked, smallRow643_block_004_checked]
  rfl

def smallRow643_small_right : List Nat := smallRow643_block_002 ++ smallRow643_small_right_right
theorem smallRow643_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 643 = smallRow643_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 643 ++ upperNeighbors PrimePairVertices.tree_small_right_right 643 = _
  rw [smallRow643_block_002_checked, smallRow643_small_right_right_checked]
  rfl

def smallRow643_small : List Nat := smallRow643_small_left ++ smallRow643_small_right
theorem smallRow643_small_checked : upperNeighbors PrimePairVertices.tree_small 643 = smallRow643_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 643 ++ upperNeighbors PrimePairVertices.tree_small_right 643 = _
  rw [smallRow643_small_left_checked, smallRow643_small_right_checked]
  rfl

def smallRow643 : List Nat := smallRow643_small
theorem smallRow643_checked : upperNeighbors PrimePairVertices.smallTree 643 = smallRow643 := smallRow643_small_checked

def smallRow647_block_000 : List Nat := [821, 953, 1019, 1511, 1667, 1733, 1811, 1889]
set_option maxRecDepth 2048 in
theorem smallRow647_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 647 = smallRow647_block_000 := by decide

def smallRow647_block_001 : List Nat := [3011, 3347, 3407, 3911, 4019]
set_option maxRecDepth 2048 in
theorem smallRow647_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 647 = smallRow647_block_001 := by decide

def smallRow647_block_002 : List Nat := [4241, 4493, 5153, 5387, 5471, 5879]
set_option maxRecDepth 2048 in
theorem smallRow647_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 647 = smallRow647_block_002 := by decide

def smallRow647_block_003 : List Nat := [6329, 7253, 7949]
set_option maxRecDepth 2048 in
theorem smallRow647_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 647 = smallRow647_block_003 := by decide

def smallRow647_block_004 : List Nat := [8273]
set_option maxRecDepth 2048 in
theorem smallRow647_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 647 = smallRow647_block_004 := by decide

def smallRow647_small_left : List Nat := smallRow647_block_000 ++ smallRow647_block_001
theorem smallRow647_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 647 = smallRow647_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 647 ++ upperNeighbors PrimePairVertices.tree_block_001 647 = _
  rw [smallRow647_block_000_checked, smallRow647_block_001_checked]
  rfl

def smallRow647_small_right_right : List Nat := smallRow647_block_003 ++ smallRow647_block_004
theorem smallRow647_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 647 = smallRow647_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 647 ++ upperNeighbors PrimePairVertices.tree_block_004 647 = _
  rw [smallRow647_block_003_checked, smallRow647_block_004_checked]
  rfl

def smallRow647_small_right : List Nat := smallRow647_block_002 ++ smallRow647_small_right_right
theorem smallRow647_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 647 = smallRow647_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 647 ++ upperNeighbors PrimePairVertices.tree_small_right_right 647 = _
  rw [smallRow647_block_002_checked, smallRow647_small_right_right_checked]
  rfl

def smallRow647_small : List Nat := smallRow647_small_left ++ smallRow647_small_right
theorem smallRow647_small_checked : upperNeighbors PrimePairVertices.tree_small 647 = smallRow647_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 647 ++ upperNeighbors PrimePairVertices.tree_small_right 647 = _
  rw [smallRow647_small_left_checked, smallRow647_small_right_checked]
  rfl

def smallRow647 : List Nat := smallRow647_small
theorem smallRow647_checked : upperNeighbors PrimePairVertices.smallTree 647 = smallRow647 := smallRow647_small_checked

def smallRow653_block_000 : List Nat := [659, 941, 1097, 1289, 1523, 2003]
set_option maxRecDepth 2048 in
theorem smallRow653_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 653 = smallRow653_block_000 := by decide

def smallRow653_block_001 : List Nat := [2399, 2531, 2693, 3167, 3209, 3251, 3347, 3911]
set_option maxRecDepth 2048 in
theorem smallRow653_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 653 = smallRow653_block_001 := by decide

def smallRow653_block_002 : List Nat := [4289, 5387, 5897]
set_option maxRecDepth 2048 in
theorem smallRow653_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 653 = smallRow653_block_002 := by decide

def smallRow653_block_003 : List Nat := [6863, 6947, 7109, 7211, 7757]
set_option maxRecDepth 2048 in
theorem smallRow653_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 653 = smallRow653_block_003 := by decide

def smallRow653_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow653_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 653 = smallRow653_block_004 := by decide

def smallRow653_small_left : List Nat := smallRow653_block_000 ++ smallRow653_block_001
theorem smallRow653_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 653 = smallRow653_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 653 ++ upperNeighbors PrimePairVertices.tree_block_001 653 = _
  rw [smallRow653_block_000_checked, smallRow653_block_001_checked]
  rfl

def smallRow653_small_right_right : List Nat := smallRow653_block_003 ++ smallRow653_block_004
theorem smallRow653_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 653 = smallRow653_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 653 ++ upperNeighbors PrimePairVertices.tree_block_004 653 = _
  rw [smallRow653_block_003_checked, smallRow653_block_004_checked]
  rfl

def smallRow653_small_right : List Nat := smallRow653_block_002 ++ smallRow653_small_right_right
theorem smallRow653_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 653 = smallRow653_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 653 ++ upperNeighbors PrimePairVertices.tree_small_right_right 653 = _
  rw [smallRow653_block_002_checked, smallRow653_small_right_right_checked]
  rfl

def smallRow653_small : List Nat := smallRow653_small_left ++ smallRow653_small_right
theorem smallRow653_small_checked : upperNeighbors PrimePairVertices.tree_small 653 = smallRow653_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 653 ++ upperNeighbors PrimePairVertices.tree_small_right 653 = _
  rw [smallRow653_small_left_checked, smallRow653_small_right_checked]
  rfl

def smallRow653 : List Nat := smallRow653_small
theorem smallRow653_checked : upperNeighbors PrimePairVertices.smallTree 653 = smallRow653 := smallRow653_small_checked

def smallRow659_block_000 : List Nat := [947, 983, 1493]
set_option maxRecDepth 2048 in
theorem smallRow659_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 659 = smallRow659_block_000 := by decide

def smallRow659_block_001 : List Nat := [2393, 2741, 3023, 3329, 3371, 3407, 3593]
set_option maxRecDepth 2048 in
theorem smallRow659_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 659 = smallRow659_block_001 := by decide

def smallRow659_block_002 : List Nat := [4127, 4139, 4799, 5009, 6089]
set_option maxRecDepth 2048 in
theorem smallRow659_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 659 = smallRow659_block_002 := by decide

def smallRow659_block_003 : List Nat := [7307, 7517, 7583, 8087]
set_option maxRecDepth 2048 in
theorem smallRow659_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 659 = smallRow659_block_003 := by decide

def smallRow659_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow659_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 659 = smallRow659_block_004 := by decide

def smallRow659_small_left : List Nat := smallRow659_block_000 ++ smallRow659_block_001
theorem smallRow659_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 659 = smallRow659_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 659 ++ upperNeighbors PrimePairVertices.tree_block_001 659 = _
  rw [smallRow659_block_000_checked, smallRow659_block_001_checked]
  rfl

def smallRow659_small_right_right : List Nat := smallRow659_block_003 ++ smallRow659_block_004
theorem smallRow659_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 659 = smallRow659_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 659 ++ upperNeighbors PrimePairVertices.tree_block_004 659 = _
  rw [smallRow659_block_003_checked, smallRow659_block_004_checked]
  rfl

def smallRow659_small_right : List Nat := smallRow659_block_002 ++ smallRow659_small_right_right
theorem smallRow659_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 659 = smallRow659_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 659 ++ upperNeighbors PrimePairVertices.tree_small_right_right 659 = _
  rw [smallRow659_block_002_checked, smallRow659_small_right_right_checked]
  rfl

def smallRow659_small : List Nat := smallRow659_small_left ++ smallRow659_small_right
theorem smallRow659_small_checked : upperNeighbors PrimePairVertices.tree_small 659 = smallRow659_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 659 ++ upperNeighbors PrimePairVertices.tree_small_right 659 = _
  rw [smallRow659_small_left_checked, smallRow659_small_right_checked]
  rfl

def smallRow659 : List Nat := smallRow659_small
theorem smallRow659_checked : upperNeighbors PrimePairVertices.smallTree 659 = smallRow659 := smallRow659_small_checked

def smallRow661_block_000 : List Nat := [769, 877, 883, 1237, 1447, 1489, 1777, 1933, 1993]
set_option maxRecDepth 2048 in
theorem smallRow661_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 661 = smallRow661_block_000 := by decide

def smallRow661_block_001 : List Nat := [2539, 2791, 2857, 2953, 3331, 3361, 3823, 3931]
set_option maxRecDepth 2048 in
theorem smallRow661_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 661 = smallRow661_block_001 := by decide

def smallRow661_block_002 : List Nat := [4759, 5503, 5641]
set_option maxRecDepth 2048 in
theorem smallRow661_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 661 = smallRow661_block_002 := by decide

def smallRow661_block_003 : List Nat := [6151, 6529, 6781, 6871, 7159, 7873]
set_option maxRecDepth 2048 in
theorem smallRow661_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 661 = smallRow661_block_003 := by decide

def smallRow661_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow661_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 661 = smallRow661_block_004 := by decide

def smallRow661_small_left : List Nat := smallRow661_block_000 ++ smallRow661_block_001
theorem smallRow661_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 661 = smallRow661_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 661 ++ upperNeighbors PrimePairVertices.tree_block_001 661 = _
  rw [smallRow661_block_000_checked, smallRow661_block_001_checked]
  rfl

def smallRow661_small_right_right : List Nat := smallRow661_block_003 ++ smallRow661_block_004
theorem smallRow661_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 661 = smallRow661_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 661 ++ upperNeighbors PrimePairVertices.tree_block_004 661 = _
  rw [smallRow661_block_003_checked, smallRow661_block_004_checked]
  rfl

def smallRow661_small_right : List Nat := smallRow661_block_002 ++ smallRow661_small_right_right
theorem smallRow661_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 661 = smallRow661_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 661 ++ upperNeighbors PrimePairVertices.tree_small_right_right 661 = _
  rw [smallRow661_block_002_checked, smallRow661_small_right_right_checked]
  rfl

def smallRow661_small : List Nat := smallRow661_small_left ++ smallRow661_small_right
theorem smallRow661_small_checked : upperNeighbors PrimePairVertices.tree_small 661 = smallRow661_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 661 ++ upperNeighbors PrimePairVertices.tree_small_right 661 = _
  rw [smallRow661_small_left_checked, smallRow661_small_right_checked]
  rfl

def smallRow661 : List Nat := smallRow661_small
theorem smallRow661_checked : upperNeighbors PrimePairVertices.smallTree 661 = smallRow661 := smallRow661_small_checked

def smallRow673_block_000 : List Nat := [769, 1117, 1279, 1327, 1489, 1579, 1657, 1759]
set_option maxRecDepth 2048 in
theorem smallRow673_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 673 = smallRow673_block_000 := by decide

def smallRow673_block_001 : List Nat := [2281, 2293, 2659, 3271]
set_option maxRecDepth 2048 in
theorem smallRow673_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 673 = smallRow673_block_001 := by decide

def smallRow673_block_002 : List Nat := [4297, 4597, 4813, 4993]
set_option maxRecDepth 2048 in
theorem smallRow673_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 673 = smallRow673_block_002 := by decide

def smallRow673_block_003 : List Nat := [6271, 6841, 7177, 7417, 8089]
set_option maxRecDepth 2048 in
theorem smallRow673_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 673 = smallRow673_block_003 := by decide

def smallRow673_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow673_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 673 = smallRow673_block_004 := by decide

def smallRow673_small_left : List Nat := smallRow673_block_000 ++ smallRow673_block_001
theorem smallRow673_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 673 = smallRow673_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 673 ++ upperNeighbors PrimePairVertices.tree_block_001 673 = _
  rw [smallRow673_block_000_checked, smallRow673_block_001_checked]
  rfl

def smallRow673_small_right_right : List Nat := smallRow673_block_003 ++ smallRow673_block_004
theorem smallRow673_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 673 = smallRow673_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 673 ++ upperNeighbors PrimePairVertices.tree_block_004 673 = _
  rw [smallRow673_block_003_checked, smallRow673_block_004_checked]
  rfl

def smallRow673_small_right : List Nat := smallRow673_block_002 ++ smallRow673_small_right_right
theorem smallRow673_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 673 = smallRow673_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 673 ++ upperNeighbors PrimePairVertices.tree_small_right_right 673 = _
  rw [smallRow673_block_002_checked, smallRow673_small_right_right_checked]
  rfl

def smallRow673_small : List Nat := smallRow673_small_left ++ smallRow673_small_right
theorem smallRow673_small_checked : upperNeighbors PrimePairVertices.tree_small 673 = smallRow673_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 673 ++ upperNeighbors PrimePairVertices.tree_small_right 673 = _
  rw [smallRow673_small_left_checked, smallRow673_small_right_checked]
  rfl

def smallRow673 : List Nat := smallRow673_small
theorem smallRow673_checked : upperNeighbors PrimePairVertices.smallTree 673 = smallRow673 := smallRow673_small_checked

end ProjectEulerSolutions.PrimePairCache
