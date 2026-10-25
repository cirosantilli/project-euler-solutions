import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow677_block_000 : List Nat := [827, 1217, 1283, 1409, 1451, 1493, 1871]
set_option maxRecDepth 2048 in
theorem smallRow677_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 677 = smallRow677_block_000 := by decide

def smallRow677_block_001 : List Nat := [2309, 2687, 2837, 3203, 3863, 4073]
set_option maxRecDepth 2048 in
theorem smallRow677_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 677 = smallRow677_block_001 := by decide

def smallRow677_block_002 : List Nat := [4409, 4451, 5273, 5387, 5393, 5399, 5519, 5807, 5861, 5987, 6101]
set_option maxRecDepth 2048 in
theorem smallRow677_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 677 = smallRow677_block_002 := by decide

def smallRow677_block_003 : List Nat := [6173, 6257, 6491, 6971, 7349, 7451, 7517, 7949, 8039, 8147]
set_option maxRecDepth 2048 in
theorem smallRow677_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 677 = smallRow677_block_003 := by decide

def smallRow677_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow677_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 677 = smallRow677_block_004 := by decide

def smallRow677_small_left : List Nat := smallRow677_block_000 ++ smallRow677_block_001
theorem smallRow677_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 677 = smallRow677_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 677 ++ upperNeighbors PrimePairVertices.tree_block_001 677 = _
  rw [smallRow677_block_000_checked, smallRow677_block_001_checked]
  rfl

def smallRow677_small_right_right : List Nat := smallRow677_block_003 ++ smallRow677_block_004
theorem smallRow677_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 677 = smallRow677_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 677 ++ upperNeighbors PrimePairVertices.tree_block_004 677 = _
  rw [smallRow677_block_003_checked, smallRow677_block_004_checked]
  rfl

def smallRow677_small_right : List Nat := smallRow677_block_002 ++ smallRow677_small_right_right
theorem smallRow677_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 677 = smallRow677_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 677 ++ upperNeighbors PrimePairVertices.tree_small_right_right 677 = _
  rw [smallRow677_block_002_checked, smallRow677_small_right_right_checked]
  rfl

def smallRow677_small : List Nat := smallRow677_small_left ++ smallRow677_small_right
theorem smallRow677_small_checked : upperNeighbors PrimePairVertices.tree_small 677 = smallRow677_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 677 ++ upperNeighbors PrimePairVertices.tree_small_right 677 = _
  rw [smallRow677_small_left_checked, smallRow677_small_right_checked]
  rfl

def smallRow677 : List Nat := smallRow677_small
theorem smallRow677_checked : upperNeighbors PrimePairVertices.smallTree 677 = smallRow677 := smallRow677_small_checked

def smallRow683_block_000 : List Nat := [719, 911, 1499]
set_option maxRecDepth 2048 in
theorem smallRow683_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 683 = smallRow683_block_000 := by decide

def smallRow683_block_001 : List Nat := [2417, 2843, 2939, 2957, 3083, 3209, 3797, 3929, 3989, 4073]
set_option maxRecDepth 2048 in
theorem smallRow683_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 683 = smallRow683_block_001 := by decide

def smallRow683_block_002 : List Nat := [4637, 4649, 4679, 4937, 5657, 6047, 6131]
set_option maxRecDepth 2048 in
theorem smallRow683_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 683 = smallRow683_block_002 := by decide

def smallRow683_block_003 : List Nat := [6317, 6653, 6659, 7499, 7643]
set_option maxRecDepth 2048 in
theorem smallRow683_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 683 = smallRow683_block_003 := by decide

def smallRow683_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow683_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 683 = smallRow683_block_004 := by decide

def smallRow683_small_left : List Nat := smallRow683_block_000 ++ smallRow683_block_001
theorem smallRow683_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 683 = smallRow683_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 683 ++ upperNeighbors PrimePairVertices.tree_block_001 683 = _
  rw [smallRow683_block_000_checked, smallRow683_block_001_checked]
  rfl

def smallRow683_small_right_right : List Nat := smallRow683_block_003 ++ smallRow683_block_004
theorem smallRow683_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 683 = smallRow683_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 683 ++ upperNeighbors PrimePairVertices.tree_block_004 683 = _
  rw [smallRow683_block_003_checked, smallRow683_block_004_checked]
  rfl

def smallRow683_small_right : List Nat := smallRow683_block_002 ++ smallRow683_small_right_right
theorem smallRow683_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 683 = smallRow683_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 683 ++ upperNeighbors PrimePairVertices.tree_small_right_right 683 = _
  rw [smallRow683_block_002_checked, smallRow683_small_right_right_checked]
  rfl

def smallRow683_small : List Nat := smallRow683_small_left ++ smallRow683_small_right
theorem smallRow683_small_checked : upperNeighbors PrimePairVertices.tree_small 683 = smallRow683_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 683 ++ upperNeighbors PrimePairVertices.tree_small_right 683 = _
  rw [smallRow683_small_left_checked, smallRow683_small_right_checked]
  rfl

def smallRow683 : List Nat := smallRow683_small
theorem smallRow683_checked : upperNeighbors PrimePairVertices.smallTree 683 = smallRow683 := smallRow683_small_checked

def smallRow691_block_000 : List Nat := [709, 727, 907, 919, 1249, 1291, 1423, 1693]
set_option maxRecDepth 2048 in
theorem smallRow691_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 691 = smallRow691_block_000 := by decide

def smallRow691_block_001 : List Nat := [2131, 2593, 3121, 3229, 3433, 3469, 3541, 3607, 3943]
set_option maxRecDepth 2048 in
theorem smallRow691_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 691 = smallRow691_block_001 := by decide

def smallRow691_block_002 : List Nat := [4603, 4657, 4729, 4861, 4951, 5197, 5431, 5743, 5827, 5851, 5869]
set_option maxRecDepth 2048 in
theorem smallRow691_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 691 = smallRow691_block_002 := by decide

def smallRow691_block_003 : List Nat := [6199, 6397, 6451, 6673, 7177]
set_option maxRecDepth 2048 in
theorem smallRow691_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 691 = smallRow691_block_003 := by decide

def smallRow691_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow691_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 691 = smallRow691_block_004 := by decide

def smallRow691_small_left : List Nat := smallRow691_block_000 ++ smallRow691_block_001
theorem smallRow691_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 691 = smallRow691_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 691 ++ upperNeighbors PrimePairVertices.tree_block_001 691 = _
  rw [smallRow691_block_000_checked, smallRow691_block_001_checked]
  rfl

def smallRow691_small_right_right : List Nat := smallRow691_block_003 ++ smallRow691_block_004
theorem smallRow691_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 691 = smallRow691_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 691 ++ upperNeighbors PrimePairVertices.tree_block_004 691 = _
  rw [smallRow691_block_003_checked, smallRow691_block_004_checked]
  rfl

def smallRow691_small_right : List Nat := smallRow691_block_002 ++ smallRow691_small_right_right
theorem smallRow691_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 691 = smallRow691_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 691 ++ upperNeighbors PrimePairVertices.tree_small_right_right 691 = _
  rw [smallRow691_block_002_checked, smallRow691_small_right_right_checked]
  rfl

def smallRow691_small : List Nat := smallRow691_small_left ++ smallRow691_small_right
theorem smallRow691_small_checked : upperNeighbors PrimePairVertices.tree_small 691 = smallRow691_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 691 ++ upperNeighbors PrimePairVertices.tree_small_right 691 = _
  rw [smallRow691_small_left_checked, smallRow691_small_right_checked]
  rfl

def smallRow691 : List Nat := smallRow691_small
theorem smallRow691_checked : upperNeighbors PrimePairVertices.smallTree 691 = smallRow691 := smallRow691_small_checked

def smallRow701_block_000 : List Nat := [1181, 1187, 1283, 1307, 1439, 1553, 1607, 1733, 1913]
set_option maxRecDepth 2048 in
theorem smallRow701_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 701 = smallRow701_block_000 := by decide

def smallRow701_block_001 : List Nat := [2267, 2477, 3581, 3623]
set_option maxRecDepth 2048 in
theorem smallRow701_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 701 = smallRow701_block_001 := by decide

def smallRow701_block_002 : List Nat := [4283, 4373, 4583, 4751, 4877, 5003, 5087, 5417, 5471, 5501, 6101]
set_option maxRecDepth 2048 in
theorem smallRow701_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 701 = smallRow701_block_002 := by decide

def smallRow701_block_003 : List Nat := [6389, 6803, 7541, 7607, 7919, 7937, 8171]
set_option maxRecDepth 2048 in
theorem smallRow701_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 701 = smallRow701_block_003 := by decide

def smallRow701_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow701_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 701 = smallRow701_block_004 := by decide

def smallRow701_small_left : List Nat := smallRow701_block_000 ++ smallRow701_block_001
theorem smallRow701_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 701 = smallRow701_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 701 ++ upperNeighbors PrimePairVertices.tree_block_001 701 = _
  rw [smallRow701_block_000_checked, smallRow701_block_001_checked]
  rfl

def smallRow701_small_right_right : List Nat := smallRow701_block_003 ++ smallRow701_block_004
theorem smallRow701_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 701 = smallRow701_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 701 ++ upperNeighbors PrimePairVertices.tree_block_004 701 = _
  rw [smallRow701_block_003_checked, smallRow701_block_004_checked]
  rfl

def smallRow701_small_right : List Nat := smallRow701_block_002 ++ smallRow701_small_right_right
theorem smallRow701_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 701 = smallRow701_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 701 ++ upperNeighbors PrimePairVertices.tree_small_right_right 701 = _
  rw [smallRow701_block_002_checked, smallRow701_small_right_right_checked]
  rfl

def smallRow701_small : List Nat := smallRow701_small_left ++ smallRow701_small_right
theorem smallRow701_small_checked : upperNeighbors PrimePairVertices.tree_small 701 = smallRow701_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 701 ++ upperNeighbors PrimePairVertices.tree_small_right 701 = _
  rw [smallRow701_small_left_checked, smallRow701_small_right_checked]
  rfl

def smallRow701 : List Nat := smallRow701_small
theorem smallRow701_checked : upperNeighbors PrimePairVertices.smallTree 701 = smallRow701 := smallRow701_small_checked

def smallRow709_block_000 : List Nat := [823, 967, 1471, 1543, 1663, 1933, 1951]
set_option maxRecDepth 2048 in
theorem smallRow709_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 709 = smallRow709_block_000 := by decide

def smallRow709_block_001 : List Nat := [2647, 2731, 2749, 3067, 3109, 3181, 3469, 3769]
set_option maxRecDepth 2048 in
theorem smallRow709_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 709 = smallRow709_block_001 := by decide

def smallRow709_block_002 : List Nat := [4177, 4519, 4597, 5521, 6079]
set_option maxRecDepth 2048 in
theorem smallRow709_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 709 = smallRow709_block_002 := by decide

def smallRow709_block_003 : List Nat := [7213, 7459]
set_option maxRecDepth 2048 in
theorem smallRow709_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 709 = smallRow709_block_003 := by decide

def smallRow709_block_004 : List Nat := [8209, 8263]
set_option maxRecDepth 2048 in
theorem smallRow709_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 709 = smallRow709_block_004 := by decide

def smallRow709_small_left : List Nat := smallRow709_block_000 ++ smallRow709_block_001
theorem smallRow709_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 709 = smallRow709_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 709 ++ upperNeighbors PrimePairVertices.tree_block_001 709 = _
  rw [smallRow709_block_000_checked, smallRow709_block_001_checked]
  rfl

def smallRow709_small_right_right : List Nat := smallRow709_block_003 ++ smallRow709_block_004
theorem smallRow709_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 709 = smallRow709_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 709 ++ upperNeighbors PrimePairVertices.tree_block_004 709 = _
  rw [smallRow709_block_003_checked, smallRow709_block_004_checked]
  rfl

def smallRow709_small_right : List Nat := smallRow709_block_002 ++ smallRow709_small_right_right
theorem smallRow709_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 709 = smallRow709_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 709 ++ upperNeighbors PrimePairVertices.tree_small_right_right 709 = _
  rw [smallRow709_block_002_checked, smallRow709_small_right_right_checked]
  rfl

def smallRow709_small : List Nat := smallRow709_small_left ++ smallRow709_small_right
theorem smallRow709_small_checked : upperNeighbors PrimePairVertices.tree_small 709 = smallRow709_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 709 ++ upperNeighbors PrimePairVertices.tree_small_right 709 = _
  rw [smallRow709_small_left_checked, smallRow709_small_right_checked]
  rfl

def smallRow709 : List Nat := smallRow709_small
theorem smallRow709_checked : upperNeighbors PrimePairVertices.smallTree 709 = smallRow709 := smallRow709_small_checked

def smallRow719_block_000 : List Nat := [773, 911, 947, 1319, 1439, 1997, 2027]
set_option maxRecDepth 2048 in
theorem smallRow719_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 719 = smallRow719_block_000 := by decide

def smallRow719_block_001 : List Nat := [2333, 2543, 2579, 3257, 3833]
set_option maxRecDepth 2048 in
theorem smallRow719_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 719 = smallRow719_block_001 := by decide

def smallRow719_block_002 : List Nat := [4799, 4973, 5039, 5261, 5387, 5807]
set_option maxRecDepth 2048 in
theorem smallRow719_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 719 = smallRow719_block_002 := by decide

def smallRow719_block_003 : List Nat := [6317, 6323, 6869, 7079, 7283, 7433, 7793, 7937]
set_option maxRecDepth 2048 in
theorem smallRow719_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 719 = smallRow719_block_003 := by decide

def smallRow719_block_004 : List Nat := [8297]
set_option maxRecDepth 2048 in
theorem smallRow719_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 719 = smallRow719_block_004 := by decide

def smallRow719_small_left : List Nat := smallRow719_block_000 ++ smallRow719_block_001
theorem smallRow719_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 719 = smallRow719_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 719 ++ upperNeighbors PrimePairVertices.tree_block_001 719 = _
  rw [smallRow719_block_000_checked, smallRow719_block_001_checked]
  rfl

def smallRow719_small_right_right : List Nat := smallRow719_block_003 ++ smallRow719_block_004
theorem smallRow719_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 719 = smallRow719_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 719 ++ upperNeighbors PrimePairVertices.tree_block_004 719 = _
  rw [smallRow719_block_003_checked, smallRow719_block_004_checked]
  rfl

def smallRow719_small_right : List Nat := smallRow719_block_002 ++ smallRow719_small_right_right
theorem smallRow719_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 719 = smallRow719_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 719 ++ upperNeighbors PrimePairVertices.tree_small_right_right 719 = _
  rw [smallRow719_block_002_checked, smallRow719_small_right_right_checked]
  rfl

def smallRow719_small : List Nat := smallRow719_small_left ++ smallRow719_small_right
theorem smallRow719_small_checked : upperNeighbors PrimePairVertices.tree_small 719 = smallRow719_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 719 ++ upperNeighbors PrimePairVertices.tree_small_right 719 = _
  rw [smallRow719_small_left_checked, smallRow719_small_right_checked]
  rfl

def smallRow719 : List Nat := smallRow719_small
theorem smallRow719_checked : upperNeighbors PrimePairVertices.smallTree 719 = smallRow719 := smallRow719_small_checked

def smallRow727_block_000 : List Nat := [823, 997, 1009, 1237, 1279, 1669, 1723, 1867]
set_option maxRecDepth 2048 in
theorem smallRow727_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 727 = smallRow727_block_000 := by decide

def smallRow727_block_001 : List Nat := [2137, 2689, 2803, 3001, 3463]
set_option maxRecDepth 2048 in
theorem smallRow727_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 727 = smallRow727_block_001 := by decide

def smallRow727_block_002 : List Nat := [4327, 5197, 5209, 5227, 5503, 5623, 5857, 5923, 6121]
set_option maxRecDepth 2048 in
theorem smallRow727_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 727 = smallRow727_block_002 := by decide

def smallRow727_block_003 : List Nat := [6277, 6427, 6637, 7639]
set_option maxRecDepth 2048 in
theorem smallRow727_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 727 = smallRow727_block_003 := by decide

def smallRow727_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow727_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 727 = smallRow727_block_004 := by decide

def smallRow727_small_left : List Nat := smallRow727_block_000 ++ smallRow727_block_001
theorem smallRow727_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 727 = smallRow727_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 727 ++ upperNeighbors PrimePairVertices.tree_block_001 727 = _
  rw [smallRow727_block_000_checked, smallRow727_block_001_checked]
  rfl

def smallRow727_small_right_right : List Nat := smallRow727_block_003 ++ smallRow727_block_004
theorem smallRow727_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 727 = smallRow727_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 727 ++ upperNeighbors PrimePairVertices.tree_block_004 727 = _
  rw [smallRow727_block_003_checked, smallRow727_block_004_checked]
  rfl

def smallRow727_small_right : List Nat := smallRow727_block_002 ++ smallRow727_small_right_right
theorem smallRow727_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 727 = smallRow727_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 727 ++ upperNeighbors PrimePairVertices.tree_small_right_right 727 = _
  rw [smallRow727_block_002_checked, smallRow727_small_right_right_checked]
  rfl

def smallRow727_small : List Nat := smallRow727_small_left ++ smallRow727_small_right
theorem smallRow727_small_checked : upperNeighbors PrimePairVertices.tree_small 727 = smallRow727_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 727 ++ upperNeighbors PrimePairVertices.tree_small_right 727 = _
  rw [smallRow727_small_left_checked, smallRow727_small_right_checked]
  rfl

def smallRow727 : List Nat := smallRow727_small
theorem smallRow727_checked : upperNeighbors PrimePairVertices.smallTree 727 = smallRow727 := smallRow727_small_checked

def smallRow733_block_000 : List Nat := [829, 883, 991, 1039, 1249, 1999]
set_option maxRecDepth 2048 in
theorem smallRow733_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 733 = smallRow733_block_000 := by decide

def smallRow733_block_001 : List Nat := [2113, 2551, 3331, 3709, 4027]
set_option maxRecDepth 2048 in
theorem smallRow733_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 733 = smallRow733_block_001 := by decide

def smallRow733_block_002 : List Nat := [4357, 4441, 4549, 4903, 5701, 5743, 5851]
set_option maxRecDepth 2048 in
theorem smallRow733_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 733 = smallRow733_block_002 := by decide

def smallRow733_block_003 : List Nat := [7717, 7723, 7879]
set_option maxRecDepth 2048 in
theorem smallRow733_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 733 = smallRow733_block_003 := by decide

def smallRow733_block_004 : List Nat := [8263]
set_option maxRecDepth 2048 in
theorem smallRow733_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 733 = smallRow733_block_004 := by decide

def smallRow733_small_left : List Nat := smallRow733_block_000 ++ smallRow733_block_001
theorem smallRow733_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 733 = smallRow733_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 733 ++ upperNeighbors PrimePairVertices.tree_block_001 733 = _
  rw [smallRow733_block_000_checked, smallRow733_block_001_checked]
  rfl

def smallRow733_small_right_right : List Nat := smallRow733_block_003 ++ smallRow733_block_004
theorem smallRow733_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 733 = smallRow733_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 733 ++ upperNeighbors PrimePairVertices.tree_block_004 733 = _
  rw [smallRow733_block_003_checked, smallRow733_block_004_checked]
  rfl

def smallRow733_small_right : List Nat := smallRow733_block_002 ++ smallRow733_small_right_right
theorem smallRow733_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 733 = smallRow733_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 733 ++ upperNeighbors PrimePairVertices.tree_small_right_right 733 = _
  rw [smallRow733_block_002_checked, smallRow733_small_right_right_checked]
  rfl

def smallRow733_small : List Nat := smallRow733_small_left ++ smallRow733_small_right
theorem smallRow733_small_checked : upperNeighbors PrimePairVertices.tree_small 733 = smallRow733_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 733 ++ upperNeighbors PrimePairVertices.tree_small_right 733 = _
  rw [smallRow733_small_left_checked, smallRow733_small_right_checked]
  rfl

def smallRow733 : List Nat := smallRow733_small
theorem smallRow733_checked : upperNeighbors PrimePairVertices.smallTree 733 = smallRow733 := smallRow733_small_checked

end ProjectEulerSolutions.PrimePairCache
