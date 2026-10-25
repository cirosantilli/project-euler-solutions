import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow797_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow797_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 797 = smallRow797_block_000 := by decide

def smallRow797_block_001 : List Nat := [2297, 2339, 2411, 2417, 2621, 2969, 3209, 3299, 3461, 3659, 3929]
set_option maxRecDepth 2048 in
theorem smallRow797_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 797 = smallRow797_block_001 := by decide

def smallRow797_block_002 : List Nat := [4259, 4481, 4937, 5147, 5189, 5519, 5531]
set_option maxRecDepth 2048 in
theorem smallRow797_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 797 = smallRow797_block_002 := by decide

def smallRow797_block_003 : List Nat := [6389, 6833, 6911, 7457, 7649, 7841]
set_option maxRecDepth 2048 in
theorem smallRow797_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 797 = smallRow797_block_003 := by decide

def smallRow797_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow797_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 797 = smallRow797_block_004 := by decide

def smallRow797_small_left : List Nat := smallRow797_block_000 ++ smallRow797_block_001
theorem smallRow797_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 797 = smallRow797_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 797 ++ upperNeighbors PrimePairVertices.tree_block_001 797 = _
  rw [smallRow797_block_000_checked, smallRow797_block_001_checked]
  rfl

def smallRow797_small_right_right : List Nat := smallRow797_block_003 ++ smallRow797_block_004
theorem smallRow797_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 797 = smallRow797_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 797 ++ upperNeighbors PrimePairVertices.tree_block_004 797 = _
  rw [smallRow797_block_003_checked, smallRow797_block_004_checked]
  rfl

def smallRow797_small_right : List Nat := smallRow797_block_002 ++ smallRow797_small_right_right
theorem smallRow797_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 797 = smallRow797_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 797 ++ upperNeighbors PrimePairVertices.tree_small_right_right 797 = _
  rw [smallRow797_block_002_checked, smallRow797_small_right_right_checked]
  rfl

def smallRow797_small : List Nat := smallRow797_small_left ++ smallRow797_small_right
theorem smallRow797_small_checked : upperNeighbors PrimePairVertices.tree_small 797 = smallRow797_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 797 ++ upperNeighbors PrimePairVertices.tree_small_right 797 = _
  rw [smallRow797_small_left_checked, smallRow797_small_right_checked]
  rfl

def smallRow797 : List Nat := smallRow797_small
theorem smallRow797_checked : upperNeighbors PrimePairVertices.smallTree 797 = smallRow797 := smallRow797_small_checked

def smallRow809_block_000 : List Nat := [821, 827, 839, 929, 983, 1091, 1361, 1427, 1553, 1583]
set_option maxRecDepth 2048 in
theorem smallRow809_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 809 = smallRow809_block_000 := by decide

def smallRow809_block_001 : List Nat := [2141, 2243, 2927, 2963, 3011, 3209, 3947]
set_option maxRecDepth 2048 in
theorem smallRow809_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 809 = smallRow809_block_001 := by decide

def smallRow809_block_002 : List Nat := [4271, 4679, 5237, 5393, 5531]
set_option maxRecDepth 2048 in
theorem smallRow809_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 809 = smallRow809_block_002 := by decide

def smallRow809_block_003 : List Nat := [6719]
set_option maxRecDepth 2048 in
theorem smallRow809_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 809 = smallRow809_block_003 := by decide

def smallRow809_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow809_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 809 = smallRow809_block_004 := by decide

def smallRow809_small_left : List Nat := smallRow809_block_000 ++ smallRow809_block_001
theorem smallRow809_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 809 = smallRow809_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 809 ++ upperNeighbors PrimePairVertices.tree_block_001 809 = _
  rw [smallRow809_block_000_checked, smallRow809_block_001_checked]
  rfl

def smallRow809_small_right_right : List Nat := smallRow809_block_003 ++ smallRow809_block_004
theorem smallRow809_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 809 = smallRow809_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 809 ++ upperNeighbors PrimePairVertices.tree_block_004 809 = _
  rw [smallRow809_block_003_checked, smallRow809_block_004_checked]
  rfl

def smallRow809_small_right : List Nat := smallRow809_block_002 ++ smallRow809_small_right_right
theorem smallRow809_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 809 = smallRow809_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 809 ++ upperNeighbors PrimePairVertices.tree_small_right_right 809 = _
  rw [smallRow809_block_002_checked, smallRow809_small_right_right_checked]
  rfl

def smallRow809_small : List Nat := smallRow809_small_left ++ smallRow809_small_right
theorem smallRow809_small_checked : upperNeighbors PrimePairVertices.tree_small 809 = smallRow809_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 809 ++ upperNeighbors PrimePairVertices.tree_small_right 809 = _
  rw [smallRow809_small_left_checked, smallRow809_small_right_checked]
  rfl

def smallRow809 : List Nat := smallRow809_small
theorem smallRow809_checked : upperNeighbors PrimePairVertices.smallTree 809 = smallRow809 := smallRow809_small_checked

def smallRow811_block_000 : List Nat := [919, 991, 997, 1171, 1447, 1459, 1531, 1657, 1723, 2011]
set_option maxRecDepth 2048 in
theorem smallRow811_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 811 = smallRow811_block_000 := by decide

def smallRow811_block_001 : List Nat := [3109, 3631, 3643, 3691, 3967]
set_option maxRecDepth 2048 in
theorem smallRow811_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 811 = smallRow811_block_001 := by decide

def smallRow811_block_002 : List Nat := [4201, 4861, 5119, 5449, 5791, 6133]
set_option maxRecDepth 2048 in
theorem smallRow811_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 811 = smallRow811_block_002 := by decide

def smallRow811_block_003 : List Nat := [6469, 6547, 6967, 7561, 8101, 8179, 8191]
set_option maxRecDepth 2048 in
theorem smallRow811_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 811 = smallRow811_block_003 := by decide

def smallRow811_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow811_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 811 = smallRow811_block_004 := by decide

def smallRow811_small_left : List Nat := smallRow811_block_000 ++ smallRow811_block_001
theorem smallRow811_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 811 = smallRow811_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 811 ++ upperNeighbors PrimePairVertices.tree_block_001 811 = _
  rw [smallRow811_block_000_checked, smallRow811_block_001_checked]
  rfl

def smallRow811_small_right_right : List Nat := smallRow811_block_003 ++ smallRow811_block_004
theorem smallRow811_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 811 = smallRow811_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 811 ++ upperNeighbors PrimePairVertices.tree_block_004 811 = _
  rw [smallRow811_block_003_checked, smallRow811_block_004_checked]
  rfl

def smallRow811_small_right : List Nat := smallRow811_block_002 ++ smallRow811_small_right_right
theorem smallRow811_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 811 = smallRow811_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 811 ++ upperNeighbors PrimePairVertices.tree_small_right_right 811 = _
  rw [smallRow811_block_002_checked, smallRow811_small_right_right_checked]
  rfl

def smallRow811_small : List Nat := smallRow811_small_left ++ smallRow811_small_right
theorem smallRow811_small_checked : upperNeighbors PrimePairVertices.tree_small 811 = smallRow811_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 811 ++ upperNeighbors PrimePairVertices.tree_small_right 811 = _
  rw [smallRow811_small_left_checked, smallRow811_small_right_checked]
  rfl

def smallRow811 : List Nat := smallRow811_small
theorem smallRow811_checked : upperNeighbors PrimePairVertices.smallTree 811 = smallRow811 := smallRow811_small_checked

def smallRow821_block_000 : List Nat := [827, 857, 971, 1109, 1163, 1433, 1607, 1949]
set_option maxRecDepth 2048 in
theorem smallRow821_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 821 = smallRow821_block_000 := by decide

def smallRow821_block_001 : List Nat := [2549, 2591, 2969, 3581]
set_option maxRecDepth 2048 in
theorem smallRow821_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 821 = smallRow821_block_001 := by decide

def smallRow821_block_002 : List Nat := [4691, 5783, 5843, 5849]
set_option maxRecDepth 2048 in
theorem smallRow821_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 821 = smallRow821_block_002 := by decide

def smallRow821_block_003 : List Nat := [6317, 6521, 6737, 7523]
set_option maxRecDepth 2048 in
theorem smallRow821_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 821 = smallRow821_block_003 := by decide

def smallRow821_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow821_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 821 = smallRow821_block_004 := by decide

def smallRow821_small_left : List Nat := smallRow821_block_000 ++ smallRow821_block_001
theorem smallRow821_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 821 = smallRow821_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 821 ++ upperNeighbors PrimePairVertices.tree_block_001 821 = _
  rw [smallRow821_block_000_checked, smallRow821_block_001_checked]
  rfl

def smallRow821_small_right_right : List Nat := smallRow821_block_003 ++ smallRow821_block_004
theorem smallRow821_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 821 = smallRow821_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 821 ++ upperNeighbors PrimePairVertices.tree_block_004 821 = _
  rw [smallRow821_block_003_checked, smallRow821_block_004_checked]
  rfl

def smallRow821_small_right : List Nat := smallRow821_block_002 ++ smallRow821_small_right_right
theorem smallRow821_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 821 = smallRow821_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 821 ++ upperNeighbors PrimePairVertices.tree_small_right_right 821 = _
  rw [smallRow821_block_002_checked, smallRow821_small_right_right_checked]
  rfl

def smallRow821_small : List Nat := smallRow821_small_left ++ smallRow821_small_right
theorem smallRow821_small_checked : upperNeighbors PrimePairVertices.tree_small 821 = smallRow821_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 821 ++ upperNeighbors PrimePairVertices.tree_small_right 821 = _
  rw [smallRow821_small_left_checked, smallRow821_small_right_checked]
  rfl

def smallRow821 : List Nat := smallRow821_small
theorem smallRow821_checked : upperNeighbors PrimePairVertices.smallTree 821 = smallRow821 := smallRow821_small_checked

def smallRow823_block_000 : List Nat := [877, 967, 1303, 1459, 1777, 1933, 2017]
set_option maxRecDepth 2048 in
theorem smallRow823_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 823 = smallRow823_block_000 := by decide

def smallRow823_block_001 : List Nat := [2113, 2179, 2281, 2311, 2347, 2557, 2953, 3541]
set_option maxRecDepth 2048 in
theorem smallRow823_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 823 = smallRow823_block_001 := by decide

def smallRow823_block_002 : List Nat := [4243, 4789, 4909, 4951, 5209, 5281, 5323, 5827, 5881]
set_option maxRecDepth 2048 in
theorem smallRow823_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 823 = smallRow823_block_002 := by decide

def smallRow823_block_003 : List Nat := [6337, 6733, 7237, 7699, 8101]
set_option maxRecDepth 2048 in
theorem smallRow823_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 823 = smallRow823_block_003 := by decide

def smallRow823_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow823_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 823 = smallRow823_block_004 := by decide

def smallRow823_small_left : List Nat := smallRow823_block_000 ++ smallRow823_block_001
theorem smallRow823_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 823 = smallRow823_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 823 ++ upperNeighbors PrimePairVertices.tree_block_001 823 = _
  rw [smallRow823_block_000_checked, smallRow823_block_001_checked]
  rfl

def smallRow823_small_right_right : List Nat := smallRow823_block_003 ++ smallRow823_block_004
theorem smallRow823_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 823 = smallRow823_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 823 ++ upperNeighbors PrimePairVertices.tree_block_004 823 = _
  rw [smallRow823_block_003_checked, smallRow823_block_004_checked]
  rfl

def smallRow823_small_right : List Nat := smallRow823_block_002 ++ smallRow823_small_right_right
theorem smallRow823_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 823 = smallRow823_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 823 ++ upperNeighbors PrimePairVertices.tree_small_right_right 823 = _
  rw [smallRow823_block_002_checked, smallRow823_small_right_right_checked]
  rfl

def smallRow823_small : List Nat := smallRow823_small_left ++ smallRow823_small_right
theorem smallRow823_small_checked : upperNeighbors PrimePairVertices.tree_small 823 = smallRow823_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 823 ++ upperNeighbors PrimePairVertices.tree_small_right 823 = _
  rw [smallRow823_small_left_checked, smallRow823_small_right_checked]
  rfl

def smallRow823 : List Nat := smallRow823_small
theorem smallRow823_checked : upperNeighbors PrimePairVertices.smallTree 823 = smallRow823 := smallRow823_small_checked

def smallRow827_block_000 : List Nat := [857, 1019, 1049, 1301, 1721, 1811, 1847, 1871]
set_option maxRecDepth 2048 in
theorem smallRow827_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 827 = smallRow827_block_000 := by decide

def smallRow827_block_001 : List Nat := [2267, 2273, 2351, 2711, 2819, 2909, 3299, 3527, 3593, 3719]
set_option maxRecDepth 2048 in
theorem smallRow827_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 827 = smallRow827_block_001 := by decide

def smallRow827_block_002 : List Nat := [4463, 5303, 5417, 5471, 5717, 6143]
set_option maxRecDepth 2048 in
theorem smallRow827_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 827 = smallRow827_block_002 := by decide

def smallRow827_block_003 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow827_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 827 = smallRow827_block_003 := by decide

def smallRow827_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow827_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 827 = smallRow827_block_004 := by decide

def smallRow827_small_left : List Nat := smallRow827_block_000 ++ smallRow827_block_001
theorem smallRow827_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 827 = smallRow827_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 827 ++ upperNeighbors PrimePairVertices.tree_block_001 827 = _
  rw [smallRow827_block_000_checked, smallRow827_block_001_checked]
  rfl

def smallRow827_small_right_right : List Nat := smallRow827_block_003 ++ smallRow827_block_004
theorem smallRow827_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 827 = smallRow827_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 827 ++ upperNeighbors PrimePairVertices.tree_block_004 827 = _
  rw [smallRow827_block_003_checked, smallRow827_block_004_checked]
  rfl

def smallRow827_small_right : List Nat := smallRow827_block_002 ++ smallRow827_small_right_right
theorem smallRow827_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 827 = smallRow827_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 827 ++ upperNeighbors PrimePairVertices.tree_small_right_right 827 = _
  rw [smallRow827_block_002_checked, smallRow827_small_right_right_checked]
  rfl

def smallRow827_small : List Nat := smallRow827_small_left ++ smallRow827_small_right
theorem smallRow827_small_checked : upperNeighbors PrimePairVertices.tree_small 827 = smallRow827_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 827 ++ upperNeighbors PrimePairVertices.tree_small_right 827 = _
  rw [smallRow827_small_left_checked, smallRow827_small_right_checked]
  rfl

def smallRow827 : List Nat := smallRow827_small
theorem smallRow827_checked : upperNeighbors PrimePairVertices.smallTree 827 = smallRow827 := smallRow827_small_checked

def smallRow829_block_000 : List Nat := [1033, 1237]
set_option maxRecDepth 2048 in
theorem smallRow829_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 829 = smallRow829_block_000 := by decide

def smallRow829_block_001 : List Nat := [2293, 2671, 3319, 3361, 3847]
set_option maxRecDepth 2048 in
theorem smallRow829_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 829 = smallRow829_block_001 := by decide

def smallRow829_block_002 : List Nat := [5107, 5527, 5563, 5839, 5857]
set_option maxRecDepth 2048 in
theorem smallRow829_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 829 = smallRow829_block_002 := by decide

def smallRow829_block_003 : List Nat := [6823, 7321, 7351, 7573]
set_option maxRecDepth 2048 in
theorem smallRow829_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 829 = smallRow829_block_003 := by decide

def smallRow829_block_004 : List Nat := [8209]
set_option maxRecDepth 2048 in
theorem smallRow829_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 829 = smallRow829_block_004 := by decide

def smallRow829_small_left : List Nat := smallRow829_block_000 ++ smallRow829_block_001
theorem smallRow829_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 829 = smallRow829_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 829 ++ upperNeighbors PrimePairVertices.tree_block_001 829 = _
  rw [smallRow829_block_000_checked, smallRow829_block_001_checked]
  rfl

def smallRow829_small_right_right : List Nat := smallRow829_block_003 ++ smallRow829_block_004
theorem smallRow829_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 829 = smallRow829_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 829 ++ upperNeighbors PrimePairVertices.tree_block_004 829 = _
  rw [smallRow829_block_003_checked, smallRow829_block_004_checked]
  rfl

def smallRow829_small_right : List Nat := smallRow829_block_002 ++ smallRow829_small_right_right
theorem smallRow829_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 829 = smallRow829_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 829 ++ upperNeighbors PrimePairVertices.tree_small_right_right 829 = _
  rw [smallRow829_block_002_checked, smallRow829_small_right_right_checked]
  rfl

def smallRow829_small : List Nat := smallRow829_small_left ++ smallRow829_small_right
theorem smallRow829_small_checked : upperNeighbors PrimePairVertices.tree_small 829 = smallRow829_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 829 ++ upperNeighbors PrimePairVertices.tree_small_right 829 = _
  rw [smallRow829_small_left_checked, smallRow829_small_right_checked]
  rfl

def smallRow829 : List Nat := smallRow829_small
theorem smallRow829_checked : upperNeighbors PrimePairVertices.smallTree 829 = smallRow829 := smallRow829_small_checked

def smallRow839_block_000 : List Nat := [887, 911, 1109, 1283, 1289, 1949]
set_option maxRecDepth 2048 in
theorem smallRow839_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 839 = smallRow839_block_000 := by decide

def smallRow839_block_001 : List Nat := [2063, 2333, 3251, 3533, 3923]
set_option maxRecDepth 2048 in
theorem smallRow839_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 839 = smallRow839_block_001 := by decide

def smallRow839_block_002 : List Nat := [4679, 4721, 5237, 5273, 5351, 5741]
set_option maxRecDepth 2048 in
theorem smallRow839_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 839 = smallRow839_block_002 := by decide

def smallRow839_block_003 : List Nat := [6263, 6599, 7121, 7283, 7331, 7457, 7841, 7877]
set_option maxRecDepth 2048 in
theorem smallRow839_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 839 = smallRow839_block_003 := by decide

def smallRow839_block_004 : List Nat := [8387]
set_option maxRecDepth 2048 in
theorem smallRow839_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 839 = smallRow839_block_004 := by decide

def smallRow839_small_left : List Nat := smallRow839_block_000 ++ smallRow839_block_001
theorem smallRow839_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 839 = smallRow839_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 839 ++ upperNeighbors PrimePairVertices.tree_block_001 839 = _
  rw [smallRow839_block_000_checked, smallRow839_block_001_checked]
  rfl

def smallRow839_small_right_right : List Nat := smallRow839_block_003 ++ smallRow839_block_004
theorem smallRow839_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 839 = smallRow839_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 839 ++ upperNeighbors PrimePairVertices.tree_block_004 839 = _
  rw [smallRow839_block_003_checked, smallRow839_block_004_checked]
  rfl

def smallRow839_small_right : List Nat := smallRow839_block_002 ++ smallRow839_small_right_right
theorem smallRow839_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 839 = smallRow839_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 839 ++ upperNeighbors PrimePairVertices.tree_small_right_right 839 = _
  rw [smallRow839_block_002_checked, smallRow839_small_right_right_checked]
  rfl

def smallRow839_small : List Nat := smallRow839_small_left ++ smallRow839_small_right
theorem smallRow839_small_checked : upperNeighbors PrimePairVertices.tree_small 839 = smallRow839_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 839 ++ upperNeighbors PrimePairVertices.tree_small_right 839 = _
  rw [smallRow839_small_left_checked, smallRow839_small_right_checked]
  rfl

def smallRow839 : List Nat := smallRow839_small
theorem smallRow839_checked : upperNeighbors PrimePairVertices.smallTree 839 = smallRow839 := smallRow839_small_checked

end ProjectEulerSolutions.PrimePairCache
