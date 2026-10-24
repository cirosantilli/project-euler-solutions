import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow31_block_000 : List Nat := [139, 151, 181, 859, 907, 991, 1009, 1123, 1153, 1237, 1279, 1447, 1951, 2029]
set_option maxRecDepth 2048 in
theorem smallRow31_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 31 = smallRow31_block_000 := by decide

def smallRow31_block_001 : List Nat := [2089, 2281, 2311, 2383, 2857, 2887, 3037, 3109, 3301, 3307, 3331, 3343, 3517, 3571, 3583, 3637, 3853, 3889, 4003, 4021]
set_option maxRecDepth 2048 in
theorem smallRow31_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 31 = smallRow31_block_001 := by decide

def smallRow31_block_002 : List Nat := [4159, 4243, 4261, 4339, 4357, 4441, 4513, 4549, 4723, 5011, 5179, 5281, 5527, 5569, 5701, 5779, 5851, 5881, 6037, 6067, 6073]
set_option maxRecDepth 2048 in
theorem smallRow31_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 31 = smallRow31_block_002 := by decide

def smallRow31_block_003 : List Nat := [6271, 6343, 6571, 7333, 7411, 7459, 7591, 7741, 8161, 8191]
set_option maxRecDepth 2048 in
theorem smallRow31_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 31 = smallRow31_block_003 := by decide

def smallRow31_block_004 : List Nat := [8287, 8377]
set_option maxRecDepth 2048 in
theorem smallRow31_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 31 = smallRow31_block_004 := by decide

def smallRow31_small_left : List Nat := smallRow31_block_000 ++ smallRow31_block_001
theorem smallRow31_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 31 = smallRow31_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 31 ++ upperNeighbors PrimePairVertices.tree_block_001 31 = _
  rw [smallRow31_block_000_checked, smallRow31_block_001_checked]
  rfl

def smallRow31_small_right_right : List Nat := smallRow31_block_003 ++ smallRow31_block_004
theorem smallRow31_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 31 = smallRow31_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 31 ++ upperNeighbors PrimePairVertices.tree_block_004 31 = _
  rw [smallRow31_block_003_checked, smallRow31_block_004_checked]
  rfl

def smallRow31_small_right : List Nat := smallRow31_block_002 ++ smallRow31_small_right_right
theorem smallRow31_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 31 = smallRow31_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 31 ++ upperNeighbors PrimePairVertices.tree_small_right_right 31 = _
  rw [smallRow31_block_002_checked, smallRow31_small_right_right_checked]
  rfl

def smallRow31_small : List Nat := smallRow31_small_left ++ smallRow31_small_right
theorem smallRow31_small_checked : upperNeighbors PrimePairVertices.tree_small 31 = smallRow31_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 31 ++ upperNeighbors PrimePairVertices.tree_small_right 31 = _
  rw [smallRow31_small_left_checked, smallRow31_small_right_checked]
  rfl

def smallRow31 : List Nat := smallRow31_small
theorem smallRow31_checked : upperNeighbors PrimePairVertices.smallTree 31 = smallRow31 := smallRow31_small_checked

def smallRow37_block_000 : List Nat := [67, 79, 199, 277, 313, 463, 607, 967, 991, 1069, 1153, 1237, 1303, 1321, 1447, 1471, 1549, 1699, 1873, 1951]
set_option maxRecDepth 2048 in
theorem smallRow37_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 37 = smallRow37_block_000 := by decide

def smallRow37_block_001 : List Nat := [2137, 2179, 2269, 2371, 2377, 2473, 2539, 2677, 2689, 2707, 2803, 3049, 3181, 3187, 3463, 3613, 4093]
set_option maxRecDepth 2048 in
theorem smallRow37_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 37 = smallRow37_block_001 := by decide

def smallRow37_block_002 : List Nat := [4159, 4177, 4603, 4729, 4789, 4909, 5101, 5113, 5227, 5281, 5449, 5623, 5779, 5857, 5923, 6133]
set_option maxRecDepth 2048 in
theorem smallRow37_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 37 = smallRow37_block_002 := by decide

def smallRow37_block_003 : List Nat := [6373, 6469, 6529, 6823, 7297, 7369, 7393, 7477, 7537, 7873, 7963, 8011, 8089, 8101]
set_option maxRecDepth 2048 in
theorem smallRow37_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 37 = smallRow37_block_003 := by decide

def smallRow37_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow37_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 37 = smallRow37_block_004 := by decide

def smallRow37_small_left : List Nat := smallRow37_block_000 ++ smallRow37_block_001
theorem smallRow37_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 37 = smallRow37_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 37 ++ upperNeighbors PrimePairVertices.tree_block_001 37 = _
  rw [smallRow37_block_000_checked, smallRow37_block_001_checked]
  rfl

def smallRow37_small_right_right : List Nat := smallRow37_block_003 ++ smallRow37_block_004
theorem smallRow37_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 37 = smallRow37_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 37 ++ upperNeighbors PrimePairVertices.tree_block_004 37 = _
  rw [smallRow37_block_003_checked, smallRow37_block_004_checked]
  rfl

def smallRow37_small_right : List Nat := smallRow37_block_002 ++ smallRow37_small_right_right
theorem smallRow37_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 37 = smallRow37_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 37 ++ upperNeighbors PrimePairVertices.tree_small_right_right 37 = _
  rw [smallRow37_block_002_checked, smallRow37_small_right_right_checked]
  rfl

def smallRow37_small : List Nat := smallRow37_small_left ++ smallRow37_small_right
theorem smallRow37_small_checked : upperNeighbors PrimePairVertices.tree_small 37 = smallRow37_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 37 ++ upperNeighbors PrimePairVertices.tree_small_right 37 = _
  rw [smallRow37_small_left_checked, smallRow37_small_right_checked]
  rfl

def smallRow37 : List Nat := smallRow37_small
theorem smallRow37_checked : upperNeighbors PrimePairVertices.smallTree 37 = smallRow37 := smallRow37_small_checked

def smallRow41_block_000 : List Nat := [227, 257, 593, 719, 863, 887, 911, 1013, 1031, 1259, 1583, 1613, 1637, 1667, 1823]
set_option maxRecDepth 2048 in
theorem smallRow41_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 41 = smallRow41_block_000 := by decide

def smallRow41_block_001 : List Nat := [2081, 2333, 2339, 2411, 2609, 2939, 3299, 3461, 3527, 3533, 3719, 4013, 4049]
set_option maxRecDepth 2048 in
theorem smallRow41_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 41 = smallRow41_block_001 := by decide

def smallRow41_block_002 : List Nat := [4217, 4451, 4457, 4679, 4691, 5147, 5477, 5717, 6089]
set_option maxRecDepth 2048 in
theorem smallRow41_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 41 = smallRow41_block_002 := by decide

def smallRow41_block_003 : List Nat := [6263, 6473, 6491, 6719, 6833, 7187, 7331, 7451, 7457, 7583, 7643, 7793, 8069]
set_option maxRecDepth 2048 in
theorem smallRow41_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 41 = smallRow41_block_003 := by decide

def smallRow41_block_004 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallRow41_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 41 = smallRow41_block_004 := by decide

def smallRow41_small_left : List Nat := smallRow41_block_000 ++ smallRow41_block_001
theorem smallRow41_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 41 = smallRow41_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 41 ++ upperNeighbors PrimePairVertices.tree_block_001 41 = _
  rw [smallRow41_block_000_checked, smallRow41_block_001_checked]
  rfl

def smallRow41_small_right_right : List Nat := smallRow41_block_003 ++ smallRow41_block_004
theorem smallRow41_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 41 = smallRow41_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 41 ++ upperNeighbors PrimePairVertices.tree_block_004 41 = _
  rw [smallRow41_block_003_checked, smallRow41_block_004_checked]
  rfl

def smallRow41_small_right : List Nat := smallRow41_block_002 ++ smallRow41_small_right_right
theorem smallRow41_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 41 = smallRow41_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 41 ++ upperNeighbors PrimePairVertices.tree_small_right_right 41 = _
  rw [smallRow41_block_002_checked, smallRow41_small_right_right_checked]
  rfl

def smallRow41_small : List Nat := smallRow41_small_left ++ smallRow41_small_right
theorem smallRow41_small_checked : upperNeighbors PrimePairVertices.tree_small 41 = smallRow41_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 41 ++ upperNeighbors PrimePairVertices.tree_small_right 41 = _
  rw [smallRow41_small_left_checked, smallRow41_small_right_checked]
  rfl

def smallRow41 : List Nat := smallRow41_small
theorem smallRow41_checked : upperNeighbors PrimePairVertices.smallTree 41 = smallRow41 := smallRow41_small_checked

def smallRow43_block_000 : List Nat := [97, 103, 223, 271, 499, 613, 691, 1051, 1153, 1213, 1381, 1399, 1621, 1693, 1777, 1867, 1993]
set_option maxRecDepth 2048 in
theorem smallRow43_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 43 = smallRow43_block_000 := by decide

def smallRow43_block_001 : List Nat := [2251, 2389, 2503, 2557, 2749, 3049, 3187, 3253, 3259, 3319, 3361, 3373, 3469, 3877, 3931]
set_option maxRecDepth 2048 in
theorem smallRow43_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 43 = smallRow43_block_001 := by decide

def smallRow43_block_002 : List Nat := [4111, 4243, 4363, 4783, 4813, 5563, 5569, 5653, 5857, 5923, 6091]
set_option maxRecDepth 2048 in
theorem smallRow43_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 43 = smallRow43_block_002 := by decide

def smallRow43_block_003 : List Nat := [6343, 6529, 6547, 6967, 7159, 7351, 7753]
set_option maxRecDepth 2048 in
theorem smallRow43_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 43 = smallRow43_block_003 := by decide

def smallRow43_block_004 : List Nat := [8287, 8329]
set_option maxRecDepth 2048 in
theorem smallRow43_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 43 = smallRow43_block_004 := by decide

def smallRow43_small_left : List Nat := smallRow43_block_000 ++ smallRow43_block_001
theorem smallRow43_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 43 = smallRow43_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 43 ++ upperNeighbors PrimePairVertices.tree_block_001 43 = _
  rw [smallRow43_block_000_checked, smallRow43_block_001_checked]
  rfl

def smallRow43_small_right_right : List Nat := smallRow43_block_003 ++ smallRow43_block_004
theorem smallRow43_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 43 = smallRow43_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 43 ++ upperNeighbors PrimePairVertices.tree_block_004 43 = _
  rw [smallRow43_block_003_checked, smallRow43_block_004_checked]
  rfl

def smallRow43_small_right : List Nat := smallRow43_block_002 ++ smallRow43_small_right_right
theorem smallRow43_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 43 = smallRow43_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 43 ++ upperNeighbors PrimePairVertices.tree_small_right_right 43 = _
  rw [smallRow43_block_002_checked, smallRow43_small_right_right_checked]
  rfl

def smallRow43_small : List Nat := smallRow43_small_left ++ smallRow43_small_right
theorem smallRow43_small_checked : upperNeighbors PrimePairVertices.tree_small 43 = smallRow43_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 43 ++ upperNeighbors PrimePairVertices.tree_small_right 43 = _
  rw [smallRow43_small_left_checked, smallRow43_small_right_checked]
  rfl

def smallRow43 : List Nat := smallRow43_small
theorem smallRow43_checked : upperNeighbors PrimePairVertices.smallTree 43 = smallRow43 := smallRow43_small_checked

def smallRow47_block_000 : List Nat := [149, 251, 269, 293, 419, 521, 947, 1091, 1187, 1277, 1283, 1301, 1439, 1481, 1487, 1619, 1721, 1901, 1931, 2027]
set_option maxRecDepth 2048 in
theorem smallRow47_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 47 = smallRow47_block_000 := by decide

def smallRow47_block_001 : List Nat := [2063, 2333, 2393, 2399, 2687, 2741, 2963, 3191, 3257, 3617, 3761, 3833, 3923, 4073]
set_option maxRecDepth 2048 in
theorem smallRow47_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 47 = smallRow47_block_001 := by decide

def smallRow47_block_002 : List Nat := [4211, 4241, 4337, 4751, 4787, 4931, 4937, 5147, 5273, 5297, 5381, 5483, 5639, 5669, 5807, 5879, 5927, 6029, 6089]
set_option maxRecDepth 2048 in
theorem smallRow47_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 47 = smallRow47_block_002 := by decide

def smallRow47_block_003 : List Nat := [6599, 6659, 6701, 6719, 6737, 6803, 6869, 6911, 7517, 8069, 8087, 8111, 8171]
set_option maxRecDepth 2048 in
theorem smallRow47_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 47 = smallRow47_block_003 := by decide

def smallRow47_block_004 : List Nat := [8273]
set_option maxRecDepth 2048 in
theorem smallRow47_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 47 = smallRow47_block_004 := by decide

def smallRow47_small_left : List Nat := smallRow47_block_000 ++ smallRow47_block_001
theorem smallRow47_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 47 = smallRow47_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 47 ++ upperNeighbors PrimePairVertices.tree_block_001 47 = _
  rw [smallRow47_block_000_checked, smallRow47_block_001_checked]
  rfl

def smallRow47_small_right_right : List Nat := smallRow47_block_003 ++ smallRow47_block_004
theorem smallRow47_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 47 = smallRow47_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 47 ++ upperNeighbors PrimePairVertices.tree_block_004 47 = _
  rw [smallRow47_block_003_checked, smallRow47_block_004_checked]
  rfl

def smallRow47_small_right : List Nat := smallRow47_block_002 ++ smallRow47_small_right_right
theorem smallRow47_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 47 = smallRow47_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 47 ++ upperNeighbors PrimePairVertices.tree_small_right_right 47 = _
  rw [smallRow47_block_002_checked, smallRow47_small_right_right_checked]
  rfl

def smallRow47_small : List Nat := smallRow47_small_left ++ smallRow47_small_right
theorem smallRow47_small_checked : upperNeighbors PrimePairVertices.tree_small 47 = smallRow47_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 47 ++ upperNeighbors PrimePairVertices.tree_small_right 47 = _
  rw [smallRow47_small_left_checked, smallRow47_small_right_checked]
  rfl

def smallRow47 : List Nat := smallRow47_small
theorem smallRow47_checked : upperNeighbors PrimePairVertices.smallTree 47 = smallRow47 := smallRow47_small_checked

def smallRow53_block_000 : List Nat := [113, 197, 269, 353, 401, 419, 653, 1229, 1481, 1637, 1823, 1847, 1997, 2027]
set_option maxRecDepth 2048 in
theorem smallRow53_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 53 = smallRow53_block_000 := by decide

def smallRow53_block_001 : List Nat := [2069, 2099, 2153, 2267, 2333, 2447, 2531, 2621, 2663, 3011, 3167, 3257, 3371, 3581, 3593, 4007, 4019, 4091]
set_option maxRecDepth 2048 in
theorem smallRow53_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 53 = smallRow53_block_001 := by decide

def smallRow53_block_002 : List Nat := [4283, 4637, 4649, 4799, 4943, 5273, 5303, 5333, 5783, 5861]
set_option maxRecDepth 2048 in
theorem smallRow53_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 53 = smallRow53_block_002 := by decide

def smallRow53_block_003 : List Nat := [6287, 6311, 6323, 6353, 6779, 6791, 6803, 6857, 7079, 7307, 7703, 7793, 7841, 7853, 7883, 8117, 8123]
set_option maxRecDepth 2048 in
theorem smallRow53_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 53 = smallRow53_block_003 := by decide

def smallRow53_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow53_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 53 = smallRow53_block_004 := by decide

def smallRow53_small_left : List Nat := smallRow53_block_000 ++ smallRow53_block_001
theorem smallRow53_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 53 = smallRow53_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 53 ++ upperNeighbors PrimePairVertices.tree_block_001 53 = _
  rw [smallRow53_block_000_checked, smallRow53_block_001_checked]
  rfl

def smallRow53_small_right_right : List Nat := smallRow53_block_003 ++ smallRow53_block_004
theorem smallRow53_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 53 = smallRow53_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 53 ++ upperNeighbors PrimePairVertices.tree_block_004 53 = _
  rw [smallRow53_block_003_checked, smallRow53_block_004_checked]
  rfl

def smallRow53_small_right : List Nat := smallRow53_block_002 ++ smallRow53_small_right_right
theorem smallRow53_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 53 = smallRow53_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 53 ++ upperNeighbors PrimePairVertices.tree_small_right_right 53 = _
  rw [smallRow53_block_002_checked, smallRow53_small_right_right_checked]
  rfl

def smallRow53_small : List Nat := smallRow53_small_left ++ smallRow53_small_right
theorem smallRow53_small_checked : upperNeighbors PrimePairVertices.tree_small 53 = smallRow53_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 53 ++ upperNeighbors PrimePairVertices.tree_small_right 53 = _
  rw [smallRow53_small_left_checked, smallRow53_small_right_checked]
  rfl

def smallRow53 : List Nat := smallRow53_small
theorem smallRow53_checked : upperNeighbors PrimePairVertices.smallTree 53 = smallRow53 := smallRow53_small_checked

def smallRow59_block_000 : List Nat := [167, 197, 419, 929, 971, 1091, 1163, 1193, 1259, 1289, 1319, 1601, 1901, 1973]
set_option maxRecDepth 2048 in
theorem smallRow59_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 59 = smallRow59_block_000 := by decide

def smallRow59_block_001 : List Nat := [2087, 2099, 2237, 2309, 2351, 2531, 2609, 2663, 2897, 2903, 3083, 3449, 3767, 3851]
set_option maxRecDepth 2048 in
theorem smallRow59_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 59 = smallRow59_block_001 := by decide

def smallRow59_block_002 : List Nat := [4157, 4211, 4397, 4721, 4751, 4889, 4931, 5039, 5081, 5261, 5303, 5351, 5381, 5519, 5717, 5741, 5807, 5843, 5927, 5981, 6047]
set_option maxRecDepth 2048 in
theorem smallRow59_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 59 = smallRow59_block_002 := by decide

def smallRow59_block_003 : List Nat := [6569, 6653, 6779, 6857, 6863, 6899, 6917, 6977, 6983, 7253, 7349, 7523, 7559, 7673, 8093, 8123]
set_option maxRecDepth 2048 in
theorem smallRow59_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 59 = smallRow59_block_003 := by decide

def smallRow59_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow59_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 59 = smallRow59_block_004 := by decide

def smallRow59_small_left : List Nat := smallRow59_block_000 ++ smallRow59_block_001
theorem smallRow59_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 59 = smallRow59_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 59 ++ upperNeighbors PrimePairVertices.tree_block_001 59 = _
  rw [smallRow59_block_000_checked, smallRow59_block_001_checked]
  rfl

def smallRow59_small_right_right : List Nat := smallRow59_block_003 ++ smallRow59_block_004
theorem smallRow59_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 59 = smallRow59_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 59 ++ upperNeighbors PrimePairVertices.tree_block_004 59 = _
  rw [smallRow59_block_003_checked, smallRow59_block_004_checked]
  rfl

def smallRow59_small_right : List Nat := smallRow59_block_002 ++ smallRow59_small_right_right
theorem smallRow59_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 59 = smallRow59_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 59 ++ upperNeighbors PrimePairVertices.tree_small_right_right 59 = _
  rw [smallRow59_block_002_checked, smallRow59_small_right_right_checked]
  rfl

def smallRow59_small : List Nat := smallRow59_small_left ++ smallRow59_small_right
theorem smallRow59_small_checked : upperNeighbors PrimePairVertices.tree_small 59 = smallRow59_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 59 ++ upperNeighbors PrimePairVertices.tree_small_right 59 = _
  rw [smallRow59_small_left_checked, smallRow59_small_right_checked]
  rfl

def smallRow59 : List Nat := smallRow59_small
theorem smallRow59_checked : upperNeighbors PrimePairVertices.smallTree 59 = smallRow59 := smallRow59_small_checked

def smallRow61_block_000 : List Nat := [151, 331, 409, 487, 751, 1069, 1453, 1483, 1693, 1753, 1801, 1873, 1951, 1999]
set_option maxRecDepth 2048 in
theorem smallRow61_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 61 = smallRow61_block_000 := by decide

def smallRow61_block_001 : List Nat := [2341, 2371, 2383, 2797, 2971, 3049, 3169, 3181, 3463, 3469, 3637, 3733, 3889]
set_option maxRecDepth 2048 in
theorem smallRow61_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 61 = smallRow61_block_001 := by decide

def smallRow61_block_002 : List Nat := [4657, 5413, 5431, 5449, 5557, 5623, 5821, 5827, 5869, 6079]
set_option maxRecDepth 2048 in
theorem smallRow61_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 61 = smallRow61_block_002 := by decide

def smallRow61_block_003 : List Nat := [6367, 6673, 6793, 6871, 6997, 7129, 7237, 7369, 7681, 7699, 7873, 7951, 7963, 8161]
set_option maxRecDepth 2048 in
theorem smallRow61_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 61 = smallRow61_block_003 := by decide

def smallRow61_block_004 : List Nat := [8311]
set_option maxRecDepth 2048 in
theorem smallRow61_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 61 = smallRow61_block_004 := by decide

def smallRow61_small_left : List Nat := smallRow61_block_000 ++ smallRow61_block_001
theorem smallRow61_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 61 = smallRow61_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 61 ++ upperNeighbors PrimePairVertices.tree_block_001 61 = _
  rw [smallRow61_block_000_checked, smallRow61_block_001_checked]
  rfl

def smallRow61_small_right_right : List Nat := smallRow61_block_003 ++ smallRow61_block_004
theorem smallRow61_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 61 = smallRow61_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 61 ++ upperNeighbors PrimePairVertices.tree_block_004 61 = _
  rw [smallRow61_block_003_checked, smallRow61_block_004_checked]
  rfl

def smallRow61_small_right : List Nat := smallRow61_block_002 ++ smallRow61_small_right_right
theorem smallRow61_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 61 = smallRow61_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 61 ++ upperNeighbors PrimePairVertices.tree_small_right_right 61 = _
  rw [smallRow61_block_002_checked, smallRow61_small_right_right_checked]
  rfl

def smallRow61_small : List Nat := smallRow61_small_left ++ smallRow61_small_right
theorem smallRow61_small_checked : upperNeighbors PrimePairVertices.tree_small 61 = smallRow61_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 61 ++ upperNeighbors PrimePairVertices.tree_small_right 61 = _
  rw [smallRow61_small_left_checked, smallRow61_small_right_checked]
  rfl

def smallRow61 : List Nat := smallRow61_small
theorem smallRow61_checked : upperNeighbors PrimePairVertices.smallTree 61 = smallRow61 := smallRow61_small_checked

end ProjectEulerSolutions.PrimePairCache
