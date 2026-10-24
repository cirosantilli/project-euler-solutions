import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow67_block_000 : List Nat := [139, 157, 547, 601, 619, 751, 757, 1039, 1063, 1093, 1201, 1303, 1609, 1831, 1933, 1999, 2029]
set_option maxRecDepth 2048 in
theorem smallRow67_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 67 = smallRow67_block_000 := by decide

def smallRow67_block_001 : List Nat := [2251, 2341, 2377, 2557, 2767, 3313, 3499, 3637, 3943, 4057]
set_option maxRecDepth 2048 in
theorem smallRow67_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 67 = smallRow67_block_001 := by decide

def smallRow67_block_002 : List Nat := [4099, 4483, 4789, 4831, 4903, 4987, 5179, 5419, 5569, 5581, 5743, 5827, 5881, 5923]
set_option maxRecDepth 2048 in
theorem smallRow67_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 67 = smallRow67_block_002 := by decide

def smallRow67_block_003 : List Nat := [6337, 6661, 6829, 7561, 7639, 7681, 7687, 8191]
set_option maxRecDepth 2048 in
theorem smallRow67_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 67 = smallRow67_block_003 := by decide

def smallRow67_block_004 : List Nat := [8221]
set_option maxRecDepth 2048 in
theorem smallRow67_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 67 = smallRow67_block_004 := by decide

def smallRow67_small_left : List Nat := smallRow67_block_000 ++ smallRow67_block_001
theorem smallRow67_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 67 = smallRow67_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 67 ++ upperNeighbors PrimePairVertices.tree_block_001 67 = _
  rw [smallRow67_block_000_checked, smallRow67_block_001_checked]
  rfl

def smallRow67_small_right_right : List Nat := smallRow67_block_003 ++ smallRow67_block_004
theorem smallRow67_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 67 = smallRow67_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 67 ++ upperNeighbors PrimePairVertices.tree_block_004 67 = _
  rw [smallRow67_block_003_checked, smallRow67_block_004_checked]
  rfl

def smallRow67_small_right : List Nat := smallRow67_block_002 ++ smallRow67_small_right_right
theorem smallRow67_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 67 = smallRow67_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 67 ++ upperNeighbors PrimePairVertices.tree_small_right_right 67 = _
  rw [smallRow67_block_002_checked, smallRow67_small_right_right_checked]
  rfl

def smallRow67_small : List Nat := smallRow67_small_left ++ smallRow67_small_right
theorem smallRow67_small_checked : upperNeighbors PrimePairVertices.tree_small 67 = smallRow67_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 67 ++ upperNeighbors PrimePairVertices.tree_small_right 67 = _
  rw [smallRow67_small_left_checked, smallRow67_small_right_checked]
  rfl

def smallRow67 : List Nat := smallRow67_small
theorem smallRow67_checked : upperNeighbors PrimePairVertices.smallTree 67 = smallRow67 := smallRow67_small_checked

def smallRow71_block_000 : List Nat := [233, 257, 263, 317, 389, 443, 719, 821, 887, 947, 971, 1049, 1163, 1181, 1427, 1499, 1583, 1709, 1973]
set_option maxRecDepth 2048 in
theorem smallRow71_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 71 = smallRow71_block_000 := by decide

def smallRow71_block_001 : List Nat := [2417, 2477, 2687, 2819, 2837, 2909, 3347, 3491, 3833, 3863]
set_option maxRecDepth 2048 in
theorem smallRow71_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 71 = smallRow71_block_001 := by decide

def smallRow71_block_002 : List Nat := [4517, 4673, 5087, 5153, 5237, 5417, 5441, 5639, 5651, 5657, 5849, 5879]
set_option maxRecDepth 2048 in
theorem smallRow71_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 71 = smallRow71_block_002 := by decide

def smallRow71_block_003 : List Nat := [6389, 6563, 6869, 7001, 7103, 7109, 7151, 7229, 7589, 7817, 7841, 7919, 8087]
set_option maxRecDepth 2048 in
theorem smallRow71_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 71 = smallRow71_block_003 := by decide

def smallRow71_block_004 : List Nat := [8387]
set_option maxRecDepth 2048 in
theorem smallRow71_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 71 = smallRow71_block_004 := by decide

def smallRow71_small_left : List Nat := smallRow71_block_000 ++ smallRow71_block_001
theorem smallRow71_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 71 = smallRow71_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 71 ++ upperNeighbors PrimePairVertices.tree_block_001 71 = _
  rw [smallRow71_block_000_checked, smallRow71_block_001_checked]
  rfl

def smallRow71_small_right_right : List Nat := smallRow71_block_003 ++ smallRow71_block_004
theorem smallRow71_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 71 = smallRow71_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 71 ++ upperNeighbors PrimePairVertices.tree_block_004 71 = _
  rw [smallRow71_block_003_checked, smallRow71_block_004_checked]
  rfl

def smallRow71_small_right : List Nat := smallRow71_block_002 ++ smallRow71_small_right_right
theorem smallRow71_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 71 = smallRow71_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 71 ++ upperNeighbors PrimePairVertices.tree_small_right_right 71 = _
  rw [smallRow71_block_002_checked, smallRow71_small_right_right_checked]
  rfl

def smallRow71_small : List Nat := smallRow71_small_left ++ smallRow71_small_right
theorem smallRow71_small_checked : upperNeighbors PrimePairVertices.tree_small 71 = smallRow71_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 71 ++ upperNeighbors PrimePairVertices.tree_small_right 71 = _
  rw [smallRow71_small_left_checked, smallRow71_small_right_checked]
  rfl

def smallRow71 : List Nat := smallRow71_small
theorem smallRow71_checked : upperNeighbors PrimePairVertices.smallTree 71 = smallRow71 := smallRow71_small_checked

def smallRow73_block_000 : List Nat := [277, 547, 571, 607, 643, 757, 823, 1117, 1447, 1597, 1933, 2029]
set_option maxRecDepth 2048 in
theorem smallRow73_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 73 = smallRow73_block_000 := by decide

def smallRow73_block_001 : List Nat := [2287, 2293, 2467, 2521, 2617, 2689, 2749, 3391, 3433, 3517, 3823, 3847]
set_option maxRecDepth 2048 in
theorem smallRow73_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 73 = smallRow73_block_001 := by decide

def smallRow73_block_002 : List Nat := [4177, 4549, 4663, 4759, 4813, 4957, 5107, 5443, 5659, 5821, 5953, 6091, 6121]
set_option maxRecDepth 2048 in
theorem smallRow73_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 73 = smallRow73_block_002 := by decide

def smallRow73_block_003 : List Nat := [6277, 6367, 6691, 6793, 6823, 7129, 7159, 7207, 7309, 7351, 7537, 7591, 7687]
set_option maxRecDepth 2048 in
theorem smallRow73_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 73 = smallRow73_block_003 := by decide

def smallRow73_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow73_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 73 = smallRow73_block_004 := by decide

def smallRow73_small_left : List Nat := smallRow73_block_000 ++ smallRow73_block_001
theorem smallRow73_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 73 = smallRow73_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 73 ++ upperNeighbors PrimePairVertices.tree_block_001 73 = _
  rw [smallRow73_block_000_checked, smallRow73_block_001_checked]
  rfl

def smallRow73_small_right_right : List Nat := smallRow73_block_003 ++ smallRow73_block_004
theorem smallRow73_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 73 = smallRow73_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 73 ++ upperNeighbors PrimePairVertices.tree_block_004 73 = _
  rw [smallRow73_block_003_checked, smallRow73_block_004_checked]
  rfl

def smallRow73_small_right : List Nat := smallRow73_block_002 ++ smallRow73_small_right_right
theorem smallRow73_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 73 = smallRow73_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 73 ++ upperNeighbors PrimePairVertices.tree_small_right_right 73 = _
  rw [smallRow73_block_002_checked, smallRow73_small_right_right_checked]
  rfl

def smallRow73_small : List Nat := smallRow73_small_left ++ smallRow73_small_right
theorem smallRow73_small_checked : upperNeighbors PrimePairVertices.tree_small 73 = smallRow73_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 73 ++ upperNeighbors PrimePairVertices.tree_small_right 73 = _
  rw [smallRow73_small_left_checked, smallRow73_small_right_checked]
  rfl

def smallRow73 : List Nat := smallRow73_small
theorem smallRow73_checked : upperNeighbors PrimePairVertices.smallTree 73 = smallRow73 := smallRow73_small_checked

def smallRow79_block_000 : List Nat := [193, 241, 367, 397, 613, 631, 967, 1093, 1117, 1447, 1627, 1657, 1801, 1933, 1993]
set_option maxRecDepth 2048 in
theorem smallRow79_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 79 = smallRow79_block_000 := by decide

def smallRow79_block_001 : List Nat := [2179, 2371, 2551, 2593, 2647, 2689, 2797, 3181, 3253, 3343, 3511, 3517, 3547, 3607, 3733, 3853]
set_option maxRecDepth 2048 in
theorem smallRow79_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 79 = smallRow79_block_001 := by decide

def smallRow79_block_002 : List Nat := [4153, 4231, 4483, 4831, 4999, 5077, 5101, 5323, 5449, 5503, 5581, 5647, 5653, 5659, 5791, 6091]
set_option maxRecDepth 2048 in
theorem smallRow79_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 79 = smallRow79_block_002 := by decide

def smallRow79_block_003 : List Nat := [6217, 6373, 6451, 6553, 6673, 6781, 6871, 7057, 7207, 7549, 7561, 7591, 7933, 8179]
set_option maxRecDepth 2048 in
theorem smallRow79_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 79 = smallRow79_block_003 := by decide

def smallRow79_block_004 : List Nat := [8263]
set_option maxRecDepth 2048 in
theorem smallRow79_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 79 = smallRow79_block_004 := by decide

def smallRow79_small_left : List Nat := smallRow79_block_000 ++ smallRow79_block_001
theorem smallRow79_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 79 = smallRow79_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 79 ++ upperNeighbors PrimePairVertices.tree_block_001 79 = _
  rw [smallRow79_block_000_checked, smallRow79_block_001_checked]
  rfl

def smallRow79_small_right_right : List Nat := smallRow79_block_003 ++ smallRow79_block_004
theorem smallRow79_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 79 = smallRow79_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 79 ++ upperNeighbors PrimePairVertices.tree_block_004 79 = _
  rw [smallRow79_block_003_checked, smallRow79_block_004_checked]
  rfl

def smallRow79_small_right : List Nat := smallRow79_block_002 ++ smallRow79_small_right_right
theorem smallRow79_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 79 = smallRow79_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 79 ++ upperNeighbors PrimePairVertices.tree_small_right_right 79 = _
  rw [smallRow79_block_002_checked, smallRow79_small_right_right_checked]
  rfl

def smallRow79_small : List Nat := smallRow79_small_left ++ smallRow79_small_right
theorem smallRow79_small_checked : upperNeighbors PrimePairVertices.tree_small 79 = smallRow79_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 79 ++ upperNeighbors PrimePairVertices.tree_small_right 79 = _
  rw [smallRow79_small_left_checked, smallRow79_small_right_checked]
  rfl

def smallRow79 : List Nat := smallRow79_small
theorem smallRow79_checked : upperNeighbors PrimePairVertices.smallTree 79 = smallRow79 := smallRow79_small_checked

def smallRow83_block_000 : List Nat := [227, 311, 443, 449, 563, 701, 719, 773, 911, 1031, 1301, 1373, 1409, 1553, 1619, 1697, 1811, 1889, 2003]
set_option maxRecDepth 2048 in
theorem smallRow83_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 83 = smallRow83_block_000 := by decide

def smallRow83_block_001 : List Nat := [2063, 2339, 2411, 2591, 2633, 2687, 2693, 2729, 2801, 2969, 3191, 3347, 3491, 3557, 3659, 3923]
set_option maxRecDepth 2048 in
theorem smallRow83_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 83 = smallRow83_block_001 := by decide

def smallRow83_block_002 : List Nat := [4643, 5039, 5441, 5591, 5717, 5897, 5987]
set_option maxRecDepth 2048 in
theorem smallRow83_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 83 = smallRow83_block_002 := by decide

def smallRow83_block_003 : List Nat := [6203, 6491, 7079, 7283, 7307, 7583, 8093, 8171]
set_option maxRecDepth 2048 in
theorem smallRow83_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 83 = smallRow83_block_003 := by decide

def smallRow83_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow83_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 83 = smallRow83_block_004 := by decide

def smallRow83_small_left : List Nat := smallRow83_block_000 ++ smallRow83_block_001
theorem smallRow83_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 83 = smallRow83_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 83 ++ upperNeighbors PrimePairVertices.tree_block_001 83 = _
  rw [smallRow83_block_000_checked, smallRow83_block_001_checked]
  rfl

def smallRow83_small_right_right : List Nat := smallRow83_block_003 ++ smallRow83_block_004
theorem smallRow83_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 83 = smallRow83_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 83 ++ upperNeighbors PrimePairVertices.tree_block_004 83 = _
  rw [smallRow83_block_003_checked, smallRow83_block_004_checked]
  rfl

def smallRow83_small_right : List Nat := smallRow83_block_002 ++ smallRow83_small_right_right
theorem smallRow83_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 83 = smallRow83_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 83 ++ upperNeighbors PrimePairVertices.tree_small_right_right 83 = _
  rw [smallRow83_block_002_checked, smallRow83_small_right_right_checked]
  rfl

def smallRow83_small : List Nat := smallRow83_small_left ++ smallRow83_small_right
theorem smallRow83_small_checked : upperNeighbors PrimePairVertices.tree_small 83 = smallRow83_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 83 ++ upperNeighbors PrimePairVertices.tree_small_right 83 = _
  rw [smallRow83_small_left_checked, smallRow83_small_right_checked]
  rfl

def smallRow83 : List Nat := smallRow83_small
theorem smallRow83_checked : upperNeighbors PrimePairVertices.smallTree 83 = smallRow83 := smallRow83_small_checked

def smallRow89_block_000 : List Nat := [107, 137, 293, 431, 443, 521, 809, 821, 977, 983, 1061, 1223, 1409, 1427, 1523, 1571, 1823]
set_option maxRecDepth 2048 in
theorem smallRow89_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 89 = smallRow89_block_000 := by decide

def smallRow89_block_001 : List Nat := [2141, 2153, 2267, 2357, 2579, 2777, 3407, 3671, 3821, 3917, 3989]
set_option maxRecDepth 2048 in
theorem smallRow89_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 89 = smallRow89_block_001 := by decide

def smallRow89_block_002 : List Nat := [4211, 4259, 4409, 4721, 4973, 5003, 5039, 5087, 5171, 5189, 5309, 5333, 5387, 5393, 5507, 5651, 5861, 5879, 5903, 5987, 6113]
set_option maxRecDepth 2048 in
theorem smallRow89_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 89 = smallRow89_block_002 := by decide

def smallRow89_block_003 : List Nat := [6263, 6299, 6323, 6353, 6521, 6947, 7103, 7307, 7649, 7727, 8147]
set_option maxRecDepth 2048 in
theorem smallRow89_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 89 = smallRow89_block_003 := by decide

def smallRow89_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow89_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 89 = smallRow89_block_004 := by decide

def smallRow89_small_left : List Nat := smallRow89_block_000 ++ smallRow89_block_001
theorem smallRow89_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 89 = smallRow89_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 89 ++ upperNeighbors PrimePairVertices.tree_block_001 89 = _
  rw [smallRow89_block_000_checked, smallRow89_block_001_checked]
  rfl

def smallRow89_small_right_right : List Nat := smallRow89_block_003 ++ smallRow89_block_004
theorem smallRow89_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 89 = smallRow89_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 89 ++ upperNeighbors PrimePairVertices.tree_block_004 89 = _
  rw [smallRow89_block_003_checked, smallRow89_block_004_checked]
  rfl

def smallRow89_small_right : List Nat := smallRow89_block_002 ++ smallRow89_small_right_right
theorem smallRow89_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 89 = smallRow89_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 89 ++ upperNeighbors PrimePairVertices.tree_small_right_right 89 = _
  rw [smallRow89_block_002_checked, smallRow89_small_right_right_checked]
  rfl

def smallRow89_small : List Nat := smallRow89_small_left ++ smallRow89_small_right
theorem smallRow89_small_checked : upperNeighbors PrimePairVertices.tree_small 89 = smallRow89_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 89 ++ upperNeighbors PrimePairVertices.tree_small_right 89 = _
  rw [smallRow89_small_left_checked, smallRow89_small_right_checked]
  rfl

def smallRow89 : List Nat := smallRow89_small
theorem smallRow89_checked : upperNeighbors PrimePairVertices.smallTree 89 = smallRow89 := smallRow89_small_checked

def smallRow97_block_000 : List Nat := [157, 241, 373, 379, 787, 829, 883, 919, 967, 1021, 1039, 1063, 1093, 1279, 1291, 1381, 1783, 1951, 2017]
set_option maxRecDepth 2048 in
theorem smallRow97_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 97 = smallRow97_block_000 := by decide

def smallRow97_block_001 : List Nat := [2221, 2833, 3331, 3373, 3691, 3727, 3853]
set_option maxRecDepth 2048 in
theorem smallRow97_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 97 = smallRow97_block_001 := by decide

def smallRow97_block_002 : List Nat := [4261, 4507, 4513, 4657, 4957, 4969, 5011, 5281, 5581]
set_option maxRecDepth 2048 in
theorem smallRow97_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 97 = smallRow97_block_002 := by decide

def smallRow97_block_003 : List Nat := [6271, 6301, 6571, 6637, 6883, 6991, 7507, 7681, 7927, 8011, 8053]
set_option maxRecDepth 2048 in
theorem smallRow97_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 97 = smallRow97_block_003 := by decide

def smallRow97_block_004 : List Nat := [8209, 8269, 8287]
set_option maxRecDepth 2048 in
theorem smallRow97_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 97 = smallRow97_block_004 := by decide

def smallRow97_small_left : List Nat := smallRow97_block_000 ++ smallRow97_block_001
theorem smallRow97_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 97 = smallRow97_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 97 ++ upperNeighbors PrimePairVertices.tree_block_001 97 = _
  rw [smallRow97_block_000_checked, smallRow97_block_001_checked]
  rfl

def smallRow97_small_right_right : List Nat := smallRow97_block_003 ++ smallRow97_block_004
theorem smallRow97_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 97 = smallRow97_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 97 ++ upperNeighbors PrimePairVertices.tree_block_004 97 = _
  rw [smallRow97_block_003_checked, smallRow97_block_004_checked]
  rfl

def smallRow97_small_right : List Nat := smallRow97_block_002 ++ smallRow97_small_right_right
theorem smallRow97_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 97 = smallRow97_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 97 ++ upperNeighbors PrimePairVertices.tree_small_right_right 97 = _
  rw [smallRow97_block_002_checked, smallRow97_small_right_right_checked]
  rfl

def smallRow97_small : List Nat := smallRow97_small_left ++ smallRow97_small_right
theorem smallRow97_small_checked : upperNeighbors PrimePairVertices.tree_small 97 = smallRow97_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 97 ++ upperNeighbors PrimePairVertices.tree_small_right 97 = _
  rw [smallRow97_small_left_checked, smallRow97_small_right_checked]
  rfl

def smallRow97 : List Nat := smallRow97_small
theorem smallRow97_checked : upperNeighbors PrimePairVertices.smallTree 97 = smallRow97 := smallRow97_small_checked

def smallRow101_block_000 : List Nat := [107, 149, 197, 359, 383, 467, 641, 719, 1733, 1889, 1979]
set_option maxRecDepth 2048 in
theorem smallRow101_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 101 = smallRow101_block_000 := by decide

def smallRow101_block_001 : List Nat := [2549, 2633, 2657, 2699, 2903, 3329, 3527, 3833, 3851]
set_option maxRecDepth 2048 in
theorem smallRow101_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 101 = smallRow101_block_001 := by decide

def smallRow101_block_002 : List Nat := [4229, 4337, 4817, 5039, 5081, 5471, 6053]
set_option maxRecDepth 2048 in
theorem smallRow101_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 101 = smallRow101_block_002 := by decide

def smallRow101_block_003 : List Nat := [6221, 6689, 6971, 7193]
set_option maxRecDepth 2048 in
theorem smallRow101_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 101 = smallRow101_block_003 := by decide

def smallRow101_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow101_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 101 = smallRow101_block_004 := by decide

def smallRow101_small_left : List Nat := smallRow101_block_000 ++ smallRow101_block_001
theorem smallRow101_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 101 = smallRow101_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 101 ++ upperNeighbors PrimePairVertices.tree_block_001 101 = _
  rw [smallRow101_block_000_checked, smallRow101_block_001_checked]
  rfl

def smallRow101_small_right_right : List Nat := smallRow101_block_003 ++ smallRow101_block_004
theorem smallRow101_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 101 = smallRow101_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 101 ++ upperNeighbors PrimePairVertices.tree_block_004 101 = _
  rw [smallRow101_block_003_checked, smallRow101_block_004_checked]
  rfl

def smallRow101_small_right : List Nat := smallRow101_block_002 ++ smallRow101_small_right_right
theorem smallRow101_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 101 = smallRow101_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 101 ++ upperNeighbors PrimePairVertices.tree_small_right_right 101 = _
  rw [smallRow101_block_002_checked, smallRow101_small_right_right_checked]
  rfl

def smallRow101_small : List Nat := smallRow101_small_left ++ smallRow101_small_right
theorem smallRow101_small_checked : upperNeighbors PrimePairVertices.tree_small 101 = smallRow101_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 101 ++ upperNeighbors PrimePairVertices.tree_small_right 101 = _
  rw [smallRow101_small_left_checked, smallRow101_small_right_checked]
  rfl

def smallRow101 : List Nat := smallRow101_small
theorem smallRow101_checked : upperNeighbors PrimePairVertices.smallTree 101 = smallRow101 := smallRow101_small_checked

end ProjectEulerSolutions.PrimePairCache
