import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow3_block_000 : List Nat := [7, 11, 17, 31, 37, 59, 67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907]
set_option maxRecDepth 2048 in
theorem smallRow3_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 3 = smallRow3_block_000 := by decide

def smallRow3_block_001 : List Nat := [2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019]
set_option maxRecDepth 2048 in
theorem smallRow3_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 3 = smallRow3_block_001 := by decide

def smallRow3_block_002 : List Nat := [4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073]
set_option maxRecDepth 2048 in
theorem smallRow3_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 3 = smallRow3_block_002 := by decide

def smallRow3_block_003 : List Nat := [6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963]
set_option maxRecDepth 2048 in
theorem smallRow3_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 3 = smallRow3_block_003 := by decide

def smallRow3_block_004 : List Nat := [8219, 8237, 8377]
set_option maxRecDepth 2048 in
theorem smallRow3_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 3 = smallRow3_block_004 := by decide

def smallRow3_small_left : List Nat := smallRow3_block_000 ++ smallRow3_block_001
theorem smallRow3_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 3 = smallRow3_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 3 ++ upperNeighbors PrimePairVertices.tree_block_001 3 = _
  rw [smallRow3_block_000_checked, smallRow3_block_001_checked]
  rfl

def smallRow3_small_right_right : List Nat := smallRow3_block_003 ++ smallRow3_block_004
theorem smallRow3_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 3 = smallRow3_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 3 ++ upperNeighbors PrimePairVertices.tree_block_004 3 = _
  rw [smallRow3_block_003_checked, smallRow3_block_004_checked]
  rfl

def smallRow3_small_right : List Nat := smallRow3_block_002 ++ smallRow3_small_right_right
theorem smallRow3_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 3 = smallRow3_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 3 ++ upperNeighbors PrimePairVertices.tree_small_right_right 3 = _
  rw [smallRow3_block_002_checked, smallRow3_small_right_right_checked]
  rfl

def smallRow3_small : List Nat := smallRow3_small_left ++ smallRow3_small_right
theorem smallRow3_small_checked : upperNeighbors PrimePairVertices.tree_small 3 = smallRow3_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 3 ++ upperNeighbors PrimePairVertices.tree_small_right 3 = _
  rw [smallRow3_small_left_checked, smallRow3_small_right_checked]
  rfl

def smallRow3 : List Nat := smallRow3_small
theorem smallRow3_checked : upperNeighbors PrimePairVertices.smallTree 3 = smallRow3 := smallRow3_small_checked

def smallRow7_block_000 : List Nat := [19, 61, 97, 109, 127, 229, 283, 433, 487, 523, 541, 547, 673, 691, 757, 823, 829, 853, 883, 937, 1171, 1237, 1249, 1399, 1453, 1471, 1549, 1693, 1741, 1861, 1879, 1993, 1999]
set_option maxRecDepth 2048 in
theorem smallRow7_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 7 = smallRow7_block_000 := by decide

def smallRow7_block_001 : List Nat := [2089, 2161, 2269, 2287, 2341, 2467, 2503, 2617, 2671, 2707, 2719, 2953, 3019, 3181, 3331, 3361, 3433, 3583, 3613, 3709, 3727, 3823, 4027]
set_option maxRecDepth 2048 in
theorem smallRow7_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 7 = smallRow7_block_001 := by decide

def smallRow7_block_002 : List Nat := [4159, 4177, 4201, 4219, 4357, 4441, 4507, 4567, 4729, 4903, 5437, 5521, 5641, 5659, 5689, 5821, 6091]
set_option maxRecDepth 2048 in
theorem smallRow7_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 7 = smallRow7_block_002 := by decide

def smallRow7_block_003 : List Nat := [6163, 6421, 6481, 6679, 6949, 7351, 7417, 7489, 7723, 7933, 8017, 8101, 8167]
set_option maxRecDepth 2048 in
theorem smallRow7_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 7 = smallRow7_block_003 := by decide

def smallRow7_block_004 : List Nat := [8311, 8317]
set_option maxRecDepth 2048 in
theorem smallRow7_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 7 = smallRow7_block_004 := by decide

def smallRow7_small_left : List Nat := smallRow7_block_000 ++ smallRow7_block_001
theorem smallRow7_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 7 = smallRow7_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 7 ++ upperNeighbors PrimePairVertices.tree_block_001 7 = _
  rw [smallRow7_block_000_checked, smallRow7_block_001_checked]
  rfl

def smallRow7_small_right_right : List Nat := smallRow7_block_003 ++ smallRow7_block_004
theorem smallRow7_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 7 = smallRow7_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 7 ++ upperNeighbors PrimePairVertices.tree_block_004 7 = _
  rw [smallRow7_block_003_checked, smallRow7_block_004_checked]
  rfl

def smallRow7_small_right : List Nat := smallRow7_block_002 ++ smallRow7_small_right_right
theorem smallRow7_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 7 = smallRow7_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 7 ++ upperNeighbors PrimePairVertices.tree_small_right_right 7 = _
  rw [smallRow7_block_002_checked, smallRow7_small_right_right_checked]
  rfl

def smallRow7_small : List Nat := smallRow7_small_left ++ smallRow7_small_right
theorem smallRow7_small_checked : upperNeighbors PrimePairVertices.tree_small 7 = smallRow7_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 7 ++ upperNeighbors PrimePairVertices.tree_small_right 7 = _
  rw [smallRow7_small_left_checked, smallRow7_small_right_checked]
  rfl

def smallRow7 : List Nat := smallRow7_small
theorem smallRow7_checked : upperNeighbors PrimePairVertices.smallTree 7 = smallRow7 := smallRow7_small_checked

def smallRow11_block_000 : List Nat := [23, 113, 239, 251, 353, 503, 587, 701, 743, 839, 863, 941, 953, 1049, 1091, 1103, 1217, 1427, 1487, 1847, 1871, 1949, 1973]
set_option maxRecDepth 2048 in
theorem smallRow11_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 11 = smallRow11_block_000 := by decide

def smallRow11_block_001 : List Nat := [2069, 2213, 2237, 2297, 2339, 2459, 2621, 2657, 2843, 2927, 3083, 3209, 3467, 3539, 3779, 4013, 4073]
set_option maxRecDepth 2048 in
theorem smallRow11_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 11 = smallRow11_block_001 := by decide

def smallRow11_block_002 : List Nat := [4217, 4229, 4259, 4493, 4547, 4643, 4967, 5099, 5153, 5309, 5783, 5807, 5849, 5861, 5987, 6047]
set_option maxRecDepth 2048 in
theorem smallRow11_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 11 = smallRow11_block_002 := by decide

def smallRow11_block_003 : List Nat := [6329, 6719, 6791, 6911, 7109, 7127, 7331, 7517, 7529, 7541, 7577, 7703, 7757, 7877, 7937, 8081, 8171]
set_option maxRecDepth 2048 in
theorem smallRow11_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 11 = smallRow11_block_003 := by decide

def smallRow11_block_004 : List Nat := [8219, 8273, 8387]
set_option maxRecDepth 2048 in
theorem smallRow11_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 11 = smallRow11_block_004 := by decide

def smallRow11_small_left : List Nat := smallRow11_block_000 ++ smallRow11_block_001
theorem smallRow11_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 11 = smallRow11_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 11 ++ upperNeighbors PrimePairVertices.tree_block_001 11 = _
  rw [smallRow11_block_000_checked, smallRow11_block_001_checked]
  rfl

def smallRow11_small_right_right : List Nat := smallRow11_block_003 ++ smallRow11_block_004
theorem smallRow11_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 11 = smallRow11_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 11 ++ upperNeighbors PrimePairVertices.tree_block_004 11 = _
  rw [smallRow11_block_003_checked, smallRow11_block_004_checked]
  rfl

def smallRow11_small_right : List Nat := smallRow11_block_002 ++ smallRow11_small_right_right
theorem smallRow11_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 11 = smallRow11_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 11 ++ upperNeighbors PrimePairVertices.tree_small_right_right 11 = _
  rw [smallRow11_block_002_checked, smallRow11_small_right_right_checked]
  rfl

def smallRow11_small : List Nat := smallRow11_small_left ++ smallRow11_small_right
theorem smallRow11_small_checked : upperNeighbors PrimePairVertices.tree_small 11 = smallRow11_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 11 ++ upperNeighbors PrimePairVertices.tree_small_right 11 = _
  rw [smallRow11_small_left_checked, smallRow11_small_right_checked]
  rfl

def smallRow11 : List Nat := smallRow11_small
theorem smallRow11_checked : upperNeighbors PrimePairVertices.smallTree 11 = smallRow11 := smallRow11_small_checked

def smallRow13_block_000 : List Nat := [19, 61, 103, 127, 241, 331, 337, 367, 523, 577, 709, 829, 997, 1009, 1129, 1213, 1231, 1321, 1381, 1489, 1543, 1627, 1861]
set_option maxRecDepth 2048 in
theorem smallRow13_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 13 = smallRow13_block_000 := by decide

def smallRow13_block_001 : List Nat := [2113, 2137, 2287, 2347, 2383, 2689, 2851, 2953, 2971, 3187, 3499, 3559, 3583, 3673, 3967]
set_option maxRecDepth 2048 in
theorem smallRow13_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 13 = smallRow13_block_001 := by decide

def smallRow13_block_002 : List Nat := [4219, 4243, 4327, 4363, 4513, 4591, 4789, 4909, 4951, 5077, 5197, 5209, 5431, 5581, 5623, 5647, 5701, 5851, 6043]
set_option maxRecDepth 2048 in
theorem smallRow13_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 13 = smallRow13_block_002 := by decide

def smallRow13_block_003 : List Nat := [6163, 6343, 6397, 6421, 6691, 6733, 6841, 6991, 7477, 7507, 7639, 7723, 7927, 7993, 8053, 8059, 8179]
set_option maxRecDepth 2048 in
theorem smallRow13_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 13 = smallRow13_block_003 := by decide

def smallRow13_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow13_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 13 = smallRow13_block_004 := by decide

def smallRow13_small_left : List Nat := smallRow13_block_000 ++ smallRow13_block_001
theorem smallRow13_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 13 = smallRow13_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 13 ++ upperNeighbors PrimePairVertices.tree_block_001 13 = _
  rw [smallRow13_block_000_checked, smallRow13_block_001_checked]
  rfl

def smallRow13_small_right_right : List Nat := smallRow13_block_003 ++ smallRow13_block_004
theorem smallRow13_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 13 = smallRow13_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 13 ++ upperNeighbors PrimePairVertices.tree_block_004 13 = _
  rw [smallRow13_block_003_checked, smallRow13_block_004_checked]
  rfl

def smallRow13_small_right : List Nat := smallRow13_block_002 ++ smallRow13_small_right_right
theorem smallRow13_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 13 = smallRow13_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 13 ++ upperNeighbors PrimePairVertices.tree_small_right_right 13 = _
  rw [smallRow13_block_002_checked, smallRow13_small_right_right_checked]
  rfl

def smallRow13_small : List Nat := smallRow13_small_left ++ smallRow13_small_right
theorem smallRow13_small_checked : upperNeighbors PrimePairVertices.tree_small 13 = smallRow13_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 13 ++ upperNeighbors PrimePairVertices.tree_small_right 13 = _
  rw [smallRow13_small_left_checked, smallRow13_small_right_checked]
  rfl

def smallRow13 : List Nat := smallRow13_small
theorem smallRow13_checked : upperNeighbors PrimePairVertices.smallTree 13 = smallRow13 := smallRow13_small_checked

def smallRow17_block_000 : List Nat := [83, 239, 257, 383, 389, 431, 449, 491, 881, 971, 1049, 1553, 2027]
set_option maxRecDepth 2048 in
theorem smallRow17_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 17 = smallRow17_block_000 := by decide

def smallRow17_block_001 : List Nat := [2069, 2153, 2213, 2243, 2297, 2351, 2411, 2657, 2741, 3023, 3137, 3191, 3347, 3359, 3539, 3917, 4019, 4079]
set_option maxRecDepth 2048 in
theorem smallRow17_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 17 = smallRow17_block_001 := by decide

def smallRow17_block_002 : List Nat := [4157, 4241, 4583, 4637, 4649, 4673, 4703, 4877, 4943, 5003, 5261, 5333, 5519, 5783, 5897]
set_option maxRecDepth 2048 in
theorem smallRow17_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 17 = smallRow17_block_002 := by decide

def smallRow17_block_003 : List Nat := [6317, 6353, 6521, 6599, 6857, 6899, 7109, 7127, 7211, 7283, 7487, 7691, 7841, 7883, 8069]
set_option maxRecDepth 2048 in
theorem smallRow17_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 17 = smallRow17_block_003 := by decide

def smallRow17_block_004 : List Nat := [8231]
set_option maxRecDepth 2048 in
theorem smallRow17_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 17 = smallRow17_block_004 := by decide

def smallRow17_small_left : List Nat := smallRow17_block_000 ++ smallRow17_block_001
theorem smallRow17_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 17 = smallRow17_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 17 ++ upperNeighbors PrimePairVertices.tree_block_001 17 = _
  rw [smallRow17_block_000_checked, smallRow17_block_001_checked]
  rfl

def smallRow17_small_right_right : List Nat := smallRow17_block_003 ++ smallRow17_block_004
theorem smallRow17_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 17 = smallRow17_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 17 ++ upperNeighbors PrimePairVertices.tree_block_004 17 = _
  rw [smallRow17_block_003_checked, smallRow17_block_004_checked]
  rfl

def smallRow17_small_right : List Nat := smallRow17_block_002 ++ smallRow17_small_right_right
theorem smallRow17_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 17 = smallRow17_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 17 ++ upperNeighbors PrimePairVertices.tree_small_right_right 17 = _
  rw [smallRow17_block_002_checked, smallRow17_small_right_right_checked]
  rfl

def smallRow17_small : List Nat := smallRow17_small_left ++ smallRow17_small_right
theorem smallRow17_small_checked : upperNeighbors PrimePairVertices.tree_small 17 = smallRow17_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 17 ++ upperNeighbors PrimePairVertices.tree_small_right 17 = _
  rw [smallRow17_small_left_checked, smallRow17_small_right_checked]
  rfl

def smallRow17 : List Nat := smallRow17_small
theorem smallRow17_checked : upperNeighbors PrimePairVertices.smallTree 17 = smallRow17 := smallRow17_small_checked

def smallRow19_block_000 : List Nat := [31, 79, 97, 163, 181, 433, 571, 577, 709, 727, 937, 991, 997, 1033, 1039, 1237, 1249, 1297, 1447, 1621, 1657, 1669, 1693, 1699, 1831, 1861]
set_option maxRecDepth 2048 in
theorem smallRow19_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 19 = smallRow19_block_000 := by decide

def smallRow19_block_001 : List Nat := [2053, 2113, 2161, 2239, 2251, 2539, 2677, 2791, 3727, 3793, 3847, 3943, 4057]
set_option maxRecDepth 2048 in
theorem smallRow19_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 19 = smallRow19_block_001 := by decide

def smallRow19_block_002 : List Nat := [4507, 4723, 4861, 5077, 5659, 5737, 5791, 5869, 6043, 6073]
set_option maxRecDepth 2048 in
theorem smallRow19_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 19 = smallRow19_block_002 := by decide

def smallRow19_block_003 : List Nat := [6271, 6661, 6709, 6907, 6961, 6991, 7159, 7297, 7507, 7573, 7699, 7741, 7759, 8179]
set_option maxRecDepth 2048 in
theorem smallRow19_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 19 = smallRow19_block_003 := by decide

def smallRow19_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow19_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 19 = smallRow19_block_004 := by decide

def smallRow19_small_left : List Nat := smallRow19_block_000 ++ smallRow19_block_001
theorem smallRow19_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 19 = smallRow19_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 19 ++ upperNeighbors PrimePairVertices.tree_block_001 19 = _
  rw [smallRow19_block_000_checked, smallRow19_block_001_checked]
  rfl

def smallRow19_small_right_right : List Nat := smallRow19_block_003 ++ smallRow19_block_004
theorem smallRow19_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 19 = smallRow19_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 19 ++ upperNeighbors PrimePairVertices.tree_block_004 19 = _
  rw [smallRow19_block_003_checked, smallRow19_block_004_checked]
  rfl

def smallRow19_small_right : List Nat := smallRow19_block_002 ++ smallRow19_small_right_right
theorem smallRow19_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 19 = smallRow19_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 19 ++ upperNeighbors PrimePairVertices.tree_small_right_right 19 = _
  rw [smallRow19_block_002_checked, smallRow19_small_right_right_checked]
  rfl

def smallRow19_small : List Nat := smallRow19_small_left ++ smallRow19_small_right
theorem smallRow19_small_checked : upperNeighbors PrimePairVertices.tree_small 19 = smallRow19_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 19 ++ upperNeighbors PrimePairVertices.tree_small_right 19 = _
  rw [smallRow19_small_left_checked, smallRow19_small_right_checked]
  rfl

def smallRow19 : List Nat := smallRow19_small
theorem smallRow19_checked : upperNeighbors PrimePairVertices.smallTree 19 = smallRow19 := smallRow19_small_checked

def smallRow23_block_000 : List Nat := [47, 89, 311, 509, 677, 743, 761, 773, 827, 1031, 1109, 1223, 1289, 1481, 1493, 1607, 1613, 1871, 2003]
set_option maxRecDepth 2048 in
theorem smallRow23_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 23 = smallRow23_block_000 := by decide

def smallRow23_block_001 : List Nat := [2129, 2333, 2357, 2591, 2633, 2699, 2741, 2753, 2819, 3083, 3407, 3557, 3617, 4007]
set_option maxRecDepth 2048 in
theorem smallRow23_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 23 = smallRow23_block_001 := by decide

def smallRow23_block_002 : List Nat := [4139, 4211, 4463, 4547, 4721, 4931, 5009, 5051, 5441, 5483, 5591, 5813, 5849, 5927, 6053]
set_option maxRecDepth 2048 in
theorem smallRow23_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 23 = smallRow23_block_002 := by decide

def smallRow23_block_003 : List Nat := [6323, 6329, 6449, 6563, 6659, 6917, 7151, 7331, 7487, 7547, 7607, 7673, 7691, 8009, 8093, 8171]
set_option maxRecDepth 2048 in
theorem smallRow23_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 23 = smallRow23_block_003 := by decide

def smallRow23_block_004 : List Nat := [8237, 8291]
set_option maxRecDepth 2048 in
theorem smallRow23_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 23 = smallRow23_block_004 := by decide

def smallRow23_small_left : List Nat := smallRow23_block_000 ++ smallRow23_block_001
theorem smallRow23_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 23 = smallRow23_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 23 ++ upperNeighbors PrimePairVertices.tree_block_001 23 = _
  rw [smallRow23_block_000_checked, smallRow23_block_001_checked]
  rfl

def smallRow23_small_right_right : List Nat := smallRow23_block_003 ++ smallRow23_block_004
theorem smallRow23_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 23 = smallRow23_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 23 ++ upperNeighbors PrimePairVertices.tree_block_004 23 = _
  rw [smallRow23_block_003_checked, smallRow23_block_004_checked]
  rfl

def smallRow23_small_right : List Nat := smallRow23_block_002 ++ smallRow23_small_right_right
theorem smallRow23_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 23 = smallRow23_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 23 ++ upperNeighbors PrimePairVertices.tree_small_right_right 23 = _
  rw [smallRow23_block_002_checked, smallRow23_small_right_right_checked]
  rfl

def smallRow23_small : List Nat := smallRow23_small_left ++ smallRow23_small_right
theorem smallRow23_small_checked : upperNeighbors PrimePairVertices.tree_small 23 = smallRow23_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 23 ++ upperNeighbors PrimePairVertices.tree_small_right 23 = _
  rw [smallRow23_small_left_checked, smallRow23_small_right_checked]
  rfl

def smallRow23 : List Nat := smallRow23_small
theorem smallRow23_checked : upperNeighbors PrimePairVertices.smallTree 23 = smallRow23 := smallRow23_small_checked

def smallRow29_block_000 : List Nat := [71, 137, 167, 179, 347, 383, 401, 569, 599, 683, 761, 881, 1901, 1997, 2027]
set_option maxRecDepth 2048 in
theorem smallRow29_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 29 = smallRow29_block_000 := by decide

def smallRow29_block_001 : List Nat := [2081, 2141, 2309, 2393, 2441, 2477, 2549, 2711, 2801, 2969, 3257, 3659, 3767, 3803, 3863, 4013]
set_option maxRecDepth 2048 in
theorem smallRow29_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 29 = smallRow29_block_001 := by decide

def smallRow29_block_002 : List Nat := [4157, 4241, 4337, 4397, 4523, 4673, 4787, 5039, 5081, 5237, 5441, 5861, 6047]
set_option maxRecDepth 2048 in
theorem smallRow29_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 29 = smallRow29_block_002 := by decide

def smallRow29_block_003 : List Nat := [6221, 6269, 6299, 6563, 6827, 6911, 6983, 7079, 7247, 7487, 7589, 7607, 7757, 7793, 7841, 7853]
set_option maxRecDepth 2048 in
theorem smallRow29_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 29 = smallRow29_block_003 := by decide

def smallRow29_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow29_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 29 = smallRow29_block_004 := by decide

def smallRow29_small_left : List Nat := smallRow29_block_000 ++ smallRow29_block_001
theorem smallRow29_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 29 = smallRow29_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 29 ++ upperNeighbors PrimePairVertices.tree_block_001 29 = _
  rw [smallRow29_block_000_checked, smallRow29_block_001_checked]
  rfl

def smallRow29_small_right_right : List Nat := smallRow29_block_003 ++ smallRow29_block_004
theorem smallRow29_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 29 = smallRow29_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 29 ++ upperNeighbors PrimePairVertices.tree_block_004 29 = _
  rw [smallRow29_block_003_checked, smallRow29_block_004_checked]
  rfl

def smallRow29_small_right : List Nat := smallRow29_block_002 ++ smallRow29_small_right_right
theorem smallRow29_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 29 = smallRow29_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 29 ++ upperNeighbors PrimePairVertices.tree_small_right_right 29 = _
  rw [smallRow29_block_002_checked, smallRow29_small_right_right_checked]
  rfl

def smallRow29_small : List Nat := smallRow29_small_left ++ smallRow29_small_right
theorem smallRow29_small_checked : upperNeighbors PrimePairVertices.tree_small 29 = smallRow29_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 29 ++ upperNeighbors PrimePairVertices.tree_small_right 29 = _
  rw [smallRow29_small_left_checked, smallRow29_small_right_checked]
  rfl

def smallRow29 : List Nat := smallRow29_small
theorem smallRow29_checked : upperNeighbors PrimePairVertices.smallTree 29 = smallRow29 := smallRow29_small_checked

end ProjectEulerSolutions.PrimePairCache
