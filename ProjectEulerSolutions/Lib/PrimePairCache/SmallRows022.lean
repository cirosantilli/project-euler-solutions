import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1063_block_000 : List Nat := [1471, 1531, 1609]
set_option maxRecDepth 2048 in
theorem smallRow1063_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1063 = smallRow1063_block_000 := by decide

def smallRow1063_block_001 : List Nat := [2551, 2719, 2749, 2953, 3001, 3067, 3307, 3463, 3499, 3547, 3919]
set_option maxRecDepth 2048 in
theorem smallRow1063_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1063 = smallRow1063_block_001 := by decide

def smallRow1063_block_002 : List Nat := [4339, 5197, 5701]
set_option maxRecDepth 2048 in
theorem smallRow1063_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1063 = smallRow1063_block_002 := by decide

def smallRow1063_block_003 : List Nat := [6247, 6793, 7213, 7507]
set_option maxRecDepth 2048 in
theorem smallRow1063_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1063 = smallRow1063_block_003 := by decide

def smallRow1063_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1063_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1063 = smallRow1063_block_004 := by decide

def smallRow1063_small_left : List Nat := smallRow1063_block_000 ++ smallRow1063_block_001
theorem smallRow1063_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1063 = smallRow1063_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1063 ++ upperNeighbors PrimePairVertices.tree_block_001 1063 = _
  rw [smallRow1063_block_000_checked, smallRow1063_block_001_checked]
  rfl

def smallRow1063_small_right_right : List Nat := smallRow1063_block_003 ++ smallRow1063_block_004
theorem smallRow1063_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1063 = smallRow1063_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1063 ++ upperNeighbors PrimePairVertices.tree_block_004 1063 = _
  rw [smallRow1063_block_003_checked, smallRow1063_block_004_checked]
  rfl

def smallRow1063_small_right : List Nat := smallRow1063_block_002 ++ smallRow1063_small_right_right
theorem smallRow1063_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1063 = smallRow1063_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1063 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1063 = _
  rw [smallRow1063_block_002_checked, smallRow1063_small_right_right_checked]
  rfl

def smallRow1063_small : List Nat := smallRow1063_small_left ++ smallRow1063_small_right
theorem smallRow1063_small_checked : upperNeighbors PrimePairVertices.tree_small 1063 = smallRow1063_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1063 ++ upperNeighbors PrimePairVertices.tree_small_right 1063 = _
  rw [smallRow1063_small_left_checked, smallRow1063_small_right_checked]
  rfl

def smallRow1063 : List Nat := smallRow1063_small
theorem smallRow1063_checked : upperNeighbors PrimePairVertices.smallTree 1063 = smallRow1063 := smallRow1063_small_checked

def smallRow1069_block_000 : List Nat := [1321, 1609]
set_option maxRecDepth 2048 in
theorem smallRow1069_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1069 = smallRow1069_block_000 := by decide

def smallRow1069_block_001 : List Nat := [2113, 2203, 2281, 2659, 3061, 3313]
set_option maxRecDepth 2048 in
theorem smallRow1069_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1069 = smallRow1069_block_001 := by decide

def smallRow1069_block_002 : List Nat := [4219, 4261, 4909, 5059, 5683]
set_option maxRecDepth 2048 in
theorem smallRow1069_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1069 = smallRow1069_block_002 := by decide

def smallRow1069_block_003 : List Nat := [6337, 6529, 6841, 6883, 7069, 7993, 8161]
set_option maxRecDepth 2048 in
theorem smallRow1069_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1069 = smallRow1069_block_003 := by decide

def smallRow1069_block_004 : List Nat := [8221, 8329]
set_option maxRecDepth 2048 in
theorem smallRow1069_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1069 = smallRow1069_block_004 := by decide

def smallRow1069_small_left : List Nat := smallRow1069_block_000 ++ smallRow1069_block_001
theorem smallRow1069_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1069 = smallRow1069_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1069 ++ upperNeighbors PrimePairVertices.tree_block_001 1069 = _
  rw [smallRow1069_block_000_checked, smallRow1069_block_001_checked]
  rfl

def smallRow1069_small_right_right : List Nat := smallRow1069_block_003 ++ smallRow1069_block_004
theorem smallRow1069_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1069 = smallRow1069_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1069 ++ upperNeighbors PrimePairVertices.tree_block_004 1069 = _
  rw [smallRow1069_block_003_checked, smallRow1069_block_004_checked]
  rfl

def smallRow1069_small_right : List Nat := smallRow1069_block_002 ++ smallRow1069_small_right_right
theorem smallRow1069_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1069 = smallRow1069_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1069 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1069 = _
  rw [smallRow1069_block_002_checked, smallRow1069_small_right_right_checked]
  rfl

def smallRow1069_small : List Nat := smallRow1069_small_left ++ smallRow1069_small_right
theorem smallRow1069_small_checked : upperNeighbors PrimePairVertices.tree_small 1069 = smallRow1069_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1069 ++ upperNeighbors PrimePairVertices.tree_small_right 1069 = _
  rw [smallRow1069_small_left_checked, smallRow1069_small_right_checked]
  rfl

def smallRow1069 : List Nat := smallRow1069_small
theorem smallRow1069_checked : upperNeighbors PrimePairVertices.smallTree 1069 = smallRow1069 := smallRow1069_small_checked

def smallRow1087_block_000 : List Nat := [1117, 1453]
set_option maxRecDepth 2048 in
theorem smallRow1087_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1087 = smallRow1087_block_000 := by decide

def smallRow1087_block_001 : List Nat := [2137, 2179, 2833, 3271, 3847]
set_option maxRecDepth 2048 in
theorem smallRow1087_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1087 = smallRow1087_block_001 := by decide

def smallRow1087_block_002 : List Nat := [4447, 4789, 4957, 5107, 5413, 5569, 6091]
set_option maxRecDepth 2048 in
theorem smallRow1087_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1087 = smallRow1087_block_002 := by decide

def smallRow1087_block_003 : List Nat := [6451, 6763, 6967]
set_option maxRecDepth 2048 in
theorem smallRow1087_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1087 = smallRow1087_block_003 := by decide

def smallRow1087_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1087_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1087 = smallRow1087_block_004 := by decide

def smallRow1087_small_left : List Nat := smallRow1087_block_000 ++ smallRow1087_block_001
theorem smallRow1087_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1087 = smallRow1087_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1087 ++ upperNeighbors PrimePairVertices.tree_block_001 1087 = _
  rw [smallRow1087_block_000_checked, smallRow1087_block_001_checked]
  rfl

def smallRow1087_small_right_right : List Nat := smallRow1087_block_003 ++ smallRow1087_block_004
theorem smallRow1087_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1087 = smallRow1087_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1087 ++ upperNeighbors PrimePairVertices.tree_block_004 1087 = _
  rw [smallRow1087_block_003_checked, smallRow1087_block_004_checked]
  rfl

def smallRow1087_small_right : List Nat := smallRow1087_block_002 ++ smallRow1087_small_right_right
theorem smallRow1087_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1087 = smallRow1087_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1087 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1087 = _
  rw [smallRow1087_block_002_checked, smallRow1087_small_right_right_checked]
  rfl

def smallRow1087_small : List Nat := smallRow1087_small_left ++ smallRow1087_small_right
theorem smallRow1087_small_checked : upperNeighbors PrimePairVertices.tree_small 1087 = smallRow1087_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1087 ++ upperNeighbors PrimePairVertices.tree_small_right 1087 = _
  rw [smallRow1087_small_left_checked, smallRow1087_small_right_checked]
  rfl

def smallRow1087 : List Nat := smallRow1087_small
theorem smallRow1087_checked : upperNeighbors PrimePairVertices.smallTree 1087 = smallRow1087 := smallRow1087_small_checked

def smallRow1091_block_000 : List Nat := [1319, 1697, 1847, 1889, 1949]
set_option maxRecDepth 2048 in
theorem smallRow1091_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1091 = smallRow1091_block_000 := by decide

def smallRow1091_block_001 : List Nat := [2267, 2633, 3041, 3527]
set_option maxRecDepth 2048 in
theorem smallRow1091_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1091 = smallRow1091_block_001 := by decide

def smallRow1091_block_002 : List Nat := [4373, 4919, 4931]
set_option maxRecDepth 2048 in
theorem smallRow1091_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1091 = smallRow1091_block_002 := by decide

def smallRow1091_block_003 : List Nat := [7307, 8081, 8111]
set_option maxRecDepth 2048 in
theorem smallRow1091_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1091 = smallRow1091_block_003 := by decide

def smallRow1091_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1091_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1091 = smallRow1091_block_004 := by decide

def smallRow1091_small_left : List Nat := smallRow1091_block_000 ++ smallRow1091_block_001
theorem smallRow1091_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1091 = smallRow1091_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1091 ++ upperNeighbors PrimePairVertices.tree_block_001 1091 = _
  rw [smallRow1091_block_000_checked, smallRow1091_block_001_checked]
  rfl

def smallRow1091_small_right_right : List Nat := smallRow1091_block_003 ++ smallRow1091_block_004
theorem smallRow1091_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1091 = smallRow1091_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1091 ++ upperNeighbors PrimePairVertices.tree_block_004 1091 = _
  rw [smallRow1091_block_003_checked, smallRow1091_block_004_checked]
  rfl

def smallRow1091_small_right : List Nat := smallRow1091_block_002 ++ smallRow1091_small_right_right
theorem smallRow1091_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1091 = smallRow1091_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1091 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1091 = _
  rw [smallRow1091_block_002_checked, smallRow1091_small_right_right_checked]
  rfl

def smallRow1091_small : List Nat := smallRow1091_small_left ++ smallRow1091_small_right
theorem smallRow1091_small_checked : upperNeighbors PrimePairVertices.tree_small 1091 = smallRow1091_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1091 ++ upperNeighbors PrimePairVertices.tree_small_right 1091 = _
  rw [smallRow1091_small_left_checked, smallRow1091_small_right_checked]
  rfl

def smallRow1091 : List Nat := smallRow1091_small
theorem smallRow1091_checked : upperNeighbors PrimePairVertices.smallTree 1091 = smallRow1091 := smallRow1091_small_checked

def smallRow1093_block_000 : List Nat := [1471, 1549, 1861]
set_option maxRecDepth 2048 in
theorem smallRow1093_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1093 = smallRow1093_block_000 := by decide

def smallRow1093_block_001 : List Nat := [2437, 3061, 3331]
set_option maxRecDepth 2048 in
theorem smallRow1093_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1093 = smallRow1093_block_001 := by decide

def smallRow1093_block_002 : List Nat := [4201, 4243, 4447, 4603, 4813, 5413, 5569, 5779]
set_option maxRecDepth 2048 in
theorem smallRow1093_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1093 = smallRow1093_block_002 := by decide

def smallRow1093_block_003 : List Nat := [6427, 6481, 6553, 6619, 7057, 7309, 7741]
set_option maxRecDepth 2048 in
theorem smallRow1093_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1093 = smallRow1093_block_003 := by decide

def smallRow1093_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1093_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1093 = smallRow1093_block_004 := by decide

def smallRow1093_small_left : List Nat := smallRow1093_block_000 ++ smallRow1093_block_001
theorem smallRow1093_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1093 = smallRow1093_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1093 ++ upperNeighbors PrimePairVertices.tree_block_001 1093 = _
  rw [smallRow1093_block_000_checked, smallRow1093_block_001_checked]
  rfl

def smallRow1093_small_right_right : List Nat := smallRow1093_block_003 ++ smallRow1093_block_004
theorem smallRow1093_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1093 = smallRow1093_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1093 ++ upperNeighbors PrimePairVertices.tree_block_004 1093 = _
  rw [smallRow1093_block_003_checked, smallRow1093_block_004_checked]
  rfl

def smallRow1093_small_right : List Nat := smallRow1093_block_002 ++ smallRow1093_small_right_right
theorem smallRow1093_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1093 = smallRow1093_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1093 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1093 = _
  rw [smallRow1093_block_002_checked, smallRow1093_small_right_right_checked]
  rfl

def smallRow1093_small : List Nat := smallRow1093_small_left ++ smallRow1093_small_right
theorem smallRow1093_small_checked : upperNeighbors PrimePairVertices.tree_small 1093 = smallRow1093_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1093 ++ upperNeighbors PrimePairVertices.tree_small_right 1093 = _
  rw [smallRow1093_small_left_checked, smallRow1093_small_right_checked]
  rfl

def smallRow1093 : List Nat := smallRow1093_small
theorem smallRow1093_checked : upperNeighbors PrimePairVertices.smallTree 1093 = smallRow1093 := smallRow1093_small_checked

def smallRow1097_block_000 : List Nat := [1637]
set_option maxRecDepth 2048 in
theorem smallRow1097_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1097 = smallRow1097_block_000 := by decide

def smallRow1097_block_001 : List Nat := [2393, 2999, 3251, 3533, 3593]
set_option maxRecDepth 2048 in
theorem smallRow1097_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1097 = smallRow1097_block_001 := by decide

def smallRow1097_block_002 : List Nat := [4283, 4409, 4493, 4637, 4751, 5087, 5651, 6113]
set_option maxRecDepth 2048 in
theorem smallRow1097_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1097 = smallRow1097_block_002 := by decide

def smallRow1097_block_003 : List Nat := [6323, 6473, 7517, 7559, 7727, 8087]
set_option maxRecDepth 2048 in
theorem smallRow1097_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1097 = smallRow1097_block_003 := by decide

def smallRow1097_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1097_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1097 = smallRow1097_block_004 := by decide

def smallRow1097_small_left : List Nat := smallRow1097_block_000 ++ smallRow1097_block_001
theorem smallRow1097_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1097 = smallRow1097_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1097 ++ upperNeighbors PrimePairVertices.tree_block_001 1097 = _
  rw [smallRow1097_block_000_checked, smallRow1097_block_001_checked]
  rfl

def smallRow1097_small_right_right : List Nat := smallRow1097_block_003 ++ smallRow1097_block_004
theorem smallRow1097_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1097 = smallRow1097_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1097 ++ upperNeighbors PrimePairVertices.tree_block_004 1097 = _
  rw [smallRow1097_block_003_checked, smallRow1097_block_004_checked]
  rfl

def smallRow1097_small_right : List Nat := smallRow1097_block_002 ++ smallRow1097_small_right_right
theorem smallRow1097_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1097 = smallRow1097_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1097 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1097 = _
  rw [smallRow1097_block_002_checked, smallRow1097_small_right_right_checked]
  rfl

def smallRow1097_small : List Nat := smallRow1097_small_left ++ smallRow1097_small_right
theorem smallRow1097_small_checked : upperNeighbors PrimePairVertices.tree_small 1097 = smallRow1097_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1097 ++ upperNeighbors PrimePairVertices.tree_small_right 1097 = _
  rw [smallRow1097_small_left_checked, smallRow1097_small_right_checked]
  rfl

def smallRow1097 : List Nat := smallRow1097_small
theorem smallRow1097_checked : upperNeighbors PrimePairVertices.smallTree 1097 = smallRow1097 := smallRow1097_small_checked

def smallRow1103_block_000 : List Nat := [1187, 1373, 1733, 2039]
set_option maxRecDepth 2048 in
theorem smallRow1103_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1103 = smallRow1103_block_000 := by decide

def smallRow1103_block_001 : List Nat := [2081, 2909, 2927, 3041, 3299, 3557]
set_option maxRecDepth 2048 in
theorem smallRow1103_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1103 = smallRow1103_block_001 := by decide

def smallRow1103_block_002 : List Nat := [5237, 5807, 5813, 5987, 6029, 6143]
set_option maxRecDepth 2048 in
theorem smallRow1103_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1103 = smallRow1103_block_002 := by decide

def smallRow1103_block_003 : List Nat := [6317, 6329, 6983, 7529, 8123]
set_option maxRecDepth 2048 in
theorem smallRow1103_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1103 = smallRow1103_block_003 := by decide

def smallRow1103_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1103_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1103 = smallRow1103_block_004 := by decide

def smallRow1103_small_left : List Nat := smallRow1103_block_000 ++ smallRow1103_block_001
theorem smallRow1103_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1103 = smallRow1103_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1103 ++ upperNeighbors PrimePairVertices.tree_block_001 1103 = _
  rw [smallRow1103_block_000_checked, smallRow1103_block_001_checked]
  rfl

def smallRow1103_small_right_right : List Nat := smallRow1103_block_003 ++ smallRow1103_block_004
theorem smallRow1103_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1103 = smallRow1103_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1103 ++ upperNeighbors PrimePairVertices.tree_block_004 1103 = _
  rw [smallRow1103_block_003_checked, smallRow1103_block_004_checked]
  rfl

def smallRow1103_small_right : List Nat := smallRow1103_block_002 ++ smallRow1103_small_right_right
theorem smallRow1103_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1103 = smallRow1103_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1103 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1103 = _
  rw [smallRow1103_block_002_checked, smallRow1103_small_right_right_checked]
  rfl

def smallRow1103_small : List Nat := smallRow1103_small_left ++ smallRow1103_small_right
theorem smallRow1103_small_checked : upperNeighbors PrimePairVertices.tree_small 1103 = smallRow1103_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1103 ++ upperNeighbors PrimePairVertices.tree_small_right 1103 = _
  rw [smallRow1103_small_left_checked, smallRow1103_small_right_checked]
  rfl

def smallRow1103 : List Nat := smallRow1103_small
theorem smallRow1103_checked : upperNeighbors PrimePairVertices.smallTree 1103 = smallRow1103 := smallRow1103_small_checked

def smallRow1109_block_000 : List Nat := [1181, 1229, 1889, 2027]
set_option maxRecDepth 2048 in
theorem smallRow1109_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1109 = smallRow1109_block_000 := by decide

def smallRow1109_block_001 : List Nat := [2069, 2441, 2999, 3209, 3491, 3917, 3923]
set_option maxRecDepth 2048 in
theorem smallRow1109_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1109 = smallRow1109_block_001 := by decide

def smallRow1109_block_002 : List Nat := [4289, 4493, 4871, 5501]
set_option maxRecDepth 2048 in
theorem smallRow1109_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1109 = smallRow1109_block_002 := by decide

def smallRow1109_block_003 : List Nat := [6521, 6719]
set_option maxRecDepth 2048 in
theorem smallRow1109_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1109 = smallRow1109_block_003 := by decide

def smallRow1109_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow1109_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1109 = smallRow1109_block_004 := by decide

def smallRow1109_small_left : List Nat := smallRow1109_block_000 ++ smallRow1109_block_001
theorem smallRow1109_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1109 = smallRow1109_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1109 ++ upperNeighbors PrimePairVertices.tree_block_001 1109 = _
  rw [smallRow1109_block_000_checked, smallRow1109_block_001_checked]
  rfl

def smallRow1109_small_right_right : List Nat := smallRow1109_block_003 ++ smallRow1109_block_004
theorem smallRow1109_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1109 = smallRow1109_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1109 ++ upperNeighbors PrimePairVertices.tree_block_004 1109 = _
  rw [smallRow1109_block_003_checked, smallRow1109_block_004_checked]
  rfl

def smallRow1109_small_right : List Nat := smallRow1109_block_002 ++ smallRow1109_small_right_right
theorem smallRow1109_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1109 = smallRow1109_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1109 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1109 = _
  rw [smallRow1109_block_002_checked, smallRow1109_small_right_right_checked]
  rfl

def smallRow1109_small : List Nat := smallRow1109_small_left ++ smallRow1109_small_right
theorem smallRow1109_small_checked : upperNeighbors PrimePairVertices.tree_small 1109 = smallRow1109_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1109 ++ upperNeighbors PrimePairVertices.tree_small_right 1109 = _
  rw [smallRow1109_small_left_checked, smallRow1109_small_right_checked]
  rfl

def smallRow1109 : List Nat := smallRow1109_small
theorem smallRow1109_checked : upperNeighbors PrimePairVertices.smallTree 1109 = smallRow1109 := smallRow1109_small_checked

end ProjectEulerSolutions.PrimePairCache
