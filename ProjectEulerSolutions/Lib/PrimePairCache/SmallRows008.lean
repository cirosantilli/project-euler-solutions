import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow331_block_000 : List Nat := [349, 577, 739, 883, 907, 937, 1303, 1531, 1669, 1831, 1933, 1999]
set_option maxRecDepth 2048 in
theorem smallRow331_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 331 = smallRow331_block_000 := by decide

def smallRow331_block_001 : List Nat := [2719, 2731, 2833, 3067, 3361, 3637, 3769]
set_option maxRecDepth 2048 in
theorem smallRow331_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 331 = smallRow331_block_001 := by decide

def smallRow331_block_002 : List Nat := [4261, 4513, 4639, 4651, 5107, 5407, 5779]
set_option maxRecDepth 2048 in
theorem smallRow331_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 331 = smallRow331_block_002 := by decide

def smallRow331_block_003 : List Nat := [6361, 6427, 6451, 6553, 6949, 7417, 7537]
set_option maxRecDepth 2048 in
theorem smallRow331_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 331 = smallRow331_block_003 := by decide

def smallRow331_block_004 : List Nat := [8221, 8377]
set_option maxRecDepth 2048 in
theorem smallRow331_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 331 = smallRow331_block_004 := by decide

def smallRow331_small_left : List Nat := smallRow331_block_000 ++ smallRow331_block_001
theorem smallRow331_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 331 = smallRow331_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 331 ++ upperNeighbors PrimePairVertices.tree_block_001 331 = _
  rw [smallRow331_block_000_checked, smallRow331_block_001_checked]
  rfl

def smallRow331_small_right_right : List Nat := smallRow331_block_003 ++ smallRow331_block_004
theorem smallRow331_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 331 = smallRow331_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 331 ++ upperNeighbors PrimePairVertices.tree_block_004 331 = _
  rw [smallRow331_block_003_checked, smallRow331_block_004_checked]
  rfl

def smallRow331_small_right : List Nat := smallRow331_block_002 ++ smallRow331_small_right_right
theorem smallRow331_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 331 = smallRow331_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 331 ++ upperNeighbors PrimePairVertices.tree_small_right_right 331 = _
  rw [smallRow331_block_002_checked, smallRow331_small_right_right_checked]
  rfl

def smallRow331_small : List Nat := smallRow331_small_left ++ smallRow331_small_right
theorem smallRow331_small_checked : upperNeighbors PrimePairVertices.tree_small 331 = smallRow331_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 331 ++ upperNeighbors PrimePairVertices.tree_small_right 331 = _
  rw [smallRow331_small_left_checked, smallRow331_small_right_checked]
  rfl

def smallRow331 : List Nat := smallRow331_small
theorem smallRow331_checked : upperNeighbors PrimePairVertices.smallTree 331 = smallRow331 := smallRow331_small_checked

def smallRow337_block_000 : List Nat := [349, 397, 607, 691, 919, 1153, 1297, 1867]
set_option maxRecDepth 2048 in
theorem smallRow337_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 337 = smallRow337_block_000 := by decide

def smallRow337_block_001 : List Nat := [2371, 2797, 3169, 3217, 3343, 3499]
set_option maxRecDepth 2048 in
theorem smallRow337_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 337 = smallRow337_block_001 := by decide

def smallRow337_block_002 : List Nat := [4507, 4729, 5431, 5857]
set_option maxRecDepth 2048 in
theorem smallRow337_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 337 = smallRow337_block_002 := by decide

def smallRow337_block_003 : List Nat := [6781, 6949, 7723]
set_option maxRecDepth 2048 in
theorem smallRow337_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 337 = smallRow337_block_003 := by decide

def smallRow337_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow337_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 337 = smallRow337_block_004 := by decide

def smallRow337_small_left : List Nat := smallRow337_block_000 ++ smallRow337_block_001
theorem smallRow337_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 337 = smallRow337_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 337 ++ upperNeighbors PrimePairVertices.tree_block_001 337 = _
  rw [smallRow337_block_000_checked, smallRow337_block_001_checked]
  rfl

def smallRow337_small_right_right : List Nat := smallRow337_block_003 ++ smallRow337_block_004
theorem smallRow337_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 337 = smallRow337_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 337 ++ upperNeighbors PrimePairVertices.tree_block_004 337 = _
  rw [smallRow337_block_003_checked, smallRow337_block_004_checked]
  rfl

def smallRow337_small_right : List Nat := smallRow337_block_002 ++ smallRow337_small_right_right
theorem smallRow337_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 337 = smallRow337_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 337 ++ upperNeighbors PrimePairVertices.tree_small_right_right 337 = _
  rw [smallRow337_block_002_checked, smallRow337_small_right_right_checked]
  rfl

def smallRow337_small : List Nat := smallRow337_small_left ++ smallRow337_small_right
theorem smallRow337_small_checked : upperNeighbors PrimePairVertices.tree_small 337 = smallRow337_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 337 ++ upperNeighbors PrimePairVertices.tree_small_right 337 = _
  rw [smallRow337_small_left_checked, smallRow337_small_right_checked]
  rfl

def smallRow337 : List Nat := smallRow337_small
theorem smallRow337_checked : upperNeighbors PrimePairVertices.smallTree 337 = smallRow337 := smallRow337_small_checked

def smallRow347_block_000 : List Nat := [401, 443, 983, 1031, 1301, 1697, 1847]
set_option maxRecDepth 2048 in
theorem smallRow347_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 347 = smallRow347_block_000 := by decide

def smallRow347_block_001 : List Nat := [2213, 3677, 3719, 3923]
set_option maxRecDepth 2048 in
theorem smallRow347_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 347 = smallRow347_block_001 := by decide

def smallRow347_block_002 : List Nat := [4517, 5477, 5861, 5981]
set_option maxRecDepth 2048 in
theorem smallRow347_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 347 = smallRow347_block_002 := by decide

def smallRow347_block_003 : List Nat := [6257, 6569, 6719, 6761, 7121, 7499, 7517, 7559, 7841, 8009]
set_option maxRecDepth 2048 in
theorem smallRow347_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 347 = smallRow347_block_003 := by decide

def smallRow347_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow347_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 347 = smallRow347_block_004 := by decide

def smallRow347_small_left : List Nat := smallRow347_block_000 ++ smallRow347_block_001
theorem smallRow347_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 347 = smallRow347_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 347 ++ upperNeighbors PrimePairVertices.tree_block_001 347 = _
  rw [smallRow347_block_000_checked, smallRow347_block_001_checked]
  rfl

def smallRow347_small_right_right : List Nat := smallRow347_block_003 ++ smallRow347_block_004
theorem smallRow347_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 347 = smallRow347_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 347 ++ upperNeighbors PrimePairVertices.tree_block_004 347 = _
  rw [smallRow347_block_003_checked, smallRow347_block_004_checked]
  rfl

def smallRow347_small_right : List Nat := smallRow347_block_002 ++ smallRow347_small_right_right
theorem smallRow347_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 347 = smallRow347_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 347 ++ upperNeighbors PrimePairVertices.tree_small_right_right 347 = _
  rw [smallRow347_block_002_checked, smallRow347_small_right_right_checked]
  rfl

def smallRow347_small : List Nat := smallRow347_small_left ++ smallRow347_small_right
theorem smallRow347_small_checked : upperNeighbors PrimePairVertices.tree_small 347 = smallRow347_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 347 ++ upperNeighbors PrimePairVertices.tree_small_right 347 = _
  rw [smallRow347_small_left_checked, smallRow347_small_right_checked]
  rfl

def smallRow347 : List Nat := smallRow347_small
theorem smallRow347_checked : upperNeighbors PrimePairVertices.smallTree 347 = smallRow347 := smallRow347_small_checked

def smallRow349_block_000 : List Nat := [373, 409, 499, 709, 829, 919, 967, 1087, 1123, 1237, 1291, 1429, 1447]
set_option maxRecDepth 2048 in
theorem smallRow349_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 349 = smallRow349_block_000 := by decide

def smallRow349_block_001 : List Nat := [2161, 2341, 2551, 2707, 2719, 2971, 3433]
set_option maxRecDepth 2048 in
theorem smallRow349_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 349 = smallRow349_block_001 := by decide

def smallRow349_block_002 : List Nat := [4159, 4357, 4657, 5581, 5743, 6007]
set_option maxRecDepth 2048 in
theorem smallRow349_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 349 = smallRow349_block_002 := by decide

def smallRow349_block_003 : List Nat := [6151, 6709]
set_option maxRecDepth 2048 in
theorem smallRow349_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 349 = smallRow349_block_003 := by decide

def smallRow349_block_004 : List Nat := [8293, 8377]
set_option maxRecDepth 2048 in
theorem smallRow349_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 349 = smallRow349_block_004 := by decide

def smallRow349_small_left : List Nat := smallRow349_block_000 ++ smallRow349_block_001
theorem smallRow349_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 349 = smallRow349_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 349 ++ upperNeighbors PrimePairVertices.tree_block_001 349 = _
  rw [smallRow349_block_000_checked, smallRow349_block_001_checked]
  rfl

def smallRow349_small_right_right : List Nat := smallRow349_block_003 ++ smallRow349_block_004
theorem smallRow349_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 349 = smallRow349_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 349 ++ upperNeighbors PrimePairVertices.tree_block_004 349 = _
  rw [smallRow349_block_003_checked, smallRow349_block_004_checked]
  rfl

def smallRow349_small_right : List Nat := smallRow349_block_002 ++ smallRow349_small_right_right
theorem smallRow349_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 349 = smallRow349_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 349 ++ upperNeighbors PrimePairVertices.tree_small_right_right 349 = _
  rw [smallRow349_block_002_checked, smallRow349_small_right_right_checked]
  rfl

def smallRow349_small : List Nat := smallRow349_small_left ++ smallRow349_small_right
theorem smallRow349_small_checked : upperNeighbors PrimePairVertices.tree_small 349 = smallRow349_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 349 ++ upperNeighbors PrimePairVertices.tree_small_right 349 = _
  rw [smallRow349_small_left_checked, smallRow349_small_right_checked]
  rfl

def smallRow349 : List Nat := smallRow349_small
theorem smallRow349_checked : upperNeighbors PrimePairVertices.smallTree 349 = smallRow349 := smallRow349_small_checked

def smallRow353_block_000 : List Nat := [359, 443, 797, 1433, 1481, 1499, 1877]
set_option maxRecDepth 2048 in
theorem smallRow353_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 353 = smallRow353_block_000 := by decide

def smallRow353_block_001 : List Nat := [2213, 2687, 2741, 3203, 3371, 3701, 3779]
set_option maxRecDepth 2048 in
theorem smallRow353_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 353 = smallRow353_block_001 := by decide

def smallRow353_block_002 : List Nat := [4337, 4397, 4493, 4673, 4919, 4967, 5087, 5573, 5849, 5879, 6131]
set_option maxRecDepth 2048 in
theorem smallRow353_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 353 = smallRow353_block_002 := by decide

def smallRow353_block_003 : List Nat := [6389, 6551, 6791, 7517, 7559, 7907]
set_option maxRecDepth 2048 in
theorem smallRow353_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 353 = smallRow353_block_003 := by decide

def smallRow353_block_004 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallRow353_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 353 = smallRow353_block_004 := by decide

def smallRow353_small_left : List Nat := smallRow353_block_000 ++ smallRow353_block_001
theorem smallRow353_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 353 = smallRow353_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 353 ++ upperNeighbors PrimePairVertices.tree_block_001 353 = _
  rw [smallRow353_block_000_checked, smallRow353_block_001_checked]
  rfl

def smallRow353_small_right_right : List Nat := smallRow353_block_003 ++ smallRow353_block_004
theorem smallRow353_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 353 = smallRow353_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 353 ++ upperNeighbors PrimePairVertices.tree_block_004 353 = _
  rw [smallRow353_block_003_checked, smallRow353_block_004_checked]
  rfl

def smallRow353_small_right : List Nat := smallRow353_block_002 ++ smallRow353_small_right_right
theorem smallRow353_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 353 = smallRow353_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 353 ++ upperNeighbors PrimePairVertices.tree_small_right_right 353 = _
  rw [smallRow353_block_002_checked, smallRow353_small_right_right_checked]
  rfl

def smallRow353_small : List Nat := smallRow353_small_left ++ smallRow353_small_right
theorem smallRow353_small_checked : upperNeighbors PrimePairVertices.tree_small 353 = smallRow353_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 353 ++ upperNeighbors PrimePairVertices.tree_small_right 353 = _
  rw [smallRow353_small_left_checked, smallRow353_small_right_checked]
  rfl

def smallRow353 : List Nat := smallRow353_small
theorem smallRow353_checked : upperNeighbors PrimePairVertices.smallTree 353 = smallRow353 := smallRow353_small_checked

def smallRow359_block_000 : List Nat := [509, 563, 599, 701, 911, 1487, 1499, 1667, 1949]
set_option maxRecDepth 2048 in
theorem smallRow359_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 359 = smallRow359_block_000 := by decide

def smallRow359_block_001 : List Nat := [2243, 2399, 2447, 2969, 3203, 3539, 3833, 3881]
set_option maxRecDepth 2048 in
theorem smallRow359_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 359 = smallRow359_block_001 := by decide

def smallRow359_block_002 : List Nat := [5591, 6143]
set_option maxRecDepth 2048 in
theorem smallRow359_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 359 = smallRow359_block_002 := by decide

def smallRow359_block_003 : List Nat := [6689, 7001, 7907, 8039]
set_option maxRecDepth 2048 in
theorem smallRow359_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 359 = smallRow359_block_003 := by decide

def smallRow359_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow359_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 359 = smallRow359_block_004 := by decide

def smallRow359_small_left : List Nat := smallRow359_block_000 ++ smallRow359_block_001
theorem smallRow359_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 359 = smallRow359_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 359 ++ upperNeighbors PrimePairVertices.tree_block_001 359 = _
  rw [smallRow359_block_000_checked, smallRow359_block_001_checked]
  rfl

def smallRow359_small_right_right : List Nat := smallRow359_block_003 ++ smallRow359_block_004
theorem smallRow359_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 359 = smallRow359_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 359 ++ upperNeighbors PrimePairVertices.tree_block_004 359 = _
  rw [smallRow359_block_003_checked, smallRow359_block_004_checked]
  rfl

def smallRow359_small_right : List Nat := smallRow359_block_002 ++ smallRow359_small_right_right
theorem smallRow359_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 359 = smallRow359_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 359 ++ upperNeighbors PrimePairVertices.tree_small_right_right 359 = _
  rw [smallRow359_block_002_checked, smallRow359_small_right_right_checked]
  rfl

def smallRow359_small : List Nat := smallRow359_small_left ++ smallRow359_small_right
theorem smallRow359_small_checked : upperNeighbors PrimePairVertices.tree_small 359 = smallRow359_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 359 ++ upperNeighbors PrimePairVertices.tree_small_right 359 = _
  rw [smallRow359_small_left_checked, smallRow359_small_right_checked]
  rfl

def smallRow359 : List Nat := smallRow359_small
theorem smallRow359_checked : upperNeighbors PrimePairVertices.smallTree 359 = smallRow359 := smallRow359_small_checked

def smallRow367_block_000 : List Nat := [457, 613, 751, 1021, 1117, 1531]
set_option maxRecDepth 2048 in
theorem smallRow367_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 367 = smallRow367_block_000 := by decide

def smallRow367_block_001 : List Nat := [2293, 2521, 2659, 2791, 3217, 3463, 3529]
set_option maxRecDepth 2048 in
theorem smallRow367_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 367 = smallRow367_block_001 := by decide

def smallRow367_block_002 : List Nat := [4339, 4597, 4663, 5059, 5563, 6091]
set_option maxRecDepth 2048 in
theorem smallRow367_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 367 = smallRow367_block_002 := by decide

def smallRow367_block_003 : List Nat := [7207, 7459, 7573, 7789]
set_option maxRecDepth 2048 in
theorem smallRow367_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 367 = smallRow367_block_003 := by decide

def smallRow367_block_004 : List Nat := [8317, 8377]
set_option maxRecDepth 2048 in
theorem smallRow367_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 367 = smallRow367_block_004 := by decide

def smallRow367_small_left : List Nat := smallRow367_block_000 ++ smallRow367_block_001
theorem smallRow367_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 367 = smallRow367_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 367 ++ upperNeighbors PrimePairVertices.tree_block_001 367 = _
  rw [smallRow367_block_000_checked, smallRow367_block_001_checked]
  rfl

def smallRow367_small_right_right : List Nat := smallRow367_block_003 ++ smallRow367_block_004
theorem smallRow367_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 367 = smallRow367_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 367 ++ upperNeighbors PrimePairVertices.tree_block_004 367 = _
  rw [smallRow367_block_003_checked, smallRow367_block_004_checked]
  rfl

def smallRow367_small_right : List Nat := smallRow367_block_002 ++ smallRow367_small_right_right
theorem smallRow367_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 367 = smallRow367_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 367 ++ upperNeighbors PrimePairVertices.tree_small_right_right 367 = _
  rw [smallRow367_block_002_checked, smallRow367_small_right_right_checked]
  rfl

def smallRow367_small : List Nat := smallRow367_small_left ++ smallRow367_small_right
theorem smallRow367_small_checked : upperNeighbors PrimePairVertices.tree_small 367 = smallRow367_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 367 ++ upperNeighbors PrimePairVertices.tree_small_right 367 = _
  rw [smallRow367_small_left_checked, smallRow367_small_right_checked]
  rfl

def smallRow367 : List Nat := smallRow367_small
theorem smallRow367_checked : upperNeighbors PrimePairVertices.smallTree 367 = smallRow367 := smallRow367_small_checked

def smallRow373_block_000 : List Nat := [661, 823, 859, 937, 1009, 1237, 1447, 1531, 1867]
set_option maxRecDepth 2048 in
theorem smallRow373_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 373 = smallRow373_block_000 := by decide

def smallRow373_block_001 : List Nat := [2161, 2467, 3967]
set_option maxRecDepth 2048 in
theorem smallRow373_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 373 = smallRow373_block_001 := by decide

def smallRow373_block_002 : List Nat := [4519, 4903, 4957, 5101, 5449, 5881, 6133]
set_option maxRecDepth 2048 in
theorem smallRow373_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 373 = smallRow373_block_002 := by decide

def smallRow373_block_003 : List Nat := [6163, 6793, 6823, 7057, 7351, 7417]
set_option maxRecDepth 2048 in
theorem smallRow373_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 373 = smallRow373_block_003 := by decide

def smallRow373_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow373_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 373 = smallRow373_block_004 := by decide

def smallRow373_small_left : List Nat := smallRow373_block_000 ++ smallRow373_block_001
theorem smallRow373_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 373 = smallRow373_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 373 ++ upperNeighbors PrimePairVertices.tree_block_001 373 = _
  rw [smallRow373_block_000_checked, smallRow373_block_001_checked]
  rfl

def smallRow373_small_right_right : List Nat := smallRow373_block_003 ++ smallRow373_block_004
theorem smallRow373_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 373 = smallRow373_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 373 ++ upperNeighbors PrimePairVertices.tree_block_004 373 = _
  rw [smallRow373_block_003_checked, smallRow373_block_004_checked]
  rfl

def smallRow373_small_right : List Nat := smallRow373_block_002 ++ smallRow373_small_right_right
theorem smallRow373_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 373 = smallRow373_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 373 ++ upperNeighbors PrimePairVertices.tree_small_right_right 373 = _
  rw [smallRow373_block_002_checked, smallRow373_small_right_right_checked]
  rfl

def smallRow373_small : List Nat := smallRow373_small_left ++ smallRow373_small_right
theorem smallRow373_small_checked : upperNeighbors PrimePairVertices.tree_small 373 = smallRow373_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 373 ++ upperNeighbors PrimePairVertices.tree_small_right 373 = _
  rw [smallRow373_small_left_checked, smallRow373_small_right_checked]
  rfl

def smallRow373 : List Nat := smallRow373_small
theorem smallRow373_checked : upperNeighbors PrimePairVertices.smallTree 373 = smallRow373 := smallRow373_small_checked

end ProjectEulerSolutions.PrimePairCache
