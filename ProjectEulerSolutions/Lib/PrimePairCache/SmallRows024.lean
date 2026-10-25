import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1187_block_000 : List Nat := [1361, 1451]
set_option maxRecDepth 2048 in
theorem smallRow1187_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1187 = smallRow1187_block_000 := by decide

def smallRow1187_block_001 : List Nat := [2879, 4001]
set_option maxRecDepth 2048 in
theorem smallRow1187_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1187 = smallRow1187_block_001 := by decide

def smallRow1187_block_002 : List Nat := [4523, 4649, 4937, 5477, 5861]
set_option maxRecDepth 2048 in
theorem smallRow1187_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1187 = smallRow1187_block_002 := by decide

def smallRow1187_block_003 : List Nat := [6203, 6569, 6701, 6833, 6857, 6983, 7079, 7253, 8093, 8117]
set_option maxRecDepth 2048 in
theorem smallRow1187_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1187 = smallRow1187_block_003 := by decide

def smallRow1187_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1187_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1187 = smallRow1187_block_004 := by decide

def smallRow1187_small_left : List Nat := smallRow1187_block_000 ++ smallRow1187_block_001
theorem smallRow1187_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1187 = smallRow1187_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1187 ++ upperNeighbors PrimePairVertices.tree_block_001 1187 = _
  rw [smallRow1187_block_000_checked, smallRow1187_block_001_checked]
  rfl

def smallRow1187_small_right_right : List Nat := smallRow1187_block_003 ++ smallRow1187_block_004
theorem smallRow1187_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1187 = smallRow1187_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1187 ++ upperNeighbors PrimePairVertices.tree_block_004 1187 = _
  rw [smallRow1187_block_003_checked, smallRow1187_block_004_checked]
  rfl

def smallRow1187_small_right : List Nat := smallRow1187_block_002 ++ smallRow1187_small_right_right
theorem smallRow1187_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1187 = smallRow1187_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1187 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1187 = _
  rw [smallRow1187_block_002_checked, smallRow1187_small_right_right_checked]
  rfl

def smallRow1187_small : List Nat := smallRow1187_small_left ++ smallRow1187_small_right
theorem smallRow1187_small_checked : upperNeighbors PrimePairVertices.tree_small 1187 = smallRow1187_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1187 ++ upperNeighbors PrimePairVertices.tree_small_right 1187 = _
  rw [smallRow1187_small_left_checked, smallRow1187_small_right_checked]
  rfl

def smallRow1187 : List Nat := smallRow1187_small
theorem smallRow1187_checked : upperNeighbors PrimePairVertices.smallTree 1187 = smallRow1187 := smallRow1187_small_checked

def smallRow1193_block_000 : List Nat := [1553, 1889, 1901]
set_option maxRecDepth 2048 in
theorem smallRow1193_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1193 = smallRow1193_block_000 := by decide

def smallRow1193_block_001 : List Nat := [2153, 2441, 2777, 3671, 3923]
set_option maxRecDepth 2048 in
theorem smallRow1193_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1193 = smallRow1193_block_001 := by decide

def smallRow1193_block_002 : List Nat := [4133, 4931, 5009, 5507, 5519, 5801, 6131]
set_option maxRecDepth 2048 in
theorem smallRow1193_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1193 = smallRow1193_block_002 := by decide

def smallRow1193_block_003 : List Nat := [6689, 6779, 7103, 7193, 7559, 7829, 8039]
set_option maxRecDepth 2048 in
theorem smallRow1193_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1193 = smallRow1193_block_003 := by decide

def smallRow1193_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1193_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1193 = smallRow1193_block_004 := by decide

def smallRow1193_small_left : List Nat := smallRow1193_block_000 ++ smallRow1193_block_001
theorem smallRow1193_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1193 = smallRow1193_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1193 ++ upperNeighbors PrimePairVertices.tree_block_001 1193 = _
  rw [smallRow1193_block_000_checked, smallRow1193_block_001_checked]
  rfl

def smallRow1193_small_right_right : List Nat := smallRow1193_block_003 ++ smallRow1193_block_004
theorem smallRow1193_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1193 = smallRow1193_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1193 ++ upperNeighbors PrimePairVertices.tree_block_004 1193 = _
  rw [smallRow1193_block_003_checked, smallRow1193_block_004_checked]
  rfl

def smallRow1193_small_right : List Nat := smallRow1193_block_002 ++ smallRow1193_small_right_right
theorem smallRow1193_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1193 = smallRow1193_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1193 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1193 = _
  rw [smallRow1193_block_002_checked, smallRow1193_small_right_right_checked]
  rfl

def smallRow1193_small : List Nat := smallRow1193_small_left ++ smallRow1193_small_right
theorem smallRow1193_small_checked : upperNeighbors PrimePairVertices.tree_small 1193 = smallRow1193_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1193 ++ upperNeighbors PrimePairVertices.tree_small_right 1193 = _
  rw [smallRow1193_small_left_checked, smallRow1193_small_right_checked]
  rfl

def smallRow1193 : List Nat := smallRow1193_small
theorem smallRow1193_checked : upperNeighbors PrimePairVertices.smallTree 1193 = smallRow1193 := smallRow1193_small_checked

def smallRow1201_block_000 : List Nat := [1303, 1459, 1621, 1873, 2017]
set_option maxRecDepth 2048 in
theorem smallRow1201_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1201 = smallRow1201_block_000 := by decide

def smallRow1201_block_001 : List Nat := [2137, 2551, 3229, 3637, 3769, 3931]
set_option maxRecDepth 2048 in
theorem smallRow1201_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1201 = smallRow1201_block_001 := by decide

def smallRow1201_block_002 : List Nat := [4201, 4663, 4861, 5167, 5281, 5953]
set_option maxRecDepth 2048 in
theorem smallRow1201_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1201 = smallRow1201_block_002 := by decide

def smallRow1201_block_003 : List Nat := [6337, 6577, 7207, 7459]
set_option maxRecDepth 2048 in
theorem smallRow1201_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1201 = smallRow1201_block_003 := by decide

def smallRow1201_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1201_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1201 = smallRow1201_block_004 := by decide

def smallRow1201_small_left : List Nat := smallRow1201_block_000 ++ smallRow1201_block_001
theorem smallRow1201_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1201 = smallRow1201_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1201 ++ upperNeighbors PrimePairVertices.tree_block_001 1201 = _
  rw [smallRow1201_block_000_checked, smallRow1201_block_001_checked]
  rfl

def smallRow1201_small_right_right : List Nat := smallRow1201_block_003 ++ smallRow1201_block_004
theorem smallRow1201_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1201 = smallRow1201_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1201 ++ upperNeighbors PrimePairVertices.tree_block_004 1201 = _
  rw [smallRow1201_block_003_checked, smallRow1201_block_004_checked]
  rfl

def smallRow1201_small_right : List Nat := smallRow1201_block_002 ++ smallRow1201_small_right_right
theorem smallRow1201_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1201 = smallRow1201_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1201 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1201 = _
  rw [smallRow1201_block_002_checked, smallRow1201_small_right_right_checked]
  rfl

def smallRow1201_small : List Nat := smallRow1201_small_left ++ smallRow1201_small_right
theorem smallRow1201_small_checked : upperNeighbors PrimePairVertices.tree_small 1201 = smallRow1201_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1201 ++ upperNeighbors PrimePairVertices.tree_small_right 1201 = _
  rw [smallRow1201_small_left_checked, smallRow1201_small_right_checked]
  rfl

def smallRow1201 : List Nat := smallRow1201_small
theorem smallRow1201_checked : upperNeighbors PrimePairVertices.smallTree 1201 = smallRow1201 := smallRow1201_small_checked

def smallRow1213_block_000 : List Nat := [1297, 1621, 1789, 2011]
set_option maxRecDepth 2048 in
theorem smallRow1213_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1213 = smallRow1213_block_000 := by decide

def smallRow1213_block_001 : List Nat := [2053, 2371, 2797, 2851, 3061, 3301, 3463]
set_option maxRecDepth 2048 in
theorem smallRow1213_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1213 = smallRow1213_block_001 := by decide

def smallRow1213_block_002 : List Nat := [4201, 5443, 5479, 6043, 6073]
set_option maxRecDepth 2048 in
theorem smallRow1213_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1213 = smallRow1213_block_002 := by decide

def smallRow1213_block_003 : List Nat := [6151, 6211, 6547, 7243]
set_option maxRecDepth 2048 in
theorem smallRow1213_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1213 = smallRow1213_block_003 := by decide

def smallRow1213_block_004 : List Nat := [8209, 8353]
set_option maxRecDepth 2048 in
theorem smallRow1213_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1213 = smallRow1213_block_004 := by decide

def smallRow1213_small_left : List Nat := smallRow1213_block_000 ++ smallRow1213_block_001
theorem smallRow1213_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1213 = smallRow1213_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1213 ++ upperNeighbors PrimePairVertices.tree_block_001 1213 = _
  rw [smallRow1213_block_000_checked, smallRow1213_block_001_checked]
  rfl

def smallRow1213_small_right_right : List Nat := smallRow1213_block_003 ++ smallRow1213_block_004
theorem smallRow1213_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1213 = smallRow1213_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1213 ++ upperNeighbors PrimePairVertices.tree_block_004 1213 = _
  rw [smallRow1213_block_003_checked, smallRow1213_block_004_checked]
  rfl

def smallRow1213_small_right : List Nat := smallRow1213_block_002 ++ smallRow1213_small_right_right
theorem smallRow1213_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1213 = smallRow1213_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1213 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1213 = _
  rw [smallRow1213_block_002_checked, smallRow1213_small_right_right_checked]
  rfl

def smallRow1213_small : List Nat := smallRow1213_small_left ++ smallRow1213_small_right
theorem smallRow1213_small_checked : upperNeighbors PrimePairVertices.tree_small 1213 = smallRow1213_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1213 ++ upperNeighbors PrimePairVertices.tree_small_right 1213 = _
  rw [smallRow1213_small_left_checked, smallRow1213_small_right_checked]
  rfl

def smallRow1213 : List Nat := smallRow1213_small
theorem smallRow1213_checked : upperNeighbors PrimePairVertices.smallTree 1213 = smallRow1213 := smallRow1213_small_checked

def smallRow1217_block_000 : List Nat := [1361, 1427, 1433, 1583, 1721, 2003]
set_option maxRecDepth 2048 in
theorem smallRow1217_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1217 = smallRow1217_block_000 := by decide

def smallRow1217_block_001 : List Nat := [2087, 2273, 2309, 2393, 2423, 2477, 2591, 2843, 2903, 3671]
set_option maxRecDepth 2048 in
theorem smallRow1217_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1217 = smallRow1217_block_001 := by decide

def smallRow1217_block_002 : List Nat := [4931, 5003, 5081, 5153, 5393, 5477, 5573, 5879]
set_option maxRecDepth 2048 in
theorem smallRow1217_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1217 = smallRow1217_block_002 := by decide

def smallRow1217_block_003 : List Nat := [6173, 7001, 7019, 7817, 8093]
set_option maxRecDepth 2048 in
theorem smallRow1217_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1217 = smallRow1217_block_003 := by decide

def smallRow1217_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1217_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1217 = smallRow1217_block_004 := by decide

def smallRow1217_small_left : List Nat := smallRow1217_block_000 ++ smallRow1217_block_001
theorem smallRow1217_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1217 = smallRow1217_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1217 ++ upperNeighbors PrimePairVertices.tree_block_001 1217 = _
  rw [smallRow1217_block_000_checked, smallRow1217_block_001_checked]
  rfl

def smallRow1217_small_right_right : List Nat := smallRow1217_block_003 ++ smallRow1217_block_004
theorem smallRow1217_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1217 = smallRow1217_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1217 ++ upperNeighbors PrimePairVertices.tree_block_004 1217 = _
  rw [smallRow1217_block_003_checked, smallRow1217_block_004_checked]
  rfl

def smallRow1217_small_right : List Nat := smallRow1217_block_002 ++ smallRow1217_small_right_right
theorem smallRow1217_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1217 = smallRow1217_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1217 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1217 = _
  rw [smallRow1217_block_002_checked, smallRow1217_small_right_right_checked]
  rfl

def smallRow1217_small : List Nat := smallRow1217_small_left ++ smallRow1217_small_right
theorem smallRow1217_small_checked : upperNeighbors PrimePairVertices.tree_small 1217 = smallRow1217_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1217 ++ upperNeighbors PrimePairVertices.tree_small_right 1217 = _
  rw [smallRow1217_small_left_checked, smallRow1217_small_right_checked]
  rfl

def smallRow1217 : List Nat := smallRow1217_small
theorem smallRow1217_checked : upperNeighbors PrimePairVertices.smallTree 1217 = smallRow1217 := smallRow1217_small_checked

def smallRow1223_block_000 : List Nat := [1283, 1433, 1493, 1697, 1931]
set_option maxRecDepth 2048 in
theorem smallRow1223_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1223 = smallRow1223_block_000 := by decide

def smallRow1223_block_001 : List Nat := [2141, 2543, 2579, 2621, 2687, 2777, 3323, 3467]
set_option maxRecDepth 2048 in
theorem smallRow1223_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1223 = smallRow1223_block_001 := by decide

def smallRow1223_block_002 : List Nat := [4517, 4793, 5381, 5897, 5939, 6113, 6131]
set_option maxRecDepth 2048 in
theorem smallRow1223_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1223 = smallRow1223_block_002 := by decide

def smallRow1223_block_003 : List Nat := [6701, 6761, 7523, 7559, 7703]
set_option maxRecDepth 2048 in
theorem smallRow1223_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1223 = smallRow1223_block_003 := by decide

def smallRow1223_block_004 : List Nat := [8297]
set_option maxRecDepth 2048 in
theorem smallRow1223_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1223 = smallRow1223_block_004 := by decide

def smallRow1223_small_left : List Nat := smallRow1223_block_000 ++ smallRow1223_block_001
theorem smallRow1223_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1223 = smallRow1223_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1223 ++ upperNeighbors PrimePairVertices.tree_block_001 1223 = _
  rw [smallRow1223_block_000_checked, smallRow1223_block_001_checked]
  rfl

def smallRow1223_small_right_right : List Nat := smallRow1223_block_003 ++ smallRow1223_block_004
theorem smallRow1223_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1223 = smallRow1223_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1223 ++ upperNeighbors PrimePairVertices.tree_block_004 1223 = _
  rw [smallRow1223_block_003_checked, smallRow1223_block_004_checked]
  rfl

def smallRow1223_small_right : List Nat := smallRow1223_block_002 ++ smallRow1223_small_right_right
theorem smallRow1223_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1223 = smallRow1223_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1223 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1223 = _
  rw [smallRow1223_block_002_checked, smallRow1223_small_right_right_checked]
  rfl

def smallRow1223_small : List Nat := smallRow1223_small_left ++ smallRow1223_small_right
theorem smallRow1223_small_checked : upperNeighbors PrimePairVertices.tree_small 1223 = smallRow1223_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1223 ++ upperNeighbors PrimePairVertices.tree_small_right 1223 = _
  rw [smallRow1223_small_left_checked, smallRow1223_small_right_checked]
  rfl

def smallRow1223 : List Nat := smallRow1223_small
theorem smallRow1223_checked : upperNeighbors PrimePairVertices.smallTree 1223 = smallRow1223 := smallRow1223_small_checked

def smallRow1229_block_000 : List Nat := [1607]
set_option maxRecDepth 2048 in
theorem smallRow1229_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1229 = smallRow1229_block_000 := by decide

def smallRow1229_block_001 : List Nat := [2081, 2663, 2963, 3089, 3389, 3881]
set_option maxRecDepth 2048 in
theorem smallRow1229_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1229 = smallRow1229_block_001 := by decide

def smallRow1229_block_002 : List Nat := [4127, 4229, 5399, 5693, 5741, 5861]
set_option maxRecDepth 2048 in
theorem smallRow1229_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1229 = smallRow1229_block_002 := by decide

def smallRow1229_block_003 : List Nat := [6353, 7877, 7907]
set_option maxRecDepth 2048 in
theorem smallRow1229_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1229 = smallRow1229_block_003 := by decide

def smallRow1229_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1229_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1229 = smallRow1229_block_004 := by decide

def smallRow1229_small_left : List Nat := smallRow1229_block_000 ++ smallRow1229_block_001
theorem smallRow1229_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1229 = smallRow1229_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1229 ++ upperNeighbors PrimePairVertices.tree_block_001 1229 = _
  rw [smallRow1229_block_000_checked, smallRow1229_block_001_checked]
  rfl

def smallRow1229_small_right_right : List Nat := smallRow1229_block_003 ++ smallRow1229_block_004
theorem smallRow1229_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1229 = smallRow1229_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1229 ++ upperNeighbors PrimePairVertices.tree_block_004 1229 = _
  rw [smallRow1229_block_003_checked, smallRow1229_block_004_checked]
  rfl

def smallRow1229_small_right : List Nat := smallRow1229_block_002 ++ smallRow1229_small_right_right
theorem smallRow1229_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1229 = smallRow1229_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1229 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1229 = _
  rw [smallRow1229_block_002_checked, smallRow1229_small_right_right_checked]
  rfl

def smallRow1229_small : List Nat := smallRow1229_small_left ++ smallRow1229_small_right
theorem smallRow1229_small_checked : upperNeighbors PrimePairVertices.tree_small 1229 = smallRow1229_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1229 ++ upperNeighbors PrimePairVertices.tree_small_right 1229 = _
  rw [smallRow1229_small_left_checked, smallRow1229_small_right_checked]
  rfl

def smallRow1229 : List Nat := smallRow1229_small
theorem smallRow1229_checked : upperNeighbors PrimePairVertices.smallTree 1229 = smallRow1229 := smallRow1229_small_checked

def smallRow1231_block_000 : List Nat := [1777, 1951]
set_option maxRecDepth 2048 in
theorem smallRow1231_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1231 = smallRow1231_block_000 := by decide

def smallRow1231_block_001 : List Nat := [2143, 2689, 2917]
set_option maxRecDepth 2048 in
theorem smallRow1231_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1231 = smallRow1231_block_001 := by decide

def smallRow1231_block_002 : List Nat := [4231, 4243, 4327, 4483, 4507, 4597, 4789, 5323, 5449, 5647, 5659, 5881, 6037, 6091]
set_option maxRecDepth 2048 in
theorem smallRow1231_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1231 = smallRow1231_block_002 := by decide

def smallRow1231_block_003 : List Nat := [6229, 6529, 6949, 7297, 7309]
set_option maxRecDepth 2048 in
theorem smallRow1231_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1231 = smallRow1231_block_003 := by decide

def smallRow1231_block_004 : List Nat := [8221]
set_option maxRecDepth 2048 in
theorem smallRow1231_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1231 = smallRow1231_block_004 := by decide

def smallRow1231_small_left : List Nat := smallRow1231_block_000 ++ smallRow1231_block_001
theorem smallRow1231_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1231 = smallRow1231_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1231 ++ upperNeighbors PrimePairVertices.tree_block_001 1231 = _
  rw [smallRow1231_block_000_checked, smallRow1231_block_001_checked]
  rfl

def smallRow1231_small_right_right : List Nat := smallRow1231_block_003 ++ smallRow1231_block_004
theorem smallRow1231_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1231 = smallRow1231_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1231 ++ upperNeighbors PrimePairVertices.tree_block_004 1231 = _
  rw [smallRow1231_block_003_checked, smallRow1231_block_004_checked]
  rfl

def smallRow1231_small_right : List Nat := smallRow1231_block_002 ++ smallRow1231_small_right_right
theorem smallRow1231_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1231 = smallRow1231_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1231 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1231 = _
  rw [smallRow1231_block_002_checked, smallRow1231_small_right_right_checked]
  rfl

def smallRow1231_small : List Nat := smallRow1231_small_left ++ smallRow1231_small_right
theorem smallRow1231_small_checked : upperNeighbors PrimePairVertices.tree_small 1231 = smallRow1231_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1231 ++ upperNeighbors PrimePairVertices.tree_small_right 1231 = _
  rw [smallRow1231_small_left_checked, smallRow1231_small_right_checked]
  rfl

def smallRow1231 : List Nat := smallRow1231_small
theorem smallRow1231_checked : upperNeighbors PrimePairVertices.smallTree 1231 = smallRow1231 := smallRow1231_small_checked

end ProjectEulerSolutions.PrimePairCache
