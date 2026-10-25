import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2447_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2447_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2447 = smallRow2447_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2447_block_001 : List Nat := [3011, 3797, 4019]
set_option maxRecDepth 2048 in
theorem smallRow2447_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2447 = smallRow2447_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2447_block_002 : List Nat := [4937, 5813, 5939, 6113]
set_option maxRecDepth 2048 in
theorem smallRow2447_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2447 = smallRow2447_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2447_block_003 : List Nat := [6323, 7001, 7283, 7907, 8093, 8123]
set_option maxRecDepth 2048 in
theorem smallRow2447_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2447 = smallRow2447_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2447_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2447_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2447 = smallRow2447_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2447_small_left : List Nat := smallRow2447_block_000 ++ smallRow2447_block_001
theorem smallRow2447_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2447 = smallRow2447_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2447 ++ upperNeighbors PrimePairVertices.tree_block_001 2447 = _
  rw [smallRow2447_block_000_checked, smallRow2447_block_001_checked]
  rfl

def smallRow2447_small_right_right : List Nat := smallRow2447_block_003 ++ smallRow2447_block_004
theorem smallRow2447_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2447 = smallRow2447_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2447 ++ upperNeighbors PrimePairVertices.tree_block_004 2447 = _
  rw [smallRow2447_block_003_checked, smallRow2447_block_004_checked]
  rfl

def smallRow2447_small_right : List Nat := smallRow2447_block_002 ++ smallRow2447_small_right_right
theorem smallRow2447_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2447 = smallRow2447_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2447 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2447 = _
  rw [smallRow2447_block_002_checked, smallRow2447_small_right_right_checked]
  rfl

def smallRow2447_small : List Nat := smallRow2447_small_left ++ smallRow2447_small_right
theorem smallRow2447_small_checked : upperNeighbors PrimePairVertices.tree_small 2447 = smallRow2447_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2447 ++ upperNeighbors PrimePairVertices.tree_small_right 2447 = _
  rw [smallRow2447_small_left_checked, smallRow2447_small_right_checked]
  rfl

def smallRow2447 : List Nat := smallRow2447_small
theorem smallRow2447_checked : upperNeighbors PrimePairVertices.smallTree 2447 = smallRow2447 := smallRow2447_small_checked

def smallRow2459_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2459_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2459 = smallRow2459_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2459_block_001 : List Nat := [2657, 2927]
set_option maxRecDepth 2048 in
theorem smallRow2459_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2459 = smallRow2459_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2459_block_002 : List Nat := [4139, 4289, 4463, 4919, 5171, 5843, 6101]
set_option maxRecDepth 2048 in
theorem smallRow2459_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2459 = smallRow2459_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2459_block_003 : List Nat := [6353, 6899, 7211, 8009, 8069]
set_option maxRecDepth 2048 in
theorem smallRow2459_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2459 = smallRow2459_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2459_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2459_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2459 = smallRow2459_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2459_small_left : List Nat := smallRow2459_block_000 ++ smallRow2459_block_001
theorem smallRow2459_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2459 = smallRow2459_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2459 ++ upperNeighbors PrimePairVertices.tree_block_001 2459 = _
  rw [smallRow2459_block_000_checked, smallRow2459_block_001_checked]
  rfl

def smallRow2459_small_right_right : List Nat := smallRow2459_block_003 ++ smallRow2459_block_004
theorem smallRow2459_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2459 = smallRow2459_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2459 ++ upperNeighbors PrimePairVertices.tree_block_004 2459 = _
  rw [smallRow2459_block_003_checked, smallRow2459_block_004_checked]
  rfl

def smallRow2459_small_right : List Nat := smallRow2459_block_002 ++ smallRow2459_small_right_right
theorem smallRow2459_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2459 = smallRow2459_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2459 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2459 = _
  rw [smallRow2459_block_002_checked, smallRow2459_small_right_right_checked]
  rfl

def smallRow2459_small : List Nat := smallRow2459_small_left ++ smallRow2459_small_right
theorem smallRow2459_small_checked : upperNeighbors PrimePairVertices.tree_small 2459 = smallRow2459_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2459 ++ upperNeighbors PrimePairVertices.tree_small_right 2459 = _
  rw [smallRow2459_small_left_checked, smallRow2459_small_right_checked]
  rfl

def smallRow2459 : List Nat := smallRow2459_small
theorem smallRow2459_checked : upperNeighbors PrimePairVertices.smallTree 2459 = smallRow2459 := smallRow2459_small_checked

def smallRow2467_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2467_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2467 = smallRow2467_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2467_block_001 : List Nat := [2539, 2551, 3217, 4051]
set_option maxRecDepth 2048 in
theorem smallRow2467_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2467 = smallRow2467_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2467_block_002 : List Nat := [4723, 4861, 5407, 5683, 5689, 6037]
set_option maxRecDepth 2048 in
theorem smallRow2467_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2467 = smallRow2467_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2467_block_003 : List Nat := [6277, 6529, 7621, 7963, 8167]
set_option maxRecDepth 2048 in
theorem smallRow2467_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2467 = smallRow2467_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2467_block_004 : List Nat := [8377]
set_option maxRecDepth 2048 in
theorem smallRow2467_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2467 = smallRow2467_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2467_small_left : List Nat := smallRow2467_block_000 ++ smallRow2467_block_001
theorem smallRow2467_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2467 = smallRow2467_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2467 ++ upperNeighbors PrimePairVertices.tree_block_001 2467 = _
  rw [smallRow2467_block_000_checked, smallRow2467_block_001_checked]
  rfl

def smallRow2467_small_right_right : List Nat := smallRow2467_block_003 ++ smallRow2467_block_004
theorem smallRow2467_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2467 = smallRow2467_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2467 ++ upperNeighbors PrimePairVertices.tree_block_004 2467 = _
  rw [smallRow2467_block_003_checked, smallRow2467_block_004_checked]
  rfl

def smallRow2467_small_right : List Nat := smallRow2467_block_002 ++ smallRow2467_small_right_right
theorem smallRow2467_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2467 = smallRow2467_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2467 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2467 = _
  rw [smallRow2467_block_002_checked, smallRow2467_small_right_right_checked]
  rfl

def smallRow2467_small : List Nat := smallRow2467_small_left ++ smallRow2467_small_right
theorem smallRow2467_small_checked : upperNeighbors PrimePairVertices.tree_small 2467 = smallRow2467_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2467 ++ upperNeighbors PrimePairVertices.tree_small_right 2467 = _
  rw [smallRow2467_small_left_checked, smallRow2467_small_right_checked]
  rfl

def smallRow2467 : List Nat := smallRow2467_small
theorem smallRow2467_checked : upperNeighbors PrimePairVertices.smallTree 2467 = smallRow2467 := smallRow2467_small_checked

def smallRow2473_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2473_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2473 = smallRow2473_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2473_block_001 : List Nat := [2647, 2713, 2791, 3001, 3343, 3889]
set_option maxRecDepth 2048 in
theorem smallRow2473_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2473 = smallRow2473_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2473_block_002 : List Nat := [4327, 5023, 5437, 5881]
set_option maxRecDepth 2048 in
theorem smallRow2473_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2473 = smallRow2473_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2473_block_003 : List Nat := [6379, 6427, 7417]
set_option maxRecDepth 2048 in
theorem smallRow2473_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2473 = smallRow2473_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2473_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow2473_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2473 = smallRow2473_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2473_small_left : List Nat := smallRow2473_block_000 ++ smallRow2473_block_001
theorem smallRow2473_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2473 = smallRow2473_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2473 ++ upperNeighbors PrimePairVertices.tree_block_001 2473 = _
  rw [smallRow2473_block_000_checked, smallRow2473_block_001_checked]
  rfl

def smallRow2473_small_right_right : List Nat := smallRow2473_block_003 ++ smallRow2473_block_004
theorem smallRow2473_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2473 = smallRow2473_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2473 ++ upperNeighbors PrimePairVertices.tree_block_004 2473 = _
  rw [smallRow2473_block_003_checked, smallRow2473_block_004_checked]
  rfl

def smallRow2473_small_right : List Nat := smallRow2473_block_002 ++ smallRow2473_small_right_right
theorem smallRow2473_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2473 = smallRow2473_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2473 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2473 = _
  rw [smallRow2473_block_002_checked, smallRow2473_small_right_right_checked]
  rfl

def smallRow2473_small : List Nat := smallRow2473_small_left ++ smallRow2473_small_right
theorem smallRow2473_small_checked : upperNeighbors PrimePairVertices.tree_small 2473 = smallRow2473_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2473 ++ upperNeighbors PrimePairVertices.tree_small_right 2473 = _
  rw [smallRow2473_small_left_checked, smallRow2473_small_right_checked]
  rfl

def smallRow2473 : List Nat := smallRow2473_small
theorem smallRow2473_checked : upperNeighbors PrimePairVertices.smallTree 2473 = smallRow2473 := smallRow2473_small_checked

def smallRow2477_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2477_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2477 = smallRow2477_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2477_block_001 : List Nat := [2663, 2801, 2939, 3797, 4073]
set_option maxRecDepth 2048 in
theorem smallRow2477_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2477 = smallRow2477_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2477_block_002 : List Nat := [4133, 5039, 5381]
set_option maxRecDepth 2048 in
theorem smallRow2477_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2477 = smallRow2477_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2477_block_003 : List Nat := [7703, 7853]
set_option maxRecDepth 2048 in
theorem smallRow2477_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2477 = smallRow2477_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2477_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2477_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2477 = smallRow2477_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2477_small_left : List Nat := smallRow2477_block_000 ++ smallRow2477_block_001
theorem smallRow2477_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2477 = smallRow2477_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2477 ++ upperNeighbors PrimePairVertices.tree_block_001 2477 = _
  rw [smallRow2477_block_000_checked, smallRow2477_block_001_checked]
  rfl

def smallRow2477_small_right_right : List Nat := smallRow2477_block_003 ++ smallRow2477_block_004
theorem smallRow2477_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2477 = smallRow2477_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2477 ++ upperNeighbors PrimePairVertices.tree_block_004 2477 = _
  rw [smallRow2477_block_003_checked, smallRow2477_block_004_checked]
  rfl

def smallRow2477_small_right : List Nat := smallRow2477_block_002 ++ smallRow2477_small_right_right
theorem smallRow2477_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2477 = smallRow2477_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2477 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2477 = _
  rw [smallRow2477_block_002_checked, smallRow2477_small_right_right_checked]
  rfl

def smallRow2477_small : List Nat := smallRow2477_small_left ++ smallRow2477_small_right
theorem smallRow2477_small_checked : upperNeighbors PrimePairVertices.tree_small 2477 = smallRow2477_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2477 ++ upperNeighbors PrimePairVertices.tree_small_right 2477 = _
  rw [smallRow2477_small_left_checked, smallRow2477_small_right_checked]
  rfl

def smallRow2477 : List Nat := smallRow2477_small
theorem smallRow2477_checked : upperNeighbors PrimePairVertices.smallTree 2477 = smallRow2477 := smallRow2477_small_checked

def smallRow2503_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2503_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2503 = smallRow2503_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2503_block_001 : List Nat := [3391, 3517, 3559]
set_option maxRecDepth 2048 in
theorem smallRow2503_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2503 = smallRow2503_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2503_block_002 : List Nat := [4447, 5281, 5581, 5623, 5869, 6079]
set_option maxRecDepth 2048 in
theorem smallRow2503_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2503 = smallRow2503_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2503_block_003 : List Nat := [6163, 6421, 6763, 6967, 7393, 7591]
set_option maxRecDepth 2048 in
theorem smallRow2503_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2503 = smallRow2503_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2503_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2503_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2503 = smallRow2503_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2503_small_left : List Nat := smallRow2503_block_000 ++ smallRow2503_block_001
theorem smallRow2503_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2503 = smallRow2503_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2503 ++ upperNeighbors PrimePairVertices.tree_block_001 2503 = _
  rw [smallRow2503_block_000_checked, smallRow2503_block_001_checked]
  rfl

def smallRow2503_small_right_right : List Nat := smallRow2503_block_003 ++ smallRow2503_block_004
theorem smallRow2503_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2503 = smallRow2503_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2503 ++ upperNeighbors PrimePairVertices.tree_block_004 2503 = _
  rw [smallRow2503_block_003_checked, smallRow2503_block_004_checked]
  rfl

def smallRow2503_small_right : List Nat := smallRow2503_block_002 ++ smallRow2503_small_right_right
theorem smallRow2503_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2503 = smallRow2503_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2503 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2503 = _
  rw [smallRow2503_block_002_checked, smallRow2503_small_right_right_checked]
  rfl

def smallRow2503_small : List Nat := smallRow2503_small_left ++ smallRow2503_small_right
theorem smallRow2503_small_checked : upperNeighbors PrimePairVertices.tree_small 2503 = smallRow2503_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2503 ++ upperNeighbors PrimePairVertices.tree_small_right 2503 = _
  rw [smallRow2503_small_left_checked, smallRow2503_small_right_checked]
  rfl

def smallRow2503 : List Nat := smallRow2503_small
theorem smallRow2503_checked : upperNeighbors PrimePairVertices.smallTree 2503 = smallRow2503 := smallRow2503_small_checked

def smallRow2521_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2521_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2521 = smallRow2521_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2521_block_001 : List Nat := [2797, 3271, 3697, 3739]
set_option maxRecDepth 2048 in
theorem smallRow2521_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2521 = smallRow2521_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2521_block_002 : List Nat := [4201, 4363, 4759, 4999, 5557]
set_option maxRecDepth 2048 in
theorem smallRow2521_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2521 = smallRow2521_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2521_block_003 : List Nat := [6343, 7561]
set_option maxRecDepth 2048 in
theorem smallRow2521_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2521 = smallRow2521_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2521_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2521_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2521 = smallRow2521_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2521_small_left : List Nat := smallRow2521_block_000 ++ smallRow2521_block_001
theorem smallRow2521_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2521 = smallRow2521_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2521 ++ upperNeighbors PrimePairVertices.tree_block_001 2521 = _
  rw [smallRow2521_block_000_checked, smallRow2521_block_001_checked]
  rfl

def smallRow2521_small_right_right : List Nat := smallRow2521_block_003 ++ smallRow2521_block_004
theorem smallRow2521_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2521 = smallRow2521_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2521 ++ upperNeighbors PrimePairVertices.tree_block_004 2521 = _
  rw [smallRow2521_block_003_checked, smallRow2521_block_004_checked]
  rfl

def smallRow2521_small_right : List Nat := smallRow2521_block_002 ++ smallRow2521_small_right_right
theorem smallRow2521_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2521 = smallRow2521_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2521 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2521 = _
  rw [smallRow2521_block_002_checked, smallRow2521_small_right_right_checked]
  rfl

def smallRow2521_small : List Nat := smallRow2521_small_left ++ smallRow2521_small_right
theorem smallRow2521_small_checked : upperNeighbors PrimePairVertices.tree_small 2521 = smallRow2521_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2521 ++ upperNeighbors PrimePairVertices.tree_small_right 2521 = _
  rw [smallRow2521_small_left_checked, smallRow2521_small_right_checked]
  rfl

def smallRow2521 : List Nat := smallRow2521_small
theorem smallRow2521_checked : upperNeighbors PrimePairVertices.smallTree 2521 = smallRow2521 := smallRow2521_small_checked

def smallRow2531_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2531_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2531 = smallRow2531_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2531_block_001 : List Nat := [2549, 2741, 3539, 3797]
set_option maxRecDepth 2048 in
theorem smallRow2531_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2531 = smallRow2531_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2531_block_002 : List Nat := [4643, 4889, 4967, 5273, 5309, 5651]
set_option maxRecDepth 2048 in
theorem smallRow2531_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2531 = smallRow2531_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2531_block_003 : List Nat := [6329, 6659, 6701, 7883, 8123, 8171]
set_option maxRecDepth 2048 in
theorem smallRow2531_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2531 = smallRow2531_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2531_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2531_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2531 = smallRow2531_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2531_small_left : List Nat := smallRow2531_block_000 ++ smallRow2531_block_001
theorem smallRow2531_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2531 = smallRow2531_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2531 ++ upperNeighbors PrimePairVertices.tree_block_001 2531 = _
  rw [smallRow2531_block_000_checked, smallRow2531_block_001_checked]
  rfl

def smallRow2531_small_right_right : List Nat := smallRow2531_block_003 ++ smallRow2531_block_004
theorem smallRow2531_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2531 = smallRow2531_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2531 ++ upperNeighbors PrimePairVertices.tree_block_004 2531 = _
  rw [smallRow2531_block_003_checked, smallRow2531_block_004_checked]
  rfl

def smallRow2531_small_right : List Nat := smallRow2531_block_002 ++ smallRow2531_small_right_right
theorem smallRow2531_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2531 = smallRow2531_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2531 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2531 = _
  rw [smallRow2531_block_002_checked, smallRow2531_small_right_right_checked]
  rfl

def smallRow2531_small : List Nat := smallRow2531_small_left ++ smallRow2531_small_right
theorem smallRow2531_small_checked : upperNeighbors PrimePairVertices.tree_small 2531 = smallRow2531_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2531 ++ upperNeighbors PrimePairVertices.tree_small_right 2531 = _
  rw [smallRow2531_small_left_checked, smallRow2531_small_right_checked]
  rfl

def smallRow2531 : List Nat := smallRow2531_small
theorem smallRow2531_checked : upperNeighbors PrimePairVertices.smallTree 2531 = smallRow2531 := smallRow2531_small_checked

end ProjectEulerSolutions.PrimePairCache
