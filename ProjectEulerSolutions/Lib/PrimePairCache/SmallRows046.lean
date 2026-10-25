import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2539_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2539_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2539 = smallRow2539_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2539_block_001 : List Nat := [3163, 3229, 3463]
set_option maxRecDepth 2048 in
theorem smallRow2539_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2539 = smallRow2539_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2539_block_002 : List Nat := [4297, 4339, 4441, 4723, 5119, 5179, 5449]
set_option maxRecDepth 2048 in
theorem smallRow2539_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2539 = smallRow2539_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2539_block_003 : List Nat := [6421, 6823, 7159, 7951]
set_option maxRecDepth 2048 in
theorem smallRow2539_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2539 = smallRow2539_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2539_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2539_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2539 = smallRow2539_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2539_small_left : List Nat := smallRow2539_block_000 ++ smallRow2539_block_001
theorem smallRow2539_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2539 = smallRow2539_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2539 ++ upperNeighbors PrimePairVertices.tree_block_001 2539 = _
  rw [smallRow2539_block_000_checked, smallRow2539_block_001_checked]
  rfl

def smallRow2539_small_right_right : List Nat := smallRow2539_block_003 ++ smallRow2539_block_004
theorem smallRow2539_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2539 = smallRow2539_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2539 ++ upperNeighbors PrimePairVertices.tree_block_004 2539 = _
  rw [smallRow2539_block_003_checked, smallRow2539_block_004_checked]
  rfl

def smallRow2539_small_right : List Nat := smallRow2539_block_002 ++ smallRow2539_small_right_right
theorem smallRow2539_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2539 = smallRow2539_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2539 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2539 = _
  rw [smallRow2539_block_002_checked, smallRow2539_small_right_right_checked]
  rfl

def smallRow2539_small : List Nat := smallRow2539_small_left ++ smallRow2539_small_right
theorem smallRow2539_small_checked : upperNeighbors PrimePairVertices.tree_small 2539 = smallRow2539_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2539 ++ upperNeighbors PrimePairVertices.tree_small_right 2539 = _
  rw [smallRow2539_small_left_checked, smallRow2539_small_right_checked]
  rfl

def smallRow2539 : List Nat := smallRow2539_small
theorem smallRow2539_checked : upperNeighbors PrimePairVertices.smallTree 2539 = smallRow2539 := smallRow2539_small_checked

def smallRow2543_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2543_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2543 = smallRow2543_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2543_block_001 : List Nat := [2609, 3371, 3539, 3851]
set_option maxRecDepth 2048 in
theorem smallRow2543_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2543 = smallRow2543_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2543_block_002 : List Nat := [4337, 4349, 4523, 5051, 5147, 5153, 5237, 5441, 5483, 5591, 6011]
set_option maxRecDepth 2048 in
theorem smallRow2543_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2543 = smallRow2543_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2543_block_003 : List Nat := [6791, 6911, 7331, 7541, 7649]
set_option maxRecDepth 2048 in
theorem smallRow2543_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2543 = smallRow2543_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2543_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2543_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2543 = smallRow2543_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2543_small_left : List Nat := smallRow2543_block_000 ++ smallRow2543_block_001
theorem smallRow2543_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2543 = smallRow2543_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2543 ++ upperNeighbors PrimePairVertices.tree_block_001 2543 = _
  rw [smallRow2543_block_000_checked, smallRow2543_block_001_checked]
  rfl

def smallRow2543_small_right_right : List Nat := smallRow2543_block_003 ++ smallRow2543_block_004
theorem smallRow2543_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2543 = smallRow2543_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2543 ++ upperNeighbors PrimePairVertices.tree_block_004 2543 = _
  rw [smallRow2543_block_003_checked, smallRow2543_block_004_checked]
  rfl

def smallRow2543_small_right : List Nat := smallRow2543_block_002 ++ smallRow2543_small_right_right
theorem smallRow2543_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2543 = smallRow2543_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2543 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2543 = _
  rw [smallRow2543_block_002_checked, smallRow2543_small_right_right_checked]
  rfl

def smallRow2543_small : List Nat := smallRow2543_small_left ++ smallRow2543_small_right
theorem smallRow2543_small_checked : upperNeighbors PrimePairVertices.tree_small 2543 = smallRow2543_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2543 ++ upperNeighbors PrimePairVertices.tree_small_right 2543 = _
  rw [smallRow2543_small_left_checked, smallRow2543_small_right_checked]
  rfl

def smallRow2543 : List Nat := smallRow2543_small
theorem smallRow2543_checked : upperNeighbors PrimePairVertices.smallTree 2543 = smallRow2543 := smallRow2543_small_checked

def smallRow2549_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2549_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2549 = smallRow2549_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2549_block_001 : List Nat := [2729, 3089, 3329, 3593, 3947, 4019]
set_option maxRecDepth 2048 in
theorem smallRow2549_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2549 = smallRow2549_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2549_block_002 : List Nat := [4241, 4349, 5471, 5501]
set_option maxRecDepth 2048 in
theorem smallRow2549_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2549 = smallRow2549_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2549_block_003 : List Nat := [6323, 6761, 6833, 7211, 7673, 7793, 7937, 7949, 8009, 8093]
set_option maxRecDepth 2048 in
theorem smallRow2549_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2549 = smallRow2549_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2549_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2549_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2549 = smallRow2549_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2549_small_left : List Nat := smallRow2549_block_000 ++ smallRow2549_block_001
theorem smallRow2549_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2549 = smallRow2549_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2549 ++ upperNeighbors PrimePairVertices.tree_block_001 2549 = _
  rw [smallRow2549_block_000_checked, smallRow2549_block_001_checked]
  rfl

def smallRow2549_small_right_right : List Nat := smallRow2549_block_003 ++ smallRow2549_block_004
theorem smallRow2549_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2549 = smallRow2549_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2549 ++ upperNeighbors PrimePairVertices.tree_block_004 2549 = _
  rw [smallRow2549_block_003_checked, smallRow2549_block_004_checked]
  rfl

def smallRow2549_small_right : List Nat := smallRow2549_block_002 ++ smallRow2549_small_right_right
theorem smallRow2549_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2549 = smallRow2549_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2549 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2549 = _
  rw [smallRow2549_block_002_checked, smallRow2549_small_right_right_checked]
  rfl

def smallRow2549_small : List Nat := smallRow2549_small_left ++ smallRow2549_small_right
theorem smallRow2549_small_checked : upperNeighbors PrimePairVertices.tree_small 2549 = smallRow2549_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2549 ++ upperNeighbors PrimePairVertices.tree_small_right 2549 = _
  rw [smallRow2549_small_left_checked, smallRow2549_small_right_checked]
  rfl

def smallRow2549 : List Nat := smallRow2549_small
theorem smallRow2549_checked : upperNeighbors PrimePairVertices.smallTree 2549 = smallRow2549 := smallRow2549_small_checked

def smallRow2551_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2551_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2551 = smallRow2551_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2551_block_001 : List Nat := [2887, 3331]
set_option maxRecDepth 2048 in
theorem smallRow2551_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2551 = smallRow2551_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2551_block_002 : List Nat := [4219, 4591, 4813, 4969, 5059, 5227, 6037]
set_option maxRecDepth 2048 in
theorem smallRow2551_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2551 = smallRow2551_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2551_block_003 : List Nat := [6163, 6793, 7537]
set_option maxRecDepth 2048 in
theorem smallRow2551_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2551 = smallRow2551_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2551_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2551_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2551 = smallRow2551_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2551_small_left : List Nat := smallRow2551_block_000 ++ smallRow2551_block_001
theorem smallRow2551_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2551 = smallRow2551_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2551 ++ upperNeighbors PrimePairVertices.tree_block_001 2551 = _
  rw [smallRow2551_block_000_checked, smallRow2551_block_001_checked]
  rfl

def smallRow2551_small_right_right : List Nat := smallRow2551_block_003 ++ smallRow2551_block_004
theorem smallRow2551_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2551 = smallRow2551_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2551 ++ upperNeighbors PrimePairVertices.tree_block_004 2551 = _
  rw [smallRow2551_block_003_checked, smallRow2551_block_004_checked]
  rfl

def smallRow2551_small_right : List Nat := smallRow2551_block_002 ++ smallRow2551_small_right_right
theorem smallRow2551_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2551 = smallRow2551_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2551 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2551 = _
  rw [smallRow2551_block_002_checked, smallRow2551_small_right_right_checked]
  rfl

def smallRow2551_small : List Nat := smallRow2551_small_left ++ smallRow2551_small_right
theorem smallRow2551_small_checked : upperNeighbors PrimePairVertices.tree_small 2551 = smallRow2551_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2551 ++ upperNeighbors PrimePairVertices.tree_small_right 2551 = _
  rw [smallRow2551_small_left_checked, smallRow2551_small_right_checked]
  rfl

def smallRow2551 : List Nat := smallRow2551_small
theorem smallRow2551_checked : upperNeighbors PrimePairVertices.smallTree 2551 = smallRow2551 := smallRow2551_small_checked

def smallRow2557_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2557_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2557 = smallRow2557_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2557_block_001 : List Nat := [2683, 3313]
set_option maxRecDepth 2048 in
theorem smallRow2557_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2557 = smallRow2557_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2557_block_002 : List Nat := [4243, 4639, 4657, 4957, 5749]
set_option maxRecDepth 2048 in
theorem smallRow2557_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2557 = smallRow2557_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2557_block_003 : List Nat := [6763, 6871, 6997, 7369, 7687, 7789, 7873, 8191]
set_option maxRecDepth 2048 in
theorem smallRow2557_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2557 = smallRow2557_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2557_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2557_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2557 = smallRow2557_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2557_small_left : List Nat := smallRow2557_block_000 ++ smallRow2557_block_001
theorem smallRow2557_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2557 = smallRow2557_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2557 ++ upperNeighbors PrimePairVertices.tree_block_001 2557 = _
  rw [smallRow2557_block_000_checked, smallRow2557_block_001_checked]
  rfl

def smallRow2557_small_right_right : List Nat := smallRow2557_block_003 ++ smallRow2557_block_004
theorem smallRow2557_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2557 = smallRow2557_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2557 ++ upperNeighbors PrimePairVertices.tree_block_004 2557 = _
  rw [smallRow2557_block_003_checked, smallRow2557_block_004_checked]
  rfl

def smallRow2557_small_right : List Nat := smallRow2557_block_002 ++ smallRow2557_small_right_right
theorem smallRow2557_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2557 = smallRow2557_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2557 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2557 = _
  rw [smallRow2557_block_002_checked, smallRow2557_small_right_right_checked]
  rfl

def smallRow2557_small : List Nat := smallRow2557_small_left ++ smallRow2557_small_right
theorem smallRow2557_small_checked : upperNeighbors PrimePairVertices.tree_small 2557 = smallRow2557_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2557 ++ upperNeighbors PrimePairVertices.tree_small_right 2557 = _
  rw [smallRow2557_small_left_checked, smallRow2557_small_right_checked]
  rfl

def smallRow2557 : List Nat := smallRow2557_small
theorem smallRow2557_checked : upperNeighbors PrimePairVertices.smallTree 2557 = smallRow2557 := smallRow2557_small_checked

def smallRow2579_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2579_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2579 = smallRow2579_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2579_block_001 : List Nat := [3359, 3623, 4007]
set_option maxRecDepth 2048 in
theorem smallRow2579_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2579 = smallRow2579_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2579_block_002 : List Nat := [4421, 4871, 4931, 5639, 5843, 6131]
set_option maxRecDepth 2048 in
theorem smallRow2579_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2579 = smallRow2579_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2579_block_003 : List Nat := [7229, 7349, 7727]
set_option maxRecDepth 2048 in
theorem smallRow2579_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2579 = smallRow2579_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2579_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2579_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2579 = smallRow2579_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2579_small_left : List Nat := smallRow2579_block_000 ++ smallRow2579_block_001
theorem smallRow2579_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2579 = smallRow2579_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2579 ++ upperNeighbors PrimePairVertices.tree_block_001 2579 = _
  rw [smallRow2579_block_000_checked, smallRow2579_block_001_checked]
  rfl

def smallRow2579_small_right_right : List Nat := smallRow2579_block_003 ++ smallRow2579_block_004
theorem smallRow2579_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2579 = smallRow2579_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2579 ++ upperNeighbors PrimePairVertices.tree_block_004 2579 = _
  rw [smallRow2579_block_003_checked, smallRow2579_block_004_checked]
  rfl

def smallRow2579_small_right : List Nat := smallRow2579_block_002 ++ smallRow2579_small_right_right
theorem smallRow2579_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2579 = smallRow2579_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2579 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2579 = _
  rw [smallRow2579_block_002_checked, smallRow2579_small_right_right_checked]
  rfl

def smallRow2579_small : List Nat := smallRow2579_small_left ++ smallRow2579_small_right
theorem smallRow2579_small_checked : upperNeighbors PrimePairVertices.tree_small 2579 = smallRow2579_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2579 ++ upperNeighbors PrimePairVertices.tree_small_right 2579 = _
  rw [smallRow2579_small_left_checked, smallRow2579_small_right_checked]
  rfl

def smallRow2579 : List Nat := smallRow2579_small
theorem smallRow2579_checked : upperNeighbors PrimePairVertices.smallTree 2579 = smallRow2579 := smallRow2579_small_checked

def smallRow2591_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2591_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2591 = smallRow2591_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2591_block_001 : List Nat := [2633, 3257, 3527, 4073]
set_option maxRecDepth 2048 in
theorem smallRow2591_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2591 = smallRow2591_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2591_block_002 : List Nat := [4211, 4283, 4937, 5741, 5807, 5849]
set_option maxRecDepth 2048 in
theorem smallRow2591_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2591 = smallRow2591_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2591_block_003 : List Nat := [6197, 6299, 7193, 7499, 7937]
set_option maxRecDepth 2048 in
theorem smallRow2591_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2591 = smallRow2591_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2591_block_004 : List Nat := [8231]
set_option maxRecDepth 2048 in
theorem smallRow2591_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2591 = smallRow2591_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2591_small_left : List Nat := smallRow2591_block_000 ++ smallRow2591_block_001
theorem smallRow2591_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2591 = smallRow2591_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2591 ++ upperNeighbors PrimePairVertices.tree_block_001 2591 = _
  rw [smallRow2591_block_000_checked, smallRow2591_block_001_checked]
  rfl

def smallRow2591_small_right_right : List Nat := smallRow2591_block_003 ++ smallRow2591_block_004
theorem smallRow2591_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2591 = smallRow2591_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2591 ++ upperNeighbors PrimePairVertices.tree_block_004 2591 = _
  rw [smallRow2591_block_003_checked, smallRow2591_block_004_checked]
  rfl

def smallRow2591_small_right : List Nat := smallRow2591_block_002 ++ smallRow2591_small_right_right
theorem smallRow2591_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2591 = smallRow2591_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2591 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2591 = _
  rw [smallRow2591_block_002_checked, smallRow2591_small_right_right_checked]
  rfl

def smallRow2591_small : List Nat := smallRow2591_small_left ++ smallRow2591_small_right
theorem smallRow2591_small_checked : upperNeighbors PrimePairVertices.tree_small 2591 = smallRow2591_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2591 ++ upperNeighbors PrimePairVertices.tree_small_right 2591 = _
  rw [smallRow2591_small_left_checked, smallRow2591_small_right_checked]
  rfl

def smallRow2591 : List Nat := smallRow2591_small
theorem smallRow2591_checked : upperNeighbors PrimePairVertices.smallTree 2591 = smallRow2591 := smallRow2591_small_checked

def smallRow2593_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2593_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2593 = smallRow2593_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2593_block_001 : List Nat := [2617, 2713, 2719, 2887, 2971, 3169, 3307, 3373, 3709, 3847]
set_option maxRecDepth 2048 in
theorem smallRow2593_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2593 = smallRow2593_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2593_block_002 : List Nat := [5011, 5101, 5281, 5443, 5683]
set_option maxRecDepth 2048 in
theorem smallRow2593_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2593 = smallRow2593_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2593_block_003 : List Nat := [7027, 7039, 7573, 7753, 7927]
set_option maxRecDepth 2048 in
theorem smallRow2593_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2593 = smallRow2593_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2593_block_004 : List Nat := [8293]
set_option maxRecDepth 2048 in
theorem smallRow2593_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2593 = smallRow2593_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2593_small_left : List Nat := smallRow2593_block_000 ++ smallRow2593_block_001
theorem smallRow2593_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2593 = smallRow2593_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2593 ++ upperNeighbors PrimePairVertices.tree_block_001 2593 = _
  rw [smallRow2593_block_000_checked, smallRow2593_block_001_checked]
  rfl

def smallRow2593_small_right_right : List Nat := smallRow2593_block_003 ++ smallRow2593_block_004
theorem smallRow2593_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2593 = smallRow2593_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2593 ++ upperNeighbors PrimePairVertices.tree_block_004 2593 = _
  rw [smallRow2593_block_003_checked, smallRow2593_block_004_checked]
  rfl

def smallRow2593_small_right : List Nat := smallRow2593_block_002 ++ smallRow2593_small_right_right
theorem smallRow2593_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2593 = smallRow2593_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2593 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2593 = _
  rw [smallRow2593_block_002_checked, smallRow2593_small_right_right_checked]
  rfl

def smallRow2593_small : List Nat := smallRow2593_small_left ++ smallRow2593_small_right
theorem smallRow2593_small_checked : upperNeighbors PrimePairVertices.tree_small 2593 = smallRow2593_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2593 ++ upperNeighbors PrimePairVertices.tree_small_right 2593 = _
  rw [smallRow2593_small_left_checked, smallRow2593_small_right_checked]
  rfl

def smallRow2593 : List Nat := smallRow2593_small
theorem smallRow2593_checked : upperNeighbors PrimePairVertices.smallTree 2593 = smallRow2593 := smallRow2593_small_checked

end ProjectEulerSolutions.PrimePairCache
