import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2609_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2609_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2609 = smallRow2609_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2609_block_001 : List Nat := [2777, 3257, 3491, 3803]
set_option maxRecDepth 2048 in
theorem smallRow2609_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2609 = smallRow2609_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2609_block_002 : List Nat := [4241, 4931, 4973, 5651]
set_option maxRecDepth 2048 in
theorem smallRow2609_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2609 = smallRow2609_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2609_block_003 : List Nat := [6197, 6569, 6599, 7451, 7829, 8069]
set_option maxRecDepth 2048 in
theorem smallRow2609_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2609 = smallRow2609_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2609_block_004 : List Nat := [8273, 8297]
set_option maxRecDepth 2048 in
theorem smallRow2609_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2609 = smallRow2609_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2609_small_left : List Nat := smallRow2609_block_000 ++ smallRow2609_block_001
theorem smallRow2609_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2609 = smallRow2609_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2609 ++ upperNeighbors PrimePairVertices.tree_block_001 2609 = _
  rw [smallRow2609_block_000_checked, smallRow2609_block_001_checked]
  rfl

def smallRow2609_small_right_right : List Nat := smallRow2609_block_003 ++ smallRow2609_block_004
theorem smallRow2609_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2609 = smallRow2609_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2609 ++ upperNeighbors PrimePairVertices.tree_block_004 2609 = _
  rw [smallRow2609_block_003_checked, smallRow2609_block_004_checked]
  rfl

def smallRow2609_small_right : List Nat := smallRow2609_block_002 ++ smallRow2609_small_right_right
theorem smallRow2609_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2609 = smallRow2609_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2609 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2609 = _
  rw [smallRow2609_block_002_checked, smallRow2609_small_right_right_checked]
  rfl

def smallRow2609_small : List Nat := smallRow2609_small_left ++ smallRow2609_small_right
theorem smallRow2609_small_checked : upperNeighbors PrimePairVertices.tree_small 2609 = smallRow2609_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2609 ++ upperNeighbors PrimePairVertices.tree_small_right 2609 = _
  rw [smallRow2609_small_left_checked, smallRow2609_small_right_checked]
  rfl

def smallRow2609 : List Nat := smallRow2609_small
theorem smallRow2609_checked : upperNeighbors PrimePairVertices.smallTree 2609 = smallRow2609 := smallRow2609_small_checked

def smallRow2617_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2617_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2617 = smallRow2617_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2617_block_001 : List Nat := [2803, 2833, 2857, 3181, 3571]
set_option maxRecDepth 2048 in
theorem smallRow2617_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2617 = smallRow2617_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2617_block_002 : List Nat := [4273, 4567, 4801, 5689, 5839]
set_option maxRecDepth 2048 in
theorem smallRow2617_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2617 = smallRow2617_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2617_block_003 : List Nat := [7213, 7507, 7741, 8161]
set_option maxRecDepth 2048 in
theorem smallRow2617_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2617 = smallRow2617_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2617_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2617_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2617 = smallRow2617_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2617_small_left : List Nat := smallRow2617_block_000 ++ smallRow2617_block_001
theorem smallRow2617_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2617 = smallRow2617_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2617 ++ upperNeighbors PrimePairVertices.tree_block_001 2617 = _
  rw [smallRow2617_block_000_checked, smallRow2617_block_001_checked]
  rfl

def smallRow2617_small_right_right : List Nat := smallRow2617_block_003 ++ smallRow2617_block_004
theorem smallRow2617_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2617 = smallRow2617_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2617 ++ upperNeighbors PrimePairVertices.tree_block_004 2617 = _
  rw [smallRow2617_block_003_checked, smallRow2617_block_004_checked]
  rfl

def smallRow2617_small_right : List Nat := smallRow2617_block_002 ++ smallRow2617_small_right_right
theorem smallRow2617_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2617 = smallRow2617_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2617 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2617 = _
  rw [smallRow2617_block_002_checked, smallRow2617_small_right_right_checked]
  rfl

def smallRow2617_small : List Nat := smallRow2617_small_left ++ smallRow2617_small_right
theorem smallRow2617_small_checked : upperNeighbors PrimePairVertices.tree_small 2617 = smallRow2617_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2617 ++ upperNeighbors PrimePairVertices.tree_small_right 2617 = _
  rw [smallRow2617_small_left_checked, smallRow2617_small_right_checked]
  rfl

def smallRow2617 : List Nat := smallRow2617_small
theorem smallRow2617_checked : upperNeighbors PrimePairVertices.smallTree 2617 = smallRow2617 := smallRow2617_small_checked

def smallRow2621_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2621_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2621 = smallRow2621_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2621_block_001 : List Nat := [2789, 3623]
set_option maxRecDepth 2048 in
theorem smallRow2621_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2621 = smallRow2621_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2621_block_002 : List Nat := [4583, 4973, 5477, 6053, 6101]
set_option maxRecDepth 2048 in
theorem smallRow2621_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2621 = smallRow2621_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2621_block_003 : List Nat := [6263, 6317, 6521, 6977, 7151, 7817]
set_option maxRecDepth 2048 in
theorem smallRow2621_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2621 = smallRow2621_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2621_block_004 : List Nat := [8363]
set_option maxRecDepth 2048 in
theorem smallRow2621_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2621 = smallRow2621_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2621_small_left : List Nat := smallRow2621_block_000 ++ smallRow2621_block_001
theorem smallRow2621_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2621 = smallRow2621_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2621 ++ upperNeighbors PrimePairVertices.tree_block_001 2621 = _
  rw [smallRow2621_block_000_checked, smallRow2621_block_001_checked]
  rfl

def smallRow2621_small_right_right : List Nat := smallRow2621_block_003 ++ smallRow2621_block_004
theorem smallRow2621_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2621 = smallRow2621_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2621 ++ upperNeighbors PrimePairVertices.tree_block_004 2621 = _
  rw [smallRow2621_block_003_checked, smallRow2621_block_004_checked]
  rfl

def smallRow2621_small_right : List Nat := smallRow2621_block_002 ++ smallRow2621_small_right_right
theorem smallRow2621_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2621 = smallRow2621_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2621 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2621 = _
  rw [smallRow2621_block_002_checked, smallRow2621_small_right_right_checked]
  rfl

def smallRow2621_small : List Nat := smallRow2621_small_left ++ smallRow2621_small_right
theorem smallRow2621_small_checked : upperNeighbors PrimePairVertices.tree_small 2621 = smallRow2621_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2621 ++ upperNeighbors PrimePairVertices.tree_small_right 2621 = _
  rw [smallRow2621_small_left_checked, smallRow2621_small_right_checked]
  rfl

def smallRow2621 : List Nat := smallRow2621_small
theorem smallRow2621_checked : upperNeighbors PrimePairVertices.smallTree 2621 = smallRow2621 := smallRow2621_small_checked

def smallRow2633_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2633_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2633 = smallRow2633_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2633_block_001 : List Nat := [2699, 3929]
set_option maxRecDepth 2048 in
theorem smallRow2633_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2633 = smallRow2633_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2633_block_002 : List Nat := [4337, 5051, 5147, 5693, 5807, 5867]
set_option maxRecDepth 2048 in
theorem smallRow2633_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2633 = smallRow2633_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2633_block_003 : List Nat := [6581, 7079, 7193, 7517, 7547]
set_option maxRecDepth 2048 in
theorem smallRow2633_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2633 = smallRow2633_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2633_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow2633_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2633 = smallRow2633_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2633_small_left : List Nat := smallRow2633_block_000 ++ smallRow2633_block_001
theorem smallRow2633_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2633 = smallRow2633_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2633 ++ upperNeighbors PrimePairVertices.tree_block_001 2633 = _
  rw [smallRow2633_block_000_checked, smallRow2633_block_001_checked]
  rfl

def smallRow2633_small_right_right : List Nat := smallRow2633_block_003 ++ smallRow2633_block_004
theorem smallRow2633_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2633 = smallRow2633_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2633 ++ upperNeighbors PrimePairVertices.tree_block_004 2633 = _
  rw [smallRow2633_block_003_checked, smallRow2633_block_004_checked]
  rfl

def smallRow2633_small_right : List Nat := smallRow2633_block_002 ++ smallRow2633_small_right_right
theorem smallRow2633_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2633 = smallRow2633_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2633 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2633 = _
  rw [smallRow2633_block_002_checked, smallRow2633_small_right_right_checked]
  rfl

def smallRow2633_small : List Nat := smallRow2633_small_left ++ smallRow2633_small_right
theorem smallRow2633_small_checked : upperNeighbors PrimePairVertices.tree_small 2633 = smallRow2633_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2633 ++ upperNeighbors PrimePairVertices.tree_small_right 2633 = _
  rw [smallRow2633_small_left_checked, smallRow2633_small_right_checked]
  rfl

def smallRow2633 : List Nat := smallRow2633_small
theorem smallRow2633_checked : upperNeighbors PrimePairVertices.smallTree 2633 = smallRow2633 := smallRow2633_small_checked

def smallRow2647_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2647_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2647 = smallRow2647_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2647_block_001 : List Nat := [2851, 3793, 4057]
set_option maxRecDepth 2048 in
theorem smallRow2647_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2647 = smallRow2647_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2647_block_002 : List Nat := [4663, 5431, 6091]
set_option maxRecDepth 2048 in
theorem smallRow2647_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2647 = smallRow2647_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2647_block_003 : List Nat := [6271, 6553, 6577, 6673, 7951, 8161]
set_option maxRecDepth 2048 in
theorem smallRow2647_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2647 = smallRow2647_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2647_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2647_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2647 = smallRow2647_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2647_small_left : List Nat := smallRow2647_block_000 ++ smallRow2647_block_001
theorem smallRow2647_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2647 = smallRow2647_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2647 ++ upperNeighbors PrimePairVertices.tree_block_001 2647 = _
  rw [smallRow2647_block_000_checked, smallRow2647_block_001_checked]
  rfl

def smallRow2647_small_right_right : List Nat := smallRow2647_block_003 ++ smallRow2647_block_004
theorem smallRow2647_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2647 = smallRow2647_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2647 ++ upperNeighbors PrimePairVertices.tree_block_004 2647 = _
  rw [smallRow2647_block_003_checked, smallRow2647_block_004_checked]
  rfl

def smallRow2647_small_right : List Nat := smallRow2647_block_002 ++ smallRow2647_small_right_right
theorem smallRow2647_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2647 = smallRow2647_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2647 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2647 = _
  rw [smallRow2647_block_002_checked, smallRow2647_small_right_right_checked]
  rfl

def smallRow2647_small : List Nat := smallRow2647_small_left ++ smallRow2647_small_right
theorem smallRow2647_small_checked : upperNeighbors PrimePairVertices.tree_small 2647 = smallRow2647_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2647 ++ upperNeighbors PrimePairVertices.tree_small_right 2647 = _
  rw [smallRow2647_small_left_checked, smallRow2647_small_right_checked]
  rfl

def smallRow2647 : List Nat := smallRow2647_small
theorem smallRow2647_checked : upperNeighbors PrimePairVertices.smallTree 2647 = smallRow2647 := smallRow2647_small_checked

def smallRow2657_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2657_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2657 = smallRow2657_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2657_block_001 : List Nat := [3719, 3923]
set_option maxRecDepth 2048 in
theorem smallRow2657_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2657 = smallRow2657_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2657_block_002 : List Nat := [4391, 4517, 5099, 5279]
set_option maxRecDepth 2048 in
theorem smallRow2657_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2657 = smallRow2657_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2657_block_003 : List Nat := [6449, 6833, 7247]
set_option maxRecDepth 2048 in
theorem smallRow2657_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2657 = smallRow2657_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2657_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow2657_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2657 = smallRow2657_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2657_small_left : List Nat := smallRow2657_block_000 ++ smallRow2657_block_001
theorem smallRow2657_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2657 = smallRow2657_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2657 ++ upperNeighbors PrimePairVertices.tree_block_001 2657 = _
  rw [smallRow2657_block_000_checked, smallRow2657_block_001_checked]
  rfl

def smallRow2657_small_right_right : List Nat := smallRow2657_block_003 ++ smallRow2657_block_004
theorem smallRow2657_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2657 = smallRow2657_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2657 ++ upperNeighbors PrimePairVertices.tree_block_004 2657 = _
  rw [smallRow2657_block_003_checked, smallRow2657_block_004_checked]
  rfl

def smallRow2657_small_right : List Nat := smallRow2657_block_002 ++ smallRow2657_small_right_right
theorem smallRow2657_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2657 = smallRow2657_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2657 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2657 = _
  rw [smallRow2657_block_002_checked, smallRow2657_small_right_right_checked]
  rfl

def smallRow2657_small : List Nat := smallRow2657_small_left ++ smallRow2657_small_right
theorem smallRow2657_small_checked : upperNeighbors PrimePairVertices.tree_small 2657 = smallRow2657_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2657 ++ upperNeighbors PrimePairVertices.tree_small_right 2657 = _
  rw [smallRow2657_small_left_checked, smallRow2657_small_right_checked]
  rfl

def smallRow2657 : List Nat := smallRow2657_small
theorem smallRow2657_checked : upperNeighbors PrimePairVertices.smallTree 2657 = smallRow2657 := smallRow2657_small_checked

def smallRow2659_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2659_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2659 = smallRow2659_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2659_block_001 : List Nat := [3079, 3169, 3631, 3739, 3907]
set_option maxRecDepth 2048 in
theorem smallRow2659_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2659 = smallRow2659_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2659_block_002 : List Nat := [4423, 4723, 4759, 4933, 5167, 5827, 5857, 5869]
set_option maxRecDepth 2048 in
theorem smallRow2659_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2659 = smallRow2659_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2659_block_003 : List Nat := [6949, 7603, 7927]
set_option maxRecDepth 2048 in
theorem smallRow2659_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2659 = smallRow2659_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2659_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2659_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2659 = smallRow2659_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2659_small_left : List Nat := smallRow2659_block_000 ++ smallRow2659_block_001
theorem smallRow2659_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2659 = smallRow2659_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2659 ++ upperNeighbors PrimePairVertices.tree_block_001 2659 = _
  rw [smallRow2659_block_000_checked, smallRow2659_block_001_checked]
  rfl

def smallRow2659_small_right_right : List Nat := smallRow2659_block_003 ++ smallRow2659_block_004
theorem smallRow2659_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2659 = smallRow2659_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2659 ++ upperNeighbors PrimePairVertices.tree_block_004 2659 = _
  rw [smallRow2659_block_003_checked, smallRow2659_block_004_checked]
  rfl

def smallRow2659_small_right : List Nat := smallRow2659_block_002 ++ smallRow2659_small_right_right
theorem smallRow2659_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2659 = smallRow2659_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2659 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2659 = _
  rw [smallRow2659_block_002_checked, smallRow2659_small_right_right_checked]
  rfl

def smallRow2659_small : List Nat := smallRow2659_small_left ++ smallRow2659_small_right
theorem smallRow2659_small_checked : upperNeighbors PrimePairVertices.tree_small 2659 = smallRow2659_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2659 ++ upperNeighbors PrimePairVertices.tree_small_right 2659 = _
  rw [smallRow2659_small_left_checked, smallRow2659_small_right_checked]
  rfl

def smallRow2659 : List Nat := smallRow2659_small
theorem smallRow2659_checked : upperNeighbors PrimePairVertices.smallTree 2659 = smallRow2659 := smallRow2659_small_checked

def smallRow2663_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2663_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2663 = smallRow2663_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2663_block_001 : List Nat := [3119, 3251, 3533, 3779]
set_option maxRecDepth 2048 in
theorem smallRow2663_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2663 = smallRow2663_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2663_block_002 : List Nat := [4457, 5471, 5669, 5813, 5849]
set_option maxRecDepth 2048 in
theorem smallRow2663_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2663 = smallRow2663_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2663_block_003 : List Nat := [6359, 7607, 7901]
set_option maxRecDepth 2048 in
theorem smallRow2663_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2663 = smallRow2663_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2663_block_004 : List Nat := [8291]
set_option maxRecDepth 2048 in
theorem smallRow2663_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2663 = smallRow2663_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2663_small_left : List Nat := smallRow2663_block_000 ++ smallRow2663_block_001
theorem smallRow2663_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2663 = smallRow2663_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2663 ++ upperNeighbors PrimePairVertices.tree_block_001 2663 = _
  rw [smallRow2663_block_000_checked, smallRow2663_block_001_checked]
  rfl

def smallRow2663_small_right_right : List Nat := smallRow2663_block_003 ++ smallRow2663_block_004
theorem smallRow2663_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2663 = smallRow2663_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2663 ++ upperNeighbors PrimePairVertices.tree_block_004 2663 = _
  rw [smallRow2663_block_003_checked, smallRow2663_block_004_checked]
  rfl

def smallRow2663_small_right : List Nat := smallRow2663_block_002 ++ smallRow2663_small_right_right
theorem smallRow2663_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2663 = smallRow2663_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2663 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2663 = _
  rw [smallRow2663_block_002_checked, smallRow2663_small_right_right_checked]
  rfl

def smallRow2663_small : List Nat := smallRow2663_small_left ++ smallRow2663_small_right
theorem smallRow2663_small_checked : upperNeighbors PrimePairVertices.tree_small 2663 = smallRow2663_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2663 ++ upperNeighbors PrimePairVertices.tree_small_right 2663 = _
  rw [smallRow2663_small_left_checked, smallRow2663_small_right_checked]
  rfl

def smallRow2663 : List Nat := smallRow2663_small
theorem smallRow2663_checked : upperNeighbors PrimePairVertices.smallTree 2663 = smallRow2663 := smallRow2663_small_checked

end ProjectEulerSolutions.PrimePairCache
