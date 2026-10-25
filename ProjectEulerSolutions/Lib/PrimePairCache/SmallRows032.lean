import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1637_block_000 : List Nat := [1709, 1889]
set_option maxRecDepth 2048 in
theorem smallRow1637_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1637 = smallRow1637_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1637_block_001 : List Nat := [2903, 3359, 3911, 4073]
set_option maxRecDepth 2048 in
theorem smallRow1637_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1637 = smallRow1637_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1637_block_002 : List Nat := [4649, 4931, 4973, 5669, 5897, 6053]
set_option maxRecDepth 2048 in
theorem smallRow1637_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1637 = smallRow1637_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1637_block_003 : List Nat := [7013, 7607, 7841, 8009, 8039]
set_option maxRecDepth 2048 in
theorem smallRow1637_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1637 = smallRow1637_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1637_block_004 : List Nat := [8237, 8363]
set_option maxRecDepth 2048 in
theorem smallRow1637_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1637 = smallRow1637_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1637_small_left : List Nat := smallRow1637_block_000 ++ smallRow1637_block_001
theorem smallRow1637_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1637 = smallRow1637_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1637 ++ upperNeighbors PrimePairVertices.tree_block_001 1637 = _
  rw [smallRow1637_block_000_checked, smallRow1637_block_001_checked]
  rfl

def smallRow1637_small_right_right : List Nat := smallRow1637_block_003 ++ smallRow1637_block_004
theorem smallRow1637_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1637 = smallRow1637_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1637 ++ upperNeighbors PrimePairVertices.tree_block_004 1637 = _
  rw [smallRow1637_block_003_checked, smallRow1637_block_004_checked]
  rfl

def smallRow1637_small_right : List Nat := smallRow1637_block_002 ++ smallRow1637_small_right_right
theorem smallRow1637_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1637 = smallRow1637_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1637 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1637 = _
  rw [smallRow1637_block_002_checked, smallRow1637_small_right_right_checked]
  rfl

def smallRow1637_small : List Nat := smallRow1637_small_left ++ smallRow1637_small_right
theorem smallRow1637_small_checked : upperNeighbors PrimePairVertices.tree_small 1637 = smallRow1637_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1637 ++ upperNeighbors PrimePairVertices.tree_small_right 1637 = _
  rw [smallRow1637_small_left_checked, smallRow1637_small_right_checked]
  rfl

def smallRow1637 : List Nat := smallRow1637_small
theorem smallRow1637_checked : upperNeighbors PrimePairVertices.smallTree 1637 = smallRow1637 := smallRow1637_small_checked

def smallRow1657_block_000 : List Nat := [1699, 1879]
set_option maxRecDepth 2048 in
theorem smallRow1657_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1657 = smallRow1657_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1657_block_001 : List Nat := [2203, 2683, 2791, 2887, 2917, 3463, 3967]
set_option maxRecDepth 2048 in
theorem smallRow1657_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1657 = smallRow1657_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1657_block_002 : List Nat := [4111, 4273, 4549, 4969, 5059, 5101]
set_option maxRecDepth 2048 in
theorem smallRow1657_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1657 = smallRow1657_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1657_block_003 : List Nat := [6397, 6733, 6907, 7237, 8017, 8167]
set_option maxRecDepth 2048 in
theorem smallRow1657_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1657 = smallRow1657_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1657_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow1657_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1657 = smallRow1657_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1657_small_left : List Nat := smallRow1657_block_000 ++ smallRow1657_block_001
theorem smallRow1657_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1657 = smallRow1657_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1657 ++ upperNeighbors PrimePairVertices.tree_block_001 1657 = _
  rw [smallRow1657_block_000_checked, smallRow1657_block_001_checked]
  rfl

def smallRow1657_small_right_right : List Nat := smallRow1657_block_003 ++ smallRow1657_block_004
theorem smallRow1657_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1657 = smallRow1657_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1657 ++ upperNeighbors PrimePairVertices.tree_block_004 1657 = _
  rw [smallRow1657_block_003_checked, smallRow1657_block_004_checked]
  rfl

def smallRow1657_small_right : List Nat := smallRow1657_block_002 ++ smallRow1657_small_right_right
theorem smallRow1657_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1657 = smallRow1657_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1657 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1657 = _
  rw [smallRow1657_block_002_checked, smallRow1657_small_right_right_checked]
  rfl

def smallRow1657_small : List Nat := smallRow1657_small_left ++ smallRow1657_small_right
theorem smallRow1657_small_checked : upperNeighbors PrimePairVertices.tree_small 1657 = smallRow1657_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1657 ++ upperNeighbors PrimePairVertices.tree_small_right 1657 = _
  rw [smallRow1657_small_left_checked, smallRow1657_small_right_checked]
  rfl

def smallRow1657 : List Nat := smallRow1657_small
theorem smallRow1657_checked : upperNeighbors PrimePairVertices.smallTree 1657 = smallRow1657 := smallRow1657_small_checked

def smallRow1663_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1663_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1663 = smallRow1663_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1663_block_001 : List Nat := [2857, 3229, 3613, 4027]
set_option maxRecDepth 2048 in
theorem smallRow1663_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1663 = smallRow1663_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1663_block_002 : List Nat := [4231, 4831, 4987, 5413, 5449, 5653, 6007]
set_option maxRecDepth 2048 in
theorem smallRow1663_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1663 = smallRow1663_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1663_block_003 : List Nat := [6577, 7219, 7879, 7963]
set_option maxRecDepth 2048 in
theorem smallRow1663_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1663 = smallRow1663_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1663_block_004 : List Nat := [8221]
set_option maxRecDepth 2048 in
theorem smallRow1663_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1663 = smallRow1663_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1663_small_left : List Nat := smallRow1663_block_000 ++ smallRow1663_block_001
theorem smallRow1663_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1663 = smallRow1663_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1663 ++ upperNeighbors PrimePairVertices.tree_block_001 1663 = _
  rw [smallRow1663_block_000_checked, smallRow1663_block_001_checked]
  rfl

def smallRow1663_small_right_right : List Nat := smallRow1663_block_003 ++ smallRow1663_block_004
theorem smallRow1663_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1663 = smallRow1663_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1663 ++ upperNeighbors PrimePairVertices.tree_block_004 1663 = _
  rw [smallRow1663_block_003_checked, smallRow1663_block_004_checked]
  rfl

def smallRow1663_small_right : List Nat := smallRow1663_block_002 ++ smallRow1663_small_right_right
theorem smallRow1663_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1663 = smallRow1663_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1663 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1663 = _
  rw [smallRow1663_block_002_checked, smallRow1663_small_right_right_checked]
  rfl

def smallRow1663_small : List Nat := smallRow1663_small_left ++ smallRow1663_small_right
theorem smallRow1663_small_checked : upperNeighbors PrimePairVertices.tree_small 1663 = smallRow1663_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1663 ++ upperNeighbors PrimePairVertices.tree_small_right 1663 = _
  rw [smallRow1663_small_left_checked, smallRow1663_small_right_checked]
  rfl

def smallRow1663 : List Nat := smallRow1663_small
theorem smallRow1663_checked : upperNeighbors PrimePairVertices.smallTree 1663 = smallRow1663 := smallRow1663_small_checked

def smallRow1667_block_000 : List Nat := [1901, 1973]
set_option maxRecDepth 2048 in
theorem smallRow1667_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1667 = smallRow1667_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1667_block_001 : List Nat := [2153, 2267, 2657, 3623, 3929]
set_option maxRecDepth 2048 in
theorem smallRow1667_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1667 = smallRow1667_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1667_block_002 : List Nat := [4157, 4481, 4703, 4817, 4967, 5051, 5531, 5573, 5669]
set_option maxRecDepth 2048 in
theorem smallRow1667_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1667 = smallRow1667_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1667_block_003 : List Nat := [6287, 6323, 6329, 6353, 6701]
set_option maxRecDepth 2048 in
theorem smallRow1667_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1667 = smallRow1667_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1667_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1667_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1667 = smallRow1667_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1667_small_left : List Nat := smallRow1667_block_000 ++ smallRow1667_block_001
theorem smallRow1667_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1667 = smallRow1667_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1667 ++ upperNeighbors PrimePairVertices.tree_block_001 1667 = _
  rw [smallRow1667_block_000_checked, smallRow1667_block_001_checked]
  rfl

def smallRow1667_small_right_right : List Nat := smallRow1667_block_003 ++ smallRow1667_block_004
theorem smallRow1667_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1667 = smallRow1667_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1667 ++ upperNeighbors PrimePairVertices.tree_block_004 1667 = _
  rw [smallRow1667_block_003_checked, smallRow1667_block_004_checked]
  rfl

def smallRow1667_small_right : List Nat := smallRow1667_block_002 ++ smallRow1667_small_right_right
theorem smallRow1667_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1667 = smallRow1667_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1667 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1667 = _
  rw [smallRow1667_block_002_checked, smallRow1667_small_right_right_checked]
  rfl

def smallRow1667_small : List Nat := smallRow1667_small_left ++ smallRow1667_small_right
theorem smallRow1667_small_checked : upperNeighbors PrimePairVertices.tree_small 1667 = smallRow1667_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1667 ++ upperNeighbors PrimePairVertices.tree_small_right 1667 = _
  rw [smallRow1667_small_left_checked, smallRow1667_small_right_checked]
  rfl

def smallRow1667 : List Nat := smallRow1667_small
theorem smallRow1667_checked : upperNeighbors PrimePairVertices.smallTree 1667 = smallRow1667 := smallRow1667_small_checked

def smallRow1669_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1669_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1669 = smallRow1669_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1669_block_001 : List Nat := [2593, 2707, 2791, 3181, 3301, 3499, 3511]
set_option maxRecDepth 2048 in
theorem smallRow1669_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1669 = smallRow1669_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1669_block_002 : List Nat := [4273, 4441, 4651, 5659, 5743]
set_option maxRecDepth 2048 in
theorem smallRow1669_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1669 = smallRow1669_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1669_block_003 : List Nat := [6577, 6661, 6709, 6733, 7669, 7867, 7873]
set_option maxRecDepth 2048 in
theorem smallRow1669_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1669 = smallRow1669_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1669_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1669_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1669 = smallRow1669_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1669_small_left : List Nat := smallRow1669_block_000 ++ smallRow1669_block_001
theorem smallRow1669_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1669 = smallRow1669_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1669 ++ upperNeighbors PrimePairVertices.tree_block_001 1669 = _
  rw [smallRow1669_block_000_checked, smallRow1669_block_001_checked]
  rfl

def smallRow1669_small_right_right : List Nat := smallRow1669_block_003 ++ smallRow1669_block_004
theorem smallRow1669_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1669 = smallRow1669_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1669 ++ upperNeighbors PrimePairVertices.tree_block_004 1669 = _
  rw [smallRow1669_block_003_checked, smallRow1669_block_004_checked]
  rfl

def smallRow1669_small_right : List Nat := smallRow1669_block_002 ++ smallRow1669_small_right_right
theorem smallRow1669_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1669 = smallRow1669_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1669 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1669 = _
  rw [smallRow1669_block_002_checked, smallRow1669_small_right_right_checked]
  rfl

def smallRow1669_small : List Nat := smallRow1669_small_left ++ smallRow1669_small_right
theorem smallRow1669_small_checked : upperNeighbors PrimePairVertices.tree_small 1669 = smallRow1669_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1669 ++ upperNeighbors PrimePairVertices.tree_small_right 1669 = _
  rw [smallRow1669_small_left_checked, smallRow1669_small_right_checked]
  rfl

def smallRow1669 : List Nat := smallRow1669_small
theorem smallRow1669_checked : upperNeighbors PrimePairVertices.smallTree 1669 = smallRow1669 := smallRow1669_small_checked

def smallRow1693_block_000 : List Nat := [1753]
set_option maxRecDepth 2048 in
theorem smallRow1693_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1693 = smallRow1693_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1693_block_001 : List Nat := [2371, 2551, 2593, 2689, 3181, 3373, 3739, 3919]
set_option maxRecDepth 2048 in
theorem smallRow1693_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1693 = smallRow1693_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1693_block_002 : List Nat := [4327, 4441, 4639]
set_option maxRecDepth 2048 in
theorem smallRow1693_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1693 = smallRow1693_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1693_block_003 : List Nat := [6217, 6991, 7309, 7477, 7489, 7927]
set_option maxRecDepth 2048 in
theorem smallRow1693_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1693 = smallRow1693_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1693_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1693_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1693 = smallRow1693_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1693_small_left : List Nat := smallRow1693_block_000 ++ smallRow1693_block_001
theorem smallRow1693_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1693 = smallRow1693_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1693 ++ upperNeighbors PrimePairVertices.tree_block_001 1693 = _
  rw [smallRow1693_block_000_checked, smallRow1693_block_001_checked]
  rfl

def smallRow1693_small_right_right : List Nat := smallRow1693_block_003 ++ smallRow1693_block_004
theorem smallRow1693_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1693 = smallRow1693_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1693 ++ upperNeighbors PrimePairVertices.tree_block_004 1693 = _
  rw [smallRow1693_block_003_checked, smallRow1693_block_004_checked]
  rfl

def smallRow1693_small_right : List Nat := smallRow1693_block_002 ++ smallRow1693_small_right_right
theorem smallRow1693_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1693 = smallRow1693_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1693 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1693 = _
  rw [smallRow1693_block_002_checked, smallRow1693_small_right_right_checked]
  rfl

def smallRow1693_small : List Nat := smallRow1693_small_left ++ smallRow1693_small_right
theorem smallRow1693_small_checked : upperNeighbors PrimePairVertices.tree_small 1693 = smallRow1693_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1693 ++ upperNeighbors PrimePairVertices.tree_small_right 1693 = _
  rw [smallRow1693_small_left_checked, smallRow1693_small_right_checked]
  rfl

def smallRow1693 : List Nat := smallRow1693_small
theorem smallRow1693_checked : upperNeighbors PrimePairVertices.smallTree 1693 = smallRow1693 := smallRow1693_small_checked

def smallRow1697_block_000 : List Nat := [1979, 2027]
set_option maxRecDepth 2048 in
theorem smallRow1697_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1697 = smallRow1697_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1697_block_001 : List Nat := [2741, 3041, 3527]
set_option maxRecDepth 2048 in
theorem smallRow1697_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1697 = smallRow1697_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1697_block_002 : List Nat := [4931, 5087, 5099, 5477]
set_option maxRecDepth 2048 in
theorem smallRow1697_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1697 = smallRow1697_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1697_block_003 : List Nat := [6689, 7193, 7643, 7901]
set_option maxRecDepth 2048 in
theorem smallRow1697_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1697 = smallRow1697_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1697_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow1697_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1697 = smallRow1697_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1697_small_left : List Nat := smallRow1697_block_000 ++ smallRow1697_block_001
theorem smallRow1697_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1697 = smallRow1697_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1697 ++ upperNeighbors PrimePairVertices.tree_block_001 1697 = _
  rw [smallRow1697_block_000_checked, smallRow1697_block_001_checked]
  rfl

def smallRow1697_small_right_right : List Nat := smallRow1697_block_003 ++ smallRow1697_block_004
theorem smallRow1697_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1697 = smallRow1697_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1697 ++ upperNeighbors PrimePairVertices.tree_block_004 1697 = _
  rw [smallRow1697_block_003_checked, smallRow1697_block_004_checked]
  rfl

def smallRow1697_small_right : List Nat := smallRow1697_block_002 ++ smallRow1697_small_right_right
theorem smallRow1697_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1697 = smallRow1697_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1697 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1697 = _
  rw [smallRow1697_block_002_checked, smallRow1697_small_right_right_checked]
  rfl

def smallRow1697_small : List Nat := smallRow1697_small_left ++ smallRow1697_small_right
theorem smallRow1697_small_checked : upperNeighbors PrimePairVertices.tree_small 1697 = smallRow1697_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1697 ++ upperNeighbors PrimePairVertices.tree_small_right 1697 = _
  rw [smallRow1697_small_left_checked, smallRow1697_small_right_checked]
  rfl

def smallRow1697 : List Nat := smallRow1697_small
theorem smallRow1697_checked : upperNeighbors PrimePairVertices.smallTree 1697 = smallRow1697 := smallRow1697_small_checked

def smallRow1699_block_000 : List Nat := [1759, 1861]
set_option maxRecDepth 2048 in
theorem smallRow1699_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1699 = smallRow1699_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1699_block_001 : List Nat := [2179, 2203, 2851, 3313, 3373, 3607, 3853]
set_option maxRecDepth 2048 in
theorem smallRow1699_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1699 = smallRow1699_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1699_block_002 : List Nat := [4177, 4363, 4567, 5659, 5791, 6121]
set_option maxRecDepth 2048 in
theorem smallRow1699_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1699 = smallRow1699_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1699_block_003 : List Nat := [6571, 6691, 7507, 7549, 7639]
set_option maxRecDepth 2048 in
theorem smallRow1699_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1699 = smallRow1699_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1699_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1699_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1699 = smallRow1699_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1699_small_left : List Nat := smallRow1699_block_000 ++ smallRow1699_block_001
theorem smallRow1699_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1699 = smallRow1699_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1699 ++ upperNeighbors PrimePairVertices.tree_block_001 1699 = _
  rw [smallRow1699_block_000_checked, smallRow1699_block_001_checked]
  rfl

def smallRow1699_small_right_right : List Nat := smallRow1699_block_003 ++ smallRow1699_block_004
theorem smallRow1699_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1699 = smallRow1699_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1699 ++ upperNeighbors PrimePairVertices.tree_block_004 1699 = _
  rw [smallRow1699_block_003_checked, smallRow1699_block_004_checked]
  rfl

def smallRow1699_small_right : List Nat := smallRow1699_block_002 ++ smallRow1699_small_right_right
theorem smallRow1699_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1699 = smallRow1699_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1699 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1699 = _
  rw [smallRow1699_block_002_checked, smallRow1699_small_right_right_checked]
  rfl

def smallRow1699_small : List Nat := smallRow1699_small_left ++ smallRow1699_small_right
theorem smallRow1699_small_checked : upperNeighbors PrimePairVertices.tree_small 1699 = smallRow1699_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1699 ++ upperNeighbors PrimePairVertices.tree_small_right 1699 = _
  rw [smallRow1699_small_left_checked, smallRow1699_small_right_checked]
  rfl

def smallRow1699 : List Nat := smallRow1699_small
theorem smallRow1699_checked : upperNeighbors PrimePairVertices.smallTree 1699 = smallRow1699 := smallRow1699_small_checked

end ProjectEulerSolutions.PrimePairCache
