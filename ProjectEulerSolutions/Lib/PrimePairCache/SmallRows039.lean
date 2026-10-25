import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2087_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2087_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2087 = smallRow2087_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2087_block_001 : List Nat := [2153, 2417, 3821]
set_option maxRecDepth 2048 in
theorem smallRow2087_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2087 = smallRow2087_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2087_block_002 : List Nat := [4463, 4937, 5009, 5333]
set_option maxRecDepth 2048 in
theorem smallRow2087_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2087 = smallRow2087_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2087_block_003 : List Nat := [6689, 7079, 7559, 7907]
set_option maxRecDepth 2048 in
theorem smallRow2087_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2087 = smallRow2087_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2087_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2087_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2087 = smallRow2087_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2087_small_left : List Nat := smallRow2087_block_000 ++ smallRow2087_block_001
theorem smallRow2087_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2087 = smallRow2087_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2087 ++ upperNeighbors PrimePairVertices.tree_block_001 2087 = _
  rw [smallRow2087_block_000_checked, smallRow2087_block_001_checked]
  rfl

def smallRow2087_small_right_right : List Nat := smallRow2087_block_003 ++ smallRow2087_block_004
theorem smallRow2087_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2087 = smallRow2087_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2087 ++ upperNeighbors PrimePairVertices.tree_block_004 2087 = _
  rw [smallRow2087_block_003_checked, smallRow2087_block_004_checked]
  rfl

def smallRow2087_small_right : List Nat := smallRow2087_block_002 ++ smallRow2087_small_right_right
theorem smallRow2087_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2087 = smallRow2087_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2087 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2087 = _
  rw [smallRow2087_block_002_checked, smallRow2087_small_right_right_checked]
  rfl

def smallRow2087_small : List Nat := smallRow2087_small_left ++ smallRow2087_small_right
theorem smallRow2087_small_checked : upperNeighbors PrimePairVertices.tree_small 2087 = smallRow2087_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2087 ++ upperNeighbors PrimePairVertices.tree_small_right 2087 = _
  rw [smallRow2087_small_left_checked, smallRow2087_small_right_checked]
  rfl

def smallRow2087 : List Nat := smallRow2087_small
theorem smallRow2087_checked : upperNeighbors PrimePairVertices.smallTree 2087 = smallRow2087 := smallRow2087_small_checked

def smallRow2089_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2089_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2089 = smallRow2089_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2089_block_001 : List Nat := [2341, 2953, 3181, 3643, 3931]
set_option maxRecDepth 2048 in
theorem smallRow2089_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2089 = smallRow2089_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2089_block_002 : List Nat := [4099, 4219, 4597, 5443, 5827]
set_option maxRecDepth 2048 in
theorem smallRow2089_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2089 = smallRow2089_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2089_block_003 : List Nat := [6277, 7039, 7237, 7297, 7411, 7873]
set_option maxRecDepth 2048 in
theorem smallRow2089_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2089 = smallRow2089_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2089_block_004 : List Nat := [8209, 8263]
set_option maxRecDepth 2048 in
theorem smallRow2089_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2089 = smallRow2089_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2089_small_left : List Nat := smallRow2089_block_000 ++ smallRow2089_block_001
theorem smallRow2089_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2089 = smallRow2089_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2089 ++ upperNeighbors PrimePairVertices.tree_block_001 2089 = _
  rw [smallRow2089_block_000_checked, smallRow2089_block_001_checked]
  rfl

def smallRow2089_small_right_right : List Nat := smallRow2089_block_003 ++ smallRow2089_block_004
theorem smallRow2089_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2089 = smallRow2089_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2089 ++ upperNeighbors PrimePairVertices.tree_block_004 2089 = _
  rw [smallRow2089_block_003_checked, smallRow2089_block_004_checked]
  rfl

def smallRow2089_small_right : List Nat := smallRow2089_block_002 ++ smallRow2089_small_right_right
theorem smallRow2089_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2089 = smallRow2089_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2089 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2089 = _
  rw [smallRow2089_block_002_checked, smallRow2089_small_right_right_checked]
  rfl

def smallRow2089_small : List Nat := smallRow2089_small_left ++ smallRow2089_small_right
theorem smallRow2089_small_checked : upperNeighbors PrimePairVertices.tree_small 2089 = smallRow2089_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2089 ++ upperNeighbors PrimePairVertices.tree_small_right 2089 = _
  rw [smallRow2089_small_left_checked, smallRow2089_small_right_checked]
  rfl

def smallRow2089 : List Nat := smallRow2089_small
theorem smallRow2089_checked : upperNeighbors PrimePairVertices.smallTree 2089 = smallRow2089 := smallRow2089_small_checked

def smallRow2099_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2099_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2099 = smallRow2099_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2099_block_001 : List Nat := [2129, 2381, 2393, 2693, 2843, 3083, 3527]
set_option maxRecDepth 2048 in
theorem smallRow2099_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2099 = smallRow2099_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2099_block_002 : List Nat := [4241, 5279, 5717]
set_option maxRecDepth 2048 in
theorem smallRow2099_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2099 = smallRow2099_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2099_block_003 : List Nat := [6863, 7211, 7517, 7841, 8069, 8093]
set_option maxRecDepth 2048 in
theorem smallRow2099_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2099 = smallRow2099_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2099_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2099_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2099 = smallRow2099_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2099_small_left : List Nat := smallRow2099_block_000 ++ smallRow2099_block_001
theorem smallRow2099_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2099 = smallRow2099_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2099 ++ upperNeighbors PrimePairVertices.tree_block_001 2099 = _
  rw [smallRow2099_block_000_checked, smallRow2099_block_001_checked]
  rfl

def smallRow2099_small_right_right : List Nat := smallRow2099_block_003 ++ smallRow2099_block_004
theorem smallRow2099_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2099 = smallRow2099_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2099 ++ upperNeighbors PrimePairVertices.tree_block_004 2099 = _
  rw [smallRow2099_block_003_checked, smallRow2099_block_004_checked]
  rfl

def smallRow2099_small_right : List Nat := smallRow2099_block_002 ++ smallRow2099_small_right_right
theorem smallRow2099_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2099 = smallRow2099_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2099 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2099 = _
  rw [smallRow2099_block_002_checked, smallRow2099_small_right_right_checked]
  rfl

def smallRow2099_small : List Nat := smallRow2099_small_left ++ smallRow2099_small_right
theorem smallRow2099_small_checked : upperNeighbors PrimePairVertices.tree_small 2099 = smallRow2099_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2099 ++ upperNeighbors PrimePairVertices.tree_small_right 2099 = _
  rw [smallRow2099_small_left_checked, smallRow2099_small_right_checked]
  rfl

def smallRow2099 : List Nat := smallRow2099_small
theorem smallRow2099_checked : upperNeighbors PrimePairVertices.smallTree 2099 = smallRow2099 := smallRow2099_small_checked

def smallRow2111_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2111_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2111 = smallRow2111_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2111_block_001 : List Nat := [2411, 2969, 2999, 3041, 3083, 3221, 3251, 3329, 3467, 4019]
set_option maxRecDepth 2048 in
theorem smallRow2111_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2111 = smallRow2111_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2111_block_002 : List Nat := [4127, 4217, 4229, 4649, 4931, 5261]
set_option maxRecDepth 2048 in
theorem smallRow2111_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2111 = smallRow2111_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2111_block_003 : List Nat := [6857, 6911, 7487, 7793]
set_option maxRecDepth 2048 in
theorem smallRow2111_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2111 = smallRow2111_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2111_block_004 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallRow2111_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2111 = smallRow2111_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2111_small_left : List Nat := smallRow2111_block_000 ++ smallRow2111_block_001
theorem smallRow2111_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2111 = smallRow2111_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2111 ++ upperNeighbors PrimePairVertices.tree_block_001 2111 = _
  rw [smallRow2111_block_000_checked, smallRow2111_block_001_checked]
  rfl

def smallRow2111_small_right_right : List Nat := smallRow2111_block_003 ++ smallRow2111_block_004
theorem smallRow2111_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2111 = smallRow2111_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2111 ++ upperNeighbors PrimePairVertices.tree_block_004 2111 = _
  rw [smallRow2111_block_003_checked, smallRow2111_block_004_checked]
  rfl

def smallRow2111_small_right : List Nat := smallRow2111_block_002 ++ smallRow2111_small_right_right
theorem smallRow2111_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2111 = smallRow2111_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2111 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2111 = _
  rw [smallRow2111_block_002_checked, smallRow2111_small_right_right_checked]
  rfl

def smallRow2111_small : List Nat := smallRow2111_small_left ++ smallRow2111_small_right
theorem smallRow2111_small_checked : upperNeighbors PrimePairVertices.tree_small 2111 = smallRow2111_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2111 ++ upperNeighbors PrimePairVertices.tree_small_right 2111 = _
  rw [smallRow2111_small_left_checked, smallRow2111_small_right_checked]
  rfl

def smallRow2111 : List Nat := smallRow2111_small
theorem smallRow2111_checked : upperNeighbors PrimePairVertices.smallTree 2111 = smallRow2111 := smallRow2111_small_checked

def smallRow2113_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2113_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2113 = smallRow2113_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2113_block_001 : List Nat := [2131, 2659, 3121, 3433, 3583, 3727, 4093]
set_option maxRecDepth 2048 in
theorem smallRow2113_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2113 = smallRow2113_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2113_block_002 : List Nat := [4507, 5227, 5419, 5431, 6067]
set_option maxRecDepth 2048 in
theorem smallRow2113_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2113 = smallRow2113_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2113_block_003 : List Nat := [6277, 6481, 6553, 7309, 7393, 7927]
set_option maxRecDepth 2048 in
theorem smallRow2113_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2113 = smallRow2113_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2113_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2113_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2113 = smallRow2113_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2113_small_left : List Nat := smallRow2113_block_000 ++ smallRow2113_block_001
theorem smallRow2113_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2113 = smallRow2113_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2113 ++ upperNeighbors PrimePairVertices.tree_block_001 2113 = _
  rw [smallRow2113_block_000_checked, smallRow2113_block_001_checked]
  rfl

def smallRow2113_small_right_right : List Nat := smallRow2113_block_003 ++ smallRow2113_block_004
theorem smallRow2113_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2113 = smallRow2113_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2113 ++ upperNeighbors PrimePairVertices.tree_block_004 2113 = _
  rw [smallRow2113_block_003_checked, smallRow2113_block_004_checked]
  rfl

def smallRow2113_small_right : List Nat := smallRow2113_block_002 ++ smallRow2113_small_right_right
theorem smallRow2113_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2113 = smallRow2113_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2113 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2113 = _
  rw [smallRow2113_block_002_checked, smallRow2113_small_right_right_checked]
  rfl

def smallRow2113_small : List Nat := smallRow2113_small_left ++ smallRow2113_small_right
theorem smallRow2113_small_checked : upperNeighbors PrimePairVertices.tree_small 2113 = smallRow2113_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2113 ++ upperNeighbors PrimePairVertices.tree_small_right 2113 = _
  rw [smallRow2113_small_left_checked, smallRow2113_small_right_checked]
  rfl

def smallRow2113 : List Nat := smallRow2113_small
theorem smallRow2113_checked : upperNeighbors PrimePairVertices.smallTree 2113 = smallRow2113 := smallRow2113_small_checked

def smallRow2129_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2129_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2129 = smallRow2129_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2129_block_001 : List Nat := [2423, 2729, 3089, 3947, 4001]
set_option maxRecDepth 2048 in
theorem smallRow2129_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2129 = smallRow2129_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2129_block_002 : List Nat := [4271, 4517, 4937, 5051, 5783]
set_option maxRecDepth 2048 in
theorem smallRow2129_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2129 = smallRow2129_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2129_block_003 : List Nat := [6203, 6269, 6491, 6791, 7121, 7193, 7247, 7559, 7937, 7949]
set_option maxRecDepth 2048 in
theorem smallRow2129_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2129 = smallRow2129_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2129_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2129_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2129 = smallRow2129_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2129_small_left : List Nat := smallRow2129_block_000 ++ smallRow2129_block_001
theorem smallRow2129_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2129 = smallRow2129_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2129 ++ upperNeighbors PrimePairVertices.tree_block_001 2129 = _
  rw [smallRow2129_block_000_checked, smallRow2129_block_001_checked]
  rfl

def smallRow2129_small_right_right : List Nat := smallRow2129_block_003 ++ smallRow2129_block_004
theorem smallRow2129_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2129 = smallRow2129_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2129 ++ upperNeighbors PrimePairVertices.tree_block_004 2129 = _
  rw [smallRow2129_block_003_checked, smallRow2129_block_004_checked]
  rfl

def smallRow2129_small_right : List Nat := smallRow2129_block_002 ++ smallRow2129_small_right_right
theorem smallRow2129_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2129 = smallRow2129_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2129 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2129 = _
  rw [smallRow2129_block_002_checked, smallRow2129_small_right_right_checked]
  rfl

def smallRow2129_small : List Nat := smallRow2129_small_left ++ smallRow2129_small_right
theorem smallRow2129_small_checked : upperNeighbors PrimePairVertices.tree_small 2129 = smallRow2129_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2129 ++ upperNeighbors PrimePairVertices.tree_small_right 2129 = _
  rw [smallRow2129_small_left_checked, smallRow2129_small_right_checked]
  rfl

def smallRow2129 : List Nat := smallRow2129_small
theorem smallRow2129_checked : upperNeighbors PrimePairVertices.smallTree 2129 = smallRow2129 := smallRow2129_small_checked

def smallRow2131_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2131_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2131 = smallRow2131_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2131_block_001 : List Nat := [2293, 2677, 2833, 3037, 3343, 3541, 3547]
set_option maxRecDepth 2048 in
theorem smallRow2131_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2131 = smallRow2131_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2131_block_002 : List Nat := [4297, 4567, 4933, 5023, 5563, 5689]
set_option maxRecDepth 2048 in
theorem smallRow2131_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2131 = smallRow2131_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2131_block_003 : List Nat := [6619, 7027, 7129, 7669, 7873]
set_option maxRecDepth 2048 in
theorem smallRow2131_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2131 = smallRow2131_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2131_block_004 : List Nat := [8377]
set_option maxRecDepth 2048 in
theorem smallRow2131_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2131 = smallRow2131_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2131_small_left : List Nat := smallRow2131_block_000 ++ smallRow2131_block_001
theorem smallRow2131_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2131 = smallRow2131_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2131 ++ upperNeighbors PrimePairVertices.tree_block_001 2131 = _
  rw [smallRow2131_block_000_checked, smallRow2131_block_001_checked]
  rfl

def smallRow2131_small_right_right : List Nat := smallRow2131_block_003 ++ smallRow2131_block_004
theorem smallRow2131_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2131 = smallRow2131_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2131 ++ upperNeighbors PrimePairVertices.tree_block_004 2131 = _
  rw [smallRow2131_block_003_checked, smallRow2131_block_004_checked]
  rfl

def smallRow2131_small_right : List Nat := smallRow2131_block_002 ++ smallRow2131_small_right_right
theorem smallRow2131_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2131 = smallRow2131_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2131 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2131 = _
  rw [smallRow2131_block_002_checked, smallRow2131_small_right_right_checked]
  rfl

def smallRow2131_small : List Nat := smallRow2131_small_left ++ smallRow2131_small_right
theorem smallRow2131_small_checked : upperNeighbors PrimePairVertices.tree_small 2131 = smallRow2131_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2131 ++ upperNeighbors PrimePairVertices.tree_small_right 2131 = _
  rw [smallRow2131_small_left_checked, smallRow2131_small_right_checked]
  rfl

def smallRow2131 : List Nat := smallRow2131_small
theorem smallRow2131_checked : upperNeighbors PrimePairVertices.smallTree 2131 = smallRow2131 := smallRow2131_small_checked

def smallRow2137_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2137_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2137 = smallRow2137_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2137_block_001 : List Nat := [2293, 2539, 3061, 3697, 4051]
set_option maxRecDepth 2048 in
theorem smallRow2137_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2137 = smallRow2137_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2137_block_002 : List Nat := [4513, 5503, 6073, 6133]
set_option maxRecDepth 2048 in
theorem smallRow2137_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2137 = smallRow2137_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2137_block_003 : List Nat := [6217, 6571, 6991, 7039, 7159, 7459, 7561]
set_option maxRecDepth 2048 in
theorem smallRow2137_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2137 = smallRow2137_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2137_block_004 : List Nat := [8233, 8317]
set_option maxRecDepth 2048 in
theorem smallRow2137_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2137 = smallRow2137_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2137_small_left : List Nat := smallRow2137_block_000 ++ smallRow2137_block_001
theorem smallRow2137_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2137 = smallRow2137_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2137 ++ upperNeighbors PrimePairVertices.tree_block_001 2137 = _
  rw [smallRow2137_block_000_checked, smallRow2137_block_001_checked]
  rfl

def smallRow2137_small_right_right : List Nat := smallRow2137_block_003 ++ smallRow2137_block_004
theorem smallRow2137_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2137 = smallRow2137_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2137 ++ upperNeighbors PrimePairVertices.tree_block_004 2137 = _
  rw [smallRow2137_block_003_checked, smallRow2137_block_004_checked]
  rfl

def smallRow2137_small_right : List Nat := smallRow2137_block_002 ++ smallRow2137_small_right_right
theorem smallRow2137_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2137 = smallRow2137_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2137 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2137 = _
  rw [smallRow2137_block_002_checked, smallRow2137_small_right_right_checked]
  rfl

def smallRow2137_small : List Nat := smallRow2137_small_left ++ smallRow2137_small_right
theorem smallRow2137_small_checked : upperNeighbors PrimePairVertices.tree_small 2137 = smallRow2137_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2137 ++ upperNeighbors PrimePairVertices.tree_small_right 2137 = _
  rw [smallRow2137_small_left_checked, smallRow2137_small_right_checked]
  rfl

def smallRow2137 : List Nat := smallRow2137_small
theorem smallRow2137_checked : upperNeighbors PrimePairVertices.smallTree 2137 = smallRow2137 := smallRow2137_small_checked

end ProjectEulerSolutions.PrimePairCache
