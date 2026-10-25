import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2341_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2341_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2341 = smallRow2341_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2341_block_001 : List Nat := [2833, 3259, 3373, 3967]
set_option maxRecDepth 2048 in
theorem smallRow2341_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2341 = smallRow2341_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2341_block_002 : List Nat := [5107]
set_option maxRecDepth 2048 in
theorem smallRow2341_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2341 = smallRow2341_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2341_block_003 : List Nat := [6571, 7489, 7879, 8179]
set_option maxRecDepth 2048 in
theorem smallRow2341_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2341 = smallRow2341_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2341_block_004 : List Nat := [8377]
set_option maxRecDepth 2048 in
theorem smallRow2341_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2341 = smallRow2341_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2341_small_left : List Nat := smallRow2341_block_000 ++ smallRow2341_block_001
theorem smallRow2341_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2341 = smallRow2341_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2341 ++ upperNeighbors PrimePairVertices.tree_block_001 2341 = _
  rw [smallRow2341_block_000_checked, smallRow2341_block_001_checked]
  rfl

def smallRow2341_small_right_right : List Nat := smallRow2341_block_003 ++ smallRow2341_block_004
theorem smallRow2341_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2341 = smallRow2341_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2341 ++ upperNeighbors PrimePairVertices.tree_block_004 2341 = _
  rw [smallRow2341_block_003_checked, smallRow2341_block_004_checked]
  rfl

def smallRow2341_small_right : List Nat := smallRow2341_block_002 ++ smallRow2341_small_right_right
theorem smallRow2341_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2341 = smallRow2341_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2341 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2341 = _
  rw [smallRow2341_block_002_checked, smallRow2341_small_right_right_checked]
  rfl

def smallRow2341_small : List Nat := smallRow2341_small_left ++ smallRow2341_small_right
theorem smallRow2341_small_checked : upperNeighbors PrimePairVertices.tree_small 2341 = smallRow2341_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2341 ++ upperNeighbors PrimePairVertices.tree_small_right 2341 = _
  rw [smallRow2341_small_left_checked, smallRow2341_small_right_checked]
  rfl

def smallRow2341 : List Nat := smallRow2341_small
theorem smallRow2341_checked : upperNeighbors PrimePairVertices.smallTree 2341 = smallRow2341 := smallRow2341_small_checked

def smallRow2347_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2347_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2347 = smallRow2347_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2347_block_001 : List Nat := [2731, 2749, 2833, 2851, 3049, 3067, 3253]
set_option maxRecDepth 2048 in
theorem smallRow2347_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2347 = smallRow2347_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2347_block_002 : List Nat := [4327, 4639, 4663]
set_option maxRecDepth 2048 in
theorem smallRow2347_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2347 = smallRow2347_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2347_block_003 : List Nat := [6199, 6469, 6883, 7057, 7417, 7681, 7873, 7879]
set_option maxRecDepth 2048 in
theorem smallRow2347_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2347 = smallRow2347_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2347_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2347_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2347 = smallRow2347_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2347_small_left : List Nat := smallRow2347_block_000 ++ smallRow2347_block_001
theorem smallRow2347_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2347 = smallRow2347_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2347 ++ upperNeighbors PrimePairVertices.tree_block_001 2347 = _
  rw [smallRow2347_block_000_checked, smallRow2347_block_001_checked]
  rfl

def smallRow2347_small_right_right : List Nat := smallRow2347_block_003 ++ smallRow2347_block_004
theorem smallRow2347_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2347 = smallRow2347_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2347 ++ upperNeighbors PrimePairVertices.tree_block_004 2347 = _
  rw [smallRow2347_block_003_checked, smallRow2347_block_004_checked]
  rfl

def smallRow2347_small_right : List Nat := smallRow2347_block_002 ++ smallRow2347_small_right_right
theorem smallRow2347_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2347 = smallRow2347_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2347 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2347 = _
  rw [smallRow2347_block_002_checked, smallRow2347_small_right_right_checked]
  rfl

def smallRow2347_small : List Nat := smallRow2347_small_left ++ smallRow2347_small_right
theorem smallRow2347_small_checked : upperNeighbors PrimePairVertices.tree_small 2347 = smallRow2347_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2347 ++ upperNeighbors PrimePairVertices.tree_small_right 2347 = _
  rw [smallRow2347_small_left_checked, smallRow2347_small_right_checked]
  rfl

def smallRow2347 : List Nat := smallRow2347_small
theorem smallRow2347_checked : upperNeighbors PrimePairVertices.smallTree 2347 = smallRow2347 := smallRow2347_small_checked

def smallRow2351_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2351_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2351 = smallRow2351_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2351_block_001 : List Nat := [3617, 3863, 3881]
set_option maxRecDepth 2048 in
theorem smallRow2351_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2351 = smallRow2351_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2351_block_002 : List Nat := [4973, 5507, 5519]
set_option maxRecDepth 2048 in
theorem smallRow2351_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2351 = smallRow2351_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2351_block_003 : List Nat := [6761, 6959, 7019, 7229, 7643, 7841, 7853]
set_option maxRecDepth 2048 in
theorem smallRow2351_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2351 = smallRow2351_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2351_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow2351_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2351 = smallRow2351_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2351_small_left : List Nat := smallRow2351_block_000 ++ smallRow2351_block_001
theorem smallRow2351_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2351 = smallRow2351_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2351 ++ upperNeighbors PrimePairVertices.tree_block_001 2351 = _
  rw [smallRow2351_block_000_checked, smallRow2351_block_001_checked]
  rfl

def smallRow2351_small_right_right : List Nat := smallRow2351_block_003 ++ smallRow2351_block_004
theorem smallRow2351_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2351 = smallRow2351_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2351 ++ upperNeighbors PrimePairVertices.tree_block_004 2351 = _
  rw [smallRow2351_block_003_checked, smallRow2351_block_004_checked]
  rfl

def smallRow2351_small_right : List Nat := smallRow2351_block_002 ++ smallRow2351_small_right_right
theorem smallRow2351_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2351 = smallRow2351_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2351 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2351 = _
  rw [smallRow2351_block_002_checked, smallRow2351_small_right_right_checked]
  rfl

def smallRow2351_small : List Nat := smallRow2351_small_left ++ smallRow2351_small_right
theorem smallRow2351_small_checked : upperNeighbors PrimePairVertices.tree_small 2351 = smallRow2351_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2351 ++ upperNeighbors PrimePairVertices.tree_small_right 2351 = _
  rw [smallRow2351_small_left_checked, smallRow2351_small_right_checked]
  rfl

def smallRow2351 : List Nat := smallRow2351_small
theorem smallRow2351_checked : upperNeighbors PrimePairVertices.smallTree 2351 = smallRow2351 := smallRow2351_small_checked

def smallRow2357_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2357_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2357 = smallRow2357_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2357_block_001 : List Nat := [2579, 2903, 3467, 3527, 3821, 4091]
set_option maxRecDepth 2048 in
theorem smallRow2357_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2357 = smallRow2357_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2357_block_002 : List Nat := [4289, 5087, 5693, 5711]
set_option maxRecDepth 2048 in
theorem smallRow2357_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2357 = smallRow2357_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2357_block_003 : List Nat := [6323, 6551, 6869, 7187, 7331, 7457, 7901, 7919, 7937]
set_option maxRecDepth 2048 in
theorem smallRow2357_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2357 = smallRow2357_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2357_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow2357_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2357 = smallRow2357_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2357_small_left : List Nat := smallRow2357_block_000 ++ smallRow2357_block_001
theorem smallRow2357_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2357 = smallRow2357_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2357 ++ upperNeighbors PrimePairVertices.tree_block_001 2357 = _
  rw [smallRow2357_block_000_checked, smallRow2357_block_001_checked]
  rfl

def smallRow2357_small_right_right : List Nat := smallRow2357_block_003 ++ smallRow2357_block_004
theorem smallRow2357_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2357 = smallRow2357_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2357 ++ upperNeighbors PrimePairVertices.tree_block_004 2357 = _
  rw [smallRow2357_block_003_checked, smallRow2357_block_004_checked]
  rfl

def smallRow2357_small_right : List Nat := smallRow2357_block_002 ++ smallRow2357_small_right_right
theorem smallRow2357_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2357 = smallRow2357_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2357 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2357 = _
  rw [smallRow2357_block_002_checked, smallRow2357_small_right_right_checked]
  rfl

def smallRow2357_small : List Nat := smallRow2357_small_left ++ smallRow2357_small_right
theorem smallRow2357_small_checked : upperNeighbors PrimePairVertices.tree_small 2357 = smallRow2357_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2357 ++ upperNeighbors PrimePairVertices.tree_small_right 2357 = _
  rw [smallRow2357_small_left_checked, smallRow2357_small_right_checked]
  rfl

def smallRow2357 : List Nat := smallRow2357_small
theorem smallRow2357_checked : upperNeighbors PrimePairVertices.smallTree 2357 = smallRow2357 := smallRow2357_small_checked

def smallRow2371_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2371_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2371 = smallRow2371_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2371_block_001 : List Nat := [2917, 3169, 3229, 3607, 3643, 4021]
set_option maxRecDepth 2048 in
theorem smallRow2371_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2371 = smallRow2371_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2371_block_002 : List Nat := [4423]
set_option maxRecDepth 2048 in
theorem smallRow2371_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2371 = smallRow2371_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2371_block_003 : List Nat := [7219]
set_option maxRecDepth 2048 in
theorem smallRow2371_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2371 = smallRow2371_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2371_block_004 : List Nat := [8287]
set_option maxRecDepth 2048 in
theorem smallRow2371_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2371 = smallRow2371_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2371_small_left : List Nat := smallRow2371_block_000 ++ smallRow2371_block_001
theorem smallRow2371_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2371 = smallRow2371_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2371 ++ upperNeighbors PrimePairVertices.tree_block_001 2371 = _
  rw [smallRow2371_block_000_checked, smallRow2371_block_001_checked]
  rfl

def smallRow2371_small_right_right : List Nat := smallRow2371_block_003 ++ smallRow2371_block_004
theorem smallRow2371_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2371 = smallRow2371_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2371 ++ upperNeighbors PrimePairVertices.tree_block_004 2371 = _
  rw [smallRow2371_block_003_checked, smallRow2371_block_004_checked]
  rfl

def smallRow2371_small_right : List Nat := smallRow2371_block_002 ++ smallRow2371_small_right_right
theorem smallRow2371_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2371 = smallRow2371_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2371 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2371 = _
  rw [smallRow2371_block_002_checked, smallRow2371_small_right_right_checked]
  rfl

def smallRow2371_small : List Nat := smallRow2371_small_left ++ smallRow2371_small_right
theorem smallRow2371_small_checked : upperNeighbors PrimePairVertices.tree_small 2371 = smallRow2371_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2371 ++ upperNeighbors PrimePairVertices.tree_small_right 2371 = _
  rw [smallRow2371_small_left_checked, smallRow2371_small_right_checked]
  rfl

def smallRow2371 : List Nat := smallRow2371_small
theorem smallRow2371_checked : upperNeighbors PrimePairVertices.smallTree 2371 = smallRow2371 := smallRow2371_small_checked

def smallRow2377_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2377_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2377 = smallRow2377_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2377_block_001 : List Nat := [2557, 2797, 3067, 3319, 3433]
set_option maxRecDepth 2048 in
theorem smallRow2377_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2377 = smallRow2377_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2377_block_002 : List Nat := [4159, 4327, 5743, 6073]
set_option maxRecDepth 2048 in
theorem smallRow2377_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2377 = smallRow2377_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2377_block_003 : List Nat := [6199, 6553, 7603]
set_option maxRecDepth 2048 in
theorem smallRow2377_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2377 = smallRow2377_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2377_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2377_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2377 = smallRow2377_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2377_small_left : List Nat := smallRow2377_block_000 ++ smallRow2377_block_001
theorem smallRow2377_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2377 = smallRow2377_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2377 ++ upperNeighbors PrimePairVertices.tree_block_001 2377 = _
  rw [smallRow2377_block_000_checked, smallRow2377_block_001_checked]
  rfl

def smallRow2377_small_right_right : List Nat := smallRow2377_block_003 ++ smallRow2377_block_004
theorem smallRow2377_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2377 = smallRow2377_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2377 ++ upperNeighbors PrimePairVertices.tree_block_004 2377 = _
  rw [smallRow2377_block_003_checked, smallRow2377_block_004_checked]
  rfl

def smallRow2377_small_right : List Nat := smallRow2377_block_002 ++ smallRow2377_small_right_right
theorem smallRow2377_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2377 = smallRow2377_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2377 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2377 = _
  rw [smallRow2377_block_002_checked, smallRow2377_small_right_right_checked]
  rfl

def smallRow2377_small : List Nat := smallRow2377_small_left ++ smallRow2377_small_right
theorem smallRow2377_small_checked : upperNeighbors PrimePairVertices.tree_small 2377 = smallRow2377_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2377 ++ upperNeighbors PrimePairVertices.tree_small_right 2377 = _
  rw [smallRow2377_small_left_checked, smallRow2377_small_right_checked]
  rfl

def smallRow2377 : List Nat := smallRow2377_small
theorem smallRow2377_checked : upperNeighbors PrimePairVertices.smallTree 2377 = smallRow2377 := smallRow2377_small_checked

def smallRow2381_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2381_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2381 = smallRow2381_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2381_block_001 : List Nat := [2447, 3221, 3539, 4073]
set_option maxRecDepth 2048 in
theorem smallRow2381_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2381 = smallRow2381_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2381_block_002 : List Nat := [5483, 5501, 5639, 5807]
set_option maxRecDepth 2048 in
theorem smallRow2381_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2381 = smallRow2381_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2381_block_003 : List Nat := [6563, 6857, 7487, 7853, 8117]
set_option maxRecDepth 2048 in
theorem smallRow2381_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2381 = smallRow2381_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2381_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow2381_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2381 = smallRow2381_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2381_small_left : List Nat := smallRow2381_block_000 ++ smallRow2381_block_001
theorem smallRow2381_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2381 = smallRow2381_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2381 ++ upperNeighbors PrimePairVertices.tree_block_001 2381 = _
  rw [smallRow2381_block_000_checked, smallRow2381_block_001_checked]
  rfl

def smallRow2381_small_right_right : List Nat := smallRow2381_block_003 ++ smallRow2381_block_004
theorem smallRow2381_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2381 = smallRow2381_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2381 ++ upperNeighbors PrimePairVertices.tree_block_004 2381 = _
  rw [smallRow2381_block_003_checked, smallRow2381_block_004_checked]
  rfl

def smallRow2381_small_right : List Nat := smallRow2381_block_002 ++ smallRow2381_small_right_right
theorem smallRow2381_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2381 = smallRow2381_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2381 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2381 = _
  rw [smallRow2381_block_002_checked, smallRow2381_small_right_right_checked]
  rfl

def smallRow2381_small : List Nat := smallRow2381_small_left ++ smallRow2381_small_right
theorem smallRow2381_small_checked : upperNeighbors PrimePairVertices.tree_small 2381 = smallRow2381_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2381 ++ upperNeighbors PrimePairVertices.tree_small_right 2381 = _
  rw [smallRow2381_small_left_checked, smallRow2381_small_right_checked]
  rfl

def smallRow2381 : List Nat := smallRow2381_small
theorem smallRow2381_checked : upperNeighbors PrimePairVertices.smallTree 2381 = smallRow2381 := smallRow2381_small_checked

def smallRow2383_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2383_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2383 = smallRow2383_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2383_block_001 : List Nat := [2659, 3583, 4021]
set_option maxRecDepth 2048 in
theorem smallRow2383_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2383 = smallRow2383_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2383_block_002 : List Nat := [4099, 4273, 4639, 5023, 5113, 5431, 5527]
set_option maxRecDepth 2048 in
theorem smallRow2383_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2383 = smallRow2383_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2383_block_003 : List Nat := [6733, 7129, 7573, 7873]
set_option maxRecDepth 2048 in
theorem smallRow2383_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2383 = smallRow2383_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2383_block_004 : List Nat := [8263]
set_option maxRecDepth 2048 in
theorem smallRow2383_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2383 = smallRow2383_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2383_small_left : List Nat := smallRow2383_block_000 ++ smallRow2383_block_001
theorem smallRow2383_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2383 = smallRow2383_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2383 ++ upperNeighbors PrimePairVertices.tree_block_001 2383 = _
  rw [smallRow2383_block_000_checked, smallRow2383_block_001_checked]
  rfl

def smallRow2383_small_right_right : List Nat := smallRow2383_block_003 ++ smallRow2383_block_004
theorem smallRow2383_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2383 = smallRow2383_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2383 ++ upperNeighbors PrimePairVertices.tree_block_004 2383 = _
  rw [smallRow2383_block_003_checked, smallRow2383_block_004_checked]
  rfl

def smallRow2383_small_right : List Nat := smallRow2383_block_002 ++ smallRow2383_small_right_right
theorem smallRow2383_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2383 = smallRow2383_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2383 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2383 = _
  rw [smallRow2383_block_002_checked, smallRow2383_small_right_right_checked]
  rfl

def smallRow2383_small : List Nat := smallRow2383_small_left ++ smallRow2383_small_right
theorem smallRow2383_small_checked : upperNeighbors PrimePairVertices.tree_small 2383 = smallRow2383_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2383 ++ upperNeighbors PrimePairVertices.tree_small_right 2383 = _
  rw [smallRow2383_small_left_checked, smallRow2383_small_right_checked]
  rfl

def smallRow2383 : List Nat := smallRow2383_small
theorem smallRow2383_checked : upperNeighbors PrimePairVertices.smallTree 2383 = smallRow2383 := smallRow2383_small_checked

end ProjectEulerSolutions.PrimePairCache
