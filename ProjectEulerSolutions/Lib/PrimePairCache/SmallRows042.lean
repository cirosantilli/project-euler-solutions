import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2281_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2281_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2281 = smallRow2281_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2281_block_001 : List Nat := [2551, 2731, 2887]
set_option maxRecDepth 2048 in
theorem smallRow2281_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2281 = smallRow2281_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2281_block_002 : List Nat := [4201, 4273, 4591, 5857]
set_option maxRecDepth 2048 in
theorem smallRow2281_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2281 = smallRow2281_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2281_block_003 : List Nat := [6361, 6373, 6481, 6793, 7351, 8179]
set_option maxRecDepth 2048 in
theorem smallRow2281_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2281 = smallRow2281_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2281_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2281_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2281 = smallRow2281_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2281_small_left : List Nat := smallRow2281_block_000 ++ smallRow2281_block_001
theorem smallRow2281_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2281 = smallRow2281_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2281 ++ upperNeighbors PrimePairVertices.tree_block_001 2281 = _
  rw [smallRow2281_block_000_checked, smallRow2281_block_001_checked]
  rfl

def smallRow2281_small_right_right : List Nat := smallRow2281_block_003 ++ smallRow2281_block_004
theorem smallRow2281_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2281 = smallRow2281_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2281 ++ upperNeighbors PrimePairVertices.tree_block_004 2281 = _
  rw [smallRow2281_block_003_checked, smallRow2281_block_004_checked]
  rfl

def smallRow2281_small_right : List Nat := smallRow2281_block_002 ++ smallRow2281_small_right_right
theorem smallRow2281_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2281 = smallRow2281_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2281 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2281 = _
  rw [smallRow2281_block_002_checked, smallRow2281_small_right_right_checked]
  rfl

def smallRow2281_small : List Nat := smallRow2281_small_left ++ smallRow2281_small_right
theorem smallRow2281_small_checked : upperNeighbors PrimePairVertices.tree_small 2281 = smallRow2281_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2281 ++ upperNeighbors PrimePairVertices.tree_small_right 2281 = _
  rw [smallRow2281_small_left_checked, smallRow2281_small_right_checked]
  rfl

def smallRow2281 : List Nat := smallRow2281_small
theorem smallRow2281_checked : upperNeighbors PrimePairVertices.smallTree 2281 = smallRow2281 := smallRow2281_small_checked

def smallRow2287_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2287_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2287 = smallRow2287_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2287_block_001 : List Nat := [2341, 2683, 3187, 3457, 3607, 3943]
set_option maxRecDepth 2048 in
theorem smallRow2287_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2287 = smallRow2287_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2287_block_002 : List Nat := [4339, 4483, 4861, 4933, 5059, 5077, 5431, 5857]
set_option maxRecDepth 2048 in
theorem smallRow2287_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2287 = smallRow2287_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2287_block_003 : List Nat := [6277, 6793, 7537, 7681, 8101]
set_option maxRecDepth 2048 in
theorem smallRow2287_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2287 = smallRow2287_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2287_block_004 : List Nat := [8221, 8377]
set_option maxRecDepth 2048 in
theorem smallRow2287_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2287 = smallRow2287_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2287_small_left : List Nat := smallRow2287_block_000 ++ smallRow2287_block_001
theorem smallRow2287_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2287 = smallRow2287_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2287 ++ upperNeighbors PrimePairVertices.tree_block_001 2287 = _
  rw [smallRow2287_block_000_checked, smallRow2287_block_001_checked]
  rfl

def smallRow2287_small_right_right : List Nat := smallRow2287_block_003 ++ smallRow2287_block_004
theorem smallRow2287_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2287 = smallRow2287_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2287 ++ upperNeighbors PrimePairVertices.tree_block_004 2287 = _
  rw [smallRow2287_block_003_checked, smallRow2287_block_004_checked]
  rfl

def smallRow2287_small_right : List Nat := smallRow2287_block_002 ++ smallRow2287_small_right_right
theorem smallRow2287_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2287 = smallRow2287_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2287 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2287 = _
  rw [smallRow2287_block_002_checked, smallRow2287_small_right_right_checked]
  rfl

def smallRow2287_small : List Nat := smallRow2287_small_left ++ smallRow2287_small_right
theorem smallRow2287_small_checked : upperNeighbors PrimePairVertices.tree_small 2287 = smallRow2287_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2287 ++ upperNeighbors PrimePairVertices.tree_small_right 2287 = _
  rw [smallRow2287_small_left_checked, smallRow2287_small_right_checked]
  rfl

def smallRow2287 : List Nat := smallRow2287_small
theorem smallRow2287_checked : upperNeighbors PrimePairVertices.smallTree 2287 = smallRow2287 := smallRow2287_small_checked

def smallRow2293_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2293_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2293 = smallRow2293_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2293_block_001 : List Nat := [2671, 2683, 3727, 3853]
set_option maxRecDepth 2048 in
theorem smallRow2293_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2293 = smallRow2293_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2293_block_002 : List Nat := [5197, 5281, 5569, 5749, 5827]
set_option maxRecDepth 2048 in
theorem smallRow2293_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2293 = smallRow2293_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2293_block_003 : List Nat := [6829, 7039, 7333, 8011]
set_option maxRecDepth 2048 in
theorem smallRow2293_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2293 = smallRow2293_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2293_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2293_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2293 = smallRow2293_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2293_small_left : List Nat := smallRow2293_block_000 ++ smallRow2293_block_001
theorem smallRow2293_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2293 = smallRow2293_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2293 ++ upperNeighbors PrimePairVertices.tree_block_001 2293 = _
  rw [smallRow2293_block_000_checked, smallRow2293_block_001_checked]
  rfl

def smallRow2293_small_right_right : List Nat := smallRow2293_block_003 ++ smallRow2293_block_004
theorem smallRow2293_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2293 = smallRow2293_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2293 ++ upperNeighbors PrimePairVertices.tree_block_004 2293 = _
  rw [smallRow2293_block_003_checked, smallRow2293_block_004_checked]
  rfl

def smallRow2293_small_right : List Nat := smallRow2293_block_002 ++ smallRow2293_small_right_right
theorem smallRow2293_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2293 = smallRow2293_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2293 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2293 = _
  rw [smallRow2293_block_002_checked, smallRow2293_small_right_right_checked]
  rfl

def smallRow2293_small : List Nat := smallRow2293_small_left ++ smallRow2293_small_right
theorem smallRow2293_small_checked : upperNeighbors PrimePairVertices.tree_small 2293 = smallRow2293_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2293 ++ upperNeighbors PrimePairVertices.tree_small_right 2293 = _
  rw [smallRow2293_small_left_checked, smallRow2293_small_right_checked]
  rfl

def smallRow2293 : List Nat := smallRow2293_small
theorem smallRow2293_checked : upperNeighbors PrimePairVertices.smallTree 2293 = smallRow2293 := smallRow2293_small_checked

def smallRow2297_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2297_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2297 = smallRow2297_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2297_block_001 : List Nat := [2381, 2531, 3581, 4073]
set_option maxRecDepth 2048 in
theorem smallRow2297_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2297 = smallRow2297_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2297_block_002 : List Nat := [4967, 5081, 6089]
set_option maxRecDepth 2048 in
theorem smallRow2297_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2297 = smallRow2297_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2297_block_003 : List Nat := [6269, 7127, 7937]
set_option maxRecDepth 2048 in
theorem smallRow2297_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2297 = smallRow2297_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2297_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2297_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2297 = smallRow2297_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2297_small_left : List Nat := smallRow2297_block_000 ++ smallRow2297_block_001
theorem smallRow2297_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2297 = smallRow2297_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2297 ++ upperNeighbors PrimePairVertices.tree_block_001 2297 = _
  rw [smallRow2297_block_000_checked, smallRow2297_block_001_checked]
  rfl

def smallRow2297_small_right_right : List Nat := smallRow2297_block_003 ++ smallRow2297_block_004
theorem smallRow2297_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2297 = smallRow2297_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2297 ++ upperNeighbors PrimePairVertices.tree_block_004 2297 = _
  rw [smallRow2297_block_003_checked, smallRow2297_block_004_checked]
  rfl

def smallRow2297_small_right : List Nat := smallRow2297_block_002 ++ smallRow2297_small_right_right
theorem smallRow2297_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2297 = smallRow2297_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2297 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2297 = _
  rw [smallRow2297_block_002_checked, smallRow2297_small_right_right_checked]
  rfl

def smallRow2297_small : List Nat := smallRow2297_small_left ++ smallRow2297_small_right
theorem smallRow2297_small_checked : upperNeighbors PrimePairVertices.tree_small 2297 = smallRow2297_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2297 ++ upperNeighbors PrimePairVertices.tree_small_right 2297 = _
  rw [smallRow2297_small_left_checked, smallRow2297_small_right_checked]
  rfl

def smallRow2297 : List Nat := smallRow2297_small
theorem smallRow2297_checked : upperNeighbors PrimePairVertices.smallTree 2297 = smallRow2297 := smallRow2297_small_checked

def smallRow2309_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2309_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2309 = smallRow2309_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2309_block_001 : List Nat := [2687, 3491]
set_option maxRecDepth 2048 in
theorem smallRow2309_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2309 = smallRow2309_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2309_block_002 : List Nat := [4133, 4451, 5081, 5087, 5417, 5783, 6131]
set_option maxRecDepth 2048 in
theorem smallRow2309_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2309 = smallRow2309_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2309_block_003 : List Nat := [6257, 6947, 7691, 7817, 7829]
set_option maxRecDepth 2048 in
theorem smallRow2309_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2309 = smallRow2309_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2309_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow2309_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2309 = smallRow2309_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2309_small_left : List Nat := smallRow2309_block_000 ++ smallRow2309_block_001
theorem smallRow2309_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2309 = smallRow2309_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2309 ++ upperNeighbors PrimePairVertices.tree_block_001 2309 = _
  rw [smallRow2309_block_000_checked, smallRow2309_block_001_checked]
  rfl

def smallRow2309_small_right_right : List Nat := smallRow2309_block_003 ++ smallRow2309_block_004
theorem smallRow2309_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2309 = smallRow2309_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2309 ++ upperNeighbors PrimePairVertices.tree_block_004 2309 = _
  rw [smallRow2309_block_003_checked, smallRow2309_block_004_checked]
  rfl

def smallRow2309_small_right : List Nat := smallRow2309_block_002 ++ smallRow2309_small_right_right
theorem smallRow2309_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2309 = smallRow2309_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2309 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2309 = _
  rw [smallRow2309_block_002_checked, smallRow2309_small_right_right_checked]
  rfl

def smallRow2309_small : List Nat := smallRow2309_small_left ++ smallRow2309_small_right
theorem smallRow2309_small_checked : upperNeighbors PrimePairVertices.tree_small 2309 = smallRow2309_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2309 ++ upperNeighbors PrimePairVertices.tree_small_right 2309 = _
  rw [smallRow2309_small_left_checked, smallRow2309_small_right_checked]
  rfl

def smallRow2309 : List Nat := smallRow2309_small
theorem smallRow2309_checked : upperNeighbors PrimePairVertices.smallTree 2309 = smallRow2309 := smallRow2309_small_checked

def smallRow2311_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2311_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2311 = smallRow2311_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2311_block_001 : List Nat := [2647, 3061, 3313, 3529, 3847, 3943]
set_option maxRecDepth 2048 in
theorem smallRow2311_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2311 = smallRow2311_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2311_block_002 : List Nat := [4549, 5179]
set_option maxRecDepth 2048 in
theorem smallRow2311_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2311 = smallRow2311_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2311_block_003 : List Nat := [6199, 6469, 6619, 6871, 6883, 7639]
set_option maxRecDepth 2048 in
theorem smallRow2311_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2311 = smallRow2311_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2311_block_004 : List Nat := [8287]
set_option maxRecDepth 2048 in
theorem smallRow2311_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2311 = smallRow2311_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2311_small_left : List Nat := smallRow2311_block_000 ++ smallRow2311_block_001
theorem smallRow2311_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2311 = smallRow2311_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2311 ++ upperNeighbors PrimePairVertices.tree_block_001 2311 = _
  rw [smallRow2311_block_000_checked, smallRow2311_block_001_checked]
  rfl

def smallRow2311_small_right_right : List Nat := smallRow2311_block_003 ++ smallRow2311_block_004
theorem smallRow2311_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2311 = smallRow2311_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2311 ++ upperNeighbors PrimePairVertices.tree_block_004 2311 = _
  rw [smallRow2311_block_003_checked, smallRow2311_block_004_checked]
  rfl

def smallRow2311_small_right : List Nat := smallRow2311_block_002 ++ smallRow2311_small_right_right
theorem smallRow2311_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2311 = smallRow2311_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2311 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2311 = _
  rw [smallRow2311_block_002_checked, smallRow2311_small_right_right_checked]
  rfl

def smallRow2311_small : List Nat := smallRow2311_small_left ++ smallRow2311_small_right
theorem smallRow2311_small_checked : upperNeighbors PrimePairVertices.tree_small 2311 = smallRow2311_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2311 ++ upperNeighbors PrimePairVertices.tree_small_right 2311 = _
  rw [smallRow2311_small_left_checked, smallRow2311_small_right_checked]
  rfl

def smallRow2311 : List Nat := smallRow2311_small
theorem smallRow2311_checked : upperNeighbors PrimePairVertices.smallTree 2311 = smallRow2311 := smallRow2311_small_checked

def smallRow2333_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2333_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2333 = smallRow2333_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2333_block_001 : List Nat := [2339, 2423, 2843, 2963, 3137, 3221, 3533, 3701]
set_option maxRecDepth 2048 in
theorem smallRow2333_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2333 = smallRow2333_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2333_block_002 : List Nat := [5189, 5261, 5297, 5927, 5987]
set_option maxRecDepth 2048 in
theorem smallRow2333_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2333 = smallRow2333_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2333_block_003 : List Nat := [6311, 6701, 7229, 7253, 7907]
set_option maxRecDepth 2048 in
theorem smallRow2333_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2333 = smallRow2333_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2333_block_004 : List Nat := [8387]
set_option maxRecDepth 2048 in
theorem smallRow2333_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2333 = smallRow2333_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2333_small_left : List Nat := smallRow2333_block_000 ++ smallRow2333_block_001
theorem smallRow2333_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2333 = smallRow2333_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2333 ++ upperNeighbors PrimePairVertices.tree_block_001 2333 = _
  rw [smallRow2333_block_000_checked, smallRow2333_block_001_checked]
  rfl

def smallRow2333_small_right_right : List Nat := smallRow2333_block_003 ++ smallRow2333_block_004
theorem smallRow2333_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2333 = smallRow2333_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2333 ++ upperNeighbors PrimePairVertices.tree_block_004 2333 = _
  rw [smallRow2333_block_003_checked, smallRow2333_block_004_checked]
  rfl

def smallRow2333_small_right : List Nat := smallRow2333_block_002 ++ smallRow2333_small_right_right
theorem smallRow2333_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2333 = smallRow2333_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2333 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2333 = _
  rw [smallRow2333_block_002_checked, smallRow2333_small_right_right_checked]
  rfl

def smallRow2333_small : List Nat := smallRow2333_small_left ++ smallRow2333_small_right
theorem smallRow2333_small_checked : upperNeighbors PrimePairVertices.tree_small 2333 = smallRow2333_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2333 ++ upperNeighbors PrimePairVertices.tree_small_right 2333 = _
  rw [smallRow2333_small_left_checked, smallRow2333_small_right_checked]
  rfl

def smallRow2333 : List Nat := smallRow2333_small
theorem smallRow2333_checked : upperNeighbors PrimePairVertices.smallTree 2333 = smallRow2333 := smallRow2333_small_checked

def smallRow2339_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2339_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2339 = smallRow2339_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2339_block_001 : List Nat := [2543, 2729, 3347, 3371]
set_option maxRecDepth 2048 in
theorem smallRow2339_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2339 = smallRow2339_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2339_block_002 : List Nat := [4127, 4337, 5039, 5081, 5303, 6047, 6101]
set_option maxRecDepth 2048 in
theorem smallRow2339_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2339 = smallRow2339_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2339_block_003 : List Nat := [6287, 7001, 7013, 7247, 7307, 7487, 7499, 7841]
set_option maxRecDepth 2048 in
theorem smallRow2339_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2339 = smallRow2339_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2339_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow2339_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2339 = smallRow2339_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2339_small_left : List Nat := smallRow2339_block_000 ++ smallRow2339_block_001
theorem smallRow2339_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2339 = smallRow2339_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2339 ++ upperNeighbors PrimePairVertices.tree_block_001 2339 = _
  rw [smallRow2339_block_000_checked, smallRow2339_block_001_checked]
  rfl

def smallRow2339_small_right_right : List Nat := smallRow2339_block_003 ++ smallRow2339_block_004
theorem smallRow2339_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2339 = smallRow2339_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2339 ++ upperNeighbors PrimePairVertices.tree_block_004 2339 = _
  rw [smallRow2339_block_003_checked, smallRow2339_block_004_checked]
  rfl

def smallRow2339_small_right : List Nat := smallRow2339_block_002 ++ smallRow2339_small_right_right
theorem smallRow2339_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2339 = smallRow2339_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2339 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2339 = _
  rw [smallRow2339_block_002_checked, smallRow2339_small_right_right_checked]
  rfl

def smallRow2339_small : List Nat := smallRow2339_small_left ++ smallRow2339_small_right
theorem smallRow2339_small_checked : upperNeighbors PrimePairVertices.tree_small 2339 = smallRow2339_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2339 ++ upperNeighbors PrimePairVertices.tree_small_right 2339 = _
  rw [smallRow2339_small_left_checked, smallRow2339_small_right_checked]
  rfl

def smallRow2339 : List Nat := smallRow2339_small
theorem smallRow2339_checked : upperNeighbors PrimePairVertices.smallTree 2339 = smallRow2339 := smallRow2339_small_checked

end ProjectEulerSolutions.PrimePairCache
