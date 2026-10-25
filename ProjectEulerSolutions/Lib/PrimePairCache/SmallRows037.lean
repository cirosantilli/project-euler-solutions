import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1979_block_000 : List Nat := [2039]
set_option maxRecDepth 2048 in
theorem smallRow1979_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1979 = smallRow1979_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1979_block_001 : List Nat := [2207, 2693, 2999, 3911]
set_option maxRecDepth 2048 in
theorem smallRow1979_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1979 = smallRow1979_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1979_block_002 : List Nat := [4241, 4937]
set_option maxRecDepth 2048 in
theorem smallRow1979_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1979 = smallRow1979_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1979_block_003 : List Nat := [6869, 7187, 7433]
set_option maxRecDepth 2048 in
theorem smallRow1979_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1979 = smallRow1979_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1979_block_004 : List Nat := [8291, 8297]
set_option maxRecDepth 2048 in
theorem smallRow1979_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1979 = smallRow1979_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1979_small_left : List Nat := smallRow1979_block_000 ++ smallRow1979_block_001
theorem smallRow1979_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1979 = smallRow1979_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1979 ++ upperNeighbors PrimePairVertices.tree_block_001 1979 = _
  rw [smallRow1979_block_000_checked, smallRow1979_block_001_checked]
  rfl

def smallRow1979_small_right_right : List Nat := smallRow1979_block_003 ++ smallRow1979_block_004
theorem smallRow1979_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1979 = smallRow1979_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1979 ++ upperNeighbors PrimePairVertices.tree_block_004 1979 = _
  rw [smallRow1979_block_003_checked, smallRow1979_block_004_checked]
  rfl

def smallRow1979_small_right : List Nat := smallRow1979_block_002 ++ smallRow1979_small_right_right
theorem smallRow1979_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1979 = smallRow1979_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1979 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1979 = _
  rw [smallRow1979_block_002_checked, smallRow1979_small_right_right_checked]
  rfl

def smallRow1979_small : List Nat := smallRow1979_small_left ++ smallRow1979_small_right
theorem smallRow1979_small_checked : upperNeighbors PrimePairVertices.tree_small 1979 = smallRow1979_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1979 ++ upperNeighbors PrimePairVertices.tree_small_right 1979 = _
  rw [smallRow1979_small_left_checked, smallRow1979_small_right_checked]
  rfl

def smallRow1979 : List Nat := smallRow1979_small
theorem smallRow1979_checked : upperNeighbors PrimePairVertices.smallTree 1979 = smallRow1979 := smallRow1979_small_checked

def smallRow1987_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1987_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1987 = smallRow1987_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1987_block_001 : List Nat := [2131, 2383, 2467, 2551, 2707, 2719, 3079, 3331, 3361, 3499, 3517, 3709, 3877]
set_option maxRecDepth 2048 in
theorem smallRow1987_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1987 = smallRow1987_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1987_block_002 : List Nat := [4423, 4639, 4801, 5521, 5563, 5659, 6079, 6133]
set_option maxRecDepth 2048 in
theorem smallRow1987_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1987 = smallRow1987_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1987_block_003 : List Nat := [6481, 7027, 7237, 7549, 8191]
set_option maxRecDepth 2048 in
theorem smallRow1987_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1987 = smallRow1987_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1987_block_004 : List Nat := [8263]
set_option maxRecDepth 2048 in
theorem smallRow1987_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1987 = smallRow1987_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1987_small_left : List Nat := smallRow1987_block_000 ++ smallRow1987_block_001
theorem smallRow1987_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1987 = smallRow1987_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1987 ++ upperNeighbors PrimePairVertices.tree_block_001 1987 = _
  rw [smallRow1987_block_000_checked, smallRow1987_block_001_checked]
  rfl

def smallRow1987_small_right_right : List Nat := smallRow1987_block_003 ++ smallRow1987_block_004
theorem smallRow1987_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1987 = smallRow1987_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1987 ++ upperNeighbors PrimePairVertices.tree_block_004 1987 = _
  rw [smallRow1987_block_003_checked, smallRow1987_block_004_checked]
  rfl

def smallRow1987_small_right : List Nat := smallRow1987_block_002 ++ smallRow1987_small_right_right
theorem smallRow1987_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1987 = smallRow1987_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1987 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1987 = _
  rw [smallRow1987_block_002_checked, smallRow1987_small_right_right_checked]
  rfl

def smallRow1987_small : List Nat := smallRow1987_small_left ++ smallRow1987_small_right
theorem smallRow1987_small_checked : upperNeighbors PrimePairVertices.tree_small 1987 = smallRow1987_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1987 ++ upperNeighbors PrimePairVertices.tree_small_right 1987 = _
  rw [smallRow1987_small_left_checked, smallRow1987_small_right_checked]
  rfl

def smallRow1987 : List Nat := smallRow1987_small
theorem smallRow1987_checked : upperNeighbors PrimePairVertices.smallTree 1987 = smallRow1987 := smallRow1987_small_checked

def smallRow1993_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1993_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1993 = smallRow1993_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1993_block_001 : List Nat := [2221, 3271, 3457, 4093]
set_option maxRecDepth 2048 in
theorem smallRow1993_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1993 = smallRow1993_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1993_block_002 : List Nat := [4483, 5011, 5581, 5743, 5869]
set_option maxRecDepth 2048 in
theorem smallRow1993_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1993 = smallRow1993_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1993_block_003 : List Nat := [6151, 6211, 6337, 6481, 6781, 7507]
set_option maxRecDepth 2048 in
theorem smallRow1993_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1993 = smallRow1993_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1993_block_004 : List Nat := [8209, 8287, 8377]
set_option maxRecDepth 2048 in
theorem smallRow1993_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1993 = smallRow1993_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1993_small_left : List Nat := smallRow1993_block_000 ++ smallRow1993_block_001
theorem smallRow1993_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1993 = smallRow1993_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1993 ++ upperNeighbors PrimePairVertices.tree_block_001 1993 = _
  rw [smallRow1993_block_000_checked, smallRow1993_block_001_checked]
  rfl

def smallRow1993_small_right_right : List Nat := smallRow1993_block_003 ++ smallRow1993_block_004
theorem smallRow1993_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1993 = smallRow1993_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1993 ++ upperNeighbors PrimePairVertices.tree_block_004 1993 = _
  rw [smallRow1993_block_003_checked, smallRow1993_block_004_checked]
  rfl

def smallRow1993_small_right : List Nat := smallRow1993_block_002 ++ smallRow1993_small_right_right
theorem smallRow1993_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1993 = smallRow1993_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1993 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1993 = _
  rw [smallRow1993_block_002_checked, smallRow1993_small_right_right_checked]
  rfl

def smallRow1993_small : List Nat := smallRow1993_small_left ++ smallRow1993_small_right
theorem smallRow1993_small_checked : upperNeighbors PrimePairVertices.tree_small 1993 = smallRow1993_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1993 ++ upperNeighbors PrimePairVertices.tree_small_right 1993 = _
  rw [smallRow1993_small_left_checked, smallRow1993_small_right_checked]
  rfl

def smallRow1993 : List Nat := smallRow1993_small
theorem smallRow1993_checked : upperNeighbors PrimePairVertices.smallTree 1993 = smallRow1993 := smallRow1993_small_checked

def smallRow1997_block_000 : List Nat := [2003]
set_option maxRecDepth 2048 in
theorem smallRow1997_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1997 = smallRow1997_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1997_block_001 : List Nat := [2423, 2903, 3119, 3323, 3677]
set_option maxRecDepth 2048 in
theorem smallRow1997_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1997 = smallRow1997_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1997_block_002 : List Nat := [4397, 4583, 4793, 6053]
set_option maxRecDepth 2048 in
theorem smallRow1997_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1997 = smallRow1997_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1997_block_003 : List Nat := [7043, 7583]
set_option maxRecDepth 2048 in
theorem smallRow1997_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1997 = smallRow1997_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1997_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1997_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1997 = smallRow1997_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1997_small_left : List Nat := smallRow1997_block_000 ++ smallRow1997_block_001
theorem smallRow1997_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1997 = smallRow1997_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1997 ++ upperNeighbors PrimePairVertices.tree_block_001 1997 = _
  rw [smallRow1997_block_000_checked, smallRow1997_block_001_checked]
  rfl

def smallRow1997_small_right_right : List Nat := smallRow1997_block_003 ++ smallRow1997_block_004
theorem smallRow1997_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1997 = smallRow1997_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1997 ++ upperNeighbors PrimePairVertices.tree_block_004 1997 = _
  rw [smallRow1997_block_003_checked, smallRow1997_block_004_checked]
  rfl

def smallRow1997_small_right : List Nat := smallRow1997_block_002 ++ smallRow1997_small_right_right
theorem smallRow1997_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1997 = smallRow1997_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1997 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1997 = _
  rw [smallRow1997_block_002_checked, smallRow1997_small_right_right_checked]
  rfl

def smallRow1997_small : List Nat := smallRow1997_small_left ++ smallRow1997_small_right
theorem smallRow1997_small_checked : upperNeighbors PrimePairVertices.tree_small 1997 = smallRow1997_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1997 ++ upperNeighbors PrimePairVertices.tree_small_right 1997 = _
  rw [smallRow1997_small_left_checked, smallRow1997_small_right_checked]
  rfl

def smallRow1997 : List Nat := smallRow1997_small
theorem smallRow1997_checked : upperNeighbors PrimePairVertices.smallTree 1997 = smallRow1997 := smallRow1997_small_checked

def smallRow1999_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1999_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1999 = smallRow1999_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1999_block_001 : List Nat := [2347, 2521, 3361, 3469, 3607, 3733]
set_option maxRecDepth 2048 in
theorem smallRow1999_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1999 = smallRow1999_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1999_block_002 : List Nat := [5869, 6121]
set_option maxRecDepth 2048 in
theorem smallRow1999_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1999 = smallRow1999_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1999_block_003 : List Nat := [6469, 6961, 7717]
set_option maxRecDepth 2048 in
theorem smallRow1999_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1999 = smallRow1999_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1999_block_004 : List Nat := [8263]
set_option maxRecDepth 2048 in
theorem smallRow1999_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1999 = smallRow1999_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1999_small_left : List Nat := smallRow1999_block_000 ++ smallRow1999_block_001
theorem smallRow1999_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1999 = smallRow1999_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1999 ++ upperNeighbors PrimePairVertices.tree_block_001 1999 = _
  rw [smallRow1999_block_000_checked, smallRow1999_block_001_checked]
  rfl

def smallRow1999_small_right_right : List Nat := smallRow1999_block_003 ++ smallRow1999_block_004
theorem smallRow1999_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1999 = smallRow1999_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1999 ++ upperNeighbors PrimePairVertices.tree_block_004 1999 = _
  rw [smallRow1999_block_003_checked, smallRow1999_block_004_checked]
  rfl

def smallRow1999_small_right : List Nat := smallRow1999_block_002 ++ smallRow1999_small_right_right
theorem smallRow1999_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1999 = smallRow1999_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1999 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1999 = _
  rw [smallRow1999_block_002_checked, smallRow1999_small_right_right_checked]
  rfl

def smallRow1999_small : List Nat := smallRow1999_small_left ++ smallRow1999_small_right
theorem smallRow1999_small_checked : upperNeighbors PrimePairVertices.tree_small 1999 = smallRow1999_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1999 ++ upperNeighbors PrimePairVertices.tree_small_right 1999 = _
  rw [smallRow1999_small_left_checked, smallRow1999_small_right_checked]
  rfl

def smallRow1999 : List Nat := smallRow1999_small
theorem smallRow1999_checked : upperNeighbors PrimePairVertices.smallTree 1999 = smallRow1999 := smallRow1999_small_checked

def smallRow2003_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2003_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2003 = smallRow2003_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2003_block_001 : List Nat := [2087, 2237, 2543, 3413, 3557]
set_option maxRecDepth 2048 in
theorem smallRow2003_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2003 = smallRow2003_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2003_block_002 : List Nat := [4271, 4673, 5039, 5051, 5153, 5189, 5651, 5657, 5693]
set_option maxRecDepth 2048 in
theorem smallRow2003_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2003 = smallRow2003_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2003_block_003 : List Nat := [6563, 6689, 6791, 6833, 6983, 7013, 7211, 8171]
set_option maxRecDepth 2048 in
theorem smallRow2003_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2003 = smallRow2003_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2003_block_004 : List Nat := [8273, 8297]
set_option maxRecDepth 2048 in
theorem smallRow2003_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2003 = smallRow2003_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2003_small_left : List Nat := smallRow2003_block_000 ++ smallRow2003_block_001
theorem smallRow2003_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2003 = smallRow2003_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2003 ++ upperNeighbors PrimePairVertices.tree_block_001 2003 = _
  rw [smallRow2003_block_000_checked, smallRow2003_block_001_checked]
  rfl

def smallRow2003_small_right_right : List Nat := smallRow2003_block_003 ++ smallRow2003_block_004
theorem smallRow2003_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2003 = smallRow2003_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2003 ++ upperNeighbors PrimePairVertices.tree_block_004 2003 = _
  rw [smallRow2003_block_003_checked, smallRow2003_block_004_checked]
  rfl

def smallRow2003_small_right : List Nat := smallRow2003_block_002 ++ smallRow2003_small_right_right
theorem smallRow2003_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2003 = smallRow2003_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2003 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2003 = _
  rw [smallRow2003_block_002_checked, smallRow2003_small_right_right_checked]
  rfl

def smallRow2003_small : List Nat := smallRow2003_small_left ++ smallRow2003_small_right
theorem smallRow2003_small_checked : upperNeighbors PrimePairVertices.tree_small 2003 = smallRow2003_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2003 ++ upperNeighbors PrimePairVertices.tree_small_right 2003 = _
  rw [smallRow2003_small_left_checked, smallRow2003_small_right_checked]
  rfl

def smallRow2003 : List Nat := smallRow2003_small
theorem smallRow2003_checked : upperNeighbors PrimePairVertices.smallTree 2003 = smallRow2003 := smallRow2003_small_checked

def smallRow2011_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2011_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2011 = smallRow2011_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2011_block_001 : List Nat := [3253, 3673, 4093]
set_option maxRecDepth 2048 in
theorem smallRow2011_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2011 = smallRow2011_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2011_block_002 : List Nat := [4099, 4177, 5413, 5521]
set_option maxRecDepth 2048 in
theorem smallRow2011_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2011 = smallRow2011_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2011_block_003 : List Nat := [6379, 6637, 6781, 7963]
set_option maxRecDepth 2048 in
theorem smallRow2011_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2011 = smallRow2011_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2011_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2011_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2011 = smallRow2011_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2011_small_left : List Nat := smallRow2011_block_000 ++ smallRow2011_block_001
theorem smallRow2011_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2011 = smallRow2011_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2011 ++ upperNeighbors PrimePairVertices.tree_block_001 2011 = _
  rw [smallRow2011_block_000_checked, smallRow2011_block_001_checked]
  rfl

def smallRow2011_small_right_right : List Nat := smallRow2011_block_003 ++ smallRow2011_block_004
theorem smallRow2011_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2011 = smallRow2011_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2011 ++ upperNeighbors PrimePairVertices.tree_block_004 2011 = _
  rw [smallRow2011_block_003_checked, smallRow2011_block_004_checked]
  rfl

def smallRow2011_small_right : List Nat := smallRow2011_block_002 ++ smallRow2011_small_right_right
theorem smallRow2011_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2011 = smallRow2011_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2011 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2011 = _
  rw [smallRow2011_block_002_checked, smallRow2011_small_right_right_checked]
  rfl

def smallRow2011_small : List Nat := smallRow2011_small_left ++ smallRow2011_small_right
theorem smallRow2011_small_checked : upperNeighbors PrimePairVertices.tree_small 2011 = smallRow2011_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2011 ++ upperNeighbors PrimePairVertices.tree_small_right 2011 = _
  rw [smallRow2011_small_left_checked, smallRow2011_small_right_checked]
  rfl

def smallRow2011 : List Nat := smallRow2011_small
theorem smallRow2011_checked : upperNeighbors PrimePairVertices.smallTree 2011 = smallRow2011 := smallRow2011_small_checked

def smallRow2017_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2017_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2017 = smallRow2017_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2017_block_001 : List Nat := [2221, 2617, 2689, 3313, 3919]
set_option maxRecDepth 2048 in
theorem smallRow2017_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2017 = smallRow2017_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2017_block_002 : List Nat := [4339, 4423, 4621, 5653]
set_option maxRecDepth 2048 in
theorem smallRow2017_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2017 = smallRow2017_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2017_block_003 : List Nat := [6661, 6967, 7237, 7603]
set_option maxRecDepth 2048 in
theorem smallRow2017_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2017 = smallRow2017_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2017_block_004 : List Nat := [8311, 8317]
set_option maxRecDepth 2048 in
theorem smallRow2017_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2017 = smallRow2017_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2017_small_left : List Nat := smallRow2017_block_000 ++ smallRow2017_block_001
theorem smallRow2017_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2017 = smallRow2017_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2017 ++ upperNeighbors PrimePairVertices.tree_block_001 2017 = _
  rw [smallRow2017_block_000_checked, smallRow2017_block_001_checked]
  rfl

def smallRow2017_small_right_right : List Nat := smallRow2017_block_003 ++ smallRow2017_block_004
theorem smallRow2017_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2017 = smallRow2017_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2017 ++ upperNeighbors PrimePairVertices.tree_block_004 2017 = _
  rw [smallRow2017_block_003_checked, smallRow2017_block_004_checked]
  rfl

def smallRow2017_small_right : List Nat := smallRow2017_block_002 ++ smallRow2017_small_right_right
theorem smallRow2017_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2017 = smallRow2017_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2017 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2017 = _
  rw [smallRow2017_block_002_checked, smallRow2017_small_right_right_checked]
  rfl

def smallRow2017_small : List Nat := smallRow2017_small_left ++ smallRow2017_small_right
theorem smallRow2017_small_checked : upperNeighbors PrimePairVertices.tree_small 2017 = smallRow2017_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2017 ++ upperNeighbors PrimePairVertices.tree_small_right 2017 = _
  rw [smallRow2017_small_left_checked, smallRow2017_small_right_checked]
  rfl

def smallRow2017 : List Nat := smallRow2017_small
theorem smallRow2017_checked : upperNeighbors PrimePairVertices.smallTree 2017 = smallRow2017 := smallRow2017_small_checked

end ProjectEulerSolutions.PrimePairCache
