import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1901_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1901_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1901 = smallRow1901_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1901_block_001 : List Nat := [2153, 2621, 2801, 2969, 3221, 4019]
set_option maxRecDepth 2048 in
theorem smallRow1901_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1901 = smallRow1901_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1901_block_002 : List Nat := [4817, 5861]
set_option maxRecDepth 2048 in
theorem smallRow1901_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1901 = smallRow1901_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1901_block_003 : List Nat := [7151, 7589, 7949]
set_option maxRecDepth 2048 in
theorem smallRow1901_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1901 = smallRow1901_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1901_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow1901_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1901 = smallRow1901_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1901_small_left : List Nat := smallRow1901_block_000 ++ smallRow1901_block_001
theorem smallRow1901_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1901 = smallRow1901_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1901 ++ upperNeighbors PrimePairVertices.tree_block_001 1901 = _
  rw [smallRow1901_block_000_checked, smallRow1901_block_001_checked]
  rfl

def smallRow1901_small_right_right : List Nat := smallRow1901_block_003 ++ smallRow1901_block_004
theorem smallRow1901_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1901 = smallRow1901_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1901 ++ upperNeighbors PrimePairVertices.tree_block_004 1901 = _
  rw [smallRow1901_block_003_checked, smallRow1901_block_004_checked]
  rfl

def smallRow1901_small_right : List Nat := smallRow1901_block_002 ++ smallRow1901_small_right_right
theorem smallRow1901_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1901 = smallRow1901_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1901 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1901 = _
  rw [smallRow1901_block_002_checked, smallRow1901_small_right_right_checked]
  rfl

def smallRow1901_small : List Nat := smallRow1901_small_left ++ smallRow1901_small_right
theorem smallRow1901_small_checked : upperNeighbors PrimePairVertices.tree_small 1901 = smallRow1901_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1901 ++ upperNeighbors PrimePairVertices.tree_small_right 1901 = _
  rw [smallRow1901_small_left_checked, smallRow1901_small_right_checked]
  rfl

def smallRow1901 : List Nat := smallRow1901_small
theorem smallRow1901_checked : upperNeighbors PrimePairVertices.smallTree 1901 = smallRow1901 := smallRow1901_small_checked

def smallRow1907_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1907_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1907 = smallRow1907_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1907_block_001 : List Nat := [2063, 2243, 2621, 2777, 3083, 3581, 3797]
set_option maxRecDepth 2048 in
theorem smallRow1907_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1907 = smallRow1907_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1907_block_002 : List Nat := [4127, 4637, 5099, 5669, 5897]
set_option maxRecDepth 2048 in
theorem smallRow1907_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1907 = smallRow1907_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1907_block_003 : List Nat := [7481]
set_option maxRecDepth 2048 in
theorem smallRow1907_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1907 = smallRow1907_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1907_block_004 : List Nat := [8231]
set_option maxRecDepth 2048 in
theorem smallRow1907_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1907 = smallRow1907_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1907_small_left : List Nat := smallRow1907_block_000 ++ smallRow1907_block_001
theorem smallRow1907_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1907 = smallRow1907_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1907 ++ upperNeighbors PrimePairVertices.tree_block_001 1907 = _
  rw [smallRow1907_block_000_checked, smallRow1907_block_001_checked]
  rfl

def smallRow1907_small_right_right : List Nat := smallRow1907_block_003 ++ smallRow1907_block_004
theorem smallRow1907_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1907 = smallRow1907_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1907 ++ upperNeighbors PrimePairVertices.tree_block_004 1907 = _
  rw [smallRow1907_block_003_checked, smallRow1907_block_004_checked]
  rfl

def smallRow1907_small_right : List Nat := smallRow1907_block_002 ++ smallRow1907_small_right_right
theorem smallRow1907_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1907 = smallRow1907_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1907 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1907 = _
  rw [smallRow1907_block_002_checked, smallRow1907_small_right_right_checked]
  rfl

def smallRow1907_small : List Nat := smallRow1907_small_left ++ smallRow1907_small_right
theorem smallRow1907_small_checked : upperNeighbors PrimePairVertices.tree_small 1907 = smallRow1907_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1907 ++ upperNeighbors PrimePairVertices.tree_small_right 1907 = _
  rw [smallRow1907_small_left_checked, smallRow1907_small_right_checked]
  rfl

def smallRow1907 : List Nat := smallRow1907_small
theorem smallRow1907_checked : upperNeighbors PrimePairVertices.smallTree 1907 = smallRow1907 := smallRow1907_small_checked

def smallRow1913_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1913_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1913 = smallRow1913_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1913_block_001 : List Nat := [2213, 2273, 2381, 2693, 2741, 2843, 3203]
set_option maxRecDepth 2048 in
theorem smallRow1913_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1913 = smallRow1913_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1913_block_002 : List Nat := [4211, 5009, 5087, 5189, 5381, 5483, 5861, 6011]
set_option maxRecDepth 2048 in
theorem smallRow1913_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1913 = smallRow1913_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1913_block_003 : List Nat := [6389, 7253, 7583, 7841, 7919]
set_option maxRecDepth 2048 in
theorem smallRow1913_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1913 = smallRow1913_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1913_block_004 : List Nat := [8219, 8297]
set_option maxRecDepth 2048 in
theorem smallRow1913_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1913 = smallRow1913_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1913_small_left : List Nat := smallRow1913_block_000 ++ smallRow1913_block_001
theorem smallRow1913_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1913 = smallRow1913_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1913 ++ upperNeighbors PrimePairVertices.tree_block_001 1913 = _
  rw [smallRow1913_block_000_checked, smallRow1913_block_001_checked]
  rfl

def smallRow1913_small_right_right : List Nat := smallRow1913_block_003 ++ smallRow1913_block_004
theorem smallRow1913_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1913 = smallRow1913_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1913 ++ upperNeighbors PrimePairVertices.tree_block_004 1913 = _
  rw [smallRow1913_block_003_checked, smallRow1913_block_004_checked]
  rfl

def smallRow1913_small_right : List Nat := smallRow1913_block_002 ++ smallRow1913_small_right_right
theorem smallRow1913_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1913 = smallRow1913_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1913 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1913 = _
  rw [smallRow1913_block_002_checked, smallRow1913_small_right_right_checked]
  rfl

def smallRow1913_small : List Nat := smallRow1913_small_left ++ smallRow1913_small_right
theorem smallRow1913_small_checked : upperNeighbors PrimePairVertices.tree_small 1913 = smallRow1913_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1913 ++ upperNeighbors PrimePairVertices.tree_small_right 1913 = _
  rw [smallRow1913_small_left_checked, smallRow1913_small_right_checked]
  rfl

def smallRow1913 : List Nat := smallRow1913_small
theorem smallRow1913_checked : upperNeighbors PrimePairVertices.smallTree 1913 = smallRow1913 := smallRow1913_small_checked

def smallRow1931_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1931_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1931 = smallRow1931_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1931_block_001 : List Nat := [2399, 2957]
set_option maxRecDepth 2048 in
theorem smallRow1931_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1931 = smallRow1931_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1931_block_002 : List Nat := [4289, 4943, 5171, 5279, 5297, 5477]
set_option maxRecDepth 2048 in
theorem smallRow1931_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1931 = smallRow1931_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1931_block_003 : List Nat := [6719, 7043]
set_option maxRecDepth 2048 in
theorem smallRow1931_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1931 = smallRow1931_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1931_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1931_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1931 = smallRow1931_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1931_small_left : List Nat := smallRow1931_block_000 ++ smallRow1931_block_001
theorem smallRow1931_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1931 = smallRow1931_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1931 ++ upperNeighbors PrimePairVertices.tree_block_001 1931 = _
  rw [smallRow1931_block_000_checked, smallRow1931_block_001_checked]
  rfl

def smallRow1931_small_right_right : List Nat := smallRow1931_block_003 ++ smallRow1931_block_004
theorem smallRow1931_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1931 = smallRow1931_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1931 ++ upperNeighbors PrimePairVertices.tree_block_004 1931 = _
  rw [smallRow1931_block_003_checked, smallRow1931_block_004_checked]
  rfl

def smallRow1931_small_right : List Nat := smallRow1931_block_002 ++ smallRow1931_small_right_right
theorem smallRow1931_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1931 = smallRow1931_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1931 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1931 = _
  rw [smallRow1931_block_002_checked, smallRow1931_small_right_right_checked]
  rfl

def smallRow1931_small : List Nat := smallRow1931_small_left ++ smallRow1931_small_right
theorem smallRow1931_small_checked : upperNeighbors PrimePairVertices.tree_small 1931 = smallRow1931_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1931 ++ upperNeighbors PrimePairVertices.tree_small_right 1931 = _
  rw [smallRow1931_small_left_checked, smallRow1931_small_right_checked]
  rfl

def smallRow1931 : List Nat := smallRow1931_small
theorem smallRow1931_checked : upperNeighbors PrimePairVertices.smallTree 1931 = smallRow1931 := smallRow1931_small_checked

def smallRow1933_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1933_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1933 = smallRow1933_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1933_block_001 : List Nat := [2293, 3313, 3733]
set_option maxRecDepth 2048 in
theorem smallRow1933_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1933 = smallRow1933_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1933_block_002 : List Nat := [4201, 5011, 5023, 5791]
set_option maxRecDepth 2048 in
theorem smallRow1933_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1933 = smallRow1933_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1933_block_003 : List Nat := [6703, 6841, 7333, 7639, 7933]
set_option maxRecDepth 2048 in
theorem smallRow1933_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1933 = smallRow1933_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1933_block_004 : List Nat := [8287]
set_option maxRecDepth 2048 in
theorem smallRow1933_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1933 = smallRow1933_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1933_small_left : List Nat := smallRow1933_block_000 ++ smallRow1933_block_001
theorem smallRow1933_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1933 = smallRow1933_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1933 ++ upperNeighbors PrimePairVertices.tree_block_001 1933 = _
  rw [smallRow1933_block_000_checked, smallRow1933_block_001_checked]
  rfl

def smallRow1933_small_right_right : List Nat := smallRow1933_block_003 ++ smallRow1933_block_004
theorem smallRow1933_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1933 = smallRow1933_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1933 ++ upperNeighbors PrimePairVertices.tree_block_004 1933 = _
  rw [smallRow1933_block_003_checked, smallRow1933_block_004_checked]
  rfl

def smallRow1933_small_right : List Nat := smallRow1933_block_002 ++ smallRow1933_small_right_right
theorem smallRow1933_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1933 = smallRow1933_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1933 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1933 = _
  rw [smallRow1933_block_002_checked, smallRow1933_small_right_right_checked]
  rfl

def smallRow1933_small : List Nat := smallRow1933_small_left ++ smallRow1933_small_right
theorem smallRow1933_small_checked : upperNeighbors PrimePairVertices.tree_small 1933 = smallRow1933_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1933 ++ upperNeighbors PrimePairVertices.tree_small_right 1933 = _
  rw [smallRow1933_small_left_checked, smallRow1933_small_right_checked]
  rfl

def smallRow1933 : List Nat := smallRow1933_small
theorem smallRow1933_checked : upperNeighbors PrimePairVertices.smallTree 1933 = smallRow1933 := smallRow1933_small_checked

def smallRow1949_block_000 : List Nat := [2027]
set_option maxRecDepth 2048 in
theorem smallRow1949_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1949 = smallRow1949_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1949_block_001 : List Nat := [2621, 2657, 2663, 2789, 2819, 2903, 3911, 3923, 3989]
set_option maxRecDepth 2048 in
theorem smallRow1949_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1949 = smallRow1949_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1949_block_002 : List Nat := [4337, 5669]
set_option maxRecDepth 2048 in
theorem smallRow1949_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1949 = smallRow1949_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1949_block_003 : List Nat := [6863, 7727, 7901]
set_option maxRecDepth 2048 in
theorem smallRow1949_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1949 = smallRow1949_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1949_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1949_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1949 = smallRow1949_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1949_small_left : List Nat := smallRow1949_block_000 ++ smallRow1949_block_001
theorem smallRow1949_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1949 = smallRow1949_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1949 ++ upperNeighbors PrimePairVertices.tree_block_001 1949 = _
  rw [smallRow1949_block_000_checked, smallRow1949_block_001_checked]
  rfl

def smallRow1949_small_right_right : List Nat := smallRow1949_block_003 ++ smallRow1949_block_004
theorem smallRow1949_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1949 = smallRow1949_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1949 ++ upperNeighbors PrimePairVertices.tree_block_004 1949 = _
  rw [smallRow1949_block_003_checked, smallRow1949_block_004_checked]
  rfl

def smallRow1949_small_right : List Nat := smallRow1949_block_002 ++ smallRow1949_small_right_right
theorem smallRow1949_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1949 = smallRow1949_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1949 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1949 = _
  rw [smallRow1949_block_002_checked, smallRow1949_small_right_right_checked]
  rfl

def smallRow1949_small : List Nat := smallRow1949_small_left ++ smallRow1949_small_right
theorem smallRow1949_small_checked : upperNeighbors PrimePairVertices.tree_small 1949 = smallRow1949_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1949 ++ upperNeighbors PrimePairVertices.tree_small_right 1949 = _
  rw [smallRow1949_small_left_checked, smallRow1949_small_right_checked]
  rfl

def smallRow1949 : List Nat := smallRow1949_small
theorem smallRow1949_checked : upperNeighbors PrimePairVertices.smallTree 1949 = smallRow1949 := smallRow1949_small_checked

def smallRow1951_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1951_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1951 = smallRow1951_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1951_block_001 : List Nat := [2341, 2683, 3547, 3643, 3733]
set_option maxRecDepth 2048 in
theorem smallRow1951_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1951 = smallRow1951_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1951_block_002 : List Nat := [4549, 4783, 4909, 5227, 5437, 6037]
set_option maxRecDepth 2048 in
theorem smallRow1951_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1951 = smallRow1951_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1951_block_003 : List Nat := [6151, 6379, 6547, 6823, 7027, 7489, 8167]
set_option maxRecDepth 2048 in
theorem smallRow1951_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1951 = smallRow1951_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1951_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1951_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1951 = smallRow1951_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1951_small_left : List Nat := smallRow1951_block_000 ++ smallRow1951_block_001
theorem smallRow1951_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1951 = smallRow1951_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1951 ++ upperNeighbors PrimePairVertices.tree_block_001 1951 = _
  rw [smallRow1951_block_000_checked, smallRow1951_block_001_checked]
  rfl

def smallRow1951_small_right_right : List Nat := smallRow1951_block_003 ++ smallRow1951_block_004
theorem smallRow1951_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1951 = smallRow1951_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1951 ++ upperNeighbors PrimePairVertices.tree_block_004 1951 = _
  rw [smallRow1951_block_003_checked, smallRow1951_block_004_checked]
  rfl

def smallRow1951_small_right : List Nat := smallRow1951_block_002 ++ smallRow1951_small_right_right
theorem smallRow1951_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1951 = smallRow1951_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1951 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1951 = _
  rw [smallRow1951_block_002_checked, smallRow1951_small_right_right_checked]
  rfl

def smallRow1951_small : List Nat := smallRow1951_small_left ++ smallRow1951_small_right
theorem smallRow1951_small_checked : upperNeighbors PrimePairVertices.tree_small 1951 = smallRow1951_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1951 ++ upperNeighbors PrimePairVertices.tree_small_right 1951 = _
  rw [smallRow1951_small_left_checked, smallRow1951_small_right_checked]
  rfl

def smallRow1951 : List Nat := smallRow1951_small
theorem smallRow1951_checked : upperNeighbors PrimePairVertices.smallTree 1951 = smallRow1951 := smallRow1951_small_checked

def smallRow1973_block_000 : List Nat := [1979]
set_option maxRecDepth 2048 in
theorem smallRow1973_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1973 = smallRow1973_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1973_block_001 : List Nat := [2381, 2591]
set_option maxRecDepth 2048 in
theorem smallRow1973_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1973 = smallRow1973_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1973_block_002 : List Nat := [4397, 4409, 4457, 4493, 4751, 4919, 5711, 5741, 5897, 6131]
set_option maxRecDepth 2048 in
theorem smallRow1973_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1973 = smallRow1973_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1973_block_003 : List Nat := [7547]
set_option maxRecDepth 2048 in
theorem smallRow1973_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1973 = smallRow1973_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1973_block_004 : List Nat := [8291, 8363]
set_option maxRecDepth 2048 in
theorem smallRow1973_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1973 = smallRow1973_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1973_small_left : List Nat := smallRow1973_block_000 ++ smallRow1973_block_001
theorem smallRow1973_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1973 = smallRow1973_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1973 ++ upperNeighbors PrimePairVertices.tree_block_001 1973 = _
  rw [smallRow1973_block_000_checked, smallRow1973_block_001_checked]
  rfl

def smallRow1973_small_right_right : List Nat := smallRow1973_block_003 ++ smallRow1973_block_004
theorem smallRow1973_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1973 = smallRow1973_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1973 ++ upperNeighbors PrimePairVertices.tree_block_004 1973 = _
  rw [smallRow1973_block_003_checked, smallRow1973_block_004_checked]
  rfl

def smallRow1973_small_right : List Nat := smallRow1973_block_002 ++ smallRow1973_small_right_right
theorem smallRow1973_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1973 = smallRow1973_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1973 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1973 = _
  rw [smallRow1973_block_002_checked, smallRow1973_small_right_right_checked]
  rfl

def smallRow1973_small : List Nat := smallRow1973_small_left ++ smallRow1973_small_right
theorem smallRow1973_small_checked : upperNeighbors PrimePairVertices.tree_small 1973 = smallRow1973_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1973 ++ upperNeighbors PrimePairVertices.tree_small_right 1973 = _
  rw [smallRow1973_small_left_checked, smallRow1973_small_right_checked]
  rfl

def smallRow1973 : List Nat := smallRow1973_small
theorem smallRow1973_checked : upperNeighbors PrimePairVertices.smallTree 1973 = smallRow1973 := smallRow1973_small_checked

end ProjectEulerSolutions.PrimePairCache
