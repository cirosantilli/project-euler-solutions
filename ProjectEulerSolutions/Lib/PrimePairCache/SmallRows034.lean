import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1777_block_000 : List Nat := [2017]
set_option maxRecDepth 2048 in
theorem smallRow1777_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1777 = smallRow1777_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1777_block_001 : List Nat := [2371, 2539, 2647, 2719, 3121, 3253, 3463, 3739]
set_option maxRecDepth 2048 in
theorem smallRow1777_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1777 = smallRow1777_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1777_block_002 : List Nat := [4621, 5011, 5839]
set_option maxRecDepth 2048 in
theorem smallRow1777_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1777 = smallRow1777_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1777_block_003 : List Nat := [6151, 6277, 7213, 7489, 7561, 7603, 7741, 7867]
set_option maxRecDepth 2048 in
theorem smallRow1777_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1777 = smallRow1777_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1777_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1777_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1777 = smallRow1777_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1777_small_left : List Nat := smallRow1777_block_000 ++ smallRow1777_block_001
theorem smallRow1777_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1777 = smallRow1777_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1777 ++ upperNeighbors PrimePairVertices.tree_block_001 1777 = _
  rw [smallRow1777_block_000_checked, smallRow1777_block_001_checked]
  rfl

def smallRow1777_small_right_right : List Nat := smallRow1777_block_003 ++ smallRow1777_block_004
theorem smallRow1777_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1777 = smallRow1777_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1777 ++ upperNeighbors PrimePairVertices.tree_block_004 1777 = _
  rw [smallRow1777_block_003_checked, smallRow1777_block_004_checked]
  rfl

def smallRow1777_small_right : List Nat := smallRow1777_block_002 ++ smallRow1777_small_right_right
theorem smallRow1777_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1777 = smallRow1777_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1777 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1777 = _
  rw [smallRow1777_block_002_checked, smallRow1777_small_right_right_checked]
  rfl

def smallRow1777_small : List Nat := smallRow1777_small_left ++ smallRow1777_small_right
theorem smallRow1777_small_checked : upperNeighbors PrimePairVertices.tree_small 1777 = smallRow1777_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1777 ++ upperNeighbors PrimePairVertices.tree_small_right 1777 = _
  rw [smallRow1777_small_left_checked, smallRow1777_small_right_checked]
  rfl

def smallRow1777 : List Nat := smallRow1777_small
theorem smallRow1777_checked : upperNeighbors PrimePairVertices.smallTree 1777 = smallRow1777 := smallRow1777_small_checked

def smallRow1783_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1783_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1783 = smallRow1783_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1783_block_001 : List Nat := [2053, 2707, 2791, 2953, 3643, 3769]
set_option maxRecDepth 2048 in
theorem smallRow1783_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1783 = smallRow1783_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1783_block_002 : List Nat := [4273, 4639, 5011, 6121]
set_option maxRecDepth 2048 in
theorem smallRow1783_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1783 = smallRow1783_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1783_block_003 : List Nat := [6211, 6277, 6373, 6823, 7129]
set_option maxRecDepth 2048 in
theorem smallRow1783_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1783 = smallRow1783_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1783_block_004 : List Nat := [8209]
set_option maxRecDepth 2048 in
theorem smallRow1783_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1783 = smallRow1783_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1783_small_left : List Nat := smallRow1783_block_000 ++ smallRow1783_block_001
theorem smallRow1783_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1783 = smallRow1783_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1783 ++ upperNeighbors PrimePairVertices.tree_block_001 1783 = _
  rw [smallRow1783_block_000_checked, smallRow1783_block_001_checked]
  rfl

def smallRow1783_small_right_right : List Nat := smallRow1783_block_003 ++ smallRow1783_block_004
theorem smallRow1783_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1783 = smallRow1783_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1783 ++ upperNeighbors PrimePairVertices.tree_block_004 1783 = _
  rw [smallRow1783_block_003_checked, smallRow1783_block_004_checked]
  rfl

def smallRow1783_small_right : List Nat := smallRow1783_block_002 ++ smallRow1783_small_right_right
theorem smallRow1783_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1783 = smallRow1783_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1783 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1783 = _
  rw [smallRow1783_block_002_checked, smallRow1783_small_right_right_checked]
  rfl

def smallRow1783_small : List Nat := smallRow1783_small_left ++ smallRow1783_small_right
theorem smallRow1783_small_checked : upperNeighbors PrimePairVertices.tree_small 1783 = smallRow1783_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1783 ++ upperNeighbors PrimePairVertices.tree_small_right 1783 = _
  rw [smallRow1783_small_left_checked, smallRow1783_small_right_checked]
  rfl

def smallRow1783 : List Nat := smallRow1783_small
theorem smallRow1783_checked : upperNeighbors PrimePairVertices.smallTree 1783 = smallRow1783 := smallRow1783_small_checked

def smallRow1787_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1787_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1787 = smallRow1787_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1787_block_001 : List Nat := [2447, 3761]
set_option maxRecDepth 2048 in
theorem smallRow1787_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1787 = smallRow1787_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1787_block_002 : List Nat := [4211, 4271, 4349, 4391, 4919, 4937, 5387, 5591, 5927, 6011]
set_option maxRecDepth 2048 in
theorem smallRow1787_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1787 = smallRow1787_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1787_block_003 : List Nat := [6269, 6323, 6983, 7547, 7757, 7823, 7901]
set_option maxRecDepth 2048 in
theorem smallRow1787_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1787 = smallRow1787_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1787_block_004 : List Nat := [8387]
set_option maxRecDepth 2048 in
theorem smallRow1787_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1787 = smallRow1787_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1787_small_left : List Nat := smallRow1787_block_000 ++ smallRow1787_block_001
theorem smallRow1787_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1787 = smallRow1787_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1787 ++ upperNeighbors PrimePairVertices.tree_block_001 1787 = _
  rw [smallRow1787_block_000_checked, smallRow1787_block_001_checked]
  rfl

def smallRow1787_small_right_right : List Nat := smallRow1787_block_003 ++ smallRow1787_block_004
theorem smallRow1787_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1787 = smallRow1787_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1787 ++ upperNeighbors PrimePairVertices.tree_block_004 1787 = _
  rw [smallRow1787_block_003_checked, smallRow1787_block_004_checked]
  rfl

def smallRow1787_small_right : List Nat := smallRow1787_block_002 ++ smallRow1787_small_right_right
theorem smallRow1787_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1787 = smallRow1787_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1787 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1787 = _
  rw [smallRow1787_block_002_checked, smallRow1787_small_right_right_checked]
  rfl

def smallRow1787_small : List Nat := smallRow1787_small_left ++ smallRow1787_small_right
theorem smallRow1787_small_checked : upperNeighbors PrimePairVertices.tree_small 1787 = smallRow1787_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1787 ++ upperNeighbors PrimePairVertices.tree_small_right 1787 = _
  rw [smallRow1787_small_left_checked, smallRow1787_small_right_checked]
  rfl

def smallRow1787 : List Nat := smallRow1787_small
theorem smallRow1787_checked : upperNeighbors PrimePairVertices.smallTree 1787 = smallRow1787 := smallRow1787_small_checked

def smallRow1789_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1789_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1789 = smallRow1789_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1789_block_001 : List Nat := [2089, 2377, 2467, 2503, 2521, 2683, 2887, 3853, 3907]
set_option maxRecDepth 2048 in
theorem smallRow1789_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1789 = smallRow1789_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1789_block_002 : List Nat := [4111, 4243, 4813, 4831, 5023, 5953, 6133]
set_option maxRecDepth 2048 in
theorem smallRow1789_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1789 = smallRow1789_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1789_block_003 : List Nat := [6343, 6421, 6547, 6661, 7393, 8011]
set_option maxRecDepth 2048 in
theorem smallRow1789_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1789 = smallRow1789_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1789_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1789_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1789 = smallRow1789_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1789_small_left : List Nat := smallRow1789_block_000 ++ smallRow1789_block_001
theorem smallRow1789_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1789 = smallRow1789_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1789 ++ upperNeighbors PrimePairVertices.tree_block_001 1789 = _
  rw [smallRow1789_block_000_checked, smallRow1789_block_001_checked]
  rfl

def smallRow1789_small_right_right : List Nat := smallRow1789_block_003 ++ smallRow1789_block_004
theorem smallRow1789_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1789 = smallRow1789_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1789 ++ upperNeighbors PrimePairVertices.tree_block_004 1789 = _
  rw [smallRow1789_block_003_checked, smallRow1789_block_004_checked]
  rfl

def smallRow1789_small_right : List Nat := smallRow1789_block_002 ++ smallRow1789_small_right_right
theorem smallRow1789_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1789 = smallRow1789_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1789 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1789 = _
  rw [smallRow1789_block_002_checked, smallRow1789_small_right_right_checked]
  rfl

def smallRow1789_small : List Nat := smallRow1789_small_left ++ smallRow1789_small_right
theorem smallRow1789_small_checked : upperNeighbors PrimePairVertices.tree_small 1789 = smallRow1789_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1789 ++ upperNeighbors PrimePairVertices.tree_small_right 1789 = _
  rw [smallRow1789_small_left_checked, smallRow1789_small_right_checked]
  rfl

def smallRow1789 : List Nat := smallRow1789_small
theorem smallRow1789_checked : upperNeighbors PrimePairVertices.smallTree 1789 = smallRow1789 := smallRow1789_small_checked

def smallRow1801_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1801_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1801 = smallRow1801_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1801_block_001 : List Nat := [2311, 2647, 3253, 3547, 3613, 3823]
set_option maxRecDepth 2048 in
theorem smallRow1801_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1801 = smallRow1801_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1801_block_002 : List Nat := [4177, 4933, 5167, 5689]
set_option maxRecDepth 2048 in
theorem smallRow1801_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1801 = smallRow1801_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1801_block_003 : List Nat := [6637, 6829, 6907, 7333, 7621, 7669]
set_option maxRecDepth 2048 in
theorem smallRow1801_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1801 = smallRow1801_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1801_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow1801_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1801 = smallRow1801_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1801_small_left : List Nat := smallRow1801_block_000 ++ smallRow1801_block_001
theorem smallRow1801_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1801 = smallRow1801_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1801 ++ upperNeighbors PrimePairVertices.tree_block_001 1801 = _
  rw [smallRow1801_block_000_checked, smallRow1801_block_001_checked]
  rfl

def smallRow1801_small_right_right : List Nat := smallRow1801_block_003 ++ smallRow1801_block_004
theorem smallRow1801_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1801 = smallRow1801_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1801 ++ upperNeighbors PrimePairVertices.tree_block_004 1801 = _
  rw [smallRow1801_block_003_checked, smallRow1801_block_004_checked]
  rfl

def smallRow1801_small_right : List Nat := smallRow1801_block_002 ++ smallRow1801_small_right_right
theorem smallRow1801_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1801 = smallRow1801_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1801 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1801 = _
  rw [smallRow1801_block_002_checked, smallRow1801_small_right_right_checked]
  rfl

def smallRow1801_small : List Nat := smallRow1801_small_left ++ smallRow1801_small_right
theorem smallRow1801_small_checked : upperNeighbors PrimePairVertices.tree_small 1801 = smallRow1801_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1801 ++ upperNeighbors PrimePairVertices.tree_small_right 1801 = _
  rw [smallRow1801_small_left_checked, smallRow1801_small_right_checked]
  rfl

def smallRow1801 : List Nat := smallRow1801_small
theorem smallRow1801_checked : upperNeighbors PrimePairVertices.smallTree 1801 = smallRow1801 := smallRow1801_small_checked

def smallRow1811_block_000 : List Nat := [1931, 1949]
set_option maxRecDepth 2048 in
theorem smallRow1811_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1811 = smallRow1811_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1811_block_001 : List Nat := [2141, 2333, 2441, 2609, 3257]
set_option maxRecDepth 2048 in
theorem smallRow1811_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1811 = smallRow1811_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1811_block_002 : List Nat := [5039, 5297, 5381]
set_option maxRecDepth 2048 in
theorem smallRow1811_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1811 = smallRow1811_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1811_block_003 : List Nat := [6473, 6653, 6863, 7559, 7817]
set_option maxRecDepth 2048 in
theorem smallRow1811_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1811 = smallRow1811_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1811_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1811_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1811 = smallRow1811_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1811_small_left : List Nat := smallRow1811_block_000 ++ smallRow1811_block_001
theorem smallRow1811_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1811 = smallRow1811_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1811 ++ upperNeighbors PrimePairVertices.tree_block_001 1811 = _
  rw [smallRow1811_block_000_checked, smallRow1811_block_001_checked]
  rfl

def smallRow1811_small_right_right : List Nat := smallRow1811_block_003 ++ smallRow1811_block_004
theorem smallRow1811_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1811 = smallRow1811_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1811 ++ upperNeighbors PrimePairVertices.tree_block_004 1811 = _
  rw [smallRow1811_block_003_checked, smallRow1811_block_004_checked]
  rfl

def smallRow1811_small_right : List Nat := smallRow1811_block_002 ++ smallRow1811_small_right_right
theorem smallRow1811_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1811 = smallRow1811_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1811 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1811 = _
  rw [smallRow1811_block_002_checked, smallRow1811_small_right_right_checked]
  rfl

def smallRow1811_small : List Nat := smallRow1811_small_left ++ smallRow1811_small_right
theorem smallRow1811_small_checked : upperNeighbors PrimePairVertices.tree_small 1811 = smallRow1811_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1811 ++ upperNeighbors PrimePairVertices.tree_small_right 1811 = _
  rw [smallRow1811_small_left_checked, smallRow1811_small_right_checked]
  rfl

def smallRow1811 : List Nat := smallRow1811_small
theorem smallRow1811_checked : upperNeighbors PrimePairVertices.smallTree 1811 = smallRow1811 := smallRow1811_small_checked

def smallRow1823_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1823_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1823 = smallRow1823_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1823_block_001 : List Nat := [2063, 2069, 2111, 2861, 3023, 3461, 3947, 4001]
set_option maxRecDepth 2048 in
theorem smallRow1823_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1823 = smallRow1823_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1823_block_002 : List Nat := [4493, 6131]
set_option maxRecDepth 2048 in
theorem smallRow1823_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1823 = smallRow1823_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1823_block_003 : List Nat := [6353, 6719, 6947, 6983, 7823, 8069]
set_option maxRecDepth 2048 in
theorem smallRow1823_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1823 = smallRow1823_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1823_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1823_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1823 = smallRow1823_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1823_small_left : List Nat := smallRow1823_block_000 ++ smallRow1823_block_001
theorem smallRow1823_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1823 = smallRow1823_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1823 ++ upperNeighbors PrimePairVertices.tree_block_001 1823 = _
  rw [smallRow1823_block_000_checked, smallRow1823_block_001_checked]
  rfl

def smallRow1823_small_right_right : List Nat := smallRow1823_block_003 ++ smallRow1823_block_004
theorem smallRow1823_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1823 = smallRow1823_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1823 ++ upperNeighbors PrimePairVertices.tree_block_004 1823 = _
  rw [smallRow1823_block_003_checked, smallRow1823_block_004_checked]
  rfl

def smallRow1823_small_right : List Nat := smallRow1823_block_002 ++ smallRow1823_small_right_right
theorem smallRow1823_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1823 = smallRow1823_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1823 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1823 = _
  rw [smallRow1823_block_002_checked, smallRow1823_small_right_right_checked]
  rfl

def smallRow1823_small : List Nat := smallRow1823_small_left ++ smallRow1823_small_right
theorem smallRow1823_small_checked : upperNeighbors PrimePairVertices.tree_small 1823 = smallRow1823_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1823 ++ upperNeighbors PrimePairVertices.tree_small_right 1823 = _
  rw [smallRow1823_small_left_checked, smallRow1823_small_right_checked]
  rfl

def smallRow1823 : List Nat := smallRow1823_small
theorem smallRow1823_checked : upperNeighbors PrimePairVertices.smallTree 1823 = smallRow1823 := smallRow1823_small_checked

def smallRow1831_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1831_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1831 = smallRow1831_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1831_block_001 : List Nat := [2803, 3259, 3769]
set_option maxRecDepth 2048 in
theorem smallRow1831_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1831 = smallRow1831_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1831_block_002 : List Nat := [5197]
set_option maxRecDepth 2048 in
theorem smallRow1831_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1831 = smallRow1831_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1831_block_003 : List Nat := [6301, 7603, 7927]
set_option maxRecDepth 2048 in
theorem smallRow1831_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1831 = smallRow1831_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1831_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow1831_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1831 = smallRow1831_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1831_small_left : List Nat := smallRow1831_block_000 ++ smallRow1831_block_001
theorem smallRow1831_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1831 = smallRow1831_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1831 ++ upperNeighbors PrimePairVertices.tree_block_001 1831 = _
  rw [smallRow1831_block_000_checked, smallRow1831_block_001_checked]
  rfl

def smallRow1831_small_right_right : List Nat := smallRow1831_block_003 ++ smallRow1831_block_004
theorem smallRow1831_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1831 = smallRow1831_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1831 ++ upperNeighbors PrimePairVertices.tree_block_004 1831 = _
  rw [smallRow1831_block_003_checked, smallRow1831_block_004_checked]
  rfl

def smallRow1831_small_right : List Nat := smallRow1831_block_002 ++ smallRow1831_small_right_right
theorem smallRow1831_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1831 = smallRow1831_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1831 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1831 = _
  rw [smallRow1831_block_002_checked, smallRow1831_small_right_right_checked]
  rfl

def smallRow1831_small : List Nat := smallRow1831_small_left ++ smallRow1831_small_right
theorem smallRow1831_small_checked : upperNeighbors PrimePairVertices.tree_small 1831 = smallRow1831_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1831 ++ upperNeighbors PrimePairVertices.tree_small_right 1831 = _
  rw [smallRow1831_small_left_checked, smallRow1831_small_right_checked]
  rfl

def smallRow1831 : List Nat := smallRow1831_small
theorem smallRow1831_checked : upperNeighbors PrimePairVertices.smallTree 1831 = smallRow1831 := smallRow1831_small_checked

end ProjectEulerSolutions.PrimePairCache
