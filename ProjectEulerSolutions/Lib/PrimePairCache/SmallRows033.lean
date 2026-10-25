import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1709_block_000 : List Nat := [1913]
set_option maxRecDepth 2048 in
theorem smallRow1709_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1709 = smallRow1709_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1709_block_001 : List Nat := [2213, 2333, 2393, 2423, 2591, 2843, 3137, 3299, 3389, 3917, 4001, 4013]
set_option maxRecDepth 2048 in
theorem smallRow1709_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1709 = smallRow1709_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1709_block_002 : List Nat := [4283, 4397, 4523]
set_option maxRecDepth 2048 in
theorem smallRow1709_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1709 = smallRow1709_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1709_block_003 : List Nat := [6551, 6857, 6869, 7001, 7247, 7823, 7877]
set_option maxRecDepth 2048 in
theorem smallRow1709_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1709 = smallRow1709_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1709_block_004 : List Nat := [8231, 8297]
set_option maxRecDepth 2048 in
theorem smallRow1709_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1709 = smallRow1709_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1709_small_left : List Nat := smallRow1709_block_000 ++ smallRow1709_block_001
theorem smallRow1709_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1709 = smallRow1709_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1709 ++ upperNeighbors PrimePairVertices.tree_block_001 1709 = _
  rw [smallRow1709_block_000_checked, smallRow1709_block_001_checked]
  rfl

def smallRow1709_small_right_right : List Nat := smallRow1709_block_003 ++ smallRow1709_block_004
theorem smallRow1709_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1709 = smallRow1709_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1709 ++ upperNeighbors PrimePairVertices.tree_block_004 1709 = _
  rw [smallRow1709_block_003_checked, smallRow1709_block_004_checked]
  rfl

def smallRow1709_small_right : List Nat := smallRow1709_block_002 ++ smallRow1709_small_right_right
theorem smallRow1709_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1709 = smallRow1709_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1709 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1709 = _
  rw [smallRow1709_block_002_checked, smallRow1709_small_right_right_checked]
  rfl

def smallRow1709_small : List Nat := smallRow1709_small_left ++ smallRow1709_small_right
theorem smallRow1709_small_checked : upperNeighbors PrimePairVertices.tree_small 1709 = smallRow1709_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1709 ++ upperNeighbors PrimePairVertices.tree_small_right 1709 = _
  rw [smallRow1709_small_left_checked, smallRow1709_small_right_checked]
  rfl

def smallRow1709 : List Nat := smallRow1709_small
theorem smallRow1709_checked : upperNeighbors PrimePairVertices.smallTree 1709 = smallRow1709 := smallRow1709_small_checked

def smallRow1721_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1721_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1721 = smallRow1721_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1721_block_001 : List Nat := [2213, 2297, 2411, 2621, 3347, 3359, 3557]
set_option maxRecDepth 2048 in
theorem smallRow1721_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1721 = smallRow1721_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1721_block_002 : List Nat := [4679]
set_option maxRecDepth 2048 in
theorem smallRow1721_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1721 = smallRow1721_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1721_block_003 : List Nat := [6389, 6779, 7283, 7589, 7829, 8039]
set_option maxRecDepth 2048 in
theorem smallRow1721_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1721 = smallRow1721_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1721_block_004 : List Nat := [8219, 8231, 8273, 8387]
set_option maxRecDepth 2048 in
theorem smallRow1721_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1721 = smallRow1721_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1721_small_left : List Nat := smallRow1721_block_000 ++ smallRow1721_block_001
theorem smallRow1721_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1721 = smallRow1721_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1721 ++ upperNeighbors PrimePairVertices.tree_block_001 1721 = _
  rw [smallRow1721_block_000_checked, smallRow1721_block_001_checked]
  rfl

def smallRow1721_small_right_right : List Nat := smallRow1721_block_003 ++ smallRow1721_block_004
theorem smallRow1721_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1721 = smallRow1721_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1721 ++ upperNeighbors PrimePairVertices.tree_block_004 1721 = _
  rw [smallRow1721_block_003_checked, smallRow1721_block_004_checked]
  rfl

def smallRow1721_small_right : List Nat := smallRow1721_block_002 ++ smallRow1721_small_right_right
theorem smallRow1721_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1721 = smallRow1721_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1721 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1721 = _
  rw [smallRow1721_block_002_checked, smallRow1721_small_right_right_checked]
  rfl

def smallRow1721_small : List Nat := smallRow1721_small_left ++ smallRow1721_small_right
theorem smallRow1721_small_checked : upperNeighbors PrimePairVertices.tree_small 1721 = smallRow1721_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1721 ++ upperNeighbors PrimePairVertices.tree_small_right 1721 = _
  rw [smallRow1721_small_left_checked, smallRow1721_small_right_checked]
  rfl

def smallRow1721 : List Nat := smallRow1721_small
theorem smallRow1721_checked : upperNeighbors PrimePairVertices.smallTree 1721 = smallRow1721 := smallRow1721_small_checked

def smallRow1723_block_000 : List Nat := [1831]
set_option maxRecDepth 2048 in
theorem smallRow1723_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1723 = smallRow1723_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1723_block_001 : List Nat := [2179, 3319, 3457, 4057]
set_option maxRecDepth 2048 in
theorem smallRow1723_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1723 = smallRow1723_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1723_block_002 : List Nat := [4363, 4423, 4621, 4783, 4813, 4993, 4999, 5419, 5431]
set_option maxRecDepth 2048 in
theorem smallRow1723_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1723 = smallRow1723_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1723_block_003 : List Nat := [6397, 6547, 7951, 7993]
set_option maxRecDepth 2048 in
theorem smallRow1723_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1723 = smallRow1723_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1723_block_004 : List Nat := [8233, 8329]
set_option maxRecDepth 2048 in
theorem smallRow1723_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1723 = smallRow1723_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1723_small_left : List Nat := smallRow1723_block_000 ++ smallRow1723_block_001
theorem smallRow1723_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1723 = smallRow1723_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1723 ++ upperNeighbors PrimePairVertices.tree_block_001 1723 = _
  rw [smallRow1723_block_000_checked, smallRow1723_block_001_checked]
  rfl

def smallRow1723_small_right_right : List Nat := smallRow1723_block_003 ++ smallRow1723_block_004
theorem smallRow1723_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1723 = smallRow1723_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1723 ++ upperNeighbors PrimePairVertices.tree_block_004 1723 = _
  rw [smallRow1723_block_003_checked, smallRow1723_block_004_checked]
  rfl

def smallRow1723_small_right : List Nat := smallRow1723_block_002 ++ smallRow1723_small_right_right
theorem smallRow1723_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1723 = smallRow1723_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1723 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1723 = _
  rw [smallRow1723_block_002_checked, smallRow1723_small_right_right_checked]
  rfl

def smallRow1723_small : List Nat := smallRow1723_small_left ++ smallRow1723_small_right
theorem smallRow1723_small_checked : upperNeighbors PrimePairVertices.tree_small 1723 = smallRow1723_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1723 ++ upperNeighbors PrimePairVertices.tree_small_right 1723 = _
  rw [smallRow1723_small_left_checked, smallRow1723_small_right_checked]
  rfl

def smallRow1723 : List Nat := smallRow1723_small
theorem smallRow1723_checked : upperNeighbors PrimePairVertices.smallTree 1723 = smallRow1723 := smallRow1723_small_checked

def smallRow1733_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1733_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1733 = smallRow1733_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1733_block_001 : List Nat := [2411, 2657, 2663, 2927, 3347, 3593, 3917, 4001]
set_option maxRecDepth 2048 in
theorem smallRow1733_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1733 = smallRow1733_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1733_block_002 : List Nat := [4133, 4817, 5051, 5273, 5867, 5897, 6101]
set_option maxRecDepth 2048 in
theorem smallRow1733_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1733 = smallRow1733_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1733_block_003 : List Nat := [6323, 6521, 6959, 7043, 7757, 7949]
set_option maxRecDepth 2048 in
theorem smallRow1733_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1733 = smallRow1733_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1733_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1733_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1733 = smallRow1733_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1733_small_left : List Nat := smallRow1733_block_000 ++ smallRow1733_block_001
theorem smallRow1733_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1733 = smallRow1733_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1733 ++ upperNeighbors PrimePairVertices.tree_block_001 1733 = _
  rw [smallRow1733_block_000_checked, smallRow1733_block_001_checked]
  rfl

def smallRow1733_small_right_right : List Nat := smallRow1733_block_003 ++ smallRow1733_block_004
theorem smallRow1733_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1733 = smallRow1733_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1733 ++ upperNeighbors PrimePairVertices.tree_block_004 1733 = _
  rw [smallRow1733_block_003_checked, smallRow1733_block_004_checked]
  rfl

def smallRow1733_small_right : List Nat := smallRow1733_block_002 ++ smallRow1733_small_right_right
theorem smallRow1733_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1733 = smallRow1733_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1733 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1733 = _
  rw [smallRow1733_block_002_checked, smallRow1733_small_right_right_checked]
  rfl

def smallRow1733_small : List Nat := smallRow1733_small_left ++ smallRow1733_small_right
theorem smallRow1733_small_checked : upperNeighbors PrimePairVertices.tree_small 1733 = smallRow1733_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1733 ++ upperNeighbors PrimePairVertices.tree_small_right 1733 = _
  rw [smallRow1733_small_left_checked, smallRow1733_small_right_checked]
  rfl

def smallRow1733 : List Nat := smallRow1733_small
theorem smallRow1733_checked : upperNeighbors PrimePairVertices.smallTree 1733 = smallRow1733 := smallRow1733_small_checked

def smallRow1741_block_000 : List Nat := [1753]
set_option maxRecDepth 2048 in
theorem smallRow1741_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1741 = smallRow1741_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1741_block_001 : List Nat := [2053, 2203, 2281, 2287, 2347, 2851, 2887, 2953, 2971, 3307, 3313, 3373, 3607, 3793, 3919, 4003]
set_option maxRecDepth 2048 in
theorem smallRow1741_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1741 = smallRow1741_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1741_block_002 : List Nat := [5227, 5479, 5557, 5581]
set_option maxRecDepth 2048 in
theorem smallRow1741_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1741 = smallRow1741_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1741_block_003 : List Nat := [6397, 7489, 7573, 7993]
set_option maxRecDepth 2048 in
theorem smallRow1741_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1741 = smallRow1741_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1741_block_004 : List Nat := [8287]
set_option maxRecDepth 2048 in
theorem smallRow1741_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1741 = smallRow1741_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1741_small_left : List Nat := smallRow1741_block_000 ++ smallRow1741_block_001
theorem smallRow1741_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1741 = smallRow1741_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1741 ++ upperNeighbors PrimePairVertices.tree_block_001 1741 = _
  rw [smallRow1741_block_000_checked, smallRow1741_block_001_checked]
  rfl

def smallRow1741_small_right_right : List Nat := smallRow1741_block_003 ++ smallRow1741_block_004
theorem smallRow1741_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1741 = smallRow1741_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1741 ++ upperNeighbors PrimePairVertices.tree_block_004 1741 = _
  rw [smallRow1741_block_003_checked, smallRow1741_block_004_checked]
  rfl

def smallRow1741_small_right : List Nat := smallRow1741_block_002 ++ smallRow1741_small_right_right
theorem smallRow1741_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1741 = smallRow1741_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1741 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1741 = _
  rw [smallRow1741_block_002_checked, smallRow1741_small_right_right_checked]
  rfl

def smallRow1741_small : List Nat := smallRow1741_small_left ++ smallRow1741_small_right
theorem smallRow1741_small_checked : upperNeighbors PrimePairVertices.tree_small 1741 = smallRow1741_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1741 ++ upperNeighbors PrimePairVertices.tree_small_right 1741 = _
  rw [smallRow1741_small_left_checked, smallRow1741_small_right_checked]
  rfl

def smallRow1741 : List Nat := smallRow1741_small
theorem smallRow1741_checked : upperNeighbors PrimePairVertices.smallTree 1741 = smallRow1741 := smallRow1741_small_checked

def smallRow1747_block_000 : List Nat := [1873, 1933]
set_option maxRecDepth 2048 in
theorem smallRow1747_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1747 = smallRow1747_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1747_block_001 : List Nat := [2053, 2179, 2671, 3691, 3769, 3889]
set_option maxRecDepth 2048 in
theorem smallRow1747_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1747 = smallRow1747_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1747_block_002 : List Nat := [4099, 4231, 4993, 5569, 5737, 6037]
set_option maxRecDepth 2048 in
theorem smallRow1747_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1747 = smallRow1747_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1747_block_003 : List Nat := [6367, 6421, 6967, 7129, 7177, 7477, 7639, 8011]
set_option maxRecDepth 2048 in
theorem smallRow1747_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1747 = smallRow1747_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1747_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1747_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1747 = smallRow1747_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1747_small_left : List Nat := smallRow1747_block_000 ++ smallRow1747_block_001
theorem smallRow1747_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1747 = smallRow1747_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1747 ++ upperNeighbors PrimePairVertices.tree_block_001 1747 = _
  rw [smallRow1747_block_000_checked, smallRow1747_block_001_checked]
  rfl

def smallRow1747_small_right_right : List Nat := smallRow1747_block_003 ++ smallRow1747_block_004
theorem smallRow1747_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1747 = smallRow1747_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1747 ++ upperNeighbors PrimePairVertices.tree_block_004 1747 = _
  rw [smallRow1747_block_003_checked, smallRow1747_block_004_checked]
  rfl

def smallRow1747_small_right : List Nat := smallRow1747_block_002 ++ smallRow1747_small_right_right
theorem smallRow1747_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1747 = smallRow1747_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1747 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1747 = _
  rw [smallRow1747_block_002_checked, smallRow1747_small_right_right_checked]
  rfl

def smallRow1747_small : List Nat := smallRow1747_small_left ++ smallRow1747_small_right
theorem smallRow1747_small_checked : upperNeighbors PrimePairVertices.tree_small 1747 = smallRow1747_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1747 ++ upperNeighbors PrimePairVertices.tree_small_right 1747 = _
  rw [smallRow1747_small_left_checked, smallRow1747_small_right_checked]
  rfl

def smallRow1747 : List Nat := smallRow1747_small
theorem smallRow1747_checked : upperNeighbors PrimePairVertices.smallTree 1747 = smallRow1747 := smallRow1747_small_checked

def smallRow1753_block_000 : List Nat := [1873, 1951]
set_option maxRecDepth 2048 in
theorem smallRow1753_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1753 = smallRow1753_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1753_block_001 : List Nat := [3547, 3643, 3673, 3709, 3847]
set_option maxRecDepth 2048 in
theorem smallRow1753_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1753 = smallRow1753_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1753_block_002 : List Nat := [4567, 4903, 4987, 5197, 5281, 5827, 5851]
set_option maxRecDepth 2048 in
theorem smallRow1753_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1753 = smallRow1753_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1753_block_003 : List Nat := [6199, 6691, 6823, 6907, 7237, 7321, 7411, 7549, 7669, 7789, 8089, 8161]
set_option maxRecDepth 2048 in
theorem smallRow1753_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1753 = smallRow1753_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1753_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1753_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1753 = smallRow1753_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1753_small_left : List Nat := smallRow1753_block_000 ++ smallRow1753_block_001
theorem smallRow1753_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1753 = smallRow1753_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1753 ++ upperNeighbors PrimePairVertices.tree_block_001 1753 = _
  rw [smallRow1753_block_000_checked, smallRow1753_block_001_checked]
  rfl

def smallRow1753_small_right_right : List Nat := smallRow1753_block_003 ++ smallRow1753_block_004
theorem smallRow1753_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1753 = smallRow1753_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1753 ++ upperNeighbors PrimePairVertices.tree_block_004 1753 = _
  rw [smallRow1753_block_003_checked, smallRow1753_block_004_checked]
  rfl

def smallRow1753_small_right : List Nat := smallRow1753_block_002 ++ smallRow1753_small_right_right
theorem smallRow1753_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1753 = smallRow1753_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1753 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1753 = _
  rw [smallRow1753_block_002_checked, smallRow1753_small_right_right_checked]
  rfl

def smallRow1753_small : List Nat := smallRow1753_small_left ++ smallRow1753_small_right
theorem smallRow1753_small_checked : upperNeighbors PrimePairVertices.tree_small 1753 = smallRow1753_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1753 ++ upperNeighbors PrimePairVertices.tree_small_right 1753 = _
  rw [smallRow1753_small_left_checked, smallRow1753_small_right_checked]
  rfl

def smallRow1753 : List Nat := smallRow1753_small
theorem smallRow1753_checked : upperNeighbors PrimePairVertices.smallTree 1753 = smallRow1753 := smallRow1753_small_checked

def smallRow1759_block_000 : List Nat := [1831]
set_option maxRecDepth 2048 in
theorem smallRow1759_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1759 = smallRow1759_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1759_block_001 : List Nat := [2749, 3001, 3319, 3571, 3733, 3847, 3889]
set_option maxRecDepth 2048 in
theorem smallRow1759_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1759 = smallRow1759_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1759_block_002 : List Nat := [4177, 5233, 5689]
set_option maxRecDepth 2048 in
theorem smallRow1759_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1759 = smallRow1759_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1759_block_003 : List Nat := [6199, 6361, 6367, 6637, 6703, 7351, 7417]
set_option maxRecDepth 2048 in
theorem smallRow1759_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1759 = smallRow1759_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1759_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1759_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1759 = smallRow1759_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1759_small_left : List Nat := smallRow1759_block_000 ++ smallRow1759_block_001
theorem smallRow1759_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1759 = smallRow1759_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1759 ++ upperNeighbors PrimePairVertices.tree_block_001 1759 = _
  rw [smallRow1759_block_000_checked, smallRow1759_block_001_checked]
  rfl

def smallRow1759_small_right_right : List Nat := smallRow1759_block_003 ++ smallRow1759_block_004
theorem smallRow1759_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1759 = smallRow1759_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1759 ++ upperNeighbors PrimePairVertices.tree_block_004 1759 = _
  rw [smallRow1759_block_003_checked, smallRow1759_block_004_checked]
  rfl

def smallRow1759_small_right : List Nat := smallRow1759_block_002 ++ smallRow1759_small_right_right
theorem smallRow1759_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1759 = smallRow1759_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1759 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1759 = _
  rw [smallRow1759_block_002_checked, smallRow1759_small_right_right_checked]
  rfl

def smallRow1759_small : List Nat := smallRow1759_small_left ++ smallRow1759_small_right
theorem smallRow1759_small_checked : upperNeighbors PrimePairVertices.tree_small 1759 = smallRow1759_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1759 ++ upperNeighbors PrimePairVertices.tree_small_right 1759 = _
  rw [smallRow1759_small_left_checked, smallRow1759_small_right_checked]
  rfl

def smallRow1759 : List Nat := smallRow1759_small
theorem smallRow1759_checked : upperNeighbors PrimePairVertices.smallTree 1759 = smallRow1759 := smallRow1759_small_checked

end ProjectEulerSolutions.PrimePairCache
