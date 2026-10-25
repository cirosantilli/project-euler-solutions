import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow1847_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1847_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1847 = smallRow1847_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1847_block_001 : List Nat := [2423, 2687, 2969, 3011, 3671, 3779, 3863, 4007]
set_option maxRecDepth 2048 in
theorem smallRow1847_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1847 = smallRow1847_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1847_block_002 : List Nat := [4283, 4373, 4457, 4691, 4919, 5519, 5801, 5927]
set_option maxRecDepth 2048 in
theorem smallRow1847_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1847 = smallRow1847_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1847_block_003 : List Nat := [6173, 7307, 7523, 7703, 8069, 8123]
set_option maxRecDepth 2048 in
theorem smallRow1847_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1847 = smallRow1847_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1847_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1847_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1847 = smallRow1847_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1847_small_left : List Nat := smallRow1847_block_000 ++ smallRow1847_block_001
theorem smallRow1847_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1847 = smallRow1847_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1847 ++ upperNeighbors PrimePairVertices.tree_block_001 1847 = _
  rw [smallRow1847_block_000_checked, smallRow1847_block_001_checked]
  rfl

def smallRow1847_small_right_right : List Nat := smallRow1847_block_003 ++ smallRow1847_block_004
theorem smallRow1847_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1847 = smallRow1847_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1847 ++ upperNeighbors PrimePairVertices.tree_block_004 1847 = _
  rw [smallRow1847_block_003_checked, smallRow1847_block_004_checked]
  rfl

def smallRow1847_small_right : List Nat := smallRow1847_block_002 ++ smallRow1847_small_right_right
theorem smallRow1847_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1847 = smallRow1847_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1847 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1847 = _
  rw [smallRow1847_block_002_checked, smallRow1847_small_right_right_checked]
  rfl

def smallRow1847_small : List Nat := smallRow1847_small_left ++ smallRow1847_small_right
theorem smallRow1847_small_checked : upperNeighbors PrimePairVertices.tree_small 1847 = smallRow1847_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1847 ++ upperNeighbors PrimePairVertices.tree_small_right 1847 = _
  rw [smallRow1847_small_left_checked, smallRow1847_small_right_checked]
  rfl

def smallRow1847 : List Nat := smallRow1847_small
theorem smallRow1847_checked : upperNeighbors PrimePairVertices.smallTree 1847 = smallRow1847 := smallRow1847_small_checked

def smallRow1861_block_000 : List Nat := [1933, 1951]
set_option maxRecDepth 2048 in
theorem smallRow1861_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1861 = smallRow1861_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1861_block_001 : List Nat := [2371, 2383, 2791, 3163, 3433, 3709, 3769]
set_option maxRecDepth 2048 in
theorem smallRow1861_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1861 = smallRow1861_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1861_block_002 : List Nat := [4789, 5011, 5059, 5479, 5521, 5881, 6007, 6121, 6133]
set_option maxRecDepth 2048 in
theorem smallRow1861_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1861 = smallRow1861_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1861_block_003 : List Nat := [6907, 7213, 7867]
set_option maxRecDepth 2048 in
theorem smallRow1861_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1861 = smallRow1861_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1861_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1861_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1861 = smallRow1861_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1861_small_left : List Nat := smallRow1861_block_000 ++ smallRow1861_block_001
theorem smallRow1861_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1861 = smallRow1861_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1861 ++ upperNeighbors PrimePairVertices.tree_block_001 1861 = _
  rw [smallRow1861_block_000_checked, smallRow1861_block_001_checked]
  rfl

def smallRow1861_small_right_right : List Nat := smallRow1861_block_003 ++ smallRow1861_block_004
theorem smallRow1861_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1861 = smallRow1861_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1861 ++ upperNeighbors PrimePairVertices.tree_block_004 1861 = _
  rw [smallRow1861_block_003_checked, smallRow1861_block_004_checked]
  rfl

def smallRow1861_small_right : List Nat := smallRow1861_block_002 ++ smallRow1861_small_right_right
theorem smallRow1861_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1861 = smallRow1861_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1861 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1861 = _
  rw [smallRow1861_block_002_checked, smallRow1861_small_right_right_checked]
  rfl

def smallRow1861_small : List Nat := smallRow1861_small_left ++ smallRow1861_small_right
theorem smallRow1861_small_checked : upperNeighbors PrimePairVertices.tree_small 1861 = smallRow1861_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1861 ++ upperNeighbors PrimePairVertices.tree_small_right 1861 = _
  rw [smallRow1861_small_left_checked, smallRow1861_small_right_checked]
  rfl

def smallRow1861 : List Nat := smallRow1861_small
theorem smallRow1861_checked : upperNeighbors PrimePairVertices.smallTree 1861 = smallRow1861 := smallRow1861_small_checked

def smallRow1867_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1867_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1867 = smallRow1867_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1867_block_001 : List Nat := [2053, 2221, 3673]
set_option maxRecDepth 2048 in
theorem smallRow1867_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1867 = smallRow1867_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1867_block_002 : List Nat := [4111, 5281, 5659, 5749, 6121]
set_option maxRecDepth 2048 in
theorem smallRow1867_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1867 = smallRow1867_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1867_block_003 : List Nat := [6733, 7129, 7639, 7873]
set_option maxRecDepth 2048 in
theorem smallRow1867_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1867 = smallRow1867_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1867_block_004 : List Nat := [8293]
set_option maxRecDepth 2048 in
theorem smallRow1867_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1867 = smallRow1867_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1867_small_left : List Nat := smallRow1867_block_000 ++ smallRow1867_block_001
theorem smallRow1867_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1867 = smallRow1867_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1867 ++ upperNeighbors PrimePairVertices.tree_block_001 1867 = _
  rw [smallRow1867_block_000_checked, smallRow1867_block_001_checked]
  rfl

def smallRow1867_small_right_right : List Nat := smallRow1867_block_003 ++ smallRow1867_block_004
theorem smallRow1867_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1867 = smallRow1867_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1867 ++ upperNeighbors PrimePairVertices.tree_block_004 1867 = _
  rw [smallRow1867_block_003_checked, smallRow1867_block_004_checked]
  rfl

def smallRow1867_small_right : List Nat := smallRow1867_block_002 ++ smallRow1867_small_right_right
theorem smallRow1867_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1867 = smallRow1867_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1867 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1867 = _
  rw [smallRow1867_block_002_checked, smallRow1867_small_right_right_checked]
  rfl

def smallRow1867_small : List Nat := smallRow1867_small_left ++ smallRow1867_small_right
theorem smallRow1867_small_checked : upperNeighbors PrimePairVertices.tree_small 1867 = smallRow1867_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1867 ++ upperNeighbors PrimePairVertices.tree_small_right 1867 = _
  rw [smallRow1867_small_left_checked, smallRow1867_small_right_checked]
  rfl

def smallRow1867 : List Nat := smallRow1867_small
theorem smallRow1867_checked : upperNeighbors PrimePairVertices.smallTree 1867 = smallRow1867 := smallRow1867_small_checked

def smallRow1871_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1871_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1871 = smallRow1871_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1871_block_001 : List Nat := [3137]
set_option maxRecDepth 2048 in
theorem smallRow1871_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1871 = smallRow1871_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1871_block_002 : List Nat := [4139, 4349, 4517, 4649, 4691, 4733, 5051, 5231, 5441]
set_option maxRecDepth 2048 in
theorem smallRow1871_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1871 = smallRow1871_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1871_block_003 : List Nat := [6323, 6959, 7001, 8171]
set_option maxRecDepth 2048 in
theorem smallRow1871_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1871 = smallRow1871_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1871_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1871_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1871 = smallRow1871_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1871_small_left : List Nat := smallRow1871_block_000 ++ smallRow1871_block_001
theorem smallRow1871_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1871 = smallRow1871_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1871 ++ upperNeighbors PrimePairVertices.tree_block_001 1871 = _
  rw [smallRow1871_block_000_checked, smallRow1871_block_001_checked]
  rfl

def smallRow1871_small_right_right : List Nat := smallRow1871_block_003 ++ smallRow1871_block_004
theorem smallRow1871_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1871 = smallRow1871_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1871 ++ upperNeighbors PrimePairVertices.tree_block_004 1871 = _
  rw [smallRow1871_block_003_checked, smallRow1871_block_004_checked]
  rfl

def smallRow1871_small_right : List Nat := smallRow1871_block_002 ++ smallRow1871_small_right_right
theorem smallRow1871_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1871 = smallRow1871_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1871 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1871 = _
  rw [smallRow1871_block_002_checked, smallRow1871_small_right_right_checked]
  rfl

def smallRow1871_small : List Nat := smallRow1871_small_left ++ smallRow1871_small_right
theorem smallRow1871_small_checked : upperNeighbors PrimePairVertices.tree_small 1871 = smallRow1871_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1871 ++ upperNeighbors PrimePairVertices.tree_small_right 1871 = _
  rw [smallRow1871_small_left_checked, smallRow1871_small_right_checked]
  rfl

def smallRow1871 : List Nat := smallRow1871_small
theorem smallRow1871_checked : upperNeighbors PrimePairVertices.smallTree 1871 = smallRow1871 := smallRow1871_small_checked

def smallRow1873_block_000 : List Nat := [2017]
set_option maxRecDepth 2048 in
theorem smallRow1873_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1873 = smallRow1873_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1873_block_001 : List Nat := [2269, 2437, 2719, 3511]
set_option maxRecDepth 2048 in
theorem smallRow1873_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1873 = smallRow1873_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1873_block_002 : List Nat := [5527, 5569]
set_option maxRecDepth 2048 in
theorem smallRow1873_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1873 = smallRow1873_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1873_block_003 : List Nat := [6301, 6427, 6709, 6841, 6961, 6997, 7507, 7963, 8011, 8059, 8179]
set_option maxRecDepth 2048 in
theorem smallRow1873_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1873 = smallRow1873_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1873_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1873_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1873 = smallRow1873_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1873_small_left : List Nat := smallRow1873_block_000 ++ smallRow1873_block_001
theorem smallRow1873_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1873 = smallRow1873_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1873 ++ upperNeighbors PrimePairVertices.tree_block_001 1873 = _
  rw [smallRow1873_block_000_checked, smallRow1873_block_001_checked]
  rfl

def smallRow1873_small_right_right : List Nat := smallRow1873_block_003 ++ smallRow1873_block_004
theorem smallRow1873_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1873 = smallRow1873_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1873 ++ upperNeighbors PrimePairVertices.tree_block_004 1873 = _
  rw [smallRow1873_block_003_checked, smallRow1873_block_004_checked]
  rfl

def smallRow1873_small_right : List Nat := smallRow1873_block_002 ++ smallRow1873_small_right_right
theorem smallRow1873_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1873 = smallRow1873_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1873 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1873 = _
  rw [smallRow1873_block_002_checked, smallRow1873_small_right_right_checked]
  rfl

def smallRow1873_small : List Nat := smallRow1873_small_left ++ smallRow1873_small_right
theorem smallRow1873_small_checked : upperNeighbors PrimePairVertices.tree_small 1873 = smallRow1873_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1873 ++ upperNeighbors PrimePairVertices.tree_small_right 1873 = _
  rw [smallRow1873_small_left_checked, smallRow1873_small_right_checked]
  rfl

def smallRow1873 : List Nat := smallRow1873_small
theorem smallRow1873_checked : upperNeighbors PrimePairVertices.smallTree 1873 = smallRow1873 := smallRow1873_small_checked

def smallRow1877_block_000 : List Nat := [1889]
set_option maxRecDepth 2048 in
theorem smallRow1877_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1877 = smallRow1877_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1877_block_001 : List Nat := [2447, 2549, 2909, 3119, 3917]
set_option maxRecDepth 2048 in
theorem smallRow1877_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1877 = smallRow1877_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1877_block_002 : List Nat := [4211, 4391, 4421, 4733, 5261, 5399, 5711, 6113]
set_option maxRecDepth 2048 in
theorem smallRow1877_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1877 = smallRow1877_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1877_block_003 : List Nat := [6287, 6491, 6971, 6983, 7823, 8087]
set_option maxRecDepth 2048 in
theorem smallRow1877_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1877 = smallRow1877_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1877_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1877_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1877 = smallRow1877_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1877_small_left : List Nat := smallRow1877_block_000 ++ smallRow1877_block_001
theorem smallRow1877_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1877 = smallRow1877_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1877 ++ upperNeighbors PrimePairVertices.tree_block_001 1877 = _
  rw [smallRow1877_block_000_checked, smallRow1877_block_001_checked]
  rfl

def smallRow1877_small_right_right : List Nat := smallRow1877_block_003 ++ smallRow1877_block_004
theorem smallRow1877_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1877 = smallRow1877_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1877 ++ upperNeighbors PrimePairVertices.tree_block_004 1877 = _
  rw [smallRow1877_block_003_checked, smallRow1877_block_004_checked]
  rfl

def smallRow1877_small_right : List Nat := smallRow1877_block_002 ++ smallRow1877_small_right_right
theorem smallRow1877_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1877 = smallRow1877_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1877 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1877 = _
  rw [smallRow1877_block_002_checked, smallRow1877_small_right_right_checked]
  rfl

def smallRow1877_small : List Nat := smallRow1877_small_left ++ smallRow1877_small_right
theorem smallRow1877_small_checked : upperNeighbors PrimePairVertices.tree_small 1877 = smallRow1877_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1877 ++ upperNeighbors PrimePairVertices.tree_small_right 1877 = _
  rw [smallRow1877_small_left_checked, smallRow1877_small_right_checked]
  rfl

def smallRow1877 : List Nat := smallRow1877_small
theorem smallRow1877_checked : upperNeighbors PrimePairVertices.smallTree 1877 = smallRow1877 := smallRow1877_small_checked

def smallRow1879_block_000 : List Nat := [1999]
set_option maxRecDepth 2048 in
theorem smallRow1879_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1879 = smallRow1879_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1879_block_001 : List Nat := [2281, 2341, 2713, 2971, 3331]
set_option maxRecDepth 2048 in
theorem smallRow1879_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1879 = smallRow1879_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1879_block_002 : List Nat := [4861]
set_option maxRecDepth 2048 in
theorem smallRow1879_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1879 = smallRow1879_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1879_block_003 : List Nat := [6823, 7297, 7411, 7951]
set_option maxRecDepth 2048 in
theorem smallRow1879_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1879 = smallRow1879_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1879_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1879_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1879 = smallRow1879_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1879_small_left : List Nat := smallRow1879_block_000 ++ smallRow1879_block_001
theorem smallRow1879_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1879 = smallRow1879_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1879 ++ upperNeighbors PrimePairVertices.tree_block_001 1879 = _
  rw [smallRow1879_block_000_checked, smallRow1879_block_001_checked]
  rfl

def smallRow1879_small_right_right : List Nat := smallRow1879_block_003 ++ smallRow1879_block_004
theorem smallRow1879_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1879 = smallRow1879_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1879 ++ upperNeighbors PrimePairVertices.tree_block_004 1879 = _
  rw [smallRow1879_block_003_checked, smallRow1879_block_004_checked]
  rfl

def smallRow1879_small_right : List Nat := smallRow1879_block_002 ++ smallRow1879_small_right_right
theorem smallRow1879_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1879 = smallRow1879_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1879 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1879 = _
  rw [smallRow1879_block_002_checked, smallRow1879_small_right_right_checked]
  rfl

def smallRow1879_small : List Nat := smallRow1879_small_left ++ smallRow1879_small_right
theorem smallRow1879_small_checked : upperNeighbors PrimePairVertices.tree_small 1879 = smallRow1879_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1879 ++ upperNeighbors PrimePairVertices.tree_small_right 1879 = _
  rw [smallRow1879_small_left_checked, smallRow1879_small_right_checked]
  rfl

def smallRow1879 : List Nat := smallRow1879_small
theorem smallRow1879_checked : upperNeighbors PrimePairVertices.smallTree 1879 = smallRow1879 := smallRow1879_small_checked

def smallRow1889_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1889_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1889 = smallRow1889_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1889_block_001 : List Nat := [2087, 2129, 2243, 2633, 2777, 2843, 2927, 2999, 3491, 4019]
set_option maxRecDepth 2048 in
theorem smallRow1889_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1889 = smallRow1889_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1889_block_002 : List Nat := [4919, 5297, 5417]
set_option maxRecDepth 2048 in
theorem smallRow1889_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1889 = smallRow1889_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1889_block_003 : List Nat := [6833, 7019, 8123]
set_option maxRecDepth 2048 in
theorem smallRow1889_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1889 = smallRow1889_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1889_block_004 : List Nat := [8291]
set_option maxRecDepth 2048 in
theorem smallRow1889_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1889 = smallRow1889_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow1889_small_left : List Nat := smallRow1889_block_000 ++ smallRow1889_block_001
theorem smallRow1889_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1889 = smallRow1889_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1889 ++ upperNeighbors PrimePairVertices.tree_block_001 1889 = _
  rw [smallRow1889_block_000_checked, smallRow1889_block_001_checked]
  rfl

def smallRow1889_small_right_right : List Nat := smallRow1889_block_003 ++ smallRow1889_block_004
theorem smallRow1889_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1889 = smallRow1889_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1889 ++ upperNeighbors PrimePairVertices.tree_block_004 1889 = _
  rw [smallRow1889_block_003_checked, smallRow1889_block_004_checked]
  rfl

def smallRow1889_small_right : List Nat := smallRow1889_block_002 ++ smallRow1889_small_right_right
theorem smallRow1889_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1889 = smallRow1889_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1889 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1889 = _
  rw [smallRow1889_block_002_checked, smallRow1889_small_right_right_checked]
  rfl

def smallRow1889_small : List Nat := smallRow1889_small_left ++ smallRow1889_small_right
theorem smallRow1889_small_checked : upperNeighbors PrimePairVertices.tree_small 1889 = smallRow1889_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1889 ++ upperNeighbors PrimePairVertices.tree_small_right 1889 = _
  rw [smallRow1889_small_left_checked, smallRow1889_small_right_checked]
  rfl

def smallRow1889 : List Nat := smallRow1889_small
theorem smallRow1889_checked : upperNeighbors PrimePairVertices.smallTree 1889 = smallRow1889 := smallRow1889_small_checked

end ProjectEulerSolutions.PrimePairCache
