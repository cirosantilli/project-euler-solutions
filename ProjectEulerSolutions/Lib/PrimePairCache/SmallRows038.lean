import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2027_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2027_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2027 = smallRow2027_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2027_block_001 : List Nat := [3119, 3761]
set_option maxRecDepth 2048 in
theorem smallRow2027_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2027 = smallRow2027_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2027_block_002 : List Nat := [4259, 5153, 5483]
set_option maxRecDepth 2048 in
theorem smallRow2027_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2027 = smallRow2027_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2027_block_003 : List Nat := [6203, 6701, 6911, 7253, 7457, 7703]
set_option maxRecDepth 2048 in
theorem smallRow2027_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2027 = smallRow2027_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2027_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2027_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2027 = smallRow2027_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2027_small_left : List Nat := smallRow2027_block_000 ++ smallRow2027_block_001
theorem smallRow2027_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2027 = smallRow2027_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2027 ++ upperNeighbors PrimePairVertices.tree_block_001 2027 = _
  rw [smallRow2027_block_000_checked, smallRow2027_block_001_checked]
  rfl

def smallRow2027_small_right_right : List Nat := smallRow2027_block_003 ++ smallRow2027_block_004
theorem smallRow2027_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2027 = smallRow2027_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2027 ++ upperNeighbors PrimePairVertices.tree_block_004 2027 = _
  rw [smallRow2027_block_003_checked, smallRow2027_block_004_checked]
  rfl

def smallRow2027_small_right : List Nat := smallRow2027_block_002 ++ smallRow2027_small_right_right
theorem smallRow2027_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2027 = smallRow2027_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2027 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2027 = _
  rw [smallRow2027_block_002_checked, smallRow2027_small_right_right_checked]
  rfl

def smallRow2027_small : List Nat := smallRow2027_small_left ++ smallRow2027_small_right
theorem smallRow2027_small_checked : upperNeighbors PrimePairVertices.tree_small 2027 = smallRow2027_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2027 ++ upperNeighbors PrimePairVertices.tree_small_right 2027 = _
  rw [smallRow2027_small_left_checked, smallRow2027_small_right_checked]
  rfl

def smallRow2027 : List Nat := smallRow2027_small
theorem smallRow2027_checked : upperNeighbors PrimePairVertices.smallTree 2027 = smallRow2027 := smallRow2027_small_checked

def smallRow2029_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2029_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2029 = smallRow2029_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2029_block_001 : List Nat := [2281, 2341, 2791, 3739]
set_option maxRecDepth 2048 in
theorem smallRow2029_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2029 = smallRow2029_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2029_block_002 : List Nat := [4507, 5101, 5227, 5281, 5419, 5689, 5701, 6121]
set_option maxRecDepth 2048 in
theorem smallRow2029_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2029 = smallRow2029_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2029_block_003 : List Nat := [6247, 7027, 7213, 7309]
set_option maxRecDepth 2048 in
theorem smallRow2029_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2029 = smallRow2029_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2029_block_004 : List Nat := [8293]
set_option maxRecDepth 2048 in
theorem smallRow2029_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2029 = smallRow2029_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2029_small_left : List Nat := smallRow2029_block_000 ++ smallRow2029_block_001
theorem smallRow2029_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2029 = smallRow2029_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2029 ++ upperNeighbors PrimePairVertices.tree_block_001 2029 = _
  rw [smallRow2029_block_000_checked, smallRow2029_block_001_checked]
  rfl

def smallRow2029_small_right_right : List Nat := smallRow2029_block_003 ++ smallRow2029_block_004
theorem smallRow2029_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2029 = smallRow2029_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2029 ++ upperNeighbors PrimePairVertices.tree_block_004 2029 = _
  rw [smallRow2029_block_003_checked, smallRow2029_block_004_checked]
  rfl

def smallRow2029_small_right : List Nat := smallRow2029_block_002 ++ smallRow2029_small_right_right
theorem smallRow2029_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2029 = smallRow2029_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2029 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2029 = _
  rw [smallRow2029_block_002_checked, smallRow2029_small_right_right_checked]
  rfl

def smallRow2029_small : List Nat := smallRow2029_small_left ++ smallRow2029_small_right
theorem smallRow2029_small_checked : upperNeighbors PrimePairVertices.tree_small 2029 = smallRow2029_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2029 ++ upperNeighbors PrimePairVertices.tree_small_right 2029 = _
  rw [smallRow2029_small_left_checked, smallRow2029_small_right_checked]
  rfl

def smallRow2029 : List Nat := smallRow2029_small
theorem smallRow2029_checked : upperNeighbors PrimePairVertices.smallTree 2029 = smallRow2029 := smallRow2029_small_checked

def smallRow2039_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2039_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2039 = smallRow2039_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2039_block_001 : List Nat := [2357, 2417, 2591, 3299, 3323, 3491, 3767, 3917, 4079]
set_option maxRecDepth 2048 in
theorem smallRow2039_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2039 = smallRow2039_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2039_block_002 : List Nat := [4523, 4799, 4937, 6053]
set_option maxRecDepth 2048 in
theorem smallRow2039_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2039 = smallRow2039_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2039_block_003 : List Nat := [6203, 6689, 7673]
set_option maxRecDepth 2048 in
theorem smallRow2039_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2039 = smallRow2039_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2039_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2039_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2039 = smallRow2039_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2039_small_left : List Nat := smallRow2039_block_000 ++ smallRow2039_block_001
theorem smallRow2039_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2039 = smallRow2039_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2039 ++ upperNeighbors PrimePairVertices.tree_block_001 2039 = _
  rw [smallRow2039_block_000_checked, smallRow2039_block_001_checked]
  rfl

def smallRow2039_small_right_right : List Nat := smallRow2039_block_003 ++ smallRow2039_block_004
theorem smallRow2039_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2039 = smallRow2039_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2039 ++ upperNeighbors PrimePairVertices.tree_block_004 2039 = _
  rw [smallRow2039_block_003_checked, smallRow2039_block_004_checked]
  rfl

def smallRow2039_small_right : List Nat := smallRow2039_block_002 ++ smallRow2039_small_right_right
theorem smallRow2039_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2039 = smallRow2039_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2039 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2039 = _
  rw [smallRow2039_block_002_checked, smallRow2039_small_right_right_checked]
  rfl

def smallRow2039_small : List Nat := smallRow2039_small_left ++ smallRow2039_small_right
theorem smallRow2039_small_checked : upperNeighbors PrimePairVertices.tree_small 2039 = smallRow2039_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2039 ++ upperNeighbors PrimePairVertices.tree_small_right 2039 = _
  rw [smallRow2039_small_left_checked, smallRow2039_small_right_checked]
  rfl

def smallRow2039 : List Nat := smallRow2039_small
theorem smallRow2039_checked : upperNeighbors PrimePairVertices.smallTree 2039 = smallRow2039 := smallRow2039_small_checked

def smallRow2053_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2053_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2053 = smallRow2053_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2053_block_001 : List Nat := [2797, 3049, 3361]
set_option maxRecDepth 2048 in
theorem smallRow2053_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2053 = smallRow2053_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2053_block_002 : List Nat := [4729, 4783, 5653]
set_option maxRecDepth 2048 in
theorem smallRow2053_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2053 = smallRow2053_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2053_block_003 : List Nat := [6343, 6361, 6421, 6619]
set_option maxRecDepth 2048 in
theorem smallRow2053_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2053 = smallRow2053_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2053_block_004 : List Nat := [8317]
set_option maxRecDepth 2048 in
theorem smallRow2053_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2053 = smallRow2053_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2053_small_left : List Nat := smallRow2053_block_000 ++ smallRow2053_block_001
theorem smallRow2053_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2053 = smallRow2053_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2053 ++ upperNeighbors PrimePairVertices.tree_block_001 2053 = _
  rw [smallRow2053_block_000_checked, smallRow2053_block_001_checked]
  rfl

def smallRow2053_small_right_right : List Nat := smallRow2053_block_003 ++ smallRow2053_block_004
theorem smallRow2053_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2053 = smallRow2053_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2053 ++ upperNeighbors PrimePairVertices.tree_block_004 2053 = _
  rw [smallRow2053_block_003_checked, smallRow2053_block_004_checked]
  rfl

def smallRow2053_small_right : List Nat := smallRow2053_block_002 ++ smallRow2053_small_right_right
theorem smallRow2053_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2053 = smallRow2053_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2053 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2053 = _
  rw [smallRow2053_block_002_checked, smallRow2053_small_right_right_checked]
  rfl

def smallRow2053_small : List Nat := smallRow2053_small_left ++ smallRow2053_small_right
theorem smallRow2053_small_checked : upperNeighbors PrimePairVertices.tree_small 2053 = smallRow2053_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2053 ++ upperNeighbors PrimePairVertices.tree_small_right 2053 = _
  rw [smallRow2053_small_left_checked, smallRow2053_small_right_checked]
  rfl

def smallRow2053 : List Nat := smallRow2053_small
theorem smallRow2053_checked : upperNeighbors PrimePairVertices.smallTree 2053 = smallRow2053 := smallRow2053_small_checked

def smallRow2063_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2063_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2063 = smallRow2063_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2063_block_001 : List Nat := [2099, 2243, 2333, 2411, 3089, 3617, 3803]
set_option maxRecDepth 2048 in
theorem smallRow2063_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2063 = smallRow2063_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2063_block_002 : List Nat := [4217, 4349, 5231, 5477, 5711, 5903, 5981]
set_option maxRecDepth 2048 in
theorem smallRow2063_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2063 = smallRow2063_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2063_block_003 : List Nat := [6449, 6599]
set_option maxRecDepth 2048 in
theorem smallRow2063_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2063 = smallRow2063_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2063_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2063_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2063 = smallRow2063_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2063_small_left : List Nat := smallRow2063_block_000 ++ smallRow2063_block_001
theorem smallRow2063_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2063 = smallRow2063_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2063 ++ upperNeighbors PrimePairVertices.tree_block_001 2063 = _
  rw [smallRow2063_block_000_checked, smallRow2063_block_001_checked]
  rfl

def smallRow2063_small_right_right : List Nat := smallRow2063_block_003 ++ smallRow2063_block_004
theorem smallRow2063_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2063 = smallRow2063_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2063 ++ upperNeighbors PrimePairVertices.tree_block_004 2063 = _
  rw [smallRow2063_block_003_checked, smallRow2063_block_004_checked]
  rfl

def smallRow2063_small_right : List Nat := smallRow2063_block_002 ++ smallRow2063_small_right_right
theorem smallRow2063_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2063 = smallRow2063_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2063 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2063 = _
  rw [smallRow2063_block_002_checked, smallRow2063_small_right_right_checked]
  rfl

def smallRow2063_small : List Nat := smallRow2063_small_left ++ smallRow2063_small_right
theorem smallRow2063_small_checked : upperNeighbors PrimePairVertices.tree_small 2063 = smallRow2063_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2063 ++ upperNeighbors PrimePairVertices.tree_small_right 2063 = _
  rw [smallRow2063_small_left_checked, smallRow2063_small_right_checked]
  rfl

def smallRow2063 : List Nat := smallRow2063_small
theorem smallRow2063_checked : upperNeighbors PrimePairVertices.smallTree 2063 = smallRow2063 := smallRow2063_small_checked

def smallRow2069_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2069_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2069 = smallRow2069_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2069_block_001 : List Nat := [2297, 2339]
set_option maxRecDepth 2048 in
theorem smallRow2069_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2069 = smallRow2069_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2069_block_002 : List Nat := [4397, 4637, 4691, 5021]
set_option maxRecDepth 2048 in
theorem smallRow2069_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2069 = smallRow2069_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2069_block_003 : List Nat := [6827, 6959, 6983, 7121, 7253, 7283, 7757, 7877]
set_option maxRecDepth 2048 in
theorem smallRow2069_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2069 = smallRow2069_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2069_block_004 : List Nat := [8219, 8369]
set_option maxRecDepth 2048 in
theorem smallRow2069_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2069 = smallRow2069_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2069_small_left : List Nat := smallRow2069_block_000 ++ smallRow2069_block_001
theorem smallRow2069_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2069 = smallRow2069_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2069 ++ upperNeighbors PrimePairVertices.tree_block_001 2069 = _
  rw [smallRow2069_block_000_checked, smallRow2069_block_001_checked]
  rfl

def smallRow2069_small_right_right : List Nat := smallRow2069_block_003 ++ smallRow2069_block_004
theorem smallRow2069_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2069 = smallRow2069_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2069 ++ upperNeighbors PrimePairVertices.tree_block_004 2069 = _
  rw [smallRow2069_block_003_checked, smallRow2069_block_004_checked]
  rfl

def smallRow2069_small_right : List Nat := smallRow2069_block_002 ++ smallRow2069_small_right_right
theorem smallRow2069_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2069 = smallRow2069_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2069 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2069 = _
  rw [smallRow2069_block_002_checked, smallRow2069_small_right_right_checked]
  rfl

def smallRow2069_small : List Nat := smallRow2069_small_left ++ smallRow2069_small_right
theorem smallRow2069_small_checked : upperNeighbors PrimePairVertices.tree_small 2069 = smallRow2069_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2069 ++ upperNeighbors PrimePairVertices.tree_small_right 2069 = _
  rw [smallRow2069_small_left_checked, smallRow2069_small_right_checked]
  rfl

def smallRow2069 : List Nat := smallRow2069_small
theorem smallRow2069_checked : upperNeighbors PrimePairVertices.smallTree 2069 = smallRow2069 := smallRow2069_small_checked

def smallRow2081_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2081_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2081 = smallRow2081_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2081_block_001 : List Nat := [2699, 3323]
set_option maxRecDepth 2048 in
theorem smallRow2081_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2081 = smallRow2081_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2081_block_002 : List Nat := [4127, 4349, 4691, 4943, 5153, 5441, 5657, 5801, 5807]
set_option maxRecDepth 2048 in
theorem smallRow2081_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2081 = smallRow2081_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2081_block_003 : List Nat := [6689, 6791, 7109, 7229, 7793]
set_option maxRecDepth 2048 in
theorem smallRow2081_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2081 = smallRow2081_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2081_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2081_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2081 = smallRow2081_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2081_small_left : List Nat := smallRow2081_block_000 ++ smallRow2081_block_001
theorem smallRow2081_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2081 = smallRow2081_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2081 ++ upperNeighbors PrimePairVertices.tree_block_001 2081 = _
  rw [smallRow2081_block_000_checked, smallRow2081_block_001_checked]
  rfl

def smallRow2081_small_right_right : List Nat := smallRow2081_block_003 ++ smallRow2081_block_004
theorem smallRow2081_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2081 = smallRow2081_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2081 ++ upperNeighbors PrimePairVertices.tree_block_004 2081 = _
  rw [smallRow2081_block_003_checked, smallRow2081_block_004_checked]
  rfl

def smallRow2081_small_right : List Nat := smallRow2081_block_002 ++ smallRow2081_small_right_right
theorem smallRow2081_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2081 = smallRow2081_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2081 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2081 = _
  rw [smallRow2081_block_002_checked, smallRow2081_small_right_right_checked]
  rfl

def smallRow2081_small : List Nat := smallRow2081_small_left ++ smallRow2081_small_right
theorem smallRow2081_small_checked : upperNeighbors PrimePairVertices.tree_small 2081 = smallRow2081_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2081 ++ upperNeighbors PrimePairVertices.tree_small_right 2081 = _
  rw [smallRow2081_small_left_checked, smallRow2081_small_right_checked]
  rfl

def smallRow2081 : List Nat := smallRow2081_small
theorem smallRow2081_checked : upperNeighbors PrimePairVertices.smallTree 2081 = smallRow2081 := smallRow2081_small_checked

def smallRow2083_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2083_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2083 = smallRow2083_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2083_block_001 : List Nat := [2971, 3343, 3511]
set_option maxRecDepth 2048 in
theorem smallRow2083_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2083 = smallRow2083_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2083_block_002 : List Nat := [4111, 4243, 4651, 5443, 5827, 5923, 6121]
set_option maxRecDepth 2048 in
theorem smallRow2083_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2083 = smallRow2083_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2083_block_003 : List Nat := [6997, 7459]
set_option maxRecDepth 2048 in
theorem smallRow2083_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2083 = smallRow2083_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2083_block_004 : List Nat := [8221]
set_option maxRecDepth 2048 in
theorem smallRow2083_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2083 = smallRow2083_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2083_small_left : List Nat := smallRow2083_block_000 ++ smallRow2083_block_001
theorem smallRow2083_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2083 = smallRow2083_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2083 ++ upperNeighbors PrimePairVertices.tree_block_001 2083 = _
  rw [smallRow2083_block_000_checked, smallRow2083_block_001_checked]
  rfl

def smallRow2083_small_right_right : List Nat := smallRow2083_block_003 ++ smallRow2083_block_004
theorem smallRow2083_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2083 = smallRow2083_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2083 ++ upperNeighbors PrimePairVertices.tree_block_004 2083 = _
  rw [smallRow2083_block_003_checked, smallRow2083_block_004_checked]
  rfl

def smallRow2083_small_right : List Nat := smallRow2083_block_002 ++ smallRow2083_small_right_right
theorem smallRow2083_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2083 = smallRow2083_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2083 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2083 = _
  rw [smallRow2083_block_002_checked, smallRow2083_small_right_right_checked]
  rfl

def smallRow2083_small : List Nat := smallRow2083_small_left ++ smallRow2083_small_right
theorem smallRow2083_small_checked : upperNeighbors PrimePairVertices.tree_small 2083 = smallRow2083_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2083 ++ upperNeighbors PrimePairVertices.tree_small_right 2083 = _
  rw [smallRow2083_small_left_checked, smallRow2083_small_right_checked]
  rfl

def smallRow2083 : List Nat := smallRow2083_small
theorem smallRow2083_checked : upperNeighbors PrimePairVertices.smallTree 2083 = smallRow2083 := smallRow2083_small_checked

end ProjectEulerSolutions.PrimePairCache
