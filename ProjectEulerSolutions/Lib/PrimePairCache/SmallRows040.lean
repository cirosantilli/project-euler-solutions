import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2141_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2141_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2141 = smallRow2141_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2141_block_001 : List Nat := [2267, 2309, 2621, 2663, 3167, 3191, 3881, 3947, 4049, 4091]
set_option maxRecDepth 2048 in
theorem smallRow2141_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2141 = smallRow2141_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2141_block_002 : List Nat := [4649, 5153, 5519, 6053]
set_option maxRecDepth 2048 in
theorem smallRow2141_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2141 = smallRow2141_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2141_block_003 : List Nat := [6737, 7349, 7547, 7691, 7727]
set_option maxRecDepth 2048 in
theorem smallRow2141_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2141 = smallRow2141_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2141_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2141_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2141 = smallRow2141_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2141_small_left : List Nat := smallRow2141_block_000 ++ smallRow2141_block_001
theorem smallRow2141_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2141 = smallRow2141_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2141 ++ upperNeighbors PrimePairVertices.tree_block_001 2141 = _
  rw [smallRow2141_block_000_checked, smallRow2141_block_001_checked]
  rfl

def smallRow2141_small_right_right : List Nat := smallRow2141_block_003 ++ smallRow2141_block_004
theorem smallRow2141_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2141 = smallRow2141_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2141 ++ upperNeighbors PrimePairVertices.tree_block_004 2141 = _
  rw [smallRow2141_block_003_checked, smallRow2141_block_004_checked]
  rfl

def smallRow2141_small_right : List Nat := smallRow2141_block_002 ++ smallRow2141_small_right_right
theorem smallRow2141_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2141 = smallRow2141_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2141 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2141 = _
  rw [smallRow2141_block_002_checked, smallRow2141_small_right_right_checked]
  rfl

def smallRow2141_small : List Nat := smallRow2141_small_left ++ smallRow2141_small_right
theorem smallRow2141_small_checked : upperNeighbors PrimePairVertices.tree_small 2141 = smallRow2141_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2141 ++ upperNeighbors PrimePairVertices.tree_small_right 2141 = _
  rw [smallRow2141_small_left_checked, smallRow2141_small_right_checked]
  rfl

def smallRow2141 : List Nat := smallRow2141_small
theorem smallRow2141_checked : upperNeighbors PrimePairVertices.smallTree 2141 = smallRow2141 := smallRow2141_small_checked

def smallRow2143_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2143_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2143 = smallRow2143_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2143_block_001 : List Nat := [2239, 2347]
set_option maxRecDepth 2048 in
theorem smallRow2143_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2143 = smallRow2143_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2143_block_002 : List Nat := [4657, 4801]
set_option maxRecDepth 2048 in
theorem smallRow2143_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2143 = smallRow2143_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2143_block_003 : List Nat := [6229, 6301, 6691, 6871, 8017, 8089]
set_option maxRecDepth 2048 in
theorem smallRow2143_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2143 = smallRow2143_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2143_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2143_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2143 = smallRow2143_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2143_small_left : List Nat := smallRow2143_block_000 ++ smallRow2143_block_001
theorem smallRow2143_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2143 = smallRow2143_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2143 ++ upperNeighbors PrimePairVertices.tree_block_001 2143 = _
  rw [smallRow2143_block_000_checked, smallRow2143_block_001_checked]
  rfl

def smallRow2143_small_right_right : List Nat := smallRow2143_block_003 ++ smallRow2143_block_004
theorem smallRow2143_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2143 = smallRow2143_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2143 ++ upperNeighbors PrimePairVertices.tree_block_004 2143 = _
  rw [smallRow2143_block_003_checked, smallRow2143_block_004_checked]
  rfl

def smallRow2143_small_right : List Nat := smallRow2143_block_002 ++ smallRow2143_small_right_right
theorem smallRow2143_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2143 = smallRow2143_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2143 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2143 = _
  rw [smallRow2143_block_002_checked, smallRow2143_small_right_right_checked]
  rfl

def smallRow2143_small : List Nat := smallRow2143_small_left ++ smallRow2143_small_right
theorem smallRow2143_small_checked : upperNeighbors PrimePairVertices.tree_small 2143 = smallRow2143_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2143 ++ upperNeighbors PrimePairVertices.tree_small_right 2143 = _
  rw [smallRow2143_small_left_checked, smallRow2143_small_right_checked]
  rfl

def smallRow2143 : List Nat := smallRow2143_small
theorem smallRow2143_checked : upperNeighbors PrimePairVertices.smallTree 2143 = smallRow2143 := smallRow2143_small_checked

def smallRow2153_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2153_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2153 = smallRow2153_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2153_block_001 : List Nat := [2417, 2663, 2969]
set_option maxRecDepth 2048 in
theorem smallRow2153_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2153 = smallRow2153_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2153_block_002 : List Nat := [4397, 4967, 5237, 5273, 5351, 6143]
set_option maxRecDepth 2048 in
theorem smallRow2153_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2153 = smallRow2153_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2153_block_003 : List Nat := [6863, 6869, 7121, 7151]
set_option maxRecDepth 2048 in
theorem smallRow2153_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2153 = smallRow2153_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2153_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2153_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2153 = smallRow2153_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2153_small_left : List Nat := smallRow2153_block_000 ++ smallRow2153_block_001
theorem smallRow2153_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2153 = smallRow2153_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2153 ++ upperNeighbors PrimePairVertices.tree_block_001 2153 = _
  rw [smallRow2153_block_000_checked, smallRow2153_block_001_checked]
  rfl

def smallRow2153_small_right_right : List Nat := smallRow2153_block_003 ++ smallRow2153_block_004
theorem smallRow2153_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2153 = smallRow2153_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2153 ++ upperNeighbors PrimePairVertices.tree_block_004 2153 = _
  rw [smallRow2153_block_003_checked, smallRow2153_block_004_checked]
  rfl

def smallRow2153_small_right : List Nat := smallRow2153_block_002 ++ smallRow2153_small_right_right
theorem smallRow2153_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2153 = smallRow2153_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2153 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2153 = _
  rw [smallRow2153_block_002_checked, smallRow2153_small_right_right_checked]
  rfl

def smallRow2153_small : List Nat := smallRow2153_small_left ++ smallRow2153_small_right
theorem smallRow2153_small_checked : upperNeighbors PrimePairVertices.tree_small 2153 = smallRow2153_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2153 ++ upperNeighbors PrimePairVertices.tree_small_right 2153 = _
  rw [smallRow2153_small_left_checked, smallRow2153_small_right_checked]
  rfl

def smallRow2153 : List Nat := smallRow2153_small
theorem smallRow2153_checked : upperNeighbors PrimePairVertices.smallTree 2153 = smallRow2153 := smallRow2153_small_checked

def smallRow2161_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2161_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2161 = smallRow2161_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2161_block_001 : List Nat := [2887]
set_option maxRecDepth 2048 in
theorem smallRow2161_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2161 = smallRow2161_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2161_block_002 : List Nat := [4357, 4597, 5479, 6079]
set_option maxRecDepth 2048 in
theorem smallRow2161_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2161 = smallRow2161_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2161_block_003 : List Nat := [6211, 6337, 6373, 6841, 6949, 7027, 7237, 7411, 7621, 7681, 7759]
set_option maxRecDepth 2048 in
theorem smallRow2161_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2161 = smallRow2161_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2161_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2161_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2161 = smallRow2161_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2161_small_left : List Nat := smallRow2161_block_000 ++ smallRow2161_block_001
theorem smallRow2161_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2161 = smallRow2161_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2161 ++ upperNeighbors PrimePairVertices.tree_block_001 2161 = _
  rw [smallRow2161_block_000_checked, smallRow2161_block_001_checked]
  rfl

def smallRow2161_small_right_right : List Nat := smallRow2161_block_003 ++ smallRow2161_block_004
theorem smallRow2161_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2161 = smallRow2161_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2161 ++ upperNeighbors PrimePairVertices.tree_block_004 2161 = _
  rw [smallRow2161_block_003_checked, smallRow2161_block_004_checked]
  rfl

def smallRow2161_small_right : List Nat := smallRow2161_block_002 ++ smallRow2161_small_right_right
theorem smallRow2161_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2161 = smallRow2161_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2161 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2161 = _
  rw [smallRow2161_block_002_checked, smallRow2161_small_right_right_checked]
  rfl

def smallRow2161_small : List Nat := smallRow2161_small_left ++ smallRow2161_small_right
theorem smallRow2161_small_checked : upperNeighbors PrimePairVertices.tree_small 2161 = smallRow2161_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2161 ++ upperNeighbors PrimePairVertices.tree_small_right 2161 = _
  rw [smallRow2161_small_left_checked, smallRow2161_small_right_checked]
  rfl

def smallRow2161 : List Nat := smallRow2161_small
theorem smallRow2161_checked : upperNeighbors PrimePairVertices.smallTree 2161 = smallRow2161 := smallRow2161_small_checked

def smallRow2179_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2179_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2179 = smallRow2179_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2179_block_001 : List Nat := [2503, 2917, 3169, 3253, 3469, 3631]
set_option maxRecDepth 2048 in
theorem smallRow2179_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2179 = smallRow2179_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2179_block_002 : List Nat := [4261, 5119, 5227, 5647, 5857]
set_option maxRecDepth 2048 in
theorem smallRow2179_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2179 = smallRow2179_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2179_block_003 : List Nat := [6343, 6529]
set_option maxRecDepth 2048 in
theorem smallRow2179_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2179 = smallRow2179_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2179_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2179_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2179 = smallRow2179_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2179_small_left : List Nat := smallRow2179_block_000 ++ smallRow2179_block_001
theorem smallRow2179_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2179 = smallRow2179_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2179 ++ upperNeighbors PrimePairVertices.tree_block_001 2179 = _
  rw [smallRow2179_block_000_checked, smallRow2179_block_001_checked]
  rfl

def smallRow2179_small_right_right : List Nat := smallRow2179_block_003 ++ smallRow2179_block_004
theorem smallRow2179_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2179 = smallRow2179_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2179 ++ upperNeighbors PrimePairVertices.tree_block_004 2179 = _
  rw [smallRow2179_block_003_checked, smallRow2179_block_004_checked]
  rfl

def smallRow2179_small_right : List Nat := smallRow2179_block_002 ++ smallRow2179_small_right_right
theorem smallRow2179_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2179 = smallRow2179_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2179 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2179 = _
  rw [smallRow2179_block_002_checked, smallRow2179_small_right_right_checked]
  rfl

def smallRow2179_small : List Nat := smallRow2179_small_left ++ smallRow2179_small_right
theorem smallRow2179_small_checked : upperNeighbors PrimePairVertices.tree_small 2179 = smallRow2179_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2179 ++ upperNeighbors PrimePairVertices.tree_small_right 2179 = _
  rw [smallRow2179_small_left_checked, smallRow2179_small_right_checked]
  rfl

def smallRow2179 : List Nat := smallRow2179_small
theorem smallRow2179_checked : upperNeighbors PrimePairVertices.smallTree 2179 = smallRow2179 := smallRow2179_small_checked

def smallRow2203_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2203_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2203 = smallRow2203_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2203_block_001 : List Nat := [2371, 2887, 3061]
set_option maxRecDepth 2048 in
theorem smallRow2203_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2203 = smallRow2203_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2203_block_002 : List Nat := [5521, 5557, 5623, 5647, 5857, 6037]
set_option maxRecDepth 2048 in
theorem smallRow2203_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2203 = smallRow2203_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2203_block_003 : List Nat := [6271, 6529, 6961, 7789, 7927, 8161]
set_option maxRecDepth 2048 in
theorem smallRow2203_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2203 = smallRow2203_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2203_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2203_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2203 = smallRow2203_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2203_small_left : List Nat := smallRow2203_block_000 ++ smallRow2203_block_001
theorem smallRow2203_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2203 = smallRow2203_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2203 ++ upperNeighbors PrimePairVertices.tree_block_001 2203 = _
  rw [smallRow2203_block_000_checked, smallRow2203_block_001_checked]
  rfl

def smallRow2203_small_right_right : List Nat := smallRow2203_block_003 ++ smallRow2203_block_004
theorem smallRow2203_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2203 = smallRow2203_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2203 ++ upperNeighbors PrimePairVertices.tree_block_004 2203 = _
  rw [smallRow2203_block_003_checked, smallRow2203_block_004_checked]
  rfl

def smallRow2203_small_right : List Nat := smallRow2203_block_002 ++ smallRow2203_small_right_right
theorem smallRow2203_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2203 = smallRow2203_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2203 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2203 = _
  rw [smallRow2203_block_002_checked, smallRow2203_small_right_right_checked]
  rfl

def smallRow2203_small : List Nat := smallRow2203_small_left ++ smallRow2203_small_right
theorem smallRow2203_small_checked : upperNeighbors PrimePairVertices.tree_small 2203 = smallRow2203_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2203 ++ upperNeighbors PrimePairVertices.tree_small_right 2203 = _
  rw [smallRow2203_small_left_checked, smallRow2203_small_right_checked]
  rfl

def smallRow2203 : List Nat := smallRow2203_small
theorem smallRow2203_checked : upperNeighbors PrimePairVertices.smallTree 2203 = smallRow2203 := smallRow2203_small_checked

def smallRow2207_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2207_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2207 = smallRow2207_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2207_block_001 : List Nat := [2339, 2753, 2777, 2819, 3347, 3929]
set_option maxRecDepth 2048 in
theorem smallRow2207_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2207 = smallRow2207_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2207_block_002 : List Nat := [4271, 4337, 5171, 6053]
set_option maxRecDepth 2048 in
theorem smallRow2207_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2207 = smallRow2207_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2207_block_003 : List Nat := [6563, 7193]
set_option maxRecDepth 2048 in
theorem smallRow2207_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2207 = smallRow2207_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2207_block_004 : List Nat := [8363]
set_option maxRecDepth 2048 in
theorem smallRow2207_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2207 = smallRow2207_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2207_small_left : List Nat := smallRow2207_block_000 ++ smallRow2207_block_001
theorem smallRow2207_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2207 = smallRow2207_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2207 ++ upperNeighbors PrimePairVertices.tree_block_001 2207 = _
  rw [smallRow2207_block_000_checked, smallRow2207_block_001_checked]
  rfl

def smallRow2207_small_right_right : List Nat := smallRow2207_block_003 ++ smallRow2207_block_004
theorem smallRow2207_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2207 = smallRow2207_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2207 ++ upperNeighbors PrimePairVertices.tree_block_004 2207 = _
  rw [smallRow2207_block_003_checked, smallRow2207_block_004_checked]
  rfl

def smallRow2207_small_right : List Nat := smallRow2207_block_002 ++ smallRow2207_small_right_right
theorem smallRow2207_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2207 = smallRow2207_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2207 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2207 = _
  rw [smallRow2207_block_002_checked, smallRow2207_small_right_right_checked]
  rfl

def smallRow2207_small : List Nat := smallRow2207_small_left ++ smallRow2207_small_right
theorem smallRow2207_small_checked : upperNeighbors PrimePairVertices.tree_small 2207 = smallRow2207_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2207 ++ upperNeighbors PrimePairVertices.tree_small_right 2207 = _
  rw [smallRow2207_small_left_checked, smallRow2207_small_right_checked]
  rfl

def smallRow2207 : List Nat := smallRow2207_small
theorem smallRow2207_checked : upperNeighbors PrimePairVertices.smallTree 2207 = smallRow2207 := smallRow2207_small_checked

def smallRow2213_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2213_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2213 = smallRow2213_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2213_block_001 : List Nat := [2711, 3581]
set_option maxRecDepth 2048 in
theorem smallRow2213_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2213 = smallRow2213_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2213_block_002 : List Nat := [4253, 4451, 4691, 5417, 6029]
set_option maxRecDepth 2048 in
theorem smallRow2213_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2213 = smallRow2213_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2213_block_003 : List Nat := [7457, 7547, 8093, 8147]
set_option maxRecDepth 2048 in
theorem smallRow2213_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2213 = smallRow2213_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2213_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow2213_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2213 = smallRow2213_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2213_small_left : List Nat := smallRow2213_block_000 ++ smallRow2213_block_001
theorem smallRow2213_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2213 = smallRow2213_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2213 ++ upperNeighbors PrimePairVertices.tree_block_001 2213 = _
  rw [smallRow2213_block_000_checked, smallRow2213_block_001_checked]
  rfl

def smallRow2213_small_right_right : List Nat := smallRow2213_block_003 ++ smallRow2213_block_004
theorem smallRow2213_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2213 = smallRow2213_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2213 ++ upperNeighbors PrimePairVertices.tree_block_004 2213 = _
  rw [smallRow2213_block_003_checked, smallRow2213_block_004_checked]
  rfl

def smallRow2213_small_right : List Nat := smallRow2213_block_002 ++ smallRow2213_small_right_right
theorem smallRow2213_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2213 = smallRow2213_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2213 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2213 = _
  rw [smallRow2213_block_002_checked, smallRow2213_small_right_right_checked]
  rfl

def smallRow2213_small : List Nat := smallRow2213_small_left ++ smallRow2213_small_right
theorem smallRow2213_small_checked : upperNeighbors PrimePairVertices.tree_small 2213 = smallRow2213_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2213 ++ upperNeighbors PrimePairVertices.tree_small_right 2213 = _
  rw [smallRow2213_small_left_checked, smallRow2213_small_right_checked]
  rfl

def smallRow2213 : List Nat := smallRow2213_small
theorem smallRow2213_checked : upperNeighbors PrimePairVertices.smallTree 2213 = smallRow2213 := smallRow2213_small_checked

end ProjectEulerSolutions.PrimePairCache
