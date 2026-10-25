import ProjectEulerSolutions.Lib.PrimePairVertices.Trees
import ProjectEulerSolutions.Lib.PrimePairFast

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow2221_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2221_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2221 = smallRow2221_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2221_block_001 : List Nat := [2389, 3049, 3847]
set_option maxRecDepth 2048 in
theorem smallRow2221_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2221 = smallRow2221_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2221_block_002 : List Nat := [4111, 4201, 4513, 4519, 4663, 4909, 5527, 5581, 5737, 5839, 6043]
set_option maxRecDepth 2048 in
theorem smallRow2221_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2221 = smallRow2221_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2221_block_003 : List Nat := [6451, 6673, 7639, 7681, 7873, 8059]
set_option maxRecDepth 2048 in
theorem smallRow2221_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2221 = smallRow2221_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2221_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2221_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2221 = smallRow2221_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2221_small_left : List Nat := smallRow2221_block_000 ++ smallRow2221_block_001
theorem smallRow2221_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2221 = smallRow2221_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2221 ++ upperNeighbors PrimePairVertices.tree_block_001 2221 = _
  rw [smallRow2221_block_000_checked, smallRow2221_block_001_checked]
  rfl

def smallRow2221_small_right_right : List Nat := smallRow2221_block_003 ++ smallRow2221_block_004
theorem smallRow2221_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2221 = smallRow2221_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2221 ++ upperNeighbors PrimePairVertices.tree_block_004 2221 = _
  rw [smallRow2221_block_003_checked, smallRow2221_block_004_checked]
  rfl

def smallRow2221_small_right : List Nat := smallRow2221_block_002 ++ smallRow2221_small_right_right
theorem smallRow2221_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2221 = smallRow2221_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2221 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2221 = _
  rw [smallRow2221_block_002_checked, smallRow2221_small_right_right_checked]
  rfl

def smallRow2221_small : List Nat := smallRow2221_small_left ++ smallRow2221_small_right
theorem smallRow2221_small_checked : upperNeighbors PrimePairVertices.tree_small 2221 = smallRow2221_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2221 ++ upperNeighbors PrimePairVertices.tree_small_right 2221 = _
  rw [smallRow2221_small_left_checked, smallRow2221_small_right_checked]
  rfl

def smallRow2221 : List Nat := smallRow2221_small
theorem smallRow2221_checked : upperNeighbors PrimePairVertices.smallTree 2221 = smallRow2221 := smallRow2221_small_checked

def smallRow2237_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2237_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2237 = smallRow2237_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2237_block_001 : List Nat := [2381, 2411, 2549, 3251, 3257, 3929, 4019, 4049]
set_option maxRecDepth 2048 in
theorem smallRow2237_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2237 = smallRow2237_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2237_block_002 : List Nat := [4139, 4349, 4421, 4481, 4637, 5693]
set_option maxRecDepth 2048 in
theorem smallRow2237_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2237 = smallRow2237_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2237_block_003 : List Nat := [6581, 6659, 6689, 6701, 6779, 6791, 6917]
set_option maxRecDepth 2048 in
theorem smallRow2237_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2237 = smallRow2237_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2237_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2237_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2237 = smallRow2237_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2237_small_left : List Nat := smallRow2237_block_000 ++ smallRow2237_block_001
theorem smallRow2237_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2237 = smallRow2237_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2237 ++ upperNeighbors PrimePairVertices.tree_block_001 2237 = _
  rw [smallRow2237_block_000_checked, smallRow2237_block_001_checked]
  rfl

def smallRow2237_small_right_right : List Nat := smallRow2237_block_003 ++ smallRow2237_block_004
theorem smallRow2237_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2237 = smallRow2237_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2237 ++ upperNeighbors PrimePairVertices.tree_block_004 2237 = _
  rw [smallRow2237_block_003_checked, smallRow2237_block_004_checked]
  rfl

def smallRow2237_small_right : List Nat := smallRow2237_block_002 ++ smallRow2237_small_right_right
theorem smallRow2237_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2237 = smallRow2237_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2237 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2237 = _
  rw [smallRow2237_block_002_checked, smallRow2237_small_right_right_checked]
  rfl

def smallRow2237_small : List Nat := smallRow2237_small_left ++ smallRow2237_small_right
theorem smallRow2237_small_checked : upperNeighbors PrimePairVertices.tree_small 2237 = smallRow2237_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2237 ++ upperNeighbors PrimePairVertices.tree_small_right 2237 = _
  rw [smallRow2237_small_left_checked, smallRow2237_small_right_checked]
  rfl

def smallRow2237 : List Nat := smallRow2237_small
theorem smallRow2237_checked : upperNeighbors PrimePairVertices.smallTree 2237 = smallRow2237 := smallRow2237_small_checked

def smallRow2239_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2239_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2239 = smallRow2239_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2239_block_001 : List Nat := [2857, 3853, 3919]
set_option maxRecDepth 2048 in
theorem smallRow2239_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2239 = smallRow2239_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2239_block_002 : List Nat := [4129, 4261, 4441, 4987, 5179, 5209, 5557, 5683, 6007]
set_option maxRecDepth 2048 in
theorem smallRow2239_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2239 = smallRow2239_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2239_block_003 : List Nat := [6481, 6823, 7027, 7237, 7351, 7393, 7411, 7951]
set_option maxRecDepth 2048 in
theorem smallRow2239_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2239 = smallRow2239_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2239_block_004 : List Nat := [8221, 8329]
set_option maxRecDepth 2048 in
theorem smallRow2239_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2239 = smallRow2239_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2239_small_left : List Nat := smallRow2239_block_000 ++ smallRow2239_block_001
theorem smallRow2239_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2239 = smallRow2239_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2239 ++ upperNeighbors PrimePairVertices.tree_block_001 2239 = _
  rw [smallRow2239_block_000_checked, smallRow2239_block_001_checked]
  rfl

def smallRow2239_small_right_right : List Nat := smallRow2239_block_003 ++ smallRow2239_block_004
theorem smallRow2239_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2239 = smallRow2239_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2239 ++ upperNeighbors PrimePairVertices.tree_block_004 2239 = _
  rw [smallRow2239_block_003_checked, smallRow2239_block_004_checked]
  rfl

def smallRow2239_small_right : List Nat := smallRow2239_block_002 ++ smallRow2239_small_right_right
theorem smallRow2239_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2239 = smallRow2239_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2239 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2239 = _
  rw [smallRow2239_block_002_checked, smallRow2239_small_right_right_checked]
  rfl

def smallRow2239_small : List Nat := smallRow2239_small_left ++ smallRow2239_small_right
theorem smallRow2239_small_checked : upperNeighbors PrimePairVertices.tree_small 2239 = smallRow2239_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2239 ++ upperNeighbors PrimePairVertices.tree_small_right 2239 = _
  rw [smallRow2239_small_left_checked, smallRow2239_small_right_checked]
  rfl

def smallRow2239 : List Nat := smallRow2239_small
theorem smallRow2239_checked : upperNeighbors PrimePairVertices.smallTree 2239 = smallRow2239 := smallRow2239_small_checked

def smallRow2243_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2243_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2243 = smallRow2243_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2243_block_001 : List Nat := [2273, 2411, 2789, 3539]
set_option maxRecDepth 2048 in
theorem smallRow2243_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2243 = smallRow2243_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2243_block_002 : List Nat := [4133, 4457, 5477, 5927]
set_option maxRecDepth 2048 in
theorem smallRow2243_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2243 = smallRow2243_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2243_block_003 : List Nat := [6599, 6899, 6977, 7193, 7487, 8081]
set_option maxRecDepth 2048 in
theorem smallRow2243_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2243 = smallRow2243_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2243_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2243_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2243 = smallRow2243_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2243_small_left : List Nat := smallRow2243_block_000 ++ smallRow2243_block_001
theorem smallRow2243_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2243 = smallRow2243_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2243 ++ upperNeighbors PrimePairVertices.tree_block_001 2243 = _
  rw [smallRow2243_block_000_checked, smallRow2243_block_001_checked]
  rfl

def smallRow2243_small_right_right : List Nat := smallRow2243_block_003 ++ smallRow2243_block_004
theorem smallRow2243_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2243 = smallRow2243_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2243 ++ upperNeighbors PrimePairVertices.tree_block_004 2243 = _
  rw [smallRow2243_block_003_checked, smallRow2243_block_004_checked]
  rfl

def smallRow2243_small_right : List Nat := smallRow2243_block_002 ++ smallRow2243_small_right_right
theorem smallRow2243_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2243 = smallRow2243_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2243 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2243 = _
  rw [smallRow2243_block_002_checked, smallRow2243_small_right_right_checked]
  rfl

def smallRow2243_small : List Nat := smallRow2243_small_left ++ smallRow2243_small_right
theorem smallRow2243_small_checked : upperNeighbors PrimePairVertices.tree_small 2243 = smallRow2243_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2243 ++ upperNeighbors PrimePairVertices.tree_small_right 2243 = _
  rw [smallRow2243_small_left_checked, smallRow2243_small_right_checked]
  rfl

def smallRow2243 : List Nat := smallRow2243_small
theorem smallRow2243_checked : upperNeighbors PrimePairVertices.smallTree 2243 = smallRow2243 := smallRow2243_small_checked

def smallRow2251_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2251_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2251 = smallRow2251_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2251_block_001 : List Nat := [2593, 3217, 3307, 4021]
set_option maxRecDepth 2048 in
theorem smallRow2251_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2251 = smallRow2251_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2251_block_002 : List Nat := [4111, 4201, 5503, 5881]
set_option maxRecDepth 2048 in
theorem smallRow2251_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2251 = smallRow2251_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2251_block_003 : List Nat := [6451, 6841, 7669, 7927, 8191]
set_option maxRecDepth 2048 in
theorem smallRow2251_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2251 = smallRow2251_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2251_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow2251_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2251 = smallRow2251_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2251_small_left : List Nat := smallRow2251_block_000 ++ smallRow2251_block_001
theorem smallRow2251_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2251 = smallRow2251_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2251 ++ upperNeighbors PrimePairVertices.tree_block_001 2251 = _
  rw [smallRow2251_block_000_checked, smallRow2251_block_001_checked]
  rfl

def smallRow2251_small_right_right : List Nat := smallRow2251_block_003 ++ smallRow2251_block_004
theorem smallRow2251_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2251 = smallRow2251_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2251 ++ upperNeighbors PrimePairVertices.tree_block_004 2251 = _
  rw [smallRow2251_block_003_checked, smallRow2251_block_004_checked]
  rfl

def smallRow2251_small_right : List Nat := smallRow2251_block_002 ++ smallRow2251_small_right_right
theorem smallRow2251_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2251 = smallRow2251_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2251 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2251 = _
  rw [smallRow2251_block_002_checked, smallRow2251_small_right_right_checked]
  rfl

def smallRow2251_small : List Nat := smallRow2251_small_left ++ smallRow2251_small_right
theorem smallRow2251_small_checked : upperNeighbors PrimePairVertices.tree_small 2251 = smallRow2251_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2251 ++ upperNeighbors PrimePairVertices.tree_small_right 2251 = _
  rw [smallRow2251_small_left_checked, smallRow2251_small_right_checked]
  rfl

def smallRow2251 : List Nat := smallRow2251_small
theorem smallRow2251_checked : upperNeighbors PrimePairVertices.smallTree 2251 = smallRow2251 := smallRow2251_small_checked

def smallRow2267_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2267_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2267 = smallRow2267_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2267_block_001 : List Nat := [2729, 2939, 3323, 3527]
set_option maxRecDepth 2048 in
theorem smallRow2267_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2267 = smallRow2267_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2267_block_002 : List Nat := [4637, 4721, 5669, 5843, 5867, 5879, 5897, 6053]
set_option maxRecDepth 2048 in
theorem smallRow2267_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2267 = smallRow2267_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2267_block_003 : List Nat := [6389, 6791, 6833, 6917, 7673, 8147]
set_option maxRecDepth 2048 in
theorem smallRow2267_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2267 = smallRow2267_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2267_block_004 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallRow2267_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2267 = smallRow2267_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2267_small_left : List Nat := smallRow2267_block_000 ++ smallRow2267_block_001
theorem smallRow2267_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2267 = smallRow2267_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2267 ++ upperNeighbors PrimePairVertices.tree_block_001 2267 = _
  rw [smallRow2267_block_000_checked, smallRow2267_block_001_checked]
  rfl

def smallRow2267_small_right_right : List Nat := smallRow2267_block_003 ++ smallRow2267_block_004
theorem smallRow2267_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2267 = smallRow2267_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2267 ++ upperNeighbors PrimePairVertices.tree_block_004 2267 = _
  rw [smallRow2267_block_003_checked, smallRow2267_block_004_checked]
  rfl

def smallRow2267_small_right : List Nat := smallRow2267_block_002 ++ smallRow2267_small_right_right
theorem smallRow2267_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2267 = smallRow2267_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2267 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2267 = _
  rw [smallRow2267_block_002_checked, smallRow2267_small_right_right_checked]
  rfl

def smallRow2267_small : List Nat := smallRow2267_small_left ++ smallRow2267_small_right
theorem smallRow2267_small_checked : upperNeighbors PrimePairVertices.tree_small 2267 = smallRow2267_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2267 ++ upperNeighbors PrimePairVertices.tree_small_right 2267 = _
  rw [smallRow2267_small_left_checked, smallRow2267_small_right_checked]
  rfl

def smallRow2267 : List Nat := smallRow2267_small
theorem smallRow2267_checked : upperNeighbors PrimePairVertices.smallTree 2267 = smallRow2267 := smallRow2267_small_checked

def smallRow2269_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2269_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2269 = smallRow2269_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2269_block_001 : List Nat := [2281, 3067, 3373, 3613, 3739, 4057]
set_option maxRecDepth 2048 in
theorem smallRow2269_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2269 = smallRow2269_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2269_block_002 : List Nat := [4621, 4663, 4909, 5107, 5347, 5821, 5923]
set_option maxRecDepth 2048 in
theorem smallRow2269_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2269 = smallRow2269_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2269_block_003 : List Nat := [7027, 7237, 7489, 7561, 7879]
set_option maxRecDepth 2048 in
theorem smallRow2269_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2269 = smallRow2269_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2269_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2269_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2269 = smallRow2269_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2269_small_left : List Nat := smallRow2269_block_000 ++ smallRow2269_block_001
theorem smallRow2269_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2269 = smallRow2269_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2269 ++ upperNeighbors PrimePairVertices.tree_block_001 2269 = _
  rw [smallRow2269_block_000_checked, smallRow2269_block_001_checked]
  rfl

def smallRow2269_small_right_right : List Nat := smallRow2269_block_003 ++ smallRow2269_block_004
theorem smallRow2269_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2269 = smallRow2269_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2269 ++ upperNeighbors PrimePairVertices.tree_block_004 2269 = _
  rw [smallRow2269_block_003_checked, smallRow2269_block_004_checked]
  rfl

def smallRow2269_small_right : List Nat := smallRow2269_block_002 ++ smallRow2269_small_right_right
theorem smallRow2269_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2269 = smallRow2269_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2269 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2269 = _
  rw [smallRow2269_block_002_checked, smallRow2269_small_right_right_checked]
  rfl

def smallRow2269_small : List Nat := smallRow2269_small_left ++ smallRow2269_small_right
theorem smallRow2269_small_checked : upperNeighbors PrimePairVertices.tree_small 2269 = smallRow2269_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2269 ++ upperNeighbors PrimePairVertices.tree_small_right 2269 = _
  rw [smallRow2269_small_left_checked, smallRow2269_small_right_checked]
  rfl

def smallRow2269 : List Nat := smallRow2269_small
theorem smallRow2269_checked : upperNeighbors PrimePairVertices.smallTree 2269 = smallRow2269 := smallRow2269_small_checked

def smallRow2273_block_000 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow2273_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 2273 = smallRow2273_block_000 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2273_block_001 : List Nat := [2609, 2789, 2903, 2999, 3041, 3461, 3761]
set_option maxRecDepth 2048 in
theorem smallRow2273_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 2273 = smallRow2273_block_001 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2273_block_002 : List Nat := [4973, 5399, 5711, 5861, 5897]
set_option maxRecDepth 2048 in
theorem smallRow2273_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 2273 = smallRow2273_block_002 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2273_block_003 : List Nat := [7283, 8147]
set_option maxRecDepth 2048 in
theorem smallRow2273_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 2273 = smallRow2273_block_003 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2273_block_004 : List Nat := [8297]
set_option maxRecDepth 2048 in
theorem smallRow2273_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 2273 = smallRow2273_block_004 := by
  rw [← Fast.upperNeighbors_eq_source]
  decide

def smallRow2273_small_left : List Nat := smallRow2273_block_000 ++ smallRow2273_block_001
theorem smallRow2273_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 2273 = smallRow2273_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 2273 ++ upperNeighbors PrimePairVertices.tree_block_001 2273 = _
  rw [smallRow2273_block_000_checked, smallRow2273_block_001_checked]
  rfl

def smallRow2273_small_right_right : List Nat := smallRow2273_block_003 ++ smallRow2273_block_004
theorem smallRow2273_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 2273 = smallRow2273_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 2273 ++ upperNeighbors PrimePairVertices.tree_block_004 2273 = _
  rw [smallRow2273_block_003_checked, smallRow2273_block_004_checked]
  rfl

def smallRow2273_small_right : List Nat := smallRow2273_block_002 ++ smallRow2273_small_right_right
theorem smallRow2273_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 2273 = smallRow2273_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 2273 ++ upperNeighbors PrimePairVertices.tree_small_right_right 2273 = _
  rw [smallRow2273_block_002_checked, smallRow2273_small_right_right_checked]
  rfl

def smallRow2273_small : List Nat := smallRow2273_small_left ++ smallRow2273_small_right
theorem smallRow2273_small_checked : upperNeighbors PrimePairVertices.tree_small 2273 = smallRow2273_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 2273 ++ upperNeighbors PrimePairVertices.tree_small_right 2273 = _
  rw [smallRow2273_small_left_checked, smallRow2273_small_right_checked]
  rfl

def smallRow2273 : List Nat := smallRow2273_small
theorem smallRow2273_checked : upperNeighbors PrimePairVertices.smallTree 2273 = smallRow2273 := smallRow2273_small_checked

end ProjectEulerSolutions.PrimePairCache
