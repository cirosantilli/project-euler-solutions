import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow277_block_000 : List Nat := [331, 499, 751, 787, 859, 1381, 1471, 1693, 1933]
set_option maxRecDepth 2048 in
theorem smallRow277_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 277 = smallRow277_block_000 := by decide

def smallRow277_block_001 : List Nat := [2281, 2437, 2593, 3037, 3079, 3307, 3919]
set_option maxRecDepth 2048 in
theorem smallRow277_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 277 = smallRow277_block_001 := by decide

def smallRow277_block_002 : List Nat := [4327, 4483, 5119, 5323, 5431]
set_option maxRecDepth 2048 in
theorem smallRow277_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 277 = smallRow277_block_002 := by decide

def smallRow277_block_003 : List Nat := [6343, 6427, 6841, 7213, 7741, 7993]
set_option maxRecDepth 2048 in
theorem smallRow277_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 277 = smallRow277_block_003 := by decide

def smallRow277_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow277_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 277 = smallRow277_block_004 := by decide

def smallRow277_small_left : List Nat := smallRow277_block_000 ++ smallRow277_block_001
theorem smallRow277_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 277 = smallRow277_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 277 ++ upperNeighbors PrimePairVertices.tree_block_001 277 = _
  rw [smallRow277_block_000_checked, smallRow277_block_001_checked]
  rfl

def smallRow277_small_right_right : List Nat := smallRow277_block_003 ++ smallRow277_block_004
theorem smallRow277_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 277 = smallRow277_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 277 ++ upperNeighbors PrimePairVertices.tree_block_004 277 = _
  rw [smallRow277_block_003_checked, smallRow277_block_004_checked]
  rfl

def smallRow277_small_right : List Nat := smallRow277_block_002 ++ smallRow277_small_right_right
theorem smallRow277_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 277 = smallRow277_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 277 ++ upperNeighbors PrimePairVertices.tree_small_right_right 277 = _
  rw [smallRow277_block_002_checked, smallRow277_small_right_right_checked]
  rfl

def smallRow277_small : List Nat := smallRow277_small_left ++ smallRow277_small_right
theorem smallRow277_small_checked : upperNeighbors PrimePairVertices.tree_small 277 = smallRow277_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 277 ++ upperNeighbors PrimePairVertices.tree_small_right 277 = _
  rw [smallRow277_small_left_checked, smallRow277_small_right_checked]
  rfl

def smallRow277 : List Nat := smallRow277_small
theorem smallRow277_checked : upperNeighbors PrimePairVertices.smallTree 277 = smallRow277 := smallRow277_small_checked

def smallRow281_block_000 : List Nat := [419, 509, 557, 653, 719, 797, 857, 971, 1181, 1427, 1559, 1619, 1973]
set_option maxRecDepth 2048 in
theorem smallRow281_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 281 = smallRow281_block_000 := by decide

def smallRow281_block_001 : List Nat := [2207, 2267, 2687, 2819, 3191, 3329, 3677, 3917]
set_option maxRecDepth 2048 in
theorem smallRow281_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 281 = smallRow281_block_001 := by decide

def smallRow281_block_002 : List Nat := [4787, 5009, 5399, 5843, 6029]
set_option maxRecDepth 2048 in
theorem smallRow281_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 281 = smallRow281_block_002 := by decide

def smallRow281_block_003 : List Nat := [6263, 6899, 7229, 7481, 7487, 8111]
set_option maxRecDepth 2048 in
theorem smallRow281_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 281 = smallRow281_block_003 := by decide

def smallRow281_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow281_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 281 = smallRow281_block_004 := by decide

def smallRow281_small_left : List Nat := smallRow281_block_000 ++ smallRow281_block_001
theorem smallRow281_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 281 = smallRow281_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 281 ++ upperNeighbors PrimePairVertices.tree_block_001 281 = _
  rw [smallRow281_block_000_checked, smallRow281_block_001_checked]
  rfl

def smallRow281_small_right_right : List Nat := smallRow281_block_003 ++ smallRow281_block_004
theorem smallRow281_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 281 = smallRow281_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 281 ++ upperNeighbors PrimePairVertices.tree_block_004 281 = _
  rw [smallRow281_block_003_checked, smallRow281_block_004_checked]
  rfl

def smallRow281_small_right : List Nat := smallRow281_block_002 ++ smallRow281_small_right_right
theorem smallRow281_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 281 = smallRow281_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 281 ++ upperNeighbors PrimePairVertices.tree_small_right_right 281 = _
  rw [smallRow281_block_002_checked, smallRow281_small_right_right_checked]
  rfl

def smallRow281_small : List Nat := smallRow281_small_left ++ smallRow281_small_right
theorem smallRow281_small_checked : upperNeighbors PrimePairVertices.tree_small 281 = smallRow281_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 281 ++ upperNeighbors PrimePairVertices.tree_small_right 281 = _
  rw [smallRow281_small_left_checked, smallRow281_small_right_checked]
  rfl

def smallRow281 : List Nat := smallRow281_small
theorem smallRow281_checked : upperNeighbors PrimePairVertices.smallTree 281 = smallRow281 := smallRow281_small_checked

def smallRow283_block_000 : List Nat := [397, 463, 487, 541, 601, 1051, 1237, 1321, 1429, 1489, 1657]
set_option maxRecDepth 2048 in
theorem smallRow283_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 283 = smallRow283_block_000 := by decide

def smallRow283_block_001 : List Nat := [2113, 2287, 2503, 3001, 3529]
set_option maxRecDepth 2048 in
theorem smallRow283_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 283 = smallRow283_block_001 := by decide

def smallRow283_block_002 : List Nat := [4813, 5653, 5737, 5851, 5857]
set_option maxRecDepth 2048 in
theorem smallRow283_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 283 = smallRow283_block_002 := by decide

def smallRow283_block_003 : List Nat := [6397, 6607, 6733, 6763, 7039, 7057, 8191]
set_option maxRecDepth 2048 in
theorem smallRow283_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 283 = smallRow283_block_003 := by decide

def smallRow283_block_004 : List Nat := [8287, 8293]
set_option maxRecDepth 2048 in
theorem smallRow283_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 283 = smallRow283_block_004 := by decide

def smallRow283_small_left : List Nat := smallRow283_block_000 ++ smallRow283_block_001
theorem smallRow283_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 283 = smallRow283_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 283 ++ upperNeighbors PrimePairVertices.tree_block_001 283 = _
  rw [smallRow283_block_000_checked, smallRow283_block_001_checked]
  rfl

def smallRow283_small_right_right : List Nat := smallRow283_block_003 ++ smallRow283_block_004
theorem smallRow283_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 283 = smallRow283_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 283 ++ upperNeighbors PrimePairVertices.tree_block_004 283 = _
  rw [smallRow283_block_003_checked, smallRow283_block_004_checked]
  rfl

def smallRow283_small_right : List Nat := smallRow283_block_002 ++ smallRow283_small_right_right
theorem smallRow283_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 283 = smallRow283_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 283 ++ upperNeighbors PrimePairVertices.tree_small_right_right 283 = _
  rw [smallRow283_block_002_checked, smallRow283_small_right_right_checked]
  rfl

def smallRow283_small : List Nat := smallRow283_small_left ++ smallRow283_small_right
theorem smallRow283_small_checked : upperNeighbors PrimePairVertices.tree_small 283 = smallRow283_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 283 ++ upperNeighbors PrimePairVertices.tree_small_right 283 = _
  rw [smallRow283_small_left_checked, smallRow283_small_right_checked]
  rfl

def smallRow283 : List Nat := smallRow283_small
theorem smallRow283_checked : upperNeighbors PrimePairVertices.smallTree 283 = smallRow283 := smallRow283_small_checked

def smallRow293_block_000 : List Nat := [311, 467, 617, 827, 1319, 1433, 1493, 1583, 1823, 1997]
set_option maxRecDepth 2048 in
theorem smallRow293_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 293 = smallRow293_block_000 := by decide

def smallRow293_block_001 : List Nat := [2411, 2459, 2777, 3083, 4091]
set_option maxRecDepth 2048 in
theorem smallRow293_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 293 = smallRow293_block_001 := by decide

def smallRow293_block_002 : List Nat := [4751, 5147, 5171, 5189, 5717]
set_option maxRecDepth 2048 in
theorem smallRow293_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 293 = smallRow293_block_002 := by decide

def smallRow293_block_003 : List Nat := [7523, 7757, 8009]
set_option maxRecDepth 2048 in
theorem smallRow293_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 293 = smallRow293_block_003 := by decide

def smallRow293_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow293_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 293 = smallRow293_block_004 := by decide

def smallRow293_small_left : List Nat := smallRow293_block_000 ++ smallRow293_block_001
theorem smallRow293_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 293 = smallRow293_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 293 ++ upperNeighbors PrimePairVertices.tree_block_001 293 = _
  rw [smallRow293_block_000_checked, smallRow293_block_001_checked]
  rfl

def smallRow293_small_right_right : List Nat := smallRow293_block_003 ++ smallRow293_block_004
theorem smallRow293_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 293 = smallRow293_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 293 ++ upperNeighbors PrimePairVertices.tree_block_004 293 = _
  rw [smallRow293_block_003_checked, smallRow293_block_004_checked]
  rfl

def smallRow293_small_right : List Nat := smallRow293_block_002 ++ smallRow293_small_right_right
theorem smallRow293_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 293 = smallRow293_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 293 ++ upperNeighbors PrimePairVertices.tree_small_right_right 293 = _
  rw [smallRow293_block_002_checked, smallRow293_small_right_right_checked]
  rfl

def smallRow293_small : List Nat := smallRow293_small_left ++ smallRow293_small_right
theorem smallRow293_small_checked : upperNeighbors PrimePairVertices.tree_small 293 = smallRow293_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 293 ++ upperNeighbors PrimePairVertices.tree_small_right 293 = _
  rw [smallRow293_small_left_checked, smallRow293_small_right_checked]
  rfl

def smallRow293 : List Nat := smallRow293_small
theorem smallRow293_checked : upperNeighbors PrimePairVertices.smallTree 293 = smallRow293 := smallRow293_small_checked

def smallRow307_block_000 : List Nat := [367, 523, 577, 631, 733, 1069, 1093, 1279, 1381, 1993]
set_option maxRecDepth 2048 in
theorem smallRow307_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 307 = smallRow307_block_000 := by decide

def smallRow307_block_001 : List Nat := [2137, 2269, 2467, 3547, 3943]
set_option maxRecDepth 2048 in
theorem smallRow307_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 307 = smallRow307_block_001 := by decide

def smallRow307_block_002 : List Nat := [4243, 5119, 5209, 5857]
set_option maxRecDepth 2048 in
theorem smallRow307_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 307 = smallRow307_block_002 := by decide

def smallRow307_block_003 : List Nat := [6373, 6679, 7159, 7681, 7927, 8059]
set_option maxRecDepth 2048 in
theorem smallRow307_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 307 = smallRow307_block_003 := by decide

def smallRow307_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow307_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 307 = smallRow307_block_004 := by decide

def smallRow307_small_left : List Nat := smallRow307_block_000 ++ smallRow307_block_001
theorem smallRow307_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 307 = smallRow307_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 307 ++ upperNeighbors PrimePairVertices.tree_block_001 307 = _
  rw [smallRow307_block_000_checked, smallRow307_block_001_checked]
  rfl

def smallRow307_small_right_right : List Nat := smallRow307_block_003 ++ smallRow307_block_004
theorem smallRow307_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 307 = smallRow307_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 307 ++ upperNeighbors PrimePairVertices.tree_block_004 307 = _
  rw [smallRow307_block_003_checked, smallRow307_block_004_checked]
  rfl

def smallRow307_small_right : List Nat := smallRow307_block_002 ++ smallRow307_small_right_right
theorem smallRow307_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 307 = smallRow307_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 307 ++ upperNeighbors PrimePairVertices.tree_small_right_right 307 = _
  rw [smallRow307_block_002_checked, smallRow307_small_right_right_checked]
  rfl

def smallRow307_small : List Nat := smallRow307_small_left ++ smallRow307_small_right
theorem smallRow307_small_checked : upperNeighbors PrimePairVertices.tree_small 307 = smallRow307_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 307 ++ upperNeighbors PrimePairVertices.tree_small_right 307 = _
  rw [smallRow307_small_left_checked, smallRow307_small_right_checked]
  rfl

def smallRow307 : List Nat := smallRow307_small
theorem smallRow307_checked : upperNeighbors PrimePairVertices.smallTree 307 = smallRow307 := smallRow307_small_checked

def smallRow311_block_000 : List Nat := [359, 653, 677, 821, 827, 881, 1223]
set_option maxRecDepth 2048 in
theorem smallRow311_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 311 = smallRow311_block_000 := by decide

def smallRow311_block_001 : List Nat := [2111, 2141, 2381, 2441, 2609, 2909, 3863, 4019]
set_option maxRecDepth 2048 in
theorem smallRow311_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 311 = smallRow311_block_001 := by decide

def smallRow311_block_002 : List Nat := [4871, 4967, 5171, 5237, 5279, 6011, 6131]
set_option maxRecDepth 2048 in
theorem smallRow311_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 311 = smallRow311_block_002 := by decide

def smallRow311_block_003 : List Nat := [6263, 6761, 7013, 7211, 7523, 7559, 8069]
set_option maxRecDepth 2048 in
theorem smallRow311_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 311 = smallRow311_block_003 := by decide

def smallRow311_block_004 : List Nat := [8237, 8387]
set_option maxRecDepth 2048 in
theorem smallRow311_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 311 = smallRow311_block_004 := by decide

def smallRow311_small_left : List Nat := smallRow311_block_000 ++ smallRow311_block_001
theorem smallRow311_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 311 = smallRow311_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 311 ++ upperNeighbors PrimePairVertices.tree_block_001 311 = _
  rw [smallRow311_block_000_checked, smallRow311_block_001_checked]
  rfl

def smallRow311_small_right_right : List Nat := smallRow311_block_003 ++ smallRow311_block_004
theorem smallRow311_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 311 = smallRow311_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 311 ++ upperNeighbors PrimePairVertices.tree_block_004 311 = _
  rw [smallRow311_block_003_checked, smallRow311_block_004_checked]
  rfl

def smallRow311_small_right : List Nat := smallRow311_block_002 ++ smallRow311_small_right_right
theorem smallRow311_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 311 = smallRow311_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 311 ++ upperNeighbors PrimePairVertices.tree_small_right_right 311 = _
  rw [smallRow311_block_002_checked, smallRow311_small_right_right_checked]
  rfl

def smallRow311_small : List Nat := smallRow311_small_left ++ smallRow311_small_right
theorem smallRow311_small_checked : upperNeighbors PrimePairVertices.tree_small 311 = smallRow311_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 311 ++ upperNeighbors PrimePairVertices.tree_small_right 311 = _
  rw [smallRow311_small_left_checked, smallRow311_small_right_checked]
  rfl

def smallRow311 : List Nat := smallRow311_small
theorem smallRow311_checked : upperNeighbors PrimePairVertices.smallTree 311 = smallRow311 := smallRow311_small_checked

def smallRow313_block_000 : List Nat := [619, 727, 991, 1231, 1669, 1747]
set_option maxRecDepth 2048 in
theorem smallRow313_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 313 = smallRow313_block_000 := by decide

def smallRow313_block_001 : List Nat := [2281, 3469, 3583, 3733, 3853]
set_option maxRecDepth 2048 in
theorem smallRow313_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 313 = smallRow313_block_001 := by decide

def smallRow313_block_002 : List Nat := [4231, 4591, 6037, 6079]
set_option maxRecDepth 2048 in
theorem smallRow313_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 313 = smallRow313_block_002 := by decide

def smallRow313_block_003 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow313_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 313 = smallRow313_block_003 := by decide

def smallRow313_block_004 : List Nat := [8221, 8329]
set_option maxRecDepth 2048 in
theorem smallRow313_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 313 = smallRow313_block_004 := by decide

def smallRow313_small_left : List Nat := smallRow313_block_000 ++ smallRow313_block_001
theorem smallRow313_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 313 = smallRow313_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 313 ++ upperNeighbors PrimePairVertices.tree_block_001 313 = _
  rw [smallRow313_block_000_checked, smallRow313_block_001_checked]
  rfl

def smallRow313_small_right_right : List Nat := smallRow313_block_003 ++ smallRow313_block_004
theorem smallRow313_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 313 = smallRow313_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 313 ++ upperNeighbors PrimePairVertices.tree_block_004 313 = _
  rw [smallRow313_block_003_checked, smallRow313_block_004_checked]
  rfl

def smallRow313_small_right : List Nat := smallRow313_block_002 ++ smallRow313_small_right_right
theorem smallRow313_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 313 = smallRow313_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 313 ++ upperNeighbors PrimePairVertices.tree_small_right_right 313 = _
  rw [smallRow313_block_002_checked, smallRow313_small_right_right_checked]
  rfl

def smallRow313_small : List Nat := smallRow313_small_left ++ smallRow313_small_right
theorem smallRow313_small_checked : upperNeighbors PrimePairVertices.tree_small 313 = smallRow313_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 313 ++ upperNeighbors PrimePairVertices.tree_small_right 313 = _
  rw [smallRow313_small_left_checked, smallRow313_small_right_checked]
  rfl

def smallRow313 : List Nat := smallRow313_small
theorem smallRow313_checked : upperNeighbors PrimePairVertices.smallTree 313 = smallRow313 := smallRow313_small_checked

def smallRow317_block_000 : List Nat := [353, 419, 503, 773, 983, 1187, 1601, 1697, 1709, 2027]
set_option maxRecDepth 2048 in
theorem smallRow317_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 317 = smallRow317_block_000 := by decide

def smallRow317_block_001 : List Nat := [2237, 2243, 2399, 2441, 2801, 3119, 3539, 3659, 4091]
set_option maxRecDepth 2048 in
theorem smallRow317_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 317 = smallRow317_block_001 := by decide

def smallRow317_block_002 : List Nat := [4793, 4799, 4973, 5009, 5099, 5801, 6053]
set_option maxRecDepth 2048 in
theorem smallRow317_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 317 = smallRow317_block_002 := by decide

def smallRow317_block_003 : List Nat := [6221, 7607, 7949]
set_option maxRecDepth 2048 in
theorem smallRow317_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 317 = smallRow317_block_003 := by decide

def smallRow317_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow317_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 317 = smallRow317_block_004 := by decide

def smallRow317_small_left : List Nat := smallRow317_block_000 ++ smallRow317_block_001
theorem smallRow317_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 317 = smallRow317_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 317 ++ upperNeighbors PrimePairVertices.tree_block_001 317 = _
  rw [smallRow317_block_000_checked, smallRow317_block_001_checked]
  rfl

def smallRow317_small_right_right : List Nat := smallRow317_block_003 ++ smallRow317_block_004
theorem smallRow317_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 317 = smallRow317_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 317 ++ upperNeighbors PrimePairVertices.tree_block_004 317 = _
  rw [smallRow317_block_003_checked, smallRow317_block_004_checked]
  rfl

def smallRow317_small_right : List Nat := smallRow317_block_002 ++ smallRow317_small_right_right
theorem smallRow317_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 317 = smallRow317_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 317 ++ upperNeighbors PrimePairVertices.tree_small_right_right 317 = _
  rw [smallRow317_block_002_checked, smallRow317_small_right_right_checked]
  rfl

def smallRow317_small : List Nat := smallRow317_small_left ++ smallRow317_small_right
theorem smallRow317_small_checked : upperNeighbors PrimePairVertices.tree_small 317 = smallRow317_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 317 ++ upperNeighbors PrimePairVertices.tree_small_right 317 = _
  rw [smallRow317_small_left_checked, smallRow317_small_right_checked]
  rfl

def smallRow317 : List Nat := smallRow317_small
theorem smallRow317_checked : upperNeighbors PrimePairVertices.smallTree 317 = smallRow317 := smallRow317_small_checked

end ProjectEulerSolutions.PrimePairCache
