import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow233_block_000 : List Nat := [239, 251, 347, 617, 881, 911, 983, 1151, 1163, 1187, 1583]
set_option maxRecDepth 2048 in
theorem smallRow233_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 233 = smallRow233_block_000 := by decide

def smallRow233_block_001 : List Nat := [2969, 3083, 3167, 3329, 3389]
set_option maxRecDepth 2048 in
theorem smallRow233_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 233 = smallRow233_block_001 := by decide

def smallRow233_block_002 : List Nat := [4337, 4793, 5009, 5297, 5591, 5807, 5813, 6011]
set_option maxRecDepth 2048 in
theorem smallRow233_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 233 = smallRow233_block_002 := by decide

def smallRow233_block_003 : List Nat := [6263, 6359, 6947, 7001, 7043, 7187, 7547, 7607, 7949, 8123]
set_option maxRecDepth 2048 in
theorem smallRow233_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 233 = smallRow233_block_003 := by decide

def smallRow233_block_004 : List Nat := [8291]
set_option maxRecDepth 2048 in
theorem smallRow233_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 233 = smallRow233_block_004 := by decide

def smallRow233_small_left : List Nat := smallRow233_block_000 ++ smallRow233_block_001
theorem smallRow233_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 233 = smallRow233_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 233 ++ upperNeighbors PrimePairVertices.tree_block_001 233 = _
  rw [smallRow233_block_000_checked, smallRow233_block_001_checked]
  rfl

def smallRow233_small_right_right : List Nat := smallRow233_block_003 ++ smallRow233_block_004
theorem smallRow233_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 233 = smallRow233_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 233 ++ upperNeighbors PrimePairVertices.tree_block_004 233 = _
  rw [smallRow233_block_003_checked, smallRow233_block_004_checked]
  rfl

def smallRow233_small_right : List Nat := smallRow233_block_002 ++ smallRow233_small_right_right
theorem smallRow233_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 233 = smallRow233_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 233 ++ upperNeighbors PrimePairVertices.tree_small_right_right 233 = _
  rw [smallRow233_block_002_checked, smallRow233_small_right_right_checked]
  rfl

def smallRow233_small : List Nat := smallRow233_small_left ++ smallRow233_small_right
theorem smallRow233_small_checked : upperNeighbors PrimePairVertices.tree_small 233 = smallRow233_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 233 ++ upperNeighbors PrimePairVertices.tree_small_right 233 = _
  rw [smallRow233_small_left_checked, smallRow233_small_right_checked]
  rfl

def smallRow233 : List Nat := smallRow233_small
theorem smallRow233_checked : upperNeighbors PrimePairVertices.smallTree 233 = smallRow233 := smallRow233_small_checked

def smallRow239_block_000 : List Nat := [263, 347, 461, 509, 641, 929, 947, 977, 1013, 1049, 1091, 1439, 1637, 1847]
set_option maxRecDepth 2048 in
theorem smallRow239_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 239 = smallRow239_block_000 := by decide

def smallRow239_block_001 : List Nat := [2333, 2447, 2459, 2729, 2861, 3257, 3467, 3761]
set_option maxRecDepth 2048 in
theorem smallRow239_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 239 = smallRow239_block_001 := by decide

def smallRow239_block_002 : List Nat := [4157, 4241, 4787, 5039, 5303, 5807, 5879, 5927, 6131]
set_option maxRecDepth 2048 in
theorem smallRow239_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 239 = smallRow239_block_002 := by decide

def smallRow239_block_003 : List Nat := [6197, 6257, 6659, 7523, 7541, 7823, 8087, 8111]
set_option maxRecDepth 2048 in
theorem smallRow239_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 239 = smallRow239_block_003 := by decide

def smallRow239_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow239_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 239 = smallRow239_block_004 := by decide

def smallRow239_small_left : List Nat := smallRow239_block_000 ++ smallRow239_block_001
theorem smallRow239_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 239 = smallRow239_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 239 ++ upperNeighbors PrimePairVertices.tree_block_001 239 = _
  rw [smallRow239_block_000_checked, smallRow239_block_001_checked]
  rfl

def smallRow239_small_right_right : List Nat := smallRow239_block_003 ++ smallRow239_block_004
theorem smallRow239_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 239 = smallRow239_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 239 ++ upperNeighbors PrimePairVertices.tree_block_004 239 = _
  rw [smallRow239_block_003_checked, smallRow239_block_004_checked]
  rfl

def smallRow239_small_right : List Nat := smallRow239_block_002 ++ smallRow239_small_right_right
theorem smallRow239_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 239 = smallRow239_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 239 ++ upperNeighbors PrimePairVertices.tree_small_right_right 239 = _
  rw [smallRow239_block_002_checked, smallRow239_small_right_right_checked]
  rfl

def smallRow239_small : List Nat := smallRow239_small_left ++ smallRow239_small_right
theorem smallRow239_small_checked : upperNeighbors PrimePairVertices.tree_small 239 = smallRow239_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 239 ++ upperNeighbors PrimePairVertices.tree_small_right 239 = _
  rw [smallRow239_small_left_checked, smallRow239_small_right_checked]
  rfl

def smallRow239 : List Nat := smallRow239_small
theorem smallRow239_checked : upperNeighbors PrimePairVertices.smallTree 239 = smallRow239 := smallRow239_small_checked

def smallRow241_block_000 : List Nat := [271, 313, 421, 601, 691, 727, 739, 811, 823, 883, 1867, 2029]
set_option maxRecDepth 2048 in
theorem smallRow241_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 241 = smallRow241_block_000 := by decide

def smallRow241_block_001 : List Nat := [2671, 2731, 2797, 2833, 3217, 3253, 3511, 3823]
set_option maxRecDepth 2048 in
theorem smallRow241_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 241 = smallRow241_block_001 := by decide

def smallRow241_block_002 : List Nat := [4129, 4549, 5431, 5503, 5689]
set_option maxRecDepth 2048 in
theorem smallRow241_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 241 = smallRow241_block_002 := by decide

def smallRow241_block_003 : List Nat := [6361, 6607, 6763, 7741]
set_option maxRecDepth 2048 in
theorem smallRow241_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 241 = smallRow241_block_003 := by decide

def smallRow241_block_004 : List Nat := [8329]
set_option maxRecDepth 2048 in
theorem smallRow241_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 241 = smallRow241_block_004 := by decide

def smallRow241_small_left : List Nat := smallRow241_block_000 ++ smallRow241_block_001
theorem smallRow241_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 241 = smallRow241_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 241 ++ upperNeighbors PrimePairVertices.tree_block_001 241 = _
  rw [smallRow241_block_000_checked, smallRow241_block_001_checked]
  rfl

def smallRow241_small_right_right : List Nat := smallRow241_block_003 ++ smallRow241_block_004
theorem smallRow241_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 241 = smallRow241_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 241 ++ upperNeighbors PrimePairVertices.tree_block_004 241 = _
  rw [smallRow241_block_003_checked, smallRow241_block_004_checked]
  rfl

def smallRow241_small_right : List Nat := smallRow241_block_002 ++ smallRow241_small_right_right
theorem smallRow241_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 241 = smallRow241_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 241 ++ upperNeighbors PrimePairVertices.tree_small_right_right 241 = _
  rw [smallRow241_block_002_checked, smallRow241_small_right_right_checked]
  rfl

def smallRow241_small : List Nat := smallRow241_small_left ++ smallRow241_small_right
theorem smallRow241_small_checked : upperNeighbors PrimePairVertices.tree_small 241 = smallRow241_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 241 ++ upperNeighbors PrimePairVertices.tree_small_right 241 = _
  rw [smallRow241_small_left_checked, smallRow241_small_right_checked]
  rfl

def smallRow241 : List Nat := smallRow241_small
theorem smallRow241_checked : upperNeighbors PrimePairVertices.smallTree 241 = smallRow241 := smallRow241_small_checked

def smallRow251_block_000 : List Nat := [347, 431, 491, 761, 941, 971, 1151, 1433, 1709, 1979]
set_option maxRecDepth 2048 in
theorem smallRow251_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 251 = smallRow251_block_000 := by decide

def smallRow251_block_001 : List Nat := [2063, 2207, 2417, 2789, 2801, 2999, 3011, 3347, 3557, 3803, 4019]
set_option maxRecDepth 2048 in
theorem smallRow251_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 251 = smallRow251_block_001 := by decide

def smallRow251_block_002 : List Nat := [4877, 5147, 5189, 5237, 5273, 5651, 5897]
set_option maxRecDepth 2048 in
theorem smallRow251_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 251 = smallRow251_block_002 := by decide

def smallRow251_block_003 : List Nat := [6197, 6473, 6581, 6863, 7079, 7127, 7499, 7901, 7919, 8093]
set_option maxRecDepth 2048 in
theorem smallRow251_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 251 = smallRow251_block_003 := by decide

def smallRow251_block_004 : List Nat := [8387]
set_option maxRecDepth 2048 in
theorem smallRow251_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 251 = smallRow251_block_004 := by decide

def smallRow251_small_left : List Nat := smallRow251_block_000 ++ smallRow251_block_001
theorem smallRow251_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 251 = smallRow251_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 251 ++ upperNeighbors PrimePairVertices.tree_block_001 251 = _
  rw [smallRow251_block_000_checked, smallRow251_block_001_checked]
  rfl

def smallRow251_small_right_right : List Nat := smallRow251_block_003 ++ smallRow251_block_004
theorem smallRow251_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 251 = smallRow251_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 251 ++ upperNeighbors PrimePairVertices.tree_block_004 251 = _
  rw [smallRow251_block_003_checked, smallRow251_block_004_checked]
  rfl

def smallRow251_small_right : List Nat := smallRow251_block_002 ++ smallRow251_small_right_right
theorem smallRow251_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 251 = smallRow251_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 251 ++ upperNeighbors PrimePairVertices.tree_small_right_right 251 = _
  rw [smallRow251_block_002_checked, smallRow251_small_right_right_checked]
  rfl

def smallRow251_small : List Nat := smallRow251_small_left ++ smallRow251_small_right
theorem smallRow251_small_checked : upperNeighbors PrimePairVertices.tree_small 251 = smallRow251_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 251 ++ upperNeighbors PrimePairVertices.tree_small_right 251 = _
  rw [smallRow251_small_left_checked, smallRow251_small_right_checked]
  rfl

def smallRow251 : List Nat := smallRow251_small
theorem smallRow251_checked : upperNeighbors PrimePairVertices.smallTree 251 = smallRow251 := smallRow251_small_checked

def smallRow257_block_000 : List Nat := [263, 293, 1277, 1451, 1619, 1697, 1721, 1823, 1949, 2027]
set_option maxRecDepth 2048 in
theorem smallRow257_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 257 = smallRow257_block_000 := by decide

def smallRow257_block_001 : List Nat := [2243, 2267, 2393, 2621, 2741, 2939, 3023, 3209, 3413, 3833]
set_option maxRecDepth 2048 in
theorem smallRow257_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 257 = smallRow257_block_001 := by decide

def smallRow257_block_002 : List Nat := [4337, 4409, 4889, 4967, 5021, 5351, 5939]
set_option maxRecDepth 2048 in
theorem smallRow257_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 257 = smallRow257_block_002 := by decide

def smallRow257_block_003 : List Nat := [6983, 7079, 7283, 7907]
set_option maxRecDepth 2048 in
theorem smallRow257_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 257 = smallRow257_block_003 := by decide

def smallRow257_block_004 : List Nat := [8273]
set_option maxRecDepth 2048 in
theorem smallRow257_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 257 = smallRow257_block_004 := by decide

def smallRow257_small_left : List Nat := smallRow257_block_000 ++ smallRow257_block_001
theorem smallRow257_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 257 = smallRow257_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 257 ++ upperNeighbors PrimePairVertices.tree_block_001 257 = _
  rw [smallRow257_block_000_checked, smallRow257_block_001_checked]
  rfl

def smallRow257_small_right_right : List Nat := smallRow257_block_003 ++ smallRow257_block_004
theorem smallRow257_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 257 = smallRow257_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 257 ++ upperNeighbors PrimePairVertices.tree_block_004 257 = _
  rw [smallRow257_block_003_checked, smallRow257_block_004_checked]
  rfl

def smallRow257_small_right : List Nat := smallRow257_block_002 ++ smallRow257_small_right_right
theorem smallRow257_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 257 = smallRow257_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 257 ++ upperNeighbors PrimePairVertices.tree_small_right_right 257 = _
  rw [smallRow257_block_002_checked, smallRow257_small_right_right_checked]
  rfl

def smallRow257_small : List Nat := smallRow257_small_left ++ smallRow257_small_right
theorem smallRow257_small_checked : upperNeighbors PrimePairVertices.tree_small 257 = smallRow257_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 257 ++ upperNeighbors PrimePairVertices.tree_small_right 257 = _
  rw [smallRow257_small_left_checked, smallRow257_small_right_checked]
  rfl

def smallRow257 : List Nat := smallRow257_small
theorem smallRow257_checked : upperNeighbors PrimePairVertices.smallTree 257 = smallRow257 := smallRow257_small_checked

def smallRow263_block_000 : List Nat := [293, 443, 647, 761, 821, 941, 1049, 1091]
set_option maxRecDepth 2048 in
theorem smallRow263_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 263 = smallRow263_block_000 := by decide

def smallRow263_block_001 : List Nat := [2099, 2129, 2339, 2477, 2633, 2789, 2999, 3359, 3467, 3557, 3923]
set_option maxRecDepth 2048 in
theorem smallRow263_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 263 = smallRow263_block_001 := by decide

def smallRow263_block_002 : List Nat := [4523, 5189, 5693, 5987]
set_option maxRecDepth 2048 in
theorem smallRow263_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 263 = smallRow263_block_002 := by decide

def smallRow263_block_003 : List Nat := [6197, 6701, 7013, 7043, 7253, 7451, 7937, 8087]
set_option maxRecDepth 2048 in
theorem smallRow263_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 263 = smallRow263_block_003 := by decide

def smallRow263_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow263_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 263 = smallRow263_block_004 := by decide

def smallRow263_small_left : List Nat := smallRow263_block_000 ++ smallRow263_block_001
theorem smallRow263_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 263 = smallRow263_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 263 ++ upperNeighbors PrimePairVertices.tree_block_001 263 = _
  rw [smallRow263_block_000_checked, smallRow263_block_001_checked]
  rfl

def smallRow263_small_right_right : List Nat := smallRow263_block_003 ++ smallRow263_block_004
theorem smallRow263_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 263 = smallRow263_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 263 ++ upperNeighbors PrimePairVertices.tree_block_004 263 = _
  rw [smallRow263_block_003_checked, smallRow263_block_004_checked]
  rfl

def smallRow263_small_right : List Nat := smallRow263_block_002 ++ smallRow263_small_right_right
theorem smallRow263_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 263 = smallRow263_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 263 ++ upperNeighbors PrimePairVertices.tree_small_right_right 263 = _
  rw [smallRow263_block_002_checked, smallRow263_small_right_right_checked]
  rfl

def smallRow263_small : List Nat := smallRow263_small_left ++ smallRow263_small_right
theorem smallRow263_small_checked : upperNeighbors PrimePairVertices.tree_small 263 = smallRow263_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 263 ++ upperNeighbors PrimePairVertices.tree_small_right 263 = _
  rw [smallRow263_small_left_checked, smallRow263_small_right_checked]
  rfl

def smallRow263 : List Nat := smallRow263_small
theorem smallRow263_checked : upperNeighbors PrimePairVertices.smallTree 263 = smallRow263 := smallRow263_small_checked

def smallRow269_block_000 : List Nat := [317, 389, 431, 461, 617, 887, 1979, 2003]
set_option maxRecDepth 2048 in
theorem smallRow269_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 269 = smallRow269_block_000 := by decide

def smallRow269_block_001 : List Nat := [2741, 4001]
set_option maxRecDepth 2048 in
theorem smallRow269_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 269 = smallRow269_block_001 := by decide

def smallRow269_block_002 : List Nat := [4157, 4457, 4733, 5051, 5237, 5471, 6053]
set_option maxRecDepth 2048 in
theorem smallRow269_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 269 = smallRow269_block_002 := by decide

def smallRow269_block_003 : List Nat := [6329, 6521, 6563, 7109]
set_option maxRecDepth 2048 in
theorem smallRow269_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 269 = smallRow269_block_003 := by decide

def smallRow269_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow269_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 269 = smallRow269_block_004 := by decide

def smallRow269_small_left : List Nat := smallRow269_block_000 ++ smallRow269_block_001
theorem smallRow269_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 269 = smallRow269_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 269 ++ upperNeighbors PrimePairVertices.tree_block_001 269 = _
  rw [smallRow269_block_000_checked, smallRow269_block_001_checked]
  rfl

def smallRow269_small_right_right : List Nat := smallRow269_block_003 ++ smallRow269_block_004
theorem smallRow269_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 269 = smallRow269_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 269 ++ upperNeighbors PrimePairVertices.tree_block_004 269 = _
  rw [smallRow269_block_003_checked, smallRow269_block_004_checked]
  rfl

def smallRow269_small_right : List Nat := smallRow269_block_002 ++ smallRow269_small_right_right
theorem smallRow269_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 269 = smallRow269_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 269 ++ upperNeighbors PrimePairVertices.tree_small_right_right 269 = _
  rw [smallRow269_block_002_checked, smallRow269_small_right_right_checked]
  rfl

def smallRow269_small : List Nat := smallRow269_small_left ++ smallRow269_small_right
theorem smallRow269_small_checked : upperNeighbors PrimePairVertices.tree_small 269 = smallRow269_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 269 ++ upperNeighbors PrimePairVertices.tree_small_right 269 = _
  rw [smallRow269_small_left_checked, smallRow269_small_right_checked]
  rfl

def smallRow269 : List Nat := smallRow269_small
theorem smallRow269_checked : upperNeighbors PrimePairVertices.smallTree 269 = smallRow269 := smallRow269_small_checked

def smallRow271_block_000 : List Nat := [409, 1087, 1213, 1399, 1549, 1567, 1747, 1873]
set_option maxRecDepth 2048 in
theorem smallRow271_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 271 = smallRow271_block_000 := by decide

def smallRow271_block_001 : List Nat := [2089, 2131, 2179, 2971, 3457, 3541]
set_option maxRecDepth 2048 in
theorem smallRow271_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 271 = smallRow271_block_001 := by decide

def smallRow271_block_002 : List Nat := [4111, 5701, 5737]
set_option maxRecDepth 2048 in
theorem smallRow271_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 271 = smallRow271_block_002 := by decide

def smallRow271_block_003 : List Nat := [6163, 6451, 6733, 7177, 7549, 7639, 8101]
set_option maxRecDepth 2048 in
theorem smallRow271_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 271 = smallRow271_block_003 := by decide

def smallRow271_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow271_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 271 = smallRow271_block_004 := by decide

def smallRow271_small_left : List Nat := smallRow271_block_000 ++ smallRow271_block_001
theorem smallRow271_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 271 = smallRow271_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 271 ++ upperNeighbors PrimePairVertices.tree_block_001 271 = _
  rw [smallRow271_block_000_checked, smallRow271_block_001_checked]
  rfl

def smallRow271_small_right_right : List Nat := smallRow271_block_003 ++ smallRow271_block_004
theorem smallRow271_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 271 = smallRow271_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 271 ++ upperNeighbors PrimePairVertices.tree_block_004 271 = _
  rw [smallRow271_block_003_checked, smallRow271_block_004_checked]
  rfl

def smallRow271_small_right : List Nat := smallRow271_block_002 ++ smallRow271_small_right_right
theorem smallRow271_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 271 = smallRow271_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 271 ++ upperNeighbors PrimePairVertices.tree_small_right_right 271 = _
  rw [smallRow271_block_002_checked, smallRow271_small_right_right_checked]
  rfl

def smallRow271_small : List Nat := smallRow271_small_left ++ smallRow271_small_right
theorem smallRow271_small_checked : upperNeighbors PrimePairVertices.tree_small 271 = smallRow271_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 271 ++ upperNeighbors PrimePairVertices.tree_small_right 271 = _
  rw [smallRow271_small_left_checked, smallRow271_small_right_checked]
  rfl

def smallRow271 : List Nat := smallRow271_small
theorem smallRow271_checked : upperNeighbors PrimePairVertices.smallTree 271 = smallRow271 := smallRow271_small_checked

end ProjectEulerSolutions.PrimePairCache
