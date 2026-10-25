import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow191_block_000 : List Nat := [227, 251, 281, 461, 599, 953, 977, 1103, 1319, 1373, 1583, 1697]
set_option maxRecDepth 2048 in
theorem smallRow191_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 191 = smallRow191_block_000 := by decide

def smallRow191_block_001 : List Nat := [2069, 2213, 2423, 2903, 3251, 3581]
set_option maxRecDepth 2048 in
theorem smallRow191_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 191 = smallRow191_block_001 := by decide

def smallRow191_block_002 : List Nat := [4127, 4259, 4673, 4691, 5099, 5153, 5813, 5843, 5981]
set_option maxRecDepth 2048 in
theorem smallRow191_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 191 = smallRow191_block_002 := by decide

def smallRow191_block_003 : List Nat := [6287, 6311, 6599, 6917]
set_option maxRecDepth 2048 in
theorem smallRow191_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 191 = smallRow191_block_003 := by decide

def smallRow191_block_004 : List Nat := [8237]
set_option maxRecDepth 2048 in
theorem smallRow191_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 191 = smallRow191_block_004 := by decide

def smallRow191_small_left : List Nat := smallRow191_block_000 ++ smallRow191_block_001
theorem smallRow191_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 191 = smallRow191_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 191 ++ upperNeighbors PrimePairVertices.tree_block_001 191 = _
  rw [smallRow191_block_000_checked, smallRow191_block_001_checked]
  rfl

def smallRow191_small_right_right : List Nat := smallRow191_block_003 ++ smallRow191_block_004
theorem smallRow191_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 191 = smallRow191_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 191 ++ upperNeighbors PrimePairVertices.tree_block_004 191 = _
  rw [smallRow191_block_003_checked, smallRow191_block_004_checked]
  rfl

def smallRow191_small_right : List Nat := smallRow191_block_002 ++ smallRow191_small_right_right
theorem smallRow191_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 191 = smallRow191_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 191 ++ upperNeighbors PrimePairVertices.tree_small_right_right 191 = _
  rw [smallRow191_block_002_checked, smallRow191_small_right_right_checked]
  rfl

def smallRow191_small : List Nat := smallRow191_small_left ++ smallRow191_small_right
theorem smallRow191_small_checked : upperNeighbors PrimePairVertices.tree_small 191 = smallRow191_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 191 ++ upperNeighbors PrimePairVertices.tree_small_right 191 = _
  rw [smallRow191_small_left_checked, smallRow191_small_right_checked]
  rfl

def smallRow191 : List Nat := smallRow191_small
theorem smallRow191_checked : upperNeighbors PrimePairVertices.smallTree 191 = smallRow191 := smallRow191_small_checked

def smallRow193_block_000 : List Nat := [283, 373, 433, 541, 577, 601, 751, 811, 883, 1009, 1291, 1297, 1399, 1627, 1669, 1723, 2011]
set_option maxRecDepth 2048 in
theorem smallRow193_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 193 = smallRow193_block_000 := by decide

def smallRow193_block_001 : List Nat := [2131, 2719, 2803, 3037, 3457, 3499, 3727]
set_option maxRecDepth 2048 in
theorem smallRow193_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 193 = smallRow193_block_001 := by decide

def smallRow193_block_002 : List Nat := [4657, 4969, 5827]
set_option maxRecDepth 2048 in
theorem smallRow193_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 193 = smallRow193_block_002 := by decide

def smallRow193_block_003 : List Nat := [6427, 6733, 7237, 8161, 8191]
set_option maxRecDepth 2048 in
theorem smallRow193_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 193 = smallRow193_block_003 := by decide

def smallRow193_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow193_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 193 = smallRow193_block_004 := by decide

def smallRow193_small_left : List Nat := smallRow193_block_000 ++ smallRow193_block_001
theorem smallRow193_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 193 = smallRow193_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 193 ++ upperNeighbors PrimePairVertices.tree_block_001 193 = _
  rw [smallRow193_block_000_checked, smallRow193_block_001_checked]
  rfl

def smallRow193_small_right_right : List Nat := smallRow193_block_003 ++ smallRow193_block_004
theorem smallRow193_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 193 = smallRow193_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 193 ++ upperNeighbors PrimePairVertices.tree_block_004 193 = _
  rw [smallRow193_block_003_checked, smallRow193_block_004_checked]
  rfl

def smallRow193_small_right : List Nat := smallRow193_block_002 ++ smallRow193_small_right_right
theorem smallRow193_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 193 = smallRow193_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 193 ++ upperNeighbors PrimePairVertices.tree_small_right_right 193 = _
  rw [smallRow193_block_002_checked, smallRow193_small_right_right_checked]
  rfl

def smallRow193_small : List Nat := smallRow193_small_left ++ smallRow193_small_right
theorem smallRow193_small_checked : upperNeighbors PrimePairVertices.tree_small 193 = smallRow193_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 193 ++ upperNeighbors PrimePairVertices.tree_small_right 193 = _
  rw [smallRow193_small_left_checked, smallRow193_small_right_checked]
  rfl

def smallRow193 : List Nat := smallRow193_small
theorem smallRow193_checked : upperNeighbors PrimePairVertices.smallTree 193 = smallRow193 := smallRow193_small_checked

def smallRow197_block_000 : List Nat := [311, 347, 569, 641, 947, 971, 1181, 1481, 1637]
set_option maxRecDepth 2048 in
theorem smallRow197_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 197 = smallRow197_block_000 := by decide

def smallRow197_block_001 : List Nat := [2099, 2207, 2657, 4073]
set_option maxRecDepth 2048 in
theorem smallRow197_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 197 = smallRow197_block_001 := by decide

def smallRow197_block_002 : List Nat := [4457, 4493, 4547, 4919, 4937, 4967, 5021, 5387, 5399, 6113]
set_option maxRecDepth 2048 in
theorem smallRow197_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 197 = smallRow197_block_002 := by decide

def smallRow197_block_003 : List Nat := [6173, 6761, 7187, 7817]
set_option maxRecDepth 2048 in
theorem smallRow197_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 197 = smallRow197_block_003 := by decide

def smallRow197_block_004 : List Nat := [8297, 8363]
set_option maxRecDepth 2048 in
theorem smallRow197_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 197 = smallRow197_block_004 := by decide

def smallRow197_small_left : List Nat := smallRow197_block_000 ++ smallRow197_block_001
theorem smallRow197_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 197 = smallRow197_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 197 ++ upperNeighbors PrimePairVertices.tree_block_001 197 = _
  rw [smallRow197_block_000_checked, smallRow197_block_001_checked]
  rfl

def smallRow197_small_right_right : List Nat := smallRow197_block_003 ++ smallRow197_block_004
theorem smallRow197_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 197 = smallRow197_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 197 ++ upperNeighbors PrimePairVertices.tree_block_004 197 = _
  rw [smallRow197_block_003_checked, smallRow197_block_004_checked]
  rfl

def smallRow197_small_right : List Nat := smallRow197_block_002 ++ smallRow197_small_right_right
theorem smallRow197_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 197 = smallRow197_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 197 ++ upperNeighbors PrimePairVertices.tree_small_right_right 197 = _
  rw [smallRow197_block_002_checked, smallRow197_small_right_right_checked]
  rfl

def smallRow197_small : List Nat := smallRow197_small_left ++ smallRow197_small_right
theorem smallRow197_small_checked : upperNeighbors PrimePairVertices.tree_small 197 = smallRow197_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 197 ++ upperNeighbors PrimePairVertices.tree_small_right 197 = _
  rw [smallRow197_small_left_checked, smallRow197_small_right_checked]
  rfl

def smallRow197 : List Nat := smallRow197_small
theorem smallRow197_checked : upperNeighbors PrimePairVertices.smallTree 197 = smallRow197 := smallRow197_small_checked

def smallRow199_block_000 : List Nat := [211, 373, 379, 673, 739, 751, 811, 877, 1153, 1489, 1609, 1933]
set_option maxRecDepth 2048 in
theorem smallRow199_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 199 = smallRow199_block_000 := by decide

def smallRow199_block_001 : List Nat := [2251, 2437, 2917, 2953, 3067, 3109, 3259, 3529]
set_option maxRecDepth 2048 in
theorem smallRow199_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 199 = smallRow199_block_001 := by decide

def smallRow199_block_002 : List Nat := [4327, 5011, 5023, 5107]
set_option maxRecDepth 2048 in
theorem smallRow199_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 199 = smallRow199_block_002 := by decide

def smallRow199_block_003 : List Nat := [6301, 6949]
set_option maxRecDepth 2048 in
theorem smallRow199_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 199 = smallRow199_block_003 := by decide

def smallRow199_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow199_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 199 = smallRow199_block_004 := by decide

def smallRow199_small_left : List Nat := smallRow199_block_000 ++ smallRow199_block_001
theorem smallRow199_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 199 = smallRow199_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 199 ++ upperNeighbors PrimePairVertices.tree_block_001 199 = _
  rw [smallRow199_block_000_checked, smallRow199_block_001_checked]
  rfl

def smallRow199_small_right_right : List Nat := smallRow199_block_003 ++ smallRow199_block_004
theorem smallRow199_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 199 = smallRow199_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 199 ++ upperNeighbors PrimePairVertices.tree_block_004 199 = _
  rw [smallRow199_block_003_checked, smallRow199_block_004_checked]
  rfl

def smallRow199_small_right : List Nat := smallRow199_block_002 ++ smallRow199_small_right_right
theorem smallRow199_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 199 = smallRow199_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 199 ++ upperNeighbors PrimePairVertices.tree_small_right_right 199 = _
  rw [smallRow199_block_002_checked, smallRow199_small_right_right_checked]
  rfl

def smallRow199_small : List Nat := smallRow199_small_left ++ smallRow199_small_right
theorem smallRow199_small_checked : upperNeighbors PrimePairVertices.tree_small 199 = smallRow199_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 199 ++ upperNeighbors PrimePairVertices.tree_small_right 199 = _
  rw [smallRow199_small_left_checked, smallRow199_small_right_checked]
  rfl

def smallRow199 : List Nat := smallRow199_small
theorem smallRow199_checked : upperNeighbors PrimePairVertices.smallTree 199 = smallRow199 := smallRow199_small_checked

def smallRow211_block_000 : List Nat := [271, 283, 313, 349, 373, 499, 571, 727, 1129, 1783, 1873]
set_option maxRecDepth 2048 in
theorem smallRow211_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 211 = smallRow211_block_000 := by decide

def smallRow211_block_001 : List Nat := [2053, 2683, 2833, 3733, 4027]
set_option maxRecDepth 2048 in
theorem smallRow211_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 211 = smallRow211_block_001 := by decide

def smallRow211_block_002 : List Nat := [4507, 4549, 5059, 5101, 5653, 5791]
set_option maxRecDepth 2048 in
theorem smallRow211_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 211 = smallRow211_block_002 := by decide

def smallRow211_block_003 : List Nat := [6199, 6397, 7039]
set_option maxRecDepth 2048 in
theorem smallRow211_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 211 = smallRow211_block_003 := by decide

def smallRow211_block_004 : List Nat := [8209, 8269]
set_option maxRecDepth 2048 in
theorem smallRow211_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 211 = smallRow211_block_004 := by decide

def smallRow211_small_left : List Nat := smallRow211_block_000 ++ smallRow211_block_001
theorem smallRow211_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 211 = smallRow211_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 211 ++ upperNeighbors PrimePairVertices.tree_block_001 211 = _
  rw [smallRow211_block_000_checked, smallRow211_block_001_checked]
  rfl

def smallRow211_small_right_right : List Nat := smallRow211_block_003 ++ smallRow211_block_004
theorem smallRow211_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 211 = smallRow211_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 211 ++ upperNeighbors PrimePairVertices.tree_block_004 211 = _
  rw [smallRow211_block_003_checked, smallRow211_block_004_checked]
  rfl

def smallRow211_small_right : List Nat := smallRow211_block_002 ++ smallRow211_small_right_right
theorem smallRow211_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 211 = smallRow211_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 211 ++ upperNeighbors PrimePairVertices.tree_small_right_right 211 = _
  rw [smallRow211_block_002_checked, smallRow211_small_right_right_checked]
  rfl

def smallRow211_small : List Nat := smallRow211_small_left ++ smallRow211_small_right
theorem smallRow211_small_checked : upperNeighbors PrimePairVertices.tree_small 211 = smallRow211_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 211 ++ upperNeighbors PrimePairVertices.tree_small_right 211 = _
  rw [smallRow211_small_left_checked, smallRow211_small_right_checked]
  rfl

def smallRow211 : List Nat := smallRow211_small
theorem smallRow211_checked : upperNeighbors PrimePairVertices.smallTree 211 = smallRow211 := smallRow211_small_checked

def smallRow223_block_000 : List Nat := [229, 277, 337, 547, 829, 919, 1093, 1447]
set_option maxRecDepth 2048 in
theorem smallRow223_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 223 = smallRow223_block_000 := by decide

def smallRow223_block_001 : List Nat := [2179, 2437, 2551, 3547]
set_option maxRecDepth 2048 in
theorem smallRow223_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 223 = smallRow223_block_001 := by decide

def smallRow223_block_002 : List Nat := [4159, 4261, 5647, 6007, 6121, 6133]
set_option maxRecDepth 2048 in
theorem smallRow223_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 223 = smallRow223_block_002 := by decide

def smallRow223_block_003 : List Nat := [6373, 6781, 6907, 7159, 7309, 7537, 7927]
set_option maxRecDepth 2048 in
theorem smallRow223_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 223 = smallRow223_block_003 := by decide

def smallRow223_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow223_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 223 = smallRow223_block_004 := by decide

def smallRow223_small_left : List Nat := smallRow223_block_000 ++ smallRow223_block_001
theorem smallRow223_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 223 = smallRow223_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 223 ++ upperNeighbors PrimePairVertices.tree_block_001 223 = _
  rw [smallRow223_block_000_checked, smallRow223_block_001_checked]
  rfl

def smallRow223_small_right_right : List Nat := smallRow223_block_003 ++ smallRow223_block_004
theorem smallRow223_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 223 = smallRow223_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 223 ++ upperNeighbors PrimePairVertices.tree_block_004 223 = _
  rw [smallRow223_block_003_checked, smallRow223_block_004_checked]
  rfl

def smallRow223_small_right : List Nat := smallRow223_block_002 ++ smallRow223_small_right_right
theorem smallRow223_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 223 = smallRow223_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 223 ++ upperNeighbors PrimePairVertices.tree_small_right_right 223 = _
  rw [smallRow223_block_002_checked, smallRow223_small_right_right_checked]
  rfl

def smallRow223_small : List Nat := smallRow223_small_left ++ smallRow223_small_right
theorem smallRow223_small_checked : upperNeighbors PrimePairVertices.tree_small 223 = smallRow223_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 223 ++ upperNeighbors PrimePairVertices.tree_small_right 223 = _
  rw [smallRow223_small_left_checked, smallRow223_small_right_checked]
  rfl

def smallRow223 : List Nat := smallRow223_small
theorem smallRow223_checked : upperNeighbors PrimePairVertices.smallTree 223 = smallRow223 := smallRow223_small_checked

def smallRow227_block_000 : List Nat := [281, 593, 719, 827, 1229, 1427, 1499, 1721]
set_option maxRecDepth 2048 in
theorem smallRow227_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 227 = smallRow227_block_000 := by decide

def smallRow227_block_001 : List Nat := [2129, 2729, 2903, 2939, 3701, 3911]
set_option maxRecDepth 2048 in
theorem smallRow227_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 227 = smallRow227_block_001 := by decide

def smallRow227_block_002 : List Nat := [4271, 4421, 4691, 4793, 5333, 5591]
set_option maxRecDepth 2048 in
theorem smallRow227_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 227 = smallRow227_block_002 := by decide

def smallRow227_block_003 : List Nat := [7823, 8093]
set_option maxRecDepth 2048 in
theorem smallRow227_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 227 = smallRow227_block_003 := by decide

def smallRow227_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow227_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 227 = smallRow227_block_004 := by decide

def smallRow227_small_left : List Nat := smallRow227_block_000 ++ smallRow227_block_001
theorem smallRow227_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 227 = smallRow227_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 227 ++ upperNeighbors PrimePairVertices.tree_block_001 227 = _
  rw [smallRow227_block_000_checked, smallRow227_block_001_checked]
  rfl

def smallRow227_small_right_right : List Nat := smallRow227_block_003 ++ smallRow227_block_004
theorem smallRow227_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 227 = smallRow227_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 227 ++ upperNeighbors PrimePairVertices.tree_block_004 227 = _
  rw [smallRow227_block_003_checked, smallRow227_block_004_checked]
  rfl

def smallRow227_small_right : List Nat := smallRow227_block_002 ++ smallRow227_small_right_right
theorem smallRow227_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 227 = smallRow227_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 227 ++ upperNeighbors PrimePairVertices.tree_small_right_right 227 = _
  rw [smallRow227_block_002_checked, smallRow227_small_right_right_checked]
  rfl

def smallRow227_small : List Nat := smallRow227_small_left ++ smallRow227_small_right
theorem smallRow227_small_checked : upperNeighbors PrimePairVertices.tree_small 227 = smallRow227_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 227 ++ upperNeighbors PrimePairVertices.tree_small_right 227 = _
  rw [smallRow227_small_left_checked, smallRow227_small_right_checked]
  rfl

def smallRow227 : List Nat := smallRow227_small
theorem smallRow227_checked : upperNeighbors PrimePairVertices.smallTree 227 = smallRow227 := smallRow227_small_checked

def smallRow229_block_000 : List Nat := [373, 433, 499, 547, 613, 631, 937, 1231, 1801]
set_option maxRecDepth 2048 in
theorem smallRow229_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 229 = smallRow229_block_000 := by decide

def smallRow229_block_001 : List Nat := [2083, 2221, 2239, 2347, 2767, 3121, 3301, 3391, 3541, 3559, 3919, 4021]
set_option maxRecDepth 2048 in
theorem smallRow229_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 229 = smallRow229_block_001 := by decide

def smallRow229_block_002 : List Nat := [5077, 5569, 5659, 5869]
set_option maxRecDepth 2048 in
theorem smallRow229_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 229 = smallRow229_block_002 := by decide

def smallRow229_block_003 : List Nat := [6247, 6733, 7213, 7591, 7759]
set_option maxRecDepth 2048 in
theorem smallRow229_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 229 = smallRow229_block_003 := by decide

def smallRow229_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow229_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 229 = smallRow229_block_004 := by decide

def smallRow229_small_left : List Nat := smallRow229_block_000 ++ smallRow229_block_001
theorem smallRow229_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 229 = smallRow229_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 229 ++ upperNeighbors PrimePairVertices.tree_block_001 229 = _
  rw [smallRow229_block_000_checked, smallRow229_block_001_checked]
  rfl

def smallRow229_small_right_right : List Nat := smallRow229_block_003 ++ smallRow229_block_004
theorem smallRow229_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 229 = smallRow229_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 229 ++ upperNeighbors PrimePairVertices.tree_block_004 229 = _
  rw [smallRow229_block_003_checked, smallRow229_block_004_checked]
  rfl

def smallRow229_small_right : List Nat := smallRow229_block_002 ++ smallRow229_small_right_right
theorem smallRow229_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 229 = smallRow229_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 229 ++ upperNeighbors PrimePairVertices.tree_small_right_right 229 = _
  rw [smallRow229_block_002_checked, smallRow229_small_right_right_checked]
  rfl

def smallRow229_small : List Nat := smallRow229_small_left ++ smallRow229_small_right
theorem smallRow229_small_checked : upperNeighbors PrimePairVertices.tree_small 229 = smallRow229_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 229 ++ upperNeighbors PrimePairVertices.tree_small_right 229 = _
  rw [smallRow229_small_left_checked, smallRow229_small_right_checked]
  rfl

def smallRow229 : List Nat := smallRow229_small
theorem smallRow229_checked : upperNeighbors PrimePairVertices.smallTree 229 = smallRow229 := smallRow229_small_checked

end ProjectEulerSolutions.PrimePairCache
