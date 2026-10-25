import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow149_block_000 : List Nat := [173, 251, 491, 563, 953, 971, 1571, 1583, 1979]
set_option maxRecDepth 2048 in
theorem smallRow149_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 149 = smallRow149_block_000 := by decide

def smallRow149_block_001 : List Nat := [2819, 3491, 3617]
set_option maxRecDepth 2048 in
theorem smallRow149_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 149 = smallRow149_block_001 := by decide

def smallRow149_block_002 : List Nat := [4133, 4289, 5009, 5279, 5717, 5867, 5939]
set_option maxRecDepth 2048 in
theorem smallRow149_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 149 = smallRow149_block_002 := by decide

def smallRow149_block_003 : List Nat := [6581, 6791, 6917, 7019, 7151, 7193, 7283, 8009]
set_option maxRecDepth 2048 in
theorem smallRow149_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 149 = smallRow149_block_003 := by decide

def smallRow149_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow149_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 149 = smallRow149_block_004 := by decide

def smallRow149_small_left : List Nat := smallRow149_block_000 ++ smallRow149_block_001
theorem smallRow149_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 149 = smallRow149_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 149 ++ upperNeighbors PrimePairVertices.tree_block_001 149 = _
  rw [smallRow149_block_000_checked, smallRow149_block_001_checked]
  rfl

def smallRow149_small_right_right : List Nat := smallRow149_block_003 ++ smallRow149_block_004
theorem smallRow149_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 149 = smallRow149_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 149 ++ upperNeighbors PrimePairVertices.tree_block_004 149 = _
  rw [smallRow149_block_003_checked, smallRow149_block_004_checked]
  rfl

def smallRow149_small_right : List Nat := smallRow149_block_002 ++ smallRow149_small_right_right
theorem smallRow149_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 149 = smallRow149_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 149 ++ upperNeighbors PrimePairVertices.tree_small_right_right 149 = _
  rw [smallRow149_block_002_checked, smallRow149_small_right_right_checked]
  rfl

def smallRow149_small : List Nat := smallRow149_small_left ++ smallRow149_small_right
theorem smallRow149_small_checked : upperNeighbors PrimePairVertices.tree_small 149 = smallRow149_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 149 ++ upperNeighbors PrimePairVertices.tree_small_right 149 = _
  rw [smallRow149_small_left_checked, smallRow149_small_right_checked]
  rfl

def smallRow149 : List Nat := smallRow149_small
theorem smallRow149_checked : upperNeighbors PrimePairVertices.smallTree 149 = smallRow149 := smallRow149_small_checked

def smallRow151_block_000 : List Nat := [163, 397, 433, 499, 607, 631, 769, 937, 1303]
set_option maxRecDepth 2048 in
theorem smallRow151_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 151 = smallRow151_block_000 := by decide

def smallRow151_block_001 : List Nat := [2281, 2311, 2683, 2689, 3019, 4027]
set_option maxRecDepth 2048 in
theorem smallRow151_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 151 = smallRow151_block_001 := by decide

def smallRow151_block_002 : List Nat := [4273, 4657, 4783, 4801, 5011, 5197, 5281, 5413, 6007]
set_option maxRecDepth 2048 in
theorem smallRow151_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 151 = smallRow151_block_002 := by decide

def smallRow151_block_003 : List Nat := [6217, 6607, 6733, 6967, 7933]
set_option maxRecDepth 2048 in
theorem smallRow151_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 151 = smallRow151_block_003 := by decide

def smallRow151_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow151_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 151 = smallRow151_block_004 := by decide

def smallRow151_small_left : List Nat := smallRow151_block_000 ++ smallRow151_block_001
theorem smallRow151_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 151 = smallRow151_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 151 ++ upperNeighbors PrimePairVertices.tree_block_001 151 = _
  rw [smallRow151_block_000_checked, smallRow151_block_001_checked]
  rfl

def smallRow151_small_right_right : List Nat := smallRow151_block_003 ++ smallRow151_block_004
theorem smallRow151_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 151 = smallRow151_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 151 ++ upperNeighbors PrimePairVertices.tree_block_004 151 = _
  rw [smallRow151_block_003_checked, smallRow151_block_004_checked]
  rfl

def smallRow151_small_right : List Nat := smallRow151_block_002 ++ smallRow151_small_right_right
theorem smallRow151_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 151 = smallRow151_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 151 ++ upperNeighbors PrimePairVertices.tree_small_right_right 151 = _
  rw [smallRow151_block_002_checked, smallRow151_small_right_right_checked]
  rfl

def smallRow151_small : List Nat := smallRow151_small_left ++ smallRow151_small_right
theorem smallRow151_small_checked : upperNeighbors PrimePairVertices.tree_small 151 = smallRow151_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 151 ++ upperNeighbors PrimePairVertices.tree_small_right 151 = _
  rw [smallRow151_small_left_checked, smallRow151_small_right_checked]
  rfl

def smallRow151 : List Nat := smallRow151_small
theorem smallRow151_checked : upperNeighbors PrimePairVertices.smallTree 151 = smallRow151 := smallRow151_small_checked

def smallRow157_block_000 : List Nat := [181, 229, 277, 571, 733, 1657, 1663]
set_option maxRecDepth 2048 in
theorem smallRow157_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 157 = smallRow157_block_000 := by decide

def smallRow157_block_001 : List Nat := [2239, 2377, 2539, 2647, 2713, 2749, 3079, 3109, 3217, 3613, 3709]
set_option maxRecDepth 2048 in
theorem smallRow157_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 157 = smallRow157_block_001 := by decide

def smallRow157_block_002 : List Nat := [4129, 5653, 6073]
set_option maxRecDepth 2048 in
theorem smallRow157_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 157 = smallRow157_block_002 := by decide

def smallRow157_block_003 : List Nat := [6661, 6907, 6997, 7309, 7561, 7699]
set_option maxRecDepth 2048 in
theorem smallRow157_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 157 = smallRow157_block_003 := by decide

def smallRow157_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow157_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 157 = smallRow157_block_004 := by decide

def smallRow157_small_left : List Nat := smallRow157_block_000 ++ smallRow157_block_001
theorem smallRow157_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 157 = smallRow157_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 157 ++ upperNeighbors PrimePairVertices.tree_block_001 157 = _
  rw [smallRow157_block_000_checked, smallRow157_block_001_checked]
  rfl

def smallRow157_small_right_right : List Nat := smallRow157_block_003 ++ smallRow157_block_004
theorem smallRow157_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 157 = smallRow157_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 157 ++ upperNeighbors PrimePairVertices.tree_block_004 157 = _
  rw [smallRow157_block_003_checked, smallRow157_block_004_checked]
  rfl

def smallRow157_small_right : List Nat := smallRow157_block_002 ++ smallRow157_small_right_right
theorem smallRow157_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 157 = smallRow157_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 157 ++ upperNeighbors PrimePairVertices.tree_small_right_right 157 = _
  rw [smallRow157_block_002_checked, smallRow157_small_right_right_checked]
  rfl

def smallRow157_small : List Nat := smallRow157_small_left ++ smallRow157_small_right
theorem smallRow157_small_checked : upperNeighbors PrimePairVertices.tree_small 157 = smallRow157_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 157 ++ upperNeighbors PrimePairVertices.tree_small_right 157 = _
  rw [smallRow157_small_left_checked, smallRow157_small_right_checked]
  rfl

def smallRow157 : List Nat := smallRow157_small
theorem smallRow157_checked : upperNeighbors PrimePairVertices.smallTree 157 = smallRow157 := smallRow157_small_checked

def smallRow163_block_000 : List Nat := [193, 307, 367, 409, 613, 811, 883, 997, 1579, 1741, 1783]
set_option maxRecDepth 2048 in
theorem smallRow163_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 163 = smallRow163_block_000 := by decide

def smallRow163_block_001 : List Nat := [2311, 2383, 2647, 3823]
set_option maxRecDepth 2048 in
theorem smallRow163_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 163 = smallRow163_block_001 := by decide

def smallRow163_block_002 : List Nat := [4177, 4441, 4597, 5563]
set_option maxRecDepth 2048 in
theorem smallRow163_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 163 = smallRow163_block_002 := by decide

def smallRow163_block_003 : List Nat := [6343, 6577, 6781, 6823, 6997, 7243, 7927]
set_option maxRecDepth 2048 in
theorem smallRow163_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 163 = smallRow163_block_003 := by decide

def smallRow163_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow163_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 163 = smallRow163_block_004 := by decide

def smallRow163_small_left : List Nat := smallRow163_block_000 ++ smallRow163_block_001
theorem smallRow163_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 163 = smallRow163_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 163 ++ upperNeighbors PrimePairVertices.tree_block_001 163 = _
  rw [smallRow163_block_000_checked, smallRow163_block_001_checked]
  rfl

def smallRow163_small_right_right : List Nat := smallRow163_block_003 ++ smallRow163_block_004
theorem smallRow163_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 163 = smallRow163_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 163 ++ upperNeighbors PrimePairVertices.tree_block_004 163 = _
  rw [smallRow163_block_003_checked, smallRow163_block_004_checked]
  rfl

def smallRow163_small_right : List Nat := smallRow163_block_002 ++ smallRow163_small_right_right
theorem smallRow163_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 163 = smallRow163_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 163 ++ upperNeighbors PrimePairVertices.tree_small_right_right 163 = _
  rw [smallRow163_block_002_checked, smallRow163_small_right_right_checked]
  rfl

def smallRow163_small : List Nat := smallRow163_small_left ++ smallRow163_small_right
theorem smallRow163_small_checked : upperNeighbors PrimePairVertices.tree_small 163 = smallRow163_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 163 ++ upperNeighbors PrimePairVertices.tree_small_right 163 = _
  rw [smallRow163_small_left_checked, smallRow163_small_right_checked]
  rfl

def smallRow163 : List Nat := smallRow163_small
theorem smallRow163_checked : upperNeighbors PrimePairVertices.smallTree 163 = smallRow163 := smallRow163_small_checked

def smallRow167_block_000 : List Nat := [269, 443, 491, 521, 641, 677, 911]
set_option maxRecDepth 2048 in
theorem smallRow167_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 167 = smallRow167_block_000 := by decide

def smallRow167_block_001 : List Nat := [2063, 2087, 2129, 2441, 2963, 2999, 3011, 3461, 3719]
set_option maxRecDepth 2048 in
theorem smallRow167_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 167 = smallRow167_block_001 := by decide

def smallRow167_block_002 : List Nat := [4271, 4733, 5441, 5981, 6053]
set_option maxRecDepth 2048 in
theorem smallRow167_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 167 = smallRow167_block_002 := by decide

def smallRow167_block_003 : List Nat := [6827, 6947, 7253, 7451, 7673, 7877, 8009]
set_option maxRecDepth 2048 in
theorem smallRow167_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 167 = smallRow167_block_003 := by decide

def smallRow167_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow167_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 167 = smallRow167_block_004 := by decide

def smallRow167_small_left : List Nat := smallRow167_block_000 ++ smallRow167_block_001
theorem smallRow167_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 167 = smallRow167_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 167 ++ upperNeighbors PrimePairVertices.tree_block_001 167 = _
  rw [smallRow167_block_000_checked, smallRow167_block_001_checked]
  rfl

def smallRow167_small_right_right : List Nat := smallRow167_block_003 ++ smallRow167_block_004
theorem smallRow167_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 167 = smallRow167_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 167 ++ upperNeighbors PrimePairVertices.tree_block_004 167 = _
  rw [smallRow167_block_003_checked, smallRow167_block_004_checked]
  rfl

def smallRow167_small_right : List Nat := smallRow167_block_002 ++ smallRow167_small_right_right
theorem smallRow167_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 167 = smallRow167_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 167 ++ upperNeighbors PrimePairVertices.tree_small_right_right 167 = _
  rw [smallRow167_block_002_checked, smallRow167_small_right_right_checked]
  rfl

def smallRow167_small : List Nat := smallRow167_small_left ++ smallRow167_small_right
theorem smallRow167_small_checked : upperNeighbors PrimePairVertices.tree_small 167 = smallRow167_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 167 ++ upperNeighbors PrimePairVertices.tree_small_right 167 = _
  rw [smallRow167_small_left_checked, smallRow167_small_right_checked]
  rfl

def smallRow167 : List Nat := smallRow167_small
theorem smallRow167_checked : upperNeighbors PrimePairVertices.smallTree 167 = smallRow167 := smallRow167_small_checked

def smallRow173_block_000 : List Nat := [191, 293, 347, 431, 659, 743, 1091, 2039]
set_option maxRecDepth 2048 in
theorem smallRow173_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 173 = smallRow173_block_000 := by decide

def smallRow173_block_001 : List Nat := [2333, 3041, 3929]
set_option maxRecDepth 2048 in
theorem smallRow173_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 173 = smallRow173_block_001 := by decide

def smallRow173_block_002 : List Nat := [4673, 4721, 4787, 4937, 5333, 5807, 5813]
set_option maxRecDepth 2048 in
theorem smallRow173_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 173 = smallRow173_block_002 := by decide

def smallRow173_block_003 : List Nat := [7331, 7529, 7559, 8039, 8117, 8171]
set_option maxRecDepth 2048 in
theorem smallRow173_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 173 = smallRow173_block_003 := by decide

def smallRow173_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow173_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 173 = smallRow173_block_004 := by decide

def smallRow173_small_left : List Nat := smallRow173_block_000 ++ smallRow173_block_001
theorem smallRow173_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 173 = smallRow173_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 173 ++ upperNeighbors PrimePairVertices.tree_block_001 173 = _
  rw [smallRow173_block_000_checked, smallRow173_block_001_checked]
  rfl

def smallRow173_small_right_right : List Nat := smallRow173_block_003 ++ smallRow173_block_004
theorem smallRow173_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 173 = smallRow173_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 173 ++ upperNeighbors PrimePairVertices.tree_block_004 173 = _
  rw [smallRow173_block_003_checked, smallRow173_block_004_checked]
  rfl

def smallRow173_small_right : List Nat := smallRow173_block_002 ++ smallRow173_small_right_right
theorem smallRow173_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 173 = smallRow173_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 173 ++ upperNeighbors PrimePairVertices.tree_small_right_right 173 = _
  rw [smallRow173_block_002_checked, smallRow173_small_right_right_checked]
  rfl

def smallRow173_small : List Nat := smallRow173_small_left ++ smallRow173_small_right
theorem smallRow173_small_checked : upperNeighbors PrimePairVertices.tree_small 173 = smallRow173_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 173 ++ upperNeighbors PrimePairVertices.tree_small_right 173 = _
  rw [smallRow173_small_left_checked, smallRow173_small_right_checked]
  rfl

def smallRow173 : List Nat := smallRow173_small
theorem smallRow173_checked : upperNeighbors PrimePairVertices.smallTree 173 = smallRow173 := smallRow173_small_checked

def smallRow179_block_000 : List Nat := [269, 317, 383, 593, 719, 743, 953, 1289, 1451, 1487, 1787, 1847, 2039]
set_option maxRecDepth 2048 in
theorem smallRow179_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 179 = smallRow179_block_000 := by decide

def smallRow179_block_001 : List Nat := [2309, 2339, 2381, 2579, 3137, 3719]
set_option maxRecDepth 2048 in
theorem smallRow179_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 179 = smallRow179_block_001 := by decide

def smallRow179_block_002 : List Nat := [4349, 4517, 4649, 4787, 4817, 4973, 5153, 5279, 5867]
set_option maxRecDepth 2048 in
theorem smallRow179_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 179 = smallRow179_block_002 := by decide

def smallRow179_block_003 : List Nat := [7331, 7541, 7547, 7673, 7877, 8009]
set_option maxRecDepth 2048 in
theorem smallRow179_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 179 = smallRow179_block_003 := by decide

def smallRow179_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow179_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 179 = smallRow179_block_004 := by decide

def smallRow179_small_left : List Nat := smallRow179_block_000 ++ smallRow179_block_001
theorem smallRow179_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 179 = smallRow179_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 179 ++ upperNeighbors PrimePairVertices.tree_block_001 179 = _
  rw [smallRow179_block_000_checked, smallRow179_block_001_checked]
  rfl

def smallRow179_small_right_right : List Nat := smallRow179_block_003 ++ smallRow179_block_004
theorem smallRow179_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 179 = smallRow179_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 179 ++ upperNeighbors PrimePairVertices.tree_block_004 179 = _
  rw [smallRow179_block_003_checked, smallRow179_block_004_checked]
  rfl

def smallRow179_small_right : List Nat := smallRow179_block_002 ++ smallRow179_small_right_right
theorem smallRow179_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 179 = smallRow179_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 179 ++ upperNeighbors PrimePairVertices.tree_small_right_right 179 = _
  rw [smallRow179_block_002_checked, smallRow179_small_right_right_checked]
  rfl

def smallRow179_small : List Nat := smallRow179_small_left ++ smallRow179_small_right
theorem smallRow179_small_checked : upperNeighbors PrimePairVertices.tree_small 179 = smallRow179_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 179 ++ upperNeighbors PrimePairVertices.tree_small_right 179 = _
  rw [smallRow179_small_left_checked, smallRow179_small_right_checked]
  rfl

def smallRow179 : List Nat := smallRow179_small
theorem smallRow179_checked : upperNeighbors PrimePairVertices.smallTree 179 = smallRow179 := smallRow179_small_checked

def smallRow181_block_000 : List Nat := [193, 199, 283, 397, 421, 499, 607, 619, 751, 757, 787, 1657, 1759]
set_option maxRecDepth 2048 in
theorem smallRow181_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 181 = smallRow181_block_000 := by decide

def smallRow181_block_001 : List Nat := [2269, 2311, 2683, 2917, 3391, 3673, 4051]
set_option maxRecDepth 2048 in
theorem smallRow181_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 181 = smallRow181_block_001 := by decide

def smallRow181_block_002 : List Nat := [5101, 5323, 5839, 6007]
set_option maxRecDepth 2048 in
theorem smallRow181_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 181 = smallRow181_block_002 := by decide

def smallRow181_block_003 : List Nat := [6337, 7549, 7687]
set_option maxRecDepth 2048 in
theorem smallRow181_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 181 = smallRow181_block_003 := by decide

def smallRow181_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow181_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 181 = smallRow181_block_004 := by decide

def smallRow181_small_left : List Nat := smallRow181_block_000 ++ smallRow181_block_001
theorem smallRow181_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 181 = smallRow181_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 181 ++ upperNeighbors PrimePairVertices.tree_block_001 181 = _
  rw [smallRow181_block_000_checked, smallRow181_block_001_checked]
  rfl

def smallRow181_small_right_right : List Nat := smallRow181_block_003 ++ smallRow181_block_004
theorem smallRow181_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 181 = smallRow181_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 181 ++ upperNeighbors PrimePairVertices.tree_block_004 181 = _
  rw [smallRow181_block_003_checked, smallRow181_block_004_checked]
  rfl

def smallRow181_small_right : List Nat := smallRow181_block_002 ++ smallRow181_small_right_right
theorem smallRow181_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 181 = smallRow181_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 181 ++ upperNeighbors PrimePairVertices.tree_small_right_right 181 = _
  rw [smallRow181_block_002_checked, smallRow181_small_right_right_checked]
  rfl

def smallRow181_small : List Nat := smallRow181_small_left ++ smallRow181_small_right
theorem smallRow181_small_checked : upperNeighbors PrimePairVertices.tree_small 181 = smallRow181_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 181 ++ upperNeighbors PrimePairVertices.tree_small_right 181 = _
  rw [smallRow181_small_left_checked, smallRow181_small_right_checked]
  rfl

def smallRow181 : List Nat := smallRow181_small
theorem smallRow181_checked : upperNeighbors PrimePairVertices.smallTree 181 = smallRow181 := smallRow181_small_checked

end ProjectEulerSolutions.PrimePairCache
