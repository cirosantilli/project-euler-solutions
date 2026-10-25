import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow379_block_000 : List Nat := [397, 751, 811, 877, 997, 1063, 1069, 1087, 1213, 1231, 1321, 1423, 1741, 1873]
set_option maxRecDepth 2048 in
theorem smallRow379_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 379 = smallRow379_block_000 := by decide

def smallRow379_block_001 : List Nat := [2221, 2281, 2791, 3061, 3079, 3121, 3259, 3931, 4051]
set_option maxRecDepth 2048 in
theorem smallRow379_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 379 = smallRow379_block_001 := by decide

def smallRow379_block_002 : List Nat := [4423, 4951, 5563, 5569, 5581]
set_option maxRecDepth 2048 in
theorem smallRow379_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 379 = smallRow379_block_002 := by decide

def smallRow379_block_003 : List Nat := [6367, 6379, 6673, 6733, 6997, 7699, 7723, 7873, 7879, 7933]
set_option maxRecDepth 2048 in
theorem smallRow379_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 379 = smallRow379_block_003 := by decide

def smallRow379_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow379_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 379 = smallRow379_block_004 := by decide

def smallRow379_small_left : List Nat := smallRow379_block_000 ++ smallRow379_block_001
theorem smallRow379_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 379 = smallRow379_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 379 ++ upperNeighbors PrimePairVertices.tree_block_001 379 = _
  rw [smallRow379_block_000_checked, smallRow379_block_001_checked]
  rfl

def smallRow379_small_right_right : List Nat := smallRow379_block_003 ++ smallRow379_block_004
theorem smallRow379_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 379 = smallRow379_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 379 ++ upperNeighbors PrimePairVertices.tree_block_004 379 = _
  rw [smallRow379_block_003_checked, smallRow379_block_004_checked]
  rfl

def smallRow379_small_right : List Nat := smallRow379_block_002 ++ smallRow379_small_right_right
theorem smallRow379_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 379 = smallRow379_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 379 ++ upperNeighbors PrimePairVertices.tree_small_right_right 379 = _
  rw [smallRow379_block_002_checked, smallRow379_small_right_right_checked]
  rfl

def smallRow379_small : List Nat := smallRow379_small_left ++ smallRow379_small_right
theorem smallRow379_small_checked : upperNeighbors PrimePairVertices.tree_small 379 = smallRow379_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 379 ++ upperNeighbors PrimePairVertices.tree_small_right 379 = _
  rw [smallRow379_small_left_checked, smallRow379_small_right_checked]
  rfl

def smallRow379 : List Nat := smallRow379_small
theorem smallRow379_checked : upperNeighbors PrimePairVertices.smallTree 379 = smallRow379 := smallRow379_small_checked

def smallRow383_block_000 : List Nat := [419, 797, 821, 941, 1427, 1493, 1721, 1733, 1871, 1889]
set_option maxRecDepth 2048 in
theorem smallRow383_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 383 = smallRow383_block_000 := by decide

def smallRow383_block_001 : List Nat := [2267, 2687, 2729, 3257, 3359, 3527, 3989, 4091]
set_option maxRecDepth 2048 in
theorem smallRow383_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 383 = smallRow383_block_001 := by decide

def smallRow383_block_002 : List Nat := [4139, 4547, 5303]
set_option maxRecDepth 2048 in
theorem smallRow383_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 383 = smallRow383_block_002 := by decide

def smallRow383_block_003 : List Nat := [6317, 6359, 7331, 7703, 7853, 7877, 8117, 8123]
set_option maxRecDepth 2048 in
theorem smallRow383_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 383 = smallRow383_block_003 := by decide

def smallRow383_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow383_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 383 = smallRow383_block_004 := by decide

def smallRow383_small_left : List Nat := smallRow383_block_000 ++ smallRow383_block_001
theorem smallRow383_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 383 = smallRow383_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 383 ++ upperNeighbors PrimePairVertices.tree_block_001 383 = _
  rw [smallRow383_block_000_checked, smallRow383_block_001_checked]
  rfl

def smallRow383_small_right_right : List Nat := smallRow383_block_003 ++ smallRow383_block_004
theorem smallRow383_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 383 = smallRow383_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 383 ++ upperNeighbors PrimePairVertices.tree_block_004 383 = _
  rw [smallRow383_block_003_checked, smallRow383_block_004_checked]
  rfl

def smallRow383_small_right : List Nat := smallRow383_block_002 ++ smallRow383_small_right_right
theorem smallRow383_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 383 = smallRow383_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 383 ++ upperNeighbors PrimePairVertices.tree_small_right_right 383 = _
  rw [smallRow383_block_002_checked, smallRow383_small_right_right_checked]
  rfl

def smallRow383_small : List Nat := smallRow383_small_left ++ smallRow383_small_right
theorem smallRow383_small_checked : upperNeighbors PrimePairVertices.tree_small 383 = smallRow383_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 383 ++ upperNeighbors PrimePairVertices.tree_small_right 383 = _
  rw [smallRow383_small_left_checked, smallRow383_small_right_checked]
  rfl

def smallRow383 : List Nat := smallRow383_small
theorem smallRow383_checked : upperNeighbors PrimePairVertices.smallTree 383 = smallRow383 := smallRow383_small_checked

def smallRow389_block_000 : List Nat := [761, 797, 947, 971, 1259, 1493, 1913]
set_option maxRecDepth 2048 in
theorem smallRow389_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 389 = smallRow389_block_000 := by decide

def smallRow389_block_001 : List Nat := [2087, 2309, 2333, 2393, 2459, 2969, 3137, 3191, 3299, 3719, 4049]
set_option maxRecDepth 2048 in
theorem smallRow389_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 389 = smallRow389_block_001 := by decide

def smallRow389_block_002 : List Nat := [4409, 4637, 4703, 4733, 5231, 5351, 6029, 6089]
set_option maxRecDepth 2048 in
theorem smallRow389_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 389 = smallRow389_block_002 := by decide

def smallRow389_block_003 : List Nat := [6257, 6947, 7499, 7703, 7727, 7853, 7937, 8087, 8123]
set_option maxRecDepth 2048 in
theorem smallRow389_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 389 = smallRow389_block_003 := by decide

def smallRow389_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow389_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 389 = smallRow389_block_004 := by decide

def smallRow389_small_left : List Nat := smallRow389_block_000 ++ smallRow389_block_001
theorem smallRow389_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 389 = smallRow389_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 389 ++ upperNeighbors PrimePairVertices.tree_block_001 389 = _
  rw [smallRow389_block_000_checked, smallRow389_block_001_checked]
  rfl

def smallRow389_small_right_right : List Nat := smallRow389_block_003 ++ smallRow389_block_004
theorem smallRow389_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 389 = smallRow389_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 389 ++ upperNeighbors PrimePairVertices.tree_block_004 389 = _
  rw [smallRow389_block_003_checked, smallRow389_block_004_checked]
  rfl

def smallRow389_small_right : List Nat := smallRow389_block_002 ++ smallRow389_small_right_right
theorem smallRow389_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 389 = smallRow389_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 389 ++ upperNeighbors PrimePairVertices.tree_small_right_right 389 = _
  rw [smallRow389_block_002_checked, smallRow389_small_right_right_checked]
  rfl

def smallRow389_small : List Nat := smallRow389_small_left ++ smallRow389_small_right
theorem smallRow389_small_checked : upperNeighbors PrimePairVertices.tree_small 389 = smallRow389_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 389 ++ upperNeighbors PrimePairVertices.tree_small_right 389 = _
  rw [smallRow389_small_left_checked, smallRow389_small_right_checked]
  rfl

def smallRow389 : List Nat := smallRow389_small
theorem smallRow389_checked : upperNeighbors PrimePairVertices.smallTree 389 = smallRow389 := smallRow389_small_checked

def smallRow397_block_000 : List Nat := [547, 673, 907, 1063, 1483, 1621, 1861]
set_option maxRecDepth 2048 in
theorem smallRow397_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 397 = smallRow397_block_000 := by decide

def smallRow397_block_001 : List Nat := [2251, 2593, 2851, 3511, 3727, 3847, 3877, 3919, 3943, 4057]
set_option maxRecDepth 2048 in
theorem smallRow397_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 397 = smallRow397_block_001 := by decide

def smallRow397_block_002 : List Nat := [4441, 4483, 4567, 4657, 4759, 4987, 5197, 5749, 5791, 6121]
set_option maxRecDepth 2048 in
theorem smallRow397_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 397 = smallRow397_block_002 := by decide

def smallRow397_block_003 : List Nat := [6397, 6703, 6829, 7177, 7639, 7741, 7927, 7933, 8101]
set_option maxRecDepth 2048 in
theorem smallRow397_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 397 = smallRow397_block_003 := by decide

def smallRow397_block_004 : List Nat := [8329]
set_option maxRecDepth 2048 in
theorem smallRow397_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 397 = smallRow397_block_004 := by decide

def smallRow397_small_left : List Nat := smallRow397_block_000 ++ smallRow397_block_001
theorem smallRow397_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 397 = smallRow397_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 397 ++ upperNeighbors PrimePairVertices.tree_block_001 397 = _
  rw [smallRow397_block_000_checked, smallRow397_block_001_checked]
  rfl

def smallRow397_small_right_right : List Nat := smallRow397_block_003 ++ smallRow397_block_004
theorem smallRow397_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 397 = smallRow397_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 397 ++ upperNeighbors PrimePairVertices.tree_block_004 397 = _
  rw [smallRow397_block_003_checked, smallRow397_block_004_checked]
  rfl

def smallRow397_small_right : List Nat := smallRow397_block_002 ++ smallRow397_small_right_right
theorem smallRow397_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 397 = smallRow397_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 397 ++ upperNeighbors PrimePairVertices.tree_small_right_right 397 = _
  rw [smallRow397_block_002_checked, smallRow397_small_right_right_checked]
  rfl

def smallRow397_small : List Nat := smallRow397_small_left ++ smallRow397_small_right
theorem smallRow397_small_checked : upperNeighbors PrimePairVertices.tree_small 397 = smallRow397_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 397 ++ upperNeighbors PrimePairVertices.tree_small_right 397 = _
  rw [smallRow397_small_left_checked, smallRow397_small_right_checked]
  rfl

def smallRow397 : List Nat := smallRow397_small
theorem smallRow397_checked : upperNeighbors PrimePairVertices.smallTree 397 = smallRow397 := smallRow397_small_checked

def smallRow401_block_000 : List Nat := [593, 743, 809, 887, 1289, 1487]
set_option maxRecDepth 2048 in
theorem smallRow401_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 401 = smallRow401_block_000 := by decide

def smallRow401_block_001 : List Nat := [2213, 2441, 2999, 3407, 3623, 3671]
set_option maxRecDepth 2048 in
theorem smallRow401_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 401 = smallRow401_block_001 := by decide

def smallRow401_block_002 : List Nat := [4133, 4211, 4217, 4271, 5051, 5801]
set_option maxRecDepth 2048 in
theorem smallRow401_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 401 = smallRow401_block_002 := by decide

def smallRow401_block_003 : List Nat := [6317, 6833, 7589, 7691]
set_option maxRecDepth 2048 in
theorem smallRow401_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 401 = smallRow401_block_003 := by decide

def smallRow401_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow401_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 401 = smallRow401_block_004 := by decide

def smallRow401_small_left : List Nat := smallRow401_block_000 ++ smallRow401_block_001
theorem smallRow401_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 401 = smallRow401_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 401 ++ upperNeighbors PrimePairVertices.tree_block_001 401 = _
  rw [smallRow401_block_000_checked, smallRow401_block_001_checked]
  rfl

def smallRow401_small_right_right : List Nat := smallRow401_block_003 ++ smallRow401_block_004
theorem smallRow401_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 401 = smallRow401_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 401 ++ upperNeighbors PrimePairVertices.tree_block_004 401 = _
  rw [smallRow401_block_003_checked, smallRow401_block_004_checked]
  rfl

def smallRow401_small_right : List Nat := smallRow401_block_002 ++ smallRow401_small_right_right
theorem smallRow401_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 401 = smallRow401_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 401 ++ upperNeighbors PrimePairVertices.tree_small_right_right 401 = _
  rw [smallRow401_block_002_checked, smallRow401_small_right_right_checked]
  rfl

def smallRow401_small : List Nat := smallRow401_small_left ++ smallRow401_small_right
theorem smallRow401_small_checked : upperNeighbors PrimePairVertices.tree_small 401 = smallRow401_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 401 ++ upperNeighbors PrimePairVertices.tree_small_right 401 = _
  rw [smallRow401_small_left_checked, smallRow401_small_right_checked]
  rfl

def smallRow401 : List Nat := smallRow401_small
theorem smallRow401_checked : upperNeighbors PrimePairVertices.smallTree 401 = smallRow401 := smallRow401_small_checked

def smallRow409_block_000 : List Nat := [691, 709, 733, 1471, 1741, 1873]
set_option maxRecDepth 2048 in
theorem smallRow409_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 409 = smallRow409_block_000 := by decide

def smallRow409_block_001 : List Nat := [2269, 2287, 2503, 2791, 3163, 3301, 3343, 3847]
set_option maxRecDepth 2048 in
theorem smallRow409_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 409 = smallRow409_block_001 := by decide

def smallRow409_block_002 : List Nat := [4243, 4561, 4999, 5647, 5857]
set_option maxRecDepth 2048 in
theorem smallRow409_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 409 = smallRow409_block_002 := by decide

def smallRow409_block_003 : List Nat := [6199, 6607, 7057, 7069, 7573, 7699, 8089]
set_option maxRecDepth 2048 in
theorem smallRow409_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 409 = smallRow409_block_003 := by decide

def smallRow409_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow409_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 409 = smallRow409_block_004 := by decide

def smallRow409_small_left : List Nat := smallRow409_block_000 ++ smallRow409_block_001
theorem smallRow409_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 409 = smallRow409_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 409 ++ upperNeighbors PrimePairVertices.tree_block_001 409 = _
  rw [smallRow409_block_000_checked, smallRow409_block_001_checked]
  rfl

def smallRow409_small_right_right : List Nat := smallRow409_block_003 ++ smallRow409_block_004
theorem smallRow409_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 409 = smallRow409_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 409 ++ upperNeighbors PrimePairVertices.tree_block_004 409 = _
  rw [smallRow409_block_003_checked, smallRow409_block_004_checked]
  rfl

def smallRow409_small_right : List Nat := smallRow409_block_002 ++ smallRow409_small_right_right
theorem smallRow409_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 409 = smallRow409_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 409 ++ upperNeighbors PrimePairVertices.tree_small_right_right 409 = _
  rw [smallRow409_block_002_checked, smallRow409_small_right_right_checked]
  rfl

def smallRow409_small : List Nat := smallRow409_small_left ++ smallRow409_small_right
theorem smallRow409_small_checked : upperNeighbors PrimePairVertices.tree_small 409 = smallRow409_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 409 ++ upperNeighbors PrimePairVertices.tree_small_right 409 = _
  rw [smallRow409_small_left_checked, smallRow409_small_right_checked]
  rfl

def smallRow409 : List Nat := smallRow409_small
theorem smallRow409_checked : upperNeighbors PrimePairVertices.smallTree 409 = smallRow409 := smallRow409_small_checked

def smallRow419_block_000 : List Nat := [443, 449, 563, 599, 701, 929, 1097, 1373]
set_option maxRecDepth 2048 in
theorem smallRow419_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 419 = smallRow419_block_000 := by decide

def smallRow419_block_001 : List Nat := [2087, 2129, 3209, 3221, 3803]
set_option maxRecDepth 2048 in
theorem smallRow419_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 419 = smallRow419_block_001 := by decide

def smallRow419_block_002 : List Nat := [4583, 5273, 5297, 5441]
set_option maxRecDepth 2048 in
theorem smallRow419_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 419 = smallRow419_block_002 := by decide

def smallRow419_block_003 : List Nat := [7253, 7757, 7883, 8111]
set_option maxRecDepth 2048 in
theorem smallRow419_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 419 = smallRow419_block_003 := by decide

def smallRow419_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow419_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 419 = smallRow419_block_004 := by decide

def smallRow419_small_left : List Nat := smallRow419_block_000 ++ smallRow419_block_001
theorem smallRow419_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 419 = smallRow419_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 419 ++ upperNeighbors PrimePairVertices.tree_block_001 419 = _
  rw [smallRow419_block_000_checked, smallRow419_block_001_checked]
  rfl

def smallRow419_small_right_right : List Nat := smallRow419_block_003 ++ smallRow419_block_004
theorem smallRow419_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 419 = smallRow419_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 419 ++ upperNeighbors PrimePairVertices.tree_block_004 419 = _
  rw [smallRow419_block_003_checked, smallRow419_block_004_checked]
  rfl

def smallRow419_small_right : List Nat := smallRow419_block_002 ++ smallRow419_small_right_right
theorem smallRow419_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 419 = smallRow419_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 419 ++ upperNeighbors PrimePairVertices.tree_small_right_right 419 = _
  rw [smallRow419_block_002_checked, smallRow419_small_right_right_checked]
  rfl

def smallRow419_small : List Nat := smallRow419_small_left ++ smallRow419_small_right
theorem smallRow419_small_checked : upperNeighbors PrimePairVertices.tree_small 419 = smallRow419_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 419 ++ upperNeighbors PrimePairVertices.tree_small_right 419 = _
  rw [smallRow419_small_left_checked, smallRow419_small_right_checked]
  rfl

def smallRow419 : List Nat := smallRow419_small
theorem smallRow419_checked : upperNeighbors PrimePairVertices.smallTree 419 = smallRow419 := smallRow419_small_checked

def smallRow421_block_000 : List Nat := [433, 607, 643, 661, 709, 1321, 1579, 1657]
set_option maxRecDepth 2048 in
theorem smallRow421_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 421 = smallRow421_block_000 := by decide

def smallRow421_block_001 : List Nat := [2113, 2287, 3217]
set_option maxRecDepth 2048 in
theorem smallRow421_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 421 = smallRow421_block_001 := by decide

def smallRow421_block_002 : List Nat := [4519, 4933, 5347, 5641, 5683, 5749, 6073]
set_option maxRecDepth 2048 in
theorem smallRow421_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 421 = smallRow421_block_002 := by decide

def smallRow421_block_003 : List Nat := [6211, 6871, 7159, 7237, 7459, 7669]
set_option maxRecDepth 2048 in
theorem smallRow421_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 421 = smallRow421_block_003 := by decide

def smallRow421_block_004 : List Nat := [8287]
set_option maxRecDepth 2048 in
theorem smallRow421_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 421 = smallRow421_block_004 := by decide

def smallRow421_small_left : List Nat := smallRow421_block_000 ++ smallRow421_block_001
theorem smallRow421_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 421 = smallRow421_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 421 ++ upperNeighbors PrimePairVertices.tree_block_001 421 = _
  rw [smallRow421_block_000_checked, smallRow421_block_001_checked]
  rfl

def smallRow421_small_right_right : List Nat := smallRow421_block_003 ++ smallRow421_block_004
theorem smallRow421_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 421 = smallRow421_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 421 ++ upperNeighbors PrimePairVertices.tree_block_004 421 = _
  rw [smallRow421_block_003_checked, smallRow421_block_004_checked]
  rfl

def smallRow421_small_right : List Nat := smallRow421_block_002 ++ smallRow421_small_right_right
theorem smallRow421_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 421 = smallRow421_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 421 ++ upperNeighbors PrimePairVertices.tree_small_right_right 421 = _
  rw [smallRow421_block_002_checked, smallRow421_small_right_right_checked]
  rfl

def smallRow421_small : List Nat := smallRow421_small_left ++ smallRow421_small_right
theorem smallRow421_small_checked : upperNeighbors PrimePairVertices.tree_small 421 = smallRow421_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 421 ++ upperNeighbors PrimePairVertices.tree_small_right 421 = _
  rw [smallRow421_small_left_checked, smallRow421_small_right_checked]
  rfl

def smallRow421 : List Nat := smallRow421_small
theorem smallRow421_checked : upperNeighbors PrimePairVertices.smallTree 421 = smallRow421 := smallRow421_small_checked

end ProjectEulerSolutions.PrimePairCache
