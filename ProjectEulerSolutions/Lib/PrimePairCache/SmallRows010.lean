import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow431_block_000 : List Nat := [479, 857, 929, 947, 983, 1031, 1163, 1193, 1361, 1823]
set_option maxRecDepth 2048 in
theorem smallRow431_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 431 = smallRow431_block_000 := by decide

def smallRow431_block_001 : List Nat := [2393, 2417, 2549, 2837, 3671, 3851]
set_option maxRecDepth 2048 in
theorem smallRow431_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 431 = smallRow431_block_001 := by decide

def smallRow431_block_002 : List Nat := [4217, 4943, 5903, 5981]
set_option maxRecDepth 2048 in
theorem smallRow431_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 431 = smallRow431_block_002 := by decide

def smallRow431_block_003 : List Nat := [6197, 6329, 6803, 7127, 7283, 7517]
set_option maxRecDepth 2048 in
theorem smallRow431_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 431 = smallRow431_block_003 := by decide

def smallRow431_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow431_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 431 = smallRow431_block_004 := by decide

def smallRow431_small_left : List Nat := smallRow431_block_000 ++ smallRow431_block_001
theorem smallRow431_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 431 = smallRow431_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 431 ++ upperNeighbors PrimePairVertices.tree_block_001 431 = _
  rw [smallRow431_block_000_checked, smallRow431_block_001_checked]
  rfl

def smallRow431_small_right_right : List Nat := smallRow431_block_003 ++ smallRow431_block_004
theorem smallRow431_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 431 = smallRow431_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 431 ++ upperNeighbors PrimePairVertices.tree_block_004 431 = _
  rw [smallRow431_block_003_checked, smallRow431_block_004_checked]
  rfl

def smallRow431_small_right : List Nat := smallRow431_block_002 ++ smallRow431_small_right_right
theorem smallRow431_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 431 = smallRow431_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 431 ++ upperNeighbors PrimePairVertices.tree_small_right_right 431 = _
  rw [smallRow431_block_002_checked, smallRow431_small_right_right_checked]
  rfl

def smallRow431_small : List Nat := smallRow431_small_left ++ smallRow431_small_right
theorem smallRow431_small_checked : upperNeighbors PrimePairVertices.tree_small 431 = smallRow431_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 431 ++ upperNeighbors PrimePairVertices.tree_small_right 431 = _
  rw [smallRow431_small_left_checked, smallRow431_small_right_checked]
  rfl

def smallRow431 : List Nat := smallRow431_small
theorem smallRow431_checked : upperNeighbors PrimePairVertices.smallTree 431 = smallRow431 := smallRow431_small_checked

def smallRow433_block_000 : List Nat := [571, 787, 859, 883, 1009, 1087, 1471, 1801]
set_option maxRecDepth 2048 in
theorem smallRow433_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 433 = smallRow433_block_000 := by decide

def smallRow433_block_001 : List Nat := [2203, 2719, 2857, 3391, 3529, 3613, 3697, 3739]
set_option maxRecDepth 2048 in
theorem smallRow433_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 433 = smallRow433_block_001 := by decide

def smallRow433_block_002 : List Nat := [4441, 4483, 4663, 4903, 5281, 5407, 6091]
set_option maxRecDepth 2048 in
theorem smallRow433_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 433 = smallRow433_block_002 := by decide

def smallRow433_block_003 : List Nat := [6247, 6379, 6781, 7393, 7717, 8167]
set_option maxRecDepth 2048 in
theorem smallRow433_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 433 = smallRow433_block_003 := by decide

def smallRow433_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow433_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 433 = smallRow433_block_004 := by decide

def smallRow433_small_left : List Nat := smallRow433_block_000 ++ smallRow433_block_001
theorem smallRow433_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 433 = smallRow433_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 433 ++ upperNeighbors PrimePairVertices.tree_block_001 433 = _
  rw [smallRow433_block_000_checked, smallRow433_block_001_checked]
  rfl

def smallRow433_small_right_right : List Nat := smallRow433_block_003 ++ smallRow433_block_004
theorem smallRow433_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 433 = smallRow433_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 433 ++ upperNeighbors PrimePairVertices.tree_block_004 433 = _
  rw [smallRow433_block_003_checked, smallRow433_block_004_checked]
  rfl

def smallRow433_small_right : List Nat := smallRow433_block_002 ++ smallRow433_small_right_right
theorem smallRow433_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 433 = smallRow433_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 433 ++ upperNeighbors PrimePairVertices.tree_small_right_right 433 = _
  rw [smallRow433_block_002_checked, smallRow433_small_right_right_checked]
  rfl

def smallRow433_small : List Nat := smallRow433_small_left ++ smallRow433_small_right
theorem smallRow433_small_checked : upperNeighbors PrimePairVertices.tree_small 433 = smallRow433_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 433 ++ upperNeighbors PrimePairVertices.tree_small_right 433 = _
  rw [smallRow433_small_left_checked, smallRow433_small_right_checked]
  rfl

def smallRow433 : List Nat := smallRow433_small
theorem smallRow433_checked : upperNeighbors PrimePairVertices.smallTree 433 = smallRow433 := smallRow433_small_checked

def smallRow439_block_000 : List Nat := [541, 601, 613, 661, 787, 853, 1399, 1483, 1549, 1579]
set_option maxRecDepth 2048 in
theorem smallRow439_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 439 = smallRow439_block_000 := by decide

def smallRow439_block_001 : List Nat := [2383, 2503, 2593, 2797, 2887, 3511, 4021]
set_option maxRecDepth 2048 in
theorem smallRow439_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 439 = smallRow439_block_001 := by decide

def smallRow439_block_002 : List Nat := [4441, 4447, 4783, 5527, 5557, 5701, 5827]
set_option maxRecDepth 2048 in
theorem smallRow439_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 439 = smallRow439_block_002 := by decide

def smallRow439_block_003 : List Nat := [6481, 8011]
set_option maxRecDepth 2048 in
theorem smallRow439_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 439 = smallRow439_block_003 := by decide

def smallRow439_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow439_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 439 = smallRow439_block_004 := by decide

def smallRow439_small_left : List Nat := smallRow439_block_000 ++ smallRow439_block_001
theorem smallRow439_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 439 = smallRow439_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 439 ++ upperNeighbors PrimePairVertices.tree_block_001 439 = _
  rw [smallRow439_block_000_checked, smallRow439_block_001_checked]
  rfl

def smallRow439_small_right_right : List Nat := smallRow439_block_003 ++ smallRow439_block_004
theorem smallRow439_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 439 = smallRow439_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 439 ++ upperNeighbors PrimePairVertices.tree_block_004 439 = _
  rw [smallRow439_block_003_checked, smallRow439_block_004_checked]
  rfl

def smallRow439_small_right : List Nat := smallRow439_block_002 ++ smallRow439_small_right_right
theorem smallRow439_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 439 = smallRow439_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 439 ++ upperNeighbors PrimePairVertices.tree_small_right_right 439 = _
  rw [smallRow439_block_002_checked, smallRow439_small_right_right_checked]
  rfl

def smallRow439_small : List Nat := smallRow439_small_left ++ smallRow439_small_right
theorem smallRow439_small_checked : upperNeighbors PrimePairVertices.tree_small 439 = smallRow439_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 439 ++ upperNeighbors PrimePairVertices.tree_small_right 439 = _
  rw [smallRow439_small_left_checked, smallRow439_small_right_checked]
  rfl

def smallRow439 : List Nat := smallRow439_small
theorem smallRow439_checked : upperNeighbors PrimePairVertices.smallTree 439 = smallRow439 := smallRow439_small_checked

def smallRow443_block_000 : List Nat := [701, 761, 953, 983, 1151, 1523, 1811]
set_option maxRecDepth 2048 in
theorem smallRow443_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 443 = smallRow443_block_000 := by decide

def smallRow443_block_001 : List Nat := [2069, 2531, 2693, 3167, 3203, 3449, 3833, 3881]
set_option maxRecDepth 2048 in
theorem smallRow443_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 443 = smallRow443_block_001 := by decide

def smallRow443_block_002 : List Nat := [4673, 4691, 5051, 5741, 5897]
set_option maxRecDepth 2048 in
theorem smallRow443_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 443 = smallRow443_block_002 := by decide

def smallRow443_block_003 : List Nat := [6389, 6737, 6863, 7283, 7523, 7529, 7547, 7757, 7883]
set_option maxRecDepth 2048 in
theorem smallRow443_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 443 = smallRow443_block_003 := by decide

def smallRow443_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow443_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 443 = smallRow443_block_004 := by decide

def smallRow443_small_left : List Nat := smallRow443_block_000 ++ smallRow443_block_001
theorem smallRow443_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 443 = smallRow443_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 443 ++ upperNeighbors PrimePairVertices.tree_block_001 443 = _
  rw [smallRow443_block_000_checked, smallRow443_block_001_checked]
  rfl

def smallRow443_small_right_right : List Nat := smallRow443_block_003 ++ smallRow443_block_004
theorem smallRow443_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 443 = smallRow443_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 443 ++ upperNeighbors PrimePairVertices.tree_block_004 443 = _
  rw [smallRow443_block_003_checked, smallRow443_block_004_checked]
  rfl

def smallRow443_small_right : List Nat := smallRow443_block_002 ++ smallRow443_small_right_right
theorem smallRow443_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 443 = smallRow443_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 443 ++ upperNeighbors PrimePairVertices.tree_small_right_right 443 = _
  rw [smallRow443_block_002_checked, smallRow443_small_right_right_checked]
  rfl

def smallRow443_small : List Nat := smallRow443_small_left ++ smallRow443_small_right
theorem smallRow443_small_checked : upperNeighbors PrimePairVertices.tree_small 443 = smallRow443_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 443 ++ upperNeighbors PrimePairVertices.tree_small_right 443 = _
  rw [smallRow443_small_left_checked, smallRow443_small_right_checked]
  rfl

def smallRow443 : List Nat := smallRow443_small
theorem smallRow443_checked : upperNeighbors PrimePairVertices.smallTree 443 = smallRow443 := smallRow443_small_checked

def smallRow449_block_000 : List Nat := [557, 563, 821, 941, 1871]
set_option maxRecDepth 2048 in
theorem smallRow449_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 449 = smallRow449_block_000 := by decide

def smallRow449_block_001 : List Nat := [2069, 2273, 2309, 2441, 2459, 3659]
set_option maxRecDepth 2048 in
theorem smallRow449_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 449 = smallRow449_block_001 := by decide

def smallRow449_block_002 : List Nat := [4157, 5009, 5171, 5297, 5801]
set_option maxRecDepth 2048 in
theorem smallRow449_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 449 = smallRow449_block_002 := by decide

def smallRow449_block_003 : List Nat := [6353, 6491, 6599, 7109, 7583, 7817, 7919]
set_option maxRecDepth 2048 in
theorem smallRow449_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 449 = smallRow449_block_003 := by decide

def smallRow449_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow449_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 449 = smallRow449_block_004 := by decide

def smallRow449_small_left : List Nat := smallRow449_block_000 ++ smallRow449_block_001
theorem smallRow449_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 449 = smallRow449_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 449 ++ upperNeighbors PrimePairVertices.tree_block_001 449 = _
  rw [smallRow449_block_000_checked, smallRow449_block_001_checked]
  rfl

def smallRow449_small_right_right : List Nat := smallRow449_block_003 ++ smallRow449_block_004
theorem smallRow449_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 449 = smallRow449_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 449 ++ upperNeighbors PrimePairVertices.tree_block_004 449 = _
  rw [smallRow449_block_003_checked, smallRow449_block_004_checked]
  rfl

def smallRow449_small_right : List Nat := smallRow449_block_002 ++ smallRow449_small_right_right
theorem smallRow449_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 449 = smallRow449_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 449 ++ upperNeighbors PrimePairVertices.tree_small_right_right 449 = _
  rw [smallRow449_block_002_checked, smallRow449_small_right_right_checked]
  rfl

def smallRow449_small : List Nat := smallRow449_small_left ++ smallRow449_small_right
theorem smallRow449_small_checked : upperNeighbors PrimePairVertices.tree_small 449 = smallRow449_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 449 ++ upperNeighbors PrimePairVertices.tree_small_right 449 = _
  rw [smallRow449_small_left_checked, smallRow449_small_right_checked]
  rfl

def smallRow449 : List Nat := smallRow449_small
theorem smallRow449_checked : upperNeighbors PrimePairVertices.smallTree 449 = smallRow449 := smallRow449_small_checked

def smallRow457_block_000 : List Nat := [643, 673, 757, 829, 1033, 1597]
set_option maxRecDepth 2048 in
theorem smallRow457_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 457 = smallRow457_block_000 := by decide

def smallRow457_block_001 : List Nat := [2131, 2383, 2551, 2707, 2917, 3571, 3631, 3931]
set_option maxRecDepth 2048 in
theorem smallRow457_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 457 = smallRow457_block_001 := by decide

def smallRow457_block_002 : List Nat := [4441, 4987, 5107, 5281, 5503, 5647]
set_option maxRecDepth 2048 in
theorem smallRow457_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 457 = smallRow457_block_002 := by decide

def smallRow457_block_003 : List Nat := [6997, 7039, 7297, 7411, 7669]
set_option maxRecDepth 2048 in
theorem smallRow457_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 457 = smallRow457_block_003 := by decide

def smallRow457_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow457_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 457 = smallRow457_block_004 := by decide

def smallRow457_small_left : List Nat := smallRow457_block_000 ++ smallRow457_block_001
theorem smallRow457_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 457 = smallRow457_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 457 ++ upperNeighbors PrimePairVertices.tree_block_001 457 = _
  rw [smallRow457_block_000_checked, smallRow457_block_001_checked]
  rfl

def smallRow457_small_right_right : List Nat := smallRow457_block_003 ++ smallRow457_block_004
theorem smallRow457_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 457 = smallRow457_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 457 ++ upperNeighbors PrimePairVertices.tree_block_004 457 = _
  rw [smallRow457_block_003_checked, smallRow457_block_004_checked]
  rfl

def smallRow457_small_right : List Nat := smallRow457_block_002 ++ smallRow457_small_right_right
theorem smallRow457_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 457 = smallRow457_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 457 ++ upperNeighbors PrimePairVertices.tree_small_right_right 457 = _
  rw [smallRow457_block_002_checked, smallRow457_small_right_right_checked]
  rfl

def smallRow457_small : List Nat := smallRow457_small_left ++ smallRow457_small_right
theorem smallRow457_small_checked : upperNeighbors PrimePairVertices.tree_small 457 = smallRow457_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 457 ++ upperNeighbors PrimePairVertices.tree_small_right 457 = _
  rw [smallRow457_small_left_checked, smallRow457_small_right_checked]
  rfl

def smallRow457 : List Nat := smallRow457_small
theorem smallRow457_checked : upperNeighbors PrimePairVertices.smallTree 457 = smallRow457 := smallRow457_small_checked

def smallRow461_block_000 : List Nat := [479, 569, 653, 677, 983, 1181, 1307, 1487, 1571, 1697]
set_option maxRecDepth 2048 in
theorem smallRow461_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 461 = smallRow461_block_000 := by decide

def smallRow461_block_001 : List Nat := [2339, 2549, 3449, 3881]
set_option maxRecDepth 2048 in
theorem smallRow461_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 461 = smallRow461_block_001 := by decide

def smallRow461_block_002 : List Nat := [4217, 4271, 4637, 4937, 5783]
set_option maxRecDepth 2048 in
theorem smallRow461_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 461 = smallRow461_block_002 := by decide

def smallRow461_block_003 : List Nat := [6473, 6977, 7187, 7451]
set_option maxRecDepth 2048 in
theorem smallRow461_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 461 = smallRow461_block_003 := by decide

def smallRow461_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow461_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 461 = smallRow461_block_004 := by decide

def smallRow461_small_left : List Nat := smallRow461_block_000 ++ smallRow461_block_001
theorem smallRow461_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 461 = smallRow461_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 461 ++ upperNeighbors PrimePairVertices.tree_block_001 461 = _
  rw [smallRow461_block_000_checked, smallRow461_block_001_checked]
  rfl

def smallRow461_small_right_right : List Nat := smallRow461_block_003 ++ smallRow461_block_004
theorem smallRow461_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 461 = smallRow461_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 461 ++ upperNeighbors PrimePairVertices.tree_block_004 461 = _
  rw [smallRow461_block_003_checked, smallRow461_block_004_checked]
  rfl

def smallRow461_small_right : List Nat := smallRow461_block_002 ++ smallRow461_small_right_right
theorem smallRow461_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 461 = smallRow461_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 461 ++ upperNeighbors PrimePairVertices.tree_small_right_right 461 = _
  rw [smallRow461_block_002_checked, smallRow461_small_right_right_checked]
  rfl

def smallRow461_small : List Nat := smallRow461_small_left ++ smallRow461_small_right
theorem smallRow461_small_checked : upperNeighbors PrimePairVertices.tree_small 461 = smallRow461_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 461 ++ upperNeighbors PrimePairVertices.tree_small_right 461 = _
  rw [smallRow461_small_left_checked, smallRow461_small_right_checked]
  rfl

def smallRow461 : List Nat := smallRow461_small
theorem smallRow461_checked : upperNeighbors PrimePairVertices.smallTree 461 = smallRow461 := smallRow461_small_checked

def smallRow463_block_000 : List Nat := [523, 613, 643, 829, 1171, 1489, 1663]
set_option maxRecDepth 2048 in
theorem smallRow463_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 463 = smallRow463_block_000 := by decide

def smallRow463_block_001 : List Nat := [2473, 2917, 3001, 3259, 3331, 3457, 3793, 3931]
set_option maxRecDepth 2048 in
theorem smallRow463_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 463 = smallRow463_block_001 := by decide

def smallRow463_block_002 : List Nat := [4111, 4723, 4933, 5437, 5569, 5623]
set_option maxRecDepth 2048 in
theorem smallRow463_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 463 = smallRow463_block_002 := by decide

def smallRow463_block_003 : List Nat := [6451, 6967, 7333, 7489, 7669]
set_option maxRecDepth 2048 in
theorem smallRow463_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 463 = smallRow463_block_003 := by decide

def smallRow463_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow463_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 463 = smallRow463_block_004 := by decide

def smallRow463_small_left : List Nat := smallRow463_block_000 ++ smallRow463_block_001
theorem smallRow463_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 463 = smallRow463_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 463 ++ upperNeighbors PrimePairVertices.tree_block_001 463 = _
  rw [smallRow463_block_000_checked, smallRow463_block_001_checked]
  rfl

def smallRow463_small_right_right : List Nat := smallRow463_block_003 ++ smallRow463_block_004
theorem smallRow463_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 463 = smallRow463_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 463 ++ upperNeighbors PrimePairVertices.tree_block_004 463 = _
  rw [smallRow463_block_003_checked, smallRow463_block_004_checked]
  rfl

def smallRow463_small_right : List Nat := smallRow463_block_002 ++ smallRow463_small_right_right
theorem smallRow463_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 463 = smallRow463_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 463 ++ upperNeighbors PrimePairVertices.tree_small_right_right 463 = _
  rw [smallRow463_block_002_checked, smallRow463_small_right_right_checked]
  rfl

def smallRow463_small : List Nat := smallRow463_small_left ++ smallRow463_small_right
theorem smallRow463_small_checked : upperNeighbors PrimePairVertices.tree_small 463 = smallRow463_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 463 ++ upperNeighbors PrimePairVertices.tree_small_right 463 = _
  rw [smallRow463_small_left_checked, smallRow463_small_right_checked]
  rfl

def smallRow463 : List Nat := smallRow463_small
theorem smallRow463_checked : upperNeighbors PrimePairVertices.smallTree 463 = smallRow463 := smallRow463_small_checked

end ProjectEulerSolutions.PrimePairCache
