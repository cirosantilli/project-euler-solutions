import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow467_block_000 : List Nat := [587, 617, 641, 941, 1301, 1571, 1913, 1973, 2039]
set_option maxRecDepth 2048 in
theorem smallRow467_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 467 = smallRow467_block_000 := by decide

def smallRow467_block_001 : List Nat := [2099, 2141, 2333, 2621, 2687, 3917, 3989, 4049]
set_option maxRecDepth 2048 in
theorem smallRow467_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 467 = smallRow467_block_001 := by decide

def smallRow467_block_002 : List Nat := [4127, 4253, 4421, 4463, 5477, 5861, 6029, 6101]
set_option maxRecDepth 2048 in
theorem smallRow467_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 467 = smallRow467_block_002 := by decide

def smallRow467_block_003 : List Nat := [6323, 6779, 6827]
set_option maxRecDepth 2048 in
theorem smallRow467_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 467 = smallRow467_block_003 := by decide

def smallRow467_block_004 : List Nat := [8231]
set_option maxRecDepth 2048 in
theorem smallRow467_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 467 = smallRow467_block_004 := by decide

def smallRow467_small_left : List Nat := smallRow467_block_000 ++ smallRow467_block_001
theorem smallRow467_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 467 = smallRow467_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 467 ++ upperNeighbors PrimePairVertices.tree_block_001 467 = _
  rw [smallRow467_block_000_checked, smallRow467_block_001_checked]
  rfl

def smallRow467_small_right_right : List Nat := smallRow467_block_003 ++ smallRow467_block_004
theorem smallRow467_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 467 = smallRow467_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 467 ++ upperNeighbors PrimePairVertices.tree_block_004 467 = _
  rw [smallRow467_block_003_checked, smallRow467_block_004_checked]
  rfl

def smallRow467_small_right : List Nat := smallRow467_block_002 ++ smallRow467_small_right_right
theorem smallRow467_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 467 = smallRow467_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 467 ++ upperNeighbors PrimePairVertices.tree_small_right_right 467 = _
  rw [smallRow467_block_002_checked, smallRow467_small_right_right_checked]
  rfl

def smallRow467_small : List Nat := smallRow467_small_left ++ smallRow467_small_right
theorem smallRow467_small_checked : upperNeighbors PrimePairVertices.tree_small 467 = smallRow467_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 467 ++ upperNeighbors PrimePairVertices.tree_small_right 467 = _
  rw [smallRow467_small_left_checked, smallRow467_small_right_checked]
  rfl

def smallRow467 : List Nat := smallRow467_small
theorem smallRow467_checked : upperNeighbors PrimePairVertices.smallTree 467 = smallRow467 := smallRow467_small_checked

def smallRow479_block_000 : List Nat := [569, 593, 599, 701, 821, 881, 971, 1049, 1187, 1307, 1559]
set_option maxRecDepth 2048 in
theorem smallRow479_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 479 = smallRow479_block_000 := by decide

def smallRow479_block_001 : List Nat := [2297, 3533]
set_option maxRecDepth 2048 in
theorem smallRow479_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 479 = smallRow479_block_001 := by decide

def smallRow479_block_002 : List Nat := [4337, 4481, 4733, 4817, 5807]
set_option maxRecDepth 2048 in
theorem smallRow479_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 479 = smallRow479_block_002 := by decide

def smallRow479_block_003 : List Nat := [6203, 7517]
set_option maxRecDepth 2048 in
theorem smallRow479_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 479 = smallRow479_block_003 := by decide

def smallRow479_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow479_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 479 = smallRow479_block_004 := by decide

def smallRow479_small_left : List Nat := smallRow479_block_000 ++ smallRow479_block_001
theorem smallRow479_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 479 = smallRow479_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 479 ++ upperNeighbors PrimePairVertices.tree_block_001 479 = _
  rw [smallRow479_block_000_checked, smallRow479_block_001_checked]
  rfl

def smallRow479_small_right_right : List Nat := smallRow479_block_003 ++ smallRow479_block_004
theorem smallRow479_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 479 = smallRow479_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 479 ++ upperNeighbors PrimePairVertices.tree_block_004 479 = _
  rw [smallRow479_block_003_checked, smallRow479_block_004_checked]
  rfl

def smallRow479_small_right : List Nat := smallRow479_block_002 ++ smallRow479_small_right_right
theorem smallRow479_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 479 = smallRow479_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 479 ++ upperNeighbors PrimePairVertices.tree_small_right_right 479 = _
  rw [smallRow479_block_002_checked, smallRow479_small_right_right_checked]
  rfl

def smallRow479_small : List Nat := smallRow479_small_left ++ smallRow479_small_right
theorem smallRow479_small_checked : upperNeighbors PrimePairVertices.tree_small 479 = smallRow479_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 479 ++ upperNeighbors PrimePairVertices.tree_small_right 479 = _
  rw [smallRow479_small_left_checked, smallRow479_small_right_checked]
  rfl

def smallRow479 : List Nat := smallRow479_small
theorem smallRow479_checked : upperNeighbors PrimePairVertices.smallTree 479 = smallRow479 := smallRow479_small_checked

def smallRow487_block_000 : List Nat := [601, 727, 757, 769, 1093, 1297, 1567, 1723]
set_option maxRecDepth 2048 in
theorem smallRow487_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 487 = smallRow487_block_000 := by decide

def smallRow487_block_001 : List Nat := [2311, 3163, 3433, 3559, 3697]
set_option maxRecDepth 2048 in
theorem smallRow487_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 487 = smallRow487_block_001 := by decide

def smallRow487_block_002 : List Nat := [4339, 5113, 5407, 5881, 6133]
set_option maxRecDepth 2048 in
theorem smallRow487_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 487 = smallRow487_block_002 := by decide

def smallRow487_block_003 : List Nat := [6301, 6607, 6709]
set_option maxRecDepth 2048 in
theorem smallRow487_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 487 = smallRow487_block_003 := by decide

def smallRow487_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow487_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 487 = smallRow487_block_004 := by decide

def smallRow487_small_left : List Nat := smallRow487_block_000 ++ smallRow487_block_001
theorem smallRow487_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 487 = smallRow487_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 487 ++ upperNeighbors PrimePairVertices.tree_block_001 487 = _
  rw [smallRow487_block_000_checked, smallRow487_block_001_checked]
  rfl

def smallRow487_small_right_right : List Nat := smallRow487_block_003 ++ smallRow487_block_004
theorem smallRow487_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 487 = smallRow487_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 487 ++ upperNeighbors PrimePairVertices.tree_block_004 487 = _
  rw [smallRow487_block_003_checked, smallRow487_block_004_checked]
  rfl

def smallRow487_small_right : List Nat := smallRow487_block_002 ++ smallRow487_small_right_right
theorem smallRow487_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 487 = smallRow487_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 487 ++ upperNeighbors PrimePairVertices.tree_small_right_right 487 = _
  rw [smallRow487_block_002_checked, smallRow487_small_right_right_checked]
  rfl

def smallRow487_small : List Nat := smallRow487_small_left ++ smallRow487_small_right
theorem smallRow487_small_checked : upperNeighbors PrimePairVertices.tree_small 487 = smallRow487_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 487 ++ upperNeighbors PrimePairVertices.tree_small_right 487 = _
  rw [smallRow487_small_left_checked, smallRow487_small_right_checked]
  rfl

def smallRow487 : List Nat := smallRow487_small
theorem smallRow487_checked : upperNeighbors PrimePairVertices.smallTree 487 = smallRow487 := smallRow487_small_checked

def smallRow491_block_000 : List Nat := [593, 653, 773, 983, 1409, 1523, 1607, 1871]
set_option maxRecDepth 2048 in
theorem smallRow491_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 491 = smallRow491_block_000 := by decide

def smallRow491_block_001 : List Nat := [2309, 2447, 2459, 2837, 3449, 3617, 3917, 3947]
set_option maxRecDepth 2048 in
theorem smallRow491_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 491 = smallRow491_block_001 := by decide

def smallRow491_block_002 : List Nat := [4211, 4457, 5843, 6053]
set_option maxRecDepth 2048 in
theorem smallRow491_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 491 = smallRow491_block_002 := by decide

def smallRow491_block_003 : List Nat := [6257, 6323, 6719, 6803, 7079, 7481, 7853]
set_option maxRecDepth 2048 in
theorem smallRow491_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 491 = smallRow491_block_003 := by decide

def smallRow491_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow491_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 491 = smallRow491_block_004 := by decide

def smallRow491_small_left : List Nat := smallRow491_block_000 ++ smallRow491_block_001
theorem smallRow491_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 491 = smallRow491_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 491 ++ upperNeighbors PrimePairVertices.tree_block_001 491 = _
  rw [smallRow491_block_000_checked, smallRow491_block_001_checked]
  rfl

def smallRow491_small_right_right : List Nat := smallRow491_block_003 ++ smallRow491_block_004
theorem smallRow491_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 491 = smallRow491_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 491 ++ upperNeighbors PrimePairVertices.tree_block_004 491 = _
  rw [smallRow491_block_003_checked, smallRow491_block_004_checked]
  rfl

def smallRow491_small_right : List Nat := smallRow491_block_002 ++ smallRow491_small_right_right
theorem smallRow491_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 491 = smallRow491_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 491 ++ upperNeighbors PrimePairVertices.tree_small_right_right 491 = _
  rw [smallRow491_block_002_checked, smallRow491_small_right_right_checked]
  rfl

def smallRow491_small : List Nat := smallRow491_small_left ++ smallRow491_small_right
theorem smallRow491_small_checked : upperNeighbors PrimePairVertices.tree_small 491 = smallRow491_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 491 ++ upperNeighbors PrimePairVertices.tree_small_right 491 = _
  rw [smallRow491_small_left_checked, smallRow491_small_right_checked]
  rfl

def smallRow491 : List Nat := smallRow491_small
theorem smallRow491_checked : upperNeighbors PrimePairVertices.smallTree 491 = smallRow491 := smallRow491_small_checked

def smallRow499_block_000 : List Nat := [673, 691, 1009, 1033, 1237, 1249, 1303, 1453, 1933]
set_option maxRecDepth 2048 in
theorem smallRow499_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 499 = smallRow499_block_000 := by decide

def smallRow499_block_001 : List Nat := [2551, 2707, 3343, 3847, 3907, 4051]
set_option maxRecDepth 2048 in
theorem smallRow499_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 499 = smallRow499_block_001 := by decide

def smallRow499_block_002 : List Nat := [4357, 4513, 4621, 5281, 5503, 5827]
set_option maxRecDepth 2048 in
theorem smallRow499_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 499 = smallRow499_block_002 := by decide

def smallRow499_block_003 : List Nat := [6361, 7393, 7687, 7717]
set_option maxRecDepth 2048 in
theorem smallRow499_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 499 = smallRow499_block_003 := by decide

def smallRow499_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow499_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 499 = smallRow499_block_004 := by decide

def smallRow499_small_left : List Nat := smallRow499_block_000 ++ smallRow499_block_001
theorem smallRow499_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 499 = smallRow499_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 499 ++ upperNeighbors PrimePairVertices.tree_block_001 499 = _
  rw [smallRow499_block_000_checked, smallRow499_block_001_checked]
  rfl

def smallRow499_small_right_right : List Nat := smallRow499_block_003 ++ smallRow499_block_004
theorem smallRow499_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 499 = smallRow499_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 499 ++ upperNeighbors PrimePairVertices.tree_block_004 499 = _
  rw [smallRow499_block_003_checked, smallRow499_block_004_checked]
  rfl

def smallRow499_small_right : List Nat := smallRow499_block_002 ++ smallRow499_small_right_right
theorem smallRow499_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 499 = smallRow499_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 499 ++ upperNeighbors PrimePairVertices.tree_small_right_right 499 = _
  rw [smallRow499_block_002_checked, smallRow499_small_right_right_checked]
  rfl

def smallRow499_small : List Nat := smallRow499_small_left ++ smallRow499_small_right
theorem smallRow499_small_checked : upperNeighbors PrimePairVertices.tree_small 499 = smallRow499_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 499 ++ upperNeighbors PrimePairVertices.tree_small_right 499 = _
  rw [smallRow499_small_left_checked, smallRow499_small_right_checked]
  rfl

def smallRow499 : List Nat := smallRow499_small
theorem smallRow499_checked : upperNeighbors PrimePairVertices.smallTree 499 = smallRow499 := smallRow499_small_checked

def smallRow503_block_000 : List Nat := [563, 647, 653, 911, 1193, 1601, 1997]
set_option maxRecDepth 2048 in
theorem smallRow503_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 503 = smallRow503_block_000 := by decide

def smallRow503_block_001 : List Nat := [2411, 2711, 3299, 3593]
set_option maxRecDepth 2048 in
theorem smallRow503_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 503 = smallRow503_block_001 := by decide

def smallRow503_block_002 : List Nat := [4229, 5003, 5021, 5309, 5351, 5507, 5843, 5903]
set_option maxRecDepth 2048 in
theorem smallRow503_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 503 = smallRow503_block_002 := by decide

def smallRow503_block_003 : List Nat := [7673, 7817]
set_option maxRecDepth 2048 in
theorem smallRow503_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 503 = smallRow503_block_003 := by decide

def smallRow503_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow503_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 503 = smallRow503_block_004 := by decide

def smallRow503_small_left : List Nat := smallRow503_block_000 ++ smallRow503_block_001
theorem smallRow503_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 503 = smallRow503_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 503 ++ upperNeighbors PrimePairVertices.tree_block_001 503 = _
  rw [smallRow503_block_000_checked, smallRow503_block_001_checked]
  rfl

def smallRow503_small_right_right : List Nat := smallRow503_block_003 ++ smallRow503_block_004
theorem smallRow503_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 503 = smallRow503_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 503 ++ upperNeighbors PrimePairVertices.tree_block_004 503 = _
  rw [smallRow503_block_003_checked, smallRow503_block_004_checked]
  rfl

def smallRow503_small_right : List Nat := smallRow503_block_002 ++ smallRow503_small_right_right
theorem smallRow503_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 503 = smallRow503_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 503 ++ upperNeighbors PrimePairVertices.tree_small_right_right 503 = _
  rw [smallRow503_block_002_checked, smallRow503_small_right_right_checked]
  rfl

def smallRow503_small : List Nat := smallRow503_small_left ++ smallRow503_small_right
theorem smallRow503_small_checked : upperNeighbors PrimePairVertices.tree_small 503 = smallRow503_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 503 ++ upperNeighbors PrimePairVertices.tree_small_right 503 = _
  rw [smallRow503_small_left_checked, smallRow503_small_right_checked]
  rfl

def smallRow503 : List Nat := smallRow503_small
theorem smallRow503_checked : upperNeighbors PrimePairVertices.smallTree 503 = smallRow503 := smallRow503_small_checked

def smallRow509_block_000 : List Nat := [647, 797, 863, 947, 1019, 1061, 1091, 1451, 1553, 1613, 1973]
set_option maxRecDepth 2048 in
theorem smallRow509_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 509 = smallRow509_block_000 := by decide

def smallRow509_block_001 : List Nat := [2111, 3359, 3929]
set_option maxRecDepth 2048 in
theorem smallRow509_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 509 = smallRow509_block_001 := by decide

def smallRow509_block_002 : List Nat := [4241, 4673, 5471, 5483, 5711]
set_option maxRecDepth 2048 in
theorem smallRow509_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 509 = smallRow509_block_002 := by decide

def smallRow509_block_003 : List Nat := [6563, 6863, 7193, 7577, 7583, 7793, 8117]
set_option maxRecDepth 2048 in
theorem smallRow509_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 509 = smallRow509_block_003 := by decide

def smallRow509_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow509_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 509 = smallRow509_block_004 := by decide

def smallRow509_small_left : List Nat := smallRow509_block_000 ++ smallRow509_block_001
theorem smallRow509_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 509 = smallRow509_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 509 ++ upperNeighbors PrimePairVertices.tree_block_001 509 = _
  rw [smallRow509_block_000_checked, smallRow509_block_001_checked]
  rfl

def smallRow509_small_right_right : List Nat := smallRow509_block_003 ++ smallRow509_block_004
theorem smallRow509_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 509 = smallRow509_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 509 ++ upperNeighbors PrimePairVertices.tree_block_004 509 = _
  rw [smallRow509_block_003_checked, smallRow509_block_004_checked]
  rfl

def smallRow509_small_right : List Nat := smallRow509_block_002 ++ smallRow509_small_right_right
theorem smallRow509_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 509 = smallRow509_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 509 ++ upperNeighbors PrimePairVertices.tree_small_right_right 509 = _
  rw [smallRow509_block_002_checked, smallRow509_small_right_right_checked]
  rfl

def smallRow509_small : List Nat := smallRow509_small_left ++ smallRow509_small_right
theorem smallRow509_small_checked : upperNeighbors PrimePairVertices.tree_small 509 = smallRow509_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 509 ++ upperNeighbors PrimePairVertices.tree_small_right 509 = _
  rw [smallRow509_small_left_checked, smallRow509_small_right_checked]
  rfl

def smallRow509 : List Nat := smallRow509_small
theorem smallRow509_checked : upperNeighbors PrimePairVertices.smallTree 509 = smallRow509 := smallRow509_small_checked

def smallRow521_block_000 : List Nat := [557, 641, 659, 809, 1091, 1229, 1373, 1523, 1637, 1721]
set_option maxRecDepth 2048 in
theorem smallRow521_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 521 = smallRow521_block_000 := by decide

def smallRow521_block_001 : List Nat := [2381, 2447, 2477, 2777, 3137, 3389, 3539]
set_option maxRecDepth 2048 in
theorem smallRow521_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 521 = smallRow521_block_001 := by decide

def smallRow521_block_002 : List Nat := [4337, 4691, 5381, 5417, 6053]
set_option maxRecDepth 2048 in
theorem smallRow521_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 521 = smallRow521_block_002 := by decide

def smallRow521_block_003 : List Nat := [6269, 6311, 6353, 6521, 7607, 7673, 7907]
set_option maxRecDepth 2048 in
theorem smallRow521_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 521 = smallRow521_block_003 := by decide

def smallRow521_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow521_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 521 = smallRow521_block_004 := by decide

def smallRow521_small_left : List Nat := smallRow521_block_000 ++ smallRow521_block_001
theorem smallRow521_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 521 = smallRow521_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 521 ++ upperNeighbors PrimePairVertices.tree_block_001 521 = _
  rw [smallRow521_block_000_checked, smallRow521_block_001_checked]
  rfl

def smallRow521_small_right_right : List Nat := smallRow521_block_003 ++ smallRow521_block_004
theorem smallRow521_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 521 = smallRow521_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 521 ++ upperNeighbors PrimePairVertices.tree_block_004 521 = _
  rw [smallRow521_block_003_checked, smallRow521_block_004_checked]
  rfl

def smallRow521_small_right : List Nat := smallRow521_block_002 ++ smallRow521_small_right_right
theorem smallRow521_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 521 = smallRow521_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 521 ++ upperNeighbors PrimePairVertices.tree_small_right_right 521 = _
  rw [smallRow521_block_002_checked, smallRow521_small_right_right_checked]
  rfl

def smallRow521_small : List Nat := smallRow521_small_left ++ smallRow521_small_right
theorem smallRow521_small_checked : upperNeighbors PrimePairVertices.tree_small 521 = smallRow521_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 521 ++ upperNeighbors PrimePairVertices.tree_small_right 521 = _
  rw [smallRow521_small_left_checked, smallRow521_small_right_checked]
  rfl

def smallRow521 : List Nat := smallRow521_small
theorem smallRow521_checked : upperNeighbors PrimePairVertices.smallTree 521 = smallRow521 := smallRow521_small_checked

end ProjectEulerSolutions.PrimePairCache
