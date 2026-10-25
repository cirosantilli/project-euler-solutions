import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow967_block_000 : List Nat := [1117, 1381, 1663, 1951]
set_option maxRecDepth 2048 in
theorem smallRow967_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 967 = smallRow967_block_000 := by decide

def smallRow967_block_001 : List Nat := [2389, 2557, 2791, 3049, 3511, 3583]
set_option maxRecDepth 2048 in
theorem smallRow967_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 967 = smallRow967_block_001 := by decide

def smallRow967_block_002 : List Nat := [4507, 5011, 5623, 5779, 5857]
set_option maxRecDepth 2048 in
theorem smallRow967_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 967 = smallRow967_block_002 := by decide

def smallRow967_block_003 : List Nat := [6421, 6607, 6829, 6949, 6967, 8017]
set_option maxRecDepth 2048 in
theorem smallRow967_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 967 = smallRow967_block_003 := by decide

def smallRow967_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow967_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 967 = smallRow967_block_004 := by decide

def smallRow967_small_left : List Nat := smallRow967_block_000 ++ smallRow967_block_001
theorem smallRow967_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 967 = smallRow967_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 967 ++ upperNeighbors PrimePairVertices.tree_block_001 967 = _
  rw [smallRow967_block_000_checked, smallRow967_block_001_checked]
  rfl

def smallRow967_small_right_right : List Nat := smallRow967_block_003 ++ smallRow967_block_004
theorem smallRow967_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 967 = smallRow967_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 967 ++ upperNeighbors PrimePairVertices.tree_block_004 967 = _
  rw [smallRow967_block_003_checked, smallRow967_block_004_checked]
  rfl

def smallRow967_small_right : List Nat := smallRow967_block_002 ++ smallRow967_small_right_right
theorem smallRow967_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 967 = smallRow967_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 967 ++ upperNeighbors PrimePairVertices.tree_small_right_right 967 = _
  rw [smallRow967_block_002_checked, smallRow967_small_right_right_checked]
  rfl

def smallRow967_small : List Nat := smallRow967_small_left ++ smallRow967_small_right
theorem smallRow967_small_checked : upperNeighbors PrimePairVertices.tree_small 967 = smallRow967_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 967 ++ upperNeighbors PrimePairVertices.tree_small_right 967 = _
  rw [smallRow967_small_left_checked, smallRow967_small_right_checked]
  rfl

def smallRow967 : List Nat := smallRow967_small
theorem smallRow967_checked : upperNeighbors PrimePairVertices.smallTree 967 = smallRow967 := smallRow967_small_checked

def smallRow971_block_000 : List Nat := [977, 1019, 1511]
set_option maxRecDepth 2048 in
theorem smallRow971_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 971 = smallRow971_block_000 := by decide

def smallRow971_block_001 : List Nat := [2111, 2447, 2543, 2699, 2897, 3863, 3947]
set_option maxRecDepth 2048 in
theorem smallRow971_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 971 = smallRow971_block_001 := by decide

def smallRow971_block_002 : List Nat := [4421, 5009, 5309, 5477, 5501]
set_option maxRecDepth 2048 in
theorem smallRow971_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 971 = smallRow971_block_002 := by decide

def smallRow971_block_003 : List Nat := [6269, 6491, 6569, 6689, 6701, 7193, 7517, 7919, 8039, 8081, 8123, 8171]
set_option maxRecDepth 2048 in
theorem smallRow971_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 971 = smallRow971_block_003 := by decide

def smallRow971_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow971_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 971 = smallRow971_block_004 := by decide

def smallRow971_small_left : List Nat := smallRow971_block_000 ++ smallRow971_block_001
theorem smallRow971_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 971 = smallRow971_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 971 ++ upperNeighbors PrimePairVertices.tree_block_001 971 = _
  rw [smallRow971_block_000_checked, smallRow971_block_001_checked]
  rfl

def smallRow971_small_right_right : List Nat := smallRow971_block_003 ++ smallRow971_block_004
theorem smallRow971_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 971 = smallRow971_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 971 ++ upperNeighbors PrimePairVertices.tree_block_004 971 = _
  rw [smallRow971_block_003_checked, smallRow971_block_004_checked]
  rfl

def smallRow971_small_right : List Nat := smallRow971_block_002 ++ smallRow971_small_right_right
theorem smallRow971_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 971 = smallRow971_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 971 ++ upperNeighbors PrimePairVertices.tree_small_right_right 971 = _
  rw [smallRow971_block_002_checked, smallRow971_small_right_right_checked]
  rfl

def smallRow971_small : List Nat := smallRow971_small_left ++ smallRow971_small_right
theorem smallRow971_small_checked : upperNeighbors PrimePairVertices.tree_small 971 = smallRow971_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 971 ++ upperNeighbors PrimePairVertices.tree_small_right 971 = _
  rw [smallRow971_small_left_checked, smallRow971_small_right_checked]
  rfl

def smallRow971 : List Nat := smallRow971_small
theorem smallRow971_checked : upperNeighbors PrimePairVertices.smallTree 971 = smallRow971 := smallRow971_small_checked

def smallRow977_block_000 : List Nat := [1217, 1259, 1409, 2039]
set_option maxRecDepth 2048 in
theorem smallRow977_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 977 = smallRow977_block_000 := by decide

def smallRow977_block_001 : List Nat := [2129, 2141, 2753, 3557, 3593, 3767, 3851, 4049]
set_option maxRecDepth 2048 in
theorem smallRow977_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 977 = smallRow977_block_001 := by decide

def smallRow977_block_002 : List Nat := [5021, 5039]
set_option maxRecDepth 2048 in
theorem smallRow977_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 977 = smallRow977_block_002 := by decide

def smallRow977_block_003 : List Nat := [7853]
set_option maxRecDepth 2048 in
theorem smallRow977_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 977 = smallRow977_block_003 := by decide

def smallRow977_block_004 : List Nat := [8297]
set_option maxRecDepth 2048 in
theorem smallRow977_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 977 = smallRow977_block_004 := by decide

def smallRow977_small_left : List Nat := smallRow977_block_000 ++ smallRow977_block_001
theorem smallRow977_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 977 = smallRow977_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 977 ++ upperNeighbors PrimePairVertices.tree_block_001 977 = _
  rw [smallRow977_block_000_checked, smallRow977_block_001_checked]
  rfl

def smallRow977_small_right_right : List Nat := smallRow977_block_003 ++ smallRow977_block_004
theorem smallRow977_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 977 = smallRow977_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 977 ++ upperNeighbors PrimePairVertices.tree_block_004 977 = _
  rw [smallRow977_block_003_checked, smallRow977_block_004_checked]
  rfl

def smallRow977_small_right : List Nat := smallRow977_block_002 ++ smallRow977_small_right_right
theorem smallRow977_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 977 = smallRow977_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 977 ++ upperNeighbors PrimePairVertices.tree_small_right_right 977 = _
  rw [smallRow977_block_002_checked, smallRow977_small_right_right_checked]
  rfl

def smallRow977_small : List Nat := smallRow977_small_left ++ smallRow977_small_right
theorem smallRow977_small_checked : upperNeighbors PrimePairVertices.tree_small 977 = smallRow977_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 977 ++ upperNeighbors PrimePairVertices.tree_small_right 977 = _
  rw [smallRow977_small_left_checked, smallRow977_small_right_checked]
  rfl

def smallRow977 : List Nat := smallRow977_small
theorem smallRow977_checked : upperNeighbors PrimePairVertices.smallTree 977 = smallRow977 := smallRow977_small_checked

def smallRow983_block_000 : List Nat := [1091, 1259, 1637]
set_option maxRecDepth 2048 in
theorem smallRow983_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 983 = smallRow983_block_000 := by decide

def smallRow983_block_001 : List Nat := [2129, 2267, 2399, 3221, 3347, 3617, 4073]
set_option maxRecDepth 2048 in
theorem smallRow983_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 983 = smallRow983_block_001 := by decide

def smallRow983_block_002 : List Nat := [4871, 5171]
set_option maxRecDepth 2048 in
theorem smallRow983_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 983 = smallRow983_block_002 := by decide

def smallRow983_block_003 : List Nat := [6389, 6719, 6917, 7109, 7151, 7481]
set_option maxRecDepth 2048 in
theorem smallRow983_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 983 = smallRow983_block_003 := by decide

def smallRow983_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow983_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 983 = smallRow983_block_004 := by decide

def smallRow983_small_left : List Nat := smallRow983_block_000 ++ smallRow983_block_001
theorem smallRow983_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 983 = smallRow983_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 983 ++ upperNeighbors PrimePairVertices.tree_block_001 983 = _
  rw [smallRow983_block_000_checked, smallRow983_block_001_checked]
  rfl

def smallRow983_small_right_right : List Nat := smallRow983_block_003 ++ smallRow983_block_004
theorem smallRow983_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 983 = smallRow983_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 983 ++ upperNeighbors PrimePairVertices.tree_block_004 983 = _
  rw [smallRow983_block_003_checked, smallRow983_block_004_checked]
  rfl

def smallRow983_small_right : List Nat := smallRow983_block_002 ++ smallRow983_small_right_right
theorem smallRow983_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 983 = smallRow983_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 983 ++ upperNeighbors PrimePairVertices.tree_small_right_right 983 = _
  rw [smallRow983_block_002_checked, smallRow983_small_right_right_checked]
  rfl

def smallRow983_small : List Nat := smallRow983_small_left ++ smallRow983_small_right
theorem smallRow983_small_checked : upperNeighbors PrimePairVertices.tree_small 983 = smallRow983_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 983 ++ upperNeighbors PrimePairVertices.tree_small_right 983 = _
  rw [smallRow983_small_left_checked, smallRow983_small_right_checked]
  rfl

def smallRow983 : List Nat := smallRow983_small
theorem smallRow983_checked : upperNeighbors PrimePairVertices.smallTree 983 = smallRow983 := smallRow983_small_checked

def smallRow991_block_000 : List Nat := [1009, 1171, 1609, 1753, 1861, 2017, 2029]
set_option maxRecDepth 2048 in
theorem smallRow991_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 991 = smallRow991_block_000 := by decide

def smallRow991_block_001 : List Nat := [2269, 2689, 2719, 2803, 3361, 3571, 3613, 3727]
set_option maxRecDepth 2048 in
theorem smallRow991_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 991 = smallRow991_block_001 := by decide

def smallRow991_block_002 : List Nat := [4153, 5881, 6091]
set_option maxRecDepth 2048 in
theorem smallRow991_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 991 = smallRow991_block_002 := by decide

def smallRow991_block_003 : List Nat := [6163, 7027, 7549, 7759, 8017]
set_option maxRecDepth 2048 in
theorem smallRow991_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 991 = smallRow991_block_003 := by decide

def smallRow991_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow991_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 991 = smallRow991_block_004 := by decide

def smallRow991_small_left : List Nat := smallRow991_block_000 ++ smallRow991_block_001
theorem smallRow991_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 991 = smallRow991_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 991 ++ upperNeighbors PrimePairVertices.tree_block_001 991 = _
  rw [smallRow991_block_000_checked, smallRow991_block_001_checked]
  rfl

def smallRow991_small_right_right : List Nat := smallRow991_block_003 ++ smallRow991_block_004
theorem smallRow991_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 991 = smallRow991_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 991 ++ upperNeighbors PrimePairVertices.tree_block_004 991 = _
  rw [smallRow991_block_003_checked, smallRow991_block_004_checked]
  rfl

def smallRow991_small_right : List Nat := smallRow991_block_002 ++ smallRow991_small_right_right
theorem smallRow991_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 991 = smallRow991_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 991 ++ upperNeighbors PrimePairVertices.tree_small_right_right 991 = _
  rw [smallRow991_block_002_checked, smallRow991_small_right_right_checked]
  rfl

def smallRow991_small : List Nat := smallRow991_small_left ++ smallRow991_small_right
theorem smallRow991_small_checked : upperNeighbors PrimePairVertices.tree_small 991 = smallRow991_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 991 ++ upperNeighbors PrimePairVertices.tree_small_right 991 = _
  rw [smallRow991_small_left_checked, smallRow991_small_right_checked]
  rfl

def smallRow991 : List Nat := smallRow991_small
theorem smallRow991_checked : upperNeighbors PrimePairVertices.smallTree 991 = smallRow991 := smallRow991_small_checked

def smallRow997_block_000 : List Nat := [1483, 1993, 2029]
set_option maxRecDepth 2048 in
theorem smallRow997_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 997 = smallRow997_block_000 := by decide

def smallRow997_block_001 : List Nat := [2377, 2617, 3217, 3331]
set_option maxRecDepth 2048 in
theorem smallRow997_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 997 = smallRow997_block_001 := by decide

def smallRow997_block_002 : List Nat := [4177, 4243, 4513, 4969, 5569, 6091]
set_option maxRecDepth 2048 in
theorem smallRow997_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 997 = smallRow997_block_002 := by decide

def smallRow997_block_003 : List Nat := [6481, 6763, 7057, 7741]
set_option maxRecDepth 2048 in
theorem smallRow997_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 997 = smallRow997_block_003 := by decide

def smallRow997_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow997_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 997 = smallRow997_block_004 := by decide

def smallRow997_small_left : List Nat := smallRow997_block_000 ++ smallRow997_block_001
theorem smallRow997_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 997 = smallRow997_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 997 ++ upperNeighbors PrimePairVertices.tree_block_001 997 = _
  rw [smallRow997_block_000_checked, smallRow997_block_001_checked]
  rfl

def smallRow997_small_right_right : List Nat := smallRow997_block_003 ++ smallRow997_block_004
theorem smallRow997_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 997 = smallRow997_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 997 ++ upperNeighbors PrimePairVertices.tree_block_004 997 = _
  rw [smallRow997_block_003_checked, smallRow997_block_004_checked]
  rfl

def smallRow997_small_right : List Nat := smallRow997_block_002 ++ smallRow997_small_right_right
theorem smallRow997_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 997 = smallRow997_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 997 ++ upperNeighbors PrimePairVertices.tree_small_right_right 997 = _
  rw [smallRow997_block_002_checked, smallRow997_small_right_right_checked]
  rfl

def smallRow997_small : List Nat := smallRow997_small_left ++ smallRow997_small_right
theorem smallRow997_small_checked : upperNeighbors PrimePairVertices.tree_small 997 = smallRow997_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 997 ++ upperNeighbors PrimePairVertices.tree_small_right 997 = _
  rw [smallRow997_small_left_checked, smallRow997_small_right_checked]
  rfl

def smallRow997 : List Nat := smallRow997_small
theorem smallRow997_checked : upperNeighbors PrimePairVertices.smallTree 997 = smallRow997 := smallRow997_small_checked

def smallRow1009_block_000 : List Nat := [1789, 1933]
set_option maxRecDepth 2048 in
theorem smallRow1009_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1009 = smallRow1009_block_000 := by decide

def smallRow1009_block_001 : List Nat := [2221, 2473, 2851, 3613, 3823]
set_option maxRecDepth 2048 in
theorem smallRow1009_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1009 = smallRow1009_block_001 := by decide

def smallRow1009_block_002 : List Nat := [4099, 4657, 5653, 6043]
set_option maxRecDepth 2048 in
theorem smallRow1009_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1009 = smallRow1009_block_002 := by decide

def smallRow1009_block_003 : List Nat := [6367, 6397, 6481, 6577, 6673, 7027, 7309, 7393, 7753]
set_option maxRecDepth 2048 in
theorem smallRow1009_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1009 = smallRow1009_block_003 := by decide

def smallRow1009_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow1009_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1009 = smallRow1009_block_004 := by decide

def smallRow1009_small_left : List Nat := smallRow1009_block_000 ++ smallRow1009_block_001
theorem smallRow1009_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1009 = smallRow1009_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1009 ++ upperNeighbors PrimePairVertices.tree_block_001 1009 = _
  rw [smallRow1009_block_000_checked, smallRow1009_block_001_checked]
  rfl

def smallRow1009_small_right_right : List Nat := smallRow1009_block_003 ++ smallRow1009_block_004
theorem smallRow1009_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1009 = smallRow1009_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1009 ++ upperNeighbors PrimePairVertices.tree_block_004 1009 = _
  rw [smallRow1009_block_003_checked, smallRow1009_block_004_checked]
  rfl

def smallRow1009_small_right : List Nat := smallRow1009_block_002 ++ smallRow1009_small_right_right
theorem smallRow1009_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1009 = smallRow1009_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1009 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1009 = _
  rw [smallRow1009_block_002_checked, smallRow1009_small_right_right_checked]
  rfl

def smallRow1009_small : List Nat := smallRow1009_small_left ++ smallRow1009_small_right
theorem smallRow1009_small_checked : upperNeighbors PrimePairVertices.tree_small 1009 = smallRow1009_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1009 ++ upperNeighbors PrimePairVertices.tree_small_right 1009 = _
  rw [smallRow1009_small_left_checked, smallRow1009_small_right_checked]
  rfl

def smallRow1009 : List Nat := smallRow1009_small
theorem smallRow1009_checked : upperNeighbors PrimePairVertices.smallTree 1009 = smallRow1009 := smallRow1009_small_checked

def smallRow1013_block_000 : List Nat := [1091, 1097, 1697, 1721, 1811]
set_option maxRecDepth 2048 in
theorem smallRow1013_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 1013 = smallRow1013_block_000 := by decide

def smallRow1013_block_001 : List Nat := [2273, 2543, 3659, 3803]
set_option maxRecDepth 2048 in
theorem smallRow1013_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 1013 = smallRow1013_block_001 := by decide

def smallRow1013_block_002 : List Nat := [4133, 4139, 4259, 4349, 5039, 5087, 6047]
set_option maxRecDepth 2048 in
theorem smallRow1013_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 1013 = smallRow1013_block_002 := by decide

def smallRow1013_block_003 : List Nat := [6221, 6323, 6719, 6761, 7187, 7247, 7331, 7433, 7577, 7937]
set_option maxRecDepth 2048 in
theorem smallRow1013_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 1013 = smallRow1013_block_003 := by decide

def smallRow1013_block_004 : List Nat := [8231]
set_option maxRecDepth 2048 in
theorem smallRow1013_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 1013 = smallRow1013_block_004 := by decide

def smallRow1013_small_left : List Nat := smallRow1013_block_000 ++ smallRow1013_block_001
theorem smallRow1013_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 1013 = smallRow1013_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 1013 ++ upperNeighbors PrimePairVertices.tree_block_001 1013 = _
  rw [smallRow1013_block_000_checked, smallRow1013_block_001_checked]
  rfl

def smallRow1013_small_right_right : List Nat := smallRow1013_block_003 ++ smallRow1013_block_004
theorem smallRow1013_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 1013 = smallRow1013_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 1013 ++ upperNeighbors PrimePairVertices.tree_block_004 1013 = _
  rw [smallRow1013_block_003_checked, smallRow1013_block_004_checked]
  rfl

def smallRow1013_small_right : List Nat := smallRow1013_block_002 ++ smallRow1013_small_right_right
theorem smallRow1013_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 1013 = smallRow1013_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 1013 ++ upperNeighbors PrimePairVertices.tree_small_right_right 1013 = _
  rw [smallRow1013_block_002_checked, smallRow1013_small_right_right_checked]
  rfl

def smallRow1013_small : List Nat := smallRow1013_small_left ++ smallRow1013_small_right
theorem smallRow1013_small_checked : upperNeighbors PrimePairVertices.tree_small 1013 = smallRow1013_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 1013 ++ upperNeighbors PrimePairVertices.tree_small_right 1013 = _
  rw [smallRow1013_small_left_checked, smallRow1013_small_right_checked]
  rfl

def smallRow1013 : List Nat := smallRow1013_small
theorem smallRow1013_checked : upperNeighbors PrimePairVertices.smallTree 1013 = smallRow1013 := smallRow1013_small_checked

end ProjectEulerSolutions.PrimePairCache
