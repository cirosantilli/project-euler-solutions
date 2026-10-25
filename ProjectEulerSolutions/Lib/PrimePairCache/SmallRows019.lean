import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow907_block_000 : List Nat := [1213, 1429, 1459, 1471, 1801]
set_option maxRecDepth 2048 in
theorem smallRow907_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 907 = smallRow907_block_000 := by decide

def smallRow907_block_001 : List Nat := [2383, 2473, 2731, 3709, 3919]
set_option maxRecDepth 2048 in
theorem smallRow907_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 907 = smallRow907_block_001 := by decide

def smallRow907_block_002 : List Nat := [4357, 4423, 4567, 4861, 4987, 4993, 5503, 5851]
set_option maxRecDepth 2048 in
theorem smallRow907_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 907 = smallRow907_block_002 := by decide

def smallRow907_block_003 : List Nat := [6229, 6337, 6733, 6841, 6961, 7027, 7591, 7723, 7993]
set_option maxRecDepth 2048 in
theorem smallRow907_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 907 = smallRow907_block_003 := by decide

def smallRow907_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow907_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 907 = smallRow907_block_004 := by decide

def smallRow907_small_left : List Nat := smallRow907_block_000 ++ smallRow907_block_001
theorem smallRow907_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 907 = smallRow907_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 907 ++ upperNeighbors PrimePairVertices.tree_block_001 907 = _
  rw [smallRow907_block_000_checked, smallRow907_block_001_checked]
  rfl

def smallRow907_small_right_right : List Nat := smallRow907_block_003 ++ smallRow907_block_004
theorem smallRow907_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 907 = smallRow907_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 907 ++ upperNeighbors PrimePairVertices.tree_block_004 907 = _
  rw [smallRow907_block_003_checked, smallRow907_block_004_checked]
  rfl

def smallRow907_small_right : List Nat := smallRow907_block_002 ++ smallRow907_small_right_right
theorem smallRow907_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 907 = smallRow907_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 907 ++ upperNeighbors PrimePairVertices.tree_small_right_right 907 = _
  rw [smallRow907_block_002_checked, smallRow907_small_right_right_checked]
  rfl

def smallRow907_small : List Nat := smallRow907_small_left ++ smallRow907_small_right
theorem smallRow907_small_checked : upperNeighbors PrimePairVertices.tree_small 907 = smallRow907_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 907 ++ upperNeighbors PrimePairVertices.tree_small_right 907 = _
  rw [smallRow907_small_left_checked, smallRow907_small_right_checked]
  rfl

def smallRow907 : List Nat := smallRow907_small
theorem smallRow907_checked : upperNeighbors PrimePairVertices.smallTree 907 = smallRow907 := smallRow907_small_checked

def smallRow911_block_000 : List Nat := [947, 1103, 1229, 1511]
set_option maxRecDepth 2048 in
theorem smallRow911_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 911 = smallRow911_block_000 := by decide

def smallRow911_block_001 : List Nat := [2309, 2399, 2657, 3413, 3821]
set_option maxRecDepth 2048 in
theorem smallRow911_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 911 = smallRow911_block_001 := by decide

def smallRow911_block_002 : List Nat := [4337, 4517, 4583, 4643, 5303, 5471, 6089, 6113]
set_option maxRecDepth 2048 in
theorem smallRow911_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 911 = smallRow911_block_002 := by decide

def smallRow911_block_003 : List Nat := [6269, 6971, 7109, 8111]
set_option maxRecDepth 2048 in
theorem smallRow911_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 911 = smallRow911_block_003 := by decide

def smallRow911_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow911_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 911 = smallRow911_block_004 := by decide

def smallRow911_small_left : List Nat := smallRow911_block_000 ++ smallRow911_block_001
theorem smallRow911_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 911 = smallRow911_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 911 ++ upperNeighbors PrimePairVertices.tree_block_001 911 = _
  rw [smallRow911_block_000_checked, smallRow911_block_001_checked]
  rfl

def smallRow911_small_right_right : List Nat := smallRow911_block_003 ++ smallRow911_block_004
theorem smallRow911_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 911 = smallRow911_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 911 ++ upperNeighbors PrimePairVertices.tree_block_004 911 = _
  rw [smallRow911_block_003_checked, smallRow911_block_004_checked]
  rfl

def smallRow911_small_right : List Nat := smallRow911_block_002 ++ smallRow911_small_right_right
theorem smallRow911_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 911 = smallRow911_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 911 ++ upperNeighbors PrimePairVertices.tree_small_right_right 911 = _
  rw [smallRow911_block_002_checked, smallRow911_small_right_right_checked]
  rfl

def smallRow911_small : List Nat := smallRow911_small_left ++ smallRow911_small_right
theorem smallRow911_small_checked : upperNeighbors PrimePairVertices.tree_small 911 = smallRow911_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 911 ++ upperNeighbors PrimePairVertices.tree_small_right 911 = _
  rw [smallRow911_small_left_checked, smallRow911_small_right_checked]
  rfl

def smallRow911 : List Nat := smallRow911_small
theorem smallRow911_checked : upperNeighbors PrimePairVertices.smallTree 911 = smallRow911 := smallRow911_small_checked

def smallRow919_block_000 : List Nat := [937, 1069, 1201, 1279, 1627, 2017]
set_option maxRecDepth 2048 in
theorem smallRow919_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 919 = smallRow919_block_000 := by decide

def smallRow919_block_001 : List Nat := [2749, 3463, 3643]
set_option maxRecDepth 2048 in
theorem smallRow919_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 919 = smallRow919_block_001 := by decide

def smallRow919_block_002 : List Nat := [4231, 4441, 4567, 4597, 5167, 5821, 5839, 5881]
set_option maxRecDepth 2048 in
theorem smallRow919_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 919 = smallRow919_block_002 := by decide

def smallRow919_block_003 : List Nat := [6679, 6961, 7039, 7207]
set_option maxRecDepth 2048 in
theorem smallRow919_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 919 = smallRow919_block_003 := by decide

def smallRow919_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow919_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 919 = smallRow919_block_004 := by decide

def smallRow919_small_left : List Nat := smallRow919_block_000 ++ smallRow919_block_001
theorem smallRow919_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 919 = smallRow919_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 919 ++ upperNeighbors PrimePairVertices.tree_block_001 919 = _
  rw [smallRow919_block_000_checked, smallRow919_block_001_checked]
  rfl

def smallRow919_small_right_right : List Nat := smallRow919_block_003 ++ smallRow919_block_004
theorem smallRow919_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 919 = smallRow919_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 919 ++ upperNeighbors PrimePairVertices.tree_block_004 919 = _
  rw [smallRow919_block_003_checked, smallRow919_block_004_checked]
  rfl

def smallRow919_small_right : List Nat := smallRow919_block_002 ++ smallRow919_small_right_right
theorem smallRow919_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 919 = smallRow919_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 919 ++ upperNeighbors PrimePairVertices.tree_small_right_right 919 = _
  rw [smallRow919_block_002_checked, smallRow919_small_right_right_checked]
  rfl

def smallRow919_small : List Nat := smallRow919_small_left ++ smallRow919_small_right
theorem smallRow919_small_checked : upperNeighbors PrimePairVertices.tree_small 919 = smallRow919_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 919 ++ upperNeighbors PrimePairVertices.tree_small_right 919 = _
  rw [smallRow919_small_left_checked, smallRow919_small_right_checked]
  rfl

def smallRow919 : List Nat := smallRow919_small
theorem smallRow919_checked : upperNeighbors PrimePairVertices.smallTree 919 = smallRow919 := smallRow919_small_checked

def smallRow929_block_000 : List Nat := [941, 953, 983, 2039]
set_option maxRecDepth 2048 in
theorem smallRow929_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 929 = smallRow929_block_000 := by decide

def smallRow929_block_001 : List Nat := [2477, 2741, 3209, 3467, 3539, 3761, 4013, 4073]
set_option maxRecDepth 2048 in
theorem smallRow929_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 929 = smallRow929_block_001 := by decide

def smallRow929_block_002 : List Nat := [4421, 4463, 4973, 5639, 5669]
set_option maxRecDepth 2048 in
theorem smallRow929_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 929 = smallRow929_block_002 := by decide

def smallRow929_block_003 : List Nat := [6563, 7109, 7853, 8117]
set_option maxRecDepth 2048 in
theorem smallRow929_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 929 = smallRow929_block_003 := by decide

def smallRow929_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow929_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 929 = smallRow929_block_004 := by decide

def smallRow929_small_left : List Nat := smallRow929_block_000 ++ smallRow929_block_001
theorem smallRow929_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 929 = smallRow929_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 929 ++ upperNeighbors PrimePairVertices.tree_block_001 929 = _
  rw [smallRow929_block_000_checked, smallRow929_block_001_checked]
  rfl

def smallRow929_small_right_right : List Nat := smallRow929_block_003 ++ smallRow929_block_004
theorem smallRow929_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 929 = smallRow929_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 929 ++ upperNeighbors PrimePairVertices.tree_block_004 929 = _
  rw [smallRow929_block_003_checked, smallRow929_block_004_checked]
  rfl

def smallRow929_small_right : List Nat := smallRow929_block_002 ++ smallRow929_small_right_right
theorem smallRow929_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 929 = smallRow929_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 929 ++ upperNeighbors PrimePairVertices.tree_small_right_right 929 = _
  rw [smallRow929_block_002_checked, smallRow929_small_right_right_checked]
  rfl

def smallRow929_small : List Nat := smallRow929_small_left ++ smallRow929_small_right
theorem smallRow929_small_checked : upperNeighbors PrimePairVertices.tree_small 929 = smallRow929_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 929 ++ upperNeighbors PrimePairVertices.tree_small_right 929 = _
  rw [smallRow929_small_left_checked, smallRow929_small_right_checked]
  rfl

def smallRow929 : List Nat := smallRow929_small
theorem smallRow929_checked : upperNeighbors PrimePairVertices.smallTree 929 = smallRow929 := smallRow929_small_checked

def smallRow937_block_000 : List Nat := [1489, 1783]
set_option maxRecDepth 2048 in
theorem smallRow937_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 937 = smallRow937_block_000 := by decide

def smallRow937_block_001 : List Nat := [2053, 2131, 2557, 2791, 3109, 3319, 3853, 3943, 4093]
set_option maxRecDepth 2048 in
theorem smallRow937_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 937 = smallRow937_block_001 := by decide

def smallRow937_block_002 : List Nat := [4159, 4273, 5059, 5281, 5407, 5437, 5791, 5827]
set_option maxRecDepth 2048 in
theorem smallRow937_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 937 = smallRow937_block_002 := by decide

def smallRow937_block_003 : List Nat := [7213, 8059]
set_option maxRecDepth 2048 in
theorem smallRow937_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 937 = smallRow937_block_003 := by decide

def smallRow937_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow937_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 937 = smallRow937_block_004 := by decide

def smallRow937_small_left : List Nat := smallRow937_block_000 ++ smallRow937_block_001
theorem smallRow937_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 937 = smallRow937_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 937 ++ upperNeighbors PrimePairVertices.tree_block_001 937 = _
  rw [smallRow937_block_000_checked, smallRow937_block_001_checked]
  rfl

def smallRow937_small_right_right : List Nat := smallRow937_block_003 ++ smallRow937_block_004
theorem smallRow937_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 937 = smallRow937_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 937 ++ upperNeighbors PrimePairVertices.tree_block_004 937 = _
  rw [smallRow937_block_003_checked, smallRow937_block_004_checked]
  rfl

def smallRow937_small_right : List Nat := smallRow937_block_002 ++ smallRow937_small_right_right
theorem smallRow937_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 937 = smallRow937_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 937 ++ upperNeighbors PrimePairVertices.tree_small_right_right 937 = _
  rw [smallRow937_block_002_checked, smallRow937_small_right_right_checked]
  rfl

def smallRow937_small : List Nat := smallRow937_small_left ++ smallRow937_small_right
theorem smallRow937_small_checked : upperNeighbors PrimePairVertices.tree_small 937 = smallRow937_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 937 ++ upperNeighbors PrimePairVertices.tree_small_right 937 = _
  rw [smallRow937_small_left_checked, smallRow937_small_right_checked]
  rfl

def smallRow937 : List Nat := smallRow937_small
theorem smallRow937_checked : upperNeighbors PrimePairVertices.smallTree 937 = smallRow937 := smallRow937_small_checked

def smallRow941_block_000 : List Nat := [1049, 1187, 1229, 1511, 1607, 1709]
set_option maxRecDepth 2048 in
theorem smallRow941_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 941 = smallRow941_block_000 := by decide

def smallRow941_block_001 : List Nat := [2099, 3137, 3221, 3389, 3527]
set_option maxRecDepth 2048 in
theorem smallRow941_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 941 = smallRow941_block_001 := by decide

def smallRow941_block_002 : List Nat := [4409, 4649, 4817, 6047, 6053]
set_option maxRecDepth 2048 in
theorem smallRow941_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 941 = smallRow941_block_002 := by decide

def smallRow941_block_003 : List Nat := [6263, 6299, 6971, 7229]
set_option maxRecDepth 2048 in
theorem smallRow941_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 941 = smallRow941_block_003 := by decide

def smallRow941_block_004 : List Nat := [8369]
set_option maxRecDepth 2048 in
theorem smallRow941_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 941 = smallRow941_block_004 := by decide

def smallRow941_small_left : List Nat := smallRow941_block_000 ++ smallRow941_block_001
theorem smallRow941_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 941 = smallRow941_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 941 ++ upperNeighbors PrimePairVertices.tree_block_001 941 = _
  rw [smallRow941_block_000_checked, smallRow941_block_001_checked]
  rfl

def smallRow941_small_right_right : List Nat := smallRow941_block_003 ++ smallRow941_block_004
theorem smallRow941_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 941 = smallRow941_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 941 ++ upperNeighbors PrimePairVertices.tree_block_004 941 = _
  rw [smallRow941_block_003_checked, smallRow941_block_004_checked]
  rfl

def smallRow941_small_right : List Nat := smallRow941_block_002 ++ smallRow941_small_right_right
theorem smallRow941_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 941 = smallRow941_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 941 ++ upperNeighbors PrimePairVertices.tree_small_right_right 941 = _
  rw [smallRow941_block_002_checked, smallRow941_small_right_right_checked]
  rfl

def smallRow941_small : List Nat := smallRow941_small_left ++ smallRow941_small_right
theorem smallRow941_small_checked : upperNeighbors PrimePairVertices.tree_small 941 = smallRow941_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 941 ++ upperNeighbors PrimePairVertices.tree_small_right 941 = _
  rw [smallRow941_small_left_checked, smallRow941_small_right_checked]
  rfl

def smallRow941 : List Nat := smallRow941_small
theorem smallRow941_checked : upperNeighbors PrimePairVertices.smallTree 941 = smallRow941 := smallRow941_small_checked

def smallRow947_block_000 : List Nat := [1097, 1439, 1901, 1949]
set_option maxRecDepth 2048 in
theorem smallRow947_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 947 = smallRow947_block_000 := by decide

def smallRow947_block_001 : List Nat := [2657, 3041, 3491, 3623]
set_option maxRecDepth 2048 in
theorem smallRow947_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 947 = smallRow947_block_001 := by decide

def smallRow947_block_002 : List Nat := [4523, 4637, 4931, 5009, 5051, 5981]
set_option maxRecDepth 2048 in
theorem smallRow947_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 947 = smallRow947_block_002 := by decide

def smallRow947_block_003 : List Nat := [6947, 8111]
set_option maxRecDepth 2048 in
theorem smallRow947_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 947 = smallRow947_block_003 := by decide

def smallRow947_block_004 : List Nat := [8387]
set_option maxRecDepth 2048 in
theorem smallRow947_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 947 = smallRow947_block_004 := by decide

def smallRow947_small_left : List Nat := smallRow947_block_000 ++ smallRow947_block_001
theorem smallRow947_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 947 = smallRow947_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 947 ++ upperNeighbors PrimePairVertices.tree_block_001 947 = _
  rw [smallRow947_block_000_checked, smallRow947_block_001_checked]
  rfl

def smallRow947_small_right_right : List Nat := smallRow947_block_003 ++ smallRow947_block_004
theorem smallRow947_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 947 = smallRow947_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 947 ++ upperNeighbors PrimePairVertices.tree_block_004 947 = _
  rw [smallRow947_block_003_checked, smallRow947_block_004_checked]
  rfl

def smallRow947_small_right : List Nat := smallRow947_block_002 ++ smallRow947_small_right_right
theorem smallRow947_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 947 = smallRow947_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 947 ++ upperNeighbors PrimePairVertices.tree_small_right_right 947 = _
  rw [smallRow947_block_002_checked, smallRow947_small_right_right_checked]
  rfl

def smallRow947_small : List Nat := smallRow947_small_left ++ smallRow947_small_right
theorem smallRow947_small_checked : upperNeighbors PrimePairVertices.tree_small 947 = smallRow947_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 947 ++ upperNeighbors PrimePairVertices.tree_small_right 947 = _
  rw [smallRow947_small_left_checked, smallRow947_small_right_checked]
  rfl

def smallRow947 : List Nat := smallRow947_small
theorem smallRow947_checked : upperNeighbors PrimePairVertices.smallTree 947 = smallRow947 := smallRow947_small_checked

def smallRow953_block_000 : List Nat := [1367, 1523, 2027]
set_option maxRecDepth 2048 in
theorem smallRow953_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 953 = smallRow953_block_000 := by decide

def smallRow953_block_001 : List Nat := [2333, 2609, 2687, 3533]
set_option maxRecDepth 2048 in
theorem smallRow953_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 953 = smallRow953_block_001 := by decide

def smallRow953_block_002 : List Nat := [4337, 4799, 4871, 5039, 5417, 5651, 6029]
set_option maxRecDepth 2048 in
theorem smallRow953_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 953 = smallRow953_block_002 := by decide

def smallRow953_block_003 : List Nat := [7109, 7457, 7727]
set_option maxRecDepth 2048 in
theorem smallRow953_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 953 = smallRow953_block_003 := by decide

def smallRow953_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow953_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 953 = smallRow953_block_004 := by decide

def smallRow953_small_left : List Nat := smallRow953_block_000 ++ smallRow953_block_001
theorem smallRow953_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 953 = smallRow953_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 953 ++ upperNeighbors PrimePairVertices.tree_block_001 953 = _
  rw [smallRow953_block_000_checked, smallRow953_block_001_checked]
  rfl

def smallRow953_small_right_right : List Nat := smallRow953_block_003 ++ smallRow953_block_004
theorem smallRow953_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 953 = smallRow953_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 953 ++ upperNeighbors PrimePairVertices.tree_block_004 953 = _
  rw [smallRow953_block_003_checked, smallRow953_block_004_checked]
  rfl

def smallRow953_small_right : List Nat := smallRow953_block_002 ++ smallRow953_small_right_right
theorem smallRow953_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 953 = smallRow953_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 953 ++ upperNeighbors PrimePairVertices.tree_small_right_right 953 = _
  rw [smallRow953_block_002_checked, smallRow953_small_right_right_checked]
  rfl

def smallRow953_small : List Nat := smallRow953_small_left ++ smallRow953_small_right
theorem smallRow953_small_checked : upperNeighbors PrimePairVertices.tree_small 953 = smallRow953_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 953 ++ upperNeighbors PrimePairVertices.tree_small_right 953 = _
  rw [smallRow953_small_left_checked, smallRow953_small_right_checked]
  rfl

def smallRow953 : List Nat := smallRow953_small
theorem smallRow953_checked : upperNeighbors PrimePairVertices.smallTree 953 = smallRow953 := smallRow953_small_checked

end ProjectEulerSolutions.PrimePairCache
