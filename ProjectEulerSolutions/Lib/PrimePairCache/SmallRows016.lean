import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow739_block_000 : List Nat := [751, 853, 967, 1093, 1123, 1381]
set_option maxRecDepth 2048 in
theorem smallRow739_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 739 = smallRow739_block_000 := by decide

def smallRow739_block_001 : List Nat := [3121, 3187, 3613, 3733]
set_option maxRecDepth 2048 in
theorem smallRow739_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 739 = smallRow739_block_001 := by decide

def smallRow739_block_002 : List Nat := [4357, 4549, 4759, 4909, 6007]
set_option maxRecDepth 2048 in
theorem smallRow739_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 739 = smallRow739_block_002 := by decide

def smallRow739_block_003 : List Nat := [6673, 6793, 6841, 7219, 7393, 7699]
set_option maxRecDepth 2048 in
theorem smallRow739_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 739 = smallRow739_block_003 := by decide

def smallRow739_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow739_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 739 = smallRow739_block_004 := by decide

def smallRow739_small_left : List Nat := smallRow739_block_000 ++ smallRow739_block_001
theorem smallRow739_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 739 = smallRow739_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 739 ++ upperNeighbors PrimePairVertices.tree_block_001 739 = _
  rw [smallRow739_block_000_checked, smallRow739_block_001_checked]
  rfl

def smallRow739_small_right_right : List Nat := smallRow739_block_003 ++ smallRow739_block_004
theorem smallRow739_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 739 = smallRow739_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 739 ++ upperNeighbors PrimePairVertices.tree_block_004 739 = _
  rw [smallRow739_block_003_checked, smallRow739_block_004_checked]
  rfl

def smallRow739_small_right : List Nat := smallRow739_block_002 ++ smallRow739_small_right_right
theorem smallRow739_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 739 = smallRow739_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 739 ++ upperNeighbors PrimePairVertices.tree_small_right_right 739 = _
  rw [smallRow739_block_002_checked, smallRow739_small_right_right_checked]
  rfl

def smallRow739_small : List Nat := smallRow739_small_left ++ smallRow739_small_right
theorem smallRow739_small_checked : upperNeighbors PrimePairVertices.tree_small 739 = smallRow739_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 739 ++ upperNeighbors PrimePairVertices.tree_small_right 739 = _
  rw [smallRow739_small_left_checked, smallRow739_small_right_checked]
  rfl

def smallRow739 : List Nat := smallRow739_small
theorem smallRow739_checked : upperNeighbors PrimePairVertices.smallTree 739 = smallRow739 := smallRow739_small_checked

def smallRow743_block_000 : List Nat := [881, 947, 1097, 1481, 1493, 1871, 1889]
set_option maxRecDepth 2048 in
theorem smallRow743_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 743 = smallRow743_block_000 := by decide

def smallRow743_block_001 : List Nat := [2099, 2273, 2309, 2753, 3359, 3527, 4001]
set_option maxRecDepth 2048 in
theorem smallRow743_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 743 = smallRow743_block_001 := by decide

def smallRow743_block_002 : List Nat := [4283, 5081, 5591, 6113]
set_option maxRecDepth 2048 in
theorem smallRow743_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 743 = smallRow743_block_002 := by decide

def smallRow743_block_003 : List Nat := [6323, 8117]
set_option maxRecDepth 2048 in
theorem smallRow743_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 743 = smallRow743_block_003 := by decide

def smallRow743_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow743_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 743 = smallRow743_block_004 := by decide

def smallRow743_small_left : List Nat := smallRow743_block_000 ++ smallRow743_block_001
theorem smallRow743_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 743 = smallRow743_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 743 ++ upperNeighbors PrimePairVertices.tree_block_001 743 = _
  rw [smallRow743_block_000_checked, smallRow743_block_001_checked]
  rfl

def smallRow743_small_right_right : List Nat := smallRow743_block_003 ++ smallRow743_block_004
theorem smallRow743_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 743 = smallRow743_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 743 ++ upperNeighbors PrimePairVertices.tree_block_004 743 = _
  rw [smallRow743_block_003_checked, smallRow743_block_004_checked]
  rfl

def smallRow743_small_right : List Nat := smallRow743_block_002 ++ smallRow743_small_right_right
theorem smallRow743_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 743 = smallRow743_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 743 ++ upperNeighbors PrimePairVertices.tree_small_right_right 743 = _
  rw [smallRow743_block_002_checked, smallRow743_small_right_right_checked]
  rfl

def smallRow743_small : List Nat := smallRow743_small_left ++ smallRow743_small_right
theorem smallRow743_small_checked : upperNeighbors PrimePairVertices.tree_small 743 = smallRow743_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 743 ++ upperNeighbors PrimePairVertices.tree_small_right 743 = _
  rw [smallRow743_small_left_checked, smallRow743_small_right_checked]
  rfl

def smallRow743 : List Nat := smallRow743_small
theorem smallRow743_checked : upperNeighbors PrimePairVertices.smallTree 743 = smallRow743 := smallRow743_small_checked

def smallRow751_block_000 : List Nat := [787, 997, 1033, 1789]
set_option maxRecDepth 2048 in
theorem smallRow751_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 751 = smallRow751_block_000 := by decide

def smallRow751_block_001 : List Nat := [2377, 2383, 2437, 2671, 3343, 3361, 3547]
set_option maxRecDepth 2048 in
theorem smallRow751_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 751 = smallRow751_block_001 := by decide

def smallRow751_block_002 : List Nat := [4231, 4597, 5281]
set_option maxRecDepth 2048 in
theorem smallRow751_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 751 = smallRow751_block_002 := by decide

def smallRow751_block_003 : List Nat := [7591, 7669]
set_option maxRecDepth 2048 in
theorem smallRow751_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 751 = smallRow751_block_003 := by decide

def smallRow751_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow751_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 751 = smallRow751_block_004 := by decide

def smallRow751_small_left : List Nat := smallRow751_block_000 ++ smallRow751_block_001
theorem smallRow751_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 751 = smallRow751_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 751 ++ upperNeighbors PrimePairVertices.tree_block_001 751 = _
  rw [smallRow751_block_000_checked, smallRow751_block_001_checked]
  rfl

def smallRow751_small_right_right : List Nat := smallRow751_block_003 ++ smallRow751_block_004
theorem smallRow751_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 751 = smallRow751_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 751 ++ upperNeighbors PrimePairVertices.tree_block_004 751 = _
  rw [smallRow751_block_003_checked, smallRow751_block_004_checked]
  rfl

def smallRow751_small_right : List Nat := smallRow751_block_002 ++ smallRow751_small_right_right
theorem smallRow751_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 751 = smallRow751_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 751 ++ upperNeighbors PrimePairVertices.tree_small_right_right 751 = _
  rw [smallRow751_block_002_checked, smallRow751_small_right_right_checked]
  rfl

def smallRow751_small : List Nat := smallRow751_small_left ++ smallRow751_small_right
theorem smallRow751_small_checked : upperNeighbors PrimePairVertices.tree_small 751 = smallRow751_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 751 ++ upperNeighbors PrimePairVertices.tree_small_right 751 = _
  rw [smallRow751_small_left_checked, smallRow751_small_right_checked]
  rfl

def smallRow751 : List Nat := smallRow751_small
theorem smallRow751_checked : upperNeighbors PrimePairVertices.smallTree 751 = smallRow751 := smallRow751_small_checked

def smallRow757_block_000 : List Nat := [811, 829, 1231, 1423]
set_option maxRecDepth 2048 in
theorem smallRow757_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 757 = smallRow757_block_000 := by decide

def smallRow757_block_001 : List Nat := [3037, 3217, 3733, 3931]
set_option maxRecDepth 2048 in
theorem smallRow757_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 757 = smallRow757_block_001 := by decide

def smallRow757_block_002 : List Nat := [4549, 4993, 5077, 5569, 5653, 5791, 6043]
set_option maxRecDepth 2048 in
theorem smallRow757_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 757 = smallRow757_block_002 := by decide

def smallRow757_block_003 : List Nat := [6211, 6619, 6661, 7321]
set_option maxRecDepth 2048 in
theorem smallRow757_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 757 = smallRow757_block_003 := by decide

def smallRow757_block_004 : List Nat := [8269]
set_option maxRecDepth 2048 in
theorem smallRow757_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 757 = smallRow757_block_004 := by decide

def smallRow757_small_left : List Nat := smallRow757_block_000 ++ smallRow757_block_001
theorem smallRow757_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 757 = smallRow757_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 757 ++ upperNeighbors PrimePairVertices.tree_block_001 757 = _
  rw [smallRow757_block_000_checked, smallRow757_block_001_checked]
  rfl

def smallRow757_small_right_right : List Nat := smallRow757_block_003 ++ smallRow757_block_004
theorem smallRow757_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 757 = smallRow757_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 757 ++ upperNeighbors PrimePairVertices.tree_block_004 757 = _
  rw [smallRow757_block_003_checked, smallRow757_block_004_checked]
  rfl

def smallRow757_small_right : List Nat := smallRow757_block_002 ++ smallRow757_small_right_right
theorem smallRow757_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 757 = smallRow757_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 757 ++ upperNeighbors PrimePairVertices.tree_small_right_right 757 = _
  rw [smallRow757_block_002_checked, smallRow757_small_right_right_checked]
  rfl

def smallRow757_small : List Nat := smallRow757_small_left ++ smallRow757_small_right
theorem smallRow757_small_checked : upperNeighbors PrimePairVertices.tree_small 757 = smallRow757_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 757 ++ upperNeighbors PrimePairVertices.tree_small_right 757 = _
  rw [smallRow757_small_left_checked, smallRow757_small_right_checked]
  rfl

def smallRow757 : List Nat := smallRow757_small
theorem smallRow757_checked : upperNeighbors PrimePairVertices.smallTree 757 = smallRow757 := smallRow757_small_checked

def smallRow761_block_000 : List Nat := [977, 1187, 1613, 1889]
set_option maxRecDepth 2048 in
theorem smallRow761_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 761 = smallRow761_block_000 := by decide

def smallRow761_block_001 : List Nat := [2351, 2531, 2621, 2663, 3209, 3251, 3329, 3539, 3779, 3923]
set_option maxRecDepth 2048 in
theorem smallRow761_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 761 = smallRow761_block_001 := by decide

def smallRow761_block_002 : List Nat := [4463, 4967, 5009, 5261, 5351, 5417, 5879]
set_option maxRecDepth 2048 in
theorem smallRow761_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 761 = smallRow761_block_002 := by decide

def smallRow761_block_003 : List Nat := [6911, 7283, 7529]
set_option maxRecDepth 2048 in
theorem smallRow761_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 761 = smallRow761_block_003 := by decide

def smallRow761_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow761_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 761 = smallRow761_block_004 := by decide

def smallRow761_small_left : List Nat := smallRow761_block_000 ++ smallRow761_block_001
theorem smallRow761_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 761 = smallRow761_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 761 ++ upperNeighbors PrimePairVertices.tree_block_001 761 = _
  rw [smallRow761_block_000_checked, smallRow761_block_001_checked]
  rfl

def smallRow761_small_right_right : List Nat := smallRow761_block_003 ++ smallRow761_block_004
theorem smallRow761_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 761 = smallRow761_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 761 ++ upperNeighbors PrimePairVertices.tree_block_004 761 = _
  rw [smallRow761_block_003_checked, smallRow761_block_004_checked]
  rfl

def smallRow761_small_right : List Nat := smallRow761_block_002 ++ smallRow761_small_right_right
theorem smallRow761_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 761 = smallRow761_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 761 ++ upperNeighbors PrimePairVertices.tree_small_right_right 761 = _
  rw [smallRow761_block_002_checked, smallRow761_small_right_right_checked]
  rfl

def smallRow761_small : List Nat := smallRow761_small_left ++ smallRow761_small_right
theorem smallRow761_small_checked : upperNeighbors PrimePairVertices.tree_small 761 = smallRow761_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 761 ++ upperNeighbors PrimePairVertices.tree_small_right 761 = _
  rw [smallRow761_small_left_checked, smallRow761_small_right_checked]
  rfl

def smallRow761 : List Nat := smallRow761_small
theorem smallRow761_checked : upperNeighbors PrimePairVertices.smallTree 761 = smallRow761 := smallRow761_small_checked

def smallRow769_block_000 : List Nat := [919, 997, 1381]
set_option maxRecDepth 2048 in
theorem smallRow769_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 769 = smallRow769_block_000 := by decide

def smallRow769_block_001 : List Nat := [2053, 2647, 3313, 4021]
set_option maxRecDepth 2048 in
theorem smallRow769_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 769 = smallRow769_block_001 := by decide

def smallRow769_block_002 : List Nat := [4651, 4987, 5113, 5197, 5227, 5503, 5791, 6133]
set_option maxRecDepth 2048 in
theorem smallRow769_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 769 = smallRow769_block_002 := by decide

def smallRow769_block_003 : List Nat := [6211, 6343, 6763, 6967, 7057, 7507]
set_option maxRecDepth 2048 in
theorem smallRow769_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 769 = smallRow769_block_003 := by decide

def smallRow769_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow769_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 769 = smallRow769_block_004 := by decide

def smallRow769_small_left : List Nat := smallRow769_block_000 ++ smallRow769_block_001
theorem smallRow769_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 769 = smallRow769_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 769 ++ upperNeighbors PrimePairVertices.tree_block_001 769 = _
  rw [smallRow769_block_000_checked, smallRow769_block_001_checked]
  rfl

def smallRow769_small_right_right : List Nat := smallRow769_block_003 ++ smallRow769_block_004
theorem smallRow769_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 769 = smallRow769_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 769 ++ upperNeighbors PrimePairVertices.tree_block_004 769 = _
  rw [smallRow769_block_003_checked, smallRow769_block_004_checked]
  rfl

def smallRow769_small_right : List Nat := smallRow769_block_002 ++ smallRow769_small_right_right
theorem smallRow769_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 769 = smallRow769_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 769 ++ upperNeighbors PrimePairVertices.tree_small_right_right 769 = _
  rw [smallRow769_block_002_checked, smallRow769_small_right_right_checked]
  rfl

def smallRow769_small : List Nat := smallRow769_small_left ++ smallRow769_small_right
theorem smallRow769_small_checked : upperNeighbors PrimePairVertices.tree_small 769 = smallRow769_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 769 ++ upperNeighbors PrimePairVertices.tree_small_right 769 = _
  rw [smallRow769_small_left_checked, smallRow769_small_right_checked]
  rfl

def smallRow769 : List Nat := smallRow769_small
theorem smallRow769_checked : upperNeighbors PrimePairVertices.smallTree 769 = smallRow769 := smallRow769_small_checked

def smallRow773_block_000 : List Nat := [953, 1439, 1667, 1721, 1877]
set_option maxRecDepth 2048 in
theorem smallRow773_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 773 = smallRow773_block_000 := by decide

def smallRow773_block_001 : List Nat := [2063, 2129, 2339, 2591, 3701]
set_option maxRecDepth 2048 in
theorem smallRow773_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 773 = smallRow773_block_001 := by decide

def smallRow773_block_002 : List Nat := [4241, 4409, 5081, 5381, 5669, 5711, 5927, 6101]
set_option maxRecDepth 2048 in
theorem smallRow773_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 773 = smallRow773_block_002 := by decide

def smallRow773_block_003 : List Nat := [6263, 6269, 7547]
set_option maxRecDepth 2048 in
theorem smallRow773_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 773 = smallRow773_block_003 := by decide

def smallRow773_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow773_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 773 = smallRow773_block_004 := by decide

def smallRow773_small_left : List Nat := smallRow773_block_000 ++ smallRow773_block_001
theorem smallRow773_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 773 = smallRow773_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 773 ++ upperNeighbors PrimePairVertices.tree_block_001 773 = _
  rw [smallRow773_block_000_checked, smallRow773_block_001_checked]
  rfl

def smallRow773_small_right_right : List Nat := smallRow773_block_003 ++ smallRow773_block_004
theorem smallRow773_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 773 = smallRow773_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 773 ++ upperNeighbors PrimePairVertices.tree_block_004 773 = _
  rw [smallRow773_block_003_checked, smallRow773_block_004_checked]
  rfl

def smallRow773_small_right : List Nat := smallRow773_block_002 ++ smallRow773_small_right_right
theorem smallRow773_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 773 = smallRow773_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 773 ++ upperNeighbors PrimePairVertices.tree_small_right_right 773 = _
  rw [smallRow773_block_002_checked, smallRow773_small_right_right_checked]
  rfl

def smallRow773_small : List Nat := smallRow773_small_left ++ smallRow773_small_right
theorem smallRow773_small_checked : upperNeighbors PrimePairVertices.tree_small 773 = smallRow773_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 773 ++ upperNeighbors PrimePairVertices.tree_small_right 773 = _
  rw [smallRow773_small_left_checked, smallRow773_small_right_checked]
  rfl

def smallRow773 : List Nat := smallRow773_small
theorem smallRow773_checked : upperNeighbors PrimePairVertices.smallTree 773 = smallRow773 := smallRow773_small_checked

def smallRow787_block_000 : List Nat := [823, 1483, 2017]
set_option maxRecDepth 2048 in
theorem smallRow787_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 787 = smallRow787_block_000 := by decide

def smallRow787_block_001 : List Nat := [2671, 2731, 3301, 3457, 3631]
set_option maxRecDepth 2048 in
theorem smallRow787_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 787 = smallRow787_block_001 := by decide

def smallRow787_block_002 : List Nat := [4129, 4177, 4507, 5347, 5653, 6007]
set_option maxRecDepth 2048 in
theorem smallRow787_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 787 = smallRow787_block_002 := by decide

def smallRow787_block_003 : List Nat := [6367, 6733, 7159, 7873, 7993]
set_option maxRecDepth 2048 in
theorem smallRow787_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 787 = smallRow787_block_003 := by decide

def smallRow787_block_004 : List Nat := [8317]
set_option maxRecDepth 2048 in
theorem smallRow787_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 787 = smallRow787_block_004 := by decide

def smallRow787_small_left : List Nat := smallRow787_block_000 ++ smallRow787_block_001
theorem smallRow787_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 787 = smallRow787_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 787 ++ upperNeighbors PrimePairVertices.tree_block_001 787 = _
  rw [smallRow787_block_000_checked, smallRow787_block_001_checked]
  rfl

def smallRow787_small_right_right : List Nat := smallRow787_block_003 ++ smallRow787_block_004
theorem smallRow787_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 787 = smallRow787_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 787 ++ upperNeighbors PrimePairVertices.tree_block_004 787 = _
  rw [smallRow787_block_003_checked, smallRow787_block_004_checked]
  rfl

def smallRow787_small_right : List Nat := smallRow787_block_002 ++ smallRow787_small_right_right
theorem smallRow787_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 787 = smallRow787_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 787 ++ upperNeighbors PrimePairVertices.tree_small_right_right 787 = _
  rw [smallRow787_block_002_checked, smallRow787_small_right_right_checked]
  rfl

def smallRow787_small : List Nat := smallRow787_small_left ++ smallRow787_small_right
theorem smallRow787_small_checked : upperNeighbors PrimePairVertices.tree_small 787 = smallRow787_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 787 ++ upperNeighbors PrimePairVertices.tree_small_right 787 = _
  rw [smallRow787_small_left_checked, smallRow787_small_right_checked]
  rfl

def smallRow787 : List Nat := smallRow787_small
theorem smallRow787_checked : upperNeighbors PrimePairVertices.smallTree 787 = smallRow787 := smallRow787_small_checked

end ProjectEulerSolutions.PrimePairCache
