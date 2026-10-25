import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow853_block_000 : List Nat := [1129, 1153, 1279, 1801]
set_option maxRecDepth 2048 in
theorem smallRow853_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 853 = smallRow853_block_000 := by decide

def smallRow853_block_001 : List Nat := [2269, 2593, 2707, 2917, 3001, 3697, 3853]
set_option maxRecDepth 2048 in
theorem smallRow853_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 853 = smallRow853_block_001 := by decide

def smallRow853_block_002 : List Nat := [4261, 4447, 4621, 4639, 4657, 4789, 5413]
set_option maxRecDepth 2048 in
theorem smallRow853_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 853 = smallRow853_block_002 := by decide

def smallRow853_block_003 : List Nat := [6379, 6637, 6961, 7603, 7759, 8017]
set_option maxRecDepth 2048 in
theorem smallRow853_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 853 = smallRow853_block_003 := by decide

def smallRow853_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow853_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 853 = smallRow853_block_004 := by decide

def smallRow853_small_left : List Nat := smallRow853_block_000 ++ smallRow853_block_001
theorem smallRow853_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 853 = smallRow853_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 853 ++ upperNeighbors PrimePairVertices.tree_block_001 853 = _
  rw [smallRow853_block_000_checked, smallRow853_block_001_checked]
  rfl

def smallRow853_small_right_right : List Nat := smallRow853_block_003 ++ smallRow853_block_004
theorem smallRow853_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 853 = smallRow853_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 853 ++ upperNeighbors PrimePairVertices.tree_block_004 853 = _
  rw [smallRow853_block_003_checked, smallRow853_block_004_checked]
  rfl

def smallRow853_small_right : List Nat := smallRow853_block_002 ++ smallRow853_small_right_right
theorem smallRow853_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 853 = smallRow853_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 853 ++ upperNeighbors PrimePairVertices.tree_small_right_right 853 = _
  rw [smallRow853_block_002_checked, smallRow853_small_right_right_checked]
  rfl

def smallRow853_small : List Nat := smallRow853_small_left ++ smallRow853_small_right
theorem smallRow853_small_checked : upperNeighbors PrimePairVertices.tree_small 853 = smallRow853_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 853 ++ upperNeighbors PrimePairVertices.tree_small_right 853 = _
  rw [smallRow853_small_left_checked, smallRow853_small_right_checked]
  rfl

def smallRow853 : List Nat := smallRow853_small
theorem smallRow853_checked : upperNeighbors PrimePairVertices.smallTree 853 = smallRow853 := smallRow853_small_checked

def smallRow857_block_000 : List Nat := [1301, 1601, 1973]
set_option maxRecDepth 2048 in
theorem smallRow857_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 857 = smallRow857_block_000 := by decide

def smallRow857_block_001 : List Nat := [2633, 2693, 3413, 4079]
set_option maxRecDepth 2048 in
theorem smallRow857_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 857 = smallRow857_block_001 := by decide

def smallRow857_block_002 : List Nat := [4463, 4877, 6131]
set_option maxRecDepth 2048 in
theorem smallRow857_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 857 = smallRow857_block_002 := by decide

def smallRow857_block_003 : List Nat := [6329, 7187, 7307, 7529, 8147]
set_option maxRecDepth 2048 in
theorem smallRow857_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 857 = smallRow857_block_003 := by decide

def smallRow857_block_004 : List Nat := [8243]
set_option maxRecDepth 2048 in
theorem smallRow857_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 857 = smallRow857_block_004 := by decide

def smallRow857_small_left : List Nat := smallRow857_block_000 ++ smallRow857_block_001
theorem smallRow857_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 857 = smallRow857_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 857 ++ upperNeighbors PrimePairVertices.tree_block_001 857 = _
  rw [smallRow857_block_000_checked, smallRow857_block_001_checked]
  rfl

def smallRow857_small_right_right : List Nat := smallRow857_block_003 ++ smallRow857_block_004
theorem smallRow857_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 857 = smallRow857_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 857 ++ upperNeighbors PrimePairVertices.tree_block_004 857 = _
  rw [smallRow857_block_003_checked, smallRow857_block_004_checked]
  rfl

def smallRow857_small_right : List Nat := smallRow857_block_002 ++ smallRow857_small_right_right
theorem smallRow857_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 857 = smallRow857_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 857 ++ upperNeighbors PrimePairVertices.tree_small_right_right 857 = _
  rw [smallRow857_block_002_checked, smallRow857_small_right_right_checked]
  rfl

def smallRow857_small : List Nat := smallRow857_small_left ++ smallRow857_small_right
theorem smallRow857_small_checked : upperNeighbors PrimePairVertices.tree_small 857 = smallRow857_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 857 ++ upperNeighbors PrimePairVertices.tree_small_right 857 = _
  rw [smallRow857_small_left_checked, smallRow857_small_right_checked]
  rfl

def smallRow857 : List Nat := smallRow857_small
theorem smallRow857_checked : upperNeighbors PrimePairVertices.smallTree 857 = smallRow857 := smallRow857_small_checked

def smallRow859_block_000 : List Nat := [919, 1129, 1381, 1543]
set_option maxRecDepth 2048 in
theorem smallRow859_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 859 = smallRow859_block_000 := by decide

def smallRow859_block_001 : List Nat := [2659, 3307, 3391, 3847, 3943]
set_option maxRecDepth 2048 in
theorem smallRow859_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 859 = smallRow859_block_001 := by decide

def smallRow859_block_002 : List Nat := [4507, 4549, 5023, 5623, 5683]
set_option maxRecDepth 2048 in
theorem smallRow859_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 859 = smallRow859_block_002 := by decide

def smallRow859_block_003 : List Nat := [6199, 7213, 7219, 7417, 7639, 7669, 7759, 8101]
set_option maxRecDepth 2048 in
theorem smallRow859_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 859 = smallRow859_block_003 := by decide

def smallRow859_block_004 : List Nat := [8221, 8353]
set_option maxRecDepth 2048 in
theorem smallRow859_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 859 = smallRow859_block_004 := by decide

def smallRow859_small_left : List Nat := smallRow859_block_000 ++ smallRow859_block_001
theorem smallRow859_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 859 = smallRow859_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 859 ++ upperNeighbors PrimePairVertices.tree_block_001 859 = _
  rw [smallRow859_block_000_checked, smallRow859_block_001_checked]
  rfl

def smallRow859_small_right_right : List Nat := smallRow859_block_003 ++ smallRow859_block_004
theorem smallRow859_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 859 = smallRow859_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 859 ++ upperNeighbors PrimePairVertices.tree_block_004 859 = _
  rw [smallRow859_block_003_checked, smallRow859_block_004_checked]
  rfl

def smallRow859_small_right : List Nat := smallRow859_block_002 ++ smallRow859_small_right_right
theorem smallRow859_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 859 = smallRow859_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 859 ++ upperNeighbors PrimePairVertices.tree_small_right_right 859 = _
  rw [smallRow859_block_002_checked, smallRow859_small_right_right_checked]
  rfl

def smallRow859_small : List Nat := smallRow859_small_left ++ smallRow859_small_right
theorem smallRow859_small_checked : upperNeighbors PrimePairVertices.tree_small 859 = smallRow859_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 859 ++ upperNeighbors PrimePairVertices.tree_small_right 859 = _
  rw [smallRow859_small_left_checked, smallRow859_small_right_checked]
  rfl

def smallRow859 : List Nat := smallRow859_small
theorem smallRow859_checked : upperNeighbors PrimePairVertices.smallTree 859 = smallRow859 := smallRow859_small_checked

def smallRow863_block_000 : List Nat := [983, 1277, 1307, 1823, 1907, 2003]
set_option maxRecDepth 2048 in
theorem smallRow863_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 863 = smallRow863_block_000 := by decide

def smallRow863_block_001 : List Nat := [2297, 2357, 2621, 2801, 2927, 3137]
set_option maxRecDepth 2048 in
theorem smallRow863_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 863 = smallRow863_block_001 := by decide

def smallRow863_block_002 : List Nat := [4253, 4259, 4397, 4421, 5381, 5717, 5849]
set_option maxRecDepth 2048 in
theorem smallRow863_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 863 = smallRow863_block_002 := by decide

def smallRow863_block_003 : List Nat := [6779, 6899, 6959, 7523]
set_option maxRecDepth 2048 in
theorem smallRow863_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 863 = smallRow863_block_003 := by decide

def smallRow863_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow863_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 863 = smallRow863_block_004 := by decide

def smallRow863_small_left : List Nat := smallRow863_block_000 ++ smallRow863_block_001
theorem smallRow863_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 863 = smallRow863_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 863 ++ upperNeighbors PrimePairVertices.tree_block_001 863 = _
  rw [smallRow863_block_000_checked, smallRow863_block_001_checked]
  rfl

def smallRow863_small_right_right : List Nat := smallRow863_block_003 ++ smallRow863_block_004
theorem smallRow863_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 863 = smallRow863_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 863 ++ upperNeighbors PrimePairVertices.tree_block_004 863 = _
  rw [smallRow863_block_003_checked, smallRow863_block_004_checked]
  rfl

def smallRow863_small_right : List Nat := smallRow863_block_002 ++ smallRow863_small_right_right
theorem smallRow863_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 863 = smallRow863_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 863 ++ upperNeighbors PrimePairVertices.tree_small_right_right 863 = _
  rw [smallRow863_block_002_checked, smallRow863_small_right_right_checked]
  rfl

def smallRow863_small : List Nat := smallRow863_small_left ++ smallRow863_small_right
theorem smallRow863_small_checked : upperNeighbors PrimePairVertices.tree_small 863 = smallRow863_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 863 ++ upperNeighbors PrimePairVertices.tree_small_right 863 = _
  rw [smallRow863_small_left_checked, smallRow863_small_right_checked]
  rfl

def smallRow863 : List Nat := smallRow863_small
theorem smallRow863_checked : upperNeighbors PrimePairVertices.smallTree 863 = smallRow863 := smallRow863_small_checked

def smallRow877_block_000 : List Nat := [883, 937, 997, 1327, 1621]
set_option maxRecDepth 2048 in
theorem smallRow877_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 877 = smallRow877_block_000 := by decide

def smallRow877_block_001 : List Nat := [2251, 2371, 2551, 2917, 3469, 3727, 3823]
set_option maxRecDepth 2048 in
theorem smallRow877_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 877 = smallRow877_block_001 := by decide

def smallRow877_block_002 : List Nat := [4483, 4723, 5419, 5527, 5659, 5743, 6121]
set_option maxRecDepth 2048 in
theorem smallRow877_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 877 = smallRow877_block_002 := by decide

def smallRow877_block_003 : List Nat := [6373, 6529, 6637, 6991, 7243, 7669, 7759, 8191]
set_option maxRecDepth 2048 in
theorem smallRow877_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 877 = smallRow877_block_003 := by decide

def smallRow877_block_004 : List Nat := [8293]
set_option maxRecDepth 2048 in
theorem smallRow877_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 877 = smallRow877_block_004 := by decide

def smallRow877_small_left : List Nat := smallRow877_block_000 ++ smallRow877_block_001
theorem smallRow877_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 877 = smallRow877_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 877 ++ upperNeighbors PrimePairVertices.tree_block_001 877 = _
  rw [smallRow877_block_000_checked, smallRow877_block_001_checked]
  rfl

def smallRow877_small_right_right : List Nat := smallRow877_block_003 ++ smallRow877_block_004
theorem smallRow877_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 877 = smallRow877_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 877 ++ upperNeighbors PrimePairVertices.tree_block_004 877 = _
  rw [smallRow877_block_003_checked, smallRow877_block_004_checked]
  rfl

def smallRow877_small_right : List Nat := smallRow877_block_002 ++ smallRow877_small_right_right
theorem smallRow877_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 877 = smallRow877_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 877 ++ upperNeighbors PrimePairVertices.tree_small_right_right 877 = _
  rw [smallRow877_block_002_checked, smallRow877_small_right_right_checked]
  rfl

def smallRow877_small : List Nat := smallRow877_small_left ++ smallRow877_small_right
theorem smallRow877_small_checked : upperNeighbors PrimePairVertices.tree_small 877 = smallRow877_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 877 ++ upperNeighbors PrimePairVertices.tree_small_right 877 = _
  rw [smallRow877_small_left_checked, smallRow877_small_right_checked]
  rfl

def smallRow877 : List Nat := smallRow877_small
theorem smallRow877_checked : upperNeighbors PrimePairVertices.smallTree 877 = smallRow877 := smallRow877_small_checked

def smallRow881_block_000 : List Nat := [953, 983, 1373, 1499, 1667, 1949]
set_option maxRecDepth 2048 in
theorem smallRow881_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 881 = smallRow881_block_000 := by decide

def smallRow881_block_001 : List Nat := [2411, 3011, 3407]
set_option maxRecDepth 2048 in
theorem smallRow881_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 881 = smallRow881_block_001 := by decide

def smallRow881_block_002 : List Nat := [4217, 4229, 4457, 4733, 4931, 5021, 5591, 5801, 6053]
set_option maxRecDepth 2048 in
theorem smallRow881_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 881 = smallRow881_block_002 := by decide

def smallRow881_block_003 : List Nat := [6449, 6473, 6701, 6791, 6827, 6959, 7103, 7253, 7331, 7541, 7607]
set_option maxRecDepth 2048 in
theorem smallRow881_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 881 = smallRow881_block_003 := by decide

def smallRow881_block_004 : List Nat := [8297]
set_option maxRecDepth 2048 in
theorem smallRow881_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 881 = smallRow881_block_004 := by decide

def smallRow881_small_left : List Nat := smallRow881_block_000 ++ smallRow881_block_001
theorem smallRow881_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 881 = smallRow881_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 881 ++ upperNeighbors PrimePairVertices.tree_block_001 881 = _
  rw [smallRow881_block_000_checked, smallRow881_block_001_checked]
  rfl

def smallRow881_small_right_right : List Nat := smallRow881_block_003 ++ smallRow881_block_004
theorem smallRow881_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 881 = smallRow881_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 881 ++ upperNeighbors PrimePairVertices.tree_block_004 881 = _
  rw [smallRow881_block_003_checked, smallRow881_block_004_checked]
  rfl

def smallRow881_small_right : List Nat := smallRow881_block_002 ++ smallRow881_small_right_right
theorem smallRow881_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 881 = smallRow881_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 881 ++ upperNeighbors PrimePairVertices.tree_small_right_right 881 = _
  rw [smallRow881_block_002_checked, smallRow881_small_right_right_checked]
  rfl

def smallRow881_small : List Nat := smallRow881_small_left ++ smallRow881_small_right
theorem smallRow881_small_checked : upperNeighbors PrimePairVertices.tree_small 881 = smallRow881_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 881 ++ upperNeighbors PrimePairVertices.tree_small_right 881 = _
  rw [smallRow881_small_left_checked, smallRow881_small_right_checked]
  rfl

def smallRow881 : List Nat := smallRow881_small
theorem smallRow881_checked : upperNeighbors PrimePairVertices.smallTree 881 = smallRow881 := smallRow881_small_checked

def smallRow883_block_000 : List Nat := [991, 1783, 1987]
set_option maxRecDepth 2048 in
theorem smallRow883_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 883 = smallRow883_block_000 := by decide

def smallRow883_block_001 : List Nat := [2377, 2749, 3049]
set_option maxRecDepth 2048 in
theorem smallRow883_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 883 = smallRow883_block_001 := by decide

def smallRow883_block_002 : List Nat := [4561, 6043]
set_option maxRecDepth 2048 in
theorem smallRow883_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 883 = smallRow883_block_002 := by decide

def smallRow883_block_003 : List Nat := [6151, 6367, 6571, 6661, 7753, 7993]
set_option maxRecDepth 2048 in
theorem smallRow883_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 883 = smallRow883_block_003 := by decide

def smallRow883_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow883_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 883 = smallRow883_block_004 := by decide

def smallRow883_small_left : List Nat := smallRow883_block_000 ++ smallRow883_block_001
theorem smallRow883_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 883 = smallRow883_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 883 ++ upperNeighbors PrimePairVertices.tree_block_001 883 = _
  rw [smallRow883_block_000_checked, smallRow883_block_001_checked]
  rfl

def smallRow883_small_right_right : List Nat := smallRow883_block_003 ++ smallRow883_block_004
theorem smallRow883_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 883 = smallRow883_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 883 ++ upperNeighbors PrimePairVertices.tree_block_004 883 = _
  rw [smallRow883_block_003_checked, smallRow883_block_004_checked]
  rfl

def smallRow883_small_right : List Nat := smallRow883_block_002 ++ smallRow883_small_right_right
theorem smallRow883_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 883 = smallRow883_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 883 ++ upperNeighbors PrimePairVertices.tree_small_right_right 883 = _
  rw [smallRow883_block_002_checked, smallRow883_small_right_right_checked]
  rfl

def smallRow883_small : List Nat := smallRow883_small_left ++ smallRow883_small_right
theorem smallRow883_small_checked : upperNeighbors PrimePairVertices.tree_small 883 = smallRow883_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 883 ++ upperNeighbors PrimePairVertices.tree_small_right 883 = _
  rw [smallRow883_small_left_checked, smallRow883_small_right_checked]
  rfl

def smallRow883 : List Nat := smallRow883_small
theorem smallRow883_checked : upperNeighbors PrimePairVertices.smallTree 883 = smallRow883 := smallRow883_small_checked

def smallRow887_block_000 : List Nat := [1091, 1427, 1553, 1697, 1847, 1931]
set_option maxRecDepth 2048 in
theorem smallRow887_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 887 = smallRow887_block_000 := by decide

def smallRow887_block_001 : List Nat := [2099, 2153, 2741, 3089, 3251, 3803, 4049]
set_option maxRecDepth 2048 in
theorem smallRow887_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 887 = smallRow887_block_001 := by decide

def smallRow887_block_002 : List Nat := [4157, 5303, 5417, 5651]
set_option maxRecDepth 2048 in
theorem smallRow887_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 887 = smallRow887_block_002 := by decide

def smallRow887_block_003 : List Nat := [6449, 6653, 6863, 6911, 7013, 7109, 7499, 7727, 8087]
set_option maxRecDepth 2048 in
theorem smallRow887_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 887 = smallRow887_block_003 := by decide

def smallRow887_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow887_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 887 = smallRow887_block_004 := by decide

def smallRow887_small_left : List Nat := smallRow887_block_000 ++ smallRow887_block_001
theorem smallRow887_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 887 = smallRow887_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 887 ++ upperNeighbors PrimePairVertices.tree_block_001 887 = _
  rw [smallRow887_block_000_checked, smallRow887_block_001_checked]
  rfl

def smallRow887_small_right_right : List Nat := smallRow887_block_003 ++ smallRow887_block_004
theorem smallRow887_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 887 = smallRow887_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 887 ++ upperNeighbors PrimePairVertices.tree_block_004 887 = _
  rw [smallRow887_block_003_checked, smallRow887_block_004_checked]
  rfl

def smallRow887_small_right : List Nat := smallRow887_block_002 ++ smallRow887_small_right_right
theorem smallRow887_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 887 = smallRow887_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 887 ++ upperNeighbors PrimePairVertices.tree_small_right_right 887 = _
  rw [smallRow887_block_002_checked, smallRow887_small_right_right_checked]
  rfl

def smallRow887_small : List Nat := smallRow887_small_left ++ smallRow887_small_right
theorem smallRow887_small_checked : upperNeighbors PrimePairVertices.tree_small 887 = smallRow887_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 887 ++ upperNeighbors PrimePairVertices.tree_small_right 887 = _
  rw [smallRow887_small_left_checked, smallRow887_small_right_checked]
  rfl

def smallRow887 : List Nat := smallRow887_small
theorem smallRow887_checked : upperNeighbors PrimePairVertices.smallTree 887 = smallRow887 := smallRow887_small_checked

end ProjectEulerSolutions.PrimePairCache
