import ProjectEulerSolutions.Lib.PrimePairVertices.Trees

/-! Increasing adjacency rows, checked against the exact source predicate. -/
namespace ProjectEulerSolutions.PrimePairCache
open PrimePairCertificate

def smallRow523_block_000 : List Nat := [541, 577, 1129, 1297, 1663, 1987]
set_option maxRecDepth 2048 in
theorem smallRow523_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 523 = smallRow523_block_000 := by decide

def smallRow523_block_001 : List Nat := [2089, 2113, 2671, 2677, 3307, 3469, 3499, 3919]
set_option maxRecDepth 2048 in
theorem smallRow523_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 523 = smallRow523_block_001 := by decide

def smallRow523_block_002 : List Nat := [4591, 4909, 4987, 5233, 5851]
set_option maxRecDepth 2048 in
theorem smallRow523_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 523 = smallRow523_block_002 := by decide

def smallRow523_block_003 : List Nat := [6367, 6967, 7237, 7753]
set_option maxRecDepth 2048 in
theorem smallRow523_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 523 = smallRow523_block_003 := by decide

def smallRow523_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow523_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 523 = smallRow523_block_004 := by decide

def smallRow523_small_left : List Nat := smallRow523_block_000 ++ smallRow523_block_001
theorem smallRow523_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 523 = smallRow523_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 523 ++ upperNeighbors PrimePairVertices.tree_block_001 523 = _
  rw [smallRow523_block_000_checked, smallRow523_block_001_checked]
  rfl

def smallRow523_small_right_right : List Nat := smallRow523_block_003 ++ smallRow523_block_004
theorem smallRow523_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 523 = smallRow523_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 523 ++ upperNeighbors PrimePairVertices.tree_block_004 523 = _
  rw [smallRow523_block_003_checked, smallRow523_block_004_checked]
  rfl

def smallRow523_small_right : List Nat := smallRow523_block_002 ++ smallRow523_small_right_right
theorem smallRow523_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 523 = smallRow523_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 523 ++ upperNeighbors PrimePairVertices.tree_small_right_right 523 = _
  rw [smallRow523_block_002_checked, smallRow523_small_right_right_checked]
  rfl

def smallRow523_small : List Nat := smallRow523_small_left ++ smallRow523_small_right
theorem smallRow523_small_checked : upperNeighbors PrimePairVertices.tree_small 523 = smallRow523_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 523 ++ upperNeighbors PrimePairVertices.tree_small_right 523 = _
  rw [smallRow523_small_left_checked, smallRow523_small_right_checked]
  rfl

def smallRow523 : List Nat := smallRow523_small
theorem smallRow523_checked : upperNeighbors PrimePairVertices.smallTree 523 = smallRow523 := smallRow523_small_checked

def smallRow541_block_000 : List Nat := [571, 661, 727, 991, 1033, 1093, 1621]
set_option maxRecDepth 2048 in
theorem smallRow541_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 541 = smallRow541_block_000 := by decide

def smallRow541_block_001 : List Nat := [2467, 2791, 2887, 3301, 3631]
set_option maxRecDepth 2048 in
theorem smallRow541_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 541 = smallRow541_block_001 := by decide

def smallRow541_block_002 : List Nat := [4159, 4231, 4261, 5101, 5479]
set_option maxRecDepth 2048 in
theorem smallRow541_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 541 = smallRow541_block_002 := by decide

def smallRow541_block_003 : List Nat := [6709, 6991, 7369, 8167]
set_option maxRecDepth 2048 in
theorem smallRow541_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 541 = smallRow541_block_003 := by decide

def smallRow541_block_004 : List Nat := [8233]
set_option maxRecDepth 2048 in
theorem smallRow541_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 541 = smallRow541_block_004 := by decide

def smallRow541_small_left : List Nat := smallRow541_block_000 ++ smallRow541_block_001
theorem smallRow541_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 541 = smallRow541_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 541 ++ upperNeighbors PrimePairVertices.tree_block_001 541 = _
  rw [smallRow541_block_000_checked, smallRow541_block_001_checked]
  rfl

def smallRow541_small_right_right : List Nat := smallRow541_block_003 ++ smallRow541_block_004
theorem smallRow541_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 541 = smallRow541_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 541 ++ upperNeighbors PrimePairVertices.tree_block_004 541 = _
  rw [smallRow541_block_003_checked, smallRow541_block_004_checked]
  rfl

def smallRow541_small_right : List Nat := smallRow541_block_002 ++ smallRow541_small_right_right
theorem smallRow541_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 541 = smallRow541_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 541 ++ upperNeighbors PrimePairVertices.tree_small_right_right 541 = _
  rw [smallRow541_block_002_checked, smallRow541_small_right_right_checked]
  rfl

def smallRow541_small : List Nat := smallRow541_small_left ++ smallRow541_small_right
theorem smallRow541_small_checked : upperNeighbors PrimePairVertices.tree_small 541 = smallRow541_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 541 ++ upperNeighbors PrimePairVertices.tree_small_right 541 = _
  rw [smallRow541_small_left_checked, smallRow541_small_right_checked]
  rfl

def smallRow541 : List Nat := smallRow541_small
theorem smallRow541_checked : upperNeighbors PrimePairVertices.smallTree 541 = smallRow541 := smallRow541_small_checked

def smallRow547_block_000 : List Nat := [577, 643, 661, 709, 769, 787, 823, 853, 1237, 1663, 1699]
set_option maxRecDepth 2048 in
theorem smallRow547_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 547 = smallRow547_block_000 := by decide

def smallRow547_block_001 : List Nat := [2473, 3001, 3229, 3271]
set_option maxRecDepth 2048 in
theorem smallRow547_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 547 = smallRow547_block_001 := by decide

def smallRow547_block_002 : List Nat := [4519, 4549, 5443, 5557, 6073]
set_option maxRecDepth 2048 in
theorem smallRow547_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 547 = smallRow547_block_002 := by decide

def smallRow547_block_003 : List Nat := [6451, 6673, 7237, 7867]
set_option maxRecDepth 2048 in
theorem smallRow547_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 547 = smallRow547_block_003 := by decide

def smallRow547_block_004 : List Nat := [8311]
set_option maxRecDepth 2048 in
theorem smallRow547_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 547 = smallRow547_block_004 := by decide

def smallRow547_small_left : List Nat := smallRow547_block_000 ++ smallRow547_block_001
theorem smallRow547_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 547 = smallRow547_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 547 ++ upperNeighbors PrimePairVertices.tree_block_001 547 = _
  rw [smallRow547_block_000_checked, smallRow547_block_001_checked]
  rfl

def smallRow547_small_right_right : List Nat := smallRow547_block_003 ++ smallRow547_block_004
theorem smallRow547_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 547 = smallRow547_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 547 ++ upperNeighbors PrimePairVertices.tree_block_004 547 = _
  rw [smallRow547_block_003_checked, smallRow547_block_004_checked]
  rfl

def smallRow547_small_right : List Nat := smallRow547_block_002 ++ smallRow547_small_right_right
theorem smallRow547_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 547 = smallRow547_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 547 ++ upperNeighbors PrimePairVertices.tree_small_right_right 547 = _
  rw [smallRow547_block_002_checked, smallRow547_small_right_right_checked]
  rfl

def smallRow547_small : List Nat := smallRow547_small_left ++ smallRow547_small_right
theorem smallRow547_small_checked : upperNeighbors PrimePairVertices.tree_small 547 = smallRow547_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 547 ++ upperNeighbors PrimePairVertices.tree_small_right 547 = _
  rw [smallRow547_small_left_checked, smallRow547_small_right_checked]
  rfl

def smallRow547 : List Nat := smallRow547_small
theorem smallRow547_checked : upperNeighbors PrimePairVertices.smallTree 547 = smallRow547 := smallRow547_small_checked

def smallRow557_block_000 : List Nat := [1487, 1721]
set_option maxRecDepth 2048 in
theorem smallRow557_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 557 = smallRow557_block_000 := by decide

def smallRow557_block_001 : List Nat := [2687, 2729, 3221, 3389, 3929, 3947]
set_option maxRecDepth 2048 in
theorem smallRow557_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 557 = smallRow557_block_001 := by decide

def smallRow557_block_002 : List Nat := [4703, 5651, 5879, 6047]
set_option maxRecDepth 2048 in
theorem smallRow557_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 557 = smallRow557_block_002 := by decide

def smallRow557_block_003 : List Nat := [6257, 6491, 7079, 8087, 8093]
set_option maxRecDepth 2048 in
theorem smallRow557_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 557 = smallRow557_block_003 := by decide

def smallRow557_block_004 : List Nat := [8291, 8387]
set_option maxRecDepth 2048 in
theorem smallRow557_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 557 = smallRow557_block_004 := by decide

def smallRow557_small_left : List Nat := smallRow557_block_000 ++ smallRow557_block_001
theorem smallRow557_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 557 = smallRow557_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 557 ++ upperNeighbors PrimePairVertices.tree_block_001 557 = _
  rw [smallRow557_block_000_checked, smallRow557_block_001_checked]
  rfl

def smallRow557_small_right_right : List Nat := smallRow557_block_003 ++ smallRow557_block_004
theorem smallRow557_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 557 = smallRow557_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 557 ++ upperNeighbors PrimePairVertices.tree_block_004 557 = _
  rw [smallRow557_block_003_checked, smallRow557_block_004_checked]
  rfl

def smallRow557_small_right : List Nat := smallRow557_block_002 ++ smallRow557_small_right_right
theorem smallRow557_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 557 = smallRow557_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 557 ++ upperNeighbors PrimePairVertices.tree_small_right_right 557 = _
  rw [smallRow557_block_002_checked, smallRow557_small_right_right_checked]
  rfl

def smallRow557_small : List Nat := smallRow557_small_left ++ smallRow557_small_right
theorem smallRow557_small_checked : upperNeighbors PrimePairVertices.tree_small 557 = smallRow557_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 557 ++ upperNeighbors PrimePairVertices.tree_small_right 557 = _
  rw [smallRow557_small_left_checked, smallRow557_small_right_checked]
  rfl

def smallRow557 : List Nat := smallRow557_small
theorem smallRow557_checked : upperNeighbors PrimePairVertices.smallTree 557 = smallRow557 := smallRow557_small_checked

def smallRow563_block_000 : List Nat := [587, 809, 971, 1013, 1259, 1427, 1523, 1601, 1973, 1979]
set_option maxRecDepth 2048 in
theorem smallRow563_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 563 = smallRow563_block_000 := by decide

def smallRow563_block_001 : List Nat := [2267, 2909, 3119, 3821]
set_option maxRecDepth 2048 in
theorem smallRow563_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 563 = smallRow563_block_001 := by decide

def smallRow563_block_002 : List Nat := [4493, 4799, 5237, 5261, 5297, 5501, 5717]
set_option maxRecDepth 2048 in
theorem smallRow563_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 563 = smallRow563_block_002 := by decide

def smallRow563_block_003 : List Nat := [6551, 6581, 6791, 7019, 7043, 7481]
set_option maxRecDepth 2048 in
theorem smallRow563_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 563 = smallRow563_block_003 := by decide

def smallRow563_block_004 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallRow563_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 563 = smallRow563_block_004 := by decide

def smallRow563_small_left : List Nat := smallRow563_block_000 ++ smallRow563_block_001
theorem smallRow563_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 563 = smallRow563_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 563 ++ upperNeighbors PrimePairVertices.tree_block_001 563 = _
  rw [smallRow563_block_000_checked, smallRow563_block_001_checked]
  rfl

def smallRow563_small_right_right : List Nat := smallRow563_block_003 ++ smallRow563_block_004
theorem smallRow563_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 563 = smallRow563_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 563 ++ upperNeighbors PrimePairVertices.tree_block_004 563 = _
  rw [smallRow563_block_003_checked, smallRow563_block_004_checked]
  rfl

def smallRow563_small_right : List Nat := smallRow563_block_002 ++ smallRow563_small_right_right
theorem smallRow563_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 563 = smallRow563_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 563 ++ upperNeighbors PrimePairVertices.tree_small_right_right 563 = _
  rw [smallRow563_block_002_checked, smallRow563_small_right_right_checked]
  rfl

def smallRow563_small : List Nat := smallRow563_small_left ++ smallRow563_small_right
theorem smallRow563_small_checked : upperNeighbors PrimePairVertices.tree_small 563 = smallRow563_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 563 ++ upperNeighbors PrimePairVertices.tree_small_right 563 = _
  rw [smallRow563_small_left_checked, smallRow563_small_right_checked]
  rfl

def smallRow563 : List Nat := smallRow563_small
theorem smallRow563_checked : upperNeighbors PrimePairVertices.smallTree 563 = smallRow563 := smallRow563_small_checked

def smallRow569_block_000 : List Nat := [659, 773, 797, 809, 887, 1151, 1487, 1601]
set_option maxRecDepth 2048 in
theorem smallRow569_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 569 = smallRow569_block_000 := by decide

def smallRow569_block_001 : List Nat := [2657, 2699, 2861, 3203, 3617]
set_option maxRecDepth 2048 in
theorem smallRow569_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 569 = smallRow569_block_001 := by decide

def smallRow569_block_002 : List Nat := [4673, 4919, 5741]
set_option maxRecDepth 2048 in
theorem smallRow569_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 569 = smallRow569_block_002 := by decide

def smallRow569_block_003 : List Nat := [6959, 7673, 8039]
set_option maxRecDepth 2048 in
theorem smallRow569_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 569 = smallRow569_block_003 := by decide

def smallRow569_block_004 : List Nat := [8219]
set_option maxRecDepth 2048 in
theorem smallRow569_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 569 = smallRow569_block_004 := by decide

def smallRow569_small_left : List Nat := smallRow569_block_000 ++ smallRow569_block_001
theorem smallRow569_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 569 = smallRow569_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 569 ++ upperNeighbors PrimePairVertices.tree_block_001 569 = _
  rw [smallRow569_block_000_checked, smallRow569_block_001_checked]
  rfl

def smallRow569_small_right_right : List Nat := smallRow569_block_003 ++ smallRow569_block_004
theorem smallRow569_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 569 = smallRow569_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 569 ++ upperNeighbors PrimePairVertices.tree_block_004 569 = _
  rw [smallRow569_block_003_checked, smallRow569_block_004_checked]
  rfl

def smallRow569_small_right : List Nat := smallRow569_block_002 ++ smallRow569_small_right_right
theorem smallRow569_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 569 = smallRow569_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 569 ++ upperNeighbors PrimePairVertices.tree_small_right_right 569 = _
  rw [smallRow569_block_002_checked, smallRow569_small_right_right_checked]
  rfl

def smallRow569_small : List Nat := smallRow569_small_left ++ smallRow569_small_right
theorem smallRow569_small_checked : upperNeighbors PrimePairVertices.tree_small 569 = smallRow569_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 569 ++ upperNeighbors PrimePairVertices.tree_small_right 569 = _
  rw [smallRow569_small_left_checked, smallRow569_small_right_checked]
  rfl

def smallRow569 : List Nat := smallRow569_small
theorem smallRow569_checked : upperNeighbors PrimePairVertices.smallTree 569 = smallRow569 := smallRow569_small_checked

def smallRow571_block_000 : List Nat := [853, 1051, 1933]
set_option maxRecDepth 2048 in
theorem smallRow571_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 571 = smallRow571_block_000 := by decide

def smallRow571_block_001 : List Nat := [2341, 2383, 2647, 2659, 2953, 3061, 3121, 3163]
set_option maxRecDepth 2048 in
theorem smallRow571_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 571 = smallRow571_block_001 := by decide

def smallRow571_block_002 : List Nat := [4483, 4759, 4987, 5701, 5839]
set_option maxRecDepth 2048 in
theorem smallRow571_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 571 = smallRow571_block_002 := by decide

def smallRow571_block_003 : List Nat := [6553, 7069, 7213, 7297, 7549, 7951]
set_option maxRecDepth 2048 in
theorem smallRow571_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 571 = smallRow571_block_003 := by decide

def smallRow571_block_004 : List Nat := [8233, 8263]
set_option maxRecDepth 2048 in
theorem smallRow571_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 571 = smallRow571_block_004 := by decide

def smallRow571_small_left : List Nat := smallRow571_block_000 ++ smallRow571_block_001
theorem smallRow571_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 571 = smallRow571_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 571 ++ upperNeighbors PrimePairVertices.tree_block_001 571 = _
  rw [smallRow571_block_000_checked, smallRow571_block_001_checked]
  rfl

def smallRow571_small_right_right : List Nat := smallRow571_block_003 ++ smallRow571_block_004
theorem smallRow571_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 571 = smallRow571_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 571 ++ upperNeighbors PrimePairVertices.tree_block_004 571 = _
  rw [smallRow571_block_003_checked, smallRow571_block_004_checked]
  rfl

def smallRow571_small_right : List Nat := smallRow571_block_002 ++ smallRow571_small_right_right
theorem smallRow571_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 571 = smallRow571_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 571 ++ upperNeighbors PrimePairVertices.tree_small_right_right 571 = _
  rw [smallRow571_block_002_checked, smallRow571_small_right_right_checked]
  rfl

def smallRow571_small : List Nat := smallRow571_small_left ++ smallRow571_small_right
theorem smallRow571_small_checked : upperNeighbors PrimePairVertices.tree_small 571 = smallRow571_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 571 ++ upperNeighbors PrimePairVertices.tree_small_right 571 = _
  rw [smallRow571_small_left_checked, smallRow571_small_right_checked]
  rfl

def smallRow571 : List Nat := smallRow571_small
theorem smallRow571_checked : upperNeighbors PrimePairVertices.smallTree 571 = smallRow571 := smallRow571_small_checked

def smallRow577_block_000 : List Nat := [613, 757, 937, 1327, 1723]
set_option maxRecDepth 2048 in
theorem smallRow577_block_000_checked : upperNeighbors PrimePairVertices.tree_block_000 577 = smallRow577_block_000 := by decide

def smallRow577_block_001 : List Nat := [2539, 2617, 3331, 3709, 3919]
set_option maxRecDepth 2048 in
theorem smallRow577_block_001_checked : upperNeighbors PrimePairVertices.tree_block_001 577 = smallRow577_block_001 := by decide

def smallRow577_block_002 : List Nat := [4597, 4759, 5431, 5683, 5821, 5839]
set_option maxRecDepth 2048 in
theorem smallRow577_block_002_checked : upperNeighbors PrimePairVertices.tree_block_002 577 = smallRow577_block_002 := by decide

def smallRow577_block_003 : List Nat := [6949, 7459, 7951, 7963]
set_option maxRecDepth 2048 in
theorem smallRow577_block_003_checked : upperNeighbors PrimePairVertices.tree_block_003 577 = smallRow577_block_003 := by decide

def smallRow577_block_004 : List Nat := [8329]
set_option maxRecDepth 2048 in
theorem smallRow577_block_004_checked : upperNeighbors PrimePairVertices.tree_block_004 577 = smallRow577_block_004 := by decide

def smallRow577_small_left : List Nat := smallRow577_block_000 ++ smallRow577_block_001
theorem smallRow577_small_left_checked : upperNeighbors PrimePairVertices.tree_small_left 577 = smallRow577_small_left := by
  change upperNeighbors PrimePairVertices.tree_block_000 577 ++ upperNeighbors PrimePairVertices.tree_block_001 577 = _
  rw [smallRow577_block_000_checked, smallRow577_block_001_checked]
  rfl

def smallRow577_small_right_right : List Nat := smallRow577_block_003 ++ smallRow577_block_004
theorem smallRow577_small_right_right_checked : upperNeighbors PrimePairVertices.tree_small_right_right 577 = smallRow577_small_right_right := by
  change upperNeighbors PrimePairVertices.tree_block_003 577 ++ upperNeighbors PrimePairVertices.tree_block_004 577 = _
  rw [smallRow577_block_003_checked, smallRow577_block_004_checked]
  rfl

def smallRow577_small_right : List Nat := smallRow577_block_002 ++ smallRow577_small_right_right
theorem smallRow577_small_right_checked : upperNeighbors PrimePairVertices.tree_small_right 577 = smallRow577_small_right := by
  change upperNeighbors PrimePairVertices.tree_block_002 577 ++ upperNeighbors PrimePairVertices.tree_small_right_right 577 = _
  rw [smallRow577_block_002_checked, smallRow577_small_right_right_checked]
  rfl

def smallRow577_small : List Nat := smallRow577_small_left ++ smallRow577_small_right
theorem smallRow577_small_checked : upperNeighbors PrimePairVertices.tree_small 577 = smallRow577_small := by
  change upperNeighbors PrimePairVertices.tree_small_left 577 ++ upperNeighbors PrimePairVertices.tree_small_right 577 = _
  rw [smallRow577_small_left_checked, smallRow577_small_right_checked]
  rfl

def smallRow577 : List Nat := smallRow577_small
theorem smallRow577_checked : upperNeighbors PrimePairVertices.smallTree 577 = smallRow577 := smallRow577_small_checked

end ProjectEulerSolutions.PrimePairCache
