import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates0 : List Nat := [7, 11, 17, 31, 37, 59, 67, 73, 109, 137, 191, 229, 271, 331, 359, 373, 449, 467, 499, 541, 557, 607, 613, 617, 673, 701, 719, 733, 739, 823, 929, 947, 1013, 1019, 1033, 1051, 1181, 1193, 1237, 1481, 1531, 1607, 1627, 1657, 1663, 1667, 1699, 1907, 2069, 2143, 2213, 2297, 2377, 2381, 2411, 2441, 2503, 2579, 2707, 2789, 2843, 2917, 2957, 3049, 3119, 3301, 3461, 3469, 3637, 3769, 3911, 3923, 3931, 4019, 4159, 4253, 4583, 4729, 4919, 5051, 5059, 5099, 5171, 5281, 5323, 5381, 5419, 5449, 5507, 5521, 5531, 5573, 5801, 5839, 5869, 5923, 6037, 6073, 6263, 6277, 6353, 6451, 6469, 6529, 6563, 6571, 6599, 6637, 6653, 6793, 6871, 6899, 6947, 7039, 7057, 7159, 7253, 7369, 7591, 7649, 7691, 7963, 8219, 8237, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates0_checked : candidates edge 5 0 3 tail1 best.bound = rootCandidates0 := by
  unfold candidates
  have row : edge 3 = PrimePairCache.smallRow3.contains := by funext q; rfl
  rw [row]
  decide

def innerTail0_125 : List Nat := []
theorem innerScan0_125 : scan edge (search edge 3) 4 [3] 3 innerTail0_125 best = best := rfl

def innerTail0_124 : List Nat := 8377 :: innerTail0_125
def innerCandidates0_124 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_124_checked : candidates edge 4 3 8377 innerTail0_125 best.bound = innerCandidates0_124 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_124 : scan edge (search edge 3) 4 [3] 3 innerTail0_124 best = best := by
  rw [innerTail0_124, scan, if_neg (by decide), innerCandidates0_124_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_125
    (if 3 ≤ innerCandidates0_124.length then search edge 3 [3, 8377] innerCandidates0_124 8380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_125

def innerTail0_123 : List Nat := 8237 :: innerTail0_124
def innerCandidates0_123 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_123_checked : candidates edge 4 3 8237 innerTail0_124 best.bound = innerCandidates0_123 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_123 : scan edge (search edge 3) 4 [3] 3 innerTail0_123 best = best := by
  rw [innerTail0_123, scan, if_neg (by decide), innerCandidates0_123_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_124
    (if 3 ≤ innerCandidates0_123.length then search edge 3 [3, 8237] innerCandidates0_123 8240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_124

def innerTail0_122 : List Nat := 8219 :: innerTail0_123
def innerCandidates0_122 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_122_checked : candidates edge 4 3 8219 innerTail0_123 best.bound = innerCandidates0_122 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_122 : scan edge (search edge 3) 4 [3] 3 innerTail0_122 best = best := by
  rw [innerTail0_122, scan, if_neg (by decide), innerCandidates0_122_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_123
    (if 3 ≤ innerCandidates0_122.length then search edge 3 [3, 8219] innerCandidates0_122 8222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_123

def innerTail0_121 : List Nat := 7963 :: innerTail0_122
def innerCandidates0_121 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_121_checked : candidates edge 4 3 7963 innerTail0_122 best.bound = innerCandidates0_121 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_121 : scan edge (search edge 3) 4 [3] 3 innerTail0_121 best = best := by
  rw [innerTail0_121, scan, if_neg (by decide), innerCandidates0_121_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_122
    (if 3 ≤ innerCandidates0_121.length then search edge 3 [3, 7963] innerCandidates0_121 7966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_122

def innerTail0_120 : List Nat := 7691 :: innerTail0_121
def innerCandidates0_120 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_120_checked : candidates edge 4 3 7691 innerTail0_121 best.bound = innerCandidates0_120 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_120 : scan edge (search edge 3) 4 [3] 3 innerTail0_120 best = best := by
  rw [innerTail0_120, scan, if_neg (by decide), innerCandidates0_120_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_121
    (if 3 ≤ innerCandidates0_120.length then search edge 3 [3, 7691] innerCandidates0_120 7694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_121

def innerTail0_119 : List Nat := 7649 :: innerTail0_120
def innerCandidates0_119 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_119_checked : candidates edge 4 3 7649 innerTail0_120 best.bound = innerCandidates0_119 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_119 : scan edge (search edge 3) 4 [3] 3 innerTail0_119 best = best := by
  rw [innerTail0_119, scan, if_neg (by decide), innerCandidates0_119_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_120
    (if 3 ≤ innerCandidates0_119.length then search edge 3 [3, 7649] innerCandidates0_119 7652 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_120

def innerTail0_118 : List Nat := 7591 :: innerTail0_119
def innerCandidates0_118 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_118_checked : candidates edge 4 3 7591 innerTail0_119 best.bound = innerCandidates0_118 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_118 : scan edge (search edge 3) 4 [3] 3 innerTail0_118 best = best := by
  rw [innerTail0_118, scan, if_neg (by decide), innerCandidates0_118_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_119
    (if 3 ≤ innerCandidates0_118.length then search edge 3 [3, 7591] innerCandidates0_118 7594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_119

def innerTail0_117 : List Nat := 7369 :: innerTail0_118
def innerCandidates0_117 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_117_checked : candidates edge 4 3 7369 innerTail0_118 best.bound = innerCandidates0_117 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_117 : scan edge (search edge 3) 4 [3] 3 innerTail0_117 best = best := by
  rw [innerTail0_117, scan, if_neg (by decide), innerCandidates0_117_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_118
    (if 3 ≤ innerCandidates0_117.length then search edge 3 [3, 7369] innerCandidates0_117 7372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_118

def innerTail0_116 : List Nat := 7253 :: innerTail0_117
def innerCandidates0_116 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_116_checked : candidates edge 4 3 7253 innerTail0_117 best.bound = innerCandidates0_116 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_116 : scan edge (search edge 3) 4 [3] 3 innerTail0_116 best = best := by
  rw [innerTail0_116, scan, if_neg (by decide), innerCandidates0_116_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_117
    (if 3 ≤ innerCandidates0_116.length then search edge 3 [3, 7253] innerCandidates0_116 7256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_117

def innerTail0_115 : List Nat := 7159 :: innerTail0_116
def innerCandidates0_115 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_115_checked : candidates edge 4 3 7159 innerTail0_116 best.bound = innerCandidates0_115 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_115 : scan edge (search edge 3) 4 [3] 3 innerTail0_115 best = best := by
  rw [innerTail0_115, scan, if_neg (by decide), innerCandidates0_115_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_116
    (if 3 ≤ innerCandidates0_115.length then search edge 3 [3, 7159] innerCandidates0_115 7162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_116

def innerTail0_114 : List Nat := 7057 :: innerTail0_115
def innerCandidates0_114 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_114_checked : candidates edge 4 3 7057 innerTail0_115 best.bound = innerCandidates0_114 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_114 : scan edge (search edge 3) 4 [3] 3 innerTail0_114 best = best := by
  rw [innerTail0_114, scan, if_neg (by decide), innerCandidates0_114_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_115
    (if 3 ≤ innerCandidates0_114.length then search edge 3 [3, 7057] innerCandidates0_114 7060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_115

def innerTail0_113 : List Nat := 7039 :: innerTail0_114
def innerCandidates0_113 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_113_checked : candidates edge 4 3 7039 innerTail0_114 best.bound = innerCandidates0_113 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_113 : scan edge (search edge 3) 4 [3] 3 innerTail0_113 best = best := by
  rw [innerTail0_113, scan, if_neg (by decide), innerCandidates0_113_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_114
    (if 3 ≤ innerCandidates0_113.length then search edge 3 [3, 7039] innerCandidates0_113 7042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_114

def innerTail0_112 : List Nat := 6947 :: innerTail0_113
def innerCandidates0_112 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_112_checked : candidates edge 4 3 6947 innerTail0_113 best.bound = innerCandidates0_112 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_112 : scan edge (search edge 3) 4 [3] 3 innerTail0_112 best = best := by
  rw [innerTail0_112, scan, if_neg (by decide), innerCandidates0_112_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_113
    (if 3 ≤ innerCandidates0_112.length then search edge 3 [3, 6947] innerCandidates0_112 6950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_113

def innerTail0_111 : List Nat := 6899 :: innerTail0_112
def innerCandidates0_111 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_111_checked : candidates edge 4 3 6899 innerTail0_112 best.bound = innerCandidates0_111 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_111 : scan edge (search edge 3) 4 [3] 3 innerTail0_111 best = best := by
  rw [innerTail0_111, scan, if_neg (by decide), innerCandidates0_111_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_112
    (if 3 ≤ innerCandidates0_111.length then search edge 3 [3, 6899] innerCandidates0_111 6902 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_112

def innerTail0_110 : List Nat := 6871 :: innerTail0_111
def innerCandidates0_110 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates0_110_checked : candidates edge 4 3 6871 innerTail0_111 best.bound = innerCandidates0_110 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_110 : scan edge (search edge 3) 4 [3] 3 innerTail0_110 best = best := by
  rw [innerTail0_110, scan, if_neg (by decide), innerCandidates0_110_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_111
    (if 3 ≤ innerCandidates0_110.length then search edge 3 [3, 6871] innerCandidates0_110 6874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_111

def innerTail0_109 : List Nat := 6793 :: innerTail0_110
def innerCandidates0_109 : List Nat := [7039, 7159]
set_option maxRecDepth 4096 in
theorem innerCandidates0_109_checked : candidates edge 4 3 6793 innerTail0_110 best.bound = innerCandidates0_109 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_109 : scan edge (search edge 3) 4 [3] 3 innerTail0_109 best = best := by
  rw [innerTail0_109, scan, if_neg (by decide), innerCandidates0_109_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_110
    (if 3 ≤ innerCandidates0_109.length then search edge 3 [3, 6793] innerCandidates0_109 6796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_110

def innerTail0_108 : List Nat := 6653 :: innerTail0_109
def innerCandidates0_108 : List Nat := [7649]
set_option maxRecDepth 4096 in
theorem innerCandidates0_108_checked : candidates edge 4 3 6653 innerTail0_109 best.bound = innerCandidates0_108 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_108 : scan edge (search edge 3) 4 [3] 3 innerTail0_108 best = best := by
  rw [innerTail0_108, scan, if_neg (by decide), innerCandidates0_108_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_109
    (if 3 ≤ innerCandidates0_108.length then search edge 3 [3, 6653] innerCandidates0_108 6656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_109

def innerTail0_107 : List Nat := 6637 :: innerTail0_108
def innerCandidates0_107 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates0_107_checked : candidates edge 4 3 6637 innerTail0_108 best.bound = innerCandidates0_107 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_107 : scan edge (search edge 3) 4 [3] 3 innerTail0_107 best = best := by
  rw [innerTail0_107, scan, if_neg (by decide), innerCandidates0_107_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_108
    (if 3 ≤ innerCandidates0_107.length then search edge 3 [3, 6637] innerCandidates0_107 6640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_108

def innerTail0_106 : List Nat := 6599 :: innerTail0_107
def innerCandidates0_106 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_106_checked : candidates edge 4 3 6599 innerTail0_107 best.bound = innerCandidates0_106 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_106 : scan edge (search edge 3) 4 [3] 3 innerTail0_106 best = best := by
  rw [innerTail0_106, scan, if_neg (by decide), innerCandidates0_106_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_107
    (if 3 ≤ innerCandidates0_106.length then search edge 3 [3, 6599] innerCandidates0_106 6602 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_107

def innerTail0_105 : List Nat := 6571 :: innerTail0_106
def innerCandidates0_105 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_105_checked : candidates edge 4 3 6571 innerTail0_106 best.bound = innerCandidates0_105 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_105 : scan edge (search edge 3) 4 [3] 3 innerTail0_105 best = best := by
  rw [innerTail0_105, scan, if_neg (by decide), innerCandidates0_105_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_106
    (if 3 ≤ innerCandidates0_105.length then search edge 3 [3, 6571] innerCandidates0_105 6574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_106

def innerTail0_104 : List Nat := 6563 :: innerTail0_105
def innerCandidates0_104 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_104_checked : candidates edge 4 3 6563 innerTail0_105 best.bound = innerCandidates0_104 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_104 : scan edge (search edge 3) 4 [3] 3 innerTail0_104 best = best := by
  rw [innerTail0_104, scan, if_neg (by decide), innerCandidates0_104_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_105
    (if 3 ≤ innerCandidates0_104.length then search edge 3 [3, 6563] innerCandidates0_104 6566 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_105

def innerTail0_103 : List Nat := 6529 :: innerTail0_104
def innerCandidates0_103 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_103_checked : candidates edge 4 3 6529 innerTail0_104 best.bound = innerCandidates0_103 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_103 : scan edge (search edge 3) 4 [3] 3 innerTail0_103 best = best := by
  rw [innerTail0_103, scan, if_neg (by decide), innerCandidates0_103_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_104
    (if 3 ≤ innerCandidates0_103.length then search edge 3 [3, 6529] innerCandidates0_103 6532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_104

def innerTail0_102 : List Nat := 6469 :: innerTail0_103
def innerCandidates0_102 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_102_checked : candidates edge 4 3 6469 innerTail0_103 best.bound = innerCandidates0_102 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_102 : scan edge (search edge 3) 4 [3] 3 innerTail0_102 best = best := by
  rw [innerTail0_102, scan, if_neg (by decide), innerCandidates0_102_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_103
    (if 3 ≤ innerCandidates0_102.length then search edge 3 [3, 6469] innerCandidates0_102 6472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_103

def innerTail0_101 : List Nat := 6451 :: innerTail0_102
def innerCandidates0_101 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_101_checked : candidates edge 4 3 6451 innerTail0_102 best.bound = innerCandidates0_101 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_101 : scan edge (search edge 3) 4 [3] 3 innerTail0_101 best = best := by
  rw [innerTail0_101, scan, if_neg (by decide), innerCandidates0_101_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_102
    (if 3 ≤ innerCandidates0_101.length then search edge 3 [3, 6451] innerCandidates0_101 6454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_102

def innerTail0_100 : List Nat := 6353 :: innerTail0_101
def innerCandidates0_100 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_100_checked : candidates edge 4 3 6353 innerTail0_101 best.bound = innerCandidates0_100 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_100 : scan edge (search edge 3) 4 [3] 3 innerTail0_100 best = best := by
  rw [innerTail0_100, scan, if_neg (by decide), innerCandidates0_100_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_101
    (if 3 ≤ innerCandidates0_100.length then search edge 3 [3, 6353] innerCandidates0_100 6356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_101

def innerTail0_99 : List Nat := 6277 :: innerTail0_100
def innerCandidates0_99 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_99_checked : candidates edge 4 3 6277 innerTail0_100 best.bound = innerCandidates0_99 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_99 : scan edge (search edge 3) 4 [3] 3 innerTail0_99 best = best := by
  rw [innerTail0_99, scan, if_neg (by decide), innerCandidates0_99_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_100
    (if 3 ≤ innerCandidates0_99.length then search edge 3 [3, 6277] innerCandidates0_99 6280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_100

def innerTail0_98 : List Nat := 6263 :: innerTail0_99
def innerCandidates0_98 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_98_checked : candidates edge 4 3 6263 innerTail0_99 best.bound = innerCandidates0_98 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_98 : scan edge (search edge 3) 4 [3] 3 innerTail0_98 best = best := by
  rw [innerTail0_98, scan, if_neg (by decide), innerCandidates0_98_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_99
    (if 3 ≤ innerCandidates0_98.length then search edge 3 [3, 6263] innerCandidates0_98 6266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_99

def innerTail0_97 : List Nat := 6073 :: innerTail0_98
def innerCandidates0_97 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_97_checked : candidates edge 4 3 6073 innerTail0_98 best.bound = innerCandidates0_97 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_97 : scan edge (search edge 3) 4 [3] 3 innerTail0_97 best = best := by
  rw [innerTail0_97, scan, if_neg (by decide), innerCandidates0_97_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_98
    (if 3 ≤ innerCandidates0_97.length then search edge 3 [3, 6073] innerCandidates0_97 6076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_98

def innerTail0_96 : List Nat := 6037 :: innerTail0_97
def innerCandidates0_96 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_96_checked : candidates edge 4 3 6037 innerTail0_97 best.bound = innerCandidates0_96 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_96 : scan edge (search edge 3) 4 [3] 3 innerTail0_96 best = best := by
  rw [innerTail0_96, scan, if_neg (by decide), innerCandidates0_96_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_97
    (if 3 ≤ innerCandidates0_96.length then search edge 3 [3, 6037] innerCandidates0_96 6040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_97

def innerTail0_95 : List Nat := 5923 :: innerTail0_96
def innerCandidates0_95 : List Nat := [6871, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates0_95_checked : candidates edge 4 3 5923 innerTail0_96 best.bound = innerCandidates0_95 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_95 : scan edge (search edge 3) 4 [3] 3 innerTail0_95 best = best := by
  rw [innerTail0_95, scan, if_neg (by decide), innerCandidates0_95_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_96
    (if 3 ≤ innerCandidates0_95.length then search edge 3 [3, 5923] innerCandidates0_95 5926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_96

def innerTail0_94 : List Nat := 5869 :: innerTail0_95
def innerCandidates0_94 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_94_checked : candidates edge 4 3 5869 innerTail0_95 best.bound = innerCandidates0_94 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_94 : scan edge (search edge 3) 4 [3] 3 innerTail0_94 best = best := by
  rw [innerTail0_94, scan, if_neg (by decide), innerCandidates0_94_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_95
    (if 3 ≤ innerCandidates0_94.length then search edge 3 [3, 5869] innerCandidates0_94 5872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_95

def innerTail0_93 : List Nat := 5839 :: innerTail0_94
def innerCandidates0_93 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_93_checked : candidates edge 4 3 5839 innerTail0_94 best.bound = innerCandidates0_93 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_93 : scan edge (search edge 3) 4 [3] 3 innerTail0_93 best = best := by
  rw [innerTail0_93, scan, if_neg (by decide), innerCandidates0_93_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_94
    (if 3 ≤ innerCandidates0_93.length then search edge 3 [3, 5839] innerCandidates0_93 5842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_94

def innerTail0_92 : List Nat := 5801 :: innerTail0_93
def innerCandidates0_92 : List Nat := [6263]
set_option maxRecDepth 4096 in
theorem innerCandidates0_92_checked : candidates edge 4 3 5801 innerTail0_93 best.bound = innerCandidates0_92 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_92 : scan edge (search edge 3) 4 [3] 3 innerTail0_92 best = best := by
  rw [innerTail0_92, scan, if_neg (by decide), innerCandidates0_92_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_93
    (if 3 ≤ innerCandidates0_92.length then search edge 3 [3, 5801] innerCandidates0_92 5804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_93

def innerTail0_91 : List Nat := 5573 :: innerTail0_92
def innerCandidates0_91 : List Nat := [8219]
set_option maxRecDepth 4096 in
theorem innerCandidates0_91_checked : candidates edge 4 3 5573 innerTail0_92 best.bound = innerCandidates0_91 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_91 : scan edge (search edge 3) 4 [3] 3 innerTail0_91 best = best := by
  rw [innerTail0_91, scan, if_neg (by decide), innerCandidates0_91_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_92
    (if 3 ≤ innerCandidates0_91.length then search edge 3 [3, 5573] innerCandidates0_91 5576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_92

def innerTail0_90 : List Nat := 5531 :: innerTail0_91
def innerCandidates0_90 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_90_checked : candidates edge 4 3 5531 innerTail0_91 best.bound = innerCandidates0_90 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_90 : scan edge (search edge 3) 4 [3] 3 innerTail0_90 best = best := by
  rw [innerTail0_90, scan, if_neg (by decide), innerCandidates0_90_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_91
    (if 3 ≤ innerCandidates0_90.length then search edge 3 [3, 5531] innerCandidates0_90 5534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_91

def innerTail0_89 : List Nat := 5521 :: innerTail0_90
def innerCandidates0_89 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates0_89_checked : candidates edge 4 3 5521 innerTail0_90 best.bound = innerCandidates0_89 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_89 : scan edge (search edge 3) 4 [3] 3 innerTail0_89 best = best := by
  rw [innerTail0_89, scan, if_neg (by decide), innerCandidates0_89_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_90
    (if 3 ≤ innerCandidates0_89.length then search edge 3 [3, 5521] innerCandidates0_89 5524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_90

def innerTail0_88 : List Nat := 5507 :: innerTail0_89
def innerCandidates0_88 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_88_checked : candidates edge 4 3 5507 innerTail0_89 best.bound = innerCandidates0_88 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_88 : scan edge (search edge 3) 4 [3] 3 innerTail0_88 best = best := by
  rw [innerTail0_88, scan, if_neg (by decide), innerCandidates0_88_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_89
    (if 3 ≤ innerCandidates0_88.length then search edge 3 [3, 5507] innerCandidates0_88 5510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_89

def innerTail0_87 : List Nat := 5449 :: innerTail0_88
def innerCandidates0_87 : List Nat := [6073]
set_option maxRecDepth 4096 in
theorem innerCandidates0_87_checked : candidates edge 4 3 5449 innerTail0_88 best.bound = innerCandidates0_87 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_87 : scan edge (search edge 3) 4 [3] 3 innerTail0_87 best = best := by
  rw [innerTail0_87, scan, if_neg (by decide), innerCandidates0_87_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_88
    (if 3 ≤ innerCandidates0_87.length then search edge 3 [3, 5449] innerCandidates0_87 5452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_88

def innerTail0_86 : List Nat := 5419 :: innerTail0_87
def innerCandidates0_86 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_86_checked : candidates edge 4 3 5419 innerTail0_87 best.bound = innerCandidates0_86 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_86 : scan edge (search edge 3) 4 [3] 3 innerTail0_86 best = best := by
  rw [innerTail0_86, scan, if_neg (by decide), innerCandidates0_86_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_87
    (if 3 ≤ innerCandidates0_86.length then search edge 3 [3, 5419] innerCandidates0_86 5422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_87

def innerTail0_85 : List Nat := 5381 :: innerTail0_86
def innerCandidates0_85 : List Nat := [5507, 6599]
set_option maxRecDepth 4096 in
theorem innerCandidates0_85_checked : candidates edge 4 3 5381 innerTail0_86 best.bound = innerCandidates0_85 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_85 : scan edge (search edge 3) 4 [3] 3 innerTail0_85 best = best := by
  rw [innerTail0_85, scan, if_neg (by decide), innerCandidates0_85_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_86
    (if 3 ≤ innerCandidates0_85.length then search edge 3 [3, 5381] innerCandidates0_85 5384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_86

def innerTail0_84 : List Nat := 5323 :: innerTail0_85
def innerCandidates0_84 : List Nat := [5521, 6037]
set_option maxRecDepth 4096 in
theorem innerCandidates0_84_checked : candidates edge 4 3 5323 innerTail0_85 best.bound = innerCandidates0_84 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_84 : scan edge (search edge 3) 4 [3] 3 innerTail0_84 best = best := by
  rw [innerTail0_84, scan, if_neg (by decide), innerCandidates0_84_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_85
    (if 3 ≤ innerCandidates0_84.length then search edge 3 [3, 5323] innerCandidates0_84 5326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_85

def innerTail0_83 : List Nat := 5281 :: innerTail0_84
def innerCandidates0_83 : List Nat := [5869, 7159]
set_option maxRecDepth 4096 in
theorem innerCandidates0_83_checked : candidates edge 4 3 5281 innerTail0_84 best.bound = innerCandidates0_83 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_83 : scan edge (search edge 3) 4 [3] 3 innerTail0_83 best = best := by
  rw [innerTail0_83, scan, if_neg (by decide), innerCandidates0_83_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_84
    (if 3 ≤ innerCandidates0_83.length then search edge 3 [3, 5281] innerCandidates0_83 5284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_84

def innerTail0_82 : List Nat := 5171 :: innerTail0_83
def innerCandidates0_82 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_82_checked : candidates edge 4 3 5171 innerTail0_83 best.bound = innerCandidates0_82 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_82 : scan edge (search edge 3) 4 [3] 3 innerTail0_82 best = best := by
  rw [innerTail0_82, scan, if_neg (by decide), innerCandidates0_82_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_83
    (if 3 ≤ innerCandidates0_82.length then search edge 3 [3, 5171] innerCandidates0_82 5174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_83

def innerTail0_81 : List Nat := 5099 :: innerTail0_82
def innerCandidates0_81 : List Nat := [5507, 7649]
set_option maxRecDepth 4096 in
theorem innerCandidates0_81_checked : candidates edge 4 3 5099 innerTail0_82 best.bound = innerCandidates0_81 := by
  unfold candidates
  have row : edge 5099 = SearchData.row5099.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_81 : scan edge (search edge 3) 4 [3] 3 innerTail0_81 best = best := by
  rw [innerTail0_81, scan, if_neg (by decide), innerCandidates0_81_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_82
    (if 3 ≤ innerCandidates0_81.length then search edge 3 [3, 5099] innerCandidates0_81 5102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_82

def innerTail0_80 : List Nat := 5059 :: innerTail0_81
def innerCandidates0_80 : List Nat := [5323]
set_option maxRecDepth 4096 in
theorem innerCandidates0_80_checked : candidates edge 4 3 5059 innerTail0_81 best.bound = innerCandidates0_80 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_80 : scan edge (search edge 3) 4 [3] 3 innerTail0_80 best = best := by
  rw [innerTail0_80, scan, if_neg (by decide), innerCandidates0_80_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_81
    (if 3 ≤ innerCandidates0_80.length then search edge 3 [3, 5059] innerCandidates0_80 5062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_81

def innerTail0_79 : List Nat := 5051 :: innerTail0_80
def innerCandidates0_79 : List Nat := [7253]
set_option maxRecDepth 4096 in
theorem innerCandidates0_79_checked : candidates edge 4 3 5051 innerTail0_80 best.bound = innerCandidates0_79 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_79 : scan edge (search edge 3) 4 [3] 3 innerTail0_79 best = best := by
  rw [innerTail0_79, scan, if_neg (by decide), innerCandidates0_79_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_80
    (if 3 ≤ innerCandidates0_79.length then search edge 3 [3, 5051] innerCandidates0_79 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_80

def innerTail0_78 : List Nat := 4919 :: innerTail0_79
def innerCandidates0_78 : List Nat := [6599]
set_option maxRecDepth 4096 in
theorem innerCandidates0_78_checked : candidates edge 4 3 4919 innerTail0_79 best.bound = innerCandidates0_78 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_78 : scan edge (search edge 3) 4 [3] 3 innerTail0_78 best = best := by
  rw [innerTail0_78, scan, if_neg (by decide), innerCandidates0_78_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_79
    (if 3 ≤ innerCandidates0_78.length then search edge 3 [3, 4919] innerCandidates0_78 4922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_79

def innerTail0_77 : List Nat := 4729 :: innerTail0_78
def innerCandidates0_77 : List Nat := [7963]
set_option maxRecDepth 4096 in
theorem innerCandidates0_77_checked : candidates edge 4 3 4729 innerTail0_78 best.bound = innerCandidates0_77 := by
  unfold candidates
  have row : edge 4729 = SearchData.row4729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_77 : scan edge (search edge 3) 4 [3] 3 innerTail0_77 best = best := by
  rw [innerTail0_77, scan, if_neg (by decide), innerCandidates0_77_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_78
    (if 3 ≤ innerCandidates0_77.length then search edge 3 [3, 4729] innerCandidates0_77 4732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_78

def innerTail0_76 : List Nat := 4583 :: innerTail0_77
def innerCandidates0_76 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_76_checked : candidates edge 4 3 4583 innerTail0_77 best.bound = innerCandidates0_76 := by
  unfold candidates
  have row : edge 4583 = SearchData.row4583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_76 : scan edge (search edge 3) 4 [3] 3 innerTail0_76 best = best := by
  rw [innerTail0_76, scan, if_neg (by decide), innerCandidates0_76_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_77
    (if 3 ≤ innerCandidates0_76.length then search edge 3 [3, 4583] innerCandidates0_76 4586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_77

def innerTail0_75 : List Nat := 4253 :: innerTail0_76
def innerCandidates0_75 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_75_checked : candidates edge 4 3 4253 innerTail0_76 best.bound = innerCandidates0_75 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_75 : scan edge (search edge 3) 4 [3] 3 innerTail0_75 best = best := by
  rw [innerTail0_75, scan, if_neg (by decide), innerCandidates0_75_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_76
    (if 3 ≤ innerCandidates0_75.length then search edge 3 [3, 4253] innerCandidates0_75 4256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_76

def innerTail0_74 : List Nat := 4159 :: innerTail0_75
def innerCandidates0_74 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_74_checked : candidates edge 4 3 4159 innerTail0_75 best.bound = innerCandidates0_74 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_74 : scan edge (search edge 3) 4 [3] 3 innerTail0_74 best = best := by
  rw [innerTail0_74, scan, if_neg (by decide), innerCandidates0_74_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_75
    (if 3 ≤ innerCandidates0_74.length then search edge 3 [3, 4159] innerCandidates0_74 4162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_75

def innerTail0_73 : List Nat := 4019 :: innerTail0_74
def innerCandidates0_73 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_73_checked : candidates edge 4 3 4019 innerTail0_74 best.bound = innerCandidates0_73 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_73 : scan edge (search edge 3) 4 [3] 3 innerTail0_73 best = best := by
  rw [innerTail0_73, scan, if_neg (by decide), innerCandidates0_73_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_74
    (if 3 ≤ innerCandidates0_73.length then search edge 3 [3, 4019] innerCandidates0_73 4022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_74

def innerTail0_72 : List Nat := 3931 :: innerTail0_73
def innerCandidates0_72 : List Nat := [5449]
set_option maxRecDepth 4096 in
theorem innerCandidates0_72_checked : candidates edge 4 3 3931 innerTail0_73 best.bound = innerCandidates0_72 := by
  unfold candidates
  have row : edge 3931 = SearchData.row3931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_72 : scan edge (search edge 3) 4 [3] 3 innerTail0_72 best = best := by
  rw [innerTail0_72, scan, if_neg (by decide), innerCandidates0_72_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_73
    (if 3 ≤ innerCandidates0_72.length then search edge 3 [3, 3931] innerCandidates0_72 3934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_73

def innerTail0_71 : List Nat := 3923 :: innerTail0_72
def innerCandidates0_71 : List Nat := [4919, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates0_71_checked : candidates edge 4 3 3923 innerTail0_72 best.bound = innerCandidates0_71 := by
  unfold candidates
  have row : edge 3923 = SearchData.row3923.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_71 : scan edge (search edge 3) 4 [3] 3 innerTail0_71 best = best := by
  rw [innerTail0_71, scan, if_neg (by decide), innerCandidates0_71_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_72
    (if 3 ≤ innerCandidates0_71.length then search edge 3 [3, 3923] innerCandidates0_71 3926 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_72

def innerTail0_70 : List Nat := 3911 :: innerTail0_71
def innerCandidates0_70 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_70_checked : candidates edge 4 3 3911 innerTail0_71 best.bound = innerCandidates0_70 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_70 : scan edge (search edge 3) 4 [3] 3 innerTail0_70 best = best := by
  rw [innerTail0_70, scan, if_neg (by decide), innerCandidates0_70_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_71
    (if 3 ≤ innerCandidates0_70.length then search edge 3 [3, 3911] innerCandidates0_70 3914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_71

def innerTail0_69 : List Nat := 3769 :: innerTail0_70
def innerCandidates0_69 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_69_checked : candidates edge 4 3 3769 innerTail0_70 best.bound = innerCandidates0_69 := by
  unfold candidates
  have row : edge 3769 = SearchData.row3769.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_69 : scan edge (search edge 3) 4 [3] 3 innerTail0_69 best = best := by
  rw [innerTail0_69, scan, if_neg (by decide), innerCandidates0_69_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_70
    (if 3 ≤ innerCandidates0_69.length then search edge 3 [3, 3769] innerCandidates0_69 3772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_70

def innerTail0_68 : List Nat := 3637 :: innerTail0_69
def innerCandidates0_68 : List Nat := [5839, 6871]
set_option maxRecDepth 4096 in
theorem innerCandidates0_68_checked : candidates edge 4 3 3637 innerTail0_69 best.bound = innerCandidates0_68 := by
  unfold candidates
  have row : edge 3637 = SearchData.row3637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_68 : scan edge (search edge 3) 4 [3] 3 innerTail0_68 best = best := by
  rw [innerTail0_68, scan, if_neg (by decide), innerCandidates0_68_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_69
    (if 3 ≤ innerCandidates0_68.length then search edge 3 [3, 3637] innerCandidates0_68 3640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_69

def innerTail0_67 : List Nat := 3469 :: innerTail0_68
def innerCandidates0_67 : List Nat := [5839]
set_option maxRecDepth 4096 in
theorem innerCandidates0_67_checked : candidates edge 4 3 3469 innerTail0_68 best.bound = innerCandidates0_67 := by
  unfold candidates
  have row : edge 3469 = SearchData.row3469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_67 : scan edge (search edge 3) 4 [3] 3 innerTail0_67 best = best := by
  rw [innerTail0_67, scan, if_neg (by decide), innerCandidates0_67_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_68
    (if 3 ≤ innerCandidates0_67.length then search edge 3 [3, 3469] innerCandidates0_67 3472 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_68

def innerTail0_66 : List Nat := 3461 :: innerTail0_67
def innerCandidates0_66 : List Nat := [4583]
set_option maxRecDepth 4096 in
theorem innerCandidates0_66_checked : candidates edge 4 3 3461 innerTail0_67 best.bound = innerCandidates0_66 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_66 : scan edge (search edge 3) 4 [3] 3 innerTail0_66 best = best := by
  rw [innerTail0_66, scan, if_neg (by decide), innerCandidates0_66_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_67
    (if 3 ≤ innerCandidates0_66.length then search edge 3 [3, 3461] innerCandidates0_66 3464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_67

def innerTail0_65 : List Nat := 3301 :: innerTail0_66
def innerCandidates0_65 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_65_checked : candidates edge 4 3 3301 innerTail0_66 best.bound = innerCandidates0_65 := by
  unfold candidates
  have row : edge 3301 = SearchData.row3301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_65 : scan edge (search edge 3) 4 [3] 3 innerTail0_65 best = best := by
  rw [innerTail0_65, scan, if_neg (by decide), innerCandidates0_65_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_66
    (if 3 ≤ innerCandidates0_65.length then search edge 3 [3, 3301] innerCandidates0_65 3304 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_66

def innerTail0_64 : List Nat := 3119 :: innerTail0_65
def innerCandidates0_64 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_64_checked : candidates edge 4 3 3119 innerTail0_65 best.bound = innerCandidates0_64 := by
  unfold candidates
  have row : edge 3119 = SearchData.row3119.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_64 : scan edge (search edge 3) 4 [3] 3 innerTail0_64 best = best := by
  rw [innerTail0_64, scan, if_neg (by decide), innerCandidates0_64_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_65
    (if 3 ≤ innerCandidates0_64.length then search edge 3 [3, 3119] innerCandidates0_64 3122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_65

def innerTail0_63 : List Nat := 3049 :: innerTail0_64
def innerCandidates0_63 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_63_checked : candidates edge 4 3 3049 innerTail0_64 best.bound = innerCandidates0_63 := by
  unfold candidates
  have row : edge 3049 = SearchData.row3049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_63 : scan edge (search edge 3) 4 [3] 3 innerTail0_63 best = best := by
  rw [innerTail0_63, scan, if_neg (by decide), innerCandidates0_63_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_64
    (if 3 ≤ innerCandidates0_63.length then search edge 3 [3, 3049] innerCandidates0_63 3052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_64

def innerTail0_62 : List Nat := 2957 :: innerTail0_63
def innerCandidates0_62 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_62_checked : candidates edge 4 3 2957 innerTail0_63 best.bound = innerCandidates0_62 := by
  unfold candidates
  have row : edge 2957 = SearchData.row2957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_62 : scan edge (search edge 3) 4 [3] 3 innerTail0_62 best = best := by
  rw [innerTail0_62, scan, if_neg (by decide), innerCandidates0_62_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_63
    (if 3 ≤ innerCandidates0_62.length then search edge 3 [3, 2957] innerCandidates0_62 2960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_63

def innerTail0_61 : List Nat := 2917 :: innerTail0_62
def innerCandidates0_61 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_61_checked : candidates edge 4 3 2917 innerTail0_62 best.bound = innerCandidates0_61 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_61 : scan edge (search edge 3) 4 [3] 3 innerTail0_61 best = best := by
  rw [innerTail0_61, scan, if_neg (by decide), innerCandidates0_61_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_62
    (if 3 ≤ innerCandidates0_61.length then search edge 3 [3, 2917] innerCandidates0_61 2920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_62

def innerTail0_60 : List Nat := 2843 :: innerTail0_61
def innerCandidates0_60 : List Nat := [4253, 6899]
set_option maxRecDepth 4096 in
theorem innerCandidates0_60_checked : candidates edge 4 3 2843 innerTail0_61 best.bound = innerCandidates0_60 := by
  unfold candidates
  have row : edge 2843 = SearchData.row2843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_60 : scan edge (search edge 3) 4 [3] 3 innerTail0_60 best = best := by
  rw [innerTail0_60, scan, if_neg (by decide), innerCandidates0_60_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_61
    (if 3 ≤ innerCandidates0_60.length then search edge 3 [3, 2843] innerCandidates0_60 2846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_61

def innerTail0_59 : List Nat := 2789 :: innerTail0_60
def innerCandidates0_59 : List Nat := [3461, 5051, 6563, 7691]
set_option maxRecDepth 4096 in
theorem innerCandidates0_59_checked : candidates edge 4 3 2789 innerTail0_60 best.bound = innerCandidates0_59 := by
  unfold candidates
  have row : edge 2789 = SearchData.row2789.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_59 : search edge 3 [3, 2789] innerCandidates0_59 2792 best = best := by decide

theorem innerScan0_59 : scan edge (search edge 3) 4 [3] 3 innerTail0_59 best = best := by
  rw [innerTail0_59, scan, if_neg (by decide), innerCandidates0_59_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_60
    (if 3 ≤ innerCandidates0_59.length then search edge 3 [3, 2789] innerCandidates0_59 2792 best else best) = best
  rw [if_pos (by decide), innerChild0_59]
  exact innerScan0_60

def innerTail0_58 : List Nat := 2707 :: innerTail0_59
def innerCandidates0_58 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_58_checked : candidates edge 4 3 2707 innerTail0_59 best.bound = innerCandidates0_58 := by
  unfold candidates
  have row : edge 2707 = SearchData.row2707.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_58 : scan edge (search edge 3) 4 [3] 3 innerTail0_58 best = best := by
  rw [innerTail0_58, scan, if_neg (by decide), innerCandidates0_58_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_59
    (if 3 ≤ innerCandidates0_58.length then search edge 3 [3, 2707] innerCandidates0_58 2710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_59

def innerTail0_57 : List Nat := 2579 :: innerTail0_58
def innerCandidates0_57 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_57_checked : candidates edge 4 3 2579 innerTail0_58 best.bound = innerCandidates0_57 := by
  unfold candidates
  have row : edge 2579 = PrimePairCache.smallRow2579.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_57 : scan edge (search edge 3) 4 [3] 3 innerTail0_57 best = best := by
  rw [innerTail0_57, scan, if_neg (by decide), innerCandidates0_57_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_58
    (if 3 ≤ innerCandidates0_57.length then search edge 3 [3, 2579] innerCandidates0_57 2582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_58

def innerTail0_56 : List Nat := 2503 :: innerTail0_57
def innerCandidates0_56 : List Nat := [5281, 5869, 7591]
set_option maxRecDepth 4096 in
theorem innerCandidates0_56_checked : candidates edge 4 3 2503 innerTail0_57 best.bound = innerCandidates0_56 := by
  unfold candidates
  have row : edge 2503 = PrimePairCache.smallRow2503.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_56 : search edge 3 [3, 2503] innerCandidates0_56 2506 best = best := by decide

theorem innerScan0_56 : scan edge (search edge 3) 4 [3] 3 innerTail0_56 best = best := by
  rw [innerTail0_56, scan, if_neg (by decide), innerCandidates0_56_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_57
    (if 3 ≤ innerCandidates0_56.length then search edge 3 [3, 2503] innerCandidates0_56 2506 best else best) = best
  rw [if_pos (by decide), innerChild0_56]
  exact innerScan0_57

def innerTail0_55 : List Nat := 2441 :: innerTail0_56
def innerCandidates0_55 : List Nat := [5381, 7691]
set_option maxRecDepth 4096 in
theorem innerCandidates0_55_checked : candidates edge 4 3 2441 innerTail0_56 best.bound = innerCandidates0_55 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_55 : scan edge (search edge 3) 4 [3] 3 innerTail0_55 best = best := by
  rw [innerTail0_55, scan, if_neg (by decide), innerCandidates0_55_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_56
    (if 3 ≤ innerCandidates0_55.length then search edge 3 [3, 2441] innerCandidates0_55 2444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_56

def innerTail0_54 : List Nat := 2411 :: innerTail0_55
def innerCandidates0_54 : List Nat := [4253, 4919, 5171]
set_option maxRecDepth 4096 in
theorem innerCandidates0_54_checked : candidates edge 4 3 2411 innerTail0_55 best.bound = innerCandidates0_54 := by
  unfold candidates
  have row : edge 2411 = PrimePairCache.smallRow2411.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_54 : search edge 3 [3, 2411] innerCandidates0_54 2414 best = best := by decide

theorem innerScan0_54 : scan edge (search edge 3) 4 [3] 3 innerTail0_54 best = best := by
  rw [innerTail0_54, scan, if_neg (by decide), innerCandidates0_54_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_55
    (if 3 ≤ innerCandidates0_54.length then search edge 3 [3, 2411] innerCandidates0_54 2414 best else best) = best
  rw [if_pos (by decide), innerChild0_54]
  exact innerScan0_55

def innerTail0_53 : List Nat := 2381 :: innerTail0_54
def innerCandidates0_53 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates0_53_checked : candidates edge 4 3 2381 innerTail0_54 best.bound = innerCandidates0_53 := by
  unfold candidates
  have row : edge 2381 = PrimePairCache.smallRow2381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_53 : scan edge (search edge 3) 4 [3] 3 innerTail0_53 best = best := by
  rw [innerTail0_53, scan, if_neg (by decide), innerCandidates0_53_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_54
    (if 3 ≤ innerCandidates0_53.length then search edge 3 [3, 2381] innerCandidates0_53 2384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_54

def innerTail0_52 : List Nat := 2377 :: innerTail0_53
def innerCandidates0_52 : List Nat := [4159, 6073]
set_option maxRecDepth 4096 in
theorem innerCandidates0_52_checked : candidates edge 4 3 2377 innerTail0_53 best.bound = innerCandidates0_52 := by
  unfold candidates
  have row : edge 2377 = PrimePairCache.smallRow2377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_52 : scan edge (search edge 3) 4 [3] 3 innerTail0_52 best = best := by
  rw [innerTail0_52, scan, if_neg (by decide), innerCandidates0_52_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_53
    (if 3 ≤ innerCandidates0_52.length then search edge 3 [3, 2377] innerCandidates0_52 2380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_53

def innerTail0_51 : List Nat := 2297 :: innerTail0_52
def innerCandidates0_51 : List Nat := [2381]
set_option maxRecDepth 4096 in
theorem innerCandidates0_51_checked : candidates edge 4 3 2297 innerTail0_52 best.bound = innerCandidates0_51 := by
  unfold candidates
  have row : edge 2297 = PrimePairCache.smallRow2297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_51 : scan edge (search edge 3) 4 [3] 3 innerTail0_51 best = best := by
  rw [innerTail0_51, scan, if_neg (by decide), innerCandidates0_51_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_52
    (if 3 ≤ innerCandidates0_51.length then search edge 3 [3, 2297] innerCandidates0_51 2300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_52

def innerTail0_50 : List Nat := 2213 :: innerTail0_51
def innerCandidates0_50 : List Nat := [4253]
set_option maxRecDepth 4096 in
theorem innerCandidates0_50_checked : candidates edge 4 3 2213 innerTail0_51 best.bound = innerCandidates0_50 := by
  unfold candidates
  have row : edge 2213 = PrimePairCache.smallRow2213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_50 : scan edge (search edge 3) 4 [3] 3 innerTail0_50 best = best := by
  rw [innerTail0_50, scan, if_neg (by decide), innerCandidates0_50_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_51
    (if 3 ≤ innerCandidates0_50.length then search edge 3 [3, 2213] innerCandidates0_50 2216 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_51

def innerTail0_49 : List Nat := 2143 :: innerTail0_50
def innerCandidates0_49 : List Nat := [6871]
set_option maxRecDepth 4096 in
theorem innerCandidates0_49_checked : candidates edge 4 3 2143 innerTail0_50 best.bound = innerCandidates0_49 := by
  unfold candidates
  have row : edge 2143 = PrimePairCache.smallRow2143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_49 : scan edge (search edge 3) 4 [3] 3 innerTail0_49 best = best := by
  rw [innerTail0_49, scan, if_neg (by decide), innerCandidates0_49_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_50
    (if 3 ≤ innerCandidates0_49.length then search edge 3 [3, 2143] innerCandidates0_49 2146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_50

def innerTail0_48 : List Nat := 2069 :: innerTail0_49
def innerCandidates0_48 : List Nat := [2297, 7253, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates0_48_checked : candidates edge 4 3 2069 innerTail0_49 best.bound = innerCandidates0_48 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_48 : search edge 3 [3, 2069] innerCandidates0_48 2072 best = best := by decide

theorem innerScan0_48 : scan edge (search edge 3) 4 [3] 3 innerTail0_48 best = best := by
  rw [innerTail0_48, scan, if_neg (by decide), innerCandidates0_48_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_49
    (if 3 ≤ innerCandidates0_48.length then search edge 3 [3, 2069] innerCandidates0_48 2072 best else best) = best
  rw [if_pos (by decide), innerChild0_48]
  exact innerScan0_49

def innerTail0_47 : List Nat := 1907 :: innerTail0_48
def innerCandidates0_47 : List Nat := [5099]
set_option maxRecDepth 4096 in
theorem innerCandidates0_47_checked : candidates edge 4 3 1907 innerTail0_48 best.bound = innerCandidates0_47 := by
  unfold candidates
  have row : edge 1907 = PrimePairCache.smallRow1907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_47 : scan edge (search edge 3) 4 [3] 3 innerTail0_47 best = best := by
  rw [innerTail0_47, scan, if_neg (by decide), innerCandidates0_47_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_48
    (if 3 ≤ innerCandidates0_47.length then search edge 3 [3, 1907] innerCandidates0_47 1910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_48

def innerTail0_46 : List Nat := 1699 :: innerTail0_47
def innerCandidates0_46 : List Nat := [6571]
set_option maxRecDepth 4096 in
theorem innerCandidates0_46_checked : candidates edge 4 3 1699 innerTail0_47 best.bound = innerCandidates0_46 := by
  unfold candidates
  have row : edge 1699 = PrimePairCache.smallRow1699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_46 : scan edge (search edge 3) 4 [3] 3 innerTail0_46 best = best := by
  rw [innerTail0_46, scan, if_neg (by decide), innerCandidates0_46_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_47
    (if 3 ≤ innerCandidates0_46.length then search edge 3 [3, 1699] innerCandidates0_46 1702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_47

def innerTail0_45 : List Nat := 1667 :: innerTail0_46
def innerCandidates0_45 : List Nat := [5051, 5531, 5573, 6353]
set_option maxRecDepth 4096 in
theorem innerCandidates0_45_checked : candidates edge 4 3 1667 innerTail0_46 best.bound = innerCandidates0_45 := by
  unfold candidates
  have row : edge 1667 = PrimePairCache.smallRow1667.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_45 : search edge 3 [3, 1667] innerCandidates0_45 1670 best = best := by decide

theorem innerScan0_45 : scan edge (search edge 3) 4 [3] 3 innerTail0_45 best = best := by
  rw [innerTail0_45, scan, if_neg (by decide), innerCandidates0_45_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_46
    (if 3 ≤ innerCandidates0_45.length then search edge 3 [3, 1667] innerCandidates0_45 1670 best else best) = best
  rw [if_pos (by decide), innerChild0_45]
  exact innerScan0_46

def innerTail0_44 : List Nat := 1663 :: innerTail0_45
def innerCandidates0_44 : List Nat := [5449, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates0_44_checked : candidates edge 4 3 1663 innerTail0_45 best.bound = innerCandidates0_44 := by
  unfold candidates
  have row : edge 1663 = PrimePairCache.smallRow1663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_44 : scan edge (search edge 3) 4 [3] 3 innerTail0_44 best = best := by
  rw [innerTail0_44, scan, if_neg (by decide), innerCandidates0_44_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_45
    (if 3 ≤ innerCandidates0_44.length then search edge 3 [3, 1663] innerCandidates0_44 1666 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_45

def innerTail0_43 : List Nat := 1657 :: innerTail0_44
def innerCandidates0_43 : List Nat := [1699, 2917, 5059]
set_option maxRecDepth 4096 in
theorem innerCandidates0_43_checked : candidates edge 4 3 1657 innerTail0_44 best.bound = innerCandidates0_43 := by
  unfold candidates
  have row : edge 1657 = PrimePairCache.smallRow1657.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_43 : search edge 3 [3, 1657] innerCandidates0_43 1660 best = best := by decide

theorem innerScan0_43 : scan edge (search edge 3) 4 [3] 3 innerTail0_43 best = best := by
  rw [innerTail0_43, scan, if_neg (by decide), innerCandidates0_43_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_44
    (if 3 ≤ innerCandidates0_43.length then search edge 3 [3, 1657] innerCandidates0_43 1660 best else best) = best
  rw [if_pos (by decide), innerChild0_43]
  exact innerScan0_44

def innerTail0_42 : List Nat := 1627 :: innerTail0_43
def innerCandidates0_42 : List Nat := [2917, 3637, 3769, 6277, 7159]
set_option maxRecDepth 4096 in
theorem innerCandidates0_42_checked : candidates edge 4 3 1627 innerTail0_43 best.bound = innerCandidates0_42 := by
  unfold candidates
  have row : edge 1627 = PrimePairCache.smallRow1627.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_42 : search edge 3 [3, 1627] innerCandidates0_42 1630 best = best := by decide

theorem innerScan0_42 : scan edge (search edge 3) 4 [3] 3 innerTail0_42 best = best := by
  rw [innerTail0_42, scan, if_neg (by decide), innerCandidates0_42_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_43
    (if 3 ≤ innerCandidates0_42.length then search edge 3 [3, 1627] innerCandidates0_42 1630 best else best) = best
  rw [if_pos (by decide), innerChild0_42]
  exact innerScan0_43

def innerTail0_41 : List Nat := 1607 :: innerTail0_42
def innerCandidates0_41 : List Nat := [5099]
set_option maxRecDepth 4096 in
theorem innerCandidates0_41_checked : candidates edge 4 3 1607 innerTail0_42 best.bound = innerCandidates0_41 := by
  unfold candidates
  have row : edge 1607 = PrimePairCache.smallRow1607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_41 : scan edge (search edge 3) 4 [3] 3 innerTail0_41 best = best := by
  rw [innerTail0_41, scan, if_neg (by decide), innerCandidates0_41_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_42
    (if 3 ≤ innerCandidates0_41.length then search edge 3 [3, 1607] innerCandidates0_41 1610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_42

def innerTail0_40 : List Nat := 1531 :: innerTail0_41
def innerCandidates0_40 : List Nat := [5059, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates0_40_checked : candidates edge 4 3 1531 innerTail0_41 best.bound = innerCandidates0_40 := by
  unfold candidates
  have row : edge 1531 = PrimePairCache.smallRow1531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_40 : scan edge (search edge 3) 4 [3] 3 innerTail0_40 best = best := by
  rw [innerTail0_40, scan, if_neg (by decide), innerCandidates0_40_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_41
    (if 3 ≤ innerCandidates0_40.length then search edge 3 [3, 1531] innerCandidates0_40 1534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_41

def innerTail0_39 : List Nat := 1481 :: innerTail0_40
def innerCandidates0_39 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_39_checked : candidates edge 4 3 1481 innerTail0_40 best.bound = innerCandidates0_39 := by
  unfold candidates
  have row : edge 1481 = PrimePairCache.smallRow1481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_39 : scan edge (search edge 3) 4 [3] 3 innerTail0_39 best = best := by
  rw [innerTail0_39, scan, if_neg (by decide), innerCandidates0_39_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_40
    (if 3 ≤ innerCandidates0_39.length then search edge 3 [3, 1481] innerCandidates0_39 1484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_40

def innerTail0_38 : List Nat := 1237 :: innerTail0_39
def innerCandidates0_38 : List Nat := [1657, 6571]
set_option maxRecDepth 4096 in
theorem innerCandidates0_38_checked : candidates edge 4 3 1237 innerTail0_39 best.bound = innerCandidates0_38 := by
  unfold candidates
  have row : edge 1237 = PrimePairCache.smallRow1237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_38 : scan edge (search edge 3) 4 [3] 3 innerTail0_38 best = best := by
  rw [innerTail0_38, scan, if_neg (by decide), innerCandidates0_38_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_39
    (if 3 ≤ innerCandidates0_38.length then search edge 3 [3, 1237] innerCandidates0_38 1240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_39

def innerTail0_37 : List Nat := 1193 :: innerTail0_38
def innerCandidates0_37 : List Nat := [2441, 3923, 5507, 5801]
set_option maxRecDepth 4096 in
theorem innerCandidates0_37_checked : candidates edge 4 3 1193 innerTail0_38 best.bound = innerCandidates0_37 := by
  unfold candidates
  have row : edge 1193 = PrimePairCache.smallRow1193.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_37 : search edge 3 [3, 1193] innerCandidates0_37 1196 best = best := by decide

theorem innerScan0_37 : scan edge (search edge 3) 4 [3] 3 innerTail0_37 best = best := by
  rw [innerTail0_37, scan, if_neg (by decide), innerCandidates0_37_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_38
    (if 3 ≤ innerCandidates0_37.length then search edge 3 [3, 1193] innerCandidates0_37 1196 best else best) = best
  rw [if_pos (by decide), innerChild0_37]
  exact innerScan0_38

def innerTail0_36 : List Nat := 1181 :: innerTail0_37
def innerCandidates0_36 : List Nat := [5171, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates0_36_checked : candidates edge 4 3 1181 innerTail0_37 best.bound = innerCandidates0_36 := by
  unfold candidates
  have row : edge 1181 = PrimePairCache.smallRow1181.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_36 : scan edge (search edge 3) 4 [3] 3 innerTail0_36 best = best := by
  rw [innerTail0_36, scan, if_neg (by decide), innerCandidates0_36_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_37
    (if 3 ≤ innerCandidates0_36.length then search edge 3 [3, 1181] innerCandidates0_36 1184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_37

def innerTail0_35 : List Nat := 1051 :: innerTail0_36
def innerCandidates0_35 : List Nat := [4729, 5839]
set_option maxRecDepth 4096 in
theorem innerCandidates0_35_checked : candidates edge 4 3 1051 innerTail0_36 best.bound = innerCandidates0_35 := by
  unfold candidates
  have row : edge 1051 = PrimePairCache.smallRow1051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_35 : scan edge (search edge 3) 4 [3] 3 innerTail0_35 best = best := by
  rw [innerTail0_35, scan, if_neg (by decide), innerCandidates0_35_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_36
    (if 3 ≤ innerCandidates0_35.length then search edge 3 [3, 1051] innerCandidates0_35 1054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_36

def innerTail0_34 : List Nat := 1033 :: innerTail0_35
def innerCandidates0_34 : List Nat := [1663, 5869, 6451]
set_option maxRecDepth 4096 in
theorem innerCandidates0_34_checked : candidates edge 4 3 1033 innerTail0_35 best.bound = innerCandidates0_34 := by
  unfold candidates
  have row : edge 1033 = PrimePairCache.smallRow1033.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_34 : search edge 3 [3, 1033] innerCandidates0_34 1036 best = best := by decide

theorem innerScan0_34 : scan edge (search edge 3) 4 [3] 3 innerTail0_34 best = best := by
  rw [innerTail0_34, scan, if_neg (by decide), innerCandidates0_34_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_35
    (if 3 ≤ innerCandidates0_34.length then search edge 3 [3, 1033] innerCandidates0_34 1036 best else best) = best
  rw [if_pos (by decide), innerChild0_34]
  exact innerScan0_35

def innerTail0_33 : List Nat := 1019 :: innerTail0_34
def innerCandidates0_33 : List Nat := [2957, 6947]
set_option maxRecDepth 4096 in
theorem innerCandidates0_33_checked : candidates edge 4 3 1019 innerTail0_34 best.bound = innerCandidates0_33 := by
  unfold candidates
  have row : edge 1019 = PrimePairCache.smallRow1019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_33 : scan edge (search edge 3) 4 [3] 3 innerTail0_33 best = best := by
  rw [innerTail0_33, scan, if_neg (by decide), innerCandidates0_33_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_34
    (if 3 ≤ innerCandidates0_33.length then search edge 3 [3, 1019] innerCandidates0_33 1022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_34

def innerTail0_32 : List Nat := 1013 :: innerTail0_33
def innerCandidates0_32 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_32_checked : candidates edge 4 3 1013 innerTail0_33 best.bound = innerCandidates0_32 := by
  unfold candidates
  have row : edge 1013 = PrimePairCache.smallRow1013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_32 : scan edge (search edge 3) 4 [3] 3 innerTail0_32 best = best := by
  rw [innerTail0_32, scan, if_neg (by decide), innerCandidates0_32_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_33
    (if 3 ≤ innerCandidates0_32.length then search edge 3 [3, 1013] innerCandidates0_32 1016 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_33

def innerTail0_31 : List Nat := 947 :: innerTail0_32
def innerCandidates0_31 : List Nat := [5051, 6947]
set_option maxRecDepth 4096 in
theorem innerCandidates0_31_checked : candidates edge 4 3 947 innerTail0_32 best.bound = innerCandidates0_31 := by
  unfold candidates
  have row : edge 947 = PrimePairCache.smallRow947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_31 : scan edge (search edge 3) 4 [3] 3 innerTail0_31 best = best := by
  rw [innerTail0_31, scan, if_neg (by decide), innerCandidates0_31_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_32
    (if 3 ≤ innerCandidates0_31.length then search edge 3 [3, 947] innerCandidates0_31 950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_32

def innerTail0_30 : List Nat := 929 :: innerTail0_31
def innerCandidates0_30 : List Nat := [6563]
set_option maxRecDepth 4096 in
theorem innerCandidates0_30_checked : candidates edge 4 3 929 innerTail0_31 best.bound = innerCandidates0_30 := by
  unfold candidates
  have row : edge 929 = PrimePairCache.smallRow929.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_30 : scan edge (search edge 3) 4 [3] 3 innerTail0_30 best = best := by
  rw [innerTail0_30, scan, if_neg (by decide), innerCandidates0_30_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_31
    (if 3 ≤ innerCandidates0_30.length then search edge 3 [3, 929] innerCandidates0_30 932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_31

def innerTail0_29 : List Nat := 823 :: innerTail0_30
def innerCandidates0_29 : List Nat := [5281, 5323]
set_option maxRecDepth 4096 in
theorem innerCandidates0_29_checked : candidates edge 4 3 823 innerTail0_30 best.bound = innerCandidates0_29 := by
  unfold candidates
  have row : edge 823 = PrimePairCache.smallRow823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_29 : scan edge (search edge 3) 4 [3] 3 innerTail0_29 best = best := by
  rw [innerTail0_29, scan, if_neg (by decide), innerCandidates0_29_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_30
    (if 3 ≤ innerCandidates0_29.length then search edge 3 [3, 823] innerCandidates0_29 826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_30

def innerTail0_28 : List Nat := 739 :: innerTail0_29
def innerCandidates0_28 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates0_28_checked : candidates edge 4 3 739 innerTail0_29 best.bound = innerCandidates0_28 := by
  unfold candidates
  have row : edge 739 = PrimePairCache.smallRow739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_28 : scan edge (search edge 3) 4 [3] 3 innerTail0_28 best = best := by
  rw [innerTail0_28, scan, if_neg (by decide), innerCandidates0_28_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_29
    (if 3 ≤ innerCandidates0_28.length then search edge 3 [3, 739] innerCandidates0_28 742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_29

def innerTail0_27 : List Nat := 733 :: innerTail0_28
def innerCandidates0_27 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_27_checked : candidates edge 4 3 733 innerTail0_28 best.bound = innerCandidates0_27 := by
  unfold candidates
  have row : edge 733 = PrimePairCache.smallRow733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_27 : scan edge (search edge 3) 4 [3] 3 innerTail0_27 best = best := by
  rw [innerTail0_27, scan, if_neg (by decide), innerCandidates0_27_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_28
    (if 3 ≤ innerCandidates0_27.length then search edge 3 [3, 733] innerCandidates0_27 736 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_28

def innerTail0_26 : List Nat := 719 :: innerTail0_27
def innerCandidates0_26 : List Nat := [947, 2579]
set_option maxRecDepth 4096 in
theorem innerCandidates0_26_checked : candidates edge 4 3 719 innerTail0_27 best.bound = innerCandidates0_26 := by
  unfold candidates
  have row : edge 719 = PrimePairCache.smallRow719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_26 : scan edge (search edge 3) 4 [3] 3 innerTail0_26 best = best := by
  rw [innerTail0_26, scan, if_neg (by decide), innerCandidates0_26_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_27
    (if 3 ≤ innerCandidates0_26.length then search edge 3 [3, 719] innerCandidates0_26 722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_27

def innerTail0_25 : List Nat := 701 :: innerTail0_26
def innerCandidates0_25 : List Nat := [1181, 1607, 4583]
set_option maxRecDepth 4096 in
theorem innerCandidates0_25_checked : candidates edge 4 3 701 innerTail0_26 best.bound = innerCandidates0_25 := by
  unfold candidates
  have row : edge 701 = PrimePairCache.smallRow701.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_25 : search edge 3 [3, 701] innerCandidates0_25 704 best = best := by decide

theorem innerScan0_25 : scan edge (search edge 3) 4 [3] 3 innerTail0_25 best = best := by
  rw [innerTail0_25, scan, if_neg (by decide), innerCandidates0_25_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_26
    (if 3 ≤ innerCandidates0_25.length then search edge 3 [3, 701] innerCandidates0_25 704 best else best) = best
  rw [if_pos (by decide), innerChild0_25]
  exact innerScan0_26

def innerTail0_24 : List Nat := 673 :: innerTail0_25
def innerCandidates0_24 : List Nat := [1657]
set_option maxRecDepth 4096 in
theorem innerCandidates0_24_checked : candidates edge 4 3 673 innerTail0_25 best.bound = innerCandidates0_24 := by
  unfold candidates
  have row : edge 673 = PrimePairCache.smallRow673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_24 : scan edge (search edge 3) 4 [3] 3 innerTail0_24 best = best := by
  rw [innerTail0_24, scan, if_neg (by decide), innerCandidates0_24_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_25
    (if 3 ≤ innerCandidates0_24.length then search edge 3 [3, 673] innerCandidates0_24 676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_25

def innerTail0_23 : List Nat := 617 :: innerTail0_24
def innerCandidates0_23 : List Nat := [4253]
set_option maxRecDepth 4096 in
theorem innerCandidates0_23_checked : candidates edge 4 3 617 innerTail0_24 best.bound = innerCandidates0_23 := by
  unfold candidates
  have row : edge 617 = PrimePairCache.smallRow617.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_23 : scan edge (search edge 3) 4 [3] 3 innerTail0_23 best = best := by
  rw [innerTail0_23, scan, if_neg (by decide), innerCandidates0_23_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_24
    (if 3 ≤ innerCandidates0_23.length then search edge 3 [3, 617] innerCandidates0_23 620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_24

def innerTail0_22 : List Nat := 613 :: innerTail0_23
def innerCandidates0_22 : List Nat := [673, 3769]
set_option maxRecDepth 4096 in
theorem innerCandidates0_22_checked : candidates edge 4 3 613 innerTail0_23 best.bound = innerCandidates0_22 := by
  unfold candidates
  have row : edge 613 = PrimePairCache.smallRow613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_22 : scan edge (search edge 3) 4 [3] 3 innerTail0_22 best = best := by
  rw [innerTail0_22, scan, if_neg (by decide), innerCandidates0_22_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_23
    (if 3 ≤ innerCandidates0_22.length then search edge 3 [3, 613] innerCandidates0_22 616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_23

def innerTail0_21 : List Nat := 607 :: innerTail0_22
def innerCandidates0_21 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates0_21_checked : candidates edge 4 3 607 innerTail0_22 best.bound = innerCandidates0_21 := by
  unfold candidates
  have row : edge 607 = PrimePairCache.smallRow607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_21 : scan edge (search edge 3) 4 [3] 3 innerTail0_21 best = best := by
  rw [innerTail0_21, scan, if_neg (by decide), innerCandidates0_21_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_22
    (if 3 ≤ innerCandidates0_21.length then search edge 3 [3, 607] innerCandidates0_21 610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_22

def innerTail0_20 : List Nat := 557 :: innerTail0_21
def innerCandidates0_20 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates0_20_checked : candidates edge 4 3 557 innerTail0_21 best.bound = innerCandidates0_20 := by
  unfold candidates
  have row : edge 557 = PrimePairCache.smallRow557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_20 : scan edge (search edge 3) 4 [3] 3 innerTail0_20 best = best := by
  rw [innerTail0_20, scan, if_neg (by decide), innerCandidates0_20_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_21
    (if 3 ≤ innerCandidates0_20.length then search edge 3 [3, 557] innerCandidates0_20 560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_21

def innerTail0_19 : List Nat := 541 :: innerTail0_20
def innerCandidates0_19 : List Nat := [1033, 3301, 4159, 7369]
set_option maxRecDepth 4096 in
theorem innerCandidates0_19_checked : candidates edge 4 3 541 innerTail0_20 best.bound = innerCandidates0_19 := by
  unfold candidates
  have row : edge 541 = PrimePairCache.smallRow541.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_19 : search edge 3 [3, 541] innerCandidates0_19 544 best = best := by decide

theorem innerScan0_19 : scan edge (search edge 3) 4 [3] 3 innerTail0_19 best = best := by
  rw [innerTail0_19, scan, if_neg (by decide), innerCandidates0_19_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_20
    (if 3 ≤ innerCandidates0_19.length then search edge 3 [3, 541] innerCandidates0_19 544 best else best) = best
  rw [if_pos (by decide), innerChild0_19]
  exact innerScan0_20

def innerTail0_18 : List Nat := 499 :: innerTail0_19
def innerCandidates0_18 : List Nat := [673, 1033, 1237, 2707, 5281]
set_option maxRecDepth 4096 in
theorem innerCandidates0_18_checked : candidates edge 4 3 499 innerTail0_19 best.bound = innerCandidates0_18 := by
  unfold candidates
  have row : edge 499 = PrimePairCache.smallRow499.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_18 : search edge 3 [3, 499] innerCandidates0_18 502 best = best := by decide

theorem innerScan0_18 : scan edge (search edge 3) 4 [3] 3 innerTail0_18 best = best := by
  rw [innerTail0_18, scan, if_neg (by decide), innerCandidates0_18_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_19
    (if 3 ≤ innerCandidates0_18.length then search edge 3 [3, 499] innerCandidates0_18 502 best else best) = best
  rw [if_pos (by decide), innerChild0_18]
  exact innerScan0_19

def innerTail0_17 : List Nat := 467 :: innerTail0_18
def innerCandidates0_17 : List Nat := [617, 4253]
set_option maxRecDepth 4096 in
theorem innerCandidates0_17_checked : candidates edge 4 3 467 innerTail0_18 best.bound = innerCandidates0_17 := by
  unfold candidates
  have row : edge 467 = PrimePairCache.smallRow467.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_17 : scan edge (search edge 3) 4 [3] 3 innerTail0_17 best = best := by
  rw [innerTail0_17, scan, if_neg (by decide), innerCandidates0_17_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_18
    (if 3 ≤ innerCandidates0_17.length then search edge 3 [3, 467] innerCandidates0_17 470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_18

def innerTail0_16 : List Nat := 449 :: innerTail0_17
def innerCandidates0_16 : List Nat := [557, 2069, 2441, 5171, 5801, 6353, 6599]
set_option maxRecDepth 4096 in
theorem innerCandidates0_16_checked : candidates edge 4 3 449 innerTail0_17 best.bound = innerCandidates0_16 := by
  unfold candidates
  have row : edge 449 = PrimePairCache.smallRow449.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_16 : search edge 3 [3, 449] innerCandidates0_16 452 best = best := by decide

theorem innerScan0_16 : scan edge (search edge 3) 4 [3] 3 innerTail0_16 best = best := by
  rw [innerTail0_16, scan, if_neg (by decide), innerCandidates0_16_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_17
    (if 3 ≤ innerCandidates0_16.length then search edge 3 [3, 449] innerCandidates0_16 452 best else best) = best
  rw [if_pos (by decide), innerChild0_16]
  exact innerScan0_17

def innerTail0_15 : List Nat := 373 :: innerTail0_16
def innerCandidates0_15 : List Nat := [823, 1237, 1531, 5449, 6793, 7057]
set_option maxRecDepth 4096 in
theorem innerCandidates0_15_checked : candidates edge 4 3 373 innerTail0_16 best.bound = innerCandidates0_15 := by
  unfold candidates
  have row : edge 373 = PrimePairCache.smallRow373.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_15 : search edge 3 [3, 373] innerCandidates0_15 376 best = best := by decide

theorem innerScan0_15 : scan edge (search edge 3) 4 [3] 3 innerTail0_15 best = best := by
  rw [innerTail0_15, scan, if_neg (by decide), innerCandidates0_15_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_16
    (if 3 ≤ innerCandidates0_15.length then search edge 3 [3, 373] innerCandidates0_15 376 best else best) = best
  rw [if_pos (by decide), innerChild0_15]
  exact innerScan0_16

def innerTail0_14 : List Nat := 359 :: innerTail0_15
def innerCandidates0_14 : List Nat := [701, 1667]
set_option maxRecDepth 4096 in
theorem innerCandidates0_14_checked : candidates edge 4 3 359 innerTail0_15 best.bound = innerCandidates0_14 := by
  unfold candidates
  have row : edge 359 = PrimePairCache.smallRow359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_14 : scan edge (search edge 3) 4 [3] 3 innerTail0_14 best = best := by
  rw [innerTail0_14, scan, if_neg (by decide), innerCandidates0_14_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_15
    (if 3 ≤ innerCandidates0_14.length then search edge 3 [3, 359] innerCandidates0_14 362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_15

def innerTail0_13 : List Nat := 331 :: innerTail0_14
def innerCandidates0_13 : List Nat := [739, 1531, 3637, 3769, 6451, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates0_13_checked : candidates edge 4 3 331 innerTail0_14 best.bound = innerCandidates0_13 := by
  unfold candidates
  have row : edge 331 = PrimePairCache.smallRow331.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_13 : search edge 3 [3, 331] innerCandidates0_13 334 best = best := by decide

theorem innerScan0_13 : scan edge (search edge 3) 4 [3] 3 innerTail0_13 best = best := by
  rw [innerTail0_13, scan, if_neg (by decide), innerCandidates0_13_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_14
    (if 3 ≤ innerCandidates0_13.length then search edge 3 [3, 331] innerCandidates0_13 334 best else best) = best
  rw [if_pos (by decide), innerChild0_13]
  exact innerScan0_14

def innerTail0_12 : List Nat := 271 :: innerTail0_13
def innerCandidates0_12 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates0_12_checked : candidates edge 4 3 271 innerTail0_13 best.bound = innerCandidates0_12 := by
  unfold candidates
  have row : edge 271 = PrimePairCache.smallRow271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_12 : scan edge (search edge 3) 4 [3] 3 innerTail0_12 best = best := by
  rw [innerTail0_12, scan, if_neg (by decide), innerCandidates0_12_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_13
    (if 3 ≤ innerCandidates0_12.length then search edge 3 [3, 271] innerCandidates0_12 274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_13

def innerTail0_11 : List Nat := 229 :: innerTail0_12
def innerCandidates0_11 : List Nat := [373, 499, 613, 3301, 5869, 7591]
set_option maxRecDepth 4096 in
theorem innerCandidates0_11_checked : candidates edge 4 3 229 innerTail0_12 best.bound = innerCandidates0_11 := by
  unfold candidates
  have row : edge 229 = PrimePairCache.smallRow229.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_11 : search edge 3 [3, 229] innerCandidates0_11 232 best = best := by decide

theorem innerScan0_11 : scan edge (search edge 3) 4 [3] 3 innerTail0_11 best = best := by
  rw [innerTail0_11, scan, if_neg (by decide), innerCandidates0_11_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_12
    (if 3 ≤ innerCandidates0_11.length then search edge 3 [3, 229] innerCandidates0_11 232 best else best) = best
  rw [if_pos (by decide), innerChild0_11]
  exact innerScan0_12

def innerTail0_10 : List Nat := 191 :: innerTail0_11
def innerCandidates0_10 : List Nat := [2069, 2213, 5099, 6599, 8237]
set_option maxRecDepth 4096 in
theorem innerCandidates0_10_checked : candidates edge 4 3 191 innerTail0_11 best.bound = innerCandidates0_10 := by
  unfold candidates
  have row : edge 191 = PrimePairCache.smallRow191.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_10 : search edge 3 [3, 191] innerCandidates0_10 194 best = best := by decide

theorem innerScan0_10 : scan edge (search edge 3) 4 [3] 3 innerTail0_10 best = best := by
  rw [innerTail0_10, scan, if_neg (by decide), innerCandidates0_10_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_11
    (if 3 ≤ innerCandidates0_10.length then search edge 3 [3, 191] innerCandidates0_10 194 best else best) = best
  rw [if_pos (by decide), innerChild0_10]
  exact innerScan0_11

def innerTail0_9 : List Nat := 137 :: innerTail0_10
def innerCandidates0_9 : List Nat := [191, 359, 947, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates0_9_checked : candidates edge 4 3 137 innerTail0_10 best.bound = innerCandidates0_9 := by
  unfold candidates
  have row : edge 137 = PrimePairCache.smallRow137.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_9 : search edge 3 [3, 137] innerCandidates0_9 140 best = best := by decide

theorem innerScan0_9 : scan edge (search edge 3) 4 [3] 3 innerTail0_9 best = best := by
  rw [innerTail0_9, scan, if_neg (by decide), innerCandidates0_9_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_10
    (if 3 ≤ innerCandidates0_9.length then search edge 3 [3, 137] innerCandidates0_9 140 best else best) = best
  rw [if_pos (by decide), innerChild0_9]
  exact innerScan0_10

def innerTail0_8 : List Nat := 109 :: innerTail0_9
def innerCandidates0_8 : List Nat := [673]
set_option maxRecDepth 4096 in
theorem innerCandidates0_8_checked : candidates edge 4 3 109 innerTail0_9 best.bound = innerCandidates0_8 := by
  unfold candidates
  have row : edge 109 = PrimePairCache.smallRow109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan0_8 : scan edge (search edge 3) 4 [3] 3 innerTail0_8 best = best := by
  rw [innerTail0_8, scan, if_neg (by decide), innerCandidates0_8_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_9
    (if 3 ≤ innerCandidates0_8.length then search edge 3 [3, 109] innerCandidates0_8 112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan0_9

def innerTail0_7 : List Nat := 73 :: innerTail0_8
def innerCandidates0_7 : List Nat := [607, 823, 6277, 6793, 7159, 7591]
set_option maxRecDepth 4096 in
theorem innerCandidates0_7_checked : candidates edge 4 3 73 innerTail0_8 best.bound = innerCandidates0_7 := by
  unfold candidates
  have row : edge 73 = PrimePairCache.smallRow73.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_7 : search edge 3 [3, 73] innerCandidates0_7 76 best = best := by decide

theorem innerScan0_7 : scan edge (search edge 3) 4 [3] 3 innerTail0_7 best = best := by
  rw [innerTail0_7, scan, if_neg (by decide), innerCandidates0_7_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_8
    (if 3 ≤ innerCandidates0_7.length then search edge 3 [3, 73] innerCandidates0_7 76 best else best) = best
  rw [if_pos (by decide), innerChild0_7]
  exact innerScan0_8

def innerTail0_6 : List Nat := 67 :: innerTail0_7
def innerCandidates0_6 : List Nat := [2377, 3637, 5419, 5923]
set_option maxRecDepth 4096 in
theorem innerCandidates0_6_checked : candidates edge 4 3 67 innerTail0_7 best.bound = innerCandidates0_6 := by
  unfold candidates
  have row : edge 67 = PrimePairCache.smallRow67.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_6 : search edge 3 [3, 67] innerCandidates0_6 70 best = best := by decide

theorem innerScan0_6 : scan edge (search edge 3) 4 [3] 3 innerTail0_6 best = best := by
  rw [innerTail0_6, scan, if_neg (by decide), innerCandidates0_6_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_7
    (if 3 ≤ innerCandidates0_6.length then search edge 3 [3, 67] innerCandidates0_6 70 best else best) = best
  rw [if_pos (by decide), innerChild0_6]
  exact innerScan0_7

def innerTail0_5 : List Nat := 59 :: innerTail0_6
def innerCandidates0_5 : List Nat := [929, 1193, 5381, 6653, 6899, 7253]
set_option maxRecDepth 4096 in
theorem innerCandidates0_5_checked : candidates edge 4 3 59 innerTail0_6 best.bound = innerCandidates0_5 := by
  unfold candidates
  have row : edge 59 = PrimePairCache.smallRow59.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_5 : search edge 3 [3, 59] innerCandidates0_5 62 best = best := by decide

theorem innerScan0_5 : scan edge (search edge 3) 4 [3] 3 innerTail0_5 best = best := by
  rw [innerTail0_5, scan, if_neg (by decide), innerCandidates0_5_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_6
    (if 3 ≤ innerCandidates0_5.length then search edge 3 [3, 59] innerCandidates0_5 62 best else best) = best
  rw [if_pos (by decide), innerChild0_5]
  exact innerScan0_6

def innerTail0_4 : List Nat := 37 :: innerTail0_5
def innerCandidates0_4 : List Nat := [67, 607, 1237, 1699, 2377, 2707, 3049, 4159, 4729, 5281, 5449, 5923, 6469, 6529, 7369, 7963]
set_option maxRecDepth 4096 in
theorem innerCandidates0_4_checked : candidates edge 4 3 37 innerTail0_5 best.bound = innerCandidates0_4 := by
  unfold candidates
  have row : edge 37 = PrimePairCache.smallRow37.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_4 : search edge 3 [3, 37] innerCandidates0_4 40 best = best := by decide

theorem innerScan0_4 : scan edge (search edge 3) 4 [3] 3 innerTail0_4 best = best := by
  rw [innerTail0_4, scan, if_neg (by decide), innerCandidates0_4_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_5
    (if 3 ≤ innerCandidates0_4.length then search edge 3 [3, 37] innerCandidates0_4 40 best else best) = best
  rw [if_pos (by decide), innerChild0_4]
  exact innerScan0_5

def innerTail0_3 : List Nat := 31 :: innerTail0_4
def innerCandidates0_3 : List Nat := [1237, 3301, 3637, 4159, 5281, 6037, 6073, 6571, 7591, 8377]
set_option maxRecDepth 4096 in
theorem innerCandidates0_3_checked : candidates edge 4 3 31 innerTail0_4 best.bound = innerCandidates0_3 := by
  unfold candidates
  have row : edge 31 = PrimePairCache.smallRow31.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_3 : search edge 3 [3, 31] innerCandidates0_3 34 best = best := by decide

theorem innerScan0_3 : scan edge (search edge 3) 4 [3] 3 innerTail0_3 best = best := by
  rw [innerTail0_3, scan, if_neg (by decide), innerCandidates0_3_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_4
    (if 3 ≤ innerCandidates0_3.length then search edge 3 [3, 31] innerCandidates0_3 34 best else best) = best
  rw [if_pos (by decide), innerChild0_3]
  exact innerScan0_4

def innerTail0_2 : List Nat := 17 :: innerTail0_3
def innerCandidates0_2 : List Nat := [449, 2069, 2213, 2297, 2411, 4019, 4583, 6353, 6599, 6899, 7691]
set_option maxRecDepth 4096 in
theorem innerCandidates0_2_checked : candidates edge 4 3 17 innerTail0_3 best.bound = innerCandidates0_2 := by
  unfold candidates
  have row : edge 17 = PrimePairCache.smallRow17.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_2 : search edge 3 [3, 17] innerCandidates0_2 20 best = best := by decide

theorem innerScan0_2 : scan edge (search edge 3) 4 [3] 3 innerTail0_2 best = best := by
  rw [innerTail0_2, scan, if_neg (by decide), innerCandidates0_2_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_3
    (if 3 ≤ innerCandidates0_2.length then search edge 3 [3, 17] innerCandidates0_2 20 best else best) = best
  rw [if_pos (by decide), innerChild0_2]
  exact innerScan0_3

def innerTail0_1 : List Nat := 11 :: innerTail0_2
def innerCandidates0_1 : List Nat := [701, 2069, 2213, 2297, 2843, 5099, 8219]
set_option maxRecDepth 4096 in
theorem innerCandidates0_1_checked : candidates edge 4 3 11 innerTail0_2 best.bound = innerCandidates0_1 := by
  unfold candidates
  have row : edge 11 = PrimePairCache.smallRow11.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_1 : search edge 3 [3, 11] innerCandidates0_1 14 best = best := by decide

theorem innerScan0_1 : scan edge (search edge 3) 4 [3] 3 innerTail0_1 best = best := by
  rw [innerTail0_1, scan, if_neg (by decide), innerCandidates0_1_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_2
    (if 3 ≤ innerCandidates0_1.length then search edge 3 [3, 11] innerCandidates0_1 14 best else best) = best
  rw [if_pos (by decide), innerChild0_1]
  exact innerScan0_2

def innerTail0_0 : List Nat := 7 :: innerTail0_1
def innerCandidates0_0 : List Nat := [109, 229, 541, 673, 823, 1237, 2503, 2707, 4159, 4729, 5521]
set_option maxRecDepth 4096 in
theorem innerCandidates0_0_checked : candidates edge 4 3 7 innerTail0_1 best.bound = innerCandidates0_0 := by
  unfold candidates
  have row : edge 7 = PrimePairCache.smallRow7.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild0_0 : search edge 3 [3, 7] innerCandidates0_0 10 best = best := by decide

theorem innerScan0_0 : scan edge (search edge 3) 4 [3] 3 innerTail0_0 best = best := by
  rw [innerTail0_0, scan, if_neg (by decide), innerCandidates0_0_checked]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_1
    (if 3 ≤ innerCandidates0_0.length then search edge 3 [3, 7] innerCandidates0_0 10 best else best) = best
  rw [if_pos (by decide), innerChild0_0]
  exact innerScan0_1

theorem rootBranch0 : search edge 4 [3] rootCandidates0 3 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3] 3 innerTail0_0 best = best
  exact innerScan0_0

end ProjectEulerSolutions.PrimePairUpper.Search
