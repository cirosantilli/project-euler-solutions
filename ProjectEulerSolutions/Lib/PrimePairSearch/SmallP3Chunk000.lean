import ProjectEulerSolutions.Lib.PrimePairRows.P3
import ProjectEulerSolutions.Lib.CliqueSearch

/-! Generated scan steps for the first small-domain branch. -/
namespace ProjectEulerSolutions.PrimePairSearch
open PrimePairCertificate CliqueSearch

def smallBest : Best := ⟨10 ^ 30, []⟩
def smallTail125 : List Nat := []
theorem smallScan125 : scan relation (search relation 3) 4 [3] 3 smallTail125 smallBest = smallBest := rfl

def smallTail124 : List Nat := 8377 :: smallTail125
def smallCandidates124 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates124_checked : candidates relation 4 3 8377 smallTail125 smallBest.bound = smallCandidates124 := by decide

theorem smallScan124 : scan relation (search relation 3) 4 [3] 3 smallTail124 smallBest = smallBest := by
  rw [smallTail124, scan, if_neg (by decide), smallCandidates124_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail125
    (if 3 ≤ smallCandidates124.length then search relation 3 [3, 8377] smallCandidates124 8380 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan125

def smallTail123 : List Nat := 8237 :: smallTail124
def smallCandidates123 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates123_checked : candidates relation 4 3 8237 smallTail124 smallBest.bound = smallCandidates123 := by decide

theorem smallScan123 : scan relation (search relation 3) 4 [3] 3 smallTail123 smallBest = smallBest := by
  rw [smallTail123, scan, if_neg (by decide), smallCandidates123_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail124
    (if 3 ≤ smallCandidates123.length then search relation 3 [3, 8237] smallCandidates123 8240 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan124

def smallTail122 : List Nat := 8219 :: smallTail123
def smallCandidates122 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates122_checked : candidates relation 4 3 8219 smallTail123 smallBest.bound = smallCandidates122 := by decide

theorem smallScan122 : scan relation (search relation 3) 4 [3] 3 smallTail122 smallBest = smallBest := by
  rw [smallTail122, scan, if_neg (by decide), smallCandidates122_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail123
    (if 3 ≤ smallCandidates122.length then search relation 3 [3, 8219] smallCandidates122 8222 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan123

def smallTail121 : List Nat := 7963 :: smallTail122
def smallCandidates121 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates121_checked : candidates relation 4 3 7963 smallTail122 smallBest.bound = smallCandidates121 := by decide

theorem smallScan121 : scan relation (search relation 3) 4 [3] 3 smallTail121 smallBest = smallBest := by
  rw [smallTail121, scan, if_neg (by decide), smallCandidates121_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail122
    (if 3 ≤ smallCandidates121.length then search relation 3 [3, 7963] smallCandidates121 7966 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan122

def smallTail120 : List Nat := 7691 :: smallTail121
def smallCandidates120 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates120_checked : candidates relation 4 3 7691 smallTail121 smallBest.bound = smallCandidates120 := by decide

theorem smallScan120 : scan relation (search relation 3) 4 [3] 3 smallTail120 smallBest = smallBest := by
  rw [smallTail120, scan, if_neg (by decide), smallCandidates120_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail121
    (if 3 ≤ smallCandidates120.length then search relation 3 [3, 7691] smallCandidates120 7694 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan121

def smallTail119 : List Nat := 7649 :: smallTail120
def smallCandidates119 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates119_checked : candidates relation 4 3 7649 smallTail120 smallBest.bound = smallCandidates119 := by decide

theorem smallScan119 : scan relation (search relation 3) 4 [3] 3 smallTail119 smallBest = smallBest := by
  rw [smallTail119, scan, if_neg (by decide), smallCandidates119_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail120
    (if 3 ≤ smallCandidates119.length then search relation 3 [3, 7649] smallCandidates119 7652 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan120

def smallTail118 : List Nat := 7591 :: smallTail119
def smallCandidates118 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates118_checked : candidates relation 4 3 7591 smallTail119 smallBest.bound = smallCandidates118 := by decide

theorem smallScan118 : scan relation (search relation 3) 4 [3] 3 smallTail118 smallBest = smallBest := by
  rw [smallTail118, scan, if_neg (by decide), smallCandidates118_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail119
    (if 3 ≤ smallCandidates118.length then search relation 3 [3, 7591] smallCandidates118 7594 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan119

def smallTail117 : List Nat := 7369 :: smallTail118
def smallCandidates117 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates117_checked : candidates relation 4 3 7369 smallTail118 smallBest.bound = smallCandidates117 := by decide

theorem smallScan117 : scan relation (search relation 3) 4 [3] 3 smallTail117 smallBest = smallBest := by
  rw [smallTail117, scan, if_neg (by decide), smallCandidates117_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail118
    (if 3 ≤ smallCandidates117.length then search relation 3 [3, 7369] smallCandidates117 7372 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan118

def smallTail116 : List Nat := 7253 :: smallTail117
def smallCandidates116 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates116_checked : candidates relation 4 3 7253 smallTail117 smallBest.bound = smallCandidates116 := by decide

theorem smallScan116 : scan relation (search relation 3) 4 [3] 3 smallTail116 smallBest = smallBest := by
  rw [smallTail116, scan, if_neg (by decide), smallCandidates116_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail117
    (if 3 ≤ smallCandidates116.length then search relation 3 [3, 7253] smallCandidates116 7256 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan117

def smallTail115 : List Nat := 7159 :: smallTail116
def smallCandidates115 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates115_checked : candidates relation 4 3 7159 smallTail116 smallBest.bound = smallCandidates115 := by decide

theorem smallScan115 : scan relation (search relation 3) 4 [3] 3 smallTail115 smallBest = smallBest := by
  rw [smallTail115, scan, if_neg (by decide), smallCandidates115_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail116
    (if 3 ≤ smallCandidates115.length then search relation 3 [3, 7159] smallCandidates115 7162 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan116

def smallTail114 : List Nat := 7057 :: smallTail115
def smallCandidates114 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates114_checked : candidates relation 4 3 7057 smallTail115 smallBest.bound = smallCandidates114 := by decide

theorem smallScan114 : scan relation (search relation 3) 4 [3] 3 smallTail114 smallBest = smallBest := by
  rw [smallTail114, scan, if_neg (by decide), smallCandidates114_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail115
    (if 3 ≤ smallCandidates114.length then search relation 3 [3, 7057] smallCandidates114 7060 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan115

def smallTail113 : List Nat := 7039 :: smallTail114
def smallCandidates113 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates113_checked : candidates relation 4 3 7039 smallTail114 smallBest.bound = smallCandidates113 := by decide

theorem smallScan113 : scan relation (search relation 3) 4 [3] 3 smallTail113 smallBest = smallBest := by
  rw [smallTail113, scan, if_neg (by decide), smallCandidates113_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail114
    (if 3 ≤ smallCandidates113.length then search relation 3 [3, 7039] smallCandidates113 7042 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan114

def smallTail112 : List Nat := 6947 :: smallTail113
def smallCandidates112 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates112_checked : candidates relation 4 3 6947 smallTail113 smallBest.bound = smallCandidates112 := by decide

theorem smallScan112 : scan relation (search relation 3) 4 [3] 3 smallTail112 smallBest = smallBest := by
  rw [smallTail112, scan, if_neg (by decide), smallCandidates112_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail113
    (if 3 ≤ smallCandidates112.length then search relation 3 [3, 6947] smallCandidates112 6950 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan113

def smallTail111 : List Nat := 6899 :: smallTail112
def smallCandidates111 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates111_checked : candidates relation 4 3 6899 smallTail112 smallBest.bound = smallCandidates111 := by decide

theorem smallScan111 : scan relation (search relation 3) 4 [3] 3 smallTail111 smallBest = smallBest := by
  rw [smallTail111, scan, if_neg (by decide), smallCandidates111_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail112
    (if 3 ≤ smallCandidates111.length then search relation 3 [3, 6899] smallCandidates111 6902 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan112

def smallTail110 : List Nat := 6871 :: smallTail111
def smallCandidates110 : List Nat := [7039]
set_option maxRecDepth 2048 in
theorem smallCandidates110_checked : candidates relation 4 3 6871 smallTail111 smallBest.bound = smallCandidates110 := by decide

theorem smallScan110 : scan relation (search relation 3) 4 [3] 3 smallTail110 smallBest = smallBest := by
  rw [smallTail110, scan, if_neg (by decide), smallCandidates110_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail111
    (if 3 ≤ smallCandidates110.length then search relation 3 [3, 6871] smallCandidates110 6874 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan111

def smallTail109 : List Nat := 6793 :: smallTail110
def smallCandidates109 : List Nat := [7039, 7159]
set_option maxRecDepth 2048 in
theorem smallCandidates109_checked : candidates relation 4 3 6793 smallTail110 smallBest.bound = smallCandidates109 := by decide

theorem smallScan109 : scan relation (search relation 3) 4 [3] 3 smallTail109 smallBest = smallBest := by
  rw [smallTail109, scan, if_neg (by decide), smallCandidates109_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail110
    (if 3 ≤ smallCandidates109.length then search relation 3 [3, 6793] smallCandidates109 6796 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan110

def smallTail108 : List Nat := 6653 :: smallTail109
def smallCandidates108 : List Nat := [7649]
set_option maxRecDepth 2048 in
theorem smallCandidates108_checked : candidates relation 4 3 6653 smallTail109 smallBest.bound = smallCandidates108 := by decide

theorem smallScan108 : scan relation (search relation 3) 4 [3] 3 smallTail108 smallBest = smallBest := by
  rw [smallTail108, scan, if_neg (by decide), smallCandidates108_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail109
    (if 3 ≤ smallCandidates108.length then search relation 3 [3, 6653] smallCandidates108 6656 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan109

def smallTail107 : List Nat := 6637 :: smallTail108
def smallCandidates107 : List Nat := [7039]
set_option maxRecDepth 2048 in
theorem smallCandidates107_checked : candidates relation 4 3 6637 smallTail108 smallBest.bound = smallCandidates107 := by decide

theorem smallScan107 : scan relation (search relation 3) 4 [3] 3 smallTail107 smallBest = smallBest := by
  rw [smallTail107, scan, if_neg (by decide), smallCandidates107_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail108
    (if 3 ≤ smallCandidates107.length then search relation 3 [3, 6637] smallCandidates107 6640 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan108

def smallTail106 : List Nat := 6599 :: smallTail107
def smallCandidates106 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates106_checked : candidates relation 4 3 6599 smallTail107 smallBest.bound = smallCandidates106 := by decide

theorem smallScan106 : scan relation (search relation 3) 4 [3] 3 smallTail106 smallBest = smallBest := by
  rw [smallTail106, scan, if_neg (by decide), smallCandidates106_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail107
    (if 3 ≤ smallCandidates106.length then search relation 3 [3, 6599] smallCandidates106 6602 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan107

def smallTail105 : List Nat := 6571 :: smallTail106
def smallCandidates105 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates105_checked : candidates relation 4 3 6571 smallTail106 smallBest.bound = smallCandidates105 := by decide

theorem smallScan105 : scan relation (search relation 3) 4 [3] 3 smallTail105 smallBest = smallBest := by
  rw [smallTail105, scan, if_neg (by decide), smallCandidates105_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail106
    (if 3 ≤ smallCandidates105.length then search relation 3 [3, 6571] smallCandidates105 6574 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan106

def smallTail104 : List Nat := 6563 :: smallTail105
def smallCandidates104 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates104_checked : candidates relation 4 3 6563 smallTail105 smallBest.bound = smallCandidates104 := by decide

theorem smallScan104 : scan relation (search relation 3) 4 [3] 3 smallTail104 smallBest = smallBest := by
  rw [smallTail104, scan, if_neg (by decide), smallCandidates104_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail105
    (if 3 ≤ smallCandidates104.length then search relation 3 [3, 6563] smallCandidates104 6566 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan105

def smallTail103 : List Nat := 6529 :: smallTail104
def smallCandidates103 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates103_checked : candidates relation 4 3 6529 smallTail104 smallBest.bound = smallCandidates103 := by decide

theorem smallScan103 : scan relation (search relation 3) 4 [3] 3 smallTail103 smallBest = smallBest := by
  rw [smallTail103, scan, if_neg (by decide), smallCandidates103_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail104
    (if 3 ≤ smallCandidates103.length then search relation 3 [3, 6529] smallCandidates103 6532 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan104

def smallTail102 : List Nat := 6469 :: smallTail103
def smallCandidates102 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates102_checked : candidates relation 4 3 6469 smallTail103 smallBest.bound = smallCandidates102 := by decide

theorem smallScan102 : scan relation (search relation 3) 4 [3] 3 smallTail102 smallBest = smallBest := by
  rw [smallTail102, scan, if_neg (by decide), smallCandidates102_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail103
    (if 3 ≤ smallCandidates102.length then search relation 3 [3, 6469] smallCandidates102 6472 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan103

def smallTail101 : List Nat := 6451 :: smallTail102
def smallCandidates101 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates101_checked : candidates relation 4 3 6451 smallTail102 smallBest.bound = smallCandidates101 := by decide

theorem smallScan101 : scan relation (search relation 3) 4 [3] 3 smallTail101 smallBest = smallBest := by
  rw [smallTail101, scan, if_neg (by decide), smallCandidates101_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail102
    (if 3 ≤ smallCandidates101.length then search relation 3 [3, 6451] smallCandidates101 6454 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan102

def smallTail100 : List Nat := 6353 :: smallTail101
def smallCandidates100 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates100_checked : candidates relation 4 3 6353 smallTail101 smallBest.bound = smallCandidates100 := by decide

theorem smallScan100 : scan relation (search relation 3) 4 [3] 3 smallTail100 smallBest = smallBest := by
  rw [smallTail100, scan, if_neg (by decide), smallCandidates100_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail101
    (if 3 ≤ smallCandidates100.length then search relation 3 [3, 6353] smallCandidates100 6356 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan101

def smallTail99 : List Nat := 6277 :: smallTail100
def smallCandidates99 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates99_checked : candidates relation 4 3 6277 smallTail100 smallBest.bound = smallCandidates99 := by decide

theorem smallScan99 : scan relation (search relation 3) 4 [3] 3 smallTail99 smallBest = smallBest := by
  rw [smallTail99, scan, if_neg (by decide), smallCandidates99_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail100
    (if 3 ≤ smallCandidates99.length then search relation 3 [3, 6277] smallCandidates99 6280 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan100

def smallTail98 : List Nat := 6263 :: smallTail99
def smallCandidates98 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates98_checked : candidates relation 4 3 6263 smallTail99 smallBest.bound = smallCandidates98 := by decide

theorem smallScan98 : scan relation (search relation 3) 4 [3] 3 smallTail98 smallBest = smallBest := by
  rw [smallTail98, scan, if_neg (by decide), smallCandidates98_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail99
    (if 3 ≤ smallCandidates98.length then search relation 3 [3, 6263] smallCandidates98 6266 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan99

def smallTail97 : List Nat := 6073 :: smallTail98
def smallCandidates97 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates97_checked : candidates relation 4 3 6073 smallTail98 smallBest.bound = smallCandidates97 := by decide

theorem smallScan97 : scan relation (search relation 3) 4 [3] 3 smallTail97 smallBest = smallBest := by
  rw [smallTail97, scan, if_neg (by decide), smallCandidates97_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail98
    (if 3 ≤ smallCandidates97.length then search relation 3 [3, 6073] smallCandidates97 6076 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan98

def smallTail96 : List Nat := 6037 :: smallTail97
def smallCandidates96 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates96_checked : candidates relation 4 3 6037 smallTail97 smallBest.bound = smallCandidates96 := by decide

theorem smallScan96 : scan relation (search relation 3) 4 [3] 3 smallTail96 smallBest = smallBest := by
  rw [smallTail96, scan, if_neg (by decide), smallCandidates96_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail97
    (if 3 ≤ smallCandidates96.length then search relation 3 [3, 6037] smallCandidates96 6040 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan97

def smallTail95 : List Nat := 5923 :: smallTail96
def smallCandidates95 : List Nat := [6871, 7963]
set_option maxRecDepth 2048 in
theorem smallCandidates95_checked : candidates relation 4 3 5923 smallTail96 smallBest.bound = smallCandidates95 := by decide

theorem smallScan95 : scan relation (search relation 3) 4 [3] 3 smallTail95 smallBest = smallBest := by
  rw [smallTail95, scan, if_neg (by decide), smallCandidates95_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail96
    (if 3 ≤ smallCandidates95.length then search relation 3 [3, 5923] smallCandidates95 5926 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan96

def smallTail94 : List Nat := 5869 :: smallTail95
def smallCandidates94 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates94_checked : candidates relation 4 3 5869 smallTail95 smallBest.bound = smallCandidates94 := by decide

theorem smallScan94 : scan relation (search relation 3) 4 [3] 3 smallTail94 smallBest = smallBest := by
  rw [smallTail94, scan, if_neg (by decide), smallCandidates94_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail95
    (if 3 ≤ smallCandidates94.length then search relation 3 [3, 5869] smallCandidates94 5872 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan95

def smallTail93 : List Nat := 5839 :: smallTail94
def smallCandidates93 : List Nat := []
set_option maxRecDepth 2048 in
theorem smallCandidates93_checked : candidates relation 4 3 5839 smallTail94 smallBest.bound = smallCandidates93 := by decide

theorem smallScan93 : scan relation (search relation 3) 4 [3] 3 smallTail93 smallBest = smallBest := by
  rw [smallTail93, scan, if_neg (by decide), smallCandidates93_checked]
  change scan relation (search relation 3) 4 [3] 3 smallTail94
    (if 3 ≤ smallCandidates93.length then search relation 3 [3, 5839] smallCandidates93 5842 smallBest else smallBest) = smallBest
  rw [if_neg (by decide)]
  exact smallScan94

end ProjectEulerSolutions.PrimePairSearch
