import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates659 : List Nat := [5107, 5563, 5821, 6343, 6781, 7537]
set_option maxRecDepth 4096 in
theorem rootCandidates659_checked : candidates edge 5 0 4951 tail660 best.bound = rootCandidates659 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext q; rfl
  rw [row]
  decide

def innerTail659_6 : List Nat := []
theorem innerScan659_6 : scan edge (search edge 3) 4 [4951] 4951 innerTail659_6 best = best := rfl

def innerTail659_5 : List Nat := 7537 :: innerTail659_6
def innerCandidates659_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates659_5_checked : candidates edge 4 4951 7537 innerTail659_6 best.bound = innerCandidates659_5 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan659_5 : scan edge (search edge 3) 4 [4951] 4951 innerTail659_5 best = best := by
  rw [innerTail659_5, scan, if_neg (by decide), innerCandidates659_5_checked]
  change scan edge (search edge 3) 4 [4951] 4951 innerTail659_6
    (if 3 ≤ innerCandidates659_5.length then search edge 3 [4951, 7537] innerCandidates659_5 12488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan659_6

def innerTail659_4 : List Nat := 6781 :: innerTail659_5
def innerCandidates659_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates659_4_checked : candidates edge 4 4951 6781 innerTail659_5 best.bound = innerCandidates659_4 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan659_4 : scan edge (search edge 3) 4 [4951] 4951 innerTail659_4 best = best := by
  rw [innerTail659_4, scan, if_neg (by decide), innerCandidates659_4_checked]
  change scan edge (search edge 3) 4 [4951] 4951 innerTail659_5
    (if 3 ≤ innerCandidates659_4.length then search edge 3 [4951, 6781] innerCandidates659_4 11732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan659_5

def innerTail659_3 : List Nat := 6343 :: innerTail659_4
def innerCandidates659_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates659_3_checked : candidates edge 4 4951 6343 innerTail659_4 best.bound = innerCandidates659_3 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan659_3 : scan edge (search edge 3) 4 [4951] 4951 innerTail659_3 best = best := by
  rw [innerTail659_3, scan, if_neg (by decide), innerCandidates659_3_checked]
  change scan edge (search edge 3) 4 [4951] 4951 innerTail659_4
    (if 3 ≤ innerCandidates659_3.length then search edge 3 [4951, 6343] innerCandidates659_3 11294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan659_4

def innerTail659_2 : List Nat := 5821 :: innerTail659_3
def innerCandidates659_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates659_2_checked : candidates edge 4 4951 5821 innerTail659_3 best.bound = innerCandidates659_2 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan659_2 : scan edge (search edge 3) 4 [4951] 4951 innerTail659_2 best = best := by
  rw [innerTail659_2, scan, if_neg (by decide), innerCandidates659_2_checked]
  change scan edge (search edge 3) 4 [4951] 4951 innerTail659_3
    (if 3 ≤ innerCandidates659_2.length then search edge 3 [4951, 5821] innerCandidates659_2 10772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan659_3

def innerTail659_1 : List Nat := 5563 :: innerTail659_2
def innerCandidates659_1 : List Nat := [6781]
set_option maxRecDepth 4096 in
theorem innerCandidates659_1_checked : candidates edge 4 4951 5563 innerTail659_2 best.bound = innerCandidates659_1 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan659_1 : scan edge (search edge 3) 4 [4951] 4951 innerTail659_1 best = best := by
  rw [innerTail659_1, scan, if_neg (by decide), innerCandidates659_1_checked]
  change scan edge (search edge 3) 4 [4951] 4951 innerTail659_2
    (if 3 ≤ innerCandidates659_1.length then search edge 3 [4951, 5563] innerCandidates659_1 10514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan659_2

def innerTail659_0 : List Nat := 5107 :: innerTail659_1
def innerCandidates659_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates659_0_checked : candidates edge 4 4951 5107 innerTail659_1 best.bound = innerCandidates659_0 := by
  unfold candidates
  have row : edge 5107 = SearchData.row5107.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan659_0 : scan edge (search edge 3) 4 [4951] 4951 innerTail659_0 best = best := by
  rw [innerTail659_0, scan, if_neg (by decide), innerCandidates659_0_checked]
  change scan edge (search edge 3) 4 [4951] 4951 innerTail659_1
    (if 3 ≤ innerCandidates659_0.length then search edge 3 [4951, 5107] innerCandidates659_0 10058 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan659_1

theorem rootBranch659 : search edge 4 [4951] rootCandidates659 4951 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4951] 4951 innerTail659_0 best = best
  exact innerScan659_0

def rootCandidates658 : List Nat := [5021, 5039, 5189, 5261, 5417, 5741, 6449, 6701, 7151, 7517, 7757, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates658_checked : candidates edge 5 0 4943 tail659 best.bound = rootCandidates658 := by
  unfold candidates
  have row : edge 4943 = SearchData.row4943.contains := by funext q; rfl
  rw [row]
  decide

def innerTail658_12 : List Nat := []
theorem innerScan658_12 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_12 best = best := rfl

def innerTail658_11 : List Nat := 7919 :: innerTail658_12
def innerCandidates658_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_11_checked : candidates edge 4 4943 7919 innerTail658_12 best.bound = innerCandidates658_11 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_11 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_11 best = best := by
  rw [innerTail658_11, scan, if_neg (by decide), innerCandidates658_11_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_12
    (if 3 ≤ innerCandidates658_11.length then search edge 3 [4943, 7919] innerCandidates658_11 12862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_12

def innerTail658_10 : List Nat := 7757 :: innerTail658_11
def innerCandidates658_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_10_checked : candidates edge 4 4943 7757 innerTail658_11 best.bound = innerCandidates658_10 := by
  unfold candidates
  have row : edge 7757 = SearchData.row7757.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_10 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_10 best = best := by
  rw [innerTail658_10, scan, if_neg (by decide), innerCandidates658_10_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_11
    (if 3 ≤ innerCandidates658_10.length then search edge 3 [4943, 7757] innerCandidates658_10 12700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_11

def innerTail658_9 : List Nat := 7517 :: innerTail658_10
def innerCandidates658_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_9_checked : candidates edge 4 4943 7517 innerTail658_10 best.bound = innerCandidates658_9 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_9 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_9 best = best := by
  rw [innerTail658_9, scan, if_neg (by decide), innerCandidates658_9_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_10
    (if 3 ≤ innerCandidates658_9.length then search edge 3 [4943, 7517] innerCandidates658_9 12460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_10

def innerTail658_8 : List Nat := 7151 :: innerTail658_9
def innerCandidates658_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_8_checked : candidates edge 4 4943 7151 innerTail658_9 best.bound = innerCandidates658_8 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_8 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_8 best = best := by
  rw [innerTail658_8, scan, if_neg (by decide), innerCandidates658_8_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_9
    (if 3 ≤ innerCandidates658_8.length then search edge 3 [4943, 7151] innerCandidates658_8 12094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_9

def innerTail658_7 : List Nat := 6701 :: innerTail658_8
def innerCandidates658_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_7_checked : candidates edge 4 4943 6701 innerTail658_8 best.bound = innerCandidates658_7 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_7 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_7 best = best := by
  rw [innerTail658_7, scan, if_neg (by decide), innerCandidates658_7_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_8
    (if 3 ≤ innerCandidates658_7.length then search edge 3 [4943, 6701] innerCandidates658_7 11644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_8

def innerTail658_6 : List Nat := 6449 :: innerTail658_7
def innerCandidates658_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_6_checked : candidates edge 4 4943 6449 innerTail658_7 best.bound = innerCandidates658_6 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_6 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_6 best = best := by
  rw [innerTail658_6, scan, if_neg (by decide), innerCandidates658_6_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_7
    (if 3 ≤ innerCandidates658_6.length then search edge 3 [4943, 6449] innerCandidates658_6 11392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_7

def innerTail658_5 : List Nat := 5741 :: innerTail658_6
def innerCandidates658_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_5_checked : candidates edge 4 4943 5741 innerTail658_6 best.bound = innerCandidates658_5 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_5 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_5 best = best := by
  rw [innerTail658_5, scan, if_neg (by decide), innerCandidates658_5_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_6
    (if 3 ≤ innerCandidates658_5.length then search edge 3 [4943, 5741] innerCandidates658_5 10684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_6

def innerTail658_4 : List Nat := 5417 :: innerTail658_5
def innerCandidates658_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_4_checked : candidates edge 4 4943 5417 innerTail658_5 best.bound = innerCandidates658_4 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_4 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_4 best = best := by
  rw [innerTail658_4, scan, if_neg (by decide), innerCandidates658_4_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_5
    (if 3 ≤ innerCandidates658_4.length then search edge 3 [4943, 5417] innerCandidates658_4 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_5

def innerTail658_3 : List Nat := 5261 :: innerTail658_4
def innerCandidates658_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_3_checked : candidates edge 4 4943 5261 innerTail658_4 best.bound = innerCandidates658_3 := by
  unfold candidates
  have row : edge 5261 = SearchData.row5261.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_3 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_3 best = best := by
  rw [innerTail658_3, scan, if_neg (by decide), innerCandidates658_3_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_4
    (if 3 ≤ innerCandidates658_3.length then search edge 3 [4943, 5261] innerCandidates658_3 10204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_4

def innerTail658_2 : List Nat := 5189 :: innerTail658_3
def innerCandidates658_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_2_checked : candidates edge 4 4943 5189 innerTail658_3 best.bound = innerCandidates658_2 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_2 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_2 best = best := by
  rw [innerTail658_2, scan, if_neg (by decide), innerCandidates658_2_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_3
    (if 3 ≤ innerCandidates658_2.length then search edge 3 [4943, 5189] innerCandidates658_2 10132 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_3

def innerTail658_1 : List Nat := 5039 :: innerTail658_2
def innerCandidates658_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates658_1_checked : candidates edge 4 4943 5039 innerTail658_2 best.bound = innerCandidates658_1 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_1 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_1 best = best := by
  rw [innerTail658_1, scan, if_neg (by decide), innerCandidates658_1_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_2
    (if 3 ≤ innerCandidates658_1.length then search edge 3 [4943, 5039] innerCandidates658_1 9982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_2

def innerTail658_0 : List Nat := 5021 :: innerTail658_1
def innerCandidates658_0 : List Nat := [5741]
set_option maxRecDepth 4096 in
theorem innerCandidates658_0_checked : candidates edge 4 4943 5021 innerTail658_1 best.bound = innerCandidates658_0 := by
  unfold candidates
  have row : edge 5021 = SearchData.row5021.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan658_0 : scan edge (search edge 3) 4 [4943] 4943 innerTail658_0 best = best := by
  rw [innerTail658_0, scan, if_neg (by decide), innerCandidates658_0_checked]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_1
    (if 3 ≤ innerCandidates658_0.length then search edge 3 [4943, 5021] innerCandidates658_0 9964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan658_1

theorem rootBranch658 : search edge 4 [4943] rootCandidates658 4943 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4943] 4943 innerTail658_0 best = best
  exact innerScan658_0

def rootCandidates657 : List Nat := [5399, 6053, 6491, 7001, 7043, 7103, 7247, 7583, 7649, 7841, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates657_checked : candidates edge 5 0 4937 tail658 best.bound = rootCandidates657 := by
  unfold candidates
  have row : edge 4937 = SearchData.row4937.contains := by funext q; rfl
  rw [row]
  decide

def innerTail657_11 : List Nat := []
theorem innerScan657_11 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_11 best = best := rfl

def innerTail657_10 : List Nat := 7919 :: innerTail657_11
def innerCandidates657_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_10_checked : candidates edge 4 4937 7919 innerTail657_11 best.bound = innerCandidates657_10 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_10 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_10 best = best := by
  rw [innerTail657_10, scan, if_neg (by decide), innerCandidates657_10_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_11
    (if 3 ≤ innerCandidates657_10.length then search edge 3 [4937, 7919] innerCandidates657_10 12856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_11

def innerTail657_9 : List Nat := 7841 :: innerTail657_10
def innerCandidates657_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_9_checked : candidates edge 4 4937 7841 innerTail657_10 best.bound = innerCandidates657_9 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_9 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_9 best = best := by
  rw [innerTail657_9, scan, if_neg (by decide), innerCandidates657_9_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_10
    (if 3 ≤ innerCandidates657_9.length then search edge 3 [4937, 7841] innerCandidates657_9 12778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_10

def innerTail657_8 : List Nat := 7649 :: innerTail657_9
def innerCandidates657_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_8_checked : candidates edge 4 4937 7649 innerTail657_9 best.bound = innerCandidates657_8 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_8 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_8 best = best := by
  rw [innerTail657_8, scan, if_neg (by decide), innerCandidates657_8_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_9
    (if 3 ≤ innerCandidates657_8.length then search edge 3 [4937, 7649] innerCandidates657_8 12586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_9

def innerTail657_7 : List Nat := 7583 :: innerTail657_8
def innerCandidates657_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_7_checked : candidates edge 4 4937 7583 innerTail657_8 best.bound = innerCandidates657_7 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_7 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_7 best = best := by
  rw [innerTail657_7, scan, if_neg (by decide), innerCandidates657_7_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_8
    (if 3 ≤ innerCandidates657_7.length then search edge 3 [4937, 7583] innerCandidates657_7 12520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_8

def innerTail657_6 : List Nat := 7247 :: innerTail657_7
def innerCandidates657_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_6_checked : candidates edge 4 4937 7247 innerTail657_7 best.bound = innerCandidates657_6 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_6 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_6 best = best := by
  rw [innerTail657_6, scan, if_neg (by decide), innerCandidates657_6_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_7
    (if 3 ≤ innerCandidates657_6.length then search edge 3 [4937, 7247] innerCandidates657_6 12184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_7

def innerTail657_5 : List Nat := 7103 :: innerTail657_6
def innerCandidates657_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_5_checked : candidates edge 4 4937 7103 innerTail657_6 best.bound = innerCandidates657_5 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_5 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_5 best = best := by
  rw [innerTail657_5, scan, if_neg (by decide), innerCandidates657_5_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_6
    (if 3 ≤ innerCandidates657_5.length then search edge 3 [4937, 7103] innerCandidates657_5 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_6

def innerTail657_4 : List Nat := 7043 :: innerTail657_5
def innerCandidates657_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_4_checked : candidates edge 4 4937 7043 innerTail657_5 best.bound = innerCandidates657_4 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_4 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_4 best = best := by
  rw [innerTail657_4, scan, if_neg (by decide), innerCandidates657_4_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_5
    (if 3 ≤ innerCandidates657_4.length then search edge 3 [4937, 7043] innerCandidates657_4 11980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_5

def innerTail657_3 : List Nat := 7001 :: innerTail657_4
def innerCandidates657_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_3_checked : candidates edge 4 4937 7001 innerTail657_4 best.bound = innerCandidates657_3 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_3 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_3 best = best := by
  rw [innerTail657_3, scan, if_neg (by decide), innerCandidates657_3_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_4
    (if 3 ≤ innerCandidates657_3.length then search edge 3 [4937, 7001] innerCandidates657_3 11938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_4

def innerTail657_2 : List Nat := 6491 :: innerTail657_3
def innerCandidates657_2 : List Nat := [7583]
set_option maxRecDepth 4096 in
theorem innerCandidates657_2_checked : candidates edge 4 4937 6491 innerTail657_3 best.bound = innerCandidates657_2 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_2 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_2 best = best := by
  rw [innerTail657_2, scan, if_neg (by decide), innerCandidates657_2_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_3
    (if 3 ≤ innerCandidates657_2.length then search edge 3 [4937, 6491] innerCandidates657_2 11428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_3

def innerTail657_1 : List Nat := 6053 :: innerTail657_2
def innerCandidates657_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates657_1_checked : candidates edge 4 4937 6053 innerTail657_2 best.bound = innerCandidates657_1 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_1 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_1 best = best := by
  rw [innerTail657_1, scan, if_neg (by decide), innerCandidates657_1_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_2
    (if 3 ≤ innerCandidates657_1.length then search edge 3 [4937, 6053] innerCandidates657_1 10990 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_2

def innerTail657_0 : List Nat := 5399 :: innerTail657_1
def innerCandidates657_0 : List Nat := [7247, 7649]
set_option maxRecDepth 4096 in
theorem innerCandidates657_0_checked : candidates edge 4 4937 5399 innerTail657_1 best.bound = innerCandidates657_0 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan657_0 : scan edge (search edge 3) 4 [4937] 4937 innerTail657_0 best = best := by
  rw [innerTail657_0, scan, if_neg (by decide), innerCandidates657_0_checked]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_1
    (if 3 ≤ innerCandidates657_0.length then search edge 3 [4937, 5399] innerCandidates657_0 10336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan657_1

theorem rootBranch657 : search edge 4 [4937] rootCandidates657 4937 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4937] 4937 innerTail657_0 best = best
  exact innerScan657_0

def rootCandidates656 : List Nat := [5701, 6271, 6361]
set_option maxRecDepth 4096 in
theorem rootCandidates656_checked : candidates edge 5 0 4933 tail657 best.bound = rootCandidates656 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates655 : List Nat := [5009, 5309, 5843, 6551, 7109, 7349, 7703, 7817]
set_option maxRecDepth 4096 in
theorem rootCandidates655_checked : candidates edge 5 0 4931 tail656 best.bound = rootCandidates655 := by
  unfold candidates
  have row : edge 4931 = SearchData.row4931.contains := by funext q; rfl
  rw [row]
  decide

def innerTail655_8 : List Nat := []
theorem innerScan655_8 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_8 best = best := rfl

def innerTail655_7 : List Nat := 7817 :: innerTail655_8
def innerCandidates655_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates655_7_checked : candidates edge 4 4931 7817 innerTail655_8 best.bound = innerCandidates655_7 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_7 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_7 best = best := by
  rw [innerTail655_7, scan, if_neg (by decide), innerCandidates655_7_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_8
    (if 3 ≤ innerCandidates655_7.length then search edge 3 [4931, 7817] innerCandidates655_7 12748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_8

def innerTail655_6 : List Nat := 7703 :: innerTail655_7
def innerCandidates655_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates655_6_checked : candidates edge 4 4931 7703 innerTail655_7 best.bound = innerCandidates655_6 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_6 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_6 best = best := by
  rw [innerTail655_6, scan, if_neg (by decide), innerCandidates655_6_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_7
    (if 3 ≤ innerCandidates655_6.length then search edge 3 [4931, 7703] innerCandidates655_6 12634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_7

def innerTail655_5 : List Nat := 7349 :: innerTail655_6
def innerCandidates655_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates655_5_checked : candidates edge 4 4931 7349 innerTail655_6 best.bound = innerCandidates655_5 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_5 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_5 best = best := by
  rw [innerTail655_5, scan, if_neg (by decide), innerCandidates655_5_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_6
    (if 3 ≤ innerCandidates655_5.length then search edge 3 [4931, 7349] innerCandidates655_5 12280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_6

def innerTail655_4 : List Nat := 7109 :: innerTail655_5
def innerCandidates655_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates655_4_checked : candidates edge 4 4931 7109 innerTail655_5 best.bound = innerCandidates655_4 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_4 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_4 best = best := by
  rw [innerTail655_4, scan, if_neg (by decide), innerCandidates655_4_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_5
    (if 3 ≤ innerCandidates655_4.length then search edge 3 [4931, 7109] innerCandidates655_4 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_5

def innerTail655_3 : List Nat := 6551 :: innerTail655_4
def innerCandidates655_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates655_3_checked : candidates edge 4 4931 6551 innerTail655_4 best.bound = innerCandidates655_3 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_3 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_3 best = best := by
  rw [innerTail655_3, scan, if_neg (by decide), innerCandidates655_3_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_4
    (if 3 ≤ innerCandidates655_3.length then search edge 3 [4931, 6551] innerCandidates655_3 11482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_4

def innerTail655_2 : List Nat := 5843 :: innerTail655_3
def innerCandidates655_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates655_2_checked : candidates edge 4 4931 5843 innerTail655_3 best.bound = innerCandidates655_2 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_2 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_2 best = best := by
  rw [innerTail655_2, scan, if_neg (by decide), innerCandidates655_2_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_3
    (if 3 ≤ innerCandidates655_2.length then search edge 3 [4931, 5843] innerCandidates655_2 10774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_3

def innerTail655_1 : List Nat := 5309 :: innerTail655_2
def innerCandidates655_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates655_1_checked : candidates edge 4 4931 5309 innerTail655_2 best.bound = innerCandidates655_1 := by
  unfold candidates
  have row : edge 5309 = SearchData.row5309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_1 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_1 best = best := by
  rw [innerTail655_1, scan, if_neg (by decide), innerCandidates655_1_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_2
    (if 3 ≤ innerCandidates655_1.length then search edge 3 [4931, 5309] innerCandidates655_1 10240 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_2

def innerTail655_0 : List Nat := 5009 :: innerTail655_1
def innerCandidates655_0 : List Nat := [7109]
set_option maxRecDepth 4096 in
theorem innerCandidates655_0_checked : candidates edge 4 4931 5009 innerTail655_1 best.bound = innerCandidates655_0 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan655_0 : scan edge (search edge 3) 4 [4931] 4931 innerTail655_0 best = best := by
  rw [innerTail655_0, scan, if_neg (by decide), innerCandidates655_0_checked]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_1
    (if 3 ≤ innerCandidates655_0.length then search edge 3 [4931, 5009] innerCandidates655_0 9940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan655_1

theorem rootBranch655 : search edge 4 [4931] rootCandidates655 4931 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4931] 4931 innerTail655_0 best = best
  exact innerScan655_0

def rootCandidates654 : List Nat := [5147, 5189, 5897, 6197, 6599, 6803, 6869, 7481, 7559, 7607, 7829, 7877]
set_option maxRecDepth 4096 in
theorem rootCandidates654_checked : candidates edge 5 0 4919 tail655 best.bound = rootCandidates654 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext q; rfl
  rw [row]
  decide

def innerTail654_12 : List Nat := []
theorem innerScan654_12 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_12 best = best := rfl

def innerTail654_11 : List Nat := 7877 :: innerTail654_12
def innerCandidates654_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_11_checked : candidates edge 4 4919 7877 innerTail654_12 best.bound = innerCandidates654_11 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_11 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_11 best = best := by
  rw [innerTail654_11, scan, if_neg (by decide), innerCandidates654_11_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_12
    (if 3 ≤ innerCandidates654_11.length then search edge 3 [4919, 7877] innerCandidates654_11 12796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_12

def innerTail654_10 : List Nat := 7829 :: innerTail654_11
def innerCandidates654_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_10_checked : candidates edge 4 4919 7829 innerTail654_11 best.bound = innerCandidates654_10 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_10 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_10 best = best := by
  rw [innerTail654_10, scan, if_neg (by decide), innerCandidates654_10_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_11
    (if 3 ≤ innerCandidates654_10.length then search edge 3 [4919, 7829] innerCandidates654_10 12748 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_11

def innerTail654_9 : List Nat := 7607 :: innerTail654_10
def innerCandidates654_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_9_checked : candidates edge 4 4919 7607 innerTail654_10 best.bound = innerCandidates654_9 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_9 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_9 best = best := by
  rw [innerTail654_9, scan, if_neg (by decide), innerCandidates654_9_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_10
    (if 3 ≤ innerCandidates654_9.length then search edge 3 [4919, 7607] innerCandidates654_9 12526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_10

def innerTail654_8 : List Nat := 7559 :: innerTail654_9
def innerCandidates654_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_8_checked : candidates edge 4 4919 7559 innerTail654_9 best.bound = innerCandidates654_8 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_8 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_8 best = best := by
  rw [innerTail654_8, scan, if_neg (by decide), innerCandidates654_8_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_9
    (if 3 ≤ innerCandidates654_8.length then search edge 3 [4919, 7559] innerCandidates654_8 12478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_9

def innerTail654_7 : List Nat := 7481 :: innerTail654_8
def innerCandidates654_7 : List Nat := [7607]
set_option maxRecDepth 4096 in
theorem innerCandidates654_7_checked : candidates edge 4 4919 7481 innerTail654_8 best.bound = innerCandidates654_7 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_7 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_7 best = best := by
  rw [innerTail654_7, scan, if_neg (by decide), innerCandidates654_7_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_8
    (if 3 ≤ innerCandidates654_7.length then search edge 3 [4919, 7481] innerCandidates654_7 12400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_8

def innerTail654_6 : List Nat := 6869 :: innerTail654_7
def innerCandidates654_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_6_checked : candidates edge 4 4919 6869 innerTail654_7 best.bound = innerCandidates654_6 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_6 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_6 best = best := by
  rw [innerTail654_6, scan, if_neg (by decide), innerCandidates654_6_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_7
    (if 3 ≤ innerCandidates654_6.length then search edge 3 [4919, 6869] innerCandidates654_6 11788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_7

def innerTail654_5 : List Nat := 6803 :: innerTail654_6
def innerCandidates654_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_5_checked : candidates edge 4 4919 6803 innerTail654_6 best.bound = innerCandidates654_5 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_5 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_5 best = best := by
  rw [innerTail654_5, scan, if_neg (by decide), innerCandidates654_5_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_6
    (if 3 ≤ innerCandidates654_5.length then search edge 3 [4919, 6803] innerCandidates654_5 11722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_6

def innerTail654_4 : List Nat := 6599 :: innerTail654_5
def innerCandidates654_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_4_checked : candidates edge 4 4919 6599 innerTail654_5 best.bound = innerCandidates654_4 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_4 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_4 best = best := by
  rw [innerTail654_4, scan, if_neg (by decide), innerCandidates654_4_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_5
    (if 3 ≤ innerCandidates654_4.length then search edge 3 [4919, 6599] innerCandidates654_4 11518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_5

def innerTail654_3 : List Nat := 6197 :: innerTail654_4
def innerCandidates654_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_3_checked : candidates edge 4 4919 6197 innerTail654_4 best.bound = innerCandidates654_3 := by
  unfold candidates
  have row : edge 6197 = SearchData.row6197.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_3 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_3 best = best := by
  rw [innerTail654_3, scan, if_neg (by decide), innerCandidates654_3_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_4
    (if 3 ≤ innerCandidates654_3.length then search edge 3 [4919, 6197] innerCandidates654_3 11116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_4

def innerTail654_2 : List Nat := 5897 :: innerTail654_3
def innerCandidates654_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_2_checked : candidates edge 4 4919 5897 innerTail654_3 best.bound = innerCandidates654_2 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_2 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_2 best = best := by
  rw [innerTail654_2, scan, if_neg (by decide), innerCandidates654_2_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_3
    (if 3 ≤ innerCandidates654_2.length then search edge 3 [4919, 5897] innerCandidates654_2 10816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_3

def innerTail654_1 : List Nat := 5189 :: innerTail654_2
def innerCandidates654_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_1_checked : candidates edge 4 4919 5189 innerTail654_2 best.bound = innerCandidates654_1 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_1 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_1 best = best := by
  rw [innerTail654_1, scan, if_neg (by decide), innerCandidates654_1_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_2
    (if 3 ≤ innerCandidates654_1.length then search edge 3 [4919, 5189] innerCandidates654_1 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_2

def innerTail654_0 : List Nat := 5147 :: innerTail654_1
def innerCandidates654_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates654_0_checked : candidates edge 4 4919 5147 innerTail654_1 best.bound = innerCandidates654_0 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan654_0 : scan edge (search edge 3) 4 [4919] 4919 innerTail654_0 best = best := by
  rw [innerTail654_0, scan, if_neg (by decide), innerCandidates654_0_checked]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_1
    (if 3 ≤ innerCandidates654_0.length then search edge 3 [4919, 5147] innerCandidates654_0 10066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan654_1

theorem rootBranch654 : search edge 4 [4919] rootCandidates654 4919 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4919] 4919 innerTail654_0 best = best
  exact innerScan654_0

def rootCandidates653 : List Nat := [5023, 5443, 5503, 5647, 5857, 7177, 7723, 7933, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates653_checked : candidates edge 5 0 4909 tail654 best.bound = rootCandidates653 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext q; rfl
  rw [row]
  decide

def innerTail653_9 : List Nat := []
theorem innerScan653_9 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_9 best = best := rfl

def innerTail653_8 : List Nat := 8293 :: innerTail653_9
def innerCandidates653_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates653_8_checked : candidates edge 4 4909 8293 innerTail653_9 best.bound = innerCandidates653_8 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_8 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_8 best = best := by
  rw [innerTail653_8, scan, if_neg (by decide), innerCandidates653_8_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_9
    (if 3 ≤ innerCandidates653_8.length then search edge 3 [4909, 8293] innerCandidates653_8 13202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_9

def innerTail653_7 : List Nat := 7933 :: innerTail653_8
def innerCandidates653_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates653_7_checked : candidates edge 4 4909 7933 innerTail653_8 best.bound = innerCandidates653_7 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_7 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_7 best = best := by
  rw [innerTail653_7, scan, if_neg (by decide), innerCandidates653_7_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_8
    (if 3 ≤ innerCandidates653_7.length then search edge 3 [4909, 7933] innerCandidates653_7 12842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_8

def innerTail653_6 : List Nat := 7723 :: innerTail653_7
def innerCandidates653_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates653_6_checked : candidates edge 4 4909 7723 innerTail653_7 best.bound = innerCandidates653_6 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_6 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_6 best = best := by
  rw [innerTail653_6, scan, if_neg (by decide), innerCandidates653_6_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_7
    (if 3 ≤ innerCandidates653_6.length then search edge 3 [4909, 7723] innerCandidates653_6 12632 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_7

def innerTail653_5 : List Nat := 7177 :: innerTail653_6
def innerCandidates653_5 : List Nat := [7933]
set_option maxRecDepth 4096 in
theorem innerCandidates653_5_checked : candidates edge 4 4909 7177 innerTail653_6 best.bound = innerCandidates653_5 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_5 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_5 best = best := by
  rw [innerTail653_5, scan, if_neg (by decide), innerCandidates653_5_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_6
    (if 3 ≤ innerCandidates653_5.length then search edge 3 [4909, 7177] innerCandidates653_5 12086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_6

def innerTail653_4 : List Nat := 5857 :: innerTail653_5
def innerCandidates653_4 : List Nat := [7723, 8293]
set_option maxRecDepth 4096 in
theorem innerCandidates653_4_checked : candidates edge 4 4909 5857 innerTail653_5 best.bound = innerCandidates653_4 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_4 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_4 best = best := by
  rw [innerTail653_4, scan, if_neg (by decide), innerCandidates653_4_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_5
    (if 3 ≤ innerCandidates653_4.length then search edge 3 [4909, 5857] innerCandidates653_4 10766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_5

def innerTail653_3 : List Nat := 5647 :: innerTail653_4
def innerCandidates653_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates653_3_checked : candidates edge 4 4909 5647 innerTail653_4 best.bound = innerCandidates653_3 := by
  unfold candidates
  have row : edge 5647 = SearchData.row5647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_3 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_3 best = best := by
  rw [innerTail653_3, scan, if_neg (by decide), innerCandidates653_3_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_4
    (if 3 ≤ innerCandidates653_3.length then search edge 3 [4909, 5647] innerCandidates653_3 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_4

def innerTail653_2 : List Nat := 5503 :: innerTail653_3
def innerCandidates653_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates653_2_checked : candidates edge 4 4909 5503 innerTail653_3 best.bound = innerCandidates653_2 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_2 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_2 best = best := by
  rw [innerTail653_2, scan, if_neg (by decide), innerCandidates653_2_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_3
    (if 3 ≤ innerCandidates653_2.length then search edge 3 [4909, 5503] innerCandidates653_2 10412 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_3

def innerTail653_1 : List Nat := 5443 :: innerTail653_2
def innerCandidates653_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates653_1_checked : candidates edge 4 4909 5443 innerTail653_2 best.bound = innerCandidates653_1 := by
  unfold candidates
  have row : edge 5443 = SearchData.row5443.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_1 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_1 best = best := by
  rw [innerTail653_1, scan, if_neg (by decide), innerCandidates653_1_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_2
    (if 3 ≤ innerCandidates653_1.length then search edge 3 [4909, 5443] innerCandidates653_1 10352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_2

def innerTail653_0 : List Nat := 5023 :: innerTail653_1
def innerCandidates653_0 : List Nat := [5443]
set_option maxRecDepth 4096 in
theorem innerCandidates653_0_checked : candidates edge 4 4909 5023 innerTail653_1 best.bound = innerCandidates653_0 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan653_0 : scan edge (search edge 3) 4 [4909] 4909 innerTail653_0 best = best := by
  rw [innerTail653_0, scan, if_neg (by decide), innerCandidates653_0_checked]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_1
    (if 3 ≤ innerCandidates653_0.length then search edge 3 [4909, 5023] innerCandidates653_0 9932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan653_1

theorem rootBranch653 : search edge 4 [4909] rootCandidates653 4909 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4909] 4909 innerTail653_0 best = best
  exact innerScan653_0

end ProjectEulerSolutions.PrimePairUpper.Search
