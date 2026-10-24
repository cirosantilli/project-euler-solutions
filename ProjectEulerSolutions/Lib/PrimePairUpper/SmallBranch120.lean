import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates320 : List Nat := [2267, 2309, 2621, 2663, 3167, 3191, 3881, 3947, 4049, 4091, 4649, 5153, 5519, 6053, 6737, 7349, 7547, 7691, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates320_checked : candidates edge 5 0 2141 tail321 best.bound = rootCandidates320 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext q; rfl
  rw [row]
  decide

def innerTail320_19 : List Nat := []
theorem innerScan320_19 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_19 best = best := rfl

def innerTail320_18 : List Nat := 7727 :: innerTail320_19
def innerCandidates320_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_18_checked : candidates edge 4 2141 7727 innerTail320_19 best.bound = innerCandidates320_18 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_18 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_18 best = best := by
  rw [innerTail320_18, scan, if_neg (by decide), innerCandidates320_18_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_19
    (if 3 ≤ innerCandidates320_18.length then search edge 3 [2141, 7727] innerCandidates320_18 9868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_19

def innerTail320_17 : List Nat := 7691 :: innerTail320_18
def innerCandidates320_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_17_checked : candidates edge 4 2141 7691 innerTail320_18 best.bound = innerCandidates320_17 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_17 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_17 best = best := by
  rw [innerTail320_17, scan, if_neg (by decide), innerCandidates320_17_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_18
    (if 3 ≤ innerCandidates320_17.length then search edge 3 [2141, 7691] innerCandidates320_17 9832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_18

def innerTail320_16 : List Nat := 7547 :: innerTail320_17
def innerCandidates320_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_16_checked : candidates edge 4 2141 7547 innerTail320_17 best.bound = innerCandidates320_16 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_16 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_16 best = best := by
  rw [innerTail320_16, scan, if_neg (by decide), innerCandidates320_16_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_17
    (if 3 ≤ innerCandidates320_16.length then search edge 3 [2141, 7547] innerCandidates320_16 9688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_17

def innerTail320_15 : List Nat := 7349 :: innerTail320_16
def innerCandidates320_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_15_checked : candidates edge 4 2141 7349 innerTail320_16 best.bound = innerCandidates320_15 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_15 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_15 best = best := by
  rw [innerTail320_15, scan, if_neg (by decide), innerCandidates320_15_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_16
    (if 3 ≤ innerCandidates320_15.length then search edge 3 [2141, 7349] innerCandidates320_15 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_16

def innerTail320_14 : List Nat := 6737 :: innerTail320_15
def innerCandidates320_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_14_checked : candidates edge 4 2141 6737 innerTail320_15 best.bound = innerCandidates320_14 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_14 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_14 best = best := by
  rw [innerTail320_14, scan, if_neg (by decide), innerCandidates320_14_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_15
    (if 3 ≤ innerCandidates320_14.length then search edge 3 [2141, 6737] innerCandidates320_14 8878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_15

def innerTail320_13 : List Nat := 6053 :: innerTail320_14
def innerCandidates320_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_13_checked : candidates edge 4 2141 6053 innerTail320_14 best.bound = innerCandidates320_13 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_13 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_13 best = best := by
  rw [innerTail320_13, scan, if_neg (by decide), innerCandidates320_13_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_14
    (if 3 ≤ innerCandidates320_13.length then search edge 3 [2141, 6053] innerCandidates320_13 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_14

def innerTail320_12 : List Nat := 5519 :: innerTail320_13
def innerCandidates320_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_12_checked : candidates edge 4 2141 5519 innerTail320_13 best.bound = innerCandidates320_12 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_12 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_12 best = best := by
  rw [innerTail320_12, scan, if_neg (by decide), innerCandidates320_12_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_13
    (if 3 ≤ innerCandidates320_12.length then search edge 3 [2141, 5519] innerCandidates320_12 7660 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_13

def innerTail320_11 : List Nat := 5153 :: innerTail320_12
def innerCandidates320_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_11_checked : candidates edge 4 2141 5153 innerTail320_12 best.bound = innerCandidates320_11 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_11 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_11 best = best := by
  rw [innerTail320_11, scan, if_neg (by decide), innerCandidates320_11_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_12
    (if 3 ≤ innerCandidates320_11.length then search edge 3 [2141, 5153] innerCandidates320_11 7294 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_12

def innerTail320_10 : List Nat := 4649 :: innerTail320_11
def innerCandidates320_10 : List Nat := [7727]
set_option maxRecDepth 4096 in
theorem innerCandidates320_10_checked : candidates edge 4 2141 4649 innerTail320_11 best.bound = innerCandidates320_10 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_10 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_10 best = best := by
  rw [innerTail320_10, scan, if_neg (by decide), innerCandidates320_10_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_11
    (if 3 ≤ innerCandidates320_10.length then search edge 3 [2141, 4649] innerCandidates320_10 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_11

def innerTail320_9 : List Nat := 4091 :: innerTail320_10
def innerCandidates320_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_9_checked : candidates edge 4 2141 4091 innerTail320_10 best.bound = innerCandidates320_9 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_9 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_9 best = best := by
  rw [innerTail320_9, scan, if_neg (by decide), innerCandidates320_9_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_10
    (if 3 ≤ innerCandidates320_9.length then search edge 3 [2141, 4091] innerCandidates320_9 6232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_10

def innerTail320_8 : List Nat := 4049 :: innerTail320_9
def innerCandidates320_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_8_checked : candidates edge 4 2141 4049 innerTail320_9 best.bound = innerCandidates320_8 := by
  unfold candidates
  have row : edge 4049 = SearchData.row4049.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_8 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_8 best = best := by
  rw [innerTail320_8, scan, if_neg (by decide), innerCandidates320_8_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_9
    (if 3 ≤ innerCandidates320_8.length then search edge 3 [2141, 4049] innerCandidates320_8 6190 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_9

def innerTail320_7 : List Nat := 3947 :: innerTail320_8
def innerCandidates320_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_7_checked : candidates edge 4 2141 3947 innerTail320_8 best.bound = innerCandidates320_7 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_7 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_7 best = best := by
  rw [innerTail320_7, scan, if_neg (by decide), innerCandidates320_7_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_8
    (if 3 ≤ innerCandidates320_7.length then search edge 3 [2141, 3947] innerCandidates320_7 6088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_8

def innerTail320_6 : List Nat := 3881 :: innerTail320_7
def innerCandidates320_6 : List Nat := [5519]
set_option maxRecDepth 4096 in
theorem innerCandidates320_6_checked : candidates edge 4 2141 3881 innerTail320_7 best.bound = innerCandidates320_6 := by
  unfold candidates
  have row : edge 3881 = SearchData.row3881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_6 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_6 best = best := by
  rw [innerTail320_6, scan, if_neg (by decide), innerCandidates320_6_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_7
    (if 3 ≤ innerCandidates320_6.length then search edge 3 [2141, 3881] innerCandidates320_6 6022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_7

def innerTail320_5 : List Nat := 3191 :: innerTail320_6
def innerCandidates320_5 : List Nat := [7547]
set_option maxRecDepth 4096 in
theorem innerCandidates320_5_checked : candidates edge 4 2141 3191 innerTail320_6 best.bound = innerCandidates320_5 := by
  unfold candidates
  have row : edge 3191 = SearchData.row3191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_5 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_5 best = best := by
  rw [innerTail320_5, scan, if_neg (by decide), innerCandidates320_5_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_6
    (if 3 ≤ innerCandidates320_5.length then search edge 3 [2141, 3191] innerCandidates320_5 5332 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_6

def innerTail320_4 : List Nat := 3167 :: innerTail320_5
def innerCandidates320_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_4_checked : candidates edge 4 2141 3167 innerTail320_5 best.bound = innerCandidates320_4 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_4 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_4 best = best := by
  rw [innerTail320_4, scan, if_neg (by decide), innerCandidates320_4_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_5
    (if 3 ≤ innerCandidates320_4.length then search edge 3 [2141, 3167] innerCandidates320_4 5308 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_5

def innerTail320_3 : List Nat := 2663 :: innerTail320_4
def innerCandidates320_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates320_3_checked : candidates edge 4 2141 2663 innerTail320_4 best.bound = innerCandidates320_3 := by
  unfold candidates
  have row : edge 2663 = PrimePairCache.smallRow2663.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_3 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_3 best = best := by
  rw [innerTail320_3, scan, if_neg (by decide), innerCandidates320_3_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_4
    (if 3 ≤ innerCandidates320_3.length then search edge 3 [2141, 2663] innerCandidates320_3 4804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_4

def innerTail320_2 : List Nat := 2621 :: innerTail320_3
def innerCandidates320_2 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates320_2_checked : candidates edge 4 2141 2621 innerTail320_3 best.bound = innerCandidates320_2 := by
  unfold candidates
  have row : edge 2621 = PrimePairCache.smallRow2621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_2 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_2 best = best := by
  rw [innerTail320_2, scan, if_neg (by decide), innerCandidates320_2_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_3
    (if 3 ≤ innerCandidates320_2.length then search edge 3 [2141, 2621] innerCandidates320_2 4762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_3

def innerTail320_1 : List Nat := 2309 :: innerTail320_2
def innerCandidates320_1 : List Nat := [7691]
set_option maxRecDepth 4096 in
theorem innerCandidates320_1_checked : candidates edge 4 2141 2309 innerTail320_2 best.bound = innerCandidates320_1 := by
  unfold candidates
  have row : edge 2309 = PrimePairCache.smallRow2309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_1 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_1 best = best := by
  rw [innerTail320_1, scan, if_neg (by decide), innerCandidates320_1_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_2
    (if 3 ≤ innerCandidates320_1.length then search edge 3 [2141, 2309] innerCandidates320_1 4450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_2

def innerTail320_0 : List Nat := 2267 :: innerTail320_1
def innerCandidates320_0 : List Nat := [6053]
set_option maxRecDepth 4096 in
theorem innerCandidates320_0_checked : candidates edge 4 2141 2267 innerTail320_1 best.bound = innerCandidates320_0 := by
  unfold candidates
  have row : edge 2267 = PrimePairCache.smallRow2267.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan320_0 : scan edge (search edge 3) 4 [2141] 2141 innerTail320_0 best = best := by
  rw [innerTail320_0, scan, if_neg (by decide), innerCandidates320_0_checked]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_1
    (if 3 ≤ innerCandidates320_0.length then search edge 3 [2141, 2267] innerCandidates320_0 4408 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan320_1

theorem rootBranch320 : search edge 4 [2141] rootCandidates320 2141 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2141] 2141 innerTail320_0 best = best
  exact innerScan320_0

def rootCandidates319 : List Nat := [2293, 2539, 3061, 3697, 4051, 4513, 5503, 6073, 6133, 6217, 6571, 6991, 7039, 7159, 7459, 7561, 8233, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates319_checked : candidates edge 5 0 2137 tail320 best.bound = rootCandidates319 := by
  unfold candidates
  have row : edge 2137 = PrimePairCache.smallRow2137.contains := by funext q; rfl
  rw [row]
  decide

def innerTail319_18 : List Nat := []
theorem innerScan319_18 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_18 best = best := rfl

def innerTail319_17 : List Nat := 8317 :: innerTail319_18
def innerCandidates319_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_17_checked : candidates edge 4 2137 8317 innerTail319_18 best.bound = innerCandidates319_17 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_17 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_17 best = best := by
  rw [innerTail319_17, scan, if_neg (by decide), innerCandidates319_17_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_18
    (if 3 ≤ innerCandidates319_17.length then search edge 3 [2137, 8317] innerCandidates319_17 10454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_18

def innerTail319_16 : List Nat := 8233 :: innerTail319_17
def innerCandidates319_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_16_checked : candidates edge 4 2137 8233 innerTail319_17 best.bound = innerCandidates319_16 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_16 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_16 best = best := by
  rw [innerTail319_16, scan, if_neg (by decide), innerCandidates319_16_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_17
    (if 3 ≤ innerCandidates319_16.length then search edge 3 [2137, 8233] innerCandidates319_16 10370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_17

def innerTail319_15 : List Nat := 7561 :: innerTail319_16
def innerCandidates319_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_15_checked : candidates edge 4 2137 7561 innerTail319_16 best.bound = innerCandidates319_15 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_15 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_15 best = best := by
  rw [innerTail319_15, scan, if_neg (by decide), innerCandidates319_15_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_16
    (if 3 ≤ innerCandidates319_15.length then search edge 3 [2137, 7561] innerCandidates319_15 9698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_16

def innerTail319_14 : List Nat := 7459 :: innerTail319_15
def innerCandidates319_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_14_checked : candidates edge 4 2137 7459 innerTail319_15 best.bound = innerCandidates319_14 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_14 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_14 best = best := by
  rw [innerTail319_14, scan, if_neg (by decide), innerCandidates319_14_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_15
    (if 3 ≤ innerCandidates319_14.length then search edge 3 [2137, 7459] innerCandidates319_14 9596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_15

def innerTail319_13 : List Nat := 7159 :: innerTail319_14
def innerCandidates319_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_13_checked : candidates edge 4 2137 7159 innerTail319_14 best.bound = innerCandidates319_13 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_13 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_13 best = best := by
  rw [innerTail319_13, scan, if_neg (by decide), innerCandidates319_13_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_14
    (if 3 ≤ innerCandidates319_13.length then search edge 3 [2137, 7159] innerCandidates319_13 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_14

def innerTail319_12 : List Nat := 7039 :: innerTail319_13
def innerCandidates319_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_12_checked : candidates edge 4 2137 7039 innerTail319_13 best.bound = innerCandidates319_12 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_12 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_12 best = best := by
  rw [innerTail319_12, scan, if_neg (by decide), innerCandidates319_12_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_13
    (if 3 ≤ innerCandidates319_12.length then search edge 3 [2137, 7039] innerCandidates319_12 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_13

def innerTail319_11 : List Nat := 6991 :: innerTail319_12
def innerCandidates319_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_11_checked : candidates edge 4 2137 6991 innerTail319_12 best.bound = innerCandidates319_11 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_11 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_11 best = best := by
  rw [innerTail319_11, scan, if_neg (by decide), innerCandidates319_11_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_12
    (if 3 ≤ innerCandidates319_11.length then search edge 3 [2137, 6991] innerCandidates319_11 9128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_12

def innerTail319_10 : List Nat := 6571 :: innerTail319_11
def innerCandidates319_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_10_checked : candidates edge 4 2137 6571 innerTail319_11 best.bound = innerCandidates319_10 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_10 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_10 best = best := by
  rw [innerTail319_10, scan, if_neg (by decide), innerCandidates319_10_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_11
    (if 3 ≤ innerCandidates319_10.length then search edge 3 [2137, 6571] innerCandidates319_10 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_11

def innerTail319_9 : List Nat := 6217 :: innerTail319_10
def innerCandidates319_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_9_checked : candidates edge 4 2137 6217 innerTail319_10 best.bound = innerCandidates319_9 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_9 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_9 best = best := by
  rw [innerTail319_9, scan, if_neg (by decide), innerCandidates319_9_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_10
    (if 3 ≤ innerCandidates319_9.length then search edge 3 [2137, 6217] innerCandidates319_9 8354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_10

def innerTail319_8 : List Nat := 6133 :: innerTail319_9
def innerCandidates319_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_8_checked : candidates edge 4 2137 6133 innerTail319_9 best.bound = innerCandidates319_8 := by
  unfold candidates
  have row : edge 6133 = SearchData.row6133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_8 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_8 best = best := by
  rw [innerTail319_8, scan, if_neg (by decide), innerCandidates319_8_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_9
    (if 3 ≤ innerCandidates319_8.length then search edge 3 [2137, 6133] innerCandidates319_8 8270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_9

def innerTail319_7 : List Nat := 6073 :: innerTail319_8
def innerCandidates319_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_7_checked : candidates edge 4 2137 6073 innerTail319_8 best.bound = innerCandidates319_7 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_7 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_7 best = best := by
  rw [innerTail319_7, scan, if_neg (by decide), innerCandidates319_7_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_8
    (if 3 ≤ innerCandidates319_7.length then search edge 3 [2137, 6073] innerCandidates319_7 8210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_8

def innerTail319_6 : List Nat := 5503 :: innerTail319_7
def innerCandidates319_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_6_checked : candidates edge 4 2137 5503 innerTail319_7 best.bound = innerCandidates319_6 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_6 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_6 best = best := by
  rw [innerTail319_6, scan, if_neg (by decide), innerCandidates319_6_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_7
    (if 3 ≤ innerCandidates319_6.length then search edge 3 [2137, 5503] innerCandidates319_6 7640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_7

def innerTail319_5 : List Nat := 4513 :: innerTail319_6
def innerCandidates319_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_5_checked : candidates edge 4 2137 4513 innerTail319_6 best.bound = innerCandidates319_5 := by
  unfold candidates
  have row : edge 4513 = SearchData.row4513.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_5 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_5 best = best := by
  rw [innerTail319_5, scan, if_neg (by decide), innerCandidates319_5_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_6
    (if 3 ≤ innerCandidates319_5.length then search edge 3 [2137, 4513] innerCandidates319_5 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_6

def innerTail319_4 : List Nat := 4051 :: innerTail319_5
def innerCandidates319_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_4_checked : candidates edge 4 2137 4051 innerTail319_5 best.bound = innerCandidates319_4 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_4 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_4 best = best := by
  rw [innerTail319_4, scan, if_neg (by decide), innerCandidates319_4_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_5
    (if 3 ≤ innerCandidates319_4.length then search edge 3 [2137, 4051] innerCandidates319_4 6188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_5

def innerTail319_3 : List Nat := 3697 :: innerTail319_4
def innerCandidates319_3 : List Nat := [7459]
set_option maxRecDepth 4096 in
theorem innerCandidates319_3_checked : candidates edge 4 2137 3697 innerTail319_4 best.bound = innerCandidates319_3 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_3 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_3 best = best := by
  rw [innerTail319_3, scan, if_neg (by decide), innerCandidates319_3_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_4
    (if 3 ≤ innerCandidates319_3.length then search edge 3 [2137, 3697] innerCandidates319_3 5834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_4

def innerTail319_2 : List Nat := 3061 :: innerTail319_3
def innerCandidates319_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates319_2_checked : candidates edge 4 2137 3061 innerTail319_3 best.bound = innerCandidates319_2 := by
  unfold candidates
  have row : edge 3061 = SearchData.row3061.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_2 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_2 best = best := by
  rw [innerTail319_2, scan, if_neg (by decide), innerCandidates319_2_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_3
    (if 3 ≤ innerCandidates319_2.length then search edge 3 [2137, 3061] innerCandidates319_2 5198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_3

def innerTail319_1 : List Nat := 2539 :: innerTail319_2
def innerCandidates319_1 : List Nat := [7159]
set_option maxRecDepth 4096 in
theorem innerCandidates319_1_checked : candidates edge 4 2137 2539 innerTail319_2 best.bound = innerCandidates319_1 := by
  unfold candidates
  have row : edge 2539 = PrimePairCache.smallRow2539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_1 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_1 best = best := by
  rw [innerTail319_1, scan, if_neg (by decide), innerCandidates319_1_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_2
    (if 3 ≤ innerCandidates319_1.length then search edge 3 [2137, 2539] innerCandidates319_1 4676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_2

def innerTail319_0 : List Nat := 2293 :: innerTail319_1
def innerCandidates319_0 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates319_0_checked : candidates edge 4 2137 2293 innerTail319_1 best.bound = innerCandidates319_0 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan319_0 : scan edge (search edge 3) 4 [2137] 2137 innerTail319_0 best = best := by
  rw [innerTail319_0, scan, if_neg (by decide), innerCandidates319_0_checked]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_1
    (if 3 ≤ innerCandidates319_0.length then search edge 3 [2137, 2293] innerCandidates319_0 4430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan319_1

theorem rootBranch319 : search edge 4 [2137] rootCandidates319 2137 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2137] 2137 innerTail319_0 best = best
  exact innerScan319_0

def rootCandidates318 : List Nat := [2293, 2677, 2833, 3037, 3343, 3541, 3547, 4297, 4567, 4933, 5023, 5563, 5689, 6619, 7027, 7129, 7669, 7873, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates318_checked : candidates edge 5 0 2131 tail319 best.bound = rootCandidates318 := by
  unfold candidates
  have row : edge 2131 = PrimePairCache.smallRow2131.contains := by funext q; rfl
  rw [row]
  decide

def innerTail318_19 : List Nat := []
theorem innerScan318_19 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_19 best = best := rfl

def innerTail318_18 : List Nat := 8377 :: innerTail318_19
def innerCandidates318_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_18_checked : candidates edge 4 2131 8377 innerTail318_19 best.bound = innerCandidates318_18 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_18 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_18 best = best := by
  rw [innerTail318_18, scan, if_neg (by decide), innerCandidates318_18_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_19
    (if 3 ≤ innerCandidates318_18.length then search edge 3 [2131, 8377] innerCandidates318_18 10508 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_19

def innerTail318_17 : List Nat := 7873 :: innerTail318_18
def innerCandidates318_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_17_checked : candidates edge 4 2131 7873 innerTail318_18 best.bound = innerCandidates318_17 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_17 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_17 best = best := by
  rw [innerTail318_17, scan, if_neg (by decide), innerCandidates318_17_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_18
    (if 3 ≤ innerCandidates318_17.length then search edge 3 [2131, 7873] innerCandidates318_17 10004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_18

def innerTail318_16 : List Nat := 7669 :: innerTail318_17
def innerCandidates318_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_16_checked : candidates edge 4 2131 7669 innerTail318_17 best.bound = innerCandidates318_16 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_16 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_16 best = best := by
  rw [innerTail318_16, scan, if_neg (by decide), innerCandidates318_16_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_17
    (if 3 ≤ innerCandidates318_16.length then search edge 3 [2131, 7669] innerCandidates318_16 9800 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_17

def innerTail318_15 : List Nat := 7129 :: innerTail318_16
def innerCandidates318_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_15_checked : candidates edge 4 2131 7129 innerTail318_16 best.bound = innerCandidates318_15 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_15 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_15 best = best := by
  rw [innerTail318_15, scan, if_neg (by decide), innerCandidates318_15_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_16
    (if 3 ≤ innerCandidates318_15.length then search edge 3 [2131, 7129] innerCandidates318_15 9260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_16

def innerTail318_14 : List Nat := 7027 :: innerTail318_15
def innerCandidates318_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_14_checked : candidates edge 4 2131 7027 innerTail318_15 best.bound = innerCandidates318_14 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_14 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_14 best = best := by
  rw [innerTail318_14, scan, if_neg (by decide), innerCandidates318_14_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_15
    (if 3 ≤ innerCandidates318_14.length then search edge 3 [2131, 7027] innerCandidates318_14 9158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_15

def innerTail318_13 : List Nat := 6619 :: innerTail318_14
def innerCandidates318_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_13_checked : candidates edge 4 2131 6619 innerTail318_14 best.bound = innerCandidates318_13 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_13 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_13 best = best := by
  rw [innerTail318_13, scan, if_neg (by decide), innerCandidates318_13_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_14
    (if 3 ≤ innerCandidates318_13.length then search edge 3 [2131, 6619] innerCandidates318_13 8750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_14

def innerTail318_12 : List Nat := 5689 :: innerTail318_13
def innerCandidates318_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_12_checked : candidates edge 4 2131 5689 innerTail318_13 best.bound = innerCandidates318_12 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_12 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_12 best = best := by
  rw [innerTail318_12, scan, if_neg (by decide), innerCandidates318_12_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_13
    (if 3 ≤ innerCandidates318_12.length then search edge 3 [2131, 5689] innerCandidates318_12 7820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_13

def innerTail318_11 : List Nat := 5563 :: innerTail318_12
def innerCandidates318_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_11_checked : candidates edge 4 2131 5563 innerTail318_12 best.bound = innerCandidates318_11 := by
  unfold candidates
  have row : edge 5563 = SearchData.row5563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_11 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_11 best = best := by
  rw [innerTail318_11, scan, if_neg (by decide), innerCandidates318_11_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_12
    (if 3 ≤ innerCandidates318_11.length then search edge 3 [2131, 5563] innerCandidates318_11 7694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_12

def innerTail318_10 : List Nat := 5023 :: innerTail318_11
def innerCandidates318_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_10_checked : candidates edge 4 2131 5023 innerTail318_11 best.bound = innerCandidates318_10 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_10 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_10 best = best := by
  rw [innerTail318_10, scan, if_neg (by decide), innerCandidates318_10_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_11
    (if 3 ≤ innerCandidates318_10.length then search edge 3 [2131, 5023] innerCandidates318_10 7154 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_11

def innerTail318_9 : List Nat := 4933 :: innerTail318_10
def innerCandidates318_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_9_checked : candidates edge 4 2131 4933 innerTail318_10 best.bound = innerCandidates318_9 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_9 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_9 best = best := by
  rw [innerTail318_9, scan, if_neg (by decide), innerCandidates318_9_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_10
    (if 3 ≤ innerCandidates318_9.length then search edge 3 [2131, 4933] innerCandidates318_9 7064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_10

def innerTail318_8 : List Nat := 4567 :: innerTail318_9
def innerCandidates318_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_8_checked : candidates edge 4 2131 4567 innerTail318_9 best.bound = innerCandidates318_8 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_8 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_8 best = best := by
  rw [innerTail318_8, scan, if_neg (by decide), innerCandidates318_8_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_9
    (if 3 ≤ innerCandidates318_8.length then search edge 3 [2131, 4567] innerCandidates318_8 6698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_9

def innerTail318_7 : List Nat := 4297 :: innerTail318_8
def innerCandidates318_7 : List Nat := [5689]
set_option maxRecDepth 4096 in
theorem innerCandidates318_7_checked : candidates edge 4 2131 4297 innerTail318_8 best.bound = innerCandidates318_7 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_7 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_7 best = best := by
  rw [innerTail318_7, scan, if_neg (by decide), innerCandidates318_7_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_8
    (if 3 ≤ innerCandidates318_7.length then search edge 3 [2131, 4297] innerCandidates318_7 6428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_8

def innerTail318_6 : List Nat := 3547 :: innerTail318_7
def innerCandidates318_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_6_checked : candidates edge 4 2131 3547 innerTail318_7 best.bound = innerCandidates318_6 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_6 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_6 best = best := by
  rw [innerTail318_6, scan, if_neg (by decide), innerCandidates318_6_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_7
    (if 3 ≤ innerCandidates318_6.length then search edge 3 [2131, 3547] innerCandidates318_6 5678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_7

def innerTail318_5 : List Nat := 3541 :: innerTail318_6
def innerCandidates318_5 : List Nat := [4567]
set_option maxRecDepth 4096 in
theorem innerCandidates318_5_checked : candidates edge 4 2131 3541 innerTail318_6 best.bound = innerCandidates318_5 := by
  unfold candidates
  have row : edge 3541 = SearchData.row3541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_5 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_5 best = best := by
  rw [innerTail318_5, scan, if_neg (by decide), innerCandidates318_5_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_6
    (if 3 ≤ innerCandidates318_5.length then search edge 3 [2131, 3541] innerCandidates318_5 5672 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_6

def innerTail318_4 : List Nat := 3343 :: innerTail318_5
def innerCandidates318_4 : List Nat := [4567]
set_option maxRecDepth 4096 in
theorem innerCandidates318_4_checked : candidates edge 4 2131 3343 innerTail318_5 best.bound = innerCandidates318_4 := by
  unfold candidates
  have row : edge 3343 = SearchData.row3343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_4 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_4 best = best := by
  rw [innerTail318_4, scan, if_neg (by decide), innerCandidates318_4_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_5
    (if 3 ≤ innerCandidates318_4.length then search edge 3 [2131, 3343] innerCandidates318_4 5474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_5

def innerTail318_3 : List Nat := 3037 :: innerTail318_4
def innerCandidates318_3 : List Nat := [5689, 7129]
set_option maxRecDepth 4096 in
theorem innerCandidates318_3_checked : candidates edge 4 2131 3037 innerTail318_4 best.bound = innerCandidates318_3 := by
  unfold candidates
  have row : edge 3037 = SearchData.row3037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_3 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_3 best = best := by
  rw [innerTail318_3, scan, if_neg (by decide), innerCandidates318_3_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_4
    (if 3 ≤ innerCandidates318_3.length then search edge 3 [2131, 3037] innerCandidates318_3 5168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_4

def innerTail318_2 : List Nat := 2833 :: innerTail318_3
def innerCandidates318_2 : List Nat := [3037]
set_option maxRecDepth 4096 in
theorem innerCandidates318_2_checked : candidates edge 4 2131 2833 innerTail318_3 best.bound = innerCandidates318_2 := by
  unfold candidates
  have row : edge 2833 = SearchData.row2833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_2 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_2 best = best := by
  rw [innerTail318_2, scan, if_neg (by decide), innerCandidates318_2_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_3
    (if 3 ≤ innerCandidates318_2.length then search edge 3 [2131, 2833] innerCandidates318_2 4964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_3

def innerTail318_1 : List Nat := 2677 :: innerTail318_2
def innerCandidates318_1 : List Nat := [2833, 5563]
set_option maxRecDepth 4096 in
theorem innerCandidates318_1_checked : candidates edge 4 2131 2677 innerTail318_2 best.bound = innerCandidates318_1 := by
  unfold candidates
  have row : edge 2677 = SearchData.row2677.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_1 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_1 best = best := by
  rw [innerTail318_1, scan, if_neg (by decide), innerCandidates318_1_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_2
    (if 3 ≤ innerCandidates318_1.length then search edge 3 [2131, 2677] innerCandidates318_1 4808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_2

def innerTail318_0 : List Nat := 2293 :: innerTail318_1
def innerCandidates318_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates318_0_checked : candidates edge 4 2131 2293 innerTail318_1 best.bound = innerCandidates318_0 := by
  unfold candidates
  have row : edge 2293 = PrimePairCache.smallRow2293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan318_0 : scan edge (search edge 3) 4 [2131] 2131 innerTail318_0 best = best := by
  rw [innerTail318_0, scan, if_neg (by decide), innerCandidates318_0_checked]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_1
    (if 3 ≤ innerCandidates318_0.length then search edge 3 [2131, 2293] innerCandidates318_0 4424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan318_1

theorem rootBranch318 : search edge 4 [2131] rootCandidates318 2131 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2131] 2131 innerTail318_0 best = best
  exact innerScan318_0

end ProjectEulerSolutions.PrimePairUpper.Search
