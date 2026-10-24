import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates284 : List Nat := [2017, 2269, 2437, 2719, 3511, 5527, 5569, 6301, 6427, 6709, 6841, 6961, 6997, 7507, 7963, 8011, 8059, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates284_checked : candidates edge 5 0 1873 tail285 best.bound = rootCandidates284 := by
  unfold candidates
  have row : edge 1873 = PrimePairCache.smallRow1873.contains := by funext q; rfl
  rw [row]
  decide

def innerTail284_18 : List Nat := []
theorem innerScan284_18 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_18 best = best := rfl

def innerTail284_17 : List Nat := 8179 :: innerTail284_18
def innerCandidates284_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_17_checked : candidates edge 4 1873 8179 innerTail284_18 best.bound = innerCandidates284_17 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_17 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_17 best = best := by
  rw [innerTail284_17, scan, if_neg (by decide), innerCandidates284_17_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_18
    (if 3 ≤ innerCandidates284_17.length then search edge 3 [1873, 8179] innerCandidates284_17 10052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_18

def innerTail284_16 : List Nat := 8059 :: innerTail284_17
def innerCandidates284_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_16_checked : candidates edge 4 1873 8059 innerTail284_17 best.bound = innerCandidates284_16 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_16 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_16 best = best := by
  rw [innerTail284_16, scan, if_neg (by decide), innerCandidates284_16_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_17
    (if 3 ≤ innerCandidates284_16.length then search edge 3 [1873, 8059] innerCandidates284_16 9932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_17

def innerTail284_15 : List Nat := 8011 :: innerTail284_16
def innerCandidates284_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_15_checked : candidates edge 4 1873 8011 innerTail284_16 best.bound = innerCandidates284_15 := by
  unfold candidates
  have row : edge 8011 = SearchData.row8011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_15 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_15 best = best := by
  rw [innerTail284_15, scan, if_neg (by decide), innerCandidates284_15_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_16
    (if 3 ≤ innerCandidates284_15.length then search edge 3 [1873, 8011] innerCandidates284_15 9884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_16

def innerTail284_14 : List Nat := 7963 :: innerTail284_15
def innerCandidates284_14 : List Nat := [8011]
set_option maxRecDepth 4096 in
theorem innerCandidates284_14_checked : candidates edge 4 1873 7963 innerTail284_15 best.bound = innerCandidates284_14 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_14 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_14 best = best := by
  rw [innerTail284_14, scan, if_neg (by decide), innerCandidates284_14_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_15
    (if 3 ≤ innerCandidates284_14.length then search edge 3 [1873, 7963] innerCandidates284_14 9836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_15

def innerTail284_13 : List Nat := 7507 :: innerTail284_14
def innerCandidates284_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_13_checked : candidates edge 4 1873 7507 innerTail284_14 best.bound = innerCandidates284_13 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_13 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_13 best = best := by
  rw [innerTail284_13, scan, if_neg (by decide), innerCandidates284_13_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_14
    (if 3 ≤ innerCandidates284_13.length then search edge 3 [1873, 7507] innerCandidates284_13 9380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_14

def innerTail284_12 : List Nat := 6997 :: innerTail284_13
def innerCandidates284_12 : List Nat := [7507, 8011]
set_option maxRecDepth 4096 in
theorem innerCandidates284_12_checked : candidates edge 4 1873 6997 innerTail284_13 best.bound = innerCandidates284_12 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_12 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_12 best = best := by
  rw [innerTail284_12, scan, if_neg (by decide), innerCandidates284_12_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_13
    (if 3 ≤ innerCandidates284_12.length then search edge 3 [1873, 6997] innerCandidates284_12 8870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_13

def innerTail284_11 : List Nat := 6961 :: innerTail284_12
def innerCandidates284_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_11_checked : candidates edge 4 1873 6961 innerTail284_12 best.bound = innerCandidates284_11 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_11 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_11 best = best := by
  rw [innerTail284_11, scan, if_neg (by decide), innerCandidates284_11_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_12
    (if 3 ≤ innerCandidates284_11.length then search edge 3 [1873, 6961] innerCandidates284_11 8834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_12

def innerTail284_10 : List Nat := 6841 :: innerTail284_11
def innerCandidates284_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_10_checked : candidates edge 4 1873 6841 innerTail284_11 best.bound = innerCandidates284_10 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_10 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_10 best = best := by
  rw [innerTail284_10, scan, if_neg (by decide), innerCandidates284_10_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_11
    (if 3 ≤ innerCandidates284_10.length then search edge 3 [1873, 6841] innerCandidates284_10 8714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_11

def innerTail284_9 : List Nat := 6709 :: innerTail284_10
def innerCandidates284_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_9_checked : candidates edge 4 1873 6709 innerTail284_10 best.bound = innerCandidates284_9 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_9 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_9 best = best := by
  rw [innerTail284_9, scan, if_neg (by decide), innerCandidates284_9_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_10
    (if 3 ≤ innerCandidates284_9.length then search edge 3 [1873, 6709] innerCandidates284_9 8582 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_10

def innerTail284_8 : List Nat := 6427 :: innerTail284_9
def innerCandidates284_8 : List Nat := [6997]
set_option maxRecDepth 4096 in
theorem innerCandidates284_8_checked : candidates edge 4 1873 6427 innerTail284_9 best.bound = innerCandidates284_8 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_8 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_8 best = best := by
  rw [innerTail284_8, scan, if_neg (by decide), innerCandidates284_8_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_9
    (if 3 ≤ innerCandidates284_8.length then search edge 3 [1873, 6427] innerCandidates284_8 8300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_9

def innerTail284_7 : List Nat := 6301 :: innerTail284_8
def innerCandidates284_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_7_checked : candidates edge 4 1873 6301 innerTail284_8 best.bound = innerCandidates284_7 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_7 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_7 best = best := by
  rw [innerTail284_7, scan, if_neg (by decide), innerCandidates284_7_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_8
    (if 3 ≤ innerCandidates284_7.length then search edge 3 [1873, 6301] innerCandidates284_7 8174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_8

def innerTail284_6 : List Nat := 5569 :: innerTail284_7
def innerCandidates284_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_6_checked : candidates edge 4 1873 5569 innerTail284_7 best.bound = innerCandidates284_6 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_6 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_6 best = best := by
  rw [innerTail284_6, scan, if_neg (by decide), innerCandidates284_6_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_7
    (if 3 ≤ innerCandidates284_6.length then search edge 3 [1873, 5569] innerCandidates284_6 7442 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_7

def innerTail284_5 : List Nat := 5527 :: innerTail284_6
def innerCandidates284_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_5_checked : candidates edge 4 1873 5527 innerTail284_6 best.bound = innerCandidates284_5 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_5 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_5 best = best := by
  rw [innerTail284_5, scan, if_neg (by decide), innerCandidates284_5_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_6
    (if 3 ≤ innerCandidates284_5.length then search edge 3 [1873, 5527] innerCandidates284_5 7400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_6

def innerTail284_4 : List Nat := 3511 :: innerTail284_5
def innerCandidates284_4 : List Nat := [7963, 8011]
set_option maxRecDepth 4096 in
theorem innerCandidates284_4_checked : candidates edge 4 1873 3511 innerTail284_5 best.bound = innerCandidates284_4 := by
  unfold candidates
  have row : edge 3511 = SearchData.row3511.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_4 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_4 best = best := by
  rw [innerTail284_4, scan, if_neg (by decide), innerCandidates284_4_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_5
    (if 3 ≤ innerCandidates284_4.length then search edge 3 [1873, 3511] innerCandidates284_4 5384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_5

def innerTail284_3 : List Nat := 2719 :: innerTail284_4
def innerCandidates284_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_3_checked : candidates edge 4 1873 2719 innerTail284_4 best.bound = innerCandidates284_3 := by
  unfold candidates
  have row : edge 2719 = SearchData.row2719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_3 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_3 best = best := by
  rw [innerTail284_3, scan, if_neg (by decide), innerCandidates284_3_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_4
    (if 3 ≤ innerCandidates284_3.length then search edge 3 [1873, 2719] innerCandidates284_3 4592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_4

def innerTail284_2 : List Nat := 2437 :: innerTail284_3
def innerCandidates284_2 : List Nat := [6301]
set_option maxRecDepth 4096 in
theorem innerCandidates284_2_checked : candidates edge 4 1873 2437 innerTail284_3 best.bound = innerCandidates284_2 := by
  unfold candidates
  have row : edge 2437 = PrimePairCache.smallRow2437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_2 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_2 best = best := by
  rw [innerTail284_2, scan, if_neg (by decide), innerCandidates284_2_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_3
    (if 3 ≤ innerCandidates284_2.length then search edge 3 [1873, 2437] innerCandidates284_2 4310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_3

def innerTail284_1 : List Nat := 2269 :: innerTail284_2
def innerCandidates284_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_1_checked : candidates edge 4 1873 2269 innerTail284_2 best.bound = innerCandidates284_1 := by
  unfold candidates
  have row : edge 2269 = PrimePairCache.smallRow2269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_1 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_1 best = best := by
  rw [innerTail284_1, scan, if_neg (by decide), innerCandidates284_1_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_2
    (if 3 ≤ innerCandidates284_1.length then search edge 3 [1873, 2269] innerCandidates284_1 4142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_2

def innerTail284_0 : List Nat := 2017 :: innerTail284_1
def innerCandidates284_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates284_0_checked : candidates edge 4 1873 2017 innerTail284_1 best.bound = innerCandidates284_0 := by
  unfold candidates
  have row : edge 2017 = PrimePairCache.smallRow2017.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan284_0 : scan edge (search edge 3) 4 [1873] 1873 innerTail284_0 best = best := by
  rw [innerTail284_0, scan, if_neg (by decide), innerCandidates284_0_checked]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_1
    (if 3 ≤ innerCandidates284_0.length then search edge 3 [1873, 2017] innerCandidates284_0 3890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan284_1

theorem rootBranch284 : search edge 4 [1873] rootCandidates284 1873 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1873] 1873 innerTail284_0 best = best
  exact innerScan284_0

def rootCandidates283 : List Nat := [3137, 4139, 4349, 4517, 4649, 4691, 4733, 5051, 5231, 5441, 6323, 6959, 7001, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates283_checked : candidates edge 5 0 1871 tail284 best.bound = rootCandidates283 := by
  unfold candidates
  have row : edge 1871 = PrimePairCache.smallRow1871.contains := by funext q; rfl
  rw [row]
  decide

def innerTail283_14 : List Nat := []
theorem innerScan283_14 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_14 best = best := rfl

def innerTail283_13 : List Nat := 8171 :: innerTail283_14
def innerCandidates283_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_13_checked : candidates edge 4 1871 8171 innerTail283_14 best.bound = innerCandidates283_13 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_13 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_13 best = best := by
  rw [innerTail283_13, scan, if_neg (by decide), innerCandidates283_13_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_14
    (if 3 ≤ innerCandidates283_13.length then search edge 3 [1871, 8171] innerCandidates283_13 10042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_14

def innerTail283_12 : List Nat := 7001 :: innerTail283_13
def innerCandidates283_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_12_checked : candidates edge 4 1871 7001 innerTail283_13 best.bound = innerCandidates283_12 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_12 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_12 best = best := by
  rw [innerTail283_12, scan, if_neg (by decide), innerCandidates283_12_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_13
    (if 3 ≤ innerCandidates283_12.length then search edge 3 [1871, 7001] innerCandidates283_12 8872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_13

def innerTail283_11 : List Nat := 6959 :: innerTail283_12
def innerCandidates283_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_11_checked : candidates edge 4 1871 6959 innerTail283_12 best.bound = innerCandidates283_11 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_11 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_11 best = best := by
  rw [innerTail283_11, scan, if_neg (by decide), innerCandidates283_11_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_12
    (if 3 ≤ innerCandidates283_11.length then search edge 3 [1871, 6959] innerCandidates283_11 8830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_12

def innerTail283_10 : List Nat := 6323 :: innerTail283_11
def innerCandidates283_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_10_checked : candidates edge 4 1871 6323 innerTail283_11 best.bound = innerCandidates283_10 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_10 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_10 best = best := by
  rw [innerTail283_10, scan, if_neg (by decide), innerCandidates283_10_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_11
    (if 3 ≤ innerCandidates283_10.length then search edge 3 [1871, 6323] innerCandidates283_10 8194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_11

def innerTail283_9 : List Nat := 5441 :: innerTail283_10
def innerCandidates283_9 : List Nat := [6959]
set_option maxRecDepth 4096 in
theorem innerCandidates283_9_checked : candidates edge 4 1871 5441 innerTail283_10 best.bound = innerCandidates283_9 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_9 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_9 best = best := by
  rw [innerTail283_9, scan, if_neg (by decide), innerCandidates283_9_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_10
    (if 3 ≤ innerCandidates283_9.length then search edge 3 [1871, 5441] innerCandidates283_9 7312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_10

def innerTail283_8 : List Nat := 5231 :: innerTail283_9
def innerCandidates283_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_8_checked : candidates edge 4 1871 5231 innerTail283_9 best.bound = innerCandidates283_8 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_8 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_8 best = best := by
  rw [innerTail283_8, scan, if_neg (by decide), innerCandidates283_8_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_9
    (if 3 ≤ innerCandidates283_8.length then search edge 3 [1871, 5231] innerCandidates283_8 7102 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_9

def innerTail283_7 : List Nat := 5051 :: innerTail283_8
def innerCandidates283_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_7_checked : candidates edge 4 1871 5051 innerTail283_8 best.bound = innerCandidates283_7 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_7 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_7 best = best := by
  rw [innerTail283_7, scan, if_neg (by decide), innerCandidates283_7_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_8
    (if 3 ≤ innerCandidates283_7.length then search edge 3 [1871, 5051] innerCandidates283_7 6922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_8

def innerTail283_6 : List Nat := 4733 :: innerTail283_7
def innerCandidates283_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_6_checked : candidates edge 4 1871 4733 innerTail283_7 best.bound = innerCandidates283_6 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_6 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_6 best = best := by
  rw [innerTail283_6, scan, if_neg (by decide), innerCandidates283_6_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_7
    (if 3 ≤ innerCandidates283_6.length then search edge 3 [1871, 4733] innerCandidates283_6 6604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_7

def innerTail283_5 : List Nat := 4691 :: innerTail283_6
def innerCandidates283_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates283_5_checked : candidates edge 4 1871 4691 innerTail283_6 best.bound = innerCandidates283_5 := by
  unfold candidates
  have row : edge 4691 = SearchData.row4691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_5 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_5 best = best := by
  rw [innerTail283_5, scan, if_neg (by decide), innerCandidates283_5_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_6
    (if 3 ≤ innerCandidates283_5.length then search edge 3 [1871, 4691] innerCandidates283_5 6562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_6

def innerTail283_4 : List Nat := 4649 :: innerTail283_5
def innerCandidates283_4 : List Nat := [6323]
set_option maxRecDepth 4096 in
theorem innerCandidates283_4_checked : candidates edge 4 1871 4649 innerTail283_5 best.bound = innerCandidates283_4 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_4 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_4 best = best := by
  rw [innerTail283_4, scan, if_neg (by decide), innerCandidates283_4_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_5
    (if 3 ≤ innerCandidates283_4.length then search edge 3 [1871, 4649] innerCandidates283_4 6520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_5

def innerTail283_3 : List Nat := 4517 :: innerTail283_4
def innerCandidates283_3 : List Nat := [6959]
set_option maxRecDepth 4096 in
theorem innerCandidates283_3_checked : candidates edge 4 1871 4517 innerTail283_4 best.bound = innerCandidates283_3 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_3 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_3 best = best := by
  rw [innerTail283_3, scan, if_neg (by decide), innerCandidates283_3_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_4
    (if 3 ≤ innerCandidates283_3.length then search edge 3 [1871, 4517] innerCandidates283_3 6388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_4

def innerTail283_2 : List Nat := 4349 :: innerTail283_3
def innerCandidates283_2 : List Nat := [5051]
set_option maxRecDepth 4096 in
theorem innerCandidates283_2_checked : candidates edge 4 1871 4349 innerTail283_3 best.bound = innerCandidates283_2 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_2 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_2 best = best := by
  rw [innerTail283_2, scan, if_neg (by decide), innerCandidates283_2_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_3
    (if 3 ≤ innerCandidates283_2.length then search edge 3 [1871, 4349] innerCandidates283_2 6220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_3

def innerTail283_1 : List Nat := 4139 :: innerTail283_2
def innerCandidates283_1 : List Nat := [6959]
set_option maxRecDepth 4096 in
theorem innerCandidates283_1_checked : candidates edge 4 1871 4139 innerTail283_2 best.bound = innerCandidates283_1 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_1 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_1 best = best := by
  rw [innerTail283_1, scan, if_neg (by decide), innerCandidates283_1_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_2
    (if 3 ≤ innerCandidates283_1.length then search edge 3 [1871, 4139] innerCandidates283_1 6010 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_2

def innerTail283_0 : List Nat := 3137 :: innerTail283_1
def innerCandidates283_0 : List Nat := [4349]
set_option maxRecDepth 4096 in
theorem innerCandidates283_0_checked : candidates edge 4 1871 3137 innerTail283_1 best.bound = innerCandidates283_0 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan283_0 : scan edge (search edge 3) 4 [1871] 1871 innerTail283_0 best = best := by
  rw [innerTail283_0, scan, if_neg (by decide), innerCandidates283_0_checked]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_1
    (if 3 ≤ innerCandidates283_0.length then search edge 3 [1871, 3137] innerCandidates283_0 5008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan283_1

theorem rootBranch283 : search edge 4 [1871] rootCandidates283 1871 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1871] 1871 innerTail283_0 best = best
  exact innerScan283_0

def rootCandidates282 : List Nat := [2053, 2221, 3673, 4111, 5281, 5659, 5749, 6121, 6733, 7129, 7639, 7873, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates282_checked : candidates edge 5 0 1867 tail283 best.bound = rootCandidates282 := by
  unfold candidates
  have row : edge 1867 = PrimePairCache.smallRow1867.contains := by funext q; rfl
  rw [row]
  decide

def innerTail282_13 : List Nat := []
theorem innerScan282_13 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_13 best = best := rfl

def innerTail282_12 : List Nat := 8293 :: innerTail282_13
def innerCandidates282_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_12_checked : candidates edge 4 1867 8293 innerTail282_13 best.bound = innerCandidates282_12 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_12 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_12 best = best := by
  rw [innerTail282_12, scan, if_neg (by decide), innerCandidates282_12_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_13
    (if 3 ≤ innerCandidates282_12.length then search edge 3 [1867, 8293] innerCandidates282_12 10160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_13

def innerTail282_11 : List Nat := 7873 :: innerTail282_12
def innerCandidates282_11 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates282_11_checked : candidates edge 4 1867 7873 innerTail282_12 best.bound = innerCandidates282_11 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_11 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_11 best = best := by
  rw [innerTail282_11, scan, if_neg (by decide), innerCandidates282_11_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_12
    (if 3 ≤ innerCandidates282_11.length then search edge 3 [1867, 7873] innerCandidates282_11 9740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_12

def innerTail282_10 : List Nat := 7639 :: innerTail282_11
def innerCandidates282_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_10_checked : candidates edge 4 1867 7639 innerTail282_11 best.bound = innerCandidates282_10 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_10 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_10 best = best := by
  rw [innerTail282_10, scan, if_neg (by decide), innerCandidates282_10_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_11
    (if 3 ≤ innerCandidates282_10.length then search edge 3 [1867, 7639] innerCandidates282_10 9506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_11

def innerTail282_9 : List Nat := 7129 :: innerTail282_10
def innerCandidates282_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_9_checked : candidates edge 4 1867 7129 innerTail282_10 best.bound = innerCandidates282_9 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_9 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_9 best = best := by
  rw [innerTail282_9, scan, if_neg (by decide), innerCandidates282_9_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_10
    (if 3 ≤ innerCandidates282_9.length then search edge 3 [1867, 7129] innerCandidates282_9 8996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_10

def innerTail282_8 : List Nat := 6733 :: innerTail282_9
def innerCandidates282_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_8_checked : candidates edge 4 1867 6733 innerTail282_9 best.bound = innerCandidates282_8 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_8 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_8 best = best := by
  rw [innerTail282_8, scan, if_neg (by decide), innerCandidates282_8_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_9
    (if 3 ≤ innerCandidates282_8.length then search edge 3 [1867, 6733] innerCandidates282_8 8600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_9

def innerTail282_7 : List Nat := 6121 :: innerTail282_8
def innerCandidates282_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_7_checked : candidates edge 4 1867 6121 innerTail282_8 best.bound = innerCandidates282_7 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_7 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_7 best = best := by
  rw [innerTail282_7, scan, if_neg (by decide), innerCandidates282_7_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_8
    (if 3 ≤ innerCandidates282_7.length then search edge 3 [1867, 6121] innerCandidates282_7 7988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_8

def innerTail282_6 : List Nat := 5749 :: innerTail282_7
def innerCandidates282_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_6_checked : candidates edge 4 1867 5749 innerTail282_7 best.bound = innerCandidates282_6 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_6 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_6 best = best := by
  rw [innerTail282_6, scan, if_neg (by decide), innerCandidates282_6_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_7
    (if 3 ≤ innerCandidates282_6.length then search edge 3 [1867, 5749] innerCandidates282_6 7616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_7

def innerTail282_5 : List Nat := 5659 :: innerTail282_6
def innerCandidates282_5 : List Nat := [6121]
set_option maxRecDepth 4096 in
theorem innerCandidates282_5_checked : candidates edge 4 1867 5659 innerTail282_6 best.bound = innerCandidates282_5 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_5 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_5 best = best := by
  rw [innerTail282_5, scan, if_neg (by decide), innerCandidates282_5_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_6
    (if 3 ≤ innerCandidates282_5.length then search edge 3 [1867, 5659] innerCandidates282_5 7526 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_6

def innerTail282_4 : List Nat := 5281 :: innerTail282_5
def innerCandidates282_4 : List Nat := [6121, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates282_4_checked : candidates edge 4 1867 5281 innerTail282_5 best.bound = innerCandidates282_4 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_4 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_4 best = best := by
  rw [innerTail282_4, scan, if_neg (by decide), innerCandidates282_4_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_5
    (if 3 ≤ innerCandidates282_4.length then search edge 3 [1867, 5281] innerCandidates282_4 7148 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_5

def innerTail282_3 : List Nat := 4111 :: innerTail282_4
def innerCandidates282_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_3_checked : candidates edge 4 1867 4111 innerTail282_4 best.bound = innerCandidates282_3 := by
  unfold candidates
  have row : edge 4111 = SearchData.row4111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_3 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_3 best = best := by
  rw [innerTail282_3, scan, if_neg (by decide), innerCandidates282_3_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_4
    (if 3 ≤ innerCandidates282_3.length then search edge 3 [1867, 4111] innerCandidates282_3 5978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_4

def innerTail282_2 : List Nat := 3673 :: innerTail282_3
def innerCandidates282_2 : List Nat := [8293]
set_option maxRecDepth 4096 in
theorem innerCandidates282_2_checked : candidates edge 4 1867 3673 innerTail282_3 best.bound = innerCandidates282_2 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_2 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_2 best = best := by
  rw [innerTail282_2, scan, if_neg (by decide), innerCandidates282_2_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_3
    (if 3 ≤ innerCandidates282_2.length then search edge 3 [1867, 3673] innerCandidates282_2 5540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_3

def innerTail282_1 : List Nat := 2221 :: innerTail282_2
def innerCandidates282_1 : List Nat := [4111, 7639, 7873]
set_option maxRecDepth 4096 in
theorem innerCandidates282_1_checked : candidates edge 4 1867 2221 innerTail282_2 best.bound = innerCandidates282_1 := by
  unfold candidates
  have row : edge 2221 = PrimePairCache.smallRow2221.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild282_1 : search edge 3 [1867, 2221] innerCandidates282_1 4088 best = best := by decide

theorem innerScan282_1 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_1 best = best := by
  rw [innerTail282_1, scan, if_neg (by decide), innerCandidates282_1_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_2
    (if 3 ≤ innerCandidates282_1.length then search edge 3 [1867, 2221] innerCandidates282_1 4088 best else best) = best
  rw [if_pos (by decide), innerChild282_1]
  exact innerScan282_2

def innerTail282_0 : List Nat := 2053 :: innerTail282_1
def innerCandidates282_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates282_0_checked : candidates edge 4 1867 2053 innerTail282_1 best.bound = innerCandidates282_0 := by
  unfold candidates
  have row : edge 2053 = PrimePairCache.smallRow2053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan282_0 : scan edge (search edge 3) 4 [1867] 1867 innerTail282_0 best = best := by
  rw [innerTail282_0, scan, if_neg (by decide), innerCandidates282_0_checked]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_1
    (if 3 ≤ innerCandidates282_0.length then search edge 3 [1867, 2053] innerCandidates282_0 3920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan282_1

theorem rootBranch282 : search edge 4 [1867] rootCandidates282 1867 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1867] 1867 innerTail282_0 best = best
  exact innerScan282_0

end ProjectEulerSolutions.PrimePairUpper.Search
