import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates371 : List Nat := [2887, 3331, 4219, 4591, 4813, 4969, 5059, 5227, 6037, 6163, 6793, 7537]
set_option maxRecDepth 4096 in
theorem rootCandidates371_checked : candidates edge 5 0 2551 tail372 best.bound = rootCandidates371 := by
  unfold candidates
  have row : edge 2551 = PrimePairCache.smallRow2551.contains := by funext q; rfl
  rw [row]
  decide

def innerTail371_12 : List Nat := []
theorem innerScan371_12 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_12 best = best := rfl

def innerTail371_11 : List Nat := 7537 :: innerTail371_12
def innerCandidates371_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_11_checked : candidates edge 4 2551 7537 innerTail371_12 best.bound = innerCandidates371_11 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_11 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_11 best = best := by
  rw [innerTail371_11, scan, if_neg (by decide), innerCandidates371_11_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_12
    (if 3 ≤ innerCandidates371_11.length then search edge 3 [2551, 7537] innerCandidates371_11 10088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_12

def innerTail371_10 : List Nat := 6793 :: innerTail371_11
def innerCandidates371_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_10_checked : candidates edge 4 2551 6793 innerTail371_11 best.bound = innerCandidates371_10 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_10 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_10 best = best := by
  rw [innerTail371_10, scan, if_neg (by decide), innerCandidates371_10_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_11
    (if 3 ≤ innerCandidates371_10.length then search edge 3 [2551, 6793] innerCandidates371_10 9344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_11

def innerTail371_9 : List Nat := 6163 :: innerTail371_10
def innerCandidates371_9 : List Nat := [6793]
set_option maxRecDepth 4096 in
theorem innerCandidates371_9_checked : candidates edge 4 2551 6163 innerTail371_10 best.bound = innerCandidates371_9 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_9 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_9 best = best := by
  rw [innerTail371_9, scan, if_neg (by decide), innerCandidates371_9_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_10
    (if 3 ≤ innerCandidates371_9.length then search edge 3 [2551, 6163] innerCandidates371_9 8714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_10

def innerTail371_8 : List Nat := 6037 :: innerTail371_9
def innerCandidates371_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_8_checked : candidates edge 4 2551 6037 innerTail371_9 best.bound = innerCandidates371_8 := by
  unfold candidates
  have row : edge 6037 = SearchData.row6037.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_8 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_8 best = best := by
  rw [innerTail371_8, scan, if_neg (by decide), innerCandidates371_8_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_9
    (if 3 ≤ innerCandidates371_8.length then search edge 3 [2551, 6037] innerCandidates371_8 8588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_9

def innerTail371_7 : List Nat := 5227 :: innerTail371_8
def innerCandidates371_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_7_checked : candidates edge 4 2551 5227 innerTail371_8 best.bound = innerCandidates371_7 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_7 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_7 best = best := by
  rw [innerTail371_7, scan, if_neg (by decide), innerCandidates371_7_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_8
    (if 3 ≤ innerCandidates371_7.length then search edge 3 [2551, 5227] innerCandidates371_7 7778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_8

def innerTail371_6 : List Nat := 5059 :: innerTail371_7
def innerCandidates371_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_6_checked : candidates edge 4 2551 5059 innerTail371_7 best.bound = innerCandidates371_6 := by
  unfold candidates
  have row : edge 5059 = SearchData.row5059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_6 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_6 best = best := by
  rw [innerTail371_6, scan, if_neg (by decide), innerCandidates371_6_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_7
    (if 3 ≤ innerCandidates371_6.length then search edge 3 [2551, 5059] innerCandidates371_6 7610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_7

def innerTail371_5 : List Nat := 4969 :: innerTail371_6
def innerCandidates371_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_5_checked : candidates edge 4 2551 4969 innerTail371_6 best.bound = innerCandidates371_5 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_5 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_5 best = best := by
  rw [innerTail371_5, scan, if_neg (by decide), innerCandidates371_5_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_6
    (if 3 ≤ innerCandidates371_5.length then search edge 3 [2551, 4969] innerCandidates371_5 7520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_6

def innerTail371_4 : List Nat := 4813 :: innerTail371_5
def innerCandidates371_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_4_checked : candidates edge 4 2551 4813 innerTail371_5 best.bound = innerCandidates371_4 := by
  unfold candidates
  have row : edge 4813 = SearchData.row4813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_4 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_4 best = best := by
  rw [innerTail371_4, scan, if_neg (by decide), innerCandidates371_4_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_5
    (if 3 ≤ innerCandidates371_4.length then search edge 3 [2551, 4813] innerCandidates371_4 7364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_5

def innerTail371_3 : List Nat := 4591 :: innerTail371_4
def innerCandidates371_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_3_checked : candidates edge 4 2551 4591 innerTail371_4 best.bound = innerCandidates371_3 := by
  unfold candidates
  have row : edge 4591 = SearchData.row4591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_3 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_3 best = best := by
  rw [innerTail371_3, scan, if_neg (by decide), innerCandidates371_3_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_4
    (if 3 ≤ innerCandidates371_3.length then search edge 3 [2551, 4591] innerCandidates371_3 7142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_4

def innerTail371_2 : List Nat := 4219 :: innerTail371_3
def innerCandidates371_2 : List Nat := [5227, 6793]
set_option maxRecDepth 4096 in
theorem innerCandidates371_2_checked : candidates edge 4 2551 4219 innerTail371_3 best.bound = innerCandidates371_2 := by
  unfold candidates
  have row : edge 4219 = SearchData.row4219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_2 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_2 best = best := by
  rw [innerTail371_2, scan, if_neg (by decide), innerCandidates371_2_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_3
    (if 3 ≤ innerCandidates371_2.length then search edge 3 [2551, 4219] innerCandidates371_2 6770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_3

def innerTail371_1 : List Nat := 3331 :: innerTail371_2
def innerCandidates371_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_1_checked : candidates edge 4 2551 3331 innerTail371_2 best.bound = innerCandidates371_1 := by
  unfold candidates
  have row : edge 3331 = SearchData.row3331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_1 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_1 best = best := by
  rw [innerTail371_1, scan, if_neg (by decide), innerCandidates371_1_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_2
    (if 3 ≤ innerCandidates371_1.length then search edge 3 [2551, 3331] innerCandidates371_1 5882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_2

def innerTail371_0 : List Nat := 2887 :: innerTail371_1
def innerCandidates371_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates371_0_checked : candidates edge 4 2551 2887 innerTail371_1 best.bound = innerCandidates371_0 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan371_0 : scan edge (search edge 3) 4 [2551] 2551 innerTail371_0 best = best := by
  rw [innerTail371_0, scan, if_neg (by decide), innerCandidates371_0_checked]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_1
    (if 3 ≤ innerCandidates371_0.length then search edge 3 [2551, 2887] innerCandidates371_0 5438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan371_1

theorem rootBranch371 : search edge 4 [2551] rootCandidates371 2551 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2551] 2551 innerTail371_0 best = best
  exact innerScan371_0

def rootCandidates370 : List Nat := [2729, 3089, 3329, 3593, 3947, 4019, 4241, 4349, 5471, 5501, 6323, 6761, 6833, 7211, 7673, 7793, 7937, 7949, 8009, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates370_checked : candidates edge 5 0 2549 tail371 best.bound = rootCandidates370 := by
  unfold candidates
  have row : edge 2549 = PrimePairCache.smallRow2549.contains := by funext q; rfl
  rw [row]
  decide

def innerTail370_20 : List Nat := []
theorem innerScan370_20 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_20 best = best := rfl

def innerTail370_19 : List Nat := 8093 :: innerTail370_20
def innerCandidates370_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_19_checked : candidates edge 4 2549 8093 innerTail370_20 best.bound = innerCandidates370_19 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_19 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_19 best = best := by
  rw [innerTail370_19, scan, if_neg (by decide), innerCandidates370_19_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_20
    (if 3 ≤ innerCandidates370_19.length then search edge 3 [2549, 8093] innerCandidates370_19 10642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_20

def innerTail370_18 : List Nat := 8009 :: innerTail370_19
def innerCandidates370_18 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates370_18_checked : candidates edge 4 2549 8009 innerTail370_19 best.bound = innerCandidates370_18 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_18 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_18 best = best := by
  rw [innerTail370_18, scan, if_neg (by decide), innerCandidates370_18_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_19
    (if 3 ≤ innerCandidates370_18.length then search edge 3 [2549, 8009] innerCandidates370_18 10558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_19

def innerTail370_17 : List Nat := 7949 :: innerTail370_18
def innerCandidates370_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_17_checked : candidates edge 4 2549 7949 innerTail370_18 best.bound = innerCandidates370_17 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_17 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_17 best = best := by
  rw [innerTail370_17, scan, if_neg (by decide), innerCandidates370_17_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_18
    (if 3 ≤ innerCandidates370_17.length then search edge 3 [2549, 7949] innerCandidates370_17 10498 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_18

def innerTail370_16 : List Nat := 7937 :: innerTail370_17
def innerCandidates370_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_16_checked : candidates edge 4 2549 7937 innerTail370_17 best.bound = innerCandidates370_16 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_16 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_16 best = best := by
  rw [innerTail370_16, scan, if_neg (by decide), innerCandidates370_16_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_17
    (if 3 ≤ innerCandidates370_16.length then search edge 3 [2549, 7937] innerCandidates370_16 10486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_17

def innerTail370_15 : List Nat := 7793 :: innerTail370_16
def innerCandidates370_15 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates370_15_checked : candidates edge 4 2549 7793 innerTail370_16 best.bound = innerCandidates370_15 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_15 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_15 best = best := by
  rw [innerTail370_15, scan, if_neg (by decide), innerCandidates370_15_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_16
    (if 3 ≤ innerCandidates370_15.length then search edge 3 [2549, 7793] innerCandidates370_15 10342 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_16

def innerTail370_14 : List Nat := 7673 :: innerTail370_15
def innerCandidates370_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_14_checked : candidates edge 4 2549 7673 innerTail370_15 best.bound = innerCandidates370_14 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_14 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_14 best = best := by
  rw [innerTail370_14, scan, if_neg (by decide), innerCandidates370_14_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_15
    (if 3 ≤ innerCandidates370_14.length then search edge 3 [2549, 7673] innerCandidates370_14 10222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_15

def innerTail370_13 : List Nat := 7211 :: innerTail370_14
def innerCandidates370_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_13_checked : candidates edge 4 2549 7211 innerTail370_14 best.bound = innerCandidates370_13 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_13 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_13 best = best := by
  rw [innerTail370_13, scan, if_neg (by decide), innerCandidates370_13_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_14
    (if 3 ≤ innerCandidates370_13.length then search edge 3 [2549, 7211] innerCandidates370_13 9760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_14

def innerTail370_12 : List Nat := 6833 :: innerTail370_13
def innerCandidates370_12 : List Nat := [7673]
set_option maxRecDepth 4096 in
theorem innerCandidates370_12_checked : candidates edge 4 2549 6833 innerTail370_13 best.bound = innerCandidates370_12 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_12 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_12 best = best := by
  rw [innerTail370_12, scan, if_neg (by decide), innerCandidates370_12_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_13
    (if 3 ≤ innerCandidates370_12.length then search edge 3 [2549, 6833] innerCandidates370_12 9382 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_13

def innerTail370_11 : List Nat := 6761 :: innerTail370_12
def innerCandidates370_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_11_checked : candidates edge 4 2549 6761 innerTail370_12 best.bound = innerCandidates370_11 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_11 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_11 best = best := by
  rw [innerTail370_11, scan, if_neg (by decide), innerCandidates370_11_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_12
    (if 3 ≤ innerCandidates370_11.length then search edge 3 [2549, 6761] innerCandidates370_11 9310 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_12

def innerTail370_10 : List Nat := 6323 :: innerTail370_11
def innerCandidates370_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_10_checked : candidates edge 4 2549 6323 innerTail370_11 best.bound = innerCandidates370_10 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_10 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_10 best = best := by
  rw [innerTail370_10, scan, if_neg (by decide), innerCandidates370_10_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_11
    (if 3 ≤ innerCandidates370_10.length then search edge 3 [2549, 6323] innerCandidates370_10 8872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_11

def innerTail370_9 : List Nat := 5501 :: innerTail370_10
def innerCandidates370_9 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates370_9_checked : candidates edge 4 2549 5501 innerTail370_10 best.bound = innerCandidates370_9 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_9 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_9 best = best := by
  rw [innerTail370_9, scan, if_neg (by decide), innerCandidates370_9_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_10
    (if 3 ≤ innerCandidates370_9.length then search edge 3 [2549, 5501] innerCandidates370_9 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_10

def innerTail370_8 : List Nat := 5471 :: innerTail370_9
def innerCandidates370_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_8_checked : candidates edge 4 2549 5471 innerTail370_9 best.bound = innerCandidates370_8 := by
  unfold candidates
  have row : edge 5471 = SearchData.row5471.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_8 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_8 best = best := by
  rw [innerTail370_8, scan, if_neg (by decide), innerCandidates370_8_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_9
    (if 3 ≤ innerCandidates370_8.length then search edge 3 [2549, 5471] innerCandidates370_8 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_9

def innerTail370_7 : List Nat := 4349 :: innerTail370_8
def innerCandidates370_7 : List Nat := [7793, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates370_7_checked : candidates edge 4 2549 4349 innerTail370_8 best.bound = innerCandidates370_7 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_7 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_7 best = best := by
  rw [innerTail370_7, scan, if_neg (by decide), innerCandidates370_7_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_8
    (if 3 ≤ innerCandidates370_7.length then search edge 3 [2549, 4349] innerCandidates370_7 6898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_8

def innerTail370_6 : List Nat := 4241 :: innerTail370_7
def innerCandidates370_6 : List Nat := [7673, 8093]
set_option maxRecDepth 4096 in
theorem innerCandidates370_6_checked : candidates edge 4 2549 4241 innerTail370_7 best.bound = innerCandidates370_6 := by
  unfold candidates
  have row : edge 4241 = SearchData.row4241.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_6 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_6 best = best := by
  rw [innerTail370_6, scan, if_neg (by decide), innerCandidates370_6_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_7
    (if 3 ≤ innerCandidates370_6.length then search edge 3 [2549, 4241] innerCandidates370_6 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_7

def innerTail370_5 : List Nat := 4019 :: innerTail370_6
def innerCandidates370_5 : List Nat := [5471, 7211]
set_option maxRecDepth 4096 in
theorem innerCandidates370_5_checked : candidates edge 4 2549 4019 innerTail370_6 best.bound = innerCandidates370_5 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_5 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_5 best = best := by
  rw [innerTail370_5, scan, if_neg (by decide), innerCandidates370_5_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_6
    (if 3 ≤ innerCandidates370_5.length then search edge 3 [2549, 4019] innerCandidates370_5 6568 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_6

def innerTail370_4 : List Nat := 3947 :: innerTail370_5
def innerCandidates370_4 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates370_4_checked : candidates edge 4 2549 3947 innerTail370_5 best.bound = innerCandidates370_4 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_4 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_4 best = best := by
  rw [innerTail370_4, scan, if_neg (by decide), innerCandidates370_4_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_5
    (if 3 ≤ innerCandidates370_4.length then search edge 3 [2549, 3947] innerCandidates370_4 6496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_5

def innerTail370_3 : List Nat := 3593 :: innerTail370_4
def innerCandidates370_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_3_checked : candidates edge 4 2549 3593 innerTail370_4 best.bound = innerCandidates370_3 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_3 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_3 best = best := by
  rw [innerTail370_3, scan, if_neg (by decide), innerCandidates370_3_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_4
    (if 3 ≤ innerCandidates370_3.length then search edge 3 [2549, 3593] innerCandidates370_3 6142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_4

def innerTail370_2 : List Nat := 3329 :: innerTail370_3
def innerCandidates370_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_2_checked : candidates edge 4 2549 3329 innerTail370_3 best.bound = innerCandidates370_2 := by
  unfold candidates
  have row : edge 3329 = SearchData.row3329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_2 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_2 best = best := by
  rw [innerTail370_2, scan, if_neg (by decide), innerCandidates370_2_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_3
    (if 3 ≤ innerCandidates370_2.length then search edge 3 [2549, 3329] innerCandidates370_2 5878 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_3

def innerTail370_1 : List Nat := 3089 :: innerTail370_2
def innerCandidates370_1 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates370_1_checked : candidates edge 4 2549 3089 innerTail370_2 best.bound = innerCandidates370_1 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_1 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_1 best = best := by
  rw [innerTail370_1, scan, if_neg (by decide), innerCandidates370_1_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_2
    (if 3 ≤ innerCandidates370_1.length then search edge 3 [2549, 3089] innerCandidates370_1 5638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_2

def innerTail370_0 : List Nat := 2729 :: innerTail370_1
def innerCandidates370_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates370_0_checked : candidates edge 4 2549 2729 innerTail370_1 best.bound = innerCandidates370_0 := by
  unfold candidates
  have row : edge 2729 = SearchData.row2729.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan370_0 : scan edge (search edge 3) 4 [2549] 2549 innerTail370_0 best = best := by
  rw [innerTail370_0, scan, if_neg (by decide), innerCandidates370_0_checked]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_1
    (if 3 ≤ innerCandidates370_0.length then search edge 3 [2549, 2729] innerCandidates370_0 5278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan370_1

theorem rootBranch370 : search edge 4 [2549] rootCandidates370 2549 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2549] 2549 innerTail370_0 best = best
  exact innerScan370_0

def rootCandidates369 : List Nat := [2609, 3371, 3539, 3851, 4337, 4349, 4523, 5051, 5147, 5153, 5237, 5441, 5483, 5591, 6011, 6791, 6911, 7331, 7541, 7649]
set_option maxRecDepth 4096 in
theorem rootCandidates369_checked : candidates edge 5 0 2543 tail370 best.bound = rootCandidates369 := by
  unfold candidates
  have row : edge 2543 = PrimePairCache.smallRow2543.contains := by funext q; rfl
  rw [row]
  decide

def innerTail369_20 : List Nat := []
theorem innerScan369_20 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_20 best = best := rfl

def innerTail369_19 : List Nat := 7649 :: innerTail369_20
def innerCandidates369_19 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_19_checked : candidates edge 4 2543 7649 innerTail369_20 best.bound = innerCandidates369_19 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_19 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_19 best = best := by
  rw [innerTail369_19, scan, if_neg (by decide), innerCandidates369_19_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_20
    (if 3 ≤ innerCandidates369_19.length then search edge 3 [2543, 7649] innerCandidates369_19 10192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_20

def innerTail369_18 : List Nat := 7541 :: innerTail369_19
def innerCandidates369_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_18_checked : candidates edge 4 2543 7541 innerTail369_19 best.bound = innerCandidates369_18 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_18 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_18 best = best := by
  rw [innerTail369_18, scan, if_neg (by decide), innerCandidates369_18_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_19
    (if 3 ≤ innerCandidates369_18.length then search edge 3 [2543, 7541] innerCandidates369_18 10084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_19

def innerTail369_17 : List Nat := 7331 :: innerTail369_18
def innerCandidates369_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_17_checked : candidates edge 4 2543 7331 innerTail369_18 best.bound = innerCandidates369_17 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_17 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_17 best = best := by
  rw [innerTail369_17, scan, if_neg (by decide), innerCandidates369_17_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_18
    (if 3 ≤ innerCandidates369_17.length then search edge 3 [2543, 7331] innerCandidates369_17 9874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_18

def innerTail369_16 : List Nat := 6911 :: innerTail369_17
def innerCandidates369_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_16_checked : candidates edge 4 2543 6911 innerTail369_17 best.bound = innerCandidates369_16 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_16 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_16 best = best := by
  rw [innerTail369_16, scan, if_neg (by decide), innerCandidates369_16_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_17
    (if 3 ≤ innerCandidates369_16.length then search edge 3 [2543, 6911] innerCandidates369_16 9454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_17

def innerTail369_15 : List Nat := 6791 :: innerTail369_16
def innerCandidates369_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_15_checked : candidates edge 4 2543 6791 innerTail369_16 best.bound = innerCandidates369_15 := by
  unfold candidates
  have row : edge 6791 = SearchData.row6791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_15 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_15 best = best := by
  rw [innerTail369_15, scan, if_neg (by decide), innerCandidates369_15_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_16
    (if 3 ≤ innerCandidates369_15.length then search edge 3 [2543, 6791] innerCandidates369_15 9334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_16

def innerTail369_14 : List Nat := 6011 :: innerTail369_15
def innerCandidates369_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_14_checked : candidates edge 4 2543 6011 innerTail369_15 best.bound = innerCandidates369_14 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_14 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_14 best = best := by
  rw [innerTail369_14, scan, if_neg (by decide), innerCandidates369_14_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_15
    (if 3 ≤ innerCandidates369_14.length then search edge 3 [2543, 6011] innerCandidates369_14 8554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_15

def innerTail369_13 : List Nat := 5591 :: innerTail369_14
def innerCandidates369_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_13_checked : candidates edge 4 2543 5591 innerTail369_14 best.bound = innerCandidates369_13 := by
  unfold candidates
  have row : edge 5591 = SearchData.row5591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_13 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_13 best = best := by
  rw [innerTail369_13, scan, if_neg (by decide), innerCandidates369_13_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_14
    (if 3 ≤ innerCandidates369_13.length then search edge 3 [2543, 5591] innerCandidates369_13 8134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_14

def innerTail369_12 : List Nat := 5483 :: innerTail369_13
def innerCandidates369_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_12_checked : candidates edge 4 2543 5483 innerTail369_13 best.bound = innerCandidates369_12 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_12 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_12 best = best := by
  rw [innerTail369_12, scan, if_neg (by decide), innerCandidates369_12_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_13
    (if 3 ≤ innerCandidates369_12.length then search edge 3 [2543, 5483] innerCandidates369_12 8026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_13

def innerTail369_11 : List Nat := 5441 :: innerTail369_12
def innerCandidates369_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_11_checked : candidates edge 4 2543 5441 innerTail369_12 best.bound = innerCandidates369_11 := by
  unfold candidates
  have row : edge 5441 = SearchData.row5441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_11 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_11 best = best := by
  rw [innerTail369_11, scan, if_neg (by decide), innerCandidates369_11_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_12
    (if 3 ≤ innerCandidates369_11.length then search edge 3 [2543, 5441] innerCandidates369_11 7984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_12

def innerTail369_10 : List Nat := 5237 :: innerTail369_11
def innerCandidates369_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_10_checked : candidates edge 4 2543 5237 innerTail369_11 best.bound = innerCandidates369_10 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_10 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_10 best = best := by
  rw [innerTail369_10, scan, if_neg (by decide), innerCandidates369_10_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_11
    (if 3 ≤ innerCandidates369_10.length then search edge 3 [2543, 5237] innerCandidates369_10 7780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_11

def innerTail369_9 : List Nat := 5153 :: innerTail369_10
def innerCandidates369_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_9_checked : candidates edge 4 2543 5153 innerTail369_10 best.bound = innerCandidates369_9 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_9 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_9 best = best := by
  rw [innerTail369_9, scan, if_neg (by decide), innerCandidates369_9_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_10
    (if 3 ≤ innerCandidates369_9.length then search edge 3 [2543, 5153] innerCandidates369_9 7696 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_10

def innerTail369_8 : List Nat := 5147 :: innerTail369_9
def innerCandidates369_8 : List Nat := [7331]
set_option maxRecDepth 4096 in
theorem innerCandidates369_8_checked : candidates edge 4 2543 5147 innerTail369_9 best.bound = innerCandidates369_8 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_8 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_8 best = best := by
  rw [innerTail369_8, scan, if_neg (by decide), innerCandidates369_8_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_9
    (if 3 ≤ innerCandidates369_8.length then search edge 3 [2543, 5147] innerCandidates369_8 7690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_9

def innerTail369_7 : List Nat := 5051 :: innerTail369_8
def innerCandidates369_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_7_checked : candidates edge 4 2543 5051 innerTail369_8 best.bound = innerCandidates369_7 := by
  unfold candidates
  have row : edge 5051 = SearchData.row5051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_7 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_7 best = best := by
  rw [innerTail369_7, scan, if_neg (by decide), innerCandidates369_7_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_8
    (if 3 ≤ innerCandidates369_7.length then search edge 3 [2543, 5051] innerCandidates369_7 7594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_8

def innerTail369_6 : List Nat := 4523 :: innerTail369_7
def innerCandidates369_6 : List Nat := [5153, 6911]
set_option maxRecDepth 4096 in
theorem innerCandidates369_6_checked : candidates edge 4 2543 4523 innerTail369_7 best.bound = innerCandidates369_6 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_6 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_6 best = best := by
  rw [innerTail369_6, scan, if_neg (by decide), innerCandidates369_6_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_7
    (if 3 ≤ innerCandidates369_6.length then search edge 3 [2543, 4523] innerCandidates369_6 7066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_7

def innerTail369_5 : List Nat := 4349 :: innerTail369_6
def innerCandidates369_5 : List Nat := [5051, 5483]
set_option maxRecDepth 4096 in
theorem innerCandidates369_5_checked : candidates edge 4 2543 4349 innerTail369_6 best.bound = innerCandidates369_5 := by
  unfold candidates
  have row : edge 4349 = SearchData.row4349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_5 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_5 best = best := by
  rw [innerTail369_5, scan, if_neg (by decide), innerCandidates369_5_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_6
    (if 3 ≤ innerCandidates369_5.length then search edge 3 [2543, 4349] innerCandidates369_5 6892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_6

def innerTail369_4 : List Nat := 4337 :: innerTail369_5
def innerCandidates369_4 : List Nat := [4523, 5237]
set_option maxRecDepth 4096 in
theorem innerCandidates369_4_checked : candidates edge 4 2543 4337 innerTail369_5 best.bound = innerCandidates369_4 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_4 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_4 best = best := by
  rw [innerTail369_4, scan, if_neg (by decide), innerCandidates369_4_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_5
    (if 3 ≤ innerCandidates369_4.length then search edge 3 [2543, 4337] innerCandidates369_4 6880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_5

def innerTail369_3 : List Nat := 3851 :: innerTail369_4
def innerCandidates369_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_3_checked : candidates edge 4 2543 3851 innerTail369_4 best.bound = innerCandidates369_3 := by
  unfold candidates
  have row : edge 3851 = SearchData.row3851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_3 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_3 best = best := by
  rw [innerTail369_3, scan, if_neg (by decide), innerCandidates369_3_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_4
    (if 3 ≤ innerCandidates369_3.length then search edge 3 [2543, 3851] innerCandidates369_3 6394 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_4

def innerTail369_2 : List Nat := 3539 :: innerTail369_3
def innerCandidates369_2 : List Nat := [5441]
set_option maxRecDepth 4096 in
theorem innerCandidates369_2_checked : candidates edge 4 2543 3539 innerTail369_3 best.bound = innerCandidates369_2 := by
  unfold candidates
  have row : edge 3539 = SearchData.row3539.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_2 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_2 best = best := by
  rw [innerTail369_2, scan, if_neg (by decide), innerCandidates369_2_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_3
    (if 3 ≤ innerCandidates369_2.length then search edge 3 [2543, 3539] innerCandidates369_2 6082 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_3

def innerTail369_1 : List Nat := 3371 :: innerTail369_2
def innerCandidates369_1 : List Nat := [5153]
set_option maxRecDepth 4096 in
theorem innerCandidates369_1_checked : candidates edge 4 2543 3371 innerTail369_2 best.bound = innerCandidates369_1 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_1 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_1 best = best := by
  rw [innerTail369_1, scan, if_neg (by decide), innerCandidates369_1_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_2
    (if 3 ≤ innerCandidates369_1.length then search edge 3 [2543, 3371] innerCandidates369_1 5914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_2

def innerTail369_0 : List Nat := 2609 :: innerTail369_1
def innerCandidates369_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates369_0_checked : candidates edge 4 2543 2609 innerTail369_1 best.bound = innerCandidates369_0 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan369_0 : scan edge (search edge 3) 4 [2543] 2543 innerTail369_0 best = best := by
  rw [innerTail369_0, scan, if_neg (by decide), innerCandidates369_0_checked]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_1
    (if 3 ≤ innerCandidates369_0.length then search edge 3 [2543, 2609] innerCandidates369_0 5152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan369_1

theorem rootBranch369 : search edge 4 [2543] rootCandidates369 2543 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2543] 2543 innerTail369_0 best = best
  exact innerScan369_0

end ProjectEulerSolutions.PrimePairUpper.Search
