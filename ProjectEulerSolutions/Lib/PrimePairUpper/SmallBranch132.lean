import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates278 : List Nat := [2063, 2069, 2111, 2861, 3023, 3461, 3947, 4001, 4493, 6131, 6353, 6719, 6947, 6983, 7823, 8069]
set_option maxRecDepth 4096 in
theorem rootCandidates278_checked : candidates edge 5 0 1823 tail279 best.bound = rootCandidates278 := by
  unfold candidates
  have row : edge 1823 = PrimePairCache.smallRow1823.contains := by funext q; rfl
  rw [row]
  decide

def innerTail278_16 : List Nat := []
theorem innerScan278_16 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_16 best = best := rfl

def innerTail278_15 : List Nat := 8069 :: innerTail278_16
def innerCandidates278_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_15_checked : candidates edge 4 1823 8069 innerTail278_16 best.bound = innerCandidates278_15 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_15 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_15 best = best := by
  rw [innerTail278_15, scan, if_neg (by decide), innerCandidates278_15_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_16
    (if 3 ≤ innerCandidates278_15.length then search edge 3 [1823, 8069] innerCandidates278_15 9892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_16

def innerTail278_14 : List Nat := 7823 :: innerTail278_15
def innerCandidates278_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_14_checked : candidates edge 4 1823 7823 innerTail278_15 best.bound = innerCandidates278_14 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_14 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_14 best = best := by
  rw [innerTail278_14, scan, if_neg (by decide), innerCandidates278_14_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_15
    (if 3 ≤ innerCandidates278_14.length then search edge 3 [1823, 7823] innerCandidates278_14 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_15

def innerTail278_13 : List Nat := 6983 :: innerTail278_14
def innerCandidates278_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_13_checked : candidates edge 4 1823 6983 innerTail278_14 best.bound = innerCandidates278_13 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_13 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_13 best = best := by
  rw [innerTail278_13, scan, if_neg (by decide), innerCandidates278_13_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_14
    (if 3 ≤ innerCandidates278_13.length then search edge 3 [1823, 6983] innerCandidates278_13 8806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_14

def innerTail278_12 : List Nat := 6947 :: innerTail278_13
def innerCandidates278_12 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates278_12_checked : candidates edge 4 1823 6947 innerTail278_13 best.bound = innerCandidates278_12 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_12 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_12 best = best := by
  rw [innerTail278_12, scan, if_neg (by decide), innerCandidates278_12_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_13
    (if 3 ≤ innerCandidates278_12.length then search edge 3 [1823, 6947] innerCandidates278_12 8770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_13

def innerTail278_11 : List Nat := 6719 :: innerTail278_12
def innerCandidates278_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_11_checked : candidates edge 4 1823 6719 innerTail278_12 best.bound = innerCandidates278_11 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_11 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_11 best = best := by
  rw [innerTail278_11, scan, if_neg (by decide), innerCandidates278_11_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_12
    (if 3 ≤ innerCandidates278_11.length then search edge 3 [1823, 6719] innerCandidates278_11 8542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_12

def innerTail278_10 : List Nat := 6353 :: innerTail278_11
def innerCandidates278_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_10_checked : candidates edge 4 1823 6353 innerTail278_11 best.bound = innerCandidates278_10 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_10 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_10 best = best := by
  rw [innerTail278_10, scan, if_neg (by decide), innerCandidates278_10_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_11
    (if 3 ≤ innerCandidates278_10.length then search edge 3 [1823, 6353] innerCandidates278_10 8176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_11

def innerTail278_9 : List Nat := 6131 :: innerTail278_10
def innerCandidates278_9 : List Nat := [6947]
set_option maxRecDepth 4096 in
theorem innerCandidates278_9_checked : candidates edge 4 1823 6131 innerTail278_10 best.bound = innerCandidates278_9 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_9 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_9 best = best := by
  rw [innerTail278_9, scan, if_neg (by decide), innerCandidates278_9_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_10
    (if 3 ≤ innerCandidates278_9.length then search edge 3 [1823, 6131] innerCandidates278_9 7954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_10

def innerTail278_8 : List Nat := 4493 :: innerTail278_9
def innerCandidates278_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_8_checked : candidates edge 4 1823 4493 innerTail278_9 best.bound = innerCandidates278_8 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_8 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_8 best = best := by
  rw [innerTail278_8, scan, if_neg (by decide), innerCandidates278_8_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_9
    (if 3 ≤ innerCandidates278_8.length then search edge 3 [1823, 4493] innerCandidates278_8 6316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_9

def innerTail278_7 : List Nat := 4001 :: innerTail278_8
def innerCandidates278_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_7_checked : candidates edge 4 1823 4001 innerTail278_8 best.bound = innerCandidates278_7 := by
  unfold candidates
  have row : edge 4001 = SearchData.row4001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_7 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_7 best = best := by
  rw [innerTail278_7, scan, if_neg (by decide), innerCandidates278_7_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_8
    (if 3 ≤ innerCandidates278_7.length then search edge 3 [1823, 4001] innerCandidates278_7 5824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_8

def innerTail278_6 : List Nat := 3947 :: innerTail278_7
def innerCandidates278_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_6_checked : candidates edge 4 1823 3947 innerTail278_7 best.bound = innerCandidates278_6 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_6 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_6 best = best := by
  rw [innerTail278_6, scan, if_neg (by decide), innerCandidates278_6_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_7
    (if 3 ≤ innerCandidates278_6.length then search edge 3 [1823, 3947] innerCandidates278_6 5770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_7

def innerTail278_5 : List Nat := 3461 :: innerTail278_6
def innerCandidates278_5 : List Nat := [7823]
set_option maxRecDepth 4096 in
theorem innerCandidates278_5_checked : candidates edge 4 1823 3461 innerTail278_6 best.bound = innerCandidates278_5 := by
  unfold candidates
  have row : edge 3461 = SearchData.row3461.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_5 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_5 best = best := by
  rw [innerTail278_5, scan, if_neg (by decide), innerCandidates278_5_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_6
    (if 3 ≤ innerCandidates278_5.length then search edge 3 [1823, 3461] innerCandidates278_5 5284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_6

def innerTail278_4 : List Nat := 3023 :: innerTail278_5
def innerCandidates278_4 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates278_4_checked : candidates edge 4 1823 3023 innerTail278_5 best.bound = innerCandidates278_4 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_4 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_4 best = best := by
  rw [innerTail278_4, scan, if_neg (by decide), innerCandidates278_4_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_5
    (if 3 ≤ innerCandidates278_4.length then search edge 3 [1823, 3023] innerCandidates278_4 4846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_5

def innerTail278_3 : List Nat := 2861 :: innerTail278_4
def innerCandidates278_3 : List Nat := [6719]
set_option maxRecDepth 4096 in
theorem innerCandidates278_3_checked : candidates edge 4 1823 2861 innerTail278_4 best.bound = innerCandidates278_3 := by
  unfold candidates
  have row : edge 2861 = SearchData.row2861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_3 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_3 best = best := by
  rw [innerTail278_3, scan, if_neg (by decide), innerCandidates278_3_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_4
    (if 3 ≤ innerCandidates278_3.length then search edge 3 [1823, 2861] innerCandidates278_3 4684 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_4

def innerTail278_2 : List Nat := 2111 :: innerTail278_3
def innerCandidates278_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_2_checked : candidates edge 4 1823 2111 innerTail278_3 best.bound = innerCandidates278_2 := by
  unfold candidates
  have row : edge 2111 = PrimePairCache.smallRow2111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_2 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_2 best = best := by
  rw [innerTail278_2, scan, if_neg (by decide), innerCandidates278_2_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_3
    (if 3 ≤ innerCandidates278_2.length then search edge 3 [1823, 2111] innerCandidates278_2 3934 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_3

def innerTail278_1 : List Nat := 2069 :: innerTail278_2
def innerCandidates278_1 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates278_1_checked : candidates edge 4 1823 2069 innerTail278_2 best.bound = innerCandidates278_1 := by
  unfold candidates
  have row : edge 2069 = PrimePairCache.smallRow2069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_1 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_1 best = best := by
  rw [innerTail278_1, scan, if_neg (by decide), innerCandidates278_1_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_2
    (if 3 ≤ innerCandidates278_1.length then search edge 3 [1823, 2069] innerCandidates278_1 3892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_2

def innerTail278_0 : List Nat := 2063 :: innerTail278_1
def innerCandidates278_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates278_0_checked : candidates edge 4 1823 2063 innerTail278_1 best.bound = innerCandidates278_0 := by
  unfold candidates
  have row : edge 2063 = PrimePairCache.smallRow2063.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan278_0 : scan edge (search edge 3) 4 [1823] 1823 innerTail278_0 best = best := by
  rw [innerTail278_0, scan, if_neg (by decide), innerCandidates278_0_checked]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_1
    (if 3 ≤ innerCandidates278_0.length then search edge 3 [1823, 2063] innerCandidates278_0 3886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan278_1

theorem rootBranch278 : search edge 4 [1823] rootCandidates278 1823 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1823] 1823 innerTail278_0 best = best
  exact innerScan278_0

def rootCandidates277 : List Nat := [1931, 1949, 2141, 2333, 2441, 2609, 3257, 5039, 5297, 5381, 6473, 6653, 6863, 7559, 7817]
set_option maxRecDepth 4096 in
theorem rootCandidates277_checked : candidates edge 5 0 1811 tail278 best.bound = rootCandidates277 := by
  unfold candidates
  have row : edge 1811 = PrimePairCache.smallRow1811.contains := by funext q; rfl
  rw [row]
  decide

def innerTail277_15 : List Nat := []
theorem innerScan277_15 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_15 best = best := rfl

def innerTail277_14 : List Nat := 7817 :: innerTail277_15
def innerCandidates277_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_14_checked : candidates edge 4 1811 7817 innerTail277_15 best.bound = innerCandidates277_14 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_14 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_14 best = best := by
  rw [innerTail277_14, scan, if_neg (by decide), innerCandidates277_14_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_15
    (if 3 ≤ innerCandidates277_14.length then search edge 3 [1811, 7817] innerCandidates277_14 9628 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_15

def innerTail277_13 : List Nat := 7559 :: innerTail277_14
def innerCandidates277_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_13_checked : candidates edge 4 1811 7559 innerTail277_14 best.bound = innerCandidates277_13 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_13 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_13 best = best := by
  rw [innerTail277_13, scan, if_neg (by decide), innerCandidates277_13_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_14
    (if 3 ≤ innerCandidates277_13.length then search edge 3 [1811, 7559] innerCandidates277_13 9370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_14

def innerTail277_12 : List Nat := 6863 :: innerTail277_13
def innerCandidates277_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_12_checked : candidates edge 4 1811 6863 innerTail277_13 best.bound = innerCandidates277_12 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_12 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_12 best = best := by
  rw [innerTail277_12, scan, if_neg (by decide), innerCandidates277_12_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_13
    (if 3 ≤ innerCandidates277_12.length then search edge 3 [1811, 6863] innerCandidates277_12 8674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_13

def innerTail277_11 : List Nat := 6653 :: innerTail277_12
def innerCandidates277_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_11_checked : candidates edge 4 1811 6653 innerTail277_12 best.bound = innerCandidates277_11 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_11 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_11 best = best := by
  rw [innerTail277_11, scan, if_neg (by decide), innerCandidates277_11_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_12
    (if 3 ≤ innerCandidates277_11.length then search edge 3 [1811, 6653] innerCandidates277_11 8464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_12

def innerTail277_10 : List Nat := 6473 :: innerTail277_11
def innerCandidates277_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_10_checked : candidates edge 4 1811 6473 innerTail277_11 best.bound = innerCandidates277_10 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_10 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_10 best = best := by
  rw [innerTail277_10, scan, if_neg (by decide), innerCandidates277_10_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_11
    (if 3 ≤ innerCandidates277_10.length then search edge 3 [1811, 6473] innerCandidates277_10 8284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_11

def innerTail277_9 : List Nat := 5381 :: innerTail277_10
def innerCandidates277_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_9_checked : candidates edge 4 1811 5381 innerTail277_10 best.bound = innerCandidates277_9 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_9 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_9 best = best := by
  rw [innerTail277_9, scan, if_neg (by decide), innerCandidates277_9_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_10
    (if 3 ≤ innerCandidates277_9.length then search edge 3 [1811, 5381] innerCandidates277_9 7192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_10

def innerTail277_8 : List Nat := 5297 :: innerTail277_9
def innerCandidates277_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_8_checked : candidates edge 4 1811 5297 innerTail277_9 best.bound = innerCandidates277_8 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_8 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_8 best = best := by
  rw [innerTail277_8, scan, if_neg (by decide), innerCandidates277_8_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_9
    (if 3 ≤ innerCandidates277_8.length then search edge 3 [1811, 5297] innerCandidates277_8 7108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_9

def innerTail277_7 : List Nat := 5039 :: innerTail277_8
def innerCandidates277_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_7_checked : candidates edge 4 1811 5039 innerTail277_8 best.bound = innerCandidates277_7 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_7 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_7 best = best := by
  rw [innerTail277_7, scan, if_neg (by decide), innerCandidates277_7_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_8
    (if 3 ≤ innerCandidates277_7.length then search edge 3 [1811, 5039] innerCandidates277_7 6850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_8

def innerTail277_6 : List Nat := 3257 :: innerTail277_7
def innerCandidates277_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_6_checked : candidates edge 4 1811 3257 innerTail277_7 best.bound = innerCandidates277_6 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_6 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_6 best = best := by
  rw [innerTail277_6, scan, if_neg (by decide), innerCandidates277_6_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_7
    (if 3 ≤ innerCandidates277_6.length then search edge 3 [1811, 3257] innerCandidates277_6 5068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_7

def innerTail277_5 : List Nat := 2609 :: innerTail277_6
def innerCandidates277_5 : List Nat := [3257]
set_option maxRecDepth 4096 in
theorem innerCandidates277_5_checked : candidates edge 4 1811 2609 innerTail277_6 best.bound = innerCandidates277_5 := by
  unfold candidates
  have row : edge 2609 = PrimePairCache.smallRow2609.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_5 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_5 best = best := by
  rw [innerTail277_5, scan, if_neg (by decide), innerCandidates277_5_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_6
    (if 3 ≤ innerCandidates277_5.length then search edge 3 [1811, 2609] innerCandidates277_5 4420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_6

def innerTail277_4 : List Nat := 2441 :: innerTail277_5
def innerCandidates277_4 : List Nat := [5297, 5381]
set_option maxRecDepth 4096 in
theorem innerCandidates277_4_checked : candidates edge 4 1811 2441 innerTail277_5 best.bound = innerCandidates277_4 := by
  unfold candidates
  have row : edge 2441 = PrimePairCache.smallRow2441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_4 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_4 best = best := by
  rw [innerTail277_4, scan, if_neg (by decide), innerCandidates277_4_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_5
    (if 3 ≤ innerCandidates277_4.length then search edge 3 [1811, 2441] innerCandidates277_4 4252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_5

def innerTail277_3 : List Nat := 2333 :: innerTail277_4
def innerCandidates277_3 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates277_3_checked : candidates edge 4 1811 2333 innerTail277_4 best.bound = innerCandidates277_3 := by
  unfold candidates
  have row : edge 2333 = PrimePairCache.smallRow2333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_3 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_3 best = best := by
  rw [innerTail277_3, scan, if_neg (by decide), innerCandidates277_3_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_4
    (if 3 ≤ innerCandidates277_3.length then search edge 3 [1811, 2333] innerCandidates277_3 4144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_4

def innerTail277_2 : List Nat := 2141 :: innerTail277_3
def innerCandidates277_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates277_2_checked : candidates edge 4 1811 2141 innerTail277_3 best.bound = innerCandidates277_2 := by
  unfold candidates
  have row : edge 2141 = PrimePairCache.smallRow2141.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_2 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_2 best = best := by
  rw [innerTail277_2, scan, if_neg (by decide), innerCandidates277_2_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_3
    (if 3 ≤ innerCandidates277_2.length then search edge 3 [1811, 2141] innerCandidates277_2 3952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_3

def innerTail277_1 : List Nat := 1949 :: innerTail277_2
def innerCandidates277_1 : List Nat := [6863]
set_option maxRecDepth 4096 in
theorem innerCandidates277_1_checked : candidates edge 4 1811 1949 innerTail277_2 best.bound = innerCandidates277_1 := by
  unfold candidates
  have row : edge 1949 = PrimePairCache.smallRow1949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_1 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_1 best = best := by
  rw [innerTail277_1, scan, if_neg (by decide), innerCandidates277_1_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_2
    (if 3 ≤ innerCandidates277_1.length then search edge 3 [1811, 1949] innerCandidates277_1 3760 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_2

def innerTail277_0 : List Nat := 1931 :: innerTail277_1
def innerCandidates277_0 : List Nat := [5297]
set_option maxRecDepth 4096 in
theorem innerCandidates277_0_checked : candidates edge 4 1811 1931 innerTail277_1 best.bound = innerCandidates277_0 := by
  unfold candidates
  have row : edge 1931 = PrimePairCache.smallRow1931.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan277_0 : scan edge (search edge 3) 4 [1811] 1811 innerTail277_0 best = best := by
  rw [innerTail277_0, scan, if_neg (by decide), innerCandidates277_0_checked]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_1
    (if 3 ≤ innerCandidates277_0.length then search edge 3 [1811, 1931] innerCandidates277_0 3742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan277_1

theorem rootBranch277 : search edge 4 [1811] rootCandidates277 1811 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1811] 1811 innerTail277_0 best = best
  exact innerScan277_0

def rootCandidates276 : List Nat := [2311, 2647, 3253, 3547, 3613, 3823, 4177, 4933, 5167, 5689, 6637, 6829, 6907, 7333, 7621, 7669, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates276_checked : candidates edge 5 0 1801 tail277 best.bound = rootCandidates276 := by
  unfold candidates
  have row : edge 1801 = PrimePairCache.smallRow1801.contains := by funext q; rfl
  rw [row]
  decide

def innerTail276_17 : List Nat := []
theorem innerScan276_17 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_17 best = best := rfl

def innerTail276_16 : List Nat := 8233 :: innerTail276_17
def innerCandidates276_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_16_checked : candidates edge 4 1801 8233 innerTail276_17 best.bound = innerCandidates276_16 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_16 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_16 best = best := by
  rw [innerTail276_16, scan, if_neg (by decide), innerCandidates276_16_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_17
    (if 3 ≤ innerCandidates276_16.length then search edge 3 [1801, 8233] innerCandidates276_16 10034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_17

def innerTail276_15 : List Nat := 7669 :: innerTail276_16
def innerCandidates276_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_15_checked : candidates edge 4 1801 7669 innerTail276_16 best.bound = innerCandidates276_15 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_15 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_15 best = best := by
  rw [innerTail276_15, scan, if_neg (by decide), innerCandidates276_15_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_16
    (if 3 ≤ innerCandidates276_15.length then search edge 3 [1801, 7669] innerCandidates276_15 9470 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_16

def innerTail276_14 : List Nat := 7621 :: innerTail276_15
def innerCandidates276_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_14_checked : candidates edge 4 1801 7621 innerTail276_15 best.bound = innerCandidates276_14 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_14 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_14 best = best := by
  rw [innerTail276_14, scan, if_neg (by decide), innerCandidates276_14_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_15
    (if 3 ≤ innerCandidates276_14.length then search edge 3 [1801, 7621] innerCandidates276_14 9422 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_15

def innerTail276_13 : List Nat := 7333 :: innerTail276_14
def innerCandidates276_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_13_checked : candidates edge 4 1801 7333 innerTail276_14 best.bound = innerCandidates276_13 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_13 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_13 best = best := by
  rw [innerTail276_13, scan, if_neg (by decide), innerCandidates276_13_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_14
    (if 3 ≤ innerCandidates276_13.length then search edge 3 [1801, 7333] innerCandidates276_13 9134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_14

def innerTail276_12 : List Nat := 6907 :: innerTail276_13
def innerCandidates276_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_12_checked : candidates edge 4 1801 6907 innerTail276_13 best.bound = innerCandidates276_12 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_12 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_12 best = best := by
  rw [innerTail276_12, scan, if_neg (by decide), innerCandidates276_12_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_13
    (if 3 ≤ innerCandidates276_12.length then search edge 3 [1801, 6907] innerCandidates276_12 8708 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_13

def innerTail276_11 : List Nat := 6829 :: innerTail276_12
def innerCandidates276_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_11_checked : candidates edge 4 1801 6829 innerTail276_12 best.bound = innerCandidates276_11 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_11 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_11 best = best := by
  rw [innerTail276_11, scan, if_neg (by decide), innerCandidates276_11_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_12
    (if 3 ≤ innerCandidates276_11.length then search edge 3 [1801, 6829] innerCandidates276_11 8630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_12

def innerTail276_10 : List Nat := 6637 :: innerTail276_11
def innerCandidates276_10 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates276_10_checked : candidates edge 4 1801 6637 innerTail276_11 best.bound = innerCandidates276_10 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_10 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_10 best = best := by
  rw [innerTail276_10, scan, if_neg (by decide), innerCandidates276_10_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_11
    (if 3 ≤ innerCandidates276_10.length then search edge 3 [1801, 6637] innerCandidates276_10 8438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_11

def innerTail276_9 : List Nat := 5689 :: innerTail276_10
def innerCandidates276_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_9_checked : candidates edge 4 1801 5689 innerTail276_10 best.bound = innerCandidates276_9 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_9 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_9 best = best := by
  rw [innerTail276_9, scan, if_neg (by decide), innerCandidates276_9_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_10
    (if 3 ≤ innerCandidates276_9.length then search edge 3 [1801, 5689] innerCandidates276_9 7490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_10

def innerTail276_8 : List Nat := 5167 :: innerTail276_9
def innerCandidates276_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_8_checked : candidates edge 4 1801 5167 innerTail276_9 best.bound = innerCandidates276_8 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_8 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_8 best = best := by
  rw [innerTail276_8, scan, if_neg (by decide), innerCandidates276_8_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_9
    (if 3 ≤ innerCandidates276_8.length then search edge 3 [1801, 5167] innerCandidates276_8 6968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_9

def innerTail276_7 : List Nat := 4933 :: innerTail276_8
def innerCandidates276_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_7_checked : candidates edge 4 1801 4933 innerTail276_8 best.bound = innerCandidates276_7 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_7 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_7 best = best := by
  rw [innerTail276_7, scan, if_neg (by decide), innerCandidates276_7_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_8
    (if 3 ≤ innerCandidates276_7.length then search edge 3 [1801, 4933] innerCandidates276_7 6734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_8

def innerTail276_6 : List Nat := 4177 :: innerTail276_7
def innerCandidates276_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_6_checked : candidates edge 4 1801 4177 innerTail276_7 best.bound = innerCandidates276_6 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_6 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_6 best = best := by
  rw [innerTail276_6, scan, if_neg (by decide), innerCandidates276_6_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_7
    (if 3 ≤ innerCandidates276_6.length then search edge 3 [1801, 4177] innerCandidates276_6 5978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_7

def innerTail276_5 : List Nat := 3823 :: innerTail276_6
def innerCandidates276_5 : List Nat := [5167]
set_option maxRecDepth 4096 in
theorem innerCandidates276_5_checked : candidates edge 4 1801 3823 innerTail276_6 best.bound = innerCandidates276_5 := by
  unfold candidates
  have row : edge 3823 = SearchData.row3823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_5 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_5 best = best := by
  rw [innerTail276_5, scan, if_neg (by decide), innerCandidates276_5_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_6
    (if 3 ≤ innerCandidates276_5.length then search edge 3 [1801, 3823] innerCandidates276_5 5624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_6

def innerTail276_4 : List Nat := 3613 :: innerTail276_5
def innerCandidates276_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_4_checked : candidates edge 4 1801 3613 innerTail276_5 best.bound = innerCandidates276_4 := by
  unfold candidates
  have row : edge 3613 = SearchData.row3613.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_4 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_4 best = best := by
  rw [innerTail276_4, scan, if_neg (by decide), innerCandidates276_4_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_5
    (if 3 ≤ innerCandidates276_4.length then search edge 3 [1801, 3613] innerCandidates276_4 5414 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_5

def innerTail276_3 : List Nat := 3547 :: innerTail276_4
def innerCandidates276_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_3_checked : candidates edge 4 1801 3547 innerTail276_4 best.bound = innerCandidates276_3 := by
  unfold candidates
  have row : edge 3547 = SearchData.row3547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_3 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_3 best = best := by
  rw [innerTail276_3, scan, if_neg (by decide), innerCandidates276_3_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_4
    (if 3 ≤ innerCandidates276_3.length then search edge 3 [1801, 3547] innerCandidates276_3 5348 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_4

def innerTail276_2 : List Nat := 3253 :: innerTail276_3
def innerCandidates276_2 : List Nat := [3547]
set_option maxRecDepth 4096 in
theorem innerCandidates276_2_checked : candidates edge 4 1801 3253 innerTail276_3 best.bound = innerCandidates276_2 := by
  unfold candidates
  have row : edge 3253 = SearchData.row3253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_2 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_2 best = best := by
  rw [innerTail276_2, scan, if_neg (by decide), innerCandidates276_2_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_3
    (if 3 ≤ innerCandidates276_2.length then search edge 3 [1801, 3253] innerCandidates276_2 5054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_3

def innerTail276_1 : List Nat := 2647 :: innerTail276_2
def innerCandidates276_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates276_1_checked : candidates edge 4 1801 2647 innerTail276_2 best.bound = innerCandidates276_1 := by
  unfold candidates
  have row : edge 2647 = PrimePairCache.smallRow2647.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_1 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_1 best = best := by
  rw [innerTail276_1, scan, if_neg (by decide), innerCandidates276_1_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_2
    (if 3 ≤ innerCandidates276_1.length then search edge 3 [1801, 2647] innerCandidates276_1 4448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_2

def innerTail276_0 : List Nat := 2311 :: innerTail276_1
def innerCandidates276_0 : List Nat := [2647]
set_option maxRecDepth 4096 in
theorem innerCandidates276_0_checked : candidates edge 4 1801 2311 innerTail276_1 best.bound = innerCandidates276_0 := by
  unfold candidates
  have row : edge 2311 = PrimePairCache.smallRow2311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan276_0 : scan edge (search edge 3) 4 [1801] 1801 innerTail276_0 best = best := by
  rw [innerTail276_0, scan, if_neg (by decide), innerCandidates276_0_checked]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_1
    (if 3 ≤ innerCandidates276_0.length then search edge 3 [1801, 2311] innerCandidates276_0 4112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan276_1

theorem rootBranch276 : search edge 4 [1801] rootCandidates276 1801 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [1801] 1801 innerTail276_0 best = best
  exact innerScan276_0

end ProjectEulerSolutions.PrimePairUpper.Search
