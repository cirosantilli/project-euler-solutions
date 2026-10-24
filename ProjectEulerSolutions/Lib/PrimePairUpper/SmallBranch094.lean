import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates416 : List Nat := [3023, 3209, 3947, 4289, 4493, 4721, 4877, 5501, 5867, 6329, 6473, 7211, 7307, 7547]
set_option maxRecDepth 4096 in
theorem rootCandidates416_checked : candidates edge 5 0 2897 tail417 best.bound = rootCandidates416 := by
  unfold candidates
  have row : edge 2897 = SearchData.row2897.contains := by funext q; rfl
  rw [row]
  decide

def innerTail416_14 : List Nat := []
theorem innerScan416_14 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_14 best = best := rfl

def innerTail416_13 : List Nat := 7547 :: innerTail416_14
def innerCandidates416_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates416_13_checked : candidates edge 4 2897 7547 innerTail416_14 best.bound = innerCandidates416_13 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_13 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_13 best = best := by
  rw [innerTail416_13, scan, if_neg (by decide), innerCandidates416_13_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_14
    (if 3 ≤ innerCandidates416_13.length then search edge 3 [2897, 7547] innerCandidates416_13 10444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_14

def innerTail416_12 : List Nat := 7307 :: innerTail416_13
def innerCandidates416_12 : List Nat := [7547]
set_option maxRecDepth 4096 in
theorem innerCandidates416_12_checked : candidates edge 4 2897 7307 innerTail416_13 best.bound = innerCandidates416_12 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_12 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_12 best = best := by
  rw [innerTail416_12, scan, if_neg (by decide), innerCandidates416_12_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_13
    (if 3 ≤ innerCandidates416_12.length then search edge 3 [2897, 7307] innerCandidates416_12 10204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_13

def innerTail416_11 : List Nat := 7211 :: innerTail416_12
def innerCandidates416_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates416_11_checked : candidates edge 4 2897 7211 innerTail416_12 best.bound = innerCandidates416_11 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_11 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_11 best = best := by
  rw [innerTail416_11, scan, if_neg (by decide), innerCandidates416_11_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_12
    (if 3 ≤ innerCandidates416_11.length then search edge 3 [2897, 7211] innerCandidates416_11 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_12

def innerTail416_10 : List Nat := 6473 :: innerTail416_11
def innerCandidates416_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates416_10_checked : candidates edge 4 2897 6473 innerTail416_11 best.bound = innerCandidates416_10 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_10 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_10 best = best := by
  rw [innerTail416_10, scan, if_neg (by decide), innerCandidates416_10_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_11
    (if 3 ≤ innerCandidates416_10.length then search edge 3 [2897, 6473] innerCandidates416_10 9370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_11

def innerTail416_9 : List Nat := 6329 :: innerTail416_10
def innerCandidates416_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates416_9_checked : candidates edge 4 2897 6329 innerTail416_10 best.bound = innerCandidates416_9 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_9 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_9 best = best := by
  rw [innerTail416_9, scan, if_neg (by decide), innerCandidates416_9_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_10
    (if 3 ≤ innerCandidates416_9.length then search edge 3 [2897, 6329] innerCandidates416_9 9226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_10

def innerTail416_8 : List Nat := 5867 :: innerTail416_9
def innerCandidates416_8 : List Nat := [6329]
set_option maxRecDepth 4096 in
theorem innerCandidates416_8_checked : candidates edge 4 2897 5867 innerTail416_9 best.bound = innerCandidates416_8 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_8 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_8 best = best := by
  rw [innerTail416_8, scan, if_neg (by decide), innerCandidates416_8_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_9
    (if 3 ≤ innerCandidates416_8.length then search edge 3 [2897, 5867] innerCandidates416_8 8764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_9

def innerTail416_7 : List Nat := 5501 :: innerTail416_8
def innerCandidates416_7 : List Nat := [5867, 7211]
set_option maxRecDepth 4096 in
theorem innerCandidates416_7_checked : candidates edge 4 2897 5501 innerTail416_8 best.bound = innerCandidates416_7 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_7 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_7 best = best := by
  rw [innerTail416_7, scan, if_neg (by decide), innerCandidates416_7_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_8
    (if 3 ≤ innerCandidates416_7.length then search edge 3 [2897, 5501] innerCandidates416_7 8398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_8

def innerTail416_6 : List Nat := 4877 :: innerTail416_7
def innerCandidates416_6 : List Nat := [5501, 7307]
set_option maxRecDepth 4096 in
theorem innerCandidates416_6_checked : candidates edge 4 2897 4877 innerTail416_7 best.bound = innerCandidates416_6 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_6 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_6 best = best := by
  rw [innerTail416_6, scan, if_neg (by decide), innerCandidates416_6_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_7
    (if 3 ≤ innerCandidates416_6.length then search edge 3 [2897, 4877] innerCandidates416_6 7774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_7

def innerTail416_5 : List Nat := 4721 :: innerTail416_6
def innerCandidates416_5 : List Nat := [4877, 5501]
set_option maxRecDepth 4096 in
theorem innerCandidates416_5_checked : candidates edge 4 2897 4721 innerTail416_6 best.bound = innerCandidates416_5 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_5 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_5 best = best := by
  rw [innerTail416_5, scan, if_neg (by decide), innerCandidates416_5_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_6
    (if 3 ≤ innerCandidates416_5.length then search edge 3 [2897, 4721] innerCandidates416_5 7618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_6

def innerTail416_4 : List Nat := 4493 :: innerTail416_5
def innerCandidates416_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates416_4_checked : candidates edge 4 2897 4493 innerTail416_5 best.bound = innerCandidates416_4 := by
  unfold candidates
  have row : edge 4493 = SearchData.row4493.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_4 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_4 best = best := by
  rw [innerTail416_4, scan, if_neg (by decide), innerCandidates416_4_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_5
    (if 3 ≤ innerCandidates416_4.length then search edge 3 [2897, 4493] innerCandidates416_4 7390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_5

def innerTail416_3 : List Nat := 4289 :: innerTail416_4
def innerCandidates416_3 : List Nat := [6473]
set_option maxRecDepth 4096 in
theorem innerCandidates416_3_checked : candidates edge 4 2897 4289 innerTail416_4 best.bound = innerCandidates416_3 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_3 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_3 best = best := by
  rw [innerTail416_3, scan, if_neg (by decide), innerCandidates416_3_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_4
    (if 3 ≤ innerCandidates416_3.length then search edge 3 [2897, 4289] innerCandidates416_3 7186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_4

def innerTail416_2 : List Nat := 3947 :: innerTail416_3
def innerCandidates416_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates416_2_checked : candidates edge 4 2897 3947 innerTail416_3 best.bound = innerCandidates416_2 := by
  unfold candidates
  have row : edge 3947 = SearchData.row3947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_2 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_2 best = best := by
  rw [innerTail416_2, scan, if_neg (by decide), innerCandidates416_2_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_3
    (if 3 ≤ innerCandidates416_2.length then search edge 3 [2897, 3947] innerCandidates416_2 6844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_3

def innerTail416_1 : List Nat := 3209 :: innerTail416_2
def innerCandidates416_1 : List Nat := [4721]
set_option maxRecDepth 4096 in
theorem innerCandidates416_1_checked : candidates edge 4 2897 3209 innerTail416_2 best.bound = innerCandidates416_1 := by
  unfold candidates
  have row : edge 3209 = SearchData.row3209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_1 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_1 best = best := by
  rw [innerTail416_1, scan, if_neg (by decide), innerCandidates416_1_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_2
    (if 3 ≤ innerCandidates416_1.length then search edge 3 [2897, 3209] innerCandidates416_1 6106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_2

def innerTail416_0 : List Nat := 3023 :: innerTail416_1
def innerCandidates416_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates416_0_checked : candidates edge 4 2897 3023 innerTail416_1 best.bound = innerCandidates416_0 := by
  unfold candidates
  have row : edge 3023 = SearchData.row3023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan416_0 : scan edge (search edge 3) 4 [2897] 2897 innerTail416_0 best = best := by
  rw [innerTail416_0, scan, if_neg (by decide), innerCandidates416_0_checked]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_1
    (if 3 ≤ innerCandidates416_0.length then search edge 3 [2897, 3023] innerCandidates416_0 5920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan416_1

theorem rootBranch416 : search edge 4 [2897] rootCandidates416 2897 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2897] 2897 innerTail416_0 best = best
  exact innerScan416_0

def rootCandidates415 : List Nat := [2917, 4231, 4273, 4597, 4987, 5653, 6709, 6733, 7573]
set_option maxRecDepth 4096 in
theorem rootCandidates415_checked : candidates edge 5 0 2887 tail416 best.bound = rootCandidates415 := by
  unfold candidates
  have row : edge 2887 = SearchData.row2887.contains := by funext q; rfl
  rw [row]
  decide

def innerTail415_9 : List Nat := []
theorem innerScan415_9 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_9 best = best := rfl

def innerTail415_8 : List Nat := 7573 :: innerTail415_9
def innerCandidates415_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_8_checked : candidates edge 4 2887 7573 innerTail415_9 best.bound = innerCandidates415_8 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_8 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_8 best = best := by
  rw [innerTail415_8, scan, if_neg (by decide), innerCandidates415_8_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_9
    (if 3 ≤ innerCandidates415_8.length then search edge 3 [2887, 7573] innerCandidates415_8 10460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_9

def innerTail415_7 : List Nat := 6733 :: innerTail415_8
def innerCandidates415_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_7_checked : candidates edge 4 2887 6733 innerTail415_8 best.bound = innerCandidates415_7 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_7 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_7 best = best := by
  rw [innerTail415_7, scan, if_neg (by decide), innerCandidates415_7_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_8
    (if 3 ≤ innerCandidates415_7.length then search edge 3 [2887, 6733] innerCandidates415_7 9620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_8

def innerTail415_6 : List Nat := 6709 :: innerTail415_7
def innerCandidates415_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_6_checked : candidates edge 4 2887 6709 innerTail415_7 best.bound = innerCandidates415_6 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_6 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_6 best = best := by
  rw [innerTail415_6, scan, if_neg (by decide), innerCandidates415_6_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_7
    (if 3 ≤ innerCandidates415_6.length then search edge 3 [2887, 6709] innerCandidates415_6 9596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_7

def innerTail415_5 : List Nat := 5653 :: innerTail415_6
def innerCandidates415_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_5_checked : candidates edge 4 2887 5653 innerTail415_6 best.bound = innerCandidates415_5 := by
  unfold candidates
  have row : edge 5653 = SearchData.row5653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_5 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_5 best = best := by
  rw [innerTail415_5, scan, if_neg (by decide), innerCandidates415_5_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_6
    (if 3 ≤ innerCandidates415_5.length then search edge 3 [2887, 5653] innerCandidates415_5 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_6

def innerTail415_4 : List Nat := 4987 :: innerTail415_5
def innerCandidates415_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_4_checked : candidates edge 4 2887 4987 innerTail415_5 best.bound = innerCandidates415_4 := by
  unfold candidates
  have row : edge 4987 = SearchData.row4987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_4 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_4 best = best := by
  rw [innerTail415_4, scan, if_neg (by decide), innerCandidates415_4_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_5
    (if 3 ≤ innerCandidates415_4.length then search edge 3 [2887, 4987] innerCandidates415_4 7874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_5

def innerTail415_3 : List Nat := 4597 :: innerTail415_4
def innerCandidates415_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_3_checked : candidates edge 4 2887 4597 innerTail415_4 best.bound = innerCandidates415_3 := by
  unfold candidates
  have row : edge 4597 = SearchData.row4597.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_3 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_3 best = best := by
  rw [innerTail415_3, scan, if_neg (by decide), innerCandidates415_3_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_4
    (if 3 ≤ innerCandidates415_3.length then search edge 3 [2887, 4597] innerCandidates415_3 7484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_4

def innerTail415_2 : List Nat := 4273 :: innerTail415_3
def innerCandidates415_2 : List Nat := [7573]
set_option maxRecDepth 4096 in
theorem innerCandidates415_2_checked : candidates edge 4 2887 4273 innerTail415_3 best.bound = innerCandidates415_2 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_2 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_2 best = best := by
  rw [innerTail415_2, scan, if_neg (by decide), innerCandidates415_2_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_3
    (if 3 ≤ innerCandidates415_2.length then search edge 3 [2887, 4273] innerCandidates415_2 7160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_3

def innerTail415_1 : List Nat := 4231 :: innerTail415_2
def innerCandidates415_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_1_checked : candidates edge 4 2887 4231 innerTail415_2 best.bound = innerCandidates415_1 := by
  unfold candidates
  have row : edge 4231 = SearchData.row4231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_1 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_1 best = best := by
  rw [innerTail415_1, scan, if_neg (by decide), innerCandidates415_1_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_2
    (if 3 ≤ innerCandidates415_1.length then search edge 3 [2887, 4231] innerCandidates415_1 7118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_2

def innerTail415_0 : List Nat := 2917 :: innerTail415_1
def innerCandidates415_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates415_0_checked : candidates edge 4 2887 2917 innerTail415_1 best.bound = innerCandidates415_0 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan415_0 : scan edge (search edge 3) 4 [2887] 2887 innerTail415_0 best = best := by
  rw [innerTail415_0, scan, if_neg (by decide), innerCandidates415_0_checked]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_1
    (if 3 ≤ innerCandidates415_0.length then search edge 3 [2887, 2917] innerCandidates415_0 5804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan415_1

theorem rootBranch415 : search edge 4 [2887] rootCandidates415 2887 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2887] 2887 innerTail415_0 best = best
  exact innerScan415_0

def rootCandidates414 : List Nat := [3449, 3581, 4397, 4673, 5171, 5303, 5387, 5531, 5573, 5693, 5867, 6143, 6221, 6701, 6899, 6959, 7151, 7883]
set_option maxRecDepth 4096 in
theorem rootCandidates414_checked : candidates edge 5 0 2879 tail415 best.bound = rootCandidates414 := by
  unfold candidates
  have row : edge 2879 = SearchData.row2879.contains := by funext q; rfl
  rw [row]
  decide

def innerTail414_18 : List Nat := []
theorem innerScan414_18 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_18 best = best := rfl

def innerTail414_17 : List Nat := 7883 :: innerTail414_18
def innerCandidates414_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_17_checked : candidates edge 4 2879 7883 innerTail414_18 best.bound = innerCandidates414_17 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_17 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_17 best = best := by
  rw [innerTail414_17, scan, if_neg (by decide), innerCandidates414_17_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_18
    (if 3 ≤ innerCandidates414_17.length then search edge 3 [2879, 7883] innerCandidates414_17 10762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_18

def innerTail414_16 : List Nat := 7151 :: innerTail414_17
def innerCandidates414_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_16_checked : candidates edge 4 2879 7151 innerTail414_17 best.bound = innerCandidates414_16 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_16 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_16 best = best := by
  rw [innerTail414_16, scan, if_neg (by decide), innerCandidates414_16_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_17
    (if 3 ≤ innerCandidates414_16.length then search edge 3 [2879, 7151] innerCandidates414_16 10030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_17

def innerTail414_15 : List Nat := 6959 :: innerTail414_16
def innerCandidates414_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_15_checked : candidates edge 4 2879 6959 innerTail414_16 best.bound = innerCandidates414_15 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_15 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_15 best = best := by
  rw [innerTail414_15, scan, if_neg (by decide), innerCandidates414_15_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_16
    (if 3 ≤ innerCandidates414_15.length then search edge 3 [2879, 6959] innerCandidates414_15 9838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_16

def innerTail414_14 : List Nat := 6899 :: innerTail414_15
def innerCandidates414_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_14_checked : candidates edge 4 2879 6899 innerTail414_15 best.bound = innerCandidates414_14 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_14 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_14 best = best := by
  rw [innerTail414_14, scan, if_neg (by decide), innerCandidates414_14_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_15
    (if 3 ≤ innerCandidates414_14.length then search edge 3 [2879, 6899] innerCandidates414_14 9778 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_15

def innerTail414_13 : List Nat := 6701 :: innerTail414_14
def innerCandidates414_13 : List Nat := [6899]
set_option maxRecDepth 4096 in
theorem innerCandidates414_13_checked : candidates edge 4 2879 6701 innerTail414_14 best.bound = innerCandidates414_13 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_13 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_13 best = best := by
  rw [innerTail414_13, scan, if_neg (by decide), innerCandidates414_13_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_14
    (if 3 ≤ innerCandidates414_13.length then search edge 3 [2879, 6701] innerCandidates414_13 9580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_14

def innerTail414_12 : List Nat := 6221 :: innerTail414_13
def innerCandidates414_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_12_checked : candidates edge 4 2879 6221 innerTail414_13 best.bound = innerCandidates414_12 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_12 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_12 best = best := by
  rw [innerTail414_12, scan, if_neg (by decide), innerCandidates414_12_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_13
    (if 3 ≤ innerCandidates414_12.length then search edge 3 [2879, 6221] innerCandidates414_12 9100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_13

def innerTail414_11 : List Nat := 6143 :: innerTail414_12
def innerCandidates414_11 : List Nat := [7151]
set_option maxRecDepth 4096 in
theorem innerCandidates414_11_checked : candidates edge 4 2879 6143 innerTail414_12 best.bound = innerCandidates414_11 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_11 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_11 best = best := by
  rw [innerTail414_11, scan, if_neg (by decide), innerCandidates414_11_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_12
    (if 3 ≤ innerCandidates414_11.length then search edge 3 [2879, 6143] innerCandidates414_11 9022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_12

def innerTail414_10 : List Nat := 5867 :: innerTail414_11
def innerCandidates414_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_10_checked : candidates edge 4 2879 5867 innerTail414_11 best.bound = innerCandidates414_10 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_10 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_10 best = best := by
  rw [innerTail414_10, scan, if_neg (by decide), innerCandidates414_10_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_11
    (if 3 ≤ innerCandidates414_10.length then search edge 3 [2879, 5867] innerCandidates414_10 8746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_11

def innerTail414_9 : List Nat := 5693 :: innerTail414_10
def innerCandidates414_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_9_checked : candidates edge 4 2879 5693 innerTail414_10 best.bound = innerCandidates414_9 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_9 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_9 best = best := by
  rw [innerTail414_9, scan, if_neg (by decide), innerCandidates414_9_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_10
    (if 3 ≤ innerCandidates414_9.length then search edge 3 [2879, 5693] innerCandidates414_9 8572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_10

def innerTail414_8 : List Nat := 5573 :: innerTail414_9
def innerCandidates414_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_8_checked : candidates edge 4 2879 5573 innerTail414_9 best.bound = innerCandidates414_8 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_8 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_8 best = best := by
  rw [innerTail414_8, scan, if_neg (by decide), innerCandidates414_8_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_9
    (if 3 ≤ innerCandidates414_8.length then search edge 3 [2879, 5573] innerCandidates414_8 8452 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_9

def innerTail414_7 : List Nat := 5531 :: innerTail414_8
def innerCandidates414_7 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates414_7_checked : candidates edge 4 2879 5531 innerTail414_8 best.bound = innerCandidates414_7 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_7 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_7 best = best := by
  rw [innerTail414_7, scan, if_neg (by decide), innerCandidates414_7_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_8
    (if 3 ≤ innerCandidates414_7.length then search edge 3 [2879, 5531] innerCandidates414_7 8410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_8

def innerTail414_6 : List Nat := 5387 :: innerTail414_7
def innerCandidates414_6 : List Nat := [6959]
set_option maxRecDepth 4096 in
theorem innerCandidates414_6_checked : candidates edge 4 2879 5387 innerTail414_7 best.bound = innerCandidates414_6 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_6 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_6 best = best := by
  rw [innerTail414_6, scan, if_neg (by decide), innerCandidates414_6_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_7
    (if 3 ≤ innerCandidates414_6.length then search edge 3 [2879, 5387] innerCandidates414_6 8266 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_7

def innerTail414_5 : List Nat := 5303 :: innerTail414_6
def innerCandidates414_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_5_checked : candidates edge 4 2879 5303 innerTail414_6 best.bound = innerCandidates414_5 := by
  unfold candidates
  have row : edge 5303 = SearchData.row5303.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_5 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_5 best = best := by
  rw [innerTail414_5, scan, if_neg (by decide), innerCandidates414_5_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_6
    (if 3 ≤ innerCandidates414_5.length then search edge 3 [2879, 5303] innerCandidates414_5 8182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_6

def innerTail414_4 : List Nat := 5171 :: innerTail414_5
def innerCandidates414_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_4_checked : candidates edge 4 2879 5171 innerTail414_5 best.bound = innerCandidates414_4 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_4 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_4 best = best := by
  rw [innerTail414_4, scan, if_neg (by decide), innerCandidates414_4_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_5
    (if 3 ≤ innerCandidates414_4.length then search edge 3 [2879, 5171] innerCandidates414_4 8050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_5

def innerTail414_3 : List Nat := 4673 :: innerTail414_4
def innerCandidates414_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates414_3_checked : candidates edge 4 2879 4673 innerTail414_4 best.bound = innerCandidates414_3 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_3 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_3 best = best := by
  rw [innerTail414_3, scan, if_neg (by decide), innerCandidates414_3_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_4
    (if 3 ≤ innerCandidates414_3.length then search edge 3 [2879, 4673] innerCandidates414_3 7552 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_4

def innerTail414_2 : List Nat := 4397 :: innerTail414_3
def innerCandidates414_2 : List Nat := [6701]
set_option maxRecDepth 4096 in
theorem innerCandidates414_2_checked : candidates edge 4 2879 4397 innerTail414_3 best.bound = innerCandidates414_2 := by
  unfold candidates
  have row : edge 4397 = SearchData.row4397.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_2 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_2 best = best := by
  rw [innerTail414_2, scan, if_neg (by decide), innerCandidates414_2_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_3
    (if 3 ≤ innerCandidates414_2.length then search edge 3 [2879, 4397] innerCandidates414_2 7276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_3

def innerTail414_1 : List Nat := 3581 :: innerTail414_2
def innerCandidates414_1 : List Nat := [5303, 5387, 6701]
set_option maxRecDepth 4096 in
theorem innerCandidates414_1_checked : candidates edge 4 2879 3581 innerTail414_2 best.bound = innerCandidates414_1 := by
  unfold candidates
  have row : edge 3581 = SearchData.row3581.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild414_1 : search edge 3 [2879, 3581] innerCandidates414_1 6460 best = best := by decide

theorem innerScan414_1 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_1 best = best := by
  rw [innerTail414_1, scan, if_neg (by decide), innerCandidates414_1_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_2
    (if 3 ≤ innerCandidates414_1.length then search edge 3 [2879, 3581] innerCandidates414_1 6460 best else best) = best
  rw [if_pos (by decide), innerChild414_1]
  exact innerScan414_2

def innerTail414_0 : List Nat := 3449 :: innerTail414_1
def innerCandidates414_0 : List Nat := [5693]
set_option maxRecDepth 4096 in
theorem innerCandidates414_0_checked : candidates edge 4 2879 3449 innerTail414_1 best.bound = innerCandidates414_0 := by
  unfold candidates
  have row : edge 3449 = SearchData.row3449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan414_0 : scan edge (search edge 3) 4 [2879] 2879 innerTail414_0 best = best := by
  rw [innerTail414_0, scan, if_neg (by decide), innerCandidates414_0_checked]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_1
    (if 3 ≤ innerCandidates414_0.length then search edge 3 [2879, 3449] innerCandidates414_0 6328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan414_1

theorem rootBranch414 : search edge 4 [2879] rootCandidates414 2879 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2879] 2879 innerTail414_0 best = best
  exact innerScan414_0

def rootCandidates413 : List Nat := [3257, 3671, 3911, 4919, 5693, 5861, 6719, 7013, 7499, 7649, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates413_checked : candidates edge 5 0 2861 tail414 best.bound = rootCandidates413 := by
  unfold candidates
  have row : edge 2861 = SearchData.row2861.contains := by funext q; rfl
  rw [row]
  decide

def innerTail413_11 : List Nat := []
theorem innerScan413_11 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_11 best = best := rfl

def innerTail413_10 : List Nat := 8171 :: innerTail413_11
def innerCandidates413_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_10_checked : candidates edge 4 2861 8171 innerTail413_11 best.bound = innerCandidates413_10 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_10 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_10 best = best := by
  rw [innerTail413_10, scan, if_neg (by decide), innerCandidates413_10_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_11
    (if 3 ≤ innerCandidates413_10.length then search edge 3 [2861, 8171] innerCandidates413_10 11032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_11

def innerTail413_9 : List Nat := 7649 :: innerTail413_10
def innerCandidates413_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_9_checked : candidates edge 4 2861 7649 innerTail413_10 best.bound = innerCandidates413_9 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_9 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_9 best = best := by
  rw [innerTail413_9, scan, if_neg (by decide), innerCandidates413_9_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_10
    (if 3 ≤ innerCandidates413_9.length then search edge 3 [2861, 7649] innerCandidates413_9 10510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_10

def innerTail413_8 : List Nat := 7499 :: innerTail413_9
def innerCandidates413_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_8_checked : candidates edge 4 2861 7499 innerTail413_9 best.bound = innerCandidates413_8 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_8 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_8 best = best := by
  rw [innerTail413_8, scan, if_neg (by decide), innerCandidates413_8_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_9
    (if 3 ≤ innerCandidates413_8.length then search edge 3 [2861, 7499] innerCandidates413_8 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_9

def innerTail413_7 : List Nat := 7013 :: innerTail413_8
def innerCandidates413_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_7_checked : candidates edge 4 2861 7013 innerTail413_8 best.bound = innerCandidates413_7 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_7 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_7 best = best := by
  rw [innerTail413_7, scan, if_neg (by decide), innerCandidates413_7_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_8
    (if 3 ≤ innerCandidates413_7.length then search edge 3 [2861, 7013] innerCandidates413_7 9874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_8

def innerTail413_6 : List Nat := 6719 :: innerTail413_7
def innerCandidates413_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_6_checked : candidates edge 4 2861 6719 innerTail413_7 best.bound = innerCandidates413_6 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_6 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_6 best = best := by
  rw [innerTail413_6, scan, if_neg (by decide), innerCandidates413_6_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_7
    (if 3 ≤ innerCandidates413_6.length then search edge 3 [2861, 6719] innerCandidates413_6 9580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_7

def innerTail413_5 : List Nat := 5861 :: innerTail413_6
def innerCandidates413_5 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates413_5_checked : candidates edge 4 2861 5861 innerTail413_6 best.bound = innerCandidates413_5 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_5 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_5 best = best := by
  rw [innerTail413_5, scan, if_neg (by decide), innerCandidates413_5_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_6
    (if 3 ≤ innerCandidates413_5.length then search edge 3 [2861, 5861] innerCandidates413_5 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_6

def innerTail413_4 : List Nat := 5693 :: innerTail413_5
def innerCandidates413_4 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates413_4_checked : candidates edge 4 2861 5693 innerTail413_5 best.bound = innerCandidates413_4 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_4 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_4 best = best := by
  rw [innerTail413_4, scan, if_neg (by decide), innerCandidates413_4_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_5
    (if 3 ≤ innerCandidates413_4.length then search edge 3 [2861, 5693] innerCandidates413_4 8554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_5

def innerTail413_3 : List Nat := 4919 :: innerTail413_4
def innerCandidates413_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_3_checked : candidates edge 4 2861 4919 innerTail413_4 best.bound = innerCandidates413_3 := by
  unfold candidates
  have row : edge 4919 = SearchData.row4919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_3 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_3 best = best := by
  rw [innerTail413_3, scan, if_neg (by decide), innerCandidates413_3_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_4
    (if 3 ≤ innerCandidates413_3.length then search edge 3 [2861, 4919] innerCandidates413_3 7780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_4

def innerTail413_2 : List Nat := 3911 :: innerTail413_3
def innerCandidates413_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_2_checked : candidates edge 4 2861 3911 innerTail413_3 best.bound = innerCandidates413_2 := by
  unfold candidates
  have row : edge 3911 = SearchData.row3911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_2 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_2 best = best := by
  rw [innerTail413_2, scan, if_neg (by decide), innerCandidates413_2_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_3
    (if 3 ≤ innerCandidates413_2.length then search edge 3 [2861, 3911] innerCandidates413_2 6772 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_3

def innerTail413_1 : List Nat := 3671 :: innerTail413_2
def innerCandidates413_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_1_checked : candidates edge 4 2861 3671 innerTail413_2 best.bound = innerCandidates413_1 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_1 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_1 best = best := by
  rw [innerTail413_1, scan, if_neg (by decide), innerCandidates413_1_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_2
    (if 3 ≤ innerCandidates413_1.length then search edge 3 [2861, 3671] innerCandidates413_1 6532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_2

def innerTail413_0 : List Nat := 3257 :: innerTail413_1
def innerCandidates413_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates413_0_checked : candidates edge 4 2861 3257 innerTail413_1 best.bound = innerCandidates413_0 := by
  unfold candidates
  have row : edge 3257 = SearchData.row3257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan413_0 : scan edge (search edge 3) 4 [2861] 2861 innerTail413_0 best = best := by
  rw [innerTail413_0, scan, if_neg (by decide), innerCandidates413_0_checked]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_1
    (if 3 ≤ innerCandidates413_0.length then search edge 3 [2861, 3257] innerCandidates413_0 6118 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan413_1

theorem rootBranch413 : search edge 4 [2861] rootCandidates413 2861 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2861] 2861 innerTail413_0 best = best
  exact innerScan413_0

end ProjectEulerSolutions.PrimePairUpper.Search
