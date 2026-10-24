import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates421 : List Nat := [2999, 3167, 3491, 3557, 3671, 4019, 4133, 4217, 4679, 5297, 5417, 5573, 5981, 6863, 7211]
set_option maxRecDepth 4096 in
theorem rootCandidates421_checked : candidates edge 5 0 2939 tail422 best.bound = rootCandidates421 := by
  unfold candidates
  have row : edge 2939 = SearchData.row2939.contains := by funext q; rfl
  rw [row]
  decide

def innerTail421_15 : List Nat := []
theorem innerScan421_15 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_15 best = best := rfl

def innerTail421_14 : List Nat := 7211 :: innerTail421_15
def innerCandidates421_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_14_checked : candidates edge 4 2939 7211 innerTail421_15 best.bound = innerCandidates421_14 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_14 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_14 best = best := by
  rw [innerTail421_14, scan, if_neg (by decide), innerCandidates421_14_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_15
    (if 3 ≤ innerCandidates421_14.length then search edge 3 [2939, 7211] innerCandidates421_14 10150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_15

def innerTail421_13 : List Nat := 6863 :: innerTail421_14
def innerCandidates421_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_13_checked : candidates edge 4 2939 6863 innerTail421_14 best.bound = innerCandidates421_13 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_13 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_13 best = best := by
  rw [innerTail421_13, scan, if_neg (by decide), innerCandidates421_13_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_14
    (if 3 ≤ innerCandidates421_13.length then search edge 3 [2939, 6863] innerCandidates421_13 9802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_14

def innerTail421_12 : List Nat := 5981 :: innerTail421_13
def innerCandidates421_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_12_checked : candidates edge 4 2939 5981 innerTail421_13 best.bound = innerCandidates421_12 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_12 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_12 best = best := by
  rw [innerTail421_12, scan, if_neg (by decide), innerCandidates421_12_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_13
    (if 3 ≤ innerCandidates421_12.length then search edge 3 [2939, 5981] innerCandidates421_12 8920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_13

def innerTail421_11 : List Nat := 5573 :: innerTail421_12
def innerCandidates421_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_11_checked : candidates edge 4 2939 5573 innerTail421_12 best.bound = innerCandidates421_11 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_11 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_11 best = best := by
  rw [innerTail421_11, scan, if_neg (by decide), innerCandidates421_11_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_12
    (if 3 ≤ innerCandidates421_11.length then search edge 3 [2939, 5573] innerCandidates421_11 8512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_12

def innerTail421_10 : List Nat := 5417 :: innerTail421_11
def innerCandidates421_10 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates421_10_checked : candidates edge 4 2939 5417 innerTail421_11 best.bound = innerCandidates421_10 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_10 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_10 best = best := by
  rw [innerTail421_10, scan, if_neg (by decide), innerCandidates421_10_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_11
    (if 3 ≤ innerCandidates421_10.length then search edge 3 [2939, 5417] innerCandidates421_10 8356 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_11

def innerTail421_9 : List Nat := 5297 :: innerTail421_10
def innerCandidates421_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_9_checked : candidates edge 4 2939 5297 innerTail421_10 best.bound = innerCandidates421_9 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_9 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_9 best = best := by
  rw [innerTail421_9, scan, if_neg (by decide), innerCandidates421_9_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_10
    (if 3 ≤ innerCandidates421_9.length then search edge 3 [2939, 5297] innerCandidates421_9 8236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_10

def innerTail421_8 : List Nat := 4679 :: innerTail421_9
def innerCandidates421_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_8_checked : candidates edge 4 2939 4679 innerTail421_9 best.bound = innerCandidates421_8 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_8 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_8 best = best := by
  rw [innerTail421_8, scan, if_neg (by decide), innerCandidates421_8_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_9
    (if 3 ≤ innerCandidates421_8.length then search edge 3 [2939, 4679] innerCandidates421_8 7618 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_9

def innerTail421_7 : List Nat := 4217 :: innerTail421_8
def innerCandidates421_7 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates421_7_checked : candidates edge 4 2939 4217 innerTail421_8 best.bound = innerCandidates421_7 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_7 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_7 best = best := by
  rw [innerTail421_7, scan, if_neg (by decide), innerCandidates421_7_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_8
    (if 3 ≤ innerCandidates421_7.length then search edge 3 [2939, 4217] innerCandidates421_7 7156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_8

def innerTail421_6 : List Nat := 4133 :: innerTail421_7
def innerCandidates421_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_6_checked : candidates edge 4 2939 4133 innerTail421_7 best.bound = innerCandidates421_6 := by
  unfold candidates
  have row : edge 4133 = SearchData.row4133.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_6 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_6 best = best := by
  rw [innerTail421_6, scan, if_neg (by decide), innerCandidates421_6_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_7
    (if 3 ≤ innerCandidates421_6.length then search edge 3 [2939, 4133] innerCandidates421_6 7072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_7

def innerTail421_5 : List Nat := 4019 :: innerTail421_6
def innerCandidates421_5 : List Nat := [7211]
set_option maxRecDepth 4096 in
theorem innerCandidates421_5_checked : candidates edge 4 2939 4019 innerTail421_6 best.bound = innerCandidates421_5 := by
  unfold candidates
  have row : edge 4019 = SearchData.row4019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_5 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_5 best = best := by
  rw [innerTail421_5, scan, if_neg (by decide), innerCandidates421_5_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_6
    (if 3 ≤ innerCandidates421_5.length then search edge 3 [2939, 4019] innerCandidates421_5 6958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_6

def innerTail421_4 : List Nat := 3671 :: innerTail421_5
def innerCandidates421_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_4_checked : candidates edge 4 2939 3671 innerTail421_5 best.bound = innerCandidates421_4 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_4 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_4 best = best := by
  rw [innerTail421_4, scan, if_neg (by decide), innerCandidates421_4_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_5
    (if 3 ≤ innerCandidates421_4.length then search edge 3 [2939, 3671] innerCandidates421_4 6610 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_5

def innerTail421_3 : List Nat := 3557 :: innerTail421_4
def innerCandidates421_3 : List Nat := [5417]
set_option maxRecDepth 4096 in
theorem innerCandidates421_3_checked : candidates edge 4 2939 3557 innerTail421_4 best.bound = innerCandidates421_3 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_3 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_3 best = best := by
  rw [innerTail421_3, scan, if_neg (by decide), innerCandidates421_3_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_4
    (if 3 ≤ innerCandidates421_3.length then search edge 3 [2939, 3557] innerCandidates421_3 6496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_4

def innerTail421_2 : List Nat := 3491 :: innerTail421_3
def innerCandidates421_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_2_checked : candidates edge 4 2939 3491 innerTail421_3 best.bound = innerCandidates421_2 := by
  unfold candidates
  have row : edge 3491 = SearchData.row3491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_2 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_2 best = best := by
  rw [innerTail421_2, scan, if_neg (by decide), innerCandidates421_2_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_3
    (if 3 ≤ innerCandidates421_2.length then search edge 3 [2939, 3491] innerCandidates421_2 6430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_3

def innerTail421_1 : List Nat := 3167 :: innerTail421_2
def innerCandidates421_1 : List Nat := [4217]
set_option maxRecDepth 4096 in
theorem innerCandidates421_1_checked : candidates edge 4 2939 3167 innerTail421_2 best.bound = innerCandidates421_1 := by
  unfold candidates
  have row : edge 3167 = SearchData.row3167.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_1 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_1 best = best := by
  rw [innerTail421_1, scan, if_neg (by decide), innerCandidates421_1_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_2
    (if 3 ≤ innerCandidates421_1.length then search edge 3 [2939, 3167] innerCandidates421_1 6106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_2

def innerTail421_0 : List Nat := 2999 :: innerTail421_1
def innerCandidates421_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates421_0_checked : candidates edge 4 2939 2999 innerTail421_1 best.bound = innerCandidates421_0 := by
  unfold candidates
  have row : edge 2999 = SearchData.row2999.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan421_0 : scan edge (search edge 3) 4 [2939] 2939 innerTail421_0 best = best := by
  rw [innerTail421_0, scan, if_neg (by decide), innerCandidates421_0_checked]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_1
    (if 3 ≤ innerCandidates421_0.length then search edge 3 [2939, 2999] innerCandidates421_0 5938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan421_1

theorem rootBranch421 : search edge 4 [2939] rootCandidates421 2939 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2939] 2939 innerTail421_0 best = best
  exact innerScan421_0

def rootCandidates420 : List Nat := [3137, 3203, 3593, 3863, 5009, 5531, 5693, 6029, 6257, 6983, 8009, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates420_checked : candidates edge 5 0 2927 tail421 best.bound = rootCandidates420 := by
  unfold candidates
  have row : edge 2927 = SearchData.row2927.contains := by funext q; rfl
  rw [row]
  decide

def innerTail420_12 : List Nat := []
theorem innerScan420_12 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_12 best = best := rfl

def innerTail420_11 : List Nat := 8243 :: innerTail420_12
def innerCandidates420_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_11_checked : candidates edge 4 2927 8243 innerTail420_12 best.bound = innerCandidates420_11 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_11 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_11 best = best := by
  rw [innerTail420_11, scan, if_neg (by decide), innerCandidates420_11_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_12
    (if 3 ≤ innerCandidates420_11.length then search edge 3 [2927, 8243] innerCandidates420_11 11170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_12

def innerTail420_10 : List Nat := 8009 :: innerTail420_11
def innerCandidates420_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_10_checked : candidates edge 4 2927 8009 innerTail420_11 best.bound = innerCandidates420_10 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_10 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_10 best = best := by
  rw [innerTail420_10, scan, if_neg (by decide), innerCandidates420_10_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_11
    (if 3 ≤ innerCandidates420_10.length then search edge 3 [2927, 8009] innerCandidates420_10 10936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_11

def innerTail420_9 : List Nat := 6983 :: innerTail420_10
def innerCandidates420_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_9_checked : candidates edge 4 2927 6983 innerTail420_10 best.bound = innerCandidates420_9 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_9 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_9 best = best := by
  rw [innerTail420_9, scan, if_neg (by decide), innerCandidates420_9_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_10
    (if 3 ≤ innerCandidates420_9.length then search edge 3 [2927, 6983] innerCandidates420_9 9910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_10

def innerTail420_8 : List Nat := 6257 :: innerTail420_9
def innerCandidates420_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_8_checked : candidates edge 4 2927 6257 innerTail420_9 best.bound = innerCandidates420_8 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_8 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_8 best = best := by
  rw [innerTail420_8, scan, if_neg (by decide), innerCandidates420_8_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_9
    (if 3 ≤ innerCandidates420_8.length then search edge 3 [2927, 6257] innerCandidates420_8 9184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_9

def innerTail420_7 : List Nat := 6029 :: innerTail420_8
def innerCandidates420_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_7_checked : candidates edge 4 2927 6029 innerTail420_8 best.bound = innerCandidates420_7 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_7 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_7 best = best := by
  rw [innerTail420_7, scan, if_neg (by decide), innerCandidates420_7_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_8
    (if 3 ≤ innerCandidates420_7.length then search edge 3 [2927, 6029] innerCandidates420_7 8956 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_8

def innerTail420_6 : List Nat := 5693 :: innerTail420_7
def innerCandidates420_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_6_checked : candidates edge 4 2927 5693 innerTail420_7 best.bound = innerCandidates420_6 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_6 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_6 best = best := by
  rw [innerTail420_6, scan, if_neg (by decide), innerCandidates420_6_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_7
    (if 3 ≤ innerCandidates420_6.length then search edge 3 [2927, 5693] innerCandidates420_6 8620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_7

def innerTail420_5 : List Nat := 5531 :: innerTail420_6
def innerCandidates420_5 : List Nat := [8009]
set_option maxRecDepth 4096 in
theorem innerCandidates420_5_checked : candidates edge 4 2927 5531 innerTail420_6 best.bound = innerCandidates420_5 := by
  unfold candidates
  have row : edge 5531 = SearchData.row5531.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_5 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_5 best = best := by
  rw [innerTail420_5, scan, if_neg (by decide), innerCandidates420_5_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_6
    (if 3 ≤ innerCandidates420_5.length then search edge 3 [2927, 5531] innerCandidates420_5 8458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_6

def innerTail420_4 : List Nat := 5009 :: innerTail420_5
def innerCandidates420_4 : List Nat := [6029]
set_option maxRecDepth 4096 in
theorem innerCandidates420_4_checked : candidates edge 4 2927 5009 innerTail420_5 best.bound = innerCandidates420_4 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_4 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_4 best = best := by
  rw [innerTail420_4, scan, if_neg (by decide), innerCandidates420_4_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_5
    (if 3 ≤ innerCandidates420_4.length then search edge 3 [2927, 5009] innerCandidates420_4 7936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_5

def innerTail420_3 : List Nat := 3863 :: innerTail420_4
def innerCandidates420_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_3_checked : candidates edge 4 2927 3863 innerTail420_4 best.bound = innerCandidates420_3 := by
  unfold candidates
  have row : edge 3863 = SearchData.row3863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_3 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_3 best = best := by
  rw [innerTail420_3, scan, if_neg (by decide), innerCandidates420_3_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_4
    (if 3 ≤ innerCandidates420_3.length then search edge 3 [2927, 3863] innerCandidates420_3 6790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_4

def innerTail420_2 : List Nat := 3593 :: innerTail420_3
def innerCandidates420_2 : List Nat := [5693]
set_option maxRecDepth 4096 in
theorem innerCandidates420_2_checked : candidates edge 4 2927 3593 innerTail420_3 best.bound = innerCandidates420_2 := by
  unfold candidates
  have row : edge 3593 = SearchData.row3593.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_2 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_2 best = best := by
  rw [innerTail420_2, scan, if_neg (by decide), innerCandidates420_2_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_3
    (if 3 ≤ innerCandidates420_2.length then search edge 3 [2927, 3593] innerCandidates420_2 6520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_3

def innerTail420_1 : List Nat := 3203 :: innerTail420_2
def innerCandidates420_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates420_1_checked : candidates edge 4 2927 3203 innerTail420_2 best.bound = innerCandidates420_1 := by
  unfold candidates
  have row : edge 3203 = SearchData.row3203.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_1 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_1 best = best := by
  rw [innerTail420_1, scan, if_neg (by decide), innerCandidates420_1_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_2
    (if 3 ≤ innerCandidates420_1.length then search edge 3 [2927, 3203] innerCandidates420_1 6130 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_2

def innerTail420_0 : List Nat := 3137 :: innerTail420_1
def innerCandidates420_0 : List Nat := [3593, 5009]
set_option maxRecDepth 4096 in
theorem innerCandidates420_0_checked : candidates edge 4 2927 3137 innerTail420_1 best.bound = innerCandidates420_0 := by
  unfold candidates
  have row : edge 3137 = SearchData.row3137.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan420_0 : scan edge (search edge 3) 4 [2927] 2927 innerTail420_0 best = best := by
  rw [innerTail420_0, scan, if_neg (by decide), innerCandidates420_0_checked]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_1
    (if 3 ≤ innerCandidates420_0.length then search edge 3 [2927, 3137] innerCandidates420_0 6064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan420_1

theorem rootBranch420 : search edge 4 [2927] rootCandidates420 2927 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2927] 2927 innerTail420_0 best = best
  exact innerScan420_0

def rootCandidates419 : List Nat := [3169, 3631, 3643, 3733, 4003, 4909, 5011, 5227, 5437, 5851, 7549, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates419_checked : candidates edge 5 0 2917 tail420 best.bound = rootCandidates419 := by
  unfold candidates
  have row : edge 2917 = SearchData.row2917.contains := by funext q; rfl
  rw [row]
  decide

def innerTail419_12 : List Nat := []
theorem innerScan419_12 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_12 best = best := rfl

def innerTail419_11 : List Nat := 8179 :: innerTail419_12
def innerCandidates419_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_11_checked : candidates edge 4 2917 8179 innerTail419_12 best.bound = innerCandidates419_11 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_11 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_11 best = best := by
  rw [innerTail419_11, scan, if_neg (by decide), innerCandidates419_11_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_12
    (if 3 ≤ innerCandidates419_11.length then search edge 3 [2917, 8179] innerCandidates419_11 11096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_12

def innerTail419_10 : List Nat := 7549 :: innerTail419_11
def innerCandidates419_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_10_checked : candidates edge 4 2917 7549 innerTail419_11 best.bound = innerCandidates419_10 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_10 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_10 best = best := by
  rw [innerTail419_10, scan, if_neg (by decide), innerCandidates419_10_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_11
    (if 3 ≤ innerCandidates419_10.length then search edge 3 [2917, 7549] innerCandidates419_10 10466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_11

def innerTail419_9 : List Nat := 5851 :: innerTail419_10
def innerCandidates419_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_9_checked : candidates edge 4 2917 5851 innerTail419_10 best.bound = innerCandidates419_9 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_9 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_9 best = best := by
  rw [innerTail419_9, scan, if_neg (by decide), innerCandidates419_9_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_10
    (if 3 ≤ innerCandidates419_9.length then search edge 3 [2917, 5851] innerCandidates419_9 8768 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_10

def innerTail419_8 : List Nat := 5437 :: innerTail419_9
def innerCandidates419_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_8_checked : candidates edge 4 2917 5437 innerTail419_9 best.bound = innerCandidates419_8 := by
  unfold candidates
  have row : edge 5437 = SearchData.row5437.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_8 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_8 best = best := by
  rw [innerTail419_8, scan, if_neg (by decide), innerCandidates419_8_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_9
    (if 3 ≤ innerCandidates419_8.length then search edge 3 [2917, 5437] innerCandidates419_8 8354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_9

def innerTail419_7 : List Nat := 5227 :: innerTail419_8
def innerCandidates419_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_7_checked : candidates edge 4 2917 5227 innerTail419_8 best.bound = innerCandidates419_7 := by
  unfold candidates
  have row : edge 5227 = SearchData.row5227.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_7 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_7 best = best := by
  rw [innerTail419_7, scan, if_neg (by decide), innerCandidates419_7_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_8
    (if 3 ≤ innerCandidates419_7.length then search edge 3 [2917, 5227] innerCandidates419_7 8144 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_8

def innerTail419_6 : List Nat := 5011 :: innerTail419_7
def innerCandidates419_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_6_checked : candidates edge 4 2917 5011 innerTail419_7 best.bound = innerCandidates419_6 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_6 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_6 best = best := by
  rw [innerTail419_6, scan, if_neg (by decide), innerCandidates419_6_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_7
    (if 3 ≤ innerCandidates419_6.length then search edge 3 [2917, 5011] innerCandidates419_6 7928 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_7

def innerTail419_5 : List Nat := 4909 :: innerTail419_6
def innerCandidates419_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_5_checked : candidates edge 4 2917 4909 innerTail419_6 best.bound = innerCandidates419_5 := by
  unfold candidates
  have row : edge 4909 = SearchData.row4909.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_5 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_5 best = best := by
  rw [innerTail419_5, scan, if_neg (by decide), innerCandidates419_5_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_6
    (if 3 ≤ innerCandidates419_5.length then search edge 3 [2917, 4909] innerCandidates419_5 7826 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_6

def innerTail419_4 : List Nat := 4003 :: innerTail419_5
def innerCandidates419_4 : List Nat := [5851]
set_option maxRecDepth 4096 in
theorem innerCandidates419_4_checked : candidates edge 4 2917 4003 innerTail419_5 best.bound = innerCandidates419_4 := by
  unfold candidates
  have row : edge 4003 = SearchData.row4003.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_4 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_4 best = best := by
  rw [innerTail419_4, scan, if_neg (by decide), innerCandidates419_4_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_5
    (if 3 ≤ innerCandidates419_4.length then search edge 3 [2917, 4003] innerCandidates419_4 6920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_5

def innerTail419_3 : List Nat := 3733 :: innerTail419_4
def innerCandidates419_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_3_checked : candidates edge 4 2917 3733 innerTail419_4 best.bound = innerCandidates419_3 := by
  unfold candidates
  have row : edge 3733 = SearchData.row3733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_3 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_3 best = best := by
  rw [innerTail419_3, scan, if_neg (by decide), innerCandidates419_3_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_4
    (if 3 ≤ innerCandidates419_3.length then search edge 3 [2917, 3733] innerCandidates419_3 6650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_4

def innerTail419_2 : List Nat := 3643 :: innerTail419_3
def innerCandidates419_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_2_checked : candidates edge 4 2917 3643 innerTail419_3 best.bound = innerCandidates419_2 := by
  unfold candidates
  have row : edge 3643 = SearchData.row3643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_2 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_2 best = best := by
  rw [innerTail419_2, scan, if_neg (by decide), innerCandidates419_2_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_3
    (if 3 ≤ innerCandidates419_2.length then search edge 3 [2917, 3643] innerCandidates419_2 6560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_3

def innerTail419_1 : List Nat := 3631 :: innerTail419_2
def innerCandidates419_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates419_1_checked : candidates edge 4 2917 3631 innerTail419_2 best.bound = innerCandidates419_1 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_1 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_1 best = best := by
  rw [innerTail419_1, scan, if_neg (by decide), innerCandidates419_1_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_2
    (if 3 ≤ innerCandidates419_1.length then search edge 3 [2917, 3631] innerCandidates419_1 6548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_2

def innerTail419_0 : List Nat := 3169 :: innerTail419_1
def innerCandidates419_0 : List Nat := [3733, 8179]
set_option maxRecDepth 4096 in
theorem innerCandidates419_0_checked : candidates edge 4 2917 3169 innerTail419_1 best.bound = innerCandidates419_0 := by
  unfold candidates
  have row : edge 3169 = SearchData.row3169.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan419_0 : scan edge (search edge 3) 4 [2917] 2917 innerTail419_0 best = best := by
  rw [innerTail419_0, scan, if_neg (by decide), innerCandidates419_0_checked]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_1
    (if 3 ≤ innerCandidates419_0.length then search edge 3 [2917, 3169] innerCandidates419_0 6086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan419_1

theorem rootBranch419 : search edge 4 [2917] rootCandidates419 2917 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2917] 2917 innerTail419_0 best = best
  exact innerScan419_0

def rootCandidates418 : List Nat := [3089, 3557, 4217, 4253, 5153, 5813, 6269, 6323, 6737, 6983, 7451]
set_option maxRecDepth 4096 in
theorem rootCandidates418_checked : candidates edge 5 0 2909 tail419 best.bound = rootCandidates418 := by
  unfold candidates
  have row : edge 2909 = SearchData.row2909.contains := by funext q; rfl
  rw [row]
  decide

def innerTail418_11 : List Nat := []
theorem innerScan418_11 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_11 best = best := rfl

def innerTail418_10 : List Nat := 7451 :: innerTail418_11
def innerCandidates418_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_10_checked : candidates edge 4 2909 7451 innerTail418_11 best.bound = innerCandidates418_10 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_10 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_10 best = best := by
  rw [innerTail418_10, scan, if_neg (by decide), innerCandidates418_10_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_11
    (if 3 ≤ innerCandidates418_10.length then search edge 3 [2909, 7451] innerCandidates418_10 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_11

def innerTail418_9 : List Nat := 6983 :: innerTail418_10
def innerCandidates418_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_9_checked : candidates edge 4 2909 6983 innerTail418_10 best.bound = innerCandidates418_9 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_9 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_9 best = best := by
  rw [innerTail418_9, scan, if_neg (by decide), innerCandidates418_9_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_10
    (if 3 ≤ innerCandidates418_9.length then search edge 3 [2909, 6983] innerCandidates418_9 9892 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_10

def innerTail418_8 : List Nat := 6737 :: innerTail418_9
def innerCandidates418_8 : List Nat := [7451]
set_option maxRecDepth 4096 in
theorem innerCandidates418_8_checked : candidates edge 4 2909 6737 innerTail418_9 best.bound = innerCandidates418_8 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_8 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_8 best = best := by
  rw [innerTail418_8, scan, if_neg (by decide), innerCandidates418_8_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_9
    (if 3 ≤ innerCandidates418_8.length then search edge 3 [2909, 6737] innerCandidates418_8 9646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_9

def innerTail418_7 : List Nat := 6323 :: innerTail418_8
def innerCandidates418_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_7_checked : candidates edge 4 2909 6323 innerTail418_8 best.bound = innerCandidates418_7 := by
  unfold candidates
  have row : edge 6323 = SearchData.row6323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_7 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_7 best = best := by
  rw [innerTail418_7, scan, if_neg (by decide), innerCandidates418_7_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_8
    (if 3 ≤ innerCandidates418_7.length then search edge 3 [2909, 6323] innerCandidates418_7 9232 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_8

def innerTail418_6 : List Nat := 6269 :: innerTail418_7
def innerCandidates418_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_6_checked : candidates edge 4 2909 6269 innerTail418_7 best.bound = innerCandidates418_6 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_6 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_6 best = best := by
  rw [innerTail418_6, scan, if_neg (by decide), innerCandidates418_6_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_7
    (if 3 ≤ innerCandidates418_6.length then search edge 3 [2909, 6269] innerCandidates418_6 9178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_7

def innerTail418_5 : List Nat := 5813 :: innerTail418_6
def innerCandidates418_5 : List Nat := [6737]
set_option maxRecDepth 4096 in
theorem innerCandidates418_5_checked : candidates edge 4 2909 5813 innerTail418_6 best.bound = innerCandidates418_5 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_5 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_5 best = best := by
  rw [innerTail418_5, scan, if_neg (by decide), innerCandidates418_5_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_6
    (if 3 ≤ innerCandidates418_5.length then search edge 3 [2909, 5813] innerCandidates418_5 8722 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_6

def innerTail418_4 : List Nat := 5153 :: innerTail418_5
def innerCandidates418_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_4_checked : candidates edge 4 2909 5153 innerTail418_5 best.bound = innerCandidates418_4 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_4 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_4 best = best := by
  rw [innerTail418_4, scan, if_neg (by decide), innerCandidates418_4_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_5
    (if 3 ≤ innerCandidates418_4.length then search edge 3 [2909, 5153] innerCandidates418_4 8062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_5

def innerTail418_3 : List Nat := 4253 :: innerTail418_4
def innerCandidates418_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_3_checked : candidates edge 4 2909 4253 innerTail418_4 best.bound = innerCandidates418_3 := by
  unfold candidates
  have row : edge 4253 = SearchData.row4253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_3 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_3 best = best := by
  rw [innerTail418_3, scan, if_neg (by decide), innerCandidates418_3_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_4
    (if 3 ≤ innerCandidates418_3.length then search edge 3 [2909, 4253] innerCandidates418_3 7162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_4

def innerTail418_2 : List Nat := 4217 :: innerTail418_3
def innerCandidates418_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_2_checked : candidates edge 4 2909 4217 innerTail418_3 best.bound = innerCandidates418_2 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_2 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_2 best = best := by
  rw [innerTail418_2, scan, if_neg (by decide), innerCandidates418_2_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_3
    (if 3 ≤ innerCandidates418_2.length then search edge 3 [2909, 4217] innerCandidates418_2 7126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_3

def innerTail418_1 : List Nat := 3557 :: innerTail418_2
def innerCandidates418_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_1_checked : candidates edge 4 2909 3557 innerTail418_2 best.bound = innerCandidates418_1 := by
  unfold candidates
  have row : edge 3557 = SearchData.row3557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_1 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_1 best = best := by
  rw [innerTail418_1, scan, if_neg (by decide), innerCandidates418_1_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_2
    (if 3 ≤ innerCandidates418_1.length then search edge 3 [2909, 3557] innerCandidates418_1 6466 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_2

def innerTail418_0 : List Nat := 3089 :: innerTail418_1
def innerCandidates418_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates418_0_checked : candidates edge 4 2909 3089 innerTail418_1 best.bound = innerCandidates418_0 := by
  unfold candidates
  have row : edge 3089 = SearchData.row3089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan418_0 : scan edge (search edge 3) 4 [2909] 2909 innerTail418_0 best = best := by
  rw [innerTail418_0, scan, if_neg (by decide), innerCandidates418_0_checked]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_1
    (if 3 ≤ innerCandidates418_0.length then search edge 3 [2909, 3089] innerCandidates418_0 5998 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan418_1

theorem rootBranch418 : search edge 4 [2909] rootCandidates418 2909 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2909] 2909 innerTail418_0 best = best
  exact innerScan418_0

def rootCandidates417 : List Nat := [2963, 4373, 4409, 4637, 4721, 5927, 6131, 6701, 6779, 7103, 7121, 7433, 7457, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates417_checked : candidates edge 5 0 2903 tail418 best.bound = rootCandidates417 := by
  unfold candidates
  have row : edge 2903 = SearchData.row2903.contains := by funext q; rfl
  rw [row]
  decide

def innerTail417_14 : List Nat := []
theorem innerScan417_14 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_14 best = best := rfl

def innerTail417_13 : List Nat := 8123 :: innerTail417_14
def innerCandidates417_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_13_checked : candidates edge 4 2903 8123 innerTail417_14 best.bound = innerCandidates417_13 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_13 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_13 best = best := by
  rw [innerTail417_13, scan, if_neg (by decide), innerCandidates417_13_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_14
    (if 3 ≤ innerCandidates417_13.length then search edge 3 [2903, 8123] innerCandidates417_13 11026 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_14

def innerTail417_12 : List Nat := 7457 :: innerTail417_13
def innerCandidates417_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_12_checked : candidates edge 4 2903 7457 innerTail417_13 best.bound = innerCandidates417_12 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_12 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_12 best = best := by
  rw [innerTail417_12, scan, if_neg (by decide), innerCandidates417_12_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_13
    (if 3 ≤ innerCandidates417_12.length then search edge 3 [2903, 7457] innerCandidates417_12 10360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_13

def innerTail417_11 : List Nat := 7433 :: innerTail417_12
def innerCandidates417_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_11_checked : candidates edge 4 2903 7433 innerTail417_12 best.bound = innerCandidates417_11 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_11 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_11 best = best := by
  rw [innerTail417_11, scan, if_neg (by decide), innerCandidates417_11_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_12
    (if 3 ≤ innerCandidates417_11.length then search edge 3 [2903, 7433] innerCandidates417_11 10336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_12

def innerTail417_10 : List Nat := 7121 :: innerTail417_11
def innerCandidates417_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_10_checked : candidates edge 4 2903 7121 innerTail417_11 best.bound = innerCandidates417_10 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_10 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_10 best = best := by
  rw [innerTail417_10, scan, if_neg (by decide), innerCandidates417_10_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_11
    (if 3 ≤ innerCandidates417_10.length then search edge 3 [2903, 7121] innerCandidates417_10 10024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_11

def innerTail417_9 : List Nat := 7103 :: innerTail417_10
def innerCandidates417_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_9_checked : candidates edge 4 2903 7103 innerTail417_10 best.bound = innerCandidates417_9 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_9 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_9 best = best := by
  rw [innerTail417_9, scan, if_neg (by decide), innerCandidates417_9_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_10
    (if 3 ≤ innerCandidates417_9.length then search edge 3 [2903, 7103] innerCandidates417_9 10006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_10

def innerTail417_8 : List Nat := 6779 :: innerTail417_9
def innerCandidates417_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_8_checked : candidates edge 4 2903 6779 innerTail417_9 best.bound = innerCandidates417_8 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_8 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_8 best = best := by
  rw [innerTail417_8, scan, if_neg (by decide), innerCandidates417_8_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_9
    (if 3 ≤ innerCandidates417_8.length then search edge 3 [2903, 6779] innerCandidates417_8 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_9

def innerTail417_7 : List Nat := 6701 :: innerTail417_8
def innerCandidates417_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_7_checked : candidates edge 4 2903 6701 innerTail417_8 best.bound = innerCandidates417_7 := by
  unfold candidates
  have row : edge 6701 = SearchData.row6701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_7 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_7 best = best := by
  rw [innerTail417_7, scan, if_neg (by decide), innerCandidates417_7_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_8
    (if 3 ≤ innerCandidates417_7.length then search edge 3 [2903, 6701] innerCandidates417_7 9604 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_8

def innerTail417_6 : List Nat := 6131 :: innerTail417_7
def innerCandidates417_6 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates417_6_checked : candidates edge 4 2903 6131 innerTail417_7 best.bound = innerCandidates417_6 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_6 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_6 best = best := by
  rw [innerTail417_6, scan, if_neg (by decide), innerCandidates417_6_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_7
    (if 3 ≤ innerCandidates417_6.length then search edge 3 [2903, 6131] innerCandidates417_6 9034 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_7

def innerTail417_5 : List Nat := 5927 :: innerTail417_6
def innerCandidates417_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_5_checked : candidates edge 4 2903 5927 innerTail417_6 best.bound = innerCandidates417_5 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_5 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_5 best = best := by
  rw [innerTail417_5, scan, if_neg (by decide), innerCandidates417_5_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_6
    (if 3 ≤ innerCandidates417_5.length then search edge 3 [2903, 5927] innerCandidates417_5 8830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_6

def innerTail417_4 : List Nat := 4721 :: innerTail417_5
def innerCandidates417_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_4_checked : candidates edge 4 2903 4721 innerTail417_5 best.bound = innerCandidates417_4 := by
  unfold candidates
  have row : edge 4721 = SearchData.row4721.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_4 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_4 best = best := by
  rw [innerTail417_4, scan, if_neg (by decide), innerCandidates417_4_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_5
    (if 3 ≤ innerCandidates417_4.length then search edge 3 [2903, 4721] innerCandidates417_4 7624 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_5

def innerTail417_3 : List Nat := 4637 :: innerTail417_4
def innerCandidates417_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_3_checked : candidates edge 4 2903 4637 innerTail417_4 best.bound = innerCandidates417_3 := by
  unfold candidates
  have row : edge 4637 = SearchData.row4637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_3 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_3 best = best := by
  rw [innerTail417_3, scan, if_neg (by decide), innerCandidates417_3_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_4
    (if 3 ≤ innerCandidates417_3.length then search edge 3 [2903, 4637] innerCandidates417_3 7540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_4

def innerTail417_2 : List Nat := 4409 :: innerTail417_3
def innerCandidates417_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_2_checked : candidates edge 4 2903 4409 innerTail417_3 best.bound = innerCandidates417_2 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_2 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_2 best = best := by
  rw [innerTail417_2, scan, if_neg (by decide), innerCandidates417_2_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_3
    (if 3 ≤ innerCandidates417_2.length then search edge 3 [2903, 4409] innerCandidates417_2 7312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_3

def innerTail417_1 : List Nat := 4373 :: innerTail417_2
def innerCandidates417_1 : List Nat := [5927, 7121]
set_option maxRecDepth 4096 in
theorem innerCandidates417_1_checked : candidates edge 4 2903 4373 innerTail417_2 best.bound = innerCandidates417_1 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_1 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_1 best = best := by
  rw [innerTail417_1, scan, if_neg (by decide), innerCandidates417_1_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_2
    (if 3 ≤ innerCandidates417_1.length then search edge 3 [2903, 4373] innerCandidates417_1 7276 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_2

def innerTail417_0 : List Nat := 2963 :: innerTail417_1
def innerCandidates417_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates417_0_checked : candidates edge 4 2903 2963 innerTail417_1 best.bound = innerCandidates417_0 := by
  unfold candidates
  have row : edge 2963 = SearchData.row2963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan417_0 : scan edge (search edge 3) 4 [2903] 2903 innerTail417_0 best = best := by
  rw [innerTail417_0, scan, if_neg (by decide), innerCandidates417_0_checked]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_1
    (if 3 ≤ innerCandidates417_0.length then search edge 3 [2903, 2963] innerCandidates417_0 5866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan417_1

theorem rootBranch417 : search edge 4 [2903] rootCandidates417 2903 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [2903] 2903 innerTail417_0 best = best
  exact innerScan417_0

end ProjectEulerSolutions.PrimePairUpper.Search
