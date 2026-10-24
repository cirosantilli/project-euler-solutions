import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates587 : List Nat := [4441, 4483, 4507, 5521, 5689, 6343, 6607, 6781, 7213, 8233, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates587_checked : candidates edge 5 0 4297 tail588 best.bound = rootCandidates587 := by
  unfold candidates
  have row : edge 4297 = SearchData.row4297.contains := by funext q; rfl
  rw [row]
  decide

def innerTail587_11 : List Nat := []
theorem innerScan587_11 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_11 best = best := rfl

def innerTail587_10 : List Nat := 8317 :: innerTail587_11
def innerCandidates587_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_10_checked : candidates edge 4 4297 8317 innerTail587_11 best.bound = innerCandidates587_10 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_10 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_10 best = best := by
  rw [innerTail587_10, scan, if_neg (by decide), innerCandidates587_10_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_11
    (if 3 ≤ innerCandidates587_10.length then search edge 3 [4297, 8317] innerCandidates587_10 12614 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_11

def innerTail587_9 : List Nat := 8233 :: innerTail587_10
def innerCandidates587_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_9_checked : candidates edge 4 4297 8233 innerTail587_10 best.bound = innerCandidates587_9 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_9 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_9 best = best := by
  rw [innerTail587_9, scan, if_neg (by decide), innerCandidates587_9_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_10
    (if 3 ≤ innerCandidates587_9.length then search edge 3 [4297, 8233] innerCandidates587_9 12530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_10

def innerTail587_8 : List Nat := 7213 :: innerTail587_9
def innerCandidates587_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_8_checked : candidates edge 4 4297 7213 innerTail587_9 best.bound = innerCandidates587_8 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_8 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_8 best = best := by
  rw [innerTail587_8, scan, if_neg (by decide), innerCandidates587_8_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_9
    (if 3 ≤ innerCandidates587_8.length then search edge 3 [4297, 7213] innerCandidates587_8 11510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_9

def innerTail587_7 : List Nat := 6781 :: innerTail587_8
def innerCandidates587_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_7_checked : candidates edge 4 4297 6781 innerTail587_8 best.bound = innerCandidates587_7 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_7 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_7 best = best := by
  rw [innerTail587_7, scan, if_neg (by decide), innerCandidates587_7_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_8
    (if 3 ≤ innerCandidates587_7.length then search edge 3 [4297, 6781] innerCandidates587_7 11078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_8

def innerTail587_6 : List Nat := 6607 :: innerTail587_7
def innerCandidates587_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_6_checked : candidates edge 4 4297 6607 innerTail587_7 best.bound = innerCandidates587_6 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_6 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_6 best = best := by
  rw [innerTail587_6, scan, if_neg (by decide), innerCandidates587_6_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_7
    (if 3 ≤ innerCandidates587_6.length then search edge 3 [4297, 6607] innerCandidates587_6 10904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_7

def innerTail587_5 : List Nat := 6343 :: innerTail587_6
def innerCandidates587_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_5_checked : candidates edge 4 4297 6343 innerTail587_6 best.bound = innerCandidates587_5 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_5 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_5 best = best := by
  rw [innerTail587_5, scan, if_neg (by decide), innerCandidates587_5_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_6
    (if 3 ≤ innerCandidates587_5.length then search edge 3 [4297, 6343] innerCandidates587_5 10640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_6

def innerTail587_4 : List Nat := 5689 :: innerTail587_5
def innerCandidates587_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_4_checked : candidates edge 4 4297 5689 innerTail587_5 best.bound = innerCandidates587_4 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_4 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_4 best = best := by
  rw [innerTail587_4, scan, if_neg (by decide), innerCandidates587_4_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_5
    (if 3 ≤ innerCandidates587_4.length then search edge 3 [4297, 5689] innerCandidates587_4 9986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_5

def innerTail587_3 : List Nat := 5521 :: innerTail587_4
def innerCandidates587_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_3_checked : candidates edge 4 4297 5521 innerTail587_4 best.bound = innerCandidates587_3 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_3 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_3 best = best := by
  rw [innerTail587_3, scan, if_neg (by decide), innerCandidates587_3_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_4
    (if 3 ≤ innerCandidates587_3.length then search edge 3 [4297, 5521] innerCandidates587_3 9818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_4

def innerTail587_2 : List Nat := 4507 :: innerTail587_3
def innerCandidates587_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_2_checked : candidates edge 4 4297 4507 innerTail587_3 best.bound = innerCandidates587_2 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_2 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_2 best = best := by
  rw [innerTail587_2, scan, if_neg (by decide), innerCandidates587_2_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_3
    (if 3 ≤ innerCandidates587_2.length then search edge 3 [4297, 4507] innerCandidates587_2 8804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_3

def innerTail587_1 : List Nat := 4483 :: innerTail587_2
def innerCandidates587_1 : List Nat := [6607]
set_option maxRecDepth 4096 in
theorem innerCandidates587_1_checked : candidates edge 4 4297 4483 innerTail587_2 best.bound = innerCandidates587_1 := by
  unfold candidates
  have row : edge 4483 = SearchData.row4483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_1 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_1 best = best := by
  rw [innerTail587_1, scan, if_neg (by decide), innerCandidates587_1_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_2
    (if 3 ≤ innerCandidates587_1.length then search edge 3 [4297, 4483] innerCandidates587_1 8780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_2

def innerTail587_0 : List Nat := 4441 :: innerTail587_1
def innerCandidates587_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates587_0_checked : candidates edge 4 4297 4441 innerTail587_1 best.bound = innerCandidates587_0 := by
  unfold candidates
  have row : edge 4441 = SearchData.row4441.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan587_0 : scan edge (search edge 3) 4 [4297] 4297 innerTail587_0 best = best := by
  rw [innerTail587_0, scan, if_neg (by decide), innerCandidates587_0_checked]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_1
    (if 3 ≤ innerCandidates587_0.length then search edge 3 [4297, 4441] innerCandidates587_0 8738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan587_1

theorem rootBranch587 : search edge 4 [4297] rootCandidates587 4297 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4297] 4297 innerTail587_0 best = best
  exact innerScan587_0

def rootCandidates586 : List Nat := [4391, 4679, 4889, 5507, 5519, 6317, 6473, 6869, 7331, 7937, 8087, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates586_checked : candidates edge 5 0 4289 tail587 best.bound = rootCandidates586 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext q; rfl
  rw [row]
  decide

def innerTail586_12 : List Nat := []
theorem innerScan586_12 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_12 best = best := rfl

def innerTail586_11 : List Nat := 8171 :: innerTail586_12
def innerCandidates586_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_11_checked : candidates edge 4 4289 8171 innerTail586_12 best.bound = innerCandidates586_11 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_11 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_11 best = best := by
  rw [innerTail586_11, scan, if_neg (by decide), innerCandidates586_11_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_12
    (if 3 ≤ innerCandidates586_11.length then search edge 3 [4289, 8171] innerCandidates586_11 12460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_12

def innerTail586_10 : List Nat := 8087 :: innerTail586_11
def innerCandidates586_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_10_checked : candidates edge 4 4289 8087 innerTail586_11 best.bound = innerCandidates586_10 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_10 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_10 best = best := by
  rw [innerTail586_10, scan, if_neg (by decide), innerCandidates586_10_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_11
    (if 3 ≤ innerCandidates586_10.length then search edge 3 [4289, 8087] innerCandidates586_10 12376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_11

def innerTail586_9 : List Nat := 7937 :: innerTail586_10
def innerCandidates586_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_9_checked : candidates edge 4 4289 7937 innerTail586_10 best.bound = innerCandidates586_9 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_9 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_9 best = best := by
  rw [innerTail586_9, scan, if_neg (by decide), innerCandidates586_9_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_10
    (if 3 ≤ innerCandidates586_9.length then search edge 3 [4289, 7937] innerCandidates586_9 12226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_10

def innerTail586_8 : List Nat := 7331 :: innerTail586_9
def innerCandidates586_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_8_checked : candidates edge 4 4289 7331 innerTail586_9 best.bound = innerCandidates586_8 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_8 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_8 best = best := by
  rw [innerTail586_8, scan, if_neg (by decide), innerCandidates586_8_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_9
    (if 3 ≤ innerCandidates586_8.length then search edge 3 [4289, 7331] innerCandidates586_8 11620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_9

def innerTail586_7 : List Nat := 6869 :: innerTail586_8
def innerCandidates586_7 : List Nat := [8087]
set_option maxRecDepth 4096 in
theorem innerCandidates586_7_checked : candidates edge 4 4289 6869 innerTail586_8 best.bound = innerCandidates586_7 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_7 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_7 best = best := by
  rw [innerTail586_7, scan, if_neg (by decide), innerCandidates586_7_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_8
    (if 3 ≤ innerCandidates586_7.length then search edge 3 [4289, 6869] innerCandidates586_7 11158 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_8

def innerTail586_6 : List Nat := 6473 :: innerTail586_7
def innerCandidates586_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_6_checked : candidates edge 4 4289 6473 innerTail586_7 best.bound = innerCandidates586_6 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_6 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_6 best = best := by
  rw [innerTail586_6, scan, if_neg (by decide), innerCandidates586_6_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_7
    (if 3 ≤ innerCandidates586_6.length then search edge 3 [4289, 6473] innerCandidates586_6 10762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_7

def innerTail586_5 : List Nat := 6317 :: innerTail586_6
def innerCandidates586_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_5_checked : candidates edge 4 4289 6317 innerTail586_6 best.bound = innerCandidates586_5 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_5 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_5 best = best := by
  rw [innerTail586_5, scan, if_neg (by decide), innerCandidates586_5_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_6
    (if 3 ≤ innerCandidates586_5.length then search edge 3 [4289, 6317] innerCandidates586_5 10606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_6

def innerTail586_4 : List Nat := 5519 :: innerTail586_5
def innerCandidates586_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_4_checked : candidates edge 4 4289 5519 innerTail586_5 best.bound = innerCandidates586_4 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_4 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_4 best = best := by
  rw [innerTail586_4, scan, if_neg (by decide), innerCandidates586_4_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_5
    (if 3 ≤ innerCandidates586_4.length then search edge 3 [4289, 5519] innerCandidates586_4 9808 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_5

def innerTail586_3 : List Nat := 5507 :: innerTail586_4
def innerCandidates586_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_3_checked : candidates edge 4 4289 5507 innerTail586_4 best.bound = innerCandidates586_3 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_3 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_3 best = best := by
  rw [innerTail586_3, scan, if_neg (by decide), innerCandidates586_3_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_4
    (if 3 ≤ innerCandidates586_3.length then search edge 3 [4289, 5507] innerCandidates586_3 9796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_4

def innerTail586_2 : List Nat := 4889 :: innerTail586_3
def innerCandidates586_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates586_2_checked : candidates edge 4 4289 4889 innerTail586_3 best.bound = innerCandidates586_2 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_2 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_2 best = best := by
  rw [innerTail586_2, scan, if_neg (by decide), innerCandidates586_2_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_3
    (if 3 ≤ innerCandidates586_2.length then search edge 3 [4289, 4889] innerCandidates586_2 9178 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_3

def innerTail586_1 : List Nat := 4679 :: innerTail586_2
def innerCandidates586_1 : List Nat := [4889]
set_option maxRecDepth 4096 in
theorem innerCandidates586_1_checked : candidates edge 4 4289 4679 innerTail586_2 best.bound = innerCandidates586_1 := by
  unfold candidates
  have row : edge 4679 = SearchData.row4679.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_1 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_1 best = best := by
  rw [innerTail586_1, scan, if_neg (by decide), innerCandidates586_1_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_2
    (if 3 ≤ innerCandidates586_1.length then search edge 3 [4289, 4679] innerCandidates586_1 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_2

def innerTail586_0 : List Nat := 4391 :: innerTail586_1
def innerCandidates586_0 : List Nat := [6869]
set_option maxRecDepth 4096 in
theorem innerCandidates586_0_checked : candidates edge 4 4289 4391 innerTail586_1 best.bound = innerCandidates586_0 := by
  unfold candidates
  have row : edge 4391 = SearchData.row4391.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan586_0 : scan edge (search edge 3) 4 [4289] 4289 innerTail586_0 best = best := by
  rw [innerTail586_0, scan, if_neg (by decide), innerCandidates586_0_checked]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_1
    (if 3 ≤ innerCandidates586_0.length then search edge 3 [4289, 4391] innerCandidates586_0 8680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan586_1

theorem rootBranch586 : search edge 4 [4289] rootCandidates586 4289 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4289] 4289 innerTail586_0 best = best
  exact innerScan586_0

def rootCandidates585 : List Nat := [4289, 4523, 5669, 7103, 8069, 8093, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates585_checked : candidates edge 5 0 4283 tail586 best.bound = rootCandidates585 := by
  unfold candidates
  have row : edge 4283 = SearchData.row4283.contains := by funext q; rfl
  rw [row]
  decide

def innerTail585_7 : List Nat := []
theorem innerScan585_7 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_7 best = best := rfl

def innerTail585_6 : List Nat := 8273 :: innerTail585_7
def innerCandidates585_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates585_6_checked : candidates edge 4 4283 8273 innerTail585_7 best.bound = innerCandidates585_6 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan585_6 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_6 best = best := by
  rw [innerTail585_6, scan, if_neg (by decide), innerCandidates585_6_checked]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_7
    (if 3 ≤ innerCandidates585_6.length then search edge 3 [4283, 8273] innerCandidates585_6 12556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan585_7

def innerTail585_5 : List Nat := 8093 :: innerTail585_6
def innerCandidates585_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates585_5_checked : candidates edge 4 4283 8093 innerTail585_6 best.bound = innerCandidates585_5 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan585_5 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_5 best = best := by
  rw [innerTail585_5, scan, if_neg (by decide), innerCandidates585_5_checked]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_6
    (if 3 ≤ innerCandidates585_5.length then search edge 3 [4283, 8093] innerCandidates585_5 12376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan585_6

def innerTail585_4 : List Nat := 8069 :: innerTail585_5
def innerCandidates585_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates585_4_checked : candidates edge 4 4283 8069 innerTail585_5 best.bound = innerCandidates585_4 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan585_4 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_4 best = best := by
  rw [innerTail585_4, scan, if_neg (by decide), innerCandidates585_4_checked]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_5
    (if 3 ≤ innerCandidates585_4.length then search edge 3 [4283, 8069] innerCandidates585_4 12352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan585_5

def innerTail585_3 : List Nat := 7103 :: innerTail585_4
def innerCandidates585_3 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates585_3_checked : candidates edge 4 4283 7103 innerTail585_4 best.bound = innerCandidates585_3 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan585_3 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_3 best = best := by
  rw [innerTail585_3, scan, if_neg (by decide), innerCandidates585_3_checked]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_4
    (if 3 ≤ innerCandidates585_3.length then search edge 3 [4283, 7103] innerCandidates585_3 11386 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan585_4

def innerTail585_2 : List Nat := 5669 :: innerTail585_3
def innerCandidates585_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates585_2_checked : candidates edge 4 4283 5669 innerTail585_3 best.bound = innerCandidates585_2 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan585_2 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_2 best = best := by
  rw [innerTail585_2, scan, if_neg (by decide), innerCandidates585_2_checked]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_3
    (if 3 ≤ innerCandidates585_2.length then search edge 3 [4283, 5669] innerCandidates585_2 9952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan585_3

def innerTail585_1 : List Nat := 4523 :: innerTail585_2
def innerCandidates585_1 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates585_1_checked : candidates edge 4 4283 4523 innerTail585_2 best.bound = innerCandidates585_1 := by
  unfold candidates
  have row : edge 4523 = SearchData.row4523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan585_1 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_1 best = best := by
  rw [innerTail585_1, scan, if_neg (by decide), innerCandidates585_1_checked]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_2
    (if 3 ≤ innerCandidates585_1.length then search edge 3 [4283, 4523] innerCandidates585_1 8806 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan585_2

def innerTail585_0 : List Nat := 4289 :: innerTail585_1
def innerCandidates585_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates585_0_checked : candidates edge 4 4283 4289 innerTail585_1 best.bound = innerCandidates585_0 := by
  unfold candidates
  have row : edge 4289 = SearchData.row4289.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan585_0 : scan edge (search edge 3) 4 [4283] 4283 innerTail585_0 best = best := by
  rw [innerTail585_0, scan, if_neg (by decide), innerCandidates585_0_checked]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_1
    (if 3 ≤ innerCandidates585_0.length then search edge 3 [4283, 4289] innerCandidates585_0 8572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan585_1

theorem rootBranch585 : search edge 4 [4283] rootCandidates585 4283 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4283] 4283 innerTail585_0 best = best
  exact innerScan585_0

def rootCandidates584 : List Nat := [4567, 4723, 4969, 5023, 6247, 6373, 6703, 7207, 7297, 7489, 7549, 7573, 7951, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates584_checked : candidates edge 5 0 4273 tail585 best.bound = rootCandidates584 := by
  unfold candidates
  have row : edge 4273 = SearchData.row4273.contains := by funext q; rfl
  rw [row]
  decide

def innerTail584_14 : List Nat := []
theorem innerScan584_14 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_14 best = best := rfl

def innerTail584_13 : List Nat := 8287 :: innerTail584_14
def innerCandidates584_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_13_checked : candidates edge 4 4273 8287 innerTail584_14 best.bound = innerCandidates584_13 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_13 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_13 best = best := by
  rw [innerTail584_13, scan, if_neg (by decide), innerCandidates584_13_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_14
    (if 3 ≤ innerCandidates584_13.length then search edge 3 [4273, 8287] innerCandidates584_13 12560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_14

def innerTail584_12 : List Nat := 7951 :: innerTail584_13
def innerCandidates584_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_12_checked : candidates edge 4 4273 7951 innerTail584_13 best.bound = innerCandidates584_12 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_12 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_12 best = best := by
  rw [innerTail584_12, scan, if_neg (by decide), innerCandidates584_12_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_13
    (if 3 ≤ innerCandidates584_12.length then search edge 3 [4273, 7951] innerCandidates584_12 12224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_13

def innerTail584_11 : List Nat := 7573 :: innerTail584_12
def innerCandidates584_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_11_checked : candidates edge 4 4273 7573 innerTail584_12 best.bound = innerCandidates584_11 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_11 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_11 best = best := by
  rw [innerTail584_11, scan, if_neg (by decide), innerCandidates584_11_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_12
    (if 3 ≤ innerCandidates584_11.length then search edge 3 [4273, 7573] innerCandidates584_11 11846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_12

def innerTail584_10 : List Nat := 7549 :: innerTail584_11
def innerCandidates584_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_10_checked : candidates edge 4 4273 7549 innerTail584_11 best.bound = innerCandidates584_10 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_10 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_10 best = best := by
  rw [innerTail584_10, scan, if_neg (by decide), innerCandidates584_10_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_11
    (if 3 ≤ innerCandidates584_10.length then search edge 3 [4273, 7549] innerCandidates584_10 11822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_11

def innerTail584_9 : List Nat := 7489 :: innerTail584_10
def innerCandidates584_9 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates584_9_checked : candidates edge 4 4273 7489 innerTail584_10 best.bound = innerCandidates584_9 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_9 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_9 best = best := by
  rw [innerTail584_9, scan, if_neg (by decide), innerCandidates584_9_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_10
    (if 3 ≤ innerCandidates584_9.length then search edge 3 [4273, 7489] innerCandidates584_9 11762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_10

def innerTail584_8 : List Nat := 7297 :: innerTail584_9
def innerCandidates584_8 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates584_8_checked : candidates edge 4 4273 7297 innerTail584_9 best.bound = innerCandidates584_8 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_8 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_8 best = best := by
  rw [innerTail584_8, scan, if_neg (by decide), innerCandidates584_8_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_9
    (if 3 ≤ innerCandidates584_8.length then search edge 3 [4273, 7297] innerCandidates584_8 11570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_9

def innerTail584_7 : List Nat := 7207 :: innerTail584_8
def innerCandidates584_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_7_checked : candidates edge 4 4273 7207 innerTail584_8 best.bound = innerCandidates584_7 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_7 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_7 best = best := by
  rw [innerTail584_7, scan, if_neg (by decide), innerCandidates584_7_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_8
    (if 3 ≤ innerCandidates584_7.length then search edge 3 [4273, 7207] innerCandidates584_7 11480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_8

def innerTail584_6 : List Nat := 6703 :: innerTail584_7
def innerCandidates584_6 : List Nat := [7297]
set_option maxRecDepth 4096 in
theorem innerCandidates584_6_checked : candidates edge 4 4273 6703 innerTail584_7 best.bound = innerCandidates584_6 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_6 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_6 best = best := by
  rw [innerTail584_6, scan, if_neg (by decide), innerCandidates584_6_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_7
    (if 3 ≤ innerCandidates584_6.length then search edge 3 [4273, 6703] innerCandidates584_6 10976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_7

def innerTail584_5 : List Nat := 6373 :: innerTail584_6
def innerCandidates584_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_5_checked : candidates edge 4 4273 6373 innerTail584_6 best.bound = innerCandidates584_5 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_5 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_5 best = best := by
  rw [innerTail584_5, scan, if_neg (by decide), innerCandidates584_5_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_6
    (if 3 ≤ innerCandidates584_5.length then search edge 3 [4273, 6373] innerCandidates584_5 10646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_6

def innerTail584_4 : List Nat := 6247 :: innerTail584_5
def innerCandidates584_4 : List Nat := [7549]
set_option maxRecDepth 4096 in
theorem innerCandidates584_4_checked : candidates edge 4 4273 6247 innerTail584_5 best.bound = innerCandidates584_4 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_4 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_4 best = best := by
  rw [innerTail584_4, scan, if_neg (by decide), innerCandidates584_4_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_5
    (if 3 ≤ innerCandidates584_4.length then search edge 3 [4273, 6247] innerCandidates584_4 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_5

def innerTail584_3 : List Nat := 5023 :: innerTail584_4
def innerCandidates584_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_3_checked : candidates edge 4 4273 5023 innerTail584_4 best.bound = innerCandidates584_3 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_3 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_3 best = best := by
  rw [innerTail584_3, scan, if_neg (by decide), innerCandidates584_3_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_4
    (if 3 ≤ innerCandidates584_3.length then search edge 3 [4273, 5023] innerCandidates584_3 9296 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_4

def innerTail584_2 : List Nat := 4969 :: innerTail584_3
def innerCandidates584_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_2_checked : candidates edge 4 4273 4969 innerTail584_3 best.bound = innerCandidates584_2 := by
  unfold candidates
  have row : edge 4969 = SearchData.row4969.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_2 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_2 best = best := by
  rw [innerTail584_2, scan, if_neg (by decide), innerCandidates584_2_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_3
    (if 3 ≤ innerCandidates584_2.length then search edge 3 [4273, 4969] innerCandidates584_2 9242 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_3

def innerTail584_1 : List Nat := 4723 :: innerTail584_2
def innerCandidates584_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_1_checked : candidates edge 4 4273 4723 innerTail584_2 best.bound = innerCandidates584_1 := by
  unfold candidates
  have row : edge 4723 = SearchData.row4723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_1 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_1 best = best := by
  rw [innerTail584_1, scan, if_neg (by decide), innerCandidates584_1_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_2
    (if 3 ≤ innerCandidates584_1.length then search edge 3 [4273, 4723] innerCandidates584_1 8996 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_2

def innerTail584_0 : List Nat := 4567 :: innerTail584_1
def innerCandidates584_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates584_0_checked : candidates edge 4 4273 4567 innerTail584_1 best.bound = innerCandidates584_0 := by
  unfold candidates
  have row : edge 4567 = SearchData.row4567.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan584_0 : scan edge (search edge 3) 4 [4273] 4273 innerTail584_0 best = best := by
  rw [innerTail584_0, scan, if_neg (by decide), innerCandidates584_0_checked]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_1
    (if 3 ≤ innerCandidates584_0.length then search edge 3 [4273, 4567] innerCandidates584_0 8840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan584_1

theorem rootBranch584 : search edge 4 [4273] rootCandidates584 4273 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4273] 4273 innerTail584_0 best = best
  exact innerScan584_0

def rootCandidates583 : List Nat := [4517, 4673, 5009, 5573, 5693, 6131, 6857, 7823]
set_option maxRecDepth 4096 in
theorem rootCandidates583_checked : candidates edge 5 0 4271 tail584 best.bound = rootCandidates583 := by
  unfold candidates
  have row : edge 4271 = SearchData.row4271.contains := by funext q; rfl
  rw [row]
  decide

def innerTail583_8 : List Nat := []
theorem innerScan583_8 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_8 best = best := rfl

def innerTail583_7 : List Nat := 7823 :: innerTail583_8
def innerCandidates583_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_7_checked : candidates edge 4 4271 7823 innerTail583_8 best.bound = innerCandidates583_7 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_7 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_7 best = best := by
  rw [innerTail583_7, scan, if_neg (by decide), innerCandidates583_7_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_8
    (if 3 ≤ innerCandidates583_7.length then search edge 3 [4271, 7823] innerCandidates583_7 12094 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_8

def innerTail583_6 : List Nat := 6857 :: innerTail583_7
def innerCandidates583_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_6_checked : candidates edge 4 4271 6857 innerTail583_7 best.bound = innerCandidates583_6 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_6 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_6 best = best := by
  rw [innerTail583_6, scan, if_neg (by decide), innerCandidates583_6_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_7
    (if 3 ≤ innerCandidates583_6.length then search edge 3 [4271, 6857] innerCandidates583_6 11128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_7

def innerTail583_5 : List Nat := 6131 :: innerTail583_6
def innerCandidates583_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_5_checked : candidates edge 4 4271 6131 innerTail583_6 best.bound = innerCandidates583_5 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_5 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_5 best = best := by
  rw [innerTail583_5, scan, if_neg (by decide), innerCandidates583_5_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_6
    (if 3 ≤ innerCandidates583_5.length then search edge 3 [4271, 6131] innerCandidates583_5 10402 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_6

def innerTail583_4 : List Nat := 5693 :: innerTail583_5
def innerCandidates583_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_4_checked : candidates edge 4 4271 5693 innerTail583_5 best.bound = innerCandidates583_4 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_4 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_4 best = best := by
  rw [innerTail583_4, scan, if_neg (by decide), innerCandidates583_4_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_5
    (if 3 ≤ innerCandidates583_4.length then search edge 3 [4271, 5693] innerCandidates583_4 9964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_5

def innerTail583_3 : List Nat := 5573 :: innerTail583_4
def innerCandidates583_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_3_checked : candidates edge 4 4271 5573 innerTail583_4 best.bound = innerCandidates583_3 := by
  unfold candidates
  have row : edge 5573 = SearchData.row5573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_3 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_3 best = best := by
  rw [innerTail583_3, scan, if_neg (by decide), innerCandidates583_3_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_4
    (if 3 ≤ innerCandidates583_3.length then search edge 3 [4271, 5573] innerCandidates583_3 9844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_4

def innerTail583_2 : List Nat := 5009 :: innerTail583_3
def innerCandidates583_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_2_checked : candidates edge 4 4271 5009 innerTail583_3 best.bound = innerCandidates583_2 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_2 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_2 best = best := by
  rw [innerTail583_2, scan, if_neg (by decide), innerCandidates583_2_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_3
    (if 3 ≤ innerCandidates583_2.length then search edge 3 [4271, 5009] innerCandidates583_2 9280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_3

def innerTail583_1 : List Nat := 4673 :: innerTail583_2
def innerCandidates583_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_1_checked : candidates edge 4 4271 4673 innerTail583_2 best.bound = innerCandidates583_1 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_1 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_1 best = best := by
  rw [innerTail583_1, scan, if_neg (by decide), innerCandidates583_1_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_2
    (if 3 ≤ innerCandidates583_1.length then search edge 3 [4271, 4673] innerCandidates583_1 8944 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_2

def innerTail583_0 : List Nat := 4517 :: innerTail583_1
def innerCandidates583_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates583_0_checked : candidates edge 4 4271 4517 innerTail583_1 best.bound = innerCandidates583_0 := by
  unfold candidates
  have row : edge 4517 = SearchData.row4517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan583_0 : scan edge (search edge 3) 4 [4271] 4271 innerTail583_0 best = best := by
  rw [innerTail583_0, scan, if_neg (by decide), innerCandidates583_0_checked]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_1
    (if 3 ≤ innerCandidates583_0.length then search edge 3 [4271, 4517] innerCandidates583_0 8788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan583_1

theorem rootBranch583 : search edge 4 [4271] rootCandidates583 4271 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4271] 4271 innerTail583_0 best = best
  exact innerScan583_0

def rootCandidates582 : List Nat := [5011, 5281, 5827, 7621, 7951, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates582_checked : candidates edge 5 0 4261 tail583 best.bound = rootCandidates582 := by
  unfold candidates
  have row : edge 4261 = SearchData.row4261.contains := by funext q; rfl
  rw [row]
  decide

def innerTail582_6 : List Nat := []
theorem innerScan582_6 : scan edge (search edge 3) 4 [4261] 4261 innerTail582_6 best = best := rfl

def innerTail582_5 : List Nat := 8179 :: innerTail582_6
def innerCandidates582_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates582_5_checked : candidates edge 4 4261 8179 innerTail582_6 best.bound = innerCandidates582_5 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan582_5 : scan edge (search edge 3) 4 [4261] 4261 innerTail582_5 best = best := by
  rw [innerTail582_5, scan, if_neg (by decide), innerCandidates582_5_checked]
  change scan edge (search edge 3) 4 [4261] 4261 innerTail582_6
    (if 3 ≤ innerCandidates582_5.length then search edge 3 [4261, 8179] innerCandidates582_5 12440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan582_6

def innerTail582_4 : List Nat := 7951 :: innerTail582_5
def innerCandidates582_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates582_4_checked : candidates edge 4 4261 7951 innerTail582_5 best.bound = innerCandidates582_4 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan582_4 : scan edge (search edge 3) 4 [4261] 4261 innerTail582_4 best = best := by
  rw [innerTail582_4, scan, if_neg (by decide), innerCandidates582_4_checked]
  change scan edge (search edge 3) 4 [4261] 4261 innerTail582_5
    (if 3 ≤ innerCandidates582_4.length then search edge 3 [4261, 7951] innerCandidates582_4 12212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan582_5

def innerTail582_3 : List Nat := 7621 :: innerTail582_4
def innerCandidates582_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates582_3_checked : candidates edge 4 4261 7621 innerTail582_4 best.bound = innerCandidates582_3 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan582_3 : scan edge (search edge 3) 4 [4261] 4261 innerTail582_3 best = best := by
  rw [innerTail582_3, scan, if_neg (by decide), innerCandidates582_3_checked]
  change scan edge (search edge 3) 4 [4261] 4261 innerTail582_4
    (if 3 ≤ innerCandidates582_3.length then search edge 3 [4261, 7621] innerCandidates582_3 11882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan582_4

def innerTail582_2 : List Nat := 5827 :: innerTail582_3
def innerCandidates582_2 : List Nat := [7621]
set_option maxRecDepth 4096 in
theorem innerCandidates582_2_checked : candidates edge 4 4261 5827 innerTail582_3 best.bound = innerCandidates582_2 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan582_2 : scan edge (search edge 3) 4 [4261] 4261 innerTail582_2 best = best := by
  rw [innerTail582_2, scan, if_neg (by decide), innerCandidates582_2_checked]
  change scan edge (search edge 3) 4 [4261] 4261 innerTail582_3
    (if 3 ≤ innerCandidates582_2.length then search edge 3 [4261, 5827] innerCandidates582_2 10088 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan582_3

def innerTail582_1 : List Nat := 5281 :: innerTail582_2
def innerCandidates582_1 : List Nat := [7951]
set_option maxRecDepth 4096 in
theorem innerCandidates582_1_checked : candidates edge 4 4261 5281 innerTail582_2 best.bound = innerCandidates582_1 := by
  unfold candidates
  have row : edge 5281 = SearchData.row5281.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan582_1 : scan edge (search edge 3) 4 [4261] 4261 innerTail582_1 best = best := by
  rw [innerTail582_1, scan, if_neg (by decide), innerCandidates582_1_checked]
  change scan edge (search edge 3) 4 [4261] 4261 innerTail582_2
    (if 3 ≤ innerCandidates582_1.length then search edge 3 [4261, 5281] innerCandidates582_1 9542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan582_2

def innerTail582_0 : List Nat := 5011 :: innerTail582_1
def innerCandidates582_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates582_0_checked : candidates edge 4 4261 5011 innerTail582_1 best.bound = innerCandidates582_0 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan582_0 : scan edge (search edge 3) 4 [4261] 4261 innerTail582_0 best = best := by
  rw [innerTail582_0, scan, if_neg (by decide), innerCandidates582_0_checked]
  change scan edge (search edge 3) 4 [4261] 4261 innerTail582_1
    (if 3 ≤ innerCandidates582_0.length then search edge 3 [4261, 5011] innerCandidates582_0 9272 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan582_1

theorem rootBranch582 : search edge 4 [4261] rootCandidates582 4261 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4261] 4261 innerTail582_0 best = best
  exact innerScan582_0

def rootCandidates581 : List Nat := [4787, 5087, 5417, 5807, 6221]
set_option maxRecDepth 4096 in
theorem rootCandidates581_checked : candidates edge 5 0 4259 tail582 best.bound = rootCandidates581 := by
  unfold candidates
  have row : edge 4259 = SearchData.row4259.contains := by funext q; rfl
  rw [row]
  decide

def innerTail581_5 : List Nat := []
theorem innerScan581_5 : scan edge (search edge 3) 4 [4259] 4259 innerTail581_5 best = best := rfl

def innerTail581_4 : List Nat := 6221 :: innerTail581_5
def innerCandidates581_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates581_4_checked : candidates edge 4 4259 6221 innerTail581_5 best.bound = innerCandidates581_4 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan581_4 : scan edge (search edge 3) 4 [4259] 4259 innerTail581_4 best = best := by
  rw [innerTail581_4, scan, if_neg (by decide), innerCandidates581_4_checked]
  change scan edge (search edge 3) 4 [4259] 4259 innerTail581_5
    (if 3 ≤ innerCandidates581_4.length then search edge 3 [4259, 6221] innerCandidates581_4 10480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan581_5

def innerTail581_3 : List Nat := 5807 :: innerTail581_4
def innerCandidates581_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates581_3_checked : candidates edge 4 4259 5807 innerTail581_4 best.bound = innerCandidates581_3 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan581_3 : scan edge (search edge 3) 4 [4259] 4259 innerTail581_3 best = best := by
  rw [innerTail581_3, scan, if_neg (by decide), innerCandidates581_3_checked]
  change scan edge (search edge 3) 4 [4259] 4259 innerTail581_4
    (if 3 ≤ innerCandidates581_3.length then search edge 3 [4259, 5807] innerCandidates581_3 10066 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan581_4

def innerTail581_2 : List Nat := 5417 :: innerTail581_3
def innerCandidates581_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates581_2_checked : candidates edge 4 4259 5417 innerTail581_3 best.bound = innerCandidates581_2 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan581_2 : scan edge (search edge 3) 4 [4259] 4259 innerTail581_2 best = best := by
  rw [innerTail581_2, scan, if_neg (by decide), innerCandidates581_2_checked]
  change scan edge (search edge 3) 4 [4259] 4259 innerTail581_3
    (if 3 ≤ innerCandidates581_2.length then search edge 3 [4259, 5417] innerCandidates581_2 9676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan581_3

def innerTail581_1 : List Nat := 5087 :: innerTail581_2
def innerCandidates581_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates581_1_checked : candidates edge 4 4259 5087 innerTail581_2 best.bound = innerCandidates581_1 := by
  unfold candidates
  have row : edge 5087 = SearchData.row5087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan581_1 : scan edge (search edge 3) 4 [4259] 4259 innerTail581_1 best = best := by
  rw [innerTail581_1, scan, if_neg (by decide), innerCandidates581_1_checked]
  change scan edge (search edge 3) 4 [4259] 4259 innerTail581_2
    (if 3 ≤ innerCandidates581_1.length then search edge 3 [4259, 5087] innerCandidates581_1 9346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan581_2

def innerTail581_0 : List Nat := 4787 :: innerTail581_1
def innerCandidates581_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates581_0_checked : candidates edge 4 4259 4787 innerTail581_1 best.bound = innerCandidates581_0 := by
  unfold candidates
  have row : edge 4787 = SearchData.row4787.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan581_0 : scan edge (search edge 3) 4 [4259] 4259 innerTail581_0 best = best := by
  rw [innerTail581_0, scan, if_neg (by decide), innerCandidates581_0_checked]
  change scan edge (search edge 3) 4 [4259] 4259 innerTail581_1
    (if 3 ≤ innerCandidates581_0.length then search edge 3 [4259, 4787] innerCandidates581_0 9046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan581_1

theorem rootBranch581 : search edge 4 [4259] rootCandidates581 4259 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4259] 4259 innerTail581_0 best = best
  exact innerScan581_0

end ProjectEulerSolutions.PrimePairUpper.Search
