import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates751 : List Nat := [6301, 7213, 7687, 8263]
set_option maxRecDepth 4096 in
theorem rootCandidates751_checked : candidates edge 5 0 5737 tail752 best.bound = rootCandidates751 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext q; rfl
  rw [row]
  decide

def innerTail751_4 : List Nat := []
theorem innerScan751_4 : scan edge (search edge 3) 4 [5737] 5737 innerTail751_4 best = best := rfl

def innerTail751_3 : List Nat := 8263 :: innerTail751_4
def innerCandidates751_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates751_3_checked : candidates edge 4 5737 8263 innerTail751_4 best.bound = innerCandidates751_3 := by
  unfold candidates
  have row : edge 8263 = SearchData.row8263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan751_3 : scan edge (search edge 3) 4 [5737] 5737 innerTail751_3 best = best := by
  rw [innerTail751_3, scan, if_neg (by decide), innerCandidates751_3_checked]
  change scan edge (search edge 3) 4 [5737] 5737 innerTail751_4
    (if 3 ≤ innerCandidates751_3.length then search edge 3 [5737, 8263] innerCandidates751_3 14000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan751_4

def innerTail751_2 : List Nat := 7687 :: innerTail751_3
def innerCandidates751_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates751_2_checked : candidates edge 4 5737 7687 innerTail751_3 best.bound = innerCandidates751_2 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan751_2 : scan edge (search edge 3) 4 [5737] 5737 innerTail751_2 best = best := by
  rw [innerTail751_2, scan, if_neg (by decide), innerCandidates751_2_checked]
  change scan edge (search edge 3) 4 [5737] 5737 innerTail751_3
    (if 3 ≤ innerCandidates751_2.length then search edge 3 [5737, 7687] innerCandidates751_2 13424 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan751_3

def innerTail751_1 : List Nat := 7213 :: innerTail751_2
def innerCandidates751_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates751_1_checked : candidates edge 4 5737 7213 innerTail751_2 best.bound = innerCandidates751_1 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan751_1 : scan edge (search edge 3) 4 [5737] 5737 innerTail751_1 best = best := by
  rw [innerTail751_1, scan, if_neg (by decide), innerCandidates751_1_checked]
  change scan edge (search edge 3) 4 [5737] 5737 innerTail751_2
    (if 3 ≤ innerCandidates751_1.length then search edge 3 [5737, 7213] innerCandidates751_1 12950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan751_2

def innerTail751_0 : List Nat := 6301 :: innerTail751_1
def innerCandidates751_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates751_0_checked : candidates edge 4 5737 6301 innerTail751_1 best.bound = innerCandidates751_0 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan751_0 : scan edge (search edge 3) 4 [5737] 5737 innerTail751_0 best = best := by
  rw [innerTail751_0, scan, if_neg (by decide), innerCandidates751_0_checked]
  change scan edge (search edge 3) 4 [5737] 5737 innerTail751_1
    (if 3 ≤ innerCandidates751_0.length then search edge 3 [5737, 6301] innerCandidates751_0 12038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan751_1

theorem rootBranch751 : search edge 4 [5737] rootCandidates751 5737 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5737] 5737 innerTail751_0 best = best
  exact innerScan751_0

def rootCandidates750 : List Nat := [7079, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates750_checked : candidates edge 5 0 5717 tail751 best.bound = rootCandidates750 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates749 : List Nat := [5783, 5867, 5939, 6173, 7013, 7103, 7451, 7523, 7883, 8069, 8273, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates749_checked : candidates edge 5 0 5711 tail750 best.bound = rootCandidates749 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext q; rfl
  rw [row]
  decide

def innerTail749_12 : List Nat := []
theorem innerScan749_12 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_12 best = best := rfl

def innerTail749_11 : List Nat := 8291 :: innerTail749_12
def innerCandidates749_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_11_checked : candidates edge 4 5711 8291 innerTail749_12 best.bound = innerCandidates749_11 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_11 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_11 best = best := by
  rw [innerTail749_11, scan, if_neg (by decide), innerCandidates749_11_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_12
    (if 3 ≤ innerCandidates749_11.length then search edge 3 [5711, 8291] innerCandidates749_11 14002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_12

def innerTail749_10 : List Nat := 8273 :: innerTail749_11
def innerCandidates749_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_10_checked : candidates edge 4 5711 8273 innerTail749_11 best.bound = innerCandidates749_10 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_10 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_10 best = best := by
  rw [innerTail749_10, scan, if_neg (by decide), innerCandidates749_10_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_11
    (if 3 ≤ innerCandidates749_10.length then search edge 3 [5711, 8273] innerCandidates749_10 13984 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_11

def innerTail749_9 : List Nat := 8069 :: innerTail749_10
def innerCandidates749_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_9_checked : candidates edge 4 5711 8069 innerTail749_10 best.bound = innerCandidates749_9 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_9 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_9 best = best := by
  rw [innerTail749_9, scan, if_neg (by decide), innerCandidates749_9_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_10
    (if 3 ≤ innerCandidates749_9.length then search edge 3 [5711, 8069] innerCandidates749_9 13780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_10

def innerTail749_8 : List Nat := 7883 :: innerTail749_9
def innerCandidates749_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_8_checked : candidates edge 4 5711 7883 innerTail749_9 best.bound = innerCandidates749_8 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_8 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_8 best = best := by
  rw [innerTail749_8, scan, if_neg (by decide), innerCandidates749_8_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_9
    (if 3 ≤ innerCandidates749_8.length then search edge 3 [5711, 7883] innerCandidates749_8 13594 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_9

def innerTail749_7 : List Nat := 7523 :: innerTail749_8
def innerCandidates749_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_7_checked : candidates edge 4 5711 7523 innerTail749_8 best.bound = innerCandidates749_7 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_7 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_7 best = best := by
  rw [innerTail749_7, scan, if_neg (by decide), innerCandidates749_7_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_8
    (if 3 ≤ innerCandidates749_7.length then search edge 3 [5711, 7523] innerCandidates749_7 13234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_8

def innerTail749_6 : List Nat := 7451 :: innerTail749_7
def innerCandidates749_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_6_checked : candidates edge 4 5711 7451 innerTail749_7 best.bound = innerCandidates749_6 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_6 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_6 best = best := by
  rw [innerTail749_6, scan, if_neg (by decide), innerCandidates749_6_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_7
    (if 3 ≤ innerCandidates749_6.length then search edge 3 [5711, 7451] innerCandidates749_6 13162 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_7

def innerTail749_5 : List Nat := 7103 :: innerTail749_6
def innerCandidates749_5 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates749_5_checked : candidates edge 4 5711 7103 innerTail749_6 best.bound = innerCandidates749_5 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_5 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_5 best = best := by
  rw [innerTail749_5, scan, if_neg (by decide), innerCandidates749_5_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_6
    (if 3 ≤ innerCandidates749_5.length then search edge 3 [5711, 7103] innerCandidates749_5 12814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_6

def innerTail749_4 : List Nat := 7013 :: innerTail749_5
def innerCandidates749_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_4_checked : candidates edge 4 5711 7013 innerTail749_5 best.bound = innerCandidates749_4 := by
  unfold candidates
  have row : edge 7013 = SearchData.row7013.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_4 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_4 best = best := by
  rw [innerTail749_4, scan, if_neg (by decide), innerCandidates749_4_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_5
    (if 3 ≤ innerCandidates749_4.length then search edge 3 [5711, 7013] innerCandidates749_4 12724 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_5

def innerTail749_3 : List Nat := 6173 :: innerTail749_4
def innerCandidates749_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_3_checked : candidates edge 4 5711 6173 innerTail749_4 best.bound = innerCandidates749_3 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_3 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_3 best = best := by
  rw [innerTail749_3, scan, if_neg (by decide), innerCandidates749_3_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_4
    (if 3 ≤ innerCandidates749_3.length then search edge 3 [5711, 6173] innerCandidates749_3 11884 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_4

def innerTail749_2 : List Nat := 5939 :: innerTail749_3
def innerCandidates749_2 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates749_2_checked : candidates edge 4 5711 5939 innerTail749_3 best.bound = innerCandidates749_2 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_2 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_2 best = best := by
  rw [innerTail749_2, scan, if_neg (by decide), innerCandidates749_2_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_3
    (if 3 ≤ innerCandidates749_2.length then search edge 3 [5711, 5939] innerCandidates749_2 11650 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_3

def innerTail749_1 : List Nat := 5867 :: innerTail749_2
def innerCandidates749_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_1_checked : candidates edge 4 5711 5867 innerTail749_2 best.bound = innerCandidates749_1 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_1 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_1 best = best := by
  rw [innerTail749_1, scan, if_neg (by decide), innerCandidates749_1_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_2
    (if 3 ≤ innerCandidates749_1.length then search edge 3 [5711, 5867] innerCandidates749_1 11578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_2

def innerTail749_0 : List Nat := 5783 :: innerTail749_1
def innerCandidates749_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates749_0_checked : candidates edge 4 5711 5783 innerTail749_1 best.bound = innerCandidates749_0 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan749_0 : scan edge (search edge 3) 4 [5711] 5711 innerTail749_0 best = best := by
  rw [innerTail749_0, scan, if_neg (by decide), innerCandidates749_0_checked]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_1
    (if 3 ≤ innerCandidates749_0.length then search edge 3 [5711, 5783] innerCandidates749_0 11494 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan749_1

theorem rootBranch749 : search edge 4 [5711] rootCandidates749 5711 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5711] 5711 innerTail749_0 best = best
  exact innerScan749_0

def rootCandidates748 : List Nat := [6367, 6481, 6733, 7459, 7669, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates748_checked : candidates edge 5 0 5701 tail749 best.bound = rootCandidates748 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext q; rfl
  rw [row]
  decide

def innerTail748_6 : List Nat := []
theorem innerScan748_6 : scan edge (search edge 3) 4 [5701] 5701 innerTail748_6 best = best := rfl

def innerTail748_5 : List Nat := 8209 :: innerTail748_6
def innerCandidates748_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates748_5_checked : candidates edge 4 5701 8209 innerTail748_6 best.bound = innerCandidates748_5 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan748_5 : scan edge (search edge 3) 4 [5701] 5701 innerTail748_5 best = best := by
  rw [innerTail748_5, scan, if_neg (by decide), innerCandidates748_5_checked]
  change scan edge (search edge 3) 4 [5701] 5701 innerTail748_6
    (if 3 ≤ innerCandidates748_5.length then search edge 3 [5701, 8209] innerCandidates748_5 13910 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan748_6

def innerTail748_4 : List Nat := 7669 :: innerTail748_5
def innerCandidates748_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates748_4_checked : candidates edge 4 5701 7669 innerTail748_5 best.bound = innerCandidates748_4 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan748_4 : scan edge (search edge 3) 4 [5701] 5701 innerTail748_4 best = best := by
  rw [innerTail748_4, scan, if_neg (by decide), innerCandidates748_4_checked]
  change scan edge (search edge 3) 4 [5701] 5701 innerTail748_5
    (if 3 ≤ innerCandidates748_4.length then search edge 3 [5701, 7669] innerCandidates748_4 13370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan748_5

def innerTail748_3 : List Nat := 7459 :: innerTail748_4
def innerCandidates748_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates748_3_checked : candidates edge 4 5701 7459 innerTail748_4 best.bound = innerCandidates748_3 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan748_3 : scan edge (search edge 3) 4 [5701] 5701 innerTail748_3 best = best := by
  rw [innerTail748_3, scan, if_neg (by decide), innerCandidates748_3_checked]
  change scan edge (search edge 3) 4 [5701] 5701 innerTail748_4
    (if 3 ≤ innerCandidates748_3.length then search edge 3 [5701, 7459] innerCandidates748_3 13160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan748_4

def innerTail748_2 : List Nat := 6733 :: innerTail748_3
def innerCandidates748_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates748_2_checked : candidates edge 4 5701 6733 innerTail748_3 best.bound = innerCandidates748_2 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan748_2 : scan edge (search edge 3) 4 [5701] 5701 innerTail748_2 best = best := by
  rw [innerTail748_2, scan, if_neg (by decide), innerCandidates748_2_checked]
  change scan edge (search edge 3) 4 [5701] 5701 innerTail748_3
    (if 3 ≤ innerCandidates748_2.length then search edge 3 [5701, 6733] innerCandidates748_2 12434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan748_3

def innerTail748_1 : List Nat := 6481 :: innerTail748_2
def innerCandidates748_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates748_1_checked : candidates edge 4 5701 6481 innerTail748_2 best.bound = innerCandidates748_1 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan748_1 : scan edge (search edge 3) 4 [5701] 5701 innerTail748_1 best = best := by
  rw [innerTail748_1, scan, if_neg (by decide), innerCandidates748_1_checked]
  change scan edge (search edge 3) 4 [5701] 5701 innerTail748_2
    (if 3 ≤ innerCandidates748_1.length then search edge 3 [5701, 6481] innerCandidates748_1 12182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan748_2

def innerTail748_0 : List Nat := 6367 :: innerTail748_1
def innerCandidates748_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates748_0_checked : candidates edge 4 5701 6367 innerTail748_1 best.bound = innerCandidates748_0 := by
  unfold candidates
  have row : edge 6367 = SearchData.row6367.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan748_0 : scan edge (search edge 3) 4 [5701] 5701 innerTail748_0 best = best := by
  rw [innerTail748_0, scan, if_neg (by decide), innerCandidates748_0_checked]
  change scan edge (search edge 3) 4 [5701] 5701 innerTail748_1
    (if 3 ≤ innerCandidates748_0.length then search edge 3 [5701, 6367] innerCandidates748_0 12068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan748_1

theorem rootBranch748 : search edge 4 [5701] rootCandidates748 5701 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5701] 5701 innerTail748_0 best = best
  exact innerScan748_0

def rootCandidates747 : List Nat := [6329, 6563, 7103, 7187, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates747_checked : candidates edge 5 0 5693 tail748 best.bound = rootCandidates747 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext q; rfl
  rw [row]
  decide

def innerTail747_5 : List Nat := []
theorem innerScan747_5 : scan edge (search edge 3) 4 [5693] 5693 innerTail747_5 best = best := rfl

def innerTail747_4 : List Nat := 8171 :: innerTail747_5
def innerCandidates747_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates747_4_checked : candidates edge 4 5693 8171 innerTail747_5 best.bound = innerCandidates747_4 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan747_4 : scan edge (search edge 3) 4 [5693] 5693 innerTail747_4 best = best := by
  rw [innerTail747_4, scan, if_neg (by decide), innerCandidates747_4_checked]
  change scan edge (search edge 3) 4 [5693] 5693 innerTail747_5
    (if 3 ≤ innerCandidates747_4.length then search edge 3 [5693, 8171] innerCandidates747_4 13864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan747_5

def innerTail747_3 : List Nat := 7187 :: innerTail747_4
def innerCandidates747_3 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates747_3_checked : candidates edge 4 5693 7187 innerTail747_4 best.bound = innerCandidates747_3 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan747_3 : scan edge (search edge 3) 4 [5693] 5693 innerTail747_3 best = best := by
  rw [innerTail747_3, scan, if_neg (by decide), innerCandidates747_3_checked]
  change scan edge (search edge 3) 4 [5693] 5693 innerTail747_4
    (if 3 ≤ innerCandidates747_3.length then search edge 3 [5693, 7187] innerCandidates747_3 12880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan747_4

def innerTail747_2 : List Nat := 7103 :: innerTail747_3
def innerCandidates747_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates747_2_checked : candidates edge 4 5693 7103 innerTail747_3 best.bound = innerCandidates747_2 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan747_2 : scan edge (search edge 3) 4 [5693] 5693 innerTail747_2 best = best := by
  rw [innerTail747_2, scan, if_neg (by decide), innerCandidates747_2_checked]
  change scan edge (search edge 3) 4 [5693] 5693 innerTail747_3
    (if 3 ≤ innerCandidates747_2.length then search edge 3 [5693, 7103] innerCandidates747_2 12796 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan747_3

def innerTail747_1 : List Nat := 6563 :: innerTail747_2
def innerCandidates747_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates747_1_checked : candidates edge 4 5693 6563 innerTail747_2 best.bound = innerCandidates747_1 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan747_1 : scan edge (search edge 3) 4 [5693] 5693 innerTail747_1 best = best := by
  rw [innerTail747_1, scan, if_neg (by decide), innerCandidates747_1_checked]
  change scan edge (search edge 3) 4 [5693] 5693 innerTail747_2
    (if 3 ≤ innerCandidates747_1.length then search edge 3 [5693, 6563] innerCandidates747_1 12256 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan747_2

def innerTail747_0 : List Nat := 6329 :: innerTail747_1
def innerCandidates747_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates747_0_checked : candidates edge 4 5693 6329 innerTail747_1 best.bound = innerCandidates747_0 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan747_0 : scan edge (search edge 3) 4 [5693] 5693 innerTail747_0 best = best := by
  rw [innerTail747_0, scan, if_neg (by decide), innerCandidates747_0_checked]
  change scan edge (search edge 3) 4 [5693] 5693 innerTail747_1
    (if 3 ≤ innerCandidates747_0.length then search edge 3 [5693, 6329] innerCandidates747_0 12022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan747_1

theorem rootBranch747 : search edge 4 [5693] rootCandidates747 5693 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5693] 5693 innerTail747_0 best = best
  exact innerScan747_0

def rootCandidates746 : List Nat := [5749, 6163, 6211, 6271, 6277, 6673, 6823, 7297, 7573, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates746_checked : candidates edge 5 0 5689 tail747 best.bound = rootCandidates746 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext q; rfl
  rw [row]
  decide

def innerTail746_10 : List Nat := []
theorem innerScan746_10 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_10 best = best := rfl

def innerTail746_9 : List Nat := 8329 :: innerTail746_10
def innerCandidates746_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates746_9_checked : candidates edge 4 5689 8329 innerTail746_10 best.bound = innerCandidates746_9 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_9 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_9 best = best := by
  rw [innerTail746_9, scan, if_neg (by decide), innerCandidates746_9_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_10
    (if 3 ≤ innerCandidates746_9.length then search edge 3 [5689, 8329] innerCandidates746_9 14018 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_10

def innerTail746_8 : List Nat := 7573 :: innerTail746_9
def innerCandidates746_8 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem innerCandidates746_8_checked : candidates edge 4 5689 7573 innerTail746_9 best.bound = innerCandidates746_8 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_8 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_8 best = best := by
  rw [innerTail746_8, scan, if_neg (by decide), innerCandidates746_8_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_9
    (if 3 ≤ innerCandidates746_8.length then search edge 3 [5689, 7573] innerCandidates746_8 13262 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_9

def innerTail746_7 : List Nat := 7297 :: innerTail746_8
def innerCandidates746_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates746_7_checked : candidates edge 4 5689 7297 innerTail746_8 best.bound = innerCandidates746_7 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_7 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_7 best = best := by
  rw [innerTail746_7, scan, if_neg (by decide), innerCandidates746_7_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_8
    (if 3 ≤ innerCandidates746_7.length then search edge 3 [5689, 7297] innerCandidates746_7 12986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_8

def innerTail746_6 : List Nat := 6823 :: innerTail746_7
def innerCandidates746_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates746_6_checked : candidates edge 4 5689 6823 innerTail746_7 best.bound = innerCandidates746_6 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_6 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_6 best = best := by
  rw [innerTail746_6, scan, if_neg (by decide), innerCandidates746_6_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_7
    (if 3 ≤ innerCandidates746_6.length then search edge 3 [5689, 6823] innerCandidates746_6 12512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_7

def innerTail746_5 : List Nat := 6673 :: innerTail746_6
def innerCandidates746_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates746_5_checked : candidates edge 4 5689 6673 innerTail746_6 best.bound = innerCandidates746_5 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_5 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_5 best = best := by
  rw [innerTail746_5, scan, if_neg (by decide), innerCandidates746_5_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_6
    (if 3 ≤ innerCandidates746_5.length then search edge 3 [5689, 6673] innerCandidates746_5 12362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_6

def innerTail746_4 : List Nat := 6277 :: innerTail746_5
def innerCandidates746_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates746_4_checked : candidates edge 4 5689 6277 innerTail746_5 best.bound = innerCandidates746_4 := by
  unfold candidates
  have row : edge 6277 = SearchData.row6277.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_4 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_4 best = best := by
  rw [innerTail746_4, scan, if_neg (by decide), innerCandidates746_4_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_5
    (if 3 ≤ innerCandidates746_4.length then search edge 3 [5689, 6277] innerCandidates746_4 11966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_5

def innerTail746_3 : List Nat := 6271 :: innerTail746_4
def innerCandidates746_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates746_3_checked : candidates edge 4 5689 6271 innerTail746_4 best.bound = innerCandidates746_3 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_3 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_3 best = best := by
  rw [innerTail746_3, scan, if_neg (by decide), innerCandidates746_3_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_4
    (if 3 ≤ innerCandidates746_3.length then search edge 3 [5689, 6271] innerCandidates746_3 11960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_4

def innerTail746_2 : List Nat := 6211 :: innerTail746_3
def innerCandidates746_2 : List Nat := [6277]
set_option maxRecDepth 4096 in
theorem innerCandidates746_2_checked : candidates edge 4 5689 6211 innerTail746_3 best.bound = innerCandidates746_2 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_2 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_2 best = best := by
  rw [innerTail746_2, scan, if_neg (by decide), innerCandidates746_2_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_3
    (if 3 ≤ innerCandidates746_2.length then search edge 3 [5689, 6211] innerCandidates746_2 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_3

def innerTail746_1 : List Nat := 6163 :: innerTail746_2
def innerCandidates746_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates746_1_checked : candidates edge 4 5689 6163 innerTail746_2 best.bound = innerCandidates746_1 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_1 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_1 best = best := by
  rw [innerTail746_1, scan, if_neg (by decide), innerCandidates746_1_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_2
    (if 3 ≤ innerCandidates746_1.length then search edge 3 [5689, 6163] innerCandidates746_1 11852 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_2

def innerTail746_0 : List Nat := 5749 :: innerTail746_1
def innerCandidates746_0 : List Nat := [6211]
set_option maxRecDepth 4096 in
theorem innerCandidates746_0_checked : candidates edge 4 5689 5749 innerTail746_1 best.bound = innerCandidates746_0 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan746_0 : scan edge (search edge 3) 4 [5689] 5689 innerTail746_0 best = best := by
  rw [innerTail746_0, scan, if_neg (by decide), innerCandidates746_0_checked]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_1
    (if 3 ≤ innerCandidates746_0.length then search edge 3 [5689, 5749] innerCandidates746_0 11438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan746_1

theorem rootBranch746 : search edge 4 [5689] rootCandidates746 5689 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5689] 5689 innerTail746_0 best = best
  exact innerScan746_0

def rootCandidates745 : List Nat := [6217, 6271, 6907, 7057, 7159, 7369]
set_option maxRecDepth 4096 in
theorem rootCandidates745_checked : candidates edge 5 0 5683 tail746 best.bound = rootCandidates745 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext q; rfl
  rw [row]
  decide

def innerTail745_6 : List Nat := []
theorem innerScan745_6 : scan edge (search edge 3) 4 [5683] 5683 innerTail745_6 best = best := rfl

def innerTail745_5 : List Nat := 7369 :: innerTail745_6
def innerCandidates745_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates745_5_checked : candidates edge 4 5683 7369 innerTail745_6 best.bound = innerCandidates745_5 := by
  unfold candidates
  have row : edge 7369 = SearchData.row7369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan745_5 : scan edge (search edge 3) 4 [5683] 5683 innerTail745_5 best = best := by
  rw [innerTail745_5, scan, if_neg (by decide), innerCandidates745_5_checked]
  change scan edge (search edge 3) 4 [5683] 5683 innerTail745_6
    (if 3 ≤ innerCandidates745_5.length then search edge 3 [5683, 7369] innerCandidates745_5 13052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan745_6

def innerTail745_4 : List Nat := 7159 :: innerTail745_5
def innerCandidates745_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates745_4_checked : candidates edge 4 5683 7159 innerTail745_5 best.bound = innerCandidates745_4 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan745_4 : scan edge (search edge 3) 4 [5683] 5683 innerTail745_4 best = best := by
  rw [innerTail745_4, scan, if_neg (by decide), innerCandidates745_4_checked]
  change scan edge (search edge 3) 4 [5683] 5683 innerTail745_5
    (if 3 ≤ innerCandidates745_4.length then search edge 3 [5683, 7159] innerCandidates745_4 12842 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan745_5

def innerTail745_3 : List Nat := 7057 :: innerTail745_4
def innerCandidates745_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates745_3_checked : candidates edge 4 5683 7057 innerTail745_4 best.bound = innerCandidates745_3 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan745_3 : scan edge (search edge 3) 4 [5683] 5683 innerTail745_3 best = best := by
  rw [innerTail745_3, scan, if_neg (by decide), innerCandidates745_3_checked]
  change scan edge (search edge 3) 4 [5683] 5683 innerTail745_4
    (if 3 ≤ innerCandidates745_3.length then search edge 3 [5683, 7057] innerCandidates745_3 12740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan745_4

def innerTail745_2 : List Nat := 6907 :: innerTail745_3
def innerCandidates745_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates745_2_checked : candidates edge 4 5683 6907 innerTail745_3 best.bound = innerCandidates745_2 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan745_2 : scan edge (search edge 3) 4 [5683] 5683 innerTail745_2 best = best := by
  rw [innerTail745_2, scan, if_neg (by decide), innerCandidates745_2_checked]
  change scan edge (search edge 3) 4 [5683] 5683 innerTail745_3
    (if 3 ≤ innerCandidates745_2.length then search edge 3 [5683, 6907] innerCandidates745_2 12590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan745_3

def innerTail745_1 : List Nat := 6271 :: innerTail745_2
def innerCandidates745_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates745_1_checked : candidates edge 4 5683 6271 innerTail745_2 best.bound = innerCandidates745_1 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan745_1 : scan edge (search edge 3) 4 [5683] 5683 innerTail745_1 best = best := by
  rw [innerTail745_1, scan, if_neg (by decide), innerCandidates745_1_checked]
  change scan edge (search edge 3) 4 [5683] 5683 innerTail745_2
    (if 3 ≤ innerCandidates745_1.length then search edge 3 [5683, 6271] innerCandidates745_1 11954 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan745_2

def innerTail745_0 : List Nat := 6217 :: innerTail745_1
def innerCandidates745_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates745_0_checked : candidates edge 4 5683 6217 innerTail745_1 best.bound = innerCandidates745_0 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan745_0 : scan edge (search edge 3) 4 [5683] 5683 innerTail745_0 best = best := by
  rw [innerTail745_0, scan, if_neg (by decide), innerCandidates745_0_checked]
  change scan edge (search edge 3) 4 [5683] 5683 innerTail745_1
    (if 3 ≤ innerCandidates745_0.length then search edge 3 [5683, 6217] innerCandidates745_0 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan745_1

theorem rootBranch745 : search edge 4 [5683] rootCandidates745 5683 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5683] 5683 innerTail745_0 best = best
  exact innerScan745_0

def rootCandidates744 : List Nat := [5879, 6011, 7517, 7559]
set_option maxRecDepth 4096 in
theorem rootCandidates744_checked : candidates edge 5 0 5669 tail745 best.bound = rootCandidates744 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext q; rfl
  rw [row]
  decide

def innerTail744_4 : List Nat := []
theorem innerScan744_4 : scan edge (search edge 3) 4 [5669] 5669 innerTail744_4 best = best := rfl

def innerTail744_3 : List Nat := 7559 :: innerTail744_4
def innerCandidates744_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates744_3_checked : candidates edge 4 5669 7559 innerTail744_4 best.bound = innerCandidates744_3 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan744_3 : scan edge (search edge 3) 4 [5669] 5669 innerTail744_3 best = best := by
  rw [innerTail744_3, scan, if_neg (by decide), innerCandidates744_3_checked]
  change scan edge (search edge 3) 4 [5669] 5669 innerTail744_4
    (if 3 ≤ innerCandidates744_3.length then search edge 3 [5669, 7559] innerCandidates744_3 13228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan744_4

def innerTail744_2 : List Nat := 7517 :: innerTail744_3
def innerCandidates744_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates744_2_checked : candidates edge 4 5669 7517 innerTail744_3 best.bound = innerCandidates744_2 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan744_2 : scan edge (search edge 3) 4 [5669] 5669 innerTail744_2 best = best := by
  rw [innerTail744_2, scan, if_neg (by decide), innerCandidates744_2_checked]
  change scan edge (search edge 3) 4 [5669] 5669 innerTail744_3
    (if 3 ≤ innerCandidates744_2.length then search edge 3 [5669, 7517] innerCandidates744_2 13186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan744_3

def innerTail744_1 : List Nat := 6011 :: innerTail744_2
def innerCandidates744_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates744_1_checked : candidates edge 4 5669 6011 innerTail744_2 best.bound = innerCandidates744_1 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan744_1 : scan edge (search edge 3) 4 [5669] 5669 innerTail744_1 best = best := by
  rw [innerTail744_1, scan, if_neg (by decide), innerCandidates744_1_checked]
  change scan edge (search edge 3) 4 [5669] 5669 innerTail744_2
    (if 3 ≤ innerCandidates744_1.length then search edge 3 [5669, 6011] innerCandidates744_1 11680 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan744_2

def innerTail744_0 : List Nat := 5879 :: innerTail744_1
def innerCandidates744_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates744_0_checked : candidates edge 4 5669 5879 innerTail744_1 best.bound = innerCandidates744_0 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan744_0 : scan edge (search edge 3) 4 [5669] 5669 innerTail744_0 best = best := by
  rw [innerTail744_0, scan, if_neg (by decide), innerCandidates744_0_checked]
  change scan edge (search edge 3) 4 [5669] 5669 innerTail744_1
    (if 3 ≤ innerCandidates744_0.length then search edge 3 [5669, 5879] innerCandidates744_0 11548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan744_1

theorem rootBranch744 : search edge 4 [5669] rootCandidates744 5669 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5669] 5669 innerTail744_0 best = best
  exact innerScan744_0

end ProjectEulerSolutions.PrimePairUpper.Search
