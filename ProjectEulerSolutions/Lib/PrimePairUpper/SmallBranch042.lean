import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates712 : List Nat := [5981, 6131, 7001, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates712_checked : candidates edge 5 0 5417 tail713 best.bound = rootCandidates712 := by
  unfold candidates
  have row : edge 5417 = SearchData.row5417.contains := by funext q; rfl
  rw [row]
  decide

def innerTail712_4 : List Nat := []
theorem innerScan712_4 : scan edge (search edge 3) 4 [5417] 5417 innerTail712_4 best = best := rfl

def innerTail712_3 : List Nat := 8219 :: innerTail712_4
def innerCandidates712_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates712_3_checked : candidates edge 4 5417 8219 innerTail712_4 best.bound = innerCandidates712_3 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan712_3 : scan edge (search edge 3) 4 [5417] 5417 innerTail712_3 best = best := by
  rw [innerTail712_3, scan, if_neg (by decide), innerCandidates712_3_checked]
  change scan edge (search edge 3) 4 [5417] 5417 innerTail712_4
    (if 3 ≤ innerCandidates712_3.length then search edge 3 [5417, 8219] innerCandidates712_3 13636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan712_4

def innerTail712_2 : List Nat := 7001 :: innerTail712_3
def innerCandidates712_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates712_2_checked : candidates edge 4 5417 7001 innerTail712_3 best.bound = innerCandidates712_2 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan712_2 : scan edge (search edge 3) 4 [5417] 5417 innerTail712_2 best = best := by
  rw [innerTail712_2, scan, if_neg (by decide), innerCandidates712_2_checked]
  change scan edge (search edge 3) 4 [5417] 5417 innerTail712_3
    (if 3 ≤ innerCandidates712_2.length then search edge 3 [5417, 7001] innerCandidates712_2 12418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan712_3

def innerTail712_1 : List Nat := 6131 :: innerTail712_2
def innerCandidates712_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates712_1_checked : candidates edge 4 5417 6131 innerTail712_2 best.bound = innerCandidates712_1 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan712_1 : scan edge (search edge 3) 4 [5417] 5417 innerTail712_1 best = best := by
  rw [innerTail712_1, scan, if_neg (by decide), innerCandidates712_1_checked]
  change scan edge (search edge 3) 4 [5417] 5417 innerTail712_2
    (if 3 ≤ innerCandidates712_1.length then search edge 3 [5417, 6131] innerCandidates712_1 11548 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan712_2

def innerTail712_0 : List Nat := 5981 :: innerTail712_1
def innerCandidates712_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates712_0_checked : candidates edge 4 5417 5981 innerTail712_1 best.bound = innerCandidates712_0 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan712_0 : scan edge (search edge 3) 4 [5417] 5417 innerTail712_0 best = best := by
  rw [innerTail712_0, scan, if_neg (by decide), innerCandidates712_0_checked]
  change scan edge (search edge 3) 4 [5417] 5417 innerTail712_1
    (if 3 ≤ innerCandidates712_0.length then search edge 3 [5417, 5981] innerCandidates712_0 11398 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan712_1

theorem rootBranch712 : search edge 4 [5417] rootCandidates712 5417 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5417] 5417 innerTail712_0 best = best
  exact innerScan712_0

def rootCandidates711 : List Nat := [5527, 5569, 5821, 6451, 7159, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates711_checked : candidates edge 5 0 5413 tail712 best.bound = rootCandidates711 := by
  unfold candidates
  have row : edge 5413 = SearchData.row5413.contains := by funext q; rfl
  rw [row]
  decide

def innerTail711_6 : List Nat := []
theorem innerScan711_6 : scan edge (search edge 3) 4 [5413] 5413 innerTail711_6 best = best := rfl

def innerTail711_5 : List Nat := 8209 :: innerTail711_6
def innerCandidates711_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates711_5_checked : candidates edge 4 5413 8209 innerTail711_6 best.bound = innerCandidates711_5 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan711_5 : scan edge (search edge 3) 4 [5413] 5413 innerTail711_5 best = best := by
  rw [innerTail711_5, scan, if_neg (by decide), innerCandidates711_5_checked]
  change scan edge (search edge 3) 4 [5413] 5413 innerTail711_6
    (if 3 ≤ innerCandidates711_5.length then search edge 3 [5413, 8209] innerCandidates711_5 13622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan711_6

def innerTail711_4 : List Nat := 7159 :: innerTail711_5
def innerCandidates711_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates711_4_checked : candidates edge 4 5413 7159 innerTail711_5 best.bound = innerCandidates711_4 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan711_4 : scan edge (search edge 3) 4 [5413] 5413 innerTail711_4 best = best := by
  rw [innerTail711_4, scan, if_neg (by decide), innerCandidates711_4_checked]
  change scan edge (search edge 3) 4 [5413] 5413 innerTail711_5
    (if 3 ≤ innerCandidates711_4.length then search edge 3 [5413, 7159] innerCandidates711_4 12572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan711_5

def innerTail711_3 : List Nat := 6451 :: innerTail711_4
def innerCandidates711_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates711_3_checked : candidates edge 4 5413 6451 innerTail711_4 best.bound = innerCandidates711_3 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan711_3 : scan edge (search edge 3) 4 [5413] 5413 innerTail711_3 best = best := by
  rw [innerTail711_3, scan, if_neg (by decide), innerCandidates711_3_checked]
  change scan edge (search edge 3) 4 [5413] 5413 innerTail711_4
    (if 3 ≤ innerCandidates711_3.length then search edge 3 [5413, 6451] innerCandidates711_3 11864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan711_4

def innerTail711_2 : List Nat := 5821 :: innerTail711_3
def innerCandidates711_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates711_2_checked : candidates edge 4 5413 5821 innerTail711_3 best.bound = innerCandidates711_2 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan711_2 : scan edge (search edge 3) 4 [5413] 5413 innerTail711_2 best = best := by
  rw [innerTail711_2, scan, if_neg (by decide), innerCandidates711_2_checked]
  change scan edge (search edge 3) 4 [5413] 5413 innerTail711_3
    (if 3 ≤ innerCandidates711_2.length then search edge 3 [5413, 5821] innerCandidates711_2 11234 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan711_3

def innerTail711_1 : List Nat := 5569 :: innerTail711_2
def innerCandidates711_1 : List Nat := [6451]
set_option maxRecDepth 4096 in
theorem innerCandidates711_1_checked : candidates edge 4 5413 5569 innerTail711_2 best.bound = innerCandidates711_1 := by
  unfold candidates
  have row : edge 5569 = SearchData.row5569.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan711_1 : scan edge (search edge 3) 4 [5413] 5413 innerTail711_1 best = best := by
  rw [innerTail711_1, scan, if_neg (by decide), innerCandidates711_1_checked]
  change scan edge (search edge 3) 4 [5413] 5413 innerTail711_2
    (if 3 ≤ innerCandidates711_1.length then search edge 3 [5413, 5569] innerCandidates711_1 10982 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan711_2

def innerTail711_0 : List Nat := 5527 :: innerTail711_1
def innerCandidates711_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates711_0_checked : candidates edge 4 5413 5527 innerTail711_1 best.bound = innerCandidates711_0 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan711_0 : scan edge (search edge 3) 4 [5413] 5413 innerTail711_0 best = best := by
  rw [innerTail711_0, scan, if_neg (by decide), innerCandidates711_0_checked]
  change scan edge (search edge 3) 4 [5413] 5413 innerTail711_1
    (if 3 ≤ innerCandidates711_0.length then search edge 3 [5413, 5527] innerCandidates711_0 10940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan711_1

theorem rootBranch711 : search edge 4 [5413] rootCandidates711 5413 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5413] 5413 innerTail711_0 best = best
  exact innerScan711_0

def rootCandidates710 : List Nat := [5449, 5743, 5779, 6163, 6961, 8287]
set_option maxRecDepth 4096 in
theorem rootCandidates710_checked : candidates edge 5 0 5407 tail711 best.bound = rootCandidates710 := by
  unfold candidates
  have row : edge 5407 = SearchData.row5407.contains := by funext q; rfl
  rw [row]
  decide

def innerTail710_6 : List Nat := []
theorem innerScan710_6 : scan edge (search edge 3) 4 [5407] 5407 innerTail710_6 best = best := rfl

def innerTail710_5 : List Nat := 8287 :: innerTail710_6
def innerCandidates710_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates710_5_checked : candidates edge 4 5407 8287 innerTail710_6 best.bound = innerCandidates710_5 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan710_5 : scan edge (search edge 3) 4 [5407] 5407 innerTail710_5 best = best := by
  rw [innerTail710_5, scan, if_neg (by decide), innerCandidates710_5_checked]
  change scan edge (search edge 3) 4 [5407] 5407 innerTail710_6
    (if 3 ≤ innerCandidates710_5.length then search edge 3 [5407, 8287] innerCandidates710_5 13694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan710_6

def innerTail710_4 : List Nat := 6961 :: innerTail710_5
def innerCandidates710_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates710_4_checked : candidates edge 4 5407 6961 innerTail710_5 best.bound = innerCandidates710_4 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan710_4 : scan edge (search edge 3) 4 [5407] 5407 innerTail710_4 best = best := by
  rw [innerTail710_4, scan, if_neg (by decide), innerCandidates710_4_checked]
  change scan edge (search edge 3) 4 [5407] 5407 innerTail710_5
    (if 3 ≤ innerCandidates710_4.length then search edge 3 [5407, 6961] innerCandidates710_4 12368 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan710_5

def innerTail710_3 : List Nat := 6163 :: innerTail710_4
def innerCandidates710_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates710_3_checked : candidates edge 4 5407 6163 innerTail710_4 best.bound = innerCandidates710_3 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan710_3 : scan edge (search edge 3) 4 [5407] 5407 innerTail710_3 best = best := by
  rw [innerTail710_3, scan, if_neg (by decide), innerCandidates710_3_checked]
  change scan edge (search edge 3) 4 [5407] 5407 innerTail710_4
    (if 3 ≤ innerCandidates710_3.length then search edge 3 [5407, 6163] innerCandidates710_3 11570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan710_4

def innerTail710_2 : List Nat := 5779 :: innerTail710_3
def innerCandidates710_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates710_2_checked : candidates edge 4 5407 5779 innerTail710_3 best.bound = innerCandidates710_2 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan710_2 : scan edge (search edge 3) 4 [5407] 5407 innerTail710_2 best = best := by
  rw [innerTail710_2, scan, if_neg (by decide), innerCandidates710_2_checked]
  change scan edge (search edge 3) 4 [5407] 5407 innerTail710_3
    (if 3 ≤ innerCandidates710_2.length then search edge 3 [5407, 5779] innerCandidates710_2 11186 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan710_3

def innerTail710_1 : List Nat := 5743 :: innerTail710_2
def innerCandidates710_1 : List Nat := [8287]
set_option maxRecDepth 4096 in
theorem innerCandidates710_1_checked : candidates edge 4 5407 5743 innerTail710_2 best.bound = innerCandidates710_1 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan710_1 : scan edge (search edge 3) 4 [5407] 5407 innerTail710_1 best = best := by
  rw [innerTail710_1, scan, if_neg (by decide), innerCandidates710_1_checked]
  change scan edge (search edge 3) 4 [5407] 5407 innerTail710_2
    (if 3 ≤ innerCandidates710_1.length then search edge 3 [5407, 5743] innerCandidates710_1 11150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan710_2

def innerTail710_0 : List Nat := 5449 :: innerTail710_1
def innerCandidates710_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates710_0_checked : candidates edge 4 5407 5449 innerTail710_1 best.bound = innerCandidates710_0 := by
  unfold candidates
  have row : edge 5449 = SearchData.row5449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan710_0 : scan edge (search edge 3) 4 [5407] 5407 innerTail710_0 best = best := by
  rw [innerTail710_0, scan, if_neg (by decide), innerCandidates710_0_checked]
  change scan edge (search edge 3) 4 [5407] 5407 innerTail710_1
    (if 3 ≤ innerCandidates710_0.length then search edge 3 [5407, 5449] innerCandidates710_0 10856 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan710_1

theorem rootBranch710 : search edge 4 [5407] rootCandidates710 5407 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5407] 5407 innerTail710_0 best = best
  exact innerScan710_0

def rootCandidates709 : List Nat := [6329, 6473, 7079, 7247, 7523, 7649, 8111, 8117]
set_option maxRecDepth 4096 in
theorem rootCandidates709_checked : candidates edge 5 0 5399 tail710 best.bound = rootCandidates709 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext q; rfl
  rw [row]
  decide

def innerTail709_8 : List Nat := []
theorem innerScan709_8 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_8 best = best := rfl

def innerTail709_7 : List Nat := 8117 :: innerTail709_8
def innerCandidates709_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates709_7_checked : candidates edge 4 5399 8117 innerTail709_8 best.bound = innerCandidates709_7 := by
  unfold candidates
  have row : edge 8117 = SearchData.row8117.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_7 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_7 best = best := by
  rw [innerTail709_7, scan, if_neg (by decide), innerCandidates709_7_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_8
    (if 3 ≤ innerCandidates709_7.length then search edge 3 [5399, 8117] innerCandidates709_7 13516 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_8

def innerTail709_6 : List Nat := 8111 :: innerTail709_7
def innerCandidates709_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates709_6_checked : candidates edge 4 5399 8111 innerTail709_7 best.bound = innerCandidates709_6 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_6 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_6 best = best := by
  rw [innerTail709_6, scan, if_neg (by decide), innerCandidates709_6_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_7
    (if 3 ≤ innerCandidates709_6.length then search edge 3 [5399, 8111] innerCandidates709_6 13510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_7

def innerTail709_5 : List Nat := 7649 :: innerTail709_6
def innerCandidates709_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates709_5_checked : candidates edge 4 5399 7649 innerTail709_6 best.bound = innerCandidates709_5 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_5 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_5 best = best := by
  rw [innerTail709_5, scan, if_neg (by decide), innerCandidates709_5_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_6
    (if 3 ≤ innerCandidates709_5.length then search edge 3 [5399, 7649] innerCandidates709_5 13048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_6

def innerTail709_4 : List Nat := 7523 :: innerTail709_5
def innerCandidates709_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates709_4_checked : candidates edge 4 5399 7523 innerTail709_5 best.bound = innerCandidates709_4 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_4 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_4 best = best := by
  rw [innerTail709_4, scan, if_neg (by decide), innerCandidates709_4_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_5
    (if 3 ≤ innerCandidates709_4.length then search edge 3 [5399, 7523] innerCandidates709_4 12922 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_5

def innerTail709_3 : List Nat := 7247 :: innerTail709_4
def innerCandidates709_3 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates709_3_checked : candidates edge 4 5399 7247 innerTail709_4 best.bound = innerCandidates709_3 := by
  unfold candidates
  have row : edge 7247 = SearchData.row7247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_3 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_3 best = best := by
  rw [innerTail709_3, scan, if_neg (by decide), innerCandidates709_3_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_4
    (if 3 ≤ innerCandidates709_3.length then search edge 3 [5399, 7247] innerCandidates709_3 12646 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_4

def innerTail709_2 : List Nat := 7079 :: innerTail709_3
def innerCandidates709_2 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates709_2_checked : candidates edge 4 5399 7079 innerTail709_3 best.bound = innerCandidates709_2 := by
  unfold candidates
  have row : edge 7079 = SearchData.row7079.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_2 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_2 best = best := by
  rw [innerTail709_2, scan, if_neg (by decide), innerCandidates709_2_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_3
    (if 3 ≤ innerCandidates709_2.length then search edge 3 [5399, 7079] innerCandidates709_2 12478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_3

def innerTail709_1 : List Nat := 6473 :: innerTail709_2
def innerCandidates709_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates709_1_checked : candidates edge 4 5399 6473 innerTail709_2 best.bound = innerCandidates709_1 := by
  unfold candidates
  have row : edge 6473 = SearchData.row6473.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_1 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_1 best = best := by
  rw [innerTail709_1, scan, if_neg (by decide), innerCandidates709_1_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_2
    (if 3 ≤ innerCandidates709_1.length then search edge 3 [5399, 6473] innerCandidates709_1 11872 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_2

def innerTail709_0 : List Nat := 6329 :: innerTail709_1
def innerCandidates709_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates709_0_checked : candidates edge 4 5399 6329 innerTail709_1 best.bound = innerCandidates709_0 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan709_0 : scan edge (search edge 3) 4 [5399] 5399 innerTail709_0 best = best := by
  rw [innerTail709_0, scan, if_neg (by decide), innerCandidates709_0_checked]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_1
    (if 3 ≤ innerCandidates709_0.length then search edge 3 [5399, 6329] innerCandidates709_0 11728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan709_1

theorem rootBranch709 : search edge 4 [5399] rootCandidates709 5399 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5399] 5399 innerTail709_0 best = best
  exact innerScan709_0

def rootCandidates708 : List Nat := [5669, 6521, 7817, 7937, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates708_checked : candidates edge 5 0 5393 tail709 best.bound = rootCandidates708 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext q; rfl
  rw [row]
  decide

def innerTail708_5 : List Nat := []
theorem innerScan708_5 : scan edge (search edge 3) 4 [5393] 5393 innerTail708_5 best = best := rfl

def innerTail708_4 : List Nat := 8369 :: innerTail708_5
def innerCandidates708_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates708_4_checked : candidates edge 4 5393 8369 innerTail708_5 best.bound = innerCandidates708_4 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan708_4 : scan edge (search edge 3) 4 [5393] 5393 innerTail708_4 best = best := by
  rw [innerTail708_4, scan, if_neg (by decide), innerCandidates708_4_checked]
  change scan edge (search edge 3) 4 [5393] 5393 innerTail708_5
    (if 3 ≤ innerCandidates708_4.length then search edge 3 [5393, 8369] innerCandidates708_4 13762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan708_5

def innerTail708_3 : List Nat := 7937 :: innerTail708_4
def innerCandidates708_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates708_3_checked : candidates edge 4 5393 7937 innerTail708_4 best.bound = innerCandidates708_3 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan708_3 : scan edge (search edge 3) 4 [5393] 5393 innerTail708_3 best = best := by
  rw [innerTail708_3, scan, if_neg (by decide), innerCandidates708_3_checked]
  change scan edge (search edge 3) 4 [5393] 5393 innerTail708_4
    (if 3 ≤ innerCandidates708_3.length then search edge 3 [5393, 7937] innerCandidates708_3 13330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan708_4

def innerTail708_2 : List Nat := 7817 :: innerTail708_3
def innerCandidates708_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates708_2_checked : candidates edge 4 5393 7817 innerTail708_3 best.bound = innerCandidates708_2 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan708_2 : scan edge (search edge 3) 4 [5393] 5393 innerTail708_2 best = best := by
  rw [innerTail708_2, scan, if_neg (by decide), innerCandidates708_2_checked]
  change scan edge (search edge 3) 4 [5393] 5393 innerTail708_3
    (if 3 ≤ innerCandidates708_2.length then search edge 3 [5393, 7817] innerCandidates708_2 13210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan708_3

def innerTail708_1 : List Nat := 6521 :: innerTail708_2
def innerCandidates708_1 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates708_1_checked : candidates edge 4 5393 6521 innerTail708_2 best.bound = innerCandidates708_1 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan708_1 : scan edge (search edge 3) 4 [5393] 5393 innerTail708_1 best = best := by
  rw [innerTail708_1, scan, if_neg (by decide), innerCandidates708_1_checked]
  change scan edge (search edge 3) 4 [5393] 5393 innerTail708_2
    (if 3 ≤ innerCandidates708_1.length then search edge 3 [5393, 6521] innerCandidates708_1 11914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan708_2

def innerTail708_0 : List Nat := 5669 :: innerTail708_1
def innerCandidates708_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates708_0_checked : candidates edge 4 5393 5669 innerTail708_1 best.bound = innerCandidates708_0 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan708_0 : scan edge (search edge 3) 4 [5393] 5393 innerTail708_0 best = best := by
  rw [innerTail708_0, scan, if_neg (by decide), innerCandidates708_0_checked]
  change scan edge (search edge 3) 4 [5393] 5393 innerTail708_1
    (if 3 ≤ innerCandidates708_0.length then search edge 3 [5393, 5669] innerCandidates708_0 11062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan708_1

theorem rootBranch708 : search edge 4 [5393] rootCandidates708 5393 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5393] 5393 innerTail708_0 best = best
  exact innerScan708_0

def rootCandidates707 : List Nat := [5399, 5807, 6269, 6353, 6653, 6959, 7127, 7487]
set_option maxRecDepth 4096 in
theorem rootCandidates707_checked : candidates edge 5 0 5387 tail708 best.bound = rootCandidates707 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext q; rfl
  rw [row]
  decide

def innerTail707_8 : List Nat := []
theorem innerScan707_8 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_8 best = best := rfl

def innerTail707_7 : List Nat := 7487 :: innerTail707_8
def innerCandidates707_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_7_checked : candidates edge 4 5387 7487 innerTail707_8 best.bound = innerCandidates707_7 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_7 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_7 best = best := by
  rw [innerTail707_7, scan, if_neg (by decide), innerCandidates707_7_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_8
    (if 3 ≤ innerCandidates707_7.length then search edge 3 [5387, 7487] innerCandidates707_7 12874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_8

def innerTail707_6 : List Nat := 7127 :: innerTail707_7
def innerCandidates707_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_6_checked : candidates edge 4 5387 7127 innerTail707_7 best.bound = innerCandidates707_6 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_6 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_6 best = best := by
  rw [innerTail707_6, scan, if_neg (by decide), innerCandidates707_6_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_7
    (if 3 ≤ innerCandidates707_6.length then search edge 3 [5387, 7127] innerCandidates707_6 12514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_7

def innerTail707_5 : List Nat := 6959 :: innerTail707_6
def innerCandidates707_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_5_checked : candidates edge 4 5387 6959 innerTail707_6 best.bound = innerCandidates707_5 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_5 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_5 best = best := by
  rw [innerTail707_5, scan, if_neg (by decide), innerCandidates707_5_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_6
    (if 3 ≤ innerCandidates707_5.length then search edge 3 [5387, 6959] innerCandidates707_5 12346 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_6

def innerTail707_4 : List Nat := 6653 :: innerTail707_5
def innerCandidates707_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_4_checked : candidates edge 4 5387 6653 innerTail707_5 best.bound = innerCandidates707_4 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_4 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_4 best = best := by
  rw [innerTail707_4, scan, if_neg (by decide), innerCandidates707_4_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_5
    (if 3 ≤ innerCandidates707_4.length then search edge 3 [5387, 6653] innerCandidates707_4 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_5

def innerTail707_3 : List Nat := 6353 :: innerTail707_4
def innerCandidates707_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_3_checked : candidates edge 4 5387 6353 innerTail707_4 best.bound = innerCandidates707_3 := by
  unfold candidates
  have row : edge 6353 = SearchData.row6353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_3 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_3 best = best := by
  rw [innerTail707_3, scan, if_neg (by decide), innerCandidates707_3_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_4
    (if 3 ≤ innerCandidates707_3.length then search edge 3 [5387, 6353] innerCandidates707_3 11740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_4

def innerTail707_2 : List Nat := 6269 :: innerTail707_3
def innerCandidates707_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_2_checked : candidates edge 4 5387 6269 innerTail707_3 best.bound = innerCandidates707_2 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_2 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_2 best = best := by
  rw [innerTail707_2, scan, if_neg (by decide), innerCandidates707_2_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_3
    (if 3 ≤ innerCandidates707_2.length then search edge 3 [5387, 6269] innerCandidates707_2 11656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_3

def innerTail707_1 : List Nat := 5807 :: innerTail707_2
def innerCandidates707_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_1_checked : candidates edge 4 5387 5807 innerTail707_2 best.bound = innerCandidates707_1 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_1 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_1 best = best := by
  rw [innerTail707_1, scan, if_neg (by decide), innerCandidates707_1_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_2
    (if 3 ≤ innerCandidates707_1.length then search edge 3 [5387, 5807] innerCandidates707_1 11194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_2

def innerTail707_0 : List Nat := 5399 :: innerTail707_1
def innerCandidates707_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates707_0_checked : candidates edge 4 5387 5399 innerTail707_1 best.bound = innerCandidates707_0 := by
  unfold candidates
  have row : edge 5399 = SearchData.row5399.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan707_0 : scan edge (search edge 3) 4 [5387] 5387 innerTail707_0 best = best := by
  rw [innerTail707_0, scan, if_neg (by decide), innerCandidates707_0_checked]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_1
    (if 3 ≤ innerCandidates707_0.length then search edge 3 [5387, 5399] innerCandidates707_0 10786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan707_1

theorem rootBranch707 : search edge 4 [5387] rootCandidates707 5387 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5387] 5387 innerTail707_0 best = best
  exact innerScan707_0

def rootCandidates706 : List Nat := [5507, 5981, 6011, 6599, 7793, 7877, 7919, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates706_checked : candidates edge 5 0 5381 tail707 best.bound = rootCandidates706 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext q; rfl
  rw [row]
  decide

def innerTail706_8 : List Nat := []
theorem innerScan706_8 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_8 best = best := rfl

def innerTail706_7 : List Nat := 8081 :: innerTail706_8
def innerCandidates706_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates706_7_checked : candidates edge 4 5381 8081 innerTail706_8 best.bound = innerCandidates706_7 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_7 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_7 best = best := by
  rw [innerTail706_7, scan, if_neg (by decide), innerCandidates706_7_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_8
    (if 3 ≤ innerCandidates706_7.length then search edge 3 [5381, 8081] innerCandidates706_7 13462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_8

def innerTail706_6 : List Nat := 7919 :: innerTail706_7
def innerCandidates706_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates706_6_checked : candidates edge 4 5381 7919 innerTail706_7 best.bound = innerCandidates706_6 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_6 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_6 best = best := by
  rw [innerTail706_6, scan, if_neg (by decide), innerCandidates706_6_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_7
    (if 3 ≤ innerCandidates706_6.length then search edge 3 [5381, 7919] innerCandidates706_6 13300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_7

def innerTail706_5 : List Nat := 7877 :: innerTail706_6
def innerCandidates706_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates706_5_checked : candidates edge 4 5381 7877 innerTail706_6 best.bound = innerCandidates706_5 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_5 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_5 best = best := by
  rw [innerTail706_5, scan, if_neg (by decide), innerCandidates706_5_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_6
    (if 3 ≤ innerCandidates706_5.length then search edge 3 [5381, 7877] innerCandidates706_5 13258 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_6

def innerTail706_4 : List Nat := 7793 :: innerTail706_5
def innerCandidates706_4 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates706_4_checked : candidates edge 4 5381 7793 innerTail706_5 best.bound = innerCandidates706_4 := by
  unfold candidates
  have row : edge 7793 = SearchData.row7793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_4 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_4 best = best := by
  rw [innerTail706_4, scan, if_neg (by decide), innerCandidates706_4_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_5
    (if 3 ≤ innerCandidates706_4.length then search edge 3 [5381, 7793] innerCandidates706_4 13174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_5

def innerTail706_3 : List Nat := 6599 :: innerTail706_4
def innerCandidates706_3 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates706_3_checked : candidates edge 4 5381 6599 innerTail706_4 best.bound = innerCandidates706_3 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_3 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_3 best = best := by
  rw [innerTail706_3, scan, if_neg (by decide), innerCandidates706_3_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_4
    (if 3 ≤ innerCandidates706_3.length then search edge 3 [5381, 6599] innerCandidates706_3 11980 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_4

def innerTail706_2 : List Nat := 6011 :: innerTail706_3
def innerCandidates706_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates706_2_checked : candidates edge 4 5381 6011 innerTail706_3 best.bound = innerCandidates706_2 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_2 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_2 best = best := by
  rw [innerTail706_2, scan, if_neg (by decide), innerCandidates706_2_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_3
    (if 3 ≤ innerCandidates706_2.length then search edge 3 [5381, 6011] innerCandidates706_2 11392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_3

def innerTail706_1 : List Nat := 5981 :: innerTail706_2
def innerCandidates706_1 : List Nat := [8081]
set_option maxRecDepth 4096 in
theorem innerCandidates706_1_checked : candidates edge 4 5381 5981 innerTail706_2 best.bound = innerCandidates706_1 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_1 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_1 best = best := by
  rw [innerTail706_1, scan, if_neg (by decide), innerCandidates706_1_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_2
    (if 3 ≤ innerCandidates706_1.length then search edge 3 [5381, 5981] innerCandidates706_1 11362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_2

def innerTail706_0 : List Nat := 5507 :: innerTail706_1
def innerCandidates706_0 : List Nat := [7877]
set_option maxRecDepth 4096 in
theorem innerCandidates706_0_checked : candidates edge 4 5381 5507 innerTail706_1 best.bound = innerCandidates706_0 := by
  unfold candidates
  have row : edge 5507 = SearchData.row5507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan706_0 : scan edge (search edge 3) 4 [5381] 5381 innerTail706_0 best = best := by
  rw [innerTail706_0, scan, if_neg (by decide), innerCandidates706_0_checked]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_1
    (if 3 ≤ innerCandidates706_0.length then search edge 3 [5381, 5507] innerCandidates706_0 10888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan706_1

theorem rootBranch706 : search edge 4 [5381] rootCandidates706 5381 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5381] 5381 innerTail706_0 best = best
  exact innerScan706_0

def rootCandidates705 : List Nat := [5519, 5879, 5927, 6011, 6131, 6761, 6899, 7019, 8039, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates705_checked : candidates edge 5 0 5351 tail706 best.bound = rootCandidates705 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext q; rfl
  rw [row]
  decide

def innerTail705_10 : List Nat := []
theorem innerScan705_10 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_10 best = best := rfl

def innerTail705_9 : List Nat := 8291 :: innerTail705_10
def innerCandidates705_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_9_checked : candidates edge 4 5351 8291 innerTail705_10 best.bound = innerCandidates705_9 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_9 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_9 best = best := by
  rw [innerTail705_9, scan, if_neg (by decide), innerCandidates705_9_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_10
    (if 3 ≤ innerCandidates705_9.length then search edge 3 [5351, 8291] innerCandidates705_9 13642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_10

def innerTail705_8 : List Nat := 8039 :: innerTail705_9
def innerCandidates705_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_8_checked : candidates edge 4 5351 8039 innerTail705_9 best.bound = innerCandidates705_8 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_8 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_8 best = best := by
  rw [innerTail705_8, scan, if_neg (by decide), innerCandidates705_8_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_9
    (if 3 ≤ innerCandidates705_8.length then search edge 3 [5351, 8039] innerCandidates705_8 13390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_9

def innerTail705_7 : List Nat := 7019 :: innerTail705_8
def innerCandidates705_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_7_checked : candidates edge 4 5351 7019 innerTail705_8 best.bound = innerCandidates705_7 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_7 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_7 best = best := by
  rw [innerTail705_7, scan, if_neg (by decide), innerCandidates705_7_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_8
    (if 3 ≤ innerCandidates705_7.length then search edge 3 [5351, 7019] innerCandidates705_7 12370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_8

def innerTail705_6 : List Nat := 6899 :: innerTail705_7
def innerCandidates705_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_6_checked : candidates edge 4 5351 6899 innerTail705_7 best.bound = innerCandidates705_6 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_6 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_6 best = best := by
  rw [innerTail705_6, scan, if_neg (by decide), innerCandidates705_6_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_7
    (if 3 ≤ innerCandidates705_6.length then search edge 3 [5351, 6899] innerCandidates705_6 12250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_7

def innerTail705_5 : List Nat := 6761 :: innerTail705_6
def innerCandidates705_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_5_checked : candidates edge 4 5351 6761 innerTail705_6 best.bound = innerCandidates705_5 := by
  unfold candidates
  have row : edge 6761 = SearchData.row6761.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_5 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_5 best = best := by
  rw [innerTail705_5, scan, if_neg (by decide), innerCandidates705_5_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_6
    (if 3 ≤ innerCandidates705_5.length then search edge 3 [5351, 6761] innerCandidates705_5 12112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_6

def innerTail705_4 : List Nat := 6131 :: innerTail705_5
def innerCandidates705_4 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates705_4_checked : candidates edge 4 5351 6131 innerTail705_5 best.bound = innerCandidates705_4 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_4 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_4 best = best := by
  rw [innerTail705_4, scan, if_neg (by decide), innerCandidates705_4_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_5
    (if 3 ≤ innerCandidates705_4.length then search edge 3 [5351, 6131] innerCandidates705_4 11482 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_5

def innerTail705_3 : List Nat := 6011 :: innerTail705_4
def innerCandidates705_3 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates705_3_checked : candidates edge 4 5351 6011 innerTail705_4 best.bound = innerCandidates705_3 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_3 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_3 best = best := by
  rw [innerTail705_3, scan, if_neg (by decide), innerCandidates705_3_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_4
    (if 3 ≤ innerCandidates705_3.length then search edge 3 [5351, 6011] innerCandidates705_3 11362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_4

def innerTail705_2 : List Nat := 5927 :: innerTail705_3
def innerCandidates705_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_2_checked : candidates edge 4 5351 5927 innerTail705_3 best.bound = innerCandidates705_2 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_2 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_2 best = best := by
  rw [innerTail705_2, scan, if_neg (by decide), innerCandidates705_2_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_3
    (if 3 ≤ innerCandidates705_2.length then search edge 3 [5351, 5927] innerCandidates705_2 11278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_3

def innerTail705_1 : List Nat := 5879 :: innerTail705_2
def innerCandidates705_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_1_checked : candidates edge 4 5351 5879 innerTail705_2 best.bound = innerCandidates705_1 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_1 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_1 best = best := by
  rw [innerTail705_1, scan, if_neg (by decide), innerCandidates705_1_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_2
    (if 3 ≤ innerCandidates705_1.length then search edge 3 [5351, 5879] innerCandidates705_1 11230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_2

def innerTail705_0 : List Nat := 5519 :: innerTail705_1
def innerCandidates705_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates705_0_checked : candidates edge 4 5351 5519 innerTail705_1 best.bound = innerCandidates705_0 := by
  unfold candidates
  have row : edge 5519 = SearchData.row5519.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan705_0 : scan edge (search edge 3) 4 [5351] 5351 innerTail705_0 best = best := by
  rw [innerTail705_0, scan, if_neg (by decide), innerCandidates705_0_checked]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_1
    (if 3 ≤ innerCandidates705_0.length then search edge 3 [5351, 5519] innerCandidates705_0 10870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan705_1

theorem rootBranch705 : search edge 4 [5351] rootCandidates705 5351 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5351] 5351 innerTail705_0 best = best
  exact innerScan705_0

end ProjectEulerSolutions.PrimePairUpper.Search
