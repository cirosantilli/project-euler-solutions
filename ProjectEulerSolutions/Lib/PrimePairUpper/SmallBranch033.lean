import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates783 : List Nat := [6221, 6389, 6491, 7529, 7607, 7649]
set_option maxRecDepth 4096 in
theorem rootCandidates783_checked : candidates edge 5 0 6029 tail784 best.bound = rootCandidates783 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext q; rfl
  rw [row]
  decide

def innerTail783_6 : List Nat := []
theorem innerScan783_6 : scan edge (search edge 3) 4 [6029] 6029 innerTail783_6 best = best := rfl

def innerTail783_5 : List Nat := 7649 :: innerTail783_6
def innerCandidates783_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates783_5_checked : candidates edge 4 6029 7649 innerTail783_6 best.bound = innerCandidates783_5 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan783_5 : scan edge (search edge 3) 4 [6029] 6029 innerTail783_5 best = best := by
  rw [innerTail783_5, scan, if_neg (by decide), innerCandidates783_5_checked]
  change scan edge (search edge 3) 4 [6029] 6029 innerTail783_6
    (if 3 ≤ innerCandidates783_5.length then search edge 3 [6029, 7649] innerCandidates783_5 13678 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan783_6

def innerTail783_4 : List Nat := 7607 :: innerTail783_5
def innerCandidates783_4 : List Nat := [7649]
set_option maxRecDepth 4096 in
theorem innerCandidates783_4_checked : candidates edge 4 6029 7607 innerTail783_5 best.bound = innerCandidates783_4 := by
  unfold candidates
  have row : edge 7607 = SearchData.row7607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan783_4 : scan edge (search edge 3) 4 [6029] 6029 innerTail783_4 best = best := by
  rw [innerTail783_4, scan, if_neg (by decide), innerCandidates783_4_checked]
  change scan edge (search edge 3) 4 [6029] 6029 innerTail783_5
    (if 3 ≤ innerCandidates783_4.length then search edge 3 [6029, 7607] innerCandidates783_4 13636 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan783_5

def innerTail783_3 : List Nat := 7529 :: innerTail783_4
def innerCandidates783_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates783_3_checked : candidates edge 4 6029 7529 innerTail783_4 best.bound = innerCandidates783_3 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan783_3 : scan edge (search edge 3) 4 [6029] 6029 innerTail783_3 best = best := by
  rw [innerTail783_3, scan, if_neg (by decide), innerCandidates783_3_checked]
  change scan edge (search edge 3) 4 [6029] 6029 innerTail783_4
    (if 3 ≤ innerCandidates783_3.length then search edge 3 [6029, 7529] innerCandidates783_3 13558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan783_4

def innerTail783_2 : List Nat := 6491 :: innerTail783_3
def innerCandidates783_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates783_2_checked : candidates edge 4 6029 6491 innerTail783_3 best.bound = innerCandidates783_2 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan783_2 : scan edge (search edge 3) 4 [6029] 6029 innerTail783_2 best = best := by
  rw [innerTail783_2, scan, if_neg (by decide), innerCandidates783_2_checked]
  change scan edge (search edge 3) 4 [6029] 6029 innerTail783_3
    (if 3 ≤ innerCandidates783_2.length then search edge 3 [6029, 6491] innerCandidates783_2 12520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan783_3

def innerTail783_1 : List Nat := 6389 :: innerTail783_2
def innerCandidates783_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates783_1_checked : candidates edge 4 6029 6389 innerTail783_2 best.bound = innerCandidates783_1 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan783_1 : scan edge (search edge 3) 4 [6029] 6029 innerTail783_1 best = best := by
  rw [innerTail783_1, scan, if_neg (by decide), innerCandidates783_1_checked]
  change scan edge (search edge 3) 4 [6029] 6029 innerTail783_2
    (if 3 ≤ innerCandidates783_1.length then search edge 3 [6029, 6389] innerCandidates783_1 12418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan783_2

def innerTail783_0 : List Nat := 6221 :: innerTail783_1
def innerCandidates783_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates783_0_checked : candidates edge 4 6029 6221 innerTail783_1 best.bound = innerCandidates783_0 := by
  unfold candidates
  have row : edge 6221 = SearchData.row6221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan783_0 : scan edge (search edge 3) 4 [6029] 6029 innerTail783_0 best = best := by
  rw [innerTail783_0, scan, if_neg (by decide), innerCandidates783_0_checked]
  change scan edge (search edge 3) 4 [6029] 6029 innerTail783_1
    (if 3 ≤ innerCandidates783_0.length then search edge 3 [6029, 6221] innerCandidates783_0 12250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan783_1

theorem rootBranch783 : search edge 4 [6029] rootCandidates783 6029 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6029] 6029 innerTail783_0 best = best
  exact innerScan783_0

def rootCandidates782 : List Nat := [7019, 7307, 7451, 7727]
set_option maxRecDepth 4096 in
theorem rootCandidates782_checked : candidates edge 5 0 6011 tail783 best.bound = rootCandidates782 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext q; rfl
  rw [row]
  decide

def innerTail782_4 : List Nat := []
theorem innerScan782_4 : scan edge (search edge 3) 4 [6011] 6011 innerTail782_4 best = best := rfl

def innerTail782_3 : List Nat := 7727 :: innerTail782_4
def innerCandidates782_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates782_3_checked : candidates edge 4 6011 7727 innerTail782_4 best.bound = innerCandidates782_3 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan782_3 : scan edge (search edge 3) 4 [6011] 6011 innerTail782_3 best = best := by
  rw [innerTail782_3, scan, if_neg (by decide), innerCandidates782_3_checked]
  change scan edge (search edge 3) 4 [6011] 6011 innerTail782_4
    (if 3 ≤ innerCandidates782_3.length then search edge 3 [6011, 7727] innerCandidates782_3 13738 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan782_4

def innerTail782_2 : List Nat := 7451 :: innerTail782_3
def innerCandidates782_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates782_2_checked : candidates edge 4 6011 7451 innerTail782_3 best.bound = innerCandidates782_2 := by
  unfold candidates
  have row : edge 7451 = SearchData.row7451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan782_2 : scan edge (search edge 3) 4 [6011] 6011 innerTail782_2 best = best := by
  rw [innerTail782_2, scan, if_neg (by decide), innerCandidates782_2_checked]
  change scan edge (search edge 3) 4 [6011] 6011 innerTail782_3
    (if 3 ≤ innerCandidates782_2.length then search edge 3 [6011, 7451] innerCandidates782_2 13462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan782_3

def innerTail782_1 : List Nat := 7307 :: innerTail782_2
def innerCandidates782_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates782_1_checked : candidates edge 4 6011 7307 innerTail782_2 best.bound = innerCandidates782_1 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan782_1 : scan edge (search edge 3) 4 [6011] 6011 innerTail782_1 best = best := by
  rw [innerTail782_1, scan, if_neg (by decide), innerCandidates782_1_checked]
  change scan edge (search edge 3) 4 [6011] 6011 innerTail782_2
    (if 3 ≤ innerCandidates782_1.length then search edge 3 [6011, 7307] innerCandidates782_1 13318 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan782_2

def innerTail782_0 : List Nat := 7019 :: innerTail782_1
def innerCandidates782_0 : List Nat := [7451]
set_option maxRecDepth 4096 in
theorem innerCandidates782_0_checked : candidates edge 4 6011 7019 innerTail782_1 best.bound = innerCandidates782_0 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan782_0 : scan edge (search edge 3) 4 [6011] 6011 innerTail782_0 best = best := by
  rw [innerTail782_0, scan, if_neg (by decide), innerCandidates782_0_checked]
  change scan edge (search edge 3) 4 [6011] 6011 innerTail782_1
    (if 3 ≤ innerCandidates782_0.length then search edge 3 [6011, 7019] innerCandidates782_0 13030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan782_1

theorem rootBranch782 : search edge 4 [6011] rootCandidates782 6011 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6011] 6011 innerTail782_0 best = best
  exact innerScan782_0

def rootCandidates781 : List Nat := [6073, 6091, 7057, 7333, 7489, 7681, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates781_checked : candidates edge 5 0 6007 tail782 best.bound = rootCandidates781 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext q; rfl
  rw [row]
  decide

def innerTail781_7 : List Nat := []
theorem innerScan781_7 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_7 best = best := rfl

def innerTail781_6 : List Nat := 7933 :: innerTail781_7
def innerCandidates781_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates781_6_checked : candidates edge 4 6007 7933 innerTail781_7 best.bound = innerCandidates781_6 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan781_6 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_6 best = best := by
  rw [innerTail781_6, scan, if_neg (by decide), innerCandidates781_6_checked]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_7
    (if 3 ≤ innerCandidates781_6.length then search edge 3 [6007, 7933] innerCandidates781_6 13940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan781_7

def innerTail781_5 : List Nat := 7681 :: innerTail781_6
def innerCandidates781_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates781_5_checked : candidates edge 4 6007 7681 innerTail781_6 best.bound = innerCandidates781_5 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan781_5 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_5 best = best := by
  rw [innerTail781_5, scan, if_neg (by decide), innerCandidates781_5_checked]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_6
    (if 3 ≤ innerCandidates781_5.length then search edge 3 [6007, 7681] innerCandidates781_5 13688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan781_6

def innerTail781_4 : List Nat := 7489 :: innerTail781_5
def innerCandidates781_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates781_4_checked : candidates edge 4 6007 7489 innerTail781_5 best.bound = innerCandidates781_4 := by
  unfold candidates
  have row : edge 7489 = SearchData.row7489.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan781_4 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_4 best = best := by
  rw [innerTail781_4, scan, if_neg (by decide), innerCandidates781_4_checked]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_5
    (if 3 ≤ innerCandidates781_4.length then search edge 3 [6007, 7489] innerCandidates781_4 13496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan781_5

def innerTail781_3 : List Nat := 7333 :: innerTail781_4
def innerCandidates781_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates781_3_checked : candidates edge 4 6007 7333 innerTail781_4 best.bound = innerCandidates781_3 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan781_3 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_3 best = best := by
  rw [innerTail781_3, scan, if_neg (by decide), innerCandidates781_3_checked]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_4
    (if 3 ≤ innerCandidates781_3.length then search edge 3 [6007, 7333] innerCandidates781_3 13340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan781_4

def innerTail781_2 : List Nat := 7057 :: innerTail781_3
def innerCandidates781_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates781_2_checked : candidates edge 4 6007 7057 innerTail781_3 best.bound = innerCandidates781_2 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan781_2 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_2 best = best := by
  rw [innerTail781_2, scan, if_neg (by decide), innerCandidates781_2_checked]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_3
    (if 3 ≤ innerCandidates781_2.length then search edge 3 [6007, 7057] innerCandidates781_2 13064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan781_3

def innerTail781_1 : List Nat := 6091 :: innerTail781_2
def innerCandidates781_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates781_1_checked : candidates edge 4 6007 6091 innerTail781_2 best.bound = innerCandidates781_1 := by
  unfold candidates
  have row : edge 6091 = SearchData.row6091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan781_1 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_1 best = best := by
  rw [innerTail781_1, scan, if_neg (by decide), innerCandidates781_1_checked]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_2
    (if 3 ≤ innerCandidates781_1.length then search edge 3 [6007, 6091] innerCandidates781_1 12098 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan781_2

def innerTail781_0 : List Nat := 6073 :: innerTail781_1
def innerCandidates781_0 : List Nat := [7681]
set_option maxRecDepth 4096 in
theorem innerCandidates781_0_checked : candidates edge 4 6007 6073 innerTail781_1 best.bound = innerCandidates781_0 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan781_0 : scan edge (search edge 3) 4 [6007] 6007 innerTail781_0 best = best := by
  rw [innerTail781_0, scan, if_neg (by decide), innerCandidates781_0_checked]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_1
    (if 3 ≤ innerCandidates781_0.length then search edge 3 [6007, 6073] innerCandidates781_0 12080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan781_1

theorem rootBranch781 : search edge 4 [6007] rootCandidates781 6007 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6007] 6007 innerTail781_0 best = best
  exact innerScan781_0

def rootCandidates780 : List Nat := [6389, 6491, 7817, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates780_checked : candidates edge 5 0 5987 tail781 best.bound = rootCandidates780 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext q; rfl
  rw [row]
  decide

def innerTail780_4 : List Nat := []
theorem innerScan780_4 : scan edge (search edge 3) 4 [5987] 5987 innerTail780_4 best = best := rfl

def innerTail780_3 : List Nat := 8363 :: innerTail780_4
def innerCandidates780_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates780_3_checked : candidates edge 4 5987 8363 innerTail780_4 best.bound = innerCandidates780_3 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan780_3 : scan edge (search edge 3) 4 [5987] 5987 innerTail780_3 best = best := by
  rw [innerTail780_3, scan, if_neg (by decide), innerCandidates780_3_checked]
  change scan edge (search edge 3) 4 [5987] 5987 innerTail780_4
    (if 3 ≤ innerCandidates780_3.length then search edge 3 [5987, 8363] innerCandidates780_3 14350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan780_4

def innerTail780_2 : List Nat := 7817 :: innerTail780_3
def innerCandidates780_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates780_2_checked : candidates edge 4 5987 7817 innerTail780_3 best.bound = innerCandidates780_2 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan780_2 : scan edge (search edge 3) 4 [5987] 5987 innerTail780_2 best = best := by
  rw [innerTail780_2, scan, if_neg (by decide), innerCandidates780_2_checked]
  change scan edge (search edge 3) 4 [5987] 5987 innerTail780_3
    (if 3 ≤ innerCandidates780_2.length then search edge 3 [5987, 7817] innerCandidates780_2 13804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan780_3

def innerTail780_1 : List Nat := 6491 :: innerTail780_2
def innerCandidates780_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates780_1_checked : candidates edge 4 5987 6491 innerTail780_2 best.bound = innerCandidates780_1 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan780_1 : scan edge (search edge 3) 4 [5987] 5987 innerTail780_1 best = best := by
  rw [innerTail780_1, scan, if_neg (by decide), innerCandidates780_1_checked]
  change scan edge (search edge 3) 4 [5987] 5987 innerTail780_2
    (if 3 ≤ innerCandidates780_1.length then search edge 3 [5987, 6491] innerCandidates780_1 12478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan780_2

def innerTail780_0 : List Nat := 6389 :: innerTail780_1
def innerCandidates780_0 : List Nat := [8363]
set_option maxRecDepth 4096 in
theorem innerCandidates780_0_checked : candidates edge 4 5987 6389 innerTail780_1 best.bound = innerCandidates780_0 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan780_0 : scan edge (search edge 3) 4 [5987] 5987 innerTail780_0 best = best := by
  rw [innerTail780_0, scan, if_neg (by decide), innerCandidates780_0_checked]
  change scan edge (search edge 3) 4 [5987] 5987 innerTail780_1
    (if 3 ≤ innerCandidates780_0.length then search edge 3 [5987, 6389] innerCandidates780_0 12376 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan780_1

theorem rootBranch780 : search edge 4 [5987] rootCandidates780 5987 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5987] 5987 innerTail780_0 best = best
  exact innerScan780_0

def rootCandidates779 : List Nat := [7283, 8081, 8237, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates779_checked : candidates edge 5 0 5981 tail780 best.bound = rootCandidates779 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext q; rfl
  rw [row]
  decide

def innerTail779_4 : List Nat := []
theorem innerScan779_4 : scan edge (search edge 3) 4 [5981] 5981 innerTail779_4 best = best := rfl

def innerTail779_3 : List Nat := 8273 :: innerTail779_4
def innerCandidates779_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates779_3_checked : candidates edge 4 5981 8273 innerTail779_4 best.bound = innerCandidates779_3 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan779_3 : scan edge (search edge 3) 4 [5981] 5981 innerTail779_3 best = best := by
  rw [innerTail779_3, scan, if_neg (by decide), innerCandidates779_3_checked]
  change scan edge (search edge 3) 4 [5981] 5981 innerTail779_4
    (if 3 ≤ innerCandidates779_3.length then search edge 3 [5981, 8273] innerCandidates779_3 14254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan779_4

def innerTail779_2 : List Nat := 8237 :: innerTail779_3
def innerCandidates779_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates779_2_checked : candidates edge 4 5981 8237 innerTail779_3 best.bound = innerCandidates779_2 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan779_2 : scan edge (search edge 3) 4 [5981] 5981 innerTail779_2 best = best := by
  rw [innerTail779_2, scan, if_neg (by decide), innerCandidates779_2_checked]
  change scan edge (search edge 3) 4 [5981] 5981 innerTail779_3
    (if 3 ≤ innerCandidates779_2.length then search edge 3 [5981, 8237] innerCandidates779_2 14218 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan779_3

def innerTail779_1 : List Nat := 8081 :: innerTail779_2
def innerCandidates779_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates779_1_checked : candidates edge 4 5981 8081 innerTail779_2 best.bound = innerCandidates779_1 := by
  unfold candidates
  have row : edge 8081 = SearchData.row8081.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan779_1 : scan edge (search edge 3) 4 [5981] 5981 innerTail779_1 best = best := by
  rw [innerTail779_1, scan, if_neg (by decide), innerCandidates779_1_checked]
  change scan edge (search edge 3) 4 [5981] 5981 innerTail779_2
    (if 3 ≤ innerCandidates779_1.length then search edge 3 [5981, 8081] innerCandidates779_1 14062 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan779_2

def innerTail779_0 : List Nat := 7283 :: innerTail779_1
def innerCandidates779_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates779_0_checked : candidates edge 4 5981 7283 innerTail779_1 best.bound = innerCandidates779_0 := by
  unfold candidates
  have row : edge 7283 = SearchData.row7283.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan779_0 : scan edge (search edge 3) 4 [5981] 5981 innerTail779_0 best = best := by
  rw [innerTail779_0, scan, if_neg (by decide), innerCandidates779_0_checked]
  change scan edge (search edge 3) 4 [5981] 5981 innerTail779_1
    (if 3 ≤ innerCandidates779_0.length then search edge 3 [5981, 7283] innerCandidates779_0 13264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan779_1

theorem rootBranch779 : search edge 4 [5981] rootCandidates779 5981 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5981] 5981 innerTail779_0 best = best
  exact innerScan779_0

def rootCandidates778 : List Nat := [6199, 6277, 7039]
set_option maxRecDepth 4096 in
theorem rootCandidates778_checked : candidates edge 5 0 5953 tail779 best.bound = rootCandidates778 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates777 : List Nat := [6053, 6101, 6173, 6269, 6551, 7529, 7817, 8243]
set_option maxRecDepth 4096 in
theorem rootCandidates777_checked : candidates edge 5 0 5939 tail778 best.bound = rootCandidates777 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext q; rfl
  rw [row]
  decide

def innerTail777_8 : List Nat := []
theorem innerScan777_8 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_8 best = best := rfl

def innerTail777_7 : List Nat := 8243 :: innerTail777_8
def innerCandidates777_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates777_7_checked : candidates edge 4 5939 8243 innerTail777_8 best.bound = innerCandidates777_7 := by
  unfold candidates
  have row : edge 8243 = SearchData.row8243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_7 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_7 best = best := by
  rw [innerTail777_7, scan, if_neg (by decide), innerCandidates777_7_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_8
    (if 3 ≤ innerCandidates777_7.length then search edge 3 [5939, 8243] innerCandidates777_7 14182 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_8

def innerTail777_6 : List Nat := 7817 :: innerTail777_7
def innerCandidates777_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates777_6_checked : candidates edge 4 5939 7817 innerTail777_7 best.bound = innerCandidates777_6 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_6 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_6 best = best := by
  rw [innerTail777_6, scan, if_neg (by decide), innerCandidates777_6_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_7
    (if 3 ≤ innerCandidates777_6.length then search edge 3 [5939, 7817] innerCandidates777_6 13756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_7

def innerTail777_5 : List Nat := 7529 :: innerTail777_6
def innerCandidates777_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates777_5_checked : candidates edge 4 5939 7529 innerTail777_6 best.bound = innerCandidates777_5 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_5 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_5 best = best := by
  rw [innerTail777_5, scan, if_neg (by decide), innerCandidates777_5_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_6
    (if 3 ≤ innerCandidates777_5.length then search edge 3 [5939, 7529] innerCandidates777_5 13468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_6

def innerTail777_4 : List Nat := 6551 :: innerTail777_5
def innerCandidates777_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates777_4_checked : candidates edge 4 5939 6551 innerTail777_5 best.bound = innerCandidates777_4 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_4 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_4 best = best := by
  rw [innerTail777_4, scan, if_neg (by decide), innerCandidates777_4_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_5
    (if 3 ≤ innerCandidates777_4.length then search edge 3 [5939, 6551] innerCandidates777_4 12490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_5

def innerTail777_3 : List Nat := 6269 :: innerTail777_4
def innerCandidates777_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates777_3_checked : candidates edge 4 5939 6269 innerTail777_4 best.bound = innerCandidates777_3 := by
  unfold candidates
  have row : edge 6269 = SearchData.row6269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_3 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_3 best = best := by
  rw [innerTail777_3, scan, if_neg (by decide), innerCandidates777_3_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_4
    (if 3 ≤ innerCandidates777_3.length then search edge 3 [5939, 6269] innerCandidates777_3 12208 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_4

def innerTail777_2 : List Nat := 6173 :: innerTail777_3
def innerCandidates777_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates777_2_checked : candidates edge 4 5939 6173 innerTail777_3 best.bound = innerCandidates777_2 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_2 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_2 best = best := by
  rw [innerTail777_2, scan, if_neg (by decide), innerCandidates777_2_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_3
    (if 3 ≤ innerCandidates777_2.length then search edge 3 [5939, 6173] innerCandidates777_2 12112 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_3

def innerTail777_1 : List Nat := 6101 :: innerTail777_2
def innerCandidates777_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates777_1_checked : candidates edge 4 5939 6101 innerTail777_2 best.bound = innerCandidates777_1 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_1 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_1 best = best := by
  rw [innerTail777_1, scan, if_neg (by decide), innerCandidates777_1_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_2
    (if 3 ≤ innerCandidates777_1.length then search edge 3 [5939, 6101] innerCandidates777_1 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_2

def innerTail777_0 : List Nat := 6053 :: innerTail777_1
def innerCandidates777_0 : List Nat := [6173]
set_option maxRecDepth 4096 in
theorem innerCandidates777_0_checked : candidates edge 4 5939 6053 innerTail777_1 best.bound = innerCandidates777_0 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan777_0 : scan edge (search edge 3) 4 [5939] 5939 innerTail777_0 best = best := by
  rw [innerTail777_0, scan, if_neg (by decide), innerCandidates777_0_checked]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_1
    (if 3 ≤ innerCandidates777_0.length then search edge 3 [5939, 6053] innerCandidates777_0 11992 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan777_1

theorem rootBranch777 : search edge 4 [5939] rootCandidates777 5939 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5939] 5939 innerTail777_0 best = best
  exact innerScan777_0

def rootCandidates776 : List Nat := [5939, 7499, 7583, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates776_checked : candidates edge 5 0 5927 tail777 best.bound = rootCandidates776 := by
  unfold candidates
  have row : edge 5927 = SearchData.row5927.contains := by funext q; rfl
  rw [row]
  decide

def innerTail776_4 : List Nat := []
theorem innerScan776_4 : scan edge (search edge 3) 4 [5927] 5927 innerTail776_4 best = best := rfl

def innerTail776_3 : List Nat := 8237 :: innerTail776_4
def innerCandidates776_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates776_3_checked : candidates edge 4 5927 8237 innerTail776_4 best.bound = innerCandidates776_3 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan776_3 : scan edge (search edge 3) 4 [5927] 5927 innerTail776_3 best = best := by
  rw [innerTail776_3, scan, if_neg (by decide), innerCandidates776_3_checked]
  change scan edge (search edge 3) 4 [5927] 5927 innerTail776_4
    (if 3 ≤ innerCandidates776_3.length then search edge 3 [5927, 8237] innerCandidates776_3 14164 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan776_4

def innerTail776_2 : List Nat := 7583 :: innerTail776_3
def innerCandidates776_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates776_2_checked : candidates edge 4 5927 7583 innerTail776_3 best.bound = innerCandidates776_2 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan776_2 : scan edge (search edge 3) 4 [5927] 5927 innerTail776_2 best = best := by
  rw [innerTail776_2, scan, if_neg (by decide), innerCandidates776_2_checked]
  change scan edge (search edge 3) 4 [5927] 5927 innerTail776_3
    (if 3 ≤ innerCandidates776_2.length then search edge 3 [5927, 7583] innerCandidates776_2 13510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan776_3

def innerTail776_1 : List Nat := 7499 :: innerTail776_2
def innerCandidates776_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates776_1_checked : candidates edge 4 5927 7499 innerTail776_2 best.bound = innerCandidates776_1 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan776_1 : scan edge (search edge 3) 4 [5927] 5927 innerTail776_1 best = best := by
  rw [innerTail776_1, scan, if_neg (by decide), innerCandidates776_1_checked]
  change scan edge (search edge 3) 4 [5927] 5927 innerTail776_2
    (if 3 ≤ innerCandidates776_1.length then search edge 3 [5927, 7499] innerCandidates776_1 13426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan776_2

def innerTail776_0 : List Nat := 5939 :: innerTail776_1
def innerCandidates776_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates776_0_checked : candidates edge 4 5927 5939 innerTail776_1 best.bound = innerCandidates776_0 := by
  unfold candidates
  have row : edge 5939 = SearchData.row5939.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan776_0 : scan edge (search edge 3) 4 [5927] 5927 innerTail776_0 best = best := by
  rw [innerTail776_0, scan, if_neg (by decide), innerCandidates776_0_checked]
  change scan edge (search edge 3) 4 [5927] 5927 innerTail776_1
    (if 3 ≤ innerCandidates776_0.length then search edge 3 [5927, 5939] innerCandidates776_0 11866 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan776_1

theorem rootBranch776 : search edge 4 [5927] rootCandidates776 5927 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5927] 5927 innerTail776_0 best = best
  exact innerScan776_0

end ProjectEulerSolutions.PrimePairUpper.Search
