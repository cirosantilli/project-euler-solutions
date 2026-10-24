import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates574 : List Nat := [4337, 4421, 4673, 4751, 5171, 5981, 7823, 7829, 8111, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates574_checked : candidates edge 5 0 4217 tail575 best.bound = rootCandidates574 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext q; rfl
  rw [row]
  decide

def innerTail574_10 : List Nat := []
theorem innerScan574_10 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_10 best = best := rfl

def innerTail574_9 : List Nat := 8363 :: innerTail574_10
def innerCandidates574_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_9_checked : candidates edge 4 4217 8363 innerTail574_10 best.bound = innerCandidates574_9 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_9 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_9 best = best := by
  rw [innerTail574_9, scan, if_neg (by decide), innerCandidates574_9_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_10
    (if 3 ≤ innerCandidates574_9.length then search edge 3 [4217, 8363] innerCandidates574_9 12580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_10

def innerTail574_8 : List Nat := 8111 :: innerTail574_9
def innerCandidates574_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_8_checked : candidates edge 4 4217 8111 innerTail574_9 best.bound = innerCandidates574_8 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_8 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_8 best = best := by
  rw [innerTail574_8, scan, if_neg (by decide), innerCandidates574_8_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_9
    (if 3 ≤ innerCandidates574_8.length then search edge 3 [4217, 8111] innerCandidates574_8 12328 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_9

def innerTail574_7 : List Nat := 7829 :: innerTail574_8
def innerCandidates574_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_7_checked : candidates edge 4 4217 7829 innerTail574_8 best.bound = innerCandidates574_7 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_7 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_7 best = best := by
  rw [innerTail574_7, scan, if_neg (by decide), innerCandidates574_7_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_8
    (if 3 ≤ innerCandidates574_7.length then search edge 3 [4217, 7829] innerCandidates574_7 12046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_8

def innerTail574_6 : List Nat := 7823 :: innerTail574_7
def innerCandidates574_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_6_checked : candidates edge 4 4217 7823 innerTail574_7 best.bound = innerCandidates574_6 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_6 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_6 best = best := by
  rw [innerTail574_6, scan, if_neg (by decide), innerCandidates574_6_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_7
    (if 3 ≤ innerCandidates574_6.length then search edge 3 [4217, 7823] innerCandidates574_6 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_7

def innerTail574_5 : List Nat := 5981 :: innerTail574_6
def innerCandidates574_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_5_checked : candidates edge 4 4217 5981 innerTail574_6 best.bound = innerCandidates574_5 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_5 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_5 best = best := by
  rw [innerTail574_5, scan, if_neg (by decide), innerCandidates574_5_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_6
    (if 3 ≤ innerCandidates574_5.length then search edge 3 [4217, 5981] innerCandidates574_5 10198 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_6

def innerTail574_4 : List Nat := 5171 :: innerTail574_5
def innerCandidates574_4 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates574_4_checked : candidates edge 4 4217 5171 innerTail574_5 best.bound = innerCandidates574_4 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_4 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_4 best = best := by
  rw [innerTail574_4, scan, if_neg (by decide), innerCandidates574_4_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_5
    (if 3 ≤ innerCandidates574_4.length then search edge 3 [4217, 5171] innerCandidates574_4 9388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_5

def innerTail574_3 : List Nat := 4751 :: innerTail574_4
def innerCandidates574_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_3_checked : candidates edge 4 4217 4751 innerTail574_4 best.bound = innerCandidates574_3 := by
  unfold candidates
  have row : edge 4751 = SearchData.row4751.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_3 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_3 best = best := by
  rw [innerTail574_3, scan, if_neg (by decide), innerCandidates574_3_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_4
    (if 3 ≤ innerCandidates574_3.length then search edge 3 [4217, 4751] innerCandidates574_3 8968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_4

def innerTail574_2 : List Nat := 4673 :: innerTail574_3
def innerCandidates574_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_2_checked : candidates edge 4 4217 4673 innerTail574_3 best.bound = innerCandidates574_2 := by
  unfold candidates
  have row : edge 4673 = SearchData.row4673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_2 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_2 best = best := by
  rw [innerTail574_2, scan, if_neg (by decide), innerCandidates574_2_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_3
    (if 3 ≤ innerCandidates574_2.length then search edge 3 [4217, 4673] innerCandidates574_2 8890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_3

def innerTail574_1 : List Nat := 4421 :: innerTail574_2
def innerCandidates574_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates574_1_checked : candidates edge 4 4217 4421 innerTail574_2 best.bound = innerCandidates574_1 := by
  unfold candidates
  have row : edge 4421 = SearchData.row4421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_1 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_1 best = best := by
  rw [innerTail574_1, scan, if_neg (by decide), innerCandidates574_1_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_2
    (if 3 ≤ innerCandidates574_1.length then search edge 3 [4217, 4421] innerCandidates574_1 8638 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_2

def innerTail574_0 : List Nat := 4337 :: innerTail574_1
def innerCandidates574_0 : List Nat := [4421]
set_option maxRecDepth 4096 in
theorem innerCandidates574_0_checked : candidates edge 4 4217 4337 innerTail574_1 best.bound = innerCandidates574_0 := by
  unfold candidates
  have row : edge 4337 = SearchData.row4337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan574_0 : scan edge (search edge 3) 4 [4217] 4217 innerTail574_0 best = best := by
  rw [innerTail574_0, scan, if_neg (by decide), innerCandidates574_0_checked]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_1
    (if 3 ≤ innerCandidates574_0.length then search edge 3 [4217, 4337] innerCandidates574_0 8554 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan574_1

theorem rootBranch574 : search edge 4 [4217] rootCandidates574 4217 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4217] 4217 innerTail574_0 best = best
  exact innerScan574_0

def rootCandidates573 : List Nat := [5237, 5351, 5483, 5693, 6911, 7703]
set_option maxRecDepth 4096 in
theorem rootCandidates573_checked : candidates edge 5 0 4211 tail574 best.bound = rootCandidates573 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext q; rfl
  rw [row]
  decide

def innerTail573_6 : List Nat := []
theorem innerScan573_6 : scan edge (search edge 3) 4 [4211] 4211 innerTail573_6 best = best := rfl

def innerTail573_5 : List Nat := 7703 :: innerTail573_6
def innerCandidates573_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates573_5_checked : candidates edge 4 4211 7703 innerTail573_6 best.bound = innerCandidates573_5 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan573_5 : scan edge (search edge 3) 4 [4211] 4211 innerTail573_5 best = best := by
  rw [innerTail573_5, scan, if_neg (by decide), innerCandidates573_5_checked]
  change scan edge (search edge 3) 4 [4211] 4211 innerTail573_6
    (if 3 ≤ innerCandidates573_5.length then search edge 3 [4211, 7703] innerCandidates573_5 11914 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan573_6

def innerTail573_4 : List Nat := 6911 :: innerTail573_5
def innerCandidates573_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates573_4_checked : candidates edge 4 4211 6911 innerTail573_5 best.bound = innerCandidates573_4 := by
  unfold candidates
  have row : edge 6911 = SearchData.row6911.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan573_4 : scan edge (search edge 3) 4 [4211] 4211 innerTail573_4 best = best := by
  rw [innerTail573_4, scan, if_neg (by decide), innerCandidates573_4_checked]
  change scan edge (search edge 3) 4 [4211] 4211 innerTail573_5
    (if 3 ≤ innerCandidates573_4.length then search edge 3 [4211, 6911] innerCandidates573_4 11122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan573_5

def innerTail573_3 : List Nat := 5693 :: innerTail573_4
def innerCandidates573_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates573_3_checked : candidates edge 4 4211 5693 innerTail573_4 best.bound = innerCandidates573_3 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan573_3 : scan edge (search edge 3) 4 [4211] 4211 innerTail573_3 best = best := by
  rw [innerTail573_3, scan, if_neg (by decide), innerCandidates573_3_checked]
  change scan edge (search edge 3) 4 [4211] 4211 innerTail573_4
    (if 3 ≤ innerCandidates573_3.length then search edge 3 [4211, 5693] innerCandidates573_3 9904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan573_4

def innerTail573_2 : List Nat := 5483 :: innerTail573_3
def innerCandidates573_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates573_2_checked : candidates edge 4 4211 5483 innerTail573_3 best.bound = innerCandidates573_2 := by
  unfold candidates
  have row : edge 5483 = SearchData.row5483.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan573_2 : scan edge (search edge 3) 4 [4211] 4211 innerTail573_2 best = best := by
  rw [innerTail573_2, scan, if_neg (by decide), innerCandidates573_2_checked]
  change scan edge (search edge 3) 4 [4211] 4211 innerTail573_3
    (if 3 ≤ innerCandidates573_2.length then search edge 3 [4211, 5483] innerCandidates573_2 9694 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan573_3

def innerTail573_1 : List Nat := 5351 :: innerTail573_2
def innerCandidates573_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates573_1_checked : candidates edge 4 4211 5351 innerTail573_2 best.bound = innerCandidates573_1 := by
  unfold candidates
  have row : edge 5351 = SearchData.row5351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan573_1 : scan edge (search edge 3) 4 [4211] 4211 innerTail573_1 best = best := by
  rw [innerTail573_1, scan, if_neg (by decide), innerCandidates573_1_checked]
  change scan edge (search edge 3) 4 [4211] 4211 innerTail573_2
    (if 3 ≤ innerCandidates573_1.length then search edge 3 [4211, 5351] innerCandidates573_1 9562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan573_2

def innerTail573_0 : List Nat := 5237 :: innerTail573_1
def innerCandidates573_0 : List Nat := [5693]
set_option maxRecDepth 4096 in
theorem innerCandidates573_0_checked : candidates edge 4 4211 5237 innerTail573_1 best.bound = innerCandidates573_0 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan573_0 : scan edge (search edge 3) 4 [4211] 4211 innerTail573_0 best = best := by
  rw [innerTail573_0, scan, if_neg (by decide), innerCandidates573_0_checked]
  change scan edge (search edge 3) 4 [4211] 4211 innerTail573_1
    (if 3 ≤ innerCandidates573_0.length then search edge 3 [4211, 5237] innerCandidates573_0 9448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan573_1

theorem rootBranch573 : search edge 4 [4211] rootCandidates573 4211 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4211] 4211 innerTail573_0 best = best
  exact innerScan573_0

def rootCandidates572 : List Nat := [4759, 5419, 5527, 5749, 6199, 6217, 6421, 8161]
set_option maxRecDepth 4096 in
theorem rootCandidates572_checked : candidates edge 5 0 4201 tail573 best.bound = rootCandidates572 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext q; rfl
  rw [row]
  decide

def innerTail572_8 : List Nat := []
theorem innerScan572_8 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_8 best = best := rfl

def innerTail572_7 : List Nat := 8161 :: innerTail572_8
def innerCandidates572_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates572_7_checked : candidates edge 4 4201 8161 innerTail572_8 best.bound = innerCandidates572_7 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_7 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_7 best = best := by
  rw [innerTail572_7, scan, if_neg (by decide), innerCandidates572_7_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_8
    (if 3 ≤ innerCandidates572_7.length then search edge 3 [4201, 8161] innerCandidates572_7 12362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_8

def innerTail572_6 : List Nat := 6421 :: innerTail572_7
def innerCandidates572_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates572_6_checked : candidates edge 4 4201 6421 innerTail572_7 best.bound = innerCandidates572_6 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_6 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_6 best = best := by
  rw [innerTail572_6, scan, if_neg (by decide), innerCandidates572_6_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_7
    (if 3 ≤ innerCandidates572_6.length then search edge 3 [4201, 6421] innerCandidates572_6 10622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_7

def innerTail572_5 : List Nat := 6217 :: innerTail572_6
def innerCandidates572_5 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates572_5_checked : candidates edge 4 4201 6217 innerTail572_6 best.bound = innerCandidates572_5 := by
  unfold candidates
  have row : edge 6217 = SearchData.row6217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_5 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_5 best = best := by
  rw [innerTail572_5, scan, if_neg (by decide), innerCandidates572_5_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_6
    (if 3 ≤ innerCandidates572_5.length then search edge 3 [4201, 6217] innerCandidates572_5 10418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_6

def innerTail572_4 : List Nat := 6199 :: innerTail572_5
def innerCandidates572_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates572_4_checked : candidates edge 4 4201 6199 innerTail572_5 best.bound = innerCandidates572_4 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_4 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_4 best = best := by
  rw [innerTail572_4, scan, if_neg (by decide), innerCandidates572_4_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_5
    (if 3 ≤ innerCandidates572_4.length then search edge 3 [4201, 6199] innerCandidates572_4 10400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_5

def innerTail572_3 : List Nat := 5749 :: innerTail572_4
def innerCandidates572_3 : List Nat := [6199]
set_option maxRecDepth 4096 in
theorem innerCandidates572_3_checked : candidates edge 4 4201 5749 innerTail572_4 best.bound = innerCandidates572_3 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_3 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_3 best = best := by
  rw [innerTail572_3, scan, if_neg (by decide), innerCandidates572_3_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_4
    (if 3 ≤ innerCandidates572_3.length then search edge 3 [4201, 5749] innerCandidates572_3 9950 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_4

def innerTail572_2 : List Nat := 5527 :: innerTail572_3
def innerCandidates572_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates572_2_checked : candidates edge 4 4201 5527 innerTail572_3 best.bound = innerCandidates572_2 := by
  unfold candidates
  have row : edge 5527 = SearchData.row5527.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_2 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_2 best = best := by
  rw [innerTail572_2, scan, if_neg (by decide), innerCandidates572_2_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_3
    (if 3 ≤ innerCandidates572_2.length then search edge 3 [4201, 5527] innerCandidates572_2 9728 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_3

def innerTail572_1 : List Nat := 5419 :: innerTail572_2
def innerCandidates572_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates572_1_checked : candidates edge 4 4201 5419 innerTail572_2 best.bound = innerCandidates572_1 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_1 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_1 best = best := by
  rw [innerTail572_1, scan, if_neg (by decide), innerCandidates572_1_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_2
    (if 3 ≤ innerCandidates572_1.length then search edge 3 [4201, 5419] innerCandidates572_1 9620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_2

def innerTail572_0 : List Nat := 4759 :: innerTail572_1
def innerCandidates572_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates572_0_checked : candidates edge 4 4201 4759 innerTail572_1 best.bound = innerCandidates572_0 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan572_0 : scan edge (search edge 3) 4 [4201] 4201 innerTail572_0 best = best := by
  rw [innerTail572_0, scan, if_neg (by decide), innerCandidates572_0_checked]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_1
    (if 3 ≤ innerCandidates572_0.length then search edge 3 [4201, 4759] innerCandidates572_0 8960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan572_1

theorem rootBranch572 : search edge 4 [4201] rootCandidates572 4201 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4201] 4201 innerTail572_0 best = best
  exact innerScan572_0

def rootCandidates571 : List Nat := [4363, 4903, 5323, 5791, 6379, 6673, 6781, 6883, 7297, 7537, 8053, 8101, 8311, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates571_checked : candidates edge 5 0 4177 tail572 best.bound = rootCandidates571 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext q; rfl
  rw [row]
  decide

def innerTail571_14 : List Nat := []
theorem innerScan571_14 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_14 best = best := rfl

def innerTail571_13 : List Nat := 8353 :: innerTail571_14
def innerCandidates571_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_13_checked : candidates edge 4 4177 8353 innerTail571_14 best.bound = innerCandidates571_13 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_13 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_13 best = best := by
  rw [innerTail571_13, scan, if_neg (by decide), innerCandidates571_13_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_14
    (if 3 ≤ innerCandidates571_13.length then search edge 3 [4177, 8353] innerCandidates571_13 12530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_14

def innerTail571_12 : List Nat := 8311 :: innerTail571_13
def innerCandidates571_12 : List Nat := [8353]
set_option maxRecDepth 4096 in
theorem innerCandidates571_12_checked : candidates edge 4 4177 8311 innerTail571_13 best.bound = innerCandidates571_12 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_12 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_12 best = best := by
  rw [innerTail571_12, scan, if_neg (by decide), innerCandidates571_12_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_13
    (if 3 ≤ innerCandidates571_12.length then search edge 3 [4177, 8311] innerCandidates571_12 12488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_13

def innerTail571_11 : List Nat := 8101 :: innerTail571_12
def innerCandidates571_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_11_checked : candidates edge 4 4177 8101 innerTail571_12 best.bound = innerCandidates571_11 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_11 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_11 best = best := by
  rw [innerTail571_11, scan, if_neg (by decide), innerCandidates571_11_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_12
    (if 3 ≤ innerCandidates571_11.length then search edge 3 [4177, 8101] innerCandidates571_11 12278 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_12

def innerTail571_10 : List Nat := 8053 :: innerTail571_11
def innerCandidates571_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_10_checked : candidates edge 4 4177 8053 innerTail571_11 best.bound = innerCandidates571_10 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_10 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_10 best = best := by
  rw [innerTail571_10, scan, if_neg (by decide), innerCandidates571_10_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_11
    (if 3 ≤ innerCandidates571_10.length then search edge 3 [4177, 8053] innerCandidates571_10 12230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_11

def innerTail571_9 : List Nat := 7537 :: innerTail571_10
def innerCandidates571_9 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates571_9_checked : candidates edge 4 4177 7537 innerTail571_10 best.bound = innerCandidates571_9 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_9 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_9 best = best := by
  rw [innerTail571_9, scan, if_neg (by decide), innerCandidates571_9_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_10
    (if 3 ≤ innerCandidates571_9.length then search edge 3 [4177, 7537] innerCandidates571_9 11714 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_10

def innerTail571_8 : List Nat := 7297 :: innerTail571_9
def innerCandidates571_8 : List Nat := [8053]
set_option maxRecDepth 4096 in
theorem innerCandidates571_8_checked : candidates edge 4 4177 7297 innerTail571_9 best.bound = innerCandidates571_8 := by
  unfold candidates
  have row : edge 7297 = SearchData.row7297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_8 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_8 best = best := by
  rw [innerTail571_8, scan, if_neg (by decide), innerCandidates571_8_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_9
    (if 3 ≤ innerCandidates571_8.length then search edge 3 [4177, 7297] innerCandidates571_8 11474 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_9

def innerTail571_7 : List Nat := 6883 :: innerTail571_8
def innerCandidates571_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_7_checked : candidates edge 4 4177 6883 innerTail571_8 best.bound = innerCandidates571_7 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_7 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_7 best = best := by
  rw [innerTail571_7, scan, if_neg (by decide), innerCandidates571_7_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_8
    (if 3 ≤ innerCandidates571_7.length then search edge 3 [4177, 6883] innerCandidates571_7 11060 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_8

def innerTail571_6 : List Nat := 6781 :: innerTail571_7
def innerCandidates571_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_6_checked : candidates edge 4 4177 6781 innerTail571_7 best.bound = innerCandidates571_6 := by
  unfold candidates
  have row : edge 6781 = SearchData.row6781.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_6 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_6 best = best := by
  rw [innerTail571_6, scan, if_neg (by decide), innerCandidates571_6_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_7
    (if 3 ≤ innerCandidates571_6.length then search edge 3 [4177, 6781] innerCandidates571_6 10958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_7

def innerTail571_5 : List Nat := 6673 :: innerTail571_6
def innerCandidates571_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_5_checked : candidates edge 4 4177 6673 innerTail571_6 best.bound = innerCandidates571_5 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_5 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_5 best = best := by
  rw [innerTail571_5, scan, if_neg (by decide), innerCandidates571_5_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_6
    (if 3 ≤ innerCandidates571_5.length then search edge 3 [4177, 6673] innerCandidates571_5 10850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_6

def innerTail571_4 : List Nat := 6379 :: innerTail571_5
def innerCandidates571_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_4_checked : candidates edge 4 4177 6379 innerTail571_5 best.bound = innerCandidates571_4 := by
  unfold candidates
  have row : edge 6379 = SearchData.row6379.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_4 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_4 best = best := by
  rw [innerTail571_4, scan, if_neg (by decide), innerCandidates571_4_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_5
    (if 3 ≤ innerCandidates571_4.length then search edge 3 [4177, 6379] innerCandidates571_4 10556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_5

def innerTail571_3 : List Nat := 5791 :: innerTail571_4
def innerCandidates571_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_3_checked : candidates edge 4 4177 5791 innerTail571_4 best.bound = innerCandidates571_3 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_3 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_3 best = best := by
  rw [innerTail571_3, scan, if_neg (by decide), innerCandidates571_3_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_4
    (if 3 ≤ innerCandidates571_3.length then search edge 3 [4177, 5791] innerCandidates571_3 9968 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_4

def innerTail571_2 : List Nat := 5323 :: innerTail571_3
def innerCandidates571_2 : List Nat := [7537]
set_option maxRecDepth 4096 in
theorem innerCandidates571_2_checked : candidates edge 4 4177 5323 innerTail571_3 best.bound = innerCandidates571_2 := by
  unfold candidates
  have row : edge 5323 = SearchData.row5323.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_2 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_2 best = best := by
  rw [innerTail571_2, scan, if_neg (by decide), innerCandidates571_2_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_3
    (if 3 ≤ innerCandidates571_2.length then search edge 3 [4177, 5323] innerCandidates571_2 9500 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_3

def innerTail571_1 : List Nat := 4903 :: innerTail571_2
def innerCandidates571_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates571_1_checked : candidates edge 4 4177 4903 innerTail571_2 best.bound = innerCandidates571_1 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_1 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_1 best = best := by
  rw [innerTail571_1, scan, if_neg (by decide), innerCandidates571_1_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_2
    (if 3 ≤ innerCandidates571_1.length then search edge 3 [4177, 4903] innerCandidates571_1 9080 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_2

def innerTail571_0 : List Nat := 4363 :: innerTail571_1
def innerCandidates571_0 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates571_0_checked : candidates edge 4 4177 4363 innerTail571_1 best.bound = innerCandidates571_0 := by
  unfold candidates
  have row : edge 4363 = SearchData.row4363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan571_0 : scan edge (search edge 3) 4 [4177] 4177 innerTail571_0 best = best := by
  rw [innerTail571_0, scan, if_neg (by decide), innerCandidates571_0_checked]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_1
    (if 3 ≤ innerCandidates571_0.length then search edge 3 [4177, 4363] innerCandidates571_0 8540 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan571_1

theorem rootBranch571 : search edge 4 [4177] rootCandidates571 4177 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4177] 4177 innerTail571_0 best = best
  exact innerScan571_0

def rootCandidates570 : List Nat := [4621, 4789, 5011, 5779, 6661, 7573, 7681, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates570_checked : candidates edge 5 0 4159 tail571 best.bound = rootCandidates570 := by
  unfold candidates
  have row : edge 4159 = SearchData.row4159.contains := by funext q; rfl
  rw [row]
  decide

def innerTail570_8 : List Nat := []
theorem innerScan570_8 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_8 best = best := rfl

def innerTail570_7 : List Nat := 8233 :: innerTail570_8
def innerCandidates570_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates570_7_checked : candidates edge 4 4159 8233 innerTail570_8 best.bound = innerCandidates570_7 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_7 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_7 best = best := by
  rw [innerTail570_7, scan, if_neg (by decide), innerCandidates570_7_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_8
    (if 3 ≤ innerCandidates570_7.length then search edge 3 [4159, 8233] innerCandidates570_7 12392 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_8

def innerTail570_6 : List Nat := 7681 :: innerTail570_7
def innerCandidates570_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates570_6_checked : candidates edge 4 4159 7681 innerTail570_7 best.bound = innerCandidates570_6 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_6 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_6 best = best := by
  rw [innerTail570_6, scan, if_neg (by decide), innerCandidates570_6_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_7
    (if 3 ≤ innerCandidates570_6.length then search edge 3 [4159, 7681] innerCandidates570_6 11840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_7

def innerTail570_5 : List Nat := 7573 :: innerTail570_6
def innerCandidates570_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates570_5_checked : candidates edge 4 4159 7573 innerTail570_6 best.bound = innerCandidates570_5 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_5 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_5 best = best := by
  rw [innerTail570_5, scan, if_neg (by decide), innerCandidates570_5_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_6
    (if 3 ≤ innerCandidates570_5.length then search edge 3 [4159, 7573] innerCandidates570_5 11732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_6

def innerTail570_4 : List Nat := 6661 :: innerTail570_5
def innerCandidates570_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates570_4_checked : candidates edge 4 4159 6661 innerTail570_5 best.bound = innerCandidates570_4 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_4 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_4 best = best := by
  rw [innerTail570_4, scan, if_neg (by decide), innerCandidates570_4_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_5
    (if 3 ≤ innerCandidates570_4.length then search edge 3 [4159, 6661] innerCandidates570_4 10820 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_5

def innerTail570_3 : List Nat := 5779 :: innerTail570_4
def innerCandidates570_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates570_3_checked : candidates edge 4 4159 5779 innerTail570_4 best.bound = innerCandidates570_3 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_3 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_3 best = best := by
  rw [innerTail570_3, scan, if_neg (by decide), innerCandidates570_3_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_4
    (if 3 ≤ innerCandidates570_3.length then search edge 3 [4159, 5779] innerCandidates570_3 9938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_4

def innerTail570_2 : List Nat := 5011 :: innerTail570_3
def innerCandidates570_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates570_2_checked : candidates edge 4 4159 5011 innerTail570_3 best.bound = innerCandidates570_2 := by
  unfold candidates
  have row : edge 5011 = SearchData.row5011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_2 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_2 best = best := by
  rw [innerTail570_2, scan, if_neg (by decide), innerCandidates570_2_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_3
    (if 3 ≤ innerCandidates570_2.length then search edge 3 [4159, 5011] innerCandidates570_2 9170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_3

def innerTail570_1 : List Nat := 4789 :: innerTail570_2
def innerCandidates570_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates570_1_checked : candidates edge 4 4159 4789 innerTail570_2 best.bound = innerCandidates570_1 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_1 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_1 best = best := by
  rw [innerTail570_1, scan, if_neg (by decide), innerCandidates570_1_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_2
    (if 3 ≤ innerCandidates570_1.length then search edge 3 [4159, 4789] innerCandidates570_1 8948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_2

def innerTail570_0 : List Nat := 4621 :: innerTail570_1
def innerCandidates570_0 : List Nat := [5779, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates570_0_checked : candidates edge 4 4159 4621 innerTail570_1 best.bound = innerCandidates570_0 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan570_0 : scan edge (search edge 3) 4 [4159] 4159 innerTail570_0 best = best := by
  rw [innerTail570_0, scan, if_neg (by decide), innerCandidates570_0_checked]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_1
    (if 3 ≤ innerCandidates570_0.length then search edge 3 [4159, 4621] innerCandidates570_0 8780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan570_1

theorem rootBranch570 : search edge 4 [4159] rootCandidates570 4159 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4159] 4159 innerTail570_0 best = best
  exact innerScan570_0

def rootCandidates569 : List Nat := [5333, 5717, 6947, 7043, 7127, 7433, 7577]
set_option maxRecDepth 4096 in
theorem rootCandidates569_checked : candidates edge 5 0 4157 tail570 best.bound = rootCandidates569 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext q; rfl
  rw [row]
  decide

def innerTail569_7 : List Nat := []
theorem innerScan569_7 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_7 best = best := rfl

def innerTail569_6 : List Nat := 7577 :: innerTail569_7
def innerCandidates569_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates569_6_checked : candidates edge 4 4157 7577 innerTail569_7 best.bound = innerCandidates569_6 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan569_6 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_6 best = best := by
  rw [innerTail569_6, scan, if_neg (by decide), innerCandidates569_6_checked]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_7
    (if 3 ≤ innerCandidates569_6.length then search edge 3 [4157, 7577] innerCandidates569_6 11734 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan569_7

def innerTail569_5 : List Nat := 7433 :: innerTail569_6
def innerCandidates569_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates569_5_checked : candidates edge 4 4157 7433 innerTail569_6 best.bound = innerCandidates569_5 := by
  unfold candidates
  have row : edge 7433 = SearchData.row7433.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan569_5 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_5 best = best := by
  rw [innerTail569_5, scan, if_neg (by decide), innerCandidates569_5_checked]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_6
    (if 3 ≤ innerCandidates569_5.length then search edge 3 [4157, 7433] innerCandidates569_5 11590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan569_6

def innerTail569_4 : List Nat := 7127 :: innerTail569_5
def innerCandidates569_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates569_4_checked : candidates edge 4 4157 7127 innerTail569_5 best.bound = innerCandidates569_4 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan569_4 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_4 best = best := by
  rw [innerTail569_4, scan, if_neg (by decide), innerCandidates569_4_checked]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_5
    (if 3 ≤ innerCandidates569_4.length then search edge 3 [4157, 7127] innerCandidates569_4 11284 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan569_5

def innerTail569_3 : List Nat := 7043 :: innerTail569_4
def innerCandidates569_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates569_3_checked : candidates edge 4 4157 7043 innerTail569_4 best.bound = innerCandidates569_3 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan569_3 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_3 best = best := by
  rw [innerTail569_3, scan, if_neg (by decide), innerCandidates569_3_checked]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_4
    (if 3 ≤ innerCandidates569_3.length then search edge 3 [4157, 7043] innerCandidates569_3 11200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan569_4

def innerTail569_2 : List Nat := 6947 :: innerTail569_3
def innerCandidates569_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates569_2_checked : candidates edge 4 4157 6947 innerTail569_3 best.bound = innerCandidates569_2 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan569_2 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_2 best = best := by
  rw [innerTail569_2, scan, if_neg (by decide), innerCandidates569_2_checked]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_3
    (if 3 ≤ innerCandidates569_2.length then search edge 3 [4157, 6947] innerCandidates569_2 11104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan569_3

def innerTail569_1 : List Nat := 5717 :: innerTail569_2
def innerCandidates569_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates569_1_checked : candidates edge 4 4157 5717 innerTail569_2 best.bound = innerCandidates569_1 := by
  unfold candidates
  have row : edge 5717 = SearchData.row5717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan569_1 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_1 best = best := by
  rw [innerTail569_1, scan, if_neg (by decide), innerCandidates569_1_checked]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_2
    (if 3 ≤ innerCandidates569_1.length then search edge 3 [4157, 5717] innerCandidates569_1 9874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan569_2

def innerTail569_0 : List Nat := 5333 :: innerTail569_1
def innerCandidates569_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates569_0_checked : candidates edge 4 4157 5333 innerTail569_1 best.bound = innerCandidates569_0 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan569_0 : scan edge (search edge 3) 4 [4157] 4157 innerTail569_0 best = best := by
  rw [innerTail569_0, scan, if_neg (by decide), innerCandidates569_0_checked]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_1
    (if 3 ≤ innerCandidates569_0.length then search edge 3 [4157, 5333] innerCandidates569_0 9490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan569_1

theorem rootBranch569 : search edge 4 [4157] rootCandidates569 4157 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4157] 4157 innerTail569_0 best = best
  exact innerScan569_0

def rootCandidates568 : List Nat := [4201, 4789, 5023, 5077, 5101, 5503, 5521, 6997, 7753, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates568_checked : candidates edge 5 0 4153 tail569 best.bound = rootCandidates568 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext q; rfl
  rw [row]
  decide

def innerTail568_10 : List Nat := []
theorem innerScan568_10 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_10 best = best := rfl

def innerTail568_9 : List Nat := 8311 :: innerTail568_10
def innerCandidates568_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_9_checked : candidates edge 4 4153 8311 innerTail568_10 best.bound = innerCandidates568_9 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_9 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_9 best = best := by
  rw [innerTail568_9, scan, if_neg (by decide), innerCandidates568_9_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_10
    (if 3 ≤ innerCandidates568_9.length then search edge 3 [4153, 8311] innerCandidates568_9 12464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_10

def innerTail568_8 : List Nat := 7753 :: innerTail568_9
def innerCandidates568_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_8_checked : candidates edge 4 4153 7753 innerTail568_9 best.bound = innerCandidates568_8 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_8 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_8 best = best := by
  rw [innerTail568_8, scan, if_neg (by decide), innerCandidates568_8_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_9
    (if 3 ≤ innerCandidates568_8.length then search edge 3 [4153, 7753] innerCandidates568_8 11906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_9

def innerTail568_7 : List Nat := 6997 :: innerTail568_8
def innerCandidates568_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_7_checked : candidates edge 4 4153 6997 innerTail568_8 best.bound = innerCandidates568_7 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_7 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_7 best = best := by
  rw [innerTail568_7, scan, if_neg (by decide), innerCandidates568_7_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_8
    (if 3 ≤ innerCandidates568_7.length then search edge 3 [4153, 6997] innerCandidates568_7 11150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_8

def innerTail568_6 : List Nat := 5521 :: innerTail568_7
def innerCandidates568_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_6_checked : candidates edge 4 4153 5521 innerTail568_7 best.bound = innerCandidates568_6 := by
  unfold candidates
  have row : edge 5521 = SearchData.row5521.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_6 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_6 best = best := by
  rw [innerTail568_6, scan, if_neg (by decide), innerCandidates568_6_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_7
    (if 3 ≤ innerCandidates568_6.length then search edge 3 [4153, 5521] innerCandidates568_6 9674 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_7

def innerTail568_5 : List Nat := 5503 :: innerTail568_6
def innerCandidates568_5 : List Nat := [8311]
set_option maxRecDepth 4096 in
theorem innerCandidates568_5_checked : candidates edge 4 4153 5503 innerTail568_6 best.bound = innerCandidates568_5 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_5 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_5 best = best := by
  rw [innerTail568_5, scan, if_neg (by decide), innerCandidates568_5_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_6
    (if 3 ≤ innerCandidates568_5.length then search edge 3 [4153, 5503] innerCandidates568_5 9656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_6

def innerTail568_4 : List Nat := 5101 :: innerTail568_5
def innerCandidates568_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_4_checked : candidates edge 4 4153 5101 innerTail568_5 best.bound = innerCandidates568_4 := by
  unfold candidates
  have row : edge 5101 = SearchData.row5101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_4 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_4 best = best := by
  rw [innerTail568_4, scan, if_neg (by decide), innerCandidates568_4_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_5
    (if 3 ≤ innerCandidates568_4.length then search edge 3 [4153, 5101] innerCandidates568_4 9254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_5

def innerTail568_3 : List Nat := 5077 :: innerTail568_4
def innerCandidates568_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_3_checked : candidates edge 4 4153 5077 innerTail568_4 best.bound = innerCandidates568_3 := by
  unfold candidates
  have row : edge 5077 = SearchData.row5077.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_3 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_3 best = best := by
  rw [innerTail568_3, scan, if_neg (by decide), innerCandidates568_3_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_4
    (if 3 ≤ innerCandidates568_3.length then search edge 3 [4153, 5077] innerCandidates568_3 9230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_4

def innerTail568_2 : List Nat := 5023 :: innerTail568_3
def innerCandidates568_2 : List Nat := [7753]
set_option maxRecDepth 4096 in
theorem innerCandidates568_2_checked : candidates edge 4 4153 5023 innerTail568_3 best.bound = innerCandidates568_2 := by
  unfold candidates
  have row : edge 5023 = SearchData.row5023.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_2 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_2 best = best := by
  rw [innerTail568_2, scan, if_neg (by decide), innerCandidates568_2_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_3
    (if 3 ≤ innerCandidates568_2.length then search edge 3 [4153, 5023] innerCandidates568_2 9176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_3

def innerTail568_1 : List Nat := 4789 :: innerTail568_2
def innerCandidates568_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_1_checked : candidates edge 4 4153 4789 innerTail568_2 best.bound = innerCandidates568_1 := by
  unfold candidates
  have row : edge 4789 = SearchData.row4789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_1 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_1 best = best := by
  rw [innerTail568_1, scan, if_neg (by decide), innerCandidates568_1_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_2
    (if 3 ≤ innerCandidates568_1.length then search edge 3 [4153, 4789] innerCandidates568_1 8942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_2

def innerTail568_0 : List Nat := 4201 :: innerTail568_1
def innerCandidates568_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates568_0_checked : candidates edge 4 4153 4201 innerTail568_1 best.bound = innerCandidates568_0 := by
  unfold candidates
  have row : edge 4201 = SearchData.row4201.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan568_0 : scan edge (search edge 3) 4 [4153] 4153 innerTail568_0 best = best := by
  rw [innerTail568_0, scan, if_neg (by decide), innerCandidates568_0_checked]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_1
    (if 3 ≤ innerCandidates568_0.length then search edge 3 [4153, 4201] innerCandidates568_0 8354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan568_1

theorem rootBranch568 : search edge 4 [4153] rootCandidates568 4153 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4153] 4153 innerTail568_0 best = best
  exact innerScan568_0

end ProjectEulerSolutions.PrimePairUpper.Search
