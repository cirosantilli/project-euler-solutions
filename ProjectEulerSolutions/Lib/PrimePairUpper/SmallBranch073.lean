import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates513 : List Nat := [3919, 4603, 5209, 5431, 5749, 6007, 6469, 6673, 7459, 7867, 7879]
set_option maxRecDepth 4096 in
theorem rootCandidates513_checked : candidates edge 5 0 3697 tail514 best.bound = rootCandidates513 := by
  unfold candidates
  have row : edge 3697 = SearchData.row3697.contains := by funext q; rfl
  rw [row]
  decide

def innerTail513_11 : List Nat := []
theorem innerScan513_11 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_11 best = best := rfl

def innerTail513_10 : List Nat := 7879 :: innerTail513_11
def innerCandidates513_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates513_10_checked : candidates edge 4 3697 7879 innerTail513_11 best.bound = innerCandidates513_10 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_10 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_10 best = best := by
  rw [innerTail513_10, scan, if_neg (by decide), innerCandidates513_10_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_11
    (if 3 ≤ innerCandidates513_10.length then search edge 3 [3697, 7879] innerCandidates513_10 11576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_11

def innerTail513_9 : List Nat := 7867 :: innerTail513_10
def innerCandidates513_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates513_9_checked : candidates edge 4 3697 7867 innerTail513_10 best.bound = innerCandidates513_9 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_9 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_9 best = best := by
  rw [innerTail513_9, scan, if_neg (by decide), innerCandidates513_9_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_10
    (if 3 ≤ innerCandidates513_9.length then search edge 3 [3697, 7867] innerCandidates513_9 11564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_10

def innerTail513_8 : List Nat := 7459 :: innerTail513_9
def innerCandidates513_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates513_8_checked : candidates edge 4 3697 7459 innerTail513_9 best.bound = innerCandidates513_8 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_8 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_8 best = best := by
  rw [innerTail513_8, scan, if_neg (by decide), innerCandidates513_8_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_9
    (if 3 ≤ innerCandidates513_8.length then search edge 3 [3697, 7459] innerCandidates513_8 11156 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_9

def innerTail513_7 : List Nat := 6673 :: innerTail513_8
def innerCandidates513_7 : List Nat := [7459]
set_option maxRecDepth 4096 in
theorem innerCandidates513_7_checked : candidates edge 4 3697 6673 innerTail513_8 best.bound = innerCandidates513_7 := by
  unfold candidates
  have row : edge 6673 = SearchData.row6673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_7 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_7 best = best := by
  rw [innerTail513_7, scan, if_neg (by decide), innerCandidates513_7_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_8
    (if 3 ≤ innerCandidates513_7.length then search edge 3 [3697, 6673] innerCandidates513_7 10370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_8

def innerTail513_6 : List Nat := 6469 :: innerTail513_7
def innerCandidates513_6 : List Nat := [6673, 7879]
set_option maxRecDepth 4096 in
theorem innerCandidates513_6_checked : candidates edge 4 3697 6469 innerTail513_7 best.bound = innerCandidates513_6 := by
  unfold candidates
  have row : edge 6469 = SearchData.row6469.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_6 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_6 best = best := by
  rw [innerTail513_6, scan, if_neg (by decide), innerCandidates513_6_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_7
    (if 3 ≤ innerCandidates513_6.length then search edge 3 [3697, 6469] innerCandidates513_6 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_7

def innerTail513_5 : List Nat := 6007 :: innerTail513_6
def innerCandidates513_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates513_5_checked : candidates edge 4 3697 6007 innerTail513_6 best.bound = innerCandidates513_5 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_5 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_5 best = best := by
  rw [innerTail513_5, scan, if_neg (by decide), innerCandidates513_5_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_6
    (if 3 ≤ innerCandidates513_5.length then search edge 3 [3697, 6007] innerCandidates513_5 9704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_6

def innerTail513_4 : List Nat := 5749 :: innerTail513_5
def innerCandidates513_4 : List Nat := [6007]
set_option maxRecDepth 4096 in
theorem innerCandidates513_4_checked : candidates edge 4 3697 5749 innerTail513_5 best.bound = innerCandidates513_4 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_4 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_4 best = best := by
  rw [innerTail513_4, scan, if_neg (by decide), innerCandidates513_4_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_5
    (if 3 ≤ innerCandidates513_4.length then search edge 3 [3697, 5749] innerCandidates513_4 9446 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_5

def innerTail513_3 : List Nat := 5431 :: innerTail513_4
def innerCandidates513_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates513_3_checked : candidates edge 4 3697 5431 innerTail513_4 best.bound = innerCandidates513_3 := by
  unfold candidates
  have row : edge 5431 = SearchData.row5431.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_3 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_3 best = best := by
  rw [innerTail513_3, scan, if_neg (by decide), innerCandidates513_3_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_4
    (if 3 ≤ innerCandidates513_3.length then search edge 3 [3697, 5431] innerCandidates513_3 9128 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_4

def innerTail513_2 : List Nat := 5209 :: innerTail513_3
def innerCandidates513_2 : List Nat := [7879]
set_option maxRecDepth 4096 in
theorem innerCandidates513_2_checked : candidates edge 4 3697 5209 innerTail513_3 best.bound = innerCandidates513_2 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_2 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_2 best = best := by
  rw [innerTail513_2, scan, if_neg (by decide), innerCandidates513_2_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_3
    (if 3 ≤ innerCandidates513_2.length then search edge 3 [3697, 5209] innerCandidates513_2 8906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_3

def innerTail513_1 : List Nat := 4603 :: innerTail513_2
def innerCandidates513_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates513_1_checked : candidates edge 4 3697 4603 innerTail513_2 best.bound = innerCandidates513_1 := by
  unfold candidates
  have row : edge 4603 = SearchData.row4603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_1 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_1 best = best := by
  rw [innerTail513_1, scan, if_neg (by decide), innerCandidates513_1_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_2
    (if 3 ≤ innerCandidates513_1.length then search edge 3 [3697, 4603] innerCandidates513_1 8300 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_2

def innerTail513_0 : List Nat := 3919 :: innerTail513_1
def innerCandidates513_0 : List Nat := [6007]
set_option maxRecDepth 4096 in
theorem innerCandidates513_0_checked : candidates edge 4 3697 3919 innerTail513_1 best.bound = innerCandidates513_0 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan513_0 : scan edge (search edge 3) 4 [3697] 3697 innerTail513_0 best = best := by
  rw [innerTail513_0, scan, if_neg (by decide), innerCandidates513_0_checked]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_1
    (if 3 ≤ innerCandidates513_0.length then search edge 3 [3697, 3919] innerCandidates513_0 7616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan513_1

theorem rootBranch513 : search edge 4 [3697] rootCandidates513 3697 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3697] 3697 innerTail513_0 best = best
  exact innerScan513_0

def rootCandidates512 : List Nat := [4051, 4153, 4177, 4561, 5419, 6337, 7333, 7477, 7561, 7669, 7789, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates512_checked : candidates edge 5 0 3691 tail513 best.bound = rootCandidates512 := by
  unfold candidates
  have row : edge 3691 = SearchData.row3691.contains := by funext q; rfl
  rw [row]
  decide

def innerTail512_12 : List Nat := []
theorem innerScan512_12 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_12 best = best := rfl

def innerTail512_11 : List Nat := 8377 :: innerTail512_12
def innerCandidates512_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_11_checked : candidates edge 4 3691 8377 innerTail512_12 best.bound = innerCandidates512_11 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_11 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_11 best = best := by
  rw [innerTail512_11, scan, if_neg (by decide), innerCandidates512_11_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_12
    (if 3 ≤ innerCandidates512_11.length then search edge 3 [3691, 8377] innerCandidates512_11 12068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_12

def innerTail512_10 : List Nat := 7789 :: innerTail512_11
def innerCandidates512_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_10_checked : candidates edge 4 3691 7789 innerTail512_11 best.bound = innerCandidates512_10 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_10 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_10 best = best := by
  rw [innerTail512_10, scan, if_neg (by decide), innerCandidates512_10_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_11
    (if 3 ≤ innerCandidates512_10.length then search edge 3 [3691, 7789] innerCandidates512_10 11480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_11

def innerTail512_9 : List Nat := 7669 :: innerTail512_10
def innerCandidates512_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_9_checked : candidates edge 4 3691 7669 innerTail512_10 best.bound = innerCandidates512_9 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_9 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_9 best = best := by
  rw [innerTail512_9, scan, if_neg (by decide), innerCandidates512_9_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_10
    (if 3 ≤ innerCandidates512_9.length then search edge 3 [3691, 7669] innerCandidates512_9 11360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_10

def innerTail512_8 : List Nat := 7561 :: innerTail512_9
def innerCandidates512_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_8_checked : candidates edge 4 3691 7561 innerTail512_9 best.bound = innerCandidates512_8 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_8 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_8 best = best := by
  rw [innerTail512_8, scan, if_neg (by decide), innerCandidates512_8_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_9
    (if 3 ≤ innerCandidates512_8.length then search edge 3 [3691, 7561] innerCandidates512_8 11252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_9

def innerTail512_7 : List Nat := 7477 :: innerTail512_8
def innerCandidates512_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_7_checked : candidates edge 4 3691 7477 innerTail512_8 best.bound = innerCandidates512_7 := by
  unfold candidates
  have row : edge 7477 = SearchData.row7477.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_7 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_7 best = best := by
  rw [innerTail512_7, scan, if_neg (by decide), innerCandidates512_7_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_8
    (if 3 ≤ innerCandidates512_7.length then search edge 3 [3691, 7477] innerCandidates512_7 11168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_8

def innerTail512_6 : List Nat := 7333 :: innerTail512_7
def innerCandidates512_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_6_checked : candidates edge 4 3691 7333 innerTail512_7 best.bound = innerCandidates512_6 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_6 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_6 best = best := by
  rw [innerTail512_6, scan, if_neg (by decide), innerCandidates512_6_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_7
    (if 3 ≤ innerCandidates512_6.length then search edge 3 [3691, 7333] innerCandidates512_6 11024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_7

def innerTail512_5 : List Nat := 6337 :: innerTail512_6
def innerCandidates512_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_5_checked : candidates edge 4 3691 6337 innerTail512_6 best.bound = innerCandidates512_5 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_5 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_5 best = best := by
  rw [innerTail512_5, scan, if_neg (by decide), innerCandidates512_5_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_6
    (if 3 ≤ innerCandidates512_5.length then search edge 3 [3691, 6337] innerCandidates512_5 10028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_6

def innerTail512_4 : List Nat := 5419 :: innerTail512_5
def innerCandidates512_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_4_checked : candidates edge 4 3691 5419 innerTail512_5 best.bound = innerCandidates512_4 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_4 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_4 best = best := by
  rw [innerTail512_4, scan, if_neg (by decide), innerCandidates512_4_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_5
    (if 3 ≤ innerCandidates512_4.length then search edge 3 [3691, 5419] innerCandidates512_4 9110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_5

def innerTail512_3 : List Nat := 4561 :: innerTail512_4
def innerCandidates512_3 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates512_3_checked : candidates edge 4 3691 4561 innerTail512_4 best.bound = innerCandidates512_3 := by
  unfold candidates
  have row : edge 4561 = SearchData.row4561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_3 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_3 best = best := by
  rw [innerTail512_3, scan, if_neg (by decide), innerCandidates512_3_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_4
    (if 3 ≤ innerCandidates512_3.length then search edge 3 [3691, 4561] innerCandidates512_3 8252 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_4

def innerTail512_2 : List Nat := 4177 :: innerTail512_3
def innerCandidates512_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_2_checked : candidates edge 4 3691 4177 innerTail512_3 best.bound = innerCandidates512_2 := by
  unfold candidates
  have row : edge 4177 = SearchData.row4177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_2 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_2 best = best := by
  rw [innerTail512_2, scan, if_neg (by decide), innerCandidates512_2_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_3
    (if 3 ≤ innerCandidates512_2.length then search edge 3 [3691, 4177] innerCandidates512_2 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_3

def innerTail512_1 : List Nat := 4153 :: innerTail512_2
def innerCandidates512_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_1_checked : candidates edge 4 3691 4153 innerTail512_2 best.bound = innerCandidates512_1 := by
  unfold candidates
  have row : edge 4153 = SearchData.row4153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_1 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_1 best = best := by
  rw [innerTail512_1, scan, if_neg (by decide), innerCandidates512_1_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_2
    (if 3 ≤ innerCandidates512_1.length then search edge 3 [3691, 4153] innerCandidates512_1 7844 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_2

def innerTail512_0 : List Nat := 4051 :: innerTail512_1
def innerCandidates512_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates512_0_checked : candidates edge 4 3691 4051 innerTail512_1 best.bound = innerCandidates512_0 := by
  unfold candidates
  have row : edge 4051 = SearchData.row4051.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan512_0 : scan edge (search edge 3) 4 [3691] 3691 innerTail512_0 best = best := by
  rw [innerTail512_0, scan, if_neg (by decide), innerCandidates512_0_checked]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_1
    (if 3 ≤ innerCandidates512_0.length then search edge 3 [3691, 4051] innerCandidates512_0 7742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan512_1

theorem rootBranch512 : search edge 4 [3691] rootCandidates512 3691 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3691] 3691 innerTail512_0 best = best
  exact innerScan512_0

def rootCandidates511 : List Nat := [4211, 4733, 5807, 5903, 6659, 7229, 7583, 8069]
set_option maxRecDepth 4096 in
theorem rootCandidates511_checked : candidates edge 5 0 3677 tail512 best.bound = rootCandidates511 := by
  unfold candidates
  have row : edge 3677 = SearchData.row3677.contains := by funext q; rfl
  rw [row]
  decide

def innerTail511_8 : List Nat := []
theorem innerScan511_8 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_8 best = best := rfl

def innerTail511_7 : List Nat := 8069 :: innerTail511_8
def innerCandidates511_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_7_checked : candidates edge 4 3677 8069 innerTail511_8 best.bound = innerCandidates511_7 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_7 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_7 best = best := by
  rw [innerTail511_7, scan, if_neg (by decide), innerCandidates511_7_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_8
    (if 3 ≤ innerCandidates511_7.length then search edge 3 [3677, 8069] innerCandidates511_7 11746 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_8

def innerTail511_6 : List Nat := 7583 :: innerTail511_7
def innerCandidates511_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_6_checked : candidates edge 4 3677 7583 innerTail511_7 best.bound = innerCandidates511_6 := by
  unfold candidates
  have row : edge 7583 = SearchData.row7583.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_6 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_6 best = best := by
  rw [innerTail511_6, scan, if_neg (by decide), innerCandidates511_6_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_7
    (if 3 ≤ innerCandidates511_6.length then search edge 3 [3677, 7583] innerCandidates511_6 11260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_7

def innerTail511_5 : List Nat := 7229 :: innerTail511_6
def innerCandidates511_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_5_checked : candidates edge 4 3677 7229 innerTail511_6 best.bound = innerCandidates511_5 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_5 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_5 best = best := by
  rw [innerTail511_5, scan, if_neg (by decide), innerCandidates511_5_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_6
    (if 3 ≤ innerCandidates511_5.length then search edge 3 [3677, 7229] innerCandidates511_5 10906 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_6

def innerTail511_4 : List Nat := 6659 :: innerTail511_5
def innerCandidates511_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_4_checked : candidates edge 4 3677 6659 innerTail511_5 best.bound = innerCandidates511_4 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_4 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_4 best = best := by
  rw [innerTail511_4, scan, if_neg (by decide), innerCandidates511_4_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_5
    (if 3 ≤ innerCandidates511_4.length then search edge 3 [3677, 6659] innerCandidates511_4 10336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_5

def innerTail511_3 : List Nat := 5903 :: innerTail511_4
def innerCandidates511_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_3_checked : candidates edge 4 3677 5903 innerTail511_4 best.bound = innerCandidates511_3 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_3 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_3 best = best := by
  rw [innerTail511_3, scan, if_neg (by decide), innerCandidates511_3_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_4
    (if 3 ≤ innerCandidates511_3.length then search edge 3 [3677, 5903] innerCandidates511_3 9580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_4

def innerTail511_2 : List Nat := 5807 :: innerTail511_3
def innerCandidates511_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_2_checked : candidates edge 4 3677 5807 innerTail511_3 best.bound = innerCandidates511_2 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_2 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_2 best = best := by
  rw [innerTail511_2, scan, if_neg (by decide), innerCandidates511_2_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_3
    (if 3 ≤ innerCandidates511_2.length then search edge 3 [3677, 5807] innerCandidates511_2 9484 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_3

def innerTail511_1 : List Nat := 4733 :: innerTail511_2
def innerCandidates511_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_1_checked : candidates edge 4 3677 4733 innerTail511_2 best.bound = innerCandidates511_1 := by
  unfold candidates
  have row : edge 4733 = SearchData.row4733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_1 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_1 best = best := by
  rw [innerTail511_1, scan, if_neg (by decide), innerCandidates511_1_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_2
    (if 3 ≤ innerCandidates511_1.length then search edge 3 [3677, 4733] innerCandidates511_1 8410 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_2

def innerTail511_0 : List Nat := 4211 :: innerTail511_1
def innerCandidates511_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates511_0_checked : candidates edge 4 3677 4211 innerTail511_1 best.bound = innerCandidates511_0 := by
  unfold candidates
  have row : edge 4211 = SearchData.row4211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan511_0 : scan edge (search edge 3) 4 [3677] 3677 innerTail511_0 best = best := by
  rw [innerTail511_0, scan, if_neg (by decide), innerCandidates511_0_checked]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_1
    (if 3 ≤ innerCandidates511_0.length then search edge 3 [3677, 4211] innerCandidates511_0 7888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan511_1

theorem rootBranch511 : search edge 4 [3677] rootCandidates511 3677 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3677] 3677 innerTail511_0 best = best
  exact innerScan511_0

def rootCandidates510 : List Nat := [4129, 4759, 5581, 6529, 6691, 7537, 7717, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates510_checked : candidates edge 5 0 3673 tail511 best.bound = rootCandidates510 := by
  unfold candidates
  have row : edge 3673 = SearchData.row3673.contains := by funext q; rfl
  rw [row]
  decide

def innerTail510_8 : List Nat := []
theorem innerScan510_8 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_8 best = best := rfl

def innerTail510_7 : List Nat := 8293 :: innerTail510_8
def innerCandidates510_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates510_7_checked : candidates edge 4 3673 8293 innerTail510_8 best.bound = innerCandidates510_7 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_7 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_7 best = best := by
  rw [innerTail510_7, scan, if_neg (by decide), innerCandidates510_7_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_8
    (if 3 ≤ innerCandidates510_7.length then search edge 3 [3673, 8293] innerCandidates510_7 11966 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_8

def innerTail510_6 : List Nat := 7717 :: innerTail510_7
def innerCandidates510_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates510_6_checked : candidates edge 4 3673 7717 innerTail510_7 best.bound = innerCandidates510_6 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_6 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_6 best = best := by
  rw [innerTail510_6, scan, if_neg (by decide), innerCandidates510_6_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_7
    (if 3 ≤ innerCandidates510_6.length then search edge 3 [3673, 7717] innerCandidates510_6 11390 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_7

def innerTail510_5 : List Nat := 7537 :: innerTail510_6
def innerCandidates510_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates510_5_checked : candidates edge 4 3673 7537 innerTail510_6 best.bound = innerCandidates510_5 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_5 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_5 best = best := by
  rw [innerTail510_5, scan, if_neg (by decide), innerCandidates510_5_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_6
    (if 3 ≤ innerCandidates510_5.length then search edge 3 [3673, 7537] innerCandidates510_5 11210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_6

def innerTail510_4 : List Nat := 6691 :: innerTail510_5
def innerCandidates510_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates510_4_checked : candidates edge 4 3673 6691 innerTail510_5 best.bound = innerCandidates510_4 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_4 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_4 best = best := by
  rw [innerTail510_4, scan, if_neg (by decide), innerCandidates510_4_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_5
    (if 3 ≤ innerCandidates510_4.length then search edge 3 [3673, 6691] innerCandidates510_4 10364 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_5

def innerTail510_3 : List Nat := 6529 :: innerTail510_4
def innerCandidates510_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates510_3_checked : candidates edge 4 3673 6529 innerTail510_4 best.bound = innerCandidates510_3 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_3 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_3 best = best := by
  rw [innerTail510_3, scan, if_neg (by decide), innerCandidates510_3_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_4
    (if 3 ≤ innerCandidates510_3.length then search edge 3 [3673, 6529] innerCandidates510_3 10202 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_4

def innerTail510_2 : List Nat := 5581 :: innerTail510_3
def innerCandidates510_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates510_2_checked : candidates edge 4 3673 5581 innerTail510_3 best.bound = innerCandidates510_2 := by
  unfold candidates
  have row : edge 5581 = SearchData.row5581.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_2 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_2 best = best := by
  rw [innerTail510_2, scan, if_neg (by decide), innerCandidates510_2_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_3
    (if 3 ≤ innerCandidates510_2.length then search edge 3 [3673, 5581] innerCandidates510_2 9254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_3

def innerTail510_1 : List Nat := 4759 :: innerTail510_2
def innerCandidates510_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates510_1_checked : candidates edge 4 3673 4759 innerTail510_2 best.bound = innerCandidates510_1 := by
  unfold candidates
  have row : edge 4759 = SearchData.row4759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_1 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_1 best = best := by
  rw [innerTail510_1, scan, if_neg (by decide), innerCandidates510_1_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_2
    (if 3 ≤ innerCandidates510_1.length then search edge 3 [3673, 4759] innerCandidates510_1 8432 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_2

def innerTail510_0 : List Nat := 4129 :: innerTail510_1
def innerCandidates510_0 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates510_0_checked : candidates edge 4 3673 4129 innerTail510_1 best.bound = innerCandidates510_0 := by
  unfold candidates
  have row : edge 4129 = SearchData.row4129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan510_0 : scan edge (search edge 3) 4 [3673] 3673 innerTail510_0 best = best := by
  rw [innerTail510_0, scan, if_neg (by decide), innerCandidates510_0_checked]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_1
    (if 3 ≤ innerCandidates510_0.length then search edge 3 [3673, 4129] innerCandidates510_0 7802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan510_1

theorem rootBranch510 : search edge 4 [3673] rootCandidates510 3673 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3673] 3673 innerTail510_0 best = best
  exact innerScan510_0

def rootCandidates509 : List Nat := [4091, 5039, 5381, 5849, 6011, 6131, 6653, 7019, 7151, 7523, 7691, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates509_checked : candidates edge 5 0 3671 tail510 best.bound = rootCandidates509 := by
  unfold candidates
  have row : edge 3671 = SearchData.row3671.contains := by funext q; rfl
  rw [row]
  decide

def innerTail509_12 : List Nat := []
theorem innerScan509_12 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_12 best = best := rfl

def innerTail509_11 : List Nat := 7853 :: innerTail509_12
def innerCandidates509_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_11_checked : candidates edge 4 3671 7853 innerTail509_12 best.bound = innerCandidates509_11 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_11 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_11 best = best := by
  rw [innerTail509_11, scan, if_neg (by decide), innerCandidates509_11_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_12
    (if 3 ≤ innerCandidates509_11.length then search edge 3 [3671, 7853] innerCandidates509_11 11524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_12

def innerTail509_10 : List Nat := 7691 :: innerTail509_11
def innerCandidates509_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_10_checked : candidates edge 4 3671 7691 innerTail509_11 best.bound = innerCandidates509_10 := by
  unfold candidates
  have row : edge 7691 = SearchData.row7691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_10 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_10 best = best := by
  rw [innerTail509_10, scan, if_neg (by decide), innerCandidates509_10_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_11
    (if 3 ≤ innerCandidates509_10.length then search edge 3 [3671, 7691] innerCandidates509_10 11362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_11

def innerTail509_9 : List Nat := 7523 :: innerTail509_10
def innerCandidates509_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_9_checked : candidates edge 4 3671 7523 innerTail509_10 best.bound = innerCandidates509_9 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_9 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_9 best = best := by
  rw [innerTail509_9, scan, if_neg (by decide), innerCandidates509_9_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_10
    (if 3 ≤ innerCandidates509_9.length then search edge 3 [3671, 7523] innerCandidates509_9 11194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_10

def innerTail509_8 : List Nat := 7151 :: innerTail509_9
def innerCandidates509_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_8_checked : candidates edge 4 3671 7151 innerTail509_9 best.bound = innerCandidates509_8 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_8 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_8 best = best := by
  rw [innerTail509_8, scan, if_neg (by decide), innerCandidates509_8_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_9
    (if 3 ≤ innerCandidates509_8.length then search edge 3 [3671, 7151] innerCandidates509_8 10822 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_9

def innerTail509_7 : List Nat := 7019 :: innerTail509_8
def innerCandidates509_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_7_checked : candidates edge 4 3671 7019 innerTail509_8 best.bound = innerCandidates509_7 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_7 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_7 best = best := by
  rw [innerTail509_7, scan, if_neg (by decide), innerCandidates509_7_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_8
    (if 3 ≤ innerCandidates509_7.length then search edge 3 [3671, 7019] innerCandidates509_7 10690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_8

def innerTail509_6 : List Nat := 6653 :: innerTail509_7
def innerCandidates509_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_6_checked : candidates edge 4 3671 6653 innerTail509_7 best.bound = innerCandidates509_6 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_6 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_6 best = best := by
  rw [innerTail509_6, scan, if_neg (by decide), innerCandidates509_6_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_7
    (if 3 ≤ innerCandidates509_6.length then search edge 3 [3671, 6653] innerCandidates509_6 10324 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_7

def innerTail509_5 : List Nat := 6131 :: innerTail509_6
def innerCandidates509_5 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates509_5_checked : candidates edge 4 3671 6131 innerTail509_6 best.bound = innerCandidates509_5 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_5 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_5 best = best := by
  rw [innerTail509_5, scan, if_neg (by decide), innerCandidates509_5_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_6
    (if 3 ≤ innerCandidates509_5.length then search edge 3 [3671, 6131] innerCandidates509_5 9802 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_6

def innerTail509_4 : List Nat := 6011 :: innerTail509_5
def innerCandidates509_4 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates509_4_checked : candidates edge 4 3671 6011 innerTail509_5 best.bound = innerCandidates509_4 := by
  unfold candidates
  have row : edge 6011 = SearchData.row6011.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_4 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_4 best = best := by
  rw [innerTail509_4, scan, if_neg (by decide), innerCandidates509_4_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_5
    (if 3 ≤ innerCandidates509_4.length then search edge 3 [3671, 6011] innerCandidates509_4 9682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_5

def innerTail509_3 : List Nat := 5849 :: innerTail509_4
def innerCandidates509_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_3_checked : candidates edge 4 3671 5849 innerTail509_4 best.bound = innerCandidates509_3 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_3 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_3 best = best := by
  rw [innerTail509_3, scan, if_neg (by decide), innerCandidates509_3_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_4
    (if 3 ≤ innerCandidates509_3.length then search edge 3 [3671, 5849] innerCandidates509_3 9520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_4

def innerTail509_2 : List Nat := 5381 :: innerTail509_3
def innerCandidates509_2 : List Nat := [6011]
set_option maxRecDepth 4096 in
theorem innerCandidates509_2_checked : candidates edge 4 3671 5381 innerTail509_3 best.bound = innerCandidates509_2 := by
  unfold candidates
  have row : edge 5381 = SearchData.row5381.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_2 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_2 best = best := by
  rw [innerTail509_2, scan, if_neg (by decide), innerCandidates509_2_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_3
    (if 3 ≤ innerCandidates509_2.length then search edge 3 [3671, 5381] innerCandidates509_2 9052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_3

def innerTail509_1 : List Nat := 5039 :: innerTail509_2
def innerCandidates509_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_1_checked : candidates edge 4 3671 5039 innerTail509_2 best.bound = innerCandidates509_1 := by
  unfold candidates
  have row : edge 5039 = SearchData.row5039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_1 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_1 best = best := by
  rw [innerTail509_1, scan, if_neg (by decide), innerCandidates509_1_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_2
    (if 3 ≤ innerCandidates509_1.length then search edge 3 [3671, 5039] innerCandidates509_1 8710 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_2

def innerTail509_0 : List Nat := 4091 :: innerTail509_1
def innerCandidates509_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates509_0_checked : candidates edge 4 3671 4091 innerTail509_1 best.bound = innerCandidates509_0 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan509_0 : scan edge (search edge 3) 4 [3671] 3671 innerTail509_0 best = best := by
  rw [innerTail509_0, scan, if_neg (by decide), innerCandidates509_0_checked]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_1
    (if 3 ≤ innerCandidates509_0.length then search edge 3 [3671, 4091] innerCandidates509_0 7762 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan509_1

theorem rootBranch509 : search edge 4 [3671] rootCandidates509 3671 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3671] 3671 innerTail509_0 best = best
  exact innerScan509_0

def rootCandidates508 : List Nat := [3803, 4091, 4157, 4373, 4643, 5981, 6047, 6317, 6449, 7229, 7703, 7829, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates508_checked : candidates edge 5 0 3659 tail509 best.bound = rootCandidates508 := by
  unfold candidates
  have row : edge 3659 = SearchData.row3659.contains := by funext q; rfl
  rw [row]
  decide

def innerTail508_13 : List Nat := []
theorem innerScan508_13 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_13 best = best := rfl

def innerTail508_12 : List Nat := 8273 :: innerTail508_13
def innerCandidates508_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_12_checked : candidates edge 4 3659 8273 innerTail508_13 best.bound = innerCandidates508_12 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_12 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_12 best = best := by
  rw [innerTail508_12, scan, if_neg (by decide), innerCandidates508_12_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_13
    (if 3 ≤ innerCandidates508_12.length then search edge 3 [3659, 8273] innerCandidates508_12 11932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_13

def innerTail508_11 : List Nat := 7829 :: innerTail508_12
def innerCandidates508_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_11_checked : candidates edge 4 3659 7829 innerTail508_12 best.bound = innerCandidates508_11 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_11 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_11 best = best := by
  rw [innerTail508_11, scan, if_neg (by decide), innerCandidates508_11_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_12
    (if 3 ≤ innerCandidates508_11.length then search edge 3 [3659, 7829] innerCandidates508_11 11488 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_12

def innerTail508_10 : List Nat := 7703 :: innerTail508_11
def innerCandidates508_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_10_checked : candidates edge 4 3659 7703 innerTail508_11 best.bound = innerCandidates508_10 := by
  unfold candidates
  have row : edge 7703 = SearchData.row7703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_10 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_10 best = best := by
  rw [innerTail508_10, scan, if_neg (by decide), innerCandidates508_10_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_11
    (if 3 ≤ innerCandidates508_10.length then search edge 3 [3659, 7703] innerCandidates508_10 11362 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_11

def innerTail508_9 : List Nat := 7229 :: innerTail508_10
def innerCandidates508_9 : List Nat := [7829]
set_option maxRecDepth 4096 in
theorem innerCandidates508_9_checked : candidates edge 4 3659 7229 innerTail508_10 best.bound = innerCandidates508_9 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_9 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_9 best = best := by
  rw [innerTail508_9, scan, if_neg (by decide), innerCandidates508_9_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_10
    (if 3 ≤ innerCandidates508_9.length then search edge 3 [3659, 7229] innerCandidates508_9 10888 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_10

def innerTail508_8 : List Nat := 6449 :: innerTail508_9
def innerCandidates508_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_8_checked : candidates edge 4 3659 6449 innerTail508_9 best.bound = innerCandidates508_8 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_8 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_8 best = best := by
  rw [innerTail508_8, scan, if_neg (by decide), innerCandidates508_8_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_9
    (if 3 ≤ innerCandidates508_8.length then search edge 3 [3659, 6449] innerCandidates508_8 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_9

def innerTail508_7 : List Nat := 6317 :: innerTail508_8
def innerCandidates508_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_7_checked : candidates edge 4 3659 6317 innerTail508_8 best.bound = innerCandidates508_7 := by
  unfold candidates
  have row : edge 6317 = SearchData.row6317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_7 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_7 best = best := by
  rw [innerTail508_7, scan, if_neg (by decide), innerCandidates508_7_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_8
    (if 3 ≤ innerCandidates508_7.length then search edge 3 [3659, 6317] innerCandidates508_7 9976 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_8

def innerTail508_6 : List Nat := 6047 :: innerTail508_7
def innerCandidates508_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_6_checked : candidates edge 4 3659 6047 innerTail508_7 best.bound = innerCandidates508_6 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_6 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_6 best = best := by
  rw [innerTail508_6, scan, if_neg (by decide), innerCandidates508_6_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_7
    (if 3 ≤ innerCandidates508_6.length then search edge 3 [3659, 6047] innerCandidates508_6 9706 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_7

def innerTail508_5 : List Nat := 5981 :: innerTail508_6
def innerCandidates508_5 : List Nat := [8273]
set_option maxRecDepth 4096 in
theorem innerCandidates508_5_checked : candidates edge 4 3659 5981 innerTail508_6 best.bound = innerCandidates508_5 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_5 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_5 best = best := by
  rw [innerTail508_5, scan, if_neg (by decide), innerCandidates508_5_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_6
    (if 3 ≤ innerCandidates508_5.length then search edge 3 [3659, 5981] innerCandidates508_5 9640 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_6

def innerTail508_4 : List Nat := 4643 :: innerTail508_5
def innerCandidates508_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_4_checked : candidates edge 4 3659 4643 innerTail508_5 best.bound = innerCandidates508_4 := by
  unfold candidates
  have row : edge 4643 = SearchData.row4643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_4 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_4 best = best := by
  rw [innerTail508_4, scan, if_neg (by decide), innerCandidates508_4_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_5
    (if 3 ≤ innerCandidates508_4.length then search edge 3 [3659, 4643] innerCandidates508_4 8302 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_5

def innerTail508_3 : List Nat := 4373 :: innerTail508_4
def innerCandidates508_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_3_checked : candidates edge 4 3659 4373 innerTail508_4 best.bound = innerCandidates508_3 := by
  unfold candidates
  have row : edge 4373 = SearchData.row4373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_3 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_3 best = best := by
  rw [innerTail508_3, scan, if_neg (by decide), innerCandidates508_3_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_4
    (if 3 ≤ innerCandidates508_3.length then search edge 3 [3659, 4373] innerCandidates508_3 8032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_4

def innerTail508_2 : List Nat := 4157 :: innerTail508_3
def innerCandidates508_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_2_checked : candidates edge 4 3659 4157 innerTail508_3 best.bound = innerCandidates508_2 := by
  unfold candidates
  have row : edge 4157 = SearchData.row4157.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_2 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_2 best = best := by
  rw [innerTail508_2, scan, if_neg (by decide), innerCandidates508_2_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_3
    (if 3 ≤ innerCandidates508_2.length then search edge 3 [3659, 4157] innerCandidates508_2 7816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_3

def innerTail508_1 : List Nat := 4091 :: innerTail508_2
def innerCandidates508_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_1_checked : candidates edge 4 3659 4091 innerTail508_2 best.bound = innerCandidates508_1 := by
  unfold candidates
  have row : edge 4091 = SearchData.row4091.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_1 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_1 best = best := by
  rw [innerTail508_1, scan, if_neg (by decide), innerCandidates508_1_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_2
    (if 3 ≤ innerCandidates508_1.length then search edge 3 [3659, 4091] innerCandidates508_1 7750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_2

def innerTail508_0 : List Nat := 3803 :: innerTail508_1
def innerCandidates508_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates508_0_checked : candidates edge 4 3659 3803 innerTail508_1 best.bound = innerCandidates508_0 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan508_0 : scan edge (search edge 3) 4 [3659] 3659 innerTail508_0 best = best := by
  rw [innerTail508_0, scan, if_neg (by decide), innerCandidates508_0_checked]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_1
    (if 3 ≤ innerCandidates508_0.length then search edge 3 [3659, 3803] innerCandidates508_0 7462 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan508_1

theorem rootBranch508 : search edge 4 [3659] rootCandidates508 3659 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3659] 3659 innerTail508_0 best = best
  exact innerScan508_0

end ProjectEulerSolutions.PrimePairUpper.Search
