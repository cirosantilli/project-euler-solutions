import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates474 : List Nat := [3803, 4139, 4409, 4799, 5153, 5387, 5879, 6449, 7547, 7727, 7841, 7937]
set_option maxRecDepth 4096 in
theorem rootCandidates474_checked : candidates edge 5 0 3389 tail475 best.bound = rootCandidates474 := by
  unfold candidates
  have row : edge 3389 = SearchData.row3389.contains := by funext q; rfl
  rw [row]
  decide

def innerTail474_12 : List Nat := []
theorem innerScan474_12 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_12 best = best := rfl

def innerTail474_11 : List Nat := 7937 :: innerTail474_12
def innerCandidates474_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_11_checked : candidates edge 4 3389 7937 innerTail474_12 best.bound = innerCandidates474_11 := by
  unfold candidates
  have row : edge 7937 = SearchData.row7937.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_11 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_11 best = best := by
  rw [innerTail474_11, scan, if_neg (by decide), innerCandidates474_11_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_12
    (if 3 ≤ innerCandidates474_11.length then search edge 3 [3389, 7937] innerCandidates474_11 11326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_12

def innerTail474_10 : List Nat := 7841 :: innerTail474_11
def innerCandidates474_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_10_checked : candidates edge 4 3389 7841 innerTail474_11 best.bound = innerCandidates474_10 := by
  unfold candidates
  have row : edge 7841 = SearchData.row7841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_10 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_10 best = best := by
  rw [innerTail474_10, scan, if_neg (by decide), innerCandidates474_10_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_11
    (if 3 ≤ innerCandidates474_10.length then search edge 3 [3389, 7841] innerCandidates474_10 11230 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_11

def innerTail474_9 : List Nat := 7727 :: innerTail474_10
def innerCandidates474_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_9_checked : candidates edge 4 3389 7727 innerTail474_10 best.bound = innerCandidates474_9 := by
  unfold candidates
  have row : edge 7727 = SearchData.row7727.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_9 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_9 best = best := by
  rw [innerTail474_9, scan, if_neg (by decide), innerCandidates474_9_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_10
    (if 3 ≤ innerCandidates474_9.length then search edge 3 [3389, 7727] innerCandidates474_9 11116 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_10

def innerTail474_8 : List Nat := 7547 :: innerTail474_9
def innerCandidates474_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_8_checked : candidates edge 4 3389 7547 innerTail474_9 best.bound = innerCandidates474_8 := by
  unfold candidates
  have row : edge 7547 = SearchData.row7547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_8 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_8 best = best := by
  rw [innerTail474_8, scan, if_neg (by decide), innerCandidates474_8_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_9
    (if 3 ≤ innerCandidates474_8.length then search edge 3 [3389, 7547] innerCandidates474_8 10936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_9

def innerTail474_7 : List Nat := 6449 :: innerTail474_8
def innerCandidates474_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_7_checked : candidates edge 4 3389 6449 innerTail474_8 best.bound = innerCandidates474_7 := by
  unfold candidates
  have row : edge 6449 = SearchData.row6449.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_7 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_7 best = best := by
  rw [innerTail474_7, scan, if_neg (by decide), innerCandidates474_7_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_8
    (if 3 ≤ innerCandidates474_7.length then search edge 3 [3389, 6449] innerCandidates474_7 9838 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_8

def innerTail474_6 : List Nat := 5879 :: innerTail474_7
def innerCandidates474_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_6_checked : candidates edge 4 3389 5879 innerTail474_7 best.bound = innerCandidates474_6 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_6 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_6 best = best := by
  rw [innerTail474_6, scan, if_neg (by decide), innerCandidates474_6_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_7
    (if 3 ≤ innerCandidates474_6.length then search edge 3 [3389, 5879] innerCandidates474_6 9268 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_7

def innerTail474_5 : List Nat := 5387 :: innerTail474_6
def innerCandidates474_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_5_checked : candidates edge 4 3389 5387 innerTail474_6 best.bound = innerCandidates474_5 := by
  unfold candidates
  have row : edge 5387 = SearchData.row5387.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_5 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_5 best = best := by
  rw [innerTail474_5, scan, if_neg (by decide), innerCandidates474_5_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_6
    (if 3 ≤ innerCandidates474_5.length then search edge 3 [3389, 5387] innerCandidates474_5 8776 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_6

def innerTail474_4 : List Nat := 5153 :: innerTail474_5
def innerCandidates474_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_4_checked : candidates edge 4 3389 5153 innerTail474_5 best.bound = innerCandidates474_4 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_4 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_4 best = best := by
  rw [innerTail474_4, scan, if_neg (by decide), innerCandidates474_4_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_5
    (if 3 ≤ innerCandidates474_4.length then search edge 3 [3389, 5153] innerCandidates474_4 8542 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_5

def innerTail474_3 : List Nat := 4799 :: innerTail474_4
def innerCandidates474_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_3_checked : candidates edge 4 3389 4799 innerTail474_4 best.bound = innerCandidates474_3 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_3 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_3 best = best := by
  rw [innerTail474_3, scan, if_neg (by decide), innerCandidates474_3_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_4
    (if 3 ≤ innerCandidates474_3.length then search edge 3 [3389, 4799] innerCandidates474_3 8188 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_4

def innerTail474_2 : List Nat := 4409 :: innerTail474_3
def innerCandidates474_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_2_checked : candidates edge 4 3389 4409 innerTail474_3 best.bound = innerCandidates474_2 := by
  unfold candidates
  have row : edge 4409 = SearchData.row4409.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_2 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_2 best = best := by
  rw [innerTail474_2, scan, if_neg (by decide), innerCandidates474_2_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_3
    (if 3 ≤ innerCandidates474_2.length then search edge 3 [3389, 4409] innerCandidates474_2 7798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_3

def innerTail474_1 : List Nat := 4139 :: innerTail474_2
def innerCandidates474_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_1_checked : candidates edge 4 3389 4139 innerTail474_2 best.bound = innerCandidates474_1 := by
  unfold candidates
  have row : edge 4139 = SearchData.row4139.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_1 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_1 best = best := by
  rw [innerTail474_1, scan, if_neg (by decide), innerCandidates474_1_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_2
    (if 3 ≤ innerCandidates474_1.length then search edge 3 [3389, 4139] innerCandidates474_1 7528 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_2

def innerTail474_0 : List Nat := 3803 :: innerTail474_1
def innerCandidates474_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates474_0_checked : candidates edge 4 3389 3803 innerTail474_1 best.bound = innerCandidates474_0 := by
  unfold candidates
  have row : edge 3803 = SearchData.row3803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan474_0 : scan edge (search edge 3) 4 [3389] 3389 innerTail474_0 best = best := by
  rw [innerTail474_0, scan, if_neg (by decide), innerCandidates474_0_checked]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_1
    (if 3 ≤ innerCandidates474_0.length then search edge 3 [3389, 3803] innerCandidates474_0 7192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan474_1

theorem rootBranch474 : search edge 4 [3389] rootCandidates474 3389 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3389] 3389 innerTail474_0 best = best
  exact innerScan474_0

def rootCandidates473 : List Nat := [3631, 4339, 4621, 4957, 5557, 5881, 6067, 6247, 6793, 6907, 7621, 8161, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates473_checked : candidates edge 5 0 3373 tail474 best.bound = rootCandidates473 := by
  unfold candidates
  have row : edge 3373 = SearchData.row3373.contains := by funext q; rfl
  rw [row]
  decide

def innerTail473_13 : List Nat := []
theorem innerScan473_13 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_13 best = best := rfl

def innerTail473_12 : List Nat := 8233 :: innerTail473_13
def innerCandidates473_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_12_checked : candidates edge 4 3373 8233 innerTail473_13 best.bound = innerCandidates473_12 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_12 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_12 best = best := by
  rw [innerTail473_12, scan, if_neg (by decide), innerCandidates473_12_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_13
    (if 3 ≤ innerCandidates473_12.length then search edge 3 [3373, 8233] innerCandidates473_12 11606 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_13

def innerTail473_11 : List Nat := 8161 :: innerTail473_12
def innerCandidates473_11 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates473_11_checked : candidates edge 4 3373 8161 innerTail473_12 best.bound = innerCandidates473_11 := by
  unfold candidates
  have row : edge 8161 = SearchData.row8161.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_11 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_11 best = best := by
  rw [innerTail473_11, scan, if_neg (by decide), innerCandidates473_11_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_12
    (if 3 ≤ innerCandidates473_11.length then search edge 3 [3373, 8161] innerCandidates473_11 11534 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_12

def innerTail473_10 : List Nat := 7621 :: innerTail473_11
def innerCandidates473_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_10_checked : candidates edge 4 3373 7621 innerTail473_11 best.bound = innerCandidates473_10 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_10 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_10 best = best := by
  rw [innerTail473_10, scan, if_neg (by decide), innerCandidates473_10_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_11
    (if 3 ≤ innerCandidates473_10.length then search edge 3 [3373, 7621] innerCandidates473_10 10994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_11

def innerTail473_9 : List Nat := 6907 :: innerTail473_10
def innerCandidates473_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_9_checked : candidates edge 4 3373 6907 innerTail473_10 best.bound = innerCandidates473_9 := by
  unfold candidates
  have row : edge 6907 = SearchData.row6907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_9 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_9 best = best := by
  rw [innerTail473_9, scan, if_neg (by decide), innerCandidates473_9_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_10
    (if 3 ≤ innerCandidates473_9.length then search edge 3 [3373, 6907] innerCandidates473_9 10280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_10

def innerTail473_8 : List Nat := 6793 :: innerTail473_9
def innerCandidates473_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_8_checked : candidates edge 4 3373 6793 innerTail473_9 best.bound = innerCandidates473_8 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_8 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_8 best = best := by
  rw [innerTail473_8, scan, if_neg (by decide), innerCandidates473_8_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_9
    (if 3 ≤ innerCandidates473_8.length then search edge 3 [3373, 6793] innerCandidates473_8 10166 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_9

def innerTail473_7 : List Nat := 6247 :: innerTail473_8
def innerCandidates473_7 : List Nat := [8161]
set_option maxRecDepth 4096 in
theorem innerCandidates473_7_checked : candidates edge 4 3373 6247 innerTail473_8 best.bound = innerCandidates473_7 := by
  unfold candidates
  have row : edge 6247 = SearchData.row6247.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_7 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_7 best = best := by
  rw [innerTail473_7, scan, if_neg (by decide), innerCandidates473_7_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_8
    (if 3 ≤ innerCandidates473_7.length then search edge 3 [3373, 6247] innerCandidates473_7 9620 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_8

def innerTail473_6 : List Nat := 6067 :: innerTail473_7
def innerCandidates473_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_6_checked : candidates edge 4 3373 6067 innerTail473_7 best.bound = innerCandidates473_6 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_6 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_6 best = best := by
  rw [innerTail473_6, scan, if_neg (by decide), innerCandidates473_6_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_7
    (if 3 ≤ innerCandidates473_6.length then search edge 3 [3373, 6067] innerCandidates473_6 9440 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_7

def innerTail473_5 : List Nat := 5881 :: innerTail473_6
def innerCandidates473_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_5_checked : candidates edge 4 3373 5881 innerTail473_6 best.bound = innerCandidates473_5 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_5 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_5 best = best := by
  rw [innerTail473_5, scan, if_neg (by decide), innerCandidates473_5_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_6
    (if 3 ≤ innerCandidates473_5.length then search edge 3 [3373, 5881] innerCandidates473_5 9254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_6

def innerTail473_4 : List Nat := 5557 :: innerTail473_5
def innerCandidates473_4 : List Nat := [6067, 8233]
set_option maxRecDepth 4096 in
theorem innerCandidates473_4_checked : candidates edge 4 3373 5557 innerTail473_5 best.bound = innerCandidates473_4 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_4 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_4 best = best := by
  rw [innerTail473_4, scan, if_neg (by decide), innerCandidates473_4_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_5
    (if 3 ≤ innerCandidates473_4.length then search edge 3 [3373, 5557] innerCandidates473_4 8930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_5

def innerTail473_3 : List Nat := 4957 :: innerTail473_4
def innerCandidates473_3 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates473_3_checked : candidates edge 4 3373 4957 innerTail473_4 best.bound = innerCandidates473_3 := by
  unfold candidates
  have row : edge 4957 = SearchData.row4957.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_3 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_3 best = best := by
  rw [innerTail473_3, scan, if_neg (by decide), innerCandidates473_3_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_4
    (if 3 ≤ innerCandidates473_3.length then search edge 3 [3373, 4957] innerCandidates473_3 8330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_4

def innerTail473_2 : List Nat := 4621 :: innerTail473_3
def innerCandidates473_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_2_checked : candidates edge 4 3373 4621 innerTail473_3 best.bound = innerCandidates473_2 := by
  unfold candidates
  have row : edge 4621 = SearchData.row4621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_2 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_2 best = best := by
  rw [innerTail473_2, scan, if_neg (by decide), innerCandidates473_2_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_3
    (if 3 ≤ innerCandidates473_2.length then search edge 3 [3373, 4621] innerCandidates473_2 7994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_3

def innerTail473_1 : List Nat := 4339 :: innerTail473_2
def innerCandidates473_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_1_checked : candidates edge 4 3373 4339 innerTail473_2 best.bound = innerCandidates473_1 := by
  unfold candidates
  have row : edge 4339 = SearchData.row4339.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_1 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_1 best = best := by
  rw [innerTail473_1, scan, if_neg (by decide), innerCandidates473_1_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_2
    (if 3 ≤ innerCandidates473_1.length then search edge 3 [3373, 4339] innerCandidates473_1 7712 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_2

def innerTail473_0 : List Nat := 3631 :: innerTail473_1
def innerCandidates473_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates473_0_checked : candidates edge 4 3373 3631 innerTail473_1 best.bound = innerCandidates473_0 := by
  unfold candidates
  have row : edge 3631 = SearchData.row3631.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan473_0 : scan edge (search edge 3) 4 [3373] 3373 innerTail473_0 best = best := by
  rw [innerTail473_0, scan, if_neg (by decide), innerCandidates473_0_checked]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_1
    (if 3 ≤ innerCandidates473_0.length then search edge 3 [3373, 3631] innerCandidates473_0 7004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan473_1

theorem rootBranch473 : search edge 4 [3373] rootCandidates473 3373 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3373] 3373 innerTail473_0 best = best
  exact innerScan473_0

def rootCandidates472 : List Nat := [3413, 4229, 4649, 4799, 4973, 5009, 5153, 5393, 5849, 6029, 6359, 6983]
set_option maxRecDepth 4096 in
theorem rootCandidates472_checked : candidates edge 5 0 3371 tail473 best.bound = rootCandidates472 := by
  unfold candidates
  have row : edge 3371 = SearchData.row3371.contains := by funext q; rfl
  rw [row]
  decide

def innerTail472_12 : List Nat := []
theorem innerScan472_12 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_12 best = best := rfl

def innerTail472_11 : List Nat := 6983 :: innerTail472_12
def innerCandidates472_11 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_11_checked : candidates edge 4 3371 6983 innerTail472_12 best.bound = innerCandidates472_11 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_11 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_11 best = best := by
  rw [innerTail472_11, scan, if_neg (by decide), innerCandidates472_11_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_12
    (if 3 ≤ innerCandidates472_11.length then search edge 3 [3371, 6983] innerCandidates472_11 10354 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_12

def innerTail472_10 : List Nat := 6359 :: innerTail472_11
def innerCandidates472_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_10_checked : candidates edge 4 3371 6359 innerTail472_11 best.bound = innerCandidates472_10 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_10 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_10 best = best := by
  rw [innerTail472_10, scan, if_neg (by decide), innerCandidates472_10_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_11
    (if 3 ≤ innerCandidates472_10.length then search edge 3 [3371, 6359] innerCandidates472_10 9730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_11

def innerTail472_9 : List Nat := 6029 :: innerTail472_10
def innerCandidates472_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_9_checked : candidates edge 4 3371 6029 innerTail472_10 best.bound = innerCandidates472_9 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_9 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_9 best = best := by
  rw [innerTail472_9, scan, if_neg (by decide), innerCandidates472_9_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_10
    (if 3 ≤ innerCandidates472_9.length then search edge 3 [3371, 6029] innerCandidates472_9 9400 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_10

def innerTail472_8 : List Nat := 5849 :: innerTail472_9
def innerCandidates472_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_8_checked : candidates edge 4 3371 5849 innerTail472_9 best.bound = innerCandidates472_8 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_8 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_8 best = best := by
  rw [innerTail472_8, scan, if_neg (by decide), innerCandidates472_8_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_9
    (if 3 ≤ innerCandidates472_8.length then search edge 3 [3371, 5849] innerCandidates472_8 9220 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_9

def innerTail472_7 : List Nat := 5393 :: innerTail472_8
def innerCandidates472_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_7_checked : candidates edge 4 3371 5393 innerTail472_8 best.bound = innerCandidates472_7 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_7 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_7 best = best := by
  rw [innerTail472_7, scan, if_neg (by decide), innerCandidates472_7_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_8
    (if 3 ≤ innerCandidates472_7.length then search edge 3 [3371, 5393] innerCandidates472_7 8764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_8

def innerTail472_6 : List Nat := 5153 :: innerTail472_7
def innerCandidates472_6 : List Nat := [5849]
set_option maxRecDepth 4096 in
theorem innerCandidates472_6_checked : candidates edge 4 3371 5153 innerTail472_7 best.bound = innerCandidates472_6 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_6 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_6 best = best := by
  rw [innerTail472_6, scan, if_neg (by decide), innerCandidates472_6_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_7
    (if 3 ≤ innerCandidates472_6.length then search edge 3 [3371, 5153] innerCandidates472_6 8524 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_7

def innerTail472_5 : List Nat := 5009 :: innerTail472_6
def innerCandidates472_5 : List Nat := [5849, 6029]
set_option maxRecDepth 4096 in
theorem innerCandidates472_5_checked : candidates edge 4 3371 5009 innerTail472_6 best.bound = innerCandidates472_5 := by
  unfold candidates
  have row : edge 5009 = SearchData.row5009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_5 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_5 best = best := by
  rw [innerTail472_5, scan, if_neg (by decide), innerCandidates472_5_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_6
    (if 3 ≤ innerCandidates472_5.length then search edge 3 [3371, 5009] innerCandidates472_5 8380 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_6

def innerTail472_4 : List Nat := 4973 :: innerTail472_5
def innerCandidates472_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_4_checked : candidates edge 4 3371 4973 innerTail472_5 best.bound = innerCandidates472_4 := by
  unfold candidates
  have row : edge 4973 = SearchData.row4973.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_4 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_4 best = best := by
  rw [innerTail472_4, scan, if_neg (by decide), innerCandidates472_4_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_5
    (if 3 ≤ innerCandidates472_4.length then search edge 3 [3371, 4973] innerCandidates472_4 8344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_5

def innerTail472_3 : List Nat := 4799 :: innerTail472_4
def innerCandidates472_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_3_checked : candidates edge 4 3371 4799 innerTail472_4 best.bound = innerCandidates472_3 := by
  unfold candidates
  have row : edge 4799 = SearchData.row4799.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_3 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_3 best = best := by
  rw [innerTail472_3, scan, if_neg (by decide), innerCandidates472_3_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_4
    (if 3 ≤ innerCandidates472_3.length then search edge 3 [3371, 4799] innerCandidates472_3 8170 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_4

def innerTail472_2 : List Nat := 4649 :: innerTail472_3
def innerCandidates472_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_2_checked : candidates edge 4 3371 4649 innerTail472_3 best.bound = innerCandidates472_2 := by
  unfold candidates
  have row : edge 4649 = SearchData.row4649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_2 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_2 best = best := by
  rw [innerTail472_2, scan, if_neg (by decide), innerCandidates472_2_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_3
    (if 3 ≤ innerCandidates472_2.length then search edge 3 [3371, 4649] innerCandidates472_2 8020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_3

def innerTail472_1 : List Nat := 4229 :: innerTail472_2
def innerCandidates472_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates472_1_checked : candidates edge 4 3371 4229 innerTail472_2 best.bound = innerCandidates472_1 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_1 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_1 best = best := by
  rw [innerTail472_1, scan, if_neg (by decide), innerCandidates472_1_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_2
    (if 3 ≤ innerCandidates472_1.length then search edge 3 [3371, 4229] innerCandidates472_1 7600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_2

def innerTail472_0 : List Nat := 3413 :: innerTail472_1
def innerCandidates472_0 : List Nat := [4649, 5009]
set_option maxRecDepth 4096 in
theorem innerCandidates472_0_checked : candidates edge 4 3371 3413 innerTail472_1 best.bound = innerCandidates472_0 := by
  unfold candidates
  have row : edge 3413 = SearchData.row3413.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan472_0 : scan edge (search edge 3) 4 [3371] 3371 innerTail472_0 best = best := by
  rw [innerTail472_0, scan, if_neg (by decide), innerCandidates472_0_checked]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_1
    (if 3 ≤ innerCandidates472_0.length then search edge 3 [3371, 3413] innerCandidates472_0 6784 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan472_1

theorem rootBranch472 : search edge 4 [3371] rootCandidates472 3371 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3371] 3371 innerTail472_0 best = best
  exact innerScan472_0

def rootCandidates471 : List Nat := [3739, 3919, 4507, 4951, 5419, 5557, 6199, 6343]
set_option maxRecDepth 4096 in
theorem rootCandidates471_checked : candidates edge 5 0 3361 tail472 best.bound = rootCandidates471 := by
  unfold candidates
  have row : edge 3361 = SearchData.row3361.contains := by funext q; rfl
  rw [row]
  decide

def innerTail471_8 : List Nat := []
theorem innerScan471_8 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_8 best = best := rfl

def innerTail471_7 : List Nat := 6343 :: innerTail471_8
def innerCandidates471_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates471_7_checked : candidates edge 4 3361 6343 innerTail471_8 best.bound = innerCandidates471_7 := by
  unfold candidates
  have row : edge 6343 = SearchData.row6343.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_7 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_7 best = best := by
  rw [innerTail471_7, scan, if_neg (by decide), innerCandidates471_7_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_8
    (if 3 ≤ innerCandidates471_7.length then search edge 3 [3361, 6343] innerCandidates471_7 9704 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_8

def innerTail471_6 : List Nat := 6199 :: innerTail471_7
def innerCandidates471_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates471_6_checked : candidates edge 4 3361 6199 innerTail471_7 best.bound = innerCandidates471_6 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_6 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_6 best = best := by
  rw [innerTail471_6, scan, if_neg (by decide), innerCandidates471_6_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_7
    (if 3 ≤ innerCandidates471_6.length then search edge 3 [3361, 6199] innerCandidates471_6 9560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_7

def innerTail471_5 : List Nat := 5557 :: innerTail471_6
def innerCandidates471_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates471_5_checked : candidates edge 4 3361 5557 innerTail471_6 best.bound = innerCandidates471_5 := by
  unfold candidates
  have row : edge 5557 = SearchData.row5557.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_5 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_5 best = best := by
  rw [innerTail471_5, scan, if_neg (by decide), innerCandidates471_5_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_6
    (if 3 ≤ innerCandidates471_5.length then search edge 3 [3361, 5557] innerCandidates471_5 8918 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_6

def innerTail471_4 : List Nat := 5419 :: innerTail471_5
def innerCandidates471_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates471_4_checked : candidates edge 4 3361 5419 innerTail471_5 best.bound = innerCandidates471_4 := by
  unfold candidates
  have row : edge 5419 = SearchData.row5419.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_4 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_4 best = best := by
  rw [innerTail471_4, scan, if_neg (by decide), innerCandidates471_4_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_5
    (if 3 ≤ innerCandidates471_4.length then search edge 3 [3361, 5419] innerCandidates471_4 8780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_5

def innerTail471_3 : List Nat := 4951 :: innerTail471_4
def innerCandidates471_3 : List Nat := [6343]
set_option maxRecDepth 4096 in
theorem innerCandidates471_3_checked : candidates edge 4 3361 4951 innerTail471_4 best.bound = innerCandidates471_3 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_3 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_3 best = best := by
  rw [innerTail471_3, scan, if_neg (by decide), innerCandidates471_3_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_4
    (if 3 ≤ innerCandidates471_3.length then search edge 3 [3361, 4951] innerCandidates471_3 8312 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_4

def innerTail471_2 : List Nat := 4507 :: innerTail471_3
def innerCandidates471_2 : List Nat := [5419]
set_option maxRecDepth 4096 in
theorem innerCandidates471_2_checked : candidates edge 4 3361 4507 innerTail471_3 best.bound = innerCandidates471_2 := by
  unfold candidates
  have row : edge 4507 = SearchData.row4507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_2 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_2 best = best := by
  rw [innerTail471_2, scan, if_neg (by decide), innerCandidates471_2_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_3
    (if 3 ≤ innerCandidates471_2.length then search edge 3 [3361, 4507] innerCandidates471_2 7868 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_3

def innerTail471_1 : List Nat := 3919 :: innerTail471_2
def innerCandidates471_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates471_1_checked : candidates edge 4 3361 3919 innerTail471_2 best.bound = innerCandidates471_1 := by
  unfold candidates
  have row : edge 3919 = SearchData.row3919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_1 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_1 best = best := by
  rw [innerTail471_1, scan, if_neg (by decide), innerCandidates471_1_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_2
    (if 3 ≤ innerCandidates471_1.length then search edge 3 [3361, 3919] innerCandidates471_1 7280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_2

def innerTail471_0 : List Nat := 3739 :: innerTail471_1
def innerCandidates471_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates471_0_checked : candidates edge 4 3361 3739 innerTail471_1 best.bound = innerCandidates471_0 := by
  unfold candidates
  have row : edge 3739 = SearchData.row3739.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan471_0 : scan edge (search edge 3) 4 [3361] 3361 innerTail471_0 best = best := by
  rw [innerTail471_0, scan, if_neg (by decide), innerCandidates471_0_checked]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_1
    (if 3 ≤ innerCandidates471_0.length then search edge 3 [3361, 3739] innerCandidates471_0 7100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan471_1

theorem rootBranch471 : search edge 4 [3361] rootCandidates471 3361 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3361] 3361 innerTail471_0 best = best
  exact innerScan471_0

def rootCandidates470 : List Nat := [4217, 4229, 5981, 6257, 6599, 6737, 6779, 7649, 7853]
set_option maxRecDepth 4096 in
theorem rootCandidates470_checked : candidates edge 5 0 3359 tail471 best.bound = rootCandidates470 := by
  unfold candidates
  have row : edge 3359 = SearchData.row3359.contains := by funext q; rfl
  rw [row]
  decide

def innerTail470_9 : List Nat := []
theorem innerScan470_9 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_9 best = best := rfl

def innerTail470_8 : List Nat := 7853 :: innerTail470_9
def innerCandidates470_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates470_8_checked : candidates edge 4 3359 7853 innerTail470_9 best.bound = innerCandidates470_8 := by
  unfold candidates
  have row : edge 7853 = SearchData.row7853.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_8 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_8 best = best := by
  rw [innerTail470_8, scan, if_neg (by decide), innerCandidates470_8_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_9
    (if 3 ≤ innerCandidates470_8.length then search edge 3 [3359, 7853] innerCandidates470_8 11212 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_9

def innerTail470_7 : List Nat := 7649 :: innerTail470_8
def innerCandidates470_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates470_7_checked : candidates edge 4 3359 7649 innerTail470_8 best.bound = innerCandidates470_7 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_7 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_7 best = best := by
  rw [innerTail470_7, scan, if_neg (by decide), innerCandidates470_7_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_8
    (if 3 ≤ innerCandidates470_7.length then search edge 3 [3359, 7649] innerCandidates470_7 11008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_8

def innerTail470_6 : List Nat := 6779 :: innerTail470_7
def innerCandidates470_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates470_6_checked : candidates edge 4 3359 6779 innerTail470_7 best.bound = innerCandidates470_6 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_6 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_6 best = best := by
  rw [innerTail470_6, scan, if_neg (by decide), innerCandidates470_6_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_7
    (if 3 ≤ innerCandidates470_6.length then search edge 3 [3359, 6779] innerCandidates470_6 10138 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_7

def innerTail470_5 : List Nat := 6737 :: innerTail470_6
def innerCandidates470_5 : List Nat := [6779]
set_option maxRecDepth 4096 in
theorem innerCandidates470_5_checked : candidates edge 4 3359 6737 innerTail470_6 best.bound = innerCandidates470_5 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_5 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_5 best = best := by
  rw [innerTail470_5, scan, if_neg (by decide), innerCandidates470_5_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_6
    (if 3 ≤ innerCandidates470_5.length then search edge 3 [3359, 6737] innerCandidates470_5 10096 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_6

def innerTail470_4 : List Nat := 6599 :: innerTail470_5
def innerCandidates470_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates470_4_checked : candidates edge 4 3359 6599 innerTail470_5 best.bound = innerCandidates470_4 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_4 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_4 best = best := by
  rw [innerTail470_4, scan, if_neg (by decide), innerCandidates470_4_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_5
    (if 3 ≤ innerCandidates470_4.length then search edge 3 [3359, 6599] innerCandidates470_4 9958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_5

def innerTail470_3 : List Nat := 6257 :: innerTail470_4
def innerCandidates470_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates470_3_checked : candidates edge 4 3359 6257 innerTail470_4 best.bound = innerCandidates470_3 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_3 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_3 best = best := by
  rw [innerTail470_3, scan, if_neg (by decide), innerCandidates470_3_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_4
    (if 3 ≤ innerCandidates470_3.length then search edge 3 [3359, 6257] innerCandidates470_3 9616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_4

def innerTail470_2 : List Nat := 5981 :: innerTail470_3
def innerCandidates470_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates470_2_checked : candidates edge 4 3359 5981 innerTail470_3 best.bound = innerCandidates470_2 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_2 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_2 best = best := by
  rw [innerTail470_2, scan, if_neg (by decide), innerCandidates470_2_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_3
    (if 3 ≤ innerCandidates470_2.length then search edge 3 [3359, 5981] innerCandidates470_2 9340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_3

def innerTail470_1 : List Nat := 4229 :: innerTail470_2
def innerCandidates470_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates470_1_checked : candidates edge 4 3359 4229 innerTail470_2 best.bound = innerCandidates470_1 := by
  unfold candidates
  have row : edge 4229 = SearchData.row4229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_1 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_1 best = best := by
  rw [innerTail470_1, scan, if_neg (by decide), innerCandidates470_1_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_2
    (if 3 ≤ innerCandidates470_1.length then search edge 3 [3359, 4229] innerCandidates470_1 7588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_2

def innerTail470_0 : List Nat := 4217 :: innerTail470_1
def innerCandidates470_0 : List Nat := [5981]
set_option maxRecDepth 4096 in
theorem innerCandidates470_0_checked : candidates edge 4 3359 4217 innerTail470_1 best.bound = innerCandidates470_0 := by
  unfold candidates
  have row : edge 4217 = SearchData.row4217.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan470_0 : scan edge (search edge 3) 4 [3359] 3359 innerTail470_0 best = best := by
  rw [innerTail470_0, scan, if_neg (by decide), innerCandidates470_0_checked]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_1
    (if 3 ≤ innerCandidates470_0.length then search edge 3 [3359, 4217] innerCandidates470_0 7576 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan470_1

theorem rootBranch470 : search edge 4 [3359] rootCandidates470 3359 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [3359] 3359 innerTail470_0 best = best
  exact innerScan470_0

end ProjectEulerSolutions.PrimePairUpper.Search
