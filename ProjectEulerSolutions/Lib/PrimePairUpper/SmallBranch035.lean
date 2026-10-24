import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates767 : List Nat := [5953, 6481, 7723, 8059, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates767_checked : candidates edge 5 0 5857 tail768 best.bound = rootCandidates767 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext q; rfl
  rw [row]
  decide

def innerTail767_5 : List Nat := []
theorem innerScan767_5 : scan edge (search edge 3) 4 [5857] 5857 innerTail767_5 best = best := rfl

def innerTail767_4 : List Nat := 8293 :: innerTail767_5
def innerCandidates767_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates767_4_checked : candidates edge 4 5857 8293 innerTail767_5 best.bound = innerCandidates767_4 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan767_4 : scan edge (search edge 3) 4 [5857] 5857 innerTail767_4 best = best := by
  rw [innerTail767_4, scan, if_neg (by decide), innerCandidates767_4_checked]
  change scan edge (search edge 3) 4 [5857] 5857 innerTail767_5
    (if 3 ≤ innerCandidates767_4.length then search edge 3 [5857, 8293] innerCandidates767_4 14150 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan767_5

def innerTail767_3 : List Nat := 8059 :: innerTail767_4
def innerCandidates767_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates767_3_checked : candidates edge 4 5857 8059 innerTail767_4 best.bound = innerCandidates767_3 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan767_3 : scan edge (search edge 3) 4 [5857] 5857 innerTail767_3 best = best := by
  rw [innerTail767_3, scan, if_neg (by decide), innerCandidates767_3_checked]
  change scan edge (search edge 3) 4 [5857] 5857 innerTail767_4
    (if 3 ≤ innerCandidates767_3.length then search edge 3 [5857, 8059] innerCandidates767_3 13916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan767_4

def innerTail767_2 : List Nat := 7723 :: innerTail767_3
def innerCandidates767_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates767_2_checked : candidates edge 4 5857 7723 innerTail767_3 best.bound = innerCandidates767_2 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan767_2 : scan edge (search edge 3) 4 [5857] 5857 innerTail767_2 best = best := by
  rw [innerTail767_2, scan, if_neg (by decide), innerCandidates767_2_checked]
  change scan edge (search edge 3) 4 [5857] 5857 innerTail767_3
    (if 3 ≤ innerCandidates767_2.length then search edge 3 [5857, 7723] innerCandidates767_2 13580 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan767_3

def innerTail767_1 : List Nat := 6481 :: innerTail767_2
def innerCandidates767_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates767_1_checked : candidates edge 4 5857 6481 innerTail767_2 best.bound = innerCandidates767_1 := by
  unfold candidates
  have row : edge 6481 = SearchData.row6481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan767_1 : scan edge (search edge 3) 4 [5857] 5857 innerTail767_1 best = best := by
  rw [innerTail767_1, scan, if_neg (by decide), innerCandidates767_1_checked]
  change scan edge (search edge 3) 4 [5857] 5857 innerTail767_2
    (if 3 ≤ innerCandidates767_1.length then search edge 3 [5857, 6481] innerCandidates767_1 12338 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan767_2

def innerTail767_0 : List Nat := 5953 :: innerTail767_1
def innerCandidates767_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates767_0_checked : candidates edge 4 5857 5953 innerTail767_1 best.bound = innerCandidates767_0 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan767_0 : scan edge (search edge 3) 4 [5857] 5857 innerTail767_0 best = best := by
  rw [innerTail767_0, scan, if_neg (by decide), innerCandidates767_0_checked]
  change scan edge (search edge 3) 4 [5857] 5857 innerTail767_1
    (if 3 ≤ innerCandidates767_0.length then search edge 3 [5857, 5953] innerCandidates767_0 11810 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan767_1

theorem rootBranch767 : search edge 4 [5857] rootCandidates767 5857 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5857] 5857 innerTail767_0 best = best
  exact innerScan767_0

def rootCandidates766 : List Nat := [6373, 6607, 6793, 6961, 7507, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates766_checked : candidates edge 5 0 5851 tail767 best.bound = rootCandidates766 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext q; rfl
  rw [row]
  decide

def innerTail766_6 : List Nat := []
theorem innerScan766_6 : scan edge (search edge 3) 4 [5851] 5851 innerTail766_6 best = best := rfl

def innerTail766_5 : List Nat := 8377 :: innerTail766_6
def innerCandidates766_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates766_5_checked : candidates edge 4 5851 8377 innerTail766_6 best.bound = innerCandidates766_5 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan766_5 : scan edge (search edge 3) 4 [5851] 5851 innerTail766_5 best = best := by
  rw [innerTail766_5, scan, if_neg (by decide), innerCandidates766_5_checked]
  change scan edge (search edge 3) 4 [5851] 5851 innerTail766_6
    (if 3 ≤ innerCandidates766_5.length then search edge 3 [5851, 8377] innerCandidates766_5 14228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan766_6

def innerTail766_4 : List Nat := 7507 :: innerTail766_5
def innerCandidates766_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates766_4_checked : candidates edge 4 5851 7507 innerTail766_5 best.bound = innerCandidates766_4 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan766_4 : scan edge (search edge 3) 4 [5851] 5851 innerTail766_4 best = best := by
  rw [innerTail766_4, scan, if_neg (by decide), innerCandidates766_4_checked]
  change scan edge (search edge 3) 4 [5851] 5851 innerTail766_5
    (if 3 ≤ innerCandidates766_4.length then search edge 3 [5851, 7507] innerCandidates766_4 13358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan766_5

def innerTail766_3 : List Nat := 6961 :: innerTail766_4
def innerCandidates766_3 : List Nat := [8377]
set_option maxRecDepth 4096 in
theorem innerCandidates766_3_checked : candidates edge 4 5851 6961 innerTail766_4 best.bound = innerCandidates766_3 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan766_3 : scan edge (search edge 3) 4 [5851] 5851 innerTail766_3 best = best := by
  rw [innerTail766_3, scan, if_neg (by decide), innerCandidates766_3_checked]
  change scan edge (search edge 3) 4 [5851] 5851 innerTail766_4
    (if 3 ≤ innerCandidates766_3.length then search edge 3 [5851, 6961] innerCandidates766_3 12812 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan766_4

def innerTail766_2 : List Nat := 6793 :: innerTail766_3
def innerCandidates766_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates766_2_checked : candidates edge 4 5851 6793 innerTail766_3 best.bound = innerCandidates766_2 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan766_2 : scan edge (search edge 3) 4 [5851] 5851 innerTail766_2 best = best := by
  rw [innerTail766_2, scan, if_neg (by decide), innerCandidates766_2_checked]
  change scan edge (search edge 3) 4 [5851] 5851 innerTail766_3
    (if 3 ≤ innerCandidates766_2.length then search edge 3 [5851, 6793] innerCandidates766_2 12644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan766_3

def innerTail766_1 : List Nat := 6607 :: innerTail766_2
def innerCandidates766_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates766_1_checked : candidates edge 4 5851 6607 innerTail766_2 best.bound = innerCandidates766_1 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan766_1 : scan edge (search edge 3) 4 [5851] 5851 innerTail766_1 best = best := by
  rw [innerTail766_1, scan, if_neg (by decide), innerCandidates766_1_checked]
  change scan edge (search edge 3) 4 [5851] 5851 innerTail766_2
    (if 3 ≤ innerCandidates766_1.length then search edge 3 [5851, 6607] innerCandidates766_1 12458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan766_2

def innerTail766_0 : List Nat := 6373 :: innerTail766_1
def innerCandidates766_0 : List Nat := [7507]
set_option maxRecDepth 4096 in
theorem innerCandidates766_0_checked : candidates edge 4 5851 6373 innerTail766_1 best.bound = innerCandidates766_0 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan766_0 : scan edge (search edge 3) 4 [5851] 5851 innerTail766_0 best = best := by
  rw [innerTail766_0, scan, if_neg (by decide), innerCandidates766_0_checked]
  change scan edge (search edge 3) 4 [5851] 5851 innerTail766_1
    (if 3 ≤ innerCandidates766_0.length then search edge 3 [5851, 6373] innerCandidates766_0 12224 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan766_1

theorem rootBranch766 : search edge 4 [5851] rootCandidates766 5851 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5851] 5851 innerTail766_0 best = best
  exact innerScan766_0

def rootCandidates765 : List Nat := [6203, 7949]
set_option maxRecDepth 4096 in
theorem rootCandidates765_checked : candidates edge 5 0 5849 tail766 best.bound = rootCandidates765 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates764 : List Nat := [6977, 7583, 7937]
set_option maxRecDepth 4096 in
theorem rootCandidates764_checked : candidates edge 5 0 5843 tail765 best.bound = rootCandidates764 := by
  unfold candidates
  have row : edge 5843 = SearchData.row5843.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates763 : List Nat := [6199, 6733, 7237, 7681, 7753, 7879, 8233, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates763_checked : candidates edge 5 0 5839 tail764 best.bound = rootCandidates763 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext q; rfl
  rw [row]
  decide

def innerTail763_8 : List Nat := []
theorem innerScan763_8 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_8 best = best := rfl

def innerTail763_7 : List Nat := 8269 :: innerTail763_8
def innerCandidates763_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates763_7_checked : candidates edge 4 5839 8269 innerTail763_8 best.bound = innerCandidates763_7 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_7 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_7 best = best := by
  rw [innerTail763_7, scan, if_neg (by decide), innerCandidates763_7_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_8
    (if 3 ≤ innerCandidates763_7.length then search edge 3 [5839, 8269] innerCandidates763_7 14108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_8

def innerTail763_6 : List Nat := 8233 :: innerTail763_7
def innerCandidates763_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates763_6_checked : candidates edge 4 5839 8233 innerTail763_7 best.bound = innerCandidates763_6 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_6 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_6 best = best := by
  rw [innerTail763_6, scan, if_neg (by decide), innerCandidates763_6_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_7
    (if 3 ≤ innerCandidates763_6.length then search edge 3 [5839, 8233] innerCandidates763_6 14072 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_7

def innerTail763_5 : List Nat := 7879 :: innerTail763_6
def innerCandidates763_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates763_5_checked : candidates edge 4 5839 7879 innerTail763_6 best.bound = innerCandidates763_5 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_5 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_5 best = best := by
  rw [innerTail763_5, scan, if_neg (by decide), innerCandidates763_5_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_6
    (if 3 ≤ innerCandidates763_5.length then search edge 3 [5839, 7879] innerCandidates763_5 13718 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_6

def innerTail763_4 : List Nat := 7753 :: innerTail763_5
def innerCandidates763_4 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates763_4_checked : candidates edge 4 5839 7753 innerTail763_5 best.bound = innerCandidates763_4 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_4 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_4 best = best := by
  rw [innerTail763_4, scan, if_neg (by decide), innerCandidates763_4_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_5
    (if 3 ≤ innerCandidates763_4.length then search edge 3 [5839, 7753] innerCandidates763_4 13592 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_5

def innerTail763_3 : List Nat := 7681 :: innerTail763_4
def innerCandidates763_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates763_3_checked : candidates edge 4 5839 7681 innerTail763_4 best.bound = innerCandidates763_3 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_3 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_3 best = best := by
  rw [innerTail763_3, scan, if_neg (by decide), innerCandidates763_3_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_4
    (if 3 ≤ innerCandidates763_3.length then search edge 3 [5839, 7681] innerCandidates763_3 13520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_4

def innerTail763_2 : List Nat := 7237 :: innerTail763_3
def innerCandidates763_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates763_2_checked : candidates edge 4 5839 7237 innerTail763_3 best.bound = innerCandidates763_2 := by
  unfold candidates
  have row : edge 7237 = SearchData.row7237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_2 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_2 best = best := by
  rw [innerTail763_2, scan, if_neg (by decide), innerCandidates763_2_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_3
    (if 3 ≤ innerCandidates763_2.length then search edge 3 [5839, 7237] innerCandidates763_2 13076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_3

def innerTail763_1 : List Nat := 6733 :: innerTail763_2
def innerCandidates763_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates763_1_checked : candidates edge 4 5839 6733 innerTail763_2 best.bound = innerCandidates763_1 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_1 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_1 best = best := by
  rw [innerTail763_1, scan, if_neg (by decide), innerCandidates763_1_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_2
    (if 3 ≤ innerCandidates763_1.length then search edge 3 [5839, 6733] innerCandidates763_1 12572 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_2

def innerTail763_0 : List Nat := 6199 :: innerTail763_1
def innerCandidates763_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates763_0_checked : candidates edge 4 5839 6199 innerTail763_1 best.bound = innerCandidates763_0 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan763_0 : scan edge (search edge 3) 4 [5839] 5839 innerTail763_0 best = best := by
  rw [innerTail763_0, scan, if_neg (by decide), innerCandidates763_0_checked]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_1
    (if 3 ≤ innerCandidates763_0.length then search edge 3 [5839, 6199] innerCandidates763_0 12038 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan763_1

theorem rootBranch763 : search edge 4 [5839] rootCandidates763 5839 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5839] 5839 innerTail763_0 best = best
  exact innerScan763_0

def rootCandidates762 : List Nat := [6151, 6991, 7159, 7213, 7507, 7621, 8053, 8089]
set_option maxRecDepth 4096 in
theorem rootCandidates762_checked : candidates edge 5 0 5827 tail763 best.bound = rootCandidates762 := by
  unfold candidates
  have row : edge 5827 = SearchData.row5827.contains := by funext q; rfl
  rw [row]
  decide

def innerTail762_8 : List Nat := []
theorem innerScan762_8 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_8 best = best := rfl

def innerTail762_7 : List Nat := 8089 :: innerTail762_8
def innerCandidates762_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates762_7_checked : candidates edge 4 5827 8089 innerTail762_8 best.bound = innerCandidates762_7 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_7 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_7 best = best := by
  rw [innerTail762_7, scan, if_neg (by decide), innerCandidates762_7_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_8
    (if 3 ≤ innerCandidates762_7.length then search edge 3 [5827, 8089] innerCandidates762_7 13916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_8

def innerTail762_6 : List Nat := 8053 :: innerTail762_7
def innerCandidates762_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates762_6_checked : candidates edge 4 5827 8053 innerTail762_7 best.bound = innerCandidates762_6 := by
  unfold candidates
  have row : edge 8053 = SearchData.row8053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_6 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_6 best = best := by
  rw [innerTail762_6, scan, if_neg (by decide), innerCandidates762_6_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_7
    (if 3 ≤ innerCandidates762_6.length then search edge 3 [5827, 8053] innerCandidates762_6 13880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_7

def innerTail762_5 : List Nat := 7621 :: innerTail762_6
def innerCandidates762_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates762_5_checked : candidates edge 4 5827 7621 innerTail762_6 best.bound = innerCandidates762_5 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_5 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_5 best = best := by
  rw [innerTail762_5, scan, if_neg (by decide), innerCandidates762_5_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_6
    (if 3 ≤ innerCandidates762_5.length then search edge 3 [5827, 7621] innerCandidates762_5 13448 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_6

def innerTail762_4 : List Nat := 7507 :: innerTail762_5
def innerCandidates762_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates762_4_checked : candidates edge 4 5827 7507 innerTail762_5 best.bound = innerCandidates762_4 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_4 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_4 best = best := by
  rw [innerTail762_4, scan, if_neg (by decide), innerCandidates762_4_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_5
    (if 3 ≤ innerCandidates762_4.length then search edge 3 [5827, 7507] innerCandidates762_4 13334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_5

def innerTail762_3 : List Nat := 7213 :: innerTail762_4
def innerCandidates762_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates762_3_checked : candidates edge 4 5827 7213 innerTail762_4 best.bound = innerCandidates762_3 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_3 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_3 best = best := by
  rw [innerTail762_3, scan, if_neg (by decide), innerCandidates762_3_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_4
    (if 3 ≤ innerCandidates762_3.length then search edge 3 [5827, 7213] innerCandidates762_3 13040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_4

def innerTail762_2 : List Nat := 7159 :: innerTail762_3
def innerCandidates762_2 : List Nat := [7213]
set_option maxRecDepth 4096 in
theorem innerCandidates762_2_checked : candidates edge 4 5827 7159 innerTail762_3 best.bound = innerCandidates762_2 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_2 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_2 best = best := by
  rw [innerTail762_2, scan, if_neg (by decide), innerCandidates762_2_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_3
    (if 3 ≤ innerCandidates762_2.length then search edge 3 [5827, 7159] innerCandidates762_2 12986 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_3

def innerTail762_1 : List Nat := 6991 :: innerTail762_2
def innerCandidates762_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates762_1_checked : candidates edge 4 5827 6991 innerTail762_2 best.bound = innerCandidates762_1 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_1 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_1 best = best := by
  rw [innerTail762_1, scan, if_neg (by decide), innerCandidates762_1_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_2
    (if 3 ≤ innerCandidates762_1.length then search edge 3 [5827, 6991] innerCandidates762_1 12818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_2

def innerTail762_0 : List Nat := 6151 :: innerTail762_1
def innerCandidates762_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates762_0_checked : candidates edge 4 5827 6151 innerTail762_1 best.bound = innerCandidates762_0 := by
  unfold candidates
  have row : edge 6151 = SearchData.row6151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan762_0 : scan edge (search edge 3) 4 [5827] 5827 innerTail762_0 best = best := by
  rw [innerTail762_0, scan, if_neg (by decide), innerCandidates762_0_checked]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_1
    (if 3 ≤ innerCandidates762_0.length then search edge 3 [5827, 6151] innerCandidates762_0 11978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan762_1

theorem rootBranch762 : search edge 4 [5827] rootCandidates762 5827 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5827] 5827 innerTail762_0 best = best
  exact innerScan762_0

def rootCandidates761 : List Nat := [6043, 6121, 6373, 6637, 6967, 8059, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates761_checked : candidates edge 5 0 5821 tail762 best.bound = rootCandidates761 := by
  unfold candidates
  have row : edge 5821 = SearchData.row5821.contains := by funext q; rfl
  rw [row]
  decide

def innerTail761_7 : List Nat := []
theorem innerScan761_7 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_7 best = best := rfl

def innerTail761_6 : List Nat := 8233 :: innerTail761_7
def innerCandidates761_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates761_6_checked : candidates edge 4 5821 8233 innerTail761_7 best.bound = innerCandidates761_6 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan761_6 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_6 best = best := by
  rw [innerTail761_6, scan, if_neg (by decide), innerCandidates761_6_checked]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_7
    (if 3 ≤ innerCandidates761_6.length then search edge 3 [5821, 8233] innerCandidates761_6 14054 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan761_7

def innerTail761_5 : List Nat := 8059 :: innerTail761_6
def innerCandidates761_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates761_5_checked : candidates edge 4 5821 8059 innerTail761_6 best.bound = innerCandidates761_5 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan761_5 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_5 best = best := by
  rw [innerTail761_5, scan, if_neg (by decide), innerCandidates761_5_checked]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_6
    (if 3 ≤ innerCandidates761_5.length then search edge 3 [5821, 8059] innerCandidates761_5 13880 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan761_6

def innerTail761_4 : List Nat := 6967 :: innerTail761_5
def innerCandidates761_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates761_4_checked : candidates edge 4 5821 6967 innerTail761_5 best.bound = innerCandidates761_4 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan761_4 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_4 best = best := by
  rw [innerTail761_4, scan, if_neg (by decide), innerCandidates761_4_checked]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_5
    (if 3 ≤ innerCandidates761_4.length then search edge 3 [5821, 6967] innerCandidates761_4 12788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan761_5

def innerTail761_3 : List Nat := 6637 :: innerTail761_4
def innerCandidates761_3 : List Nat := [8233]
set_option maxRecDepth 4096 in
theorem innerCandidates761_3_checked : candidates edge 4 5821 6637 innerTail761_4 best.bound = innerCandidates761_3 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan761_3 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_3 best = best := by
  rw [innerTail761_3, scan, if_neg (by decide), innerCandidates761_3_checked]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_4
    (if 3 ≤ innerCandidates761_3.length then search edge 3 [5821, 6637] innerCandidates761_3 12458 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan761_4

def innerTail761_2 : List Nat := 6373 :: innerTail761_3
def innerCandidates761_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates761_2_checked : candidates edge 4 5821 6373 innerTail761_3 best.bound = innerCandidates761_2 := by
  unfold candidates
  have row : edge 6373 = SearchData.row6373.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan761_2 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_2 best = best := by
  rw [innerTail761_2, scan, if_neg (by decide), innerCandidates761_2_checked]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_3
    (if 3 ≤ innerCandidates761_2.length then search edge 3 [5821, 6373] innerCandidates761_2 12194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan761_3

def innerTail761_1 : List Nat := 6121 :: innerTail761_2
def innerCandidates761_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates761_1_checked : candidates edge 4 5821 6121 innerTail761_2 best.bound = innerCandidates761_1 := by
  unfold candidates
  have row : edge 6121 = SearchData.row6121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan761_1 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_1 best = best := by
  rw [innerTail761_1, scan, if_neg (by decide), innerCandidates761_1_checked]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_2
    (if 3 ≤ innerCandidates761_1.length then search edge 3 [5821, 6121] innerCandidates761_1 11942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan761_2

def innerTail761_0 : List Nat := 6043 :: innerTail761_1
def innerCandidates761_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates761_0_checked : candidates edge 4 5821 6043 innerTail761_1 best.bound = innerCandidates761_0 := by
  unfold candidates
  have row : edge 6043 = SearchData.row6043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan761_0 : scan edge (search edge 3) 4 [5821] 5821 innerTail761_0 best = best := by
  rw [innerTail761_0, scan, if_neg (by decide), innerCandidates761_0_checked]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_1
    (if 3 ≤ innerCandidates761_0.length then search edge 3 [5821, 6043] innerCandidates761_0 11864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan761_1

theorem rootBranch761 : search edge 4 [5821] rootCandidates761 5821 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5821] 5821 innerTail761_0 best = best
  exact innerScan761_0

def rootCandidates760 : List Nat := [6737, 6803, 6863, 7151, 7193, 7517, 7829, 7907, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates760_checked : candidates edge 5 0 5813 tail761 best.bound = rootCandidates760 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext q; rfl
  rw [row]
  decide

def innerTail760_9 : List Nat := []
theorem innerScan760_9 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_9 best = best := rfl

def innerTail760_8 : List Nat := 8363 :: innerTail760_9
def innerCandidates760_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_8_checked : candidates edge 4 5813 8363 innerTail760_9 best.bound = innerCandidates760_8 := by
  unfold candidates
  have row : edge 8363 = SearchData.row8363.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_8 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_8 best = best := by
  rw [innerTail760_8, scan, if_neg (by decide), innerCandidates760_8_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_9
    (if 3 ≤ innerCandidates760_8.length then search edge 3 [5813, 8363] innerCandidates760_8 14176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_9

def innerTail760_7 : List Nat := 7907 :: innerTail760_8
def innerCandidates760_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_7_checked : candidates edge 4 5813 7907 innerTail760_8 best.bound = innerCandidates760_7 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_7 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_7 best = best := by
  rw [innerTail760_7, scan, if_neg (by decide), innerCandidates760_7_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_8
    (if 3 ≤ innerCandidates760_7.length then search edge 3 [5813, 7907] innerCandidates760_7 13720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_8

def innerTail760_6 : List Nat := 7829 :: innerTail760_7
def innerCandidates760_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_6_checked : candidates edge 4 5813 7829 innerTail760_7 best.bound = innerCandidates760_6 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_6 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_6 best = best := by
  rw [innerTail760_6, scan, if_neg (by decide), innerCandidates760_6_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_7
    (if 3 ≤ innerCandidates760_6.length then search edge 3 [5813, 7829] innerCandidates760_6 13642 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_7

def innerTail760_5 : List Nat := 7517 :: innerTail760_6
def innerCandidates760_5 : List Nat := [7829]
set_option maxRecDepth 4096 in
theorem innerCandidates760_5_checked : candidates edge 4 5813 7517 innerTail760_6 best.bound = innerCandidates760_5 := by
  unfold candidates
  have row : edge 7517 = SearchData.row7517.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_5 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_5 best = best := by
  rw [innerTail760_5, scan, if_neg (by decide), innerCandidates760_5_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_6
    (if 3 ≤ innerCandidates760_5.length then search edge 3 [5813, 7517] innerCandidates760_5 13330 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_6

def innerTail760_4 : List Nat := 7193 :: innerTail760_5
def innerCandidates760_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_4_checked : candidates edge 4 5813 7193 innerTail760_5 best.bound = innerCandidates760_4 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_4 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_4 best = best := by
  rw [innerTail760_4, scan, if_neg (by decide), innerCandidates760_4_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_5
    (if 3 ≤ innerCandidates760_4.length then search edge 3 [5813, 7193] innerCandidates760_4 13006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_5

def innerTail760_3 : List Nat := 7151 :: innerTail760_4
def innerCandidates760_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_3_checked : candidates edge 4 5813 7151 innerTail760_4 best.bound = innerCandidates760_3 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_3 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_3 best = best := by
  rw [innerTail760_3, scan, if_neg (by decide), innerCandidates760_3_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_4
    (if 3 ≤ innerCandidates760_3.length then search edge 3 [5813, 7151] innerCandidates760_3 12964 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_4

def innerTail760_2 : List Nat := 6863 :: innerTail760_3
def innerCandidates760_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_2_checked : candidates edge 4 5813 6863 innerTail760_3 best.bound = innerCandidates760_2 := by
  unfold candidates
  have row : edge 6863 = SearchData.row6863.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_2 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_2 best = best := by
  rw [innerTail760_2, scan, if_neg (by decide), innerCandidates760_2_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_3
    (if 3 ≤ innerCandidates760_2.length then search edge 3 [5813, 6863] innerCandidates760_2 12676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_3

def innerTail760_1 : List Nat := 6803 :: innerTail760_2
def innerCandidates760_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_1_checked : candidates edge 4 5813 6803 innerTail760_2 best.bound = innerCandidates760_1 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_1 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_1 best = best := by
  rw [innerTail760_1, scan, if_neg (by decide), innerCandidates760_1_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_2
    (if 3 ≤ innerCandidates760_1.length then search edge 3 [5813, 6803] innerCandidates760_1 12616 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_2

def innerTail760_0 : List Nat := 6737 :: innerTail760_1
def innerCandidates760_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates760_0_checked : candidates edge 4 5813 6737 innerTail760_1 best.bound = innerCandidates760_0 := by
  unfold candidates
  have row : edge 6737 = SearchData.row6737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan760_0 : scan edge (search edge 3) 4 [5813] 5813 innerTail760_0 best = best := by
  rw [innerTail760_0, scan, if_neg (by decide), innerCandidates760_0_checked]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_1
    (if 3 ≤ innerCandidates760_0.length then search edge 3 [5813, 6737] innerCandidates760_0 12550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan760_1

theorem rootBranch760 : search edge 4 [5813] rootCandidates760 5813 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5813] 5813 innerTail760_0 best = best
  exact innerScan760_0

end ProjectEulerSolutions.PrimePairUpper.Search
