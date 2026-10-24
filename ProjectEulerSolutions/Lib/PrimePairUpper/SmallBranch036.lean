import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates759 : List Nat := [6329, 7499, 7577, 7949, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates759_checked : candidates edge 5 0 5807 tail760 best.bound = rootCandidates759 := by
  unfold candidates
  have row : edge 5807 = SearchData.row5807.contains := by funext q; rfl
  rw [row]
  decide

def innerTail759_5 : List Nat := []
theorem innerScan759_5 : scan edge (search edge 3) 4 [5807] 5807 innerTail759_5 best = best := rfl

def innerTail759_4 : List Nat := 8297 :: innerTail759_5
def innerCandidates759_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates759_4_checked : candidates edge 4 5807 8297 innerTail759_5 best.bound = innerCandidates759_4 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan759_4 : scan edge (search edge 3) 4 [5807] 5807 innerTail759_4 best = best := by
  rw [innerTail759_4, scan, if_neg (by decide), innerCandidates759_4_checked]
  change scan edge (search edge 3) 4 [5807] 5807 innerTail759_5
    (if 3 ≤ innerCandidates759_4.length then search edge 3 [5807, 8297] innerCandidates759_4 14104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan759_5

def innerTail759_3 : List Nat := 7949 :: innerTail759_4
def innerCandidates759_3 : List Nat := [8297]
set_option maxRecDepth 4096 in
theorem innerCandidates759_3_checked : candidates edge 4 5807 7949 innerTail759_4 best.bound = innerCandidates759_3 := by
  unfold candidates
  have row : edge 7949 = SearchData.row7949.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan759_3 : scan edge (search edge 3) 4 [5807] 5807 innerTail759_3 best = best := by
  rw [innerTail759_3, scan, if_neg (by decide), innerCandidates759_3_checked]
  change scan edge (search edge 3) 4 [5807] 5807 innerTail759_4
    (if 3 ≤ innerCandidates759_3.length then search edge 3 [5807, 7949] innerCandidates759_3 13756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan759_4

def innerTail759_2 : List Nat := 7577 :: innerTail759_3
def innerCandidates759_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates759_2_checked : candidates edge 4 5807 7577 innerTail759_3 best.bound = innerCandidates759_2 := by
  unfold candidates
  have row : edge 7577 = SearchData.row7577.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan759_2 : scan edge (search edge 3) 4 [5807] 5807 innerTail759_2 best = best := by
  rw [innerTail759_2, scan, if_neg (by decide), innerCandidates759_2_checked]
  change scan edge (search edge 3) 4 [5807] 5807 innerTail759_3
    (if 3 ≤ innerCandidates759_2.length then search edge 3 [5807, 7577] innerCandidates759_2 13384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan759_3

def innerTail759_1 : List Nat := 7499 :: innerTail759_2
def innerCandidates759_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates759_1_checked : candidates edge 4 5807 7499 innerTail759_2 best.bound = innerCandidates759_1 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan759_1 : scan edge (search edge 3) 4 [5807] 5807 innerTail759_1 best = best := by
  rw [innerTail759_1, scan, if_neg (by decide), innerCandidates759_1_checked]
  change scan edge (search edge 3) 4 [5807] 5807 innerTail759_2
    (if 3 ≤ innerCandidates759_1.length then search edge 3 [5807, 7499] innerCandidates759_1 13306 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan759_2

def innerTail759_0 : List Nat := 6329 :: innerTail759_1
def innerCandidates759_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates759_0_checked : candidates edge 4 5807 6329 innerTail759_1 best.bound = innerCandidates759_0 := by
  unfold candidates
  have row : edge 6329 = SearchData.row6329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan759_0 : scan edge (search edge 3) 4 [5807] 5807 innerTail759_0 best = best := by
  rw [innerTail759_0, scan, if_neg (by decide), innerCandidates759_0_checked]
  change scan edge (search edge 3) 4 [5807] 5807 innerTail759_1
    (if 3 ≤ innerCandidates759_0.length then search edge 3 [5807, 6329] innerCandidates759_0 12136 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan759_1

theorem rootBranch759 : search edge 4 [5807] rootCandidates759 5807 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5807] 5807 innerTail759_0 best = best
  exact innerScan759_0

def rootCandidates758 : List Nat := [5897, 6029, 6131, 6173, 6263, 7919]
set_option maxRecDepth 4096 in
theorem rootCandidates758_checked : candidates edge 5 0 5801 tail759 best.bound = rootCandidates758 := by
  unfold candidates
  have row : edge 5801 = SearchData.row5801.contains := by funext q; rfl
  rw [row]
  decide

def innerTail758_6 : List Nat := []
theorem innerScan758_6 : scan edge (search edge 3) 4 [5801] 5801 innerTail758_6 best = best := rfl

def innerTail758_5 : List Nat := 7919 :: innerTail758_6
def innerCandidates758_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates758_5_checked : candidates edge 4 5801 7919 innerTail758_6 best.bound = innerCandidates758_5 := by
  unfold candidates
  have row : edge 7919 = SearchData.row7919.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan758_5 : scan edge (search edge 3) 4 [5801] 5801 innerTail758_5 best = best := by
  rw [innerTail758_5, scan, if_neg (by decide), innerCandidates758_5_checked]
  change scan edge (search edge 3) 4 [5801] 5801 innerTail758_6
    (if 3 ≤ innerCandidates758_5.length then search edge 3 [5801, 7919] innerCandidates758_5 13720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan758_6

def innerTail758_4 : List Nat := 6263 :: innerTail758_5
def innerCandidates758_4 : List Nat := [7919]
set_option maxRecDepth 4096 in
theorem innerCandidates758_4_checked : candidates edge 4 5801 6263 innerTail758_5 best.bound = innerCandidates758_4 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan758_4 : scan edge (search edge 3) 4 [5801] 5801 innerTail758_4 best = best := by
  rw [innerTail758_4, scan, if_neg (by decide), innerCandidates758_4_checked]
  change scan edge (search edge 3) 4 [5801] 5801 innerTail758_5
    (if 3 ≤ innerCandidates758_4.length then search edge 3 [5801, 6263] innerCandidates758_4 12064 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan758_5

def innerTail758_3 : List Nat := 6173 :: innerTail758_4
def innerCandidates758_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates758_3_checked : candidates edge 4 5801 6173 innerTail758_4 best.bound = innerCandidates758_3 := by
  unfold candidates
  have row : edge 6173 = SearchData.row6173.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan758_3 : scan edge (search edge 3) 4 [5801] 5801 innerTail758_3 best = best := by
  rw [innerTail758_3, scan, if_neg (by decide), innerCandidates758_3_checked]
  change scan edge (search edge 3) 4 [5801] 5801 innerTail758_4
    (if 3 ≤ innerCandidates758_3.length then search edge 3 [5801, 6173] innerCandidates758_3 11974 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan758_4

def innerTail758_2 : List Nat := 6131 :: innerTail758_3
def innerCandidates758_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates758_2_checked : candidates edge 4 5801 6131 innerTail758_3 best.bound = innerCandidates758_2 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan758_2 : scan edge (search edge 3) 4 [5801] 5801 innerTail758_2 best = best := by
  rw [innerTail758_2, scan, if_neg (by decide), innerCandidates758_2_checked]
  change scan edge (search edge 3) 4 [5801] 5801 innerTail758_3
    (if 3 ≤ innerCandidates758_2.length then search edge 3 [5801, 6131] innerCandidates758_2 11932 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan758_3

def innerTail758_1 : List Nat := 6029 :: innerTail758_2
def innerCandidates758_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates758_1_checked : candidates edge 4 5801 6029 innerTail758_2 best.bound = innerCandidates758_1 := by
  unfold candidates
  have row : edge 6029 = SearchData.row6029.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan758_1 : scan edge (search edge 3) 4 [5801] 5801 innerTail758_1 best = best := by
  rw [innerTail758_1, scan, if_neg (by decide), innerCandidates758_1_checked]
  change scan edge (search edge 3) 4 [5801] 5801 innerTail758_2
    (if 3 ≤ innerCandidates758_1.length then search edge 3 [5801, 6029] innerCandidates758_1 11830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan758_2

def innerTail758_0 : List Nat := 5897 :: innerTail758_1
def innerCandidates758_0 : List Nat := [6131]
set_option maxRecDepth 4096 in
theorem innerCandidates758_0_checked : candidates edge 4 5801 5897 innerTail758_1 best.bound = innerCandidates758_0 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan758_0 : scan edge (search edge 3) 4 [5801] 5801 innerTail758_0 best = best := by
  rw [innerTail758_0, scan, if_neg (by decide), innerCandidates758_0_checked]
  change scan edge (search edge 3) 4 [5801] 5801 innerTail758_1
    (if 3 ≤ innerCandidates758_0.length then search edge 3 [5801, 5897] innerCandidates758_0 11698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan758_1

theorem rootBranch758 : search edge 4 [5801] rootCandidates758 5801 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5801] 5801 innerTail758_0 best = best
  exact innerScan758_0

def rootCandidates757 : List Nat := [5839, 7333, 8293, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates757_checked : candidates edge 5 0 5791 tail758 best.bound = rootCandidates757 := by
  unfold candidates
  have row : edge 5791 = SearchData.row5791.contains := by funext q; rfl
  rw [row]
  decide

def innerTail757_4 : List Nat := []
theorem innerScan757_4 : scan edge (search edge 3) 4 [5791] 5791 innerTail757_4 best = best := rfl

def innerTail757_3 : List Nat := 8317 :: innerTail757_4
def innerCandidates757_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates757_3_checked : candidates edge 4 5791 8317 innerTail757_4 best.bound = innerCandidates757_3 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan757_3 : scan edge (search edge 3) 4 [5791] 5791 innerTail757_3 best = best := by
  rw [innerTail757_3, scan, if_neg (by decide), innerCandidates757_3_checked]
  change scan edge (search edge 3) 4 [5791] 5791 innerTail757_4
    (if 3 ≤ innerCandidates757_3.length then search edge 3 [5791, 8317] innerCandidates757_3 14108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan757_4

def innerTail757_2 : List Nat := 8293 :: innerTail757_3
def innerCandidates757_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates757_2_checked : candidates edge 4 5791 8293 innerTail757_3 best.bound = innerCandidates757_2 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan757_2 : scan edge (search edge 3) 4 [5791] 5791 innerTail757_2 best = best := by
  rw [innerTail757_2, scan, if_neg (by decide), innerCandidates757_2_checked]
  change scan edge (search edge 3) 4 [5791] 5791 innerTail757_3
    (if 3 ≤ innerCandidates757_2.length then search edge 3 [5791, 8293] innerCandidates757_2 14084 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan757_3

def innerTail757_1 : List Nat := 7333 :: innerTail757_2
def innerCandidates757_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates757_1_checked : candidates edge 4 5791 7333 innerTail757_2 best.bound = innerCandidates757_1 := by
  unfold candidates
  have row : edge 7333 = SearchData.row7333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan757_1 : scan edge (search edge 3) 4 [5791] 5791 innerTail757_1 best = best := by
  rw [innerTail757_1, scan, if_neg (by decide), innerCandidates757_1_checked]
  change scan edge (search edge 3) 4 [5791] 5791 innerTail757_2
    (if 3 ≤ innerCandidates757_1.length then search edge 3 [5791, 7333] innerCandidates757_1 13124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan757_2

def innerTail757_0 : List Nat := 5839 :: innerTail757_1
def innerCandidates757_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates757_0_checked : candidates edge 4 5791 5839 innerTail757_1 best.bound = innerCandidates757_0 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan757_0 : scan edge (search edge 3) 4 [5791] 5791 innerTail757_0 best = best := by
  rw [innerTail757_0, scan, if_neg (by decide), innerCandidates757_0_checked]
  change scan edge (search edge 3) 4 [5791] 5791 innerTail757_1
    (if 3 ≤ innerCandidates757_0.length then search edge 3 [5791, 5839] innerCandidates757_0 11630 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan757_1

theorem rootBranch757 : search edge 4 [5791] rootCandidates757 5791 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5791] 5791 innerTail757_0 best = best
  exact innerScan757_0

def rootCandidates756 : List Nat := [6491, 6719, 6959, 6983, 7331, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates756_checked : candidates edge 5 0 5783 tail757 best.bound = rootCandidates756 := by
  unfold candidates
  have row : edge 5783 = SearchData.row5783.contains := by funext q; rfl
  rw [row]
  decide

def innerTail756_6 : List Nat := []
theorem innerScan756_6 : scan edge (search edge 3) 4 [5783] 5783 innerTail756_6 best = best := rfl

def innerTail756_5 : List Nat := 8009 :: innerTail756_6
def innerCandidates756_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates756_5_checked : candidates edge 4 5783 8009 innerTail756_6 best.bound = innerCandidates756_5 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan756_5 : scan edge (search edge 3) 4 [5783] 5783 innerTail756_5 best = best := by
  rw [innerTail756_5, scan, if_neg (by decide), innerCandidates756_5_checked]
  change scan edge (search edge 3) 4 [5783] 5783 innerTail756_6
    (if 3 ≤ innerCandidates756_5.length then search edge 3 [5783, 8009] innerCandidates756_5 13792 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan756_6

def innerTail756_4 : List Nat := 7331 :: innerTail756_5
def innerCandidates756_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates756_4_checked : candidates edge 4 5783 7331 innerTail756_5 best.bound = innerCandidates756_4 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan756_4 : scan edge (search edge 3) 4 [5783] 5783 innerTail756_4 best = best := by
  rw [innerTail756_4, scan, if_neg (by decide), innerCandidates756_4_checked]
  change scan edge (search edge 3) 4 [5783] 5783 innerTail756_5
    (if 3 ≤ innerCandidates756_4.length then search edge 3 [5783, 7331] innerCandidates756_4 13114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan756_5

def innerTail756_3 : List Nat := 6983 :: innerTail756_4
def innerCandidates756_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates756_3_checked : candidates edge 4 5783 6983 innerTail756_4 best.bound = innerCandidates756_3 := by
  unfold candidates
  have row : edge 6983 = SearchData.row6983.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan756_3 : scan edge (search edge 3) 4 [5783] 5783 innerTail756_3 best = best := by
  rw [innerTail756_3, scan, if_neg (by decide), innerCandidates756_3_checked]
  change scan edge (search edge 3) 4 [5783] 5783 innerTail756_4
    (if 3 ≤ innerCandidates756_3.length then search edge 3 [5783, 6983] innerCandidates756_3 12766 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan756_4

def innerTail756_2 : List Nat := 6959 :: innerTail756_3
def innerCandidates756_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates756_2_checked : candidates edge 4 5783 6959 innerTail756_3 best.bound = innerCandidates756_2 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan756_2 : scan edge (search edge 3) 4 [5783] 5783 innerTail756_2 best = best := by
  rw [innerTail756_2, scan, if_neg (by decide), innerCandidates756_2_checked]
  change scan edge (search edge 3) 4 [5783] 5783 innerTail756_3
    (if 3 ≤ innerCandidates756_2.length then search edge 3 [5783, 6959] innerCandidates756_2 12742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan756_3

def innerTail756_1 : List Nat := 6719 :: innerTail756_2
def innerCandidates756_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates756_1_checked : candidates edge 4 5783 6719 innerTail756_2 best.bound = innerCandidates756_1 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan756_1 : scan edge (search edge 3) 4 [5783] 5783 innerTail756_1 best = best := by
  rw [innerTail756_1, scan, if_neg (by decide), innerCandidates756_1_checked]
  change scan edge (search edge 3) 4 [5783] 5783 innerTail756_2
    (if 3 ≤ innerCandidates756_1.length then search edge 3 [5783, 6719] innerCandidates756_1 12502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan756_2

def innerTail756_0 : List Nat := 6491 :: innerTail756_1
def innerCandidates756_0 : List Nat := [6983]
set_option maxRecDepth 4096 in
theorem innerCandidates756_0_checked : candidates edge 4 5783 6491 innerTail756_1 best.bound = innerCandidates756_0 := by
  unfold candidates
  have row : edge 6491 = SearchData.row6491.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan756_0 : scan edge (search edge 3) 4 [5783] 5783 innerTail756_0 best = best := by
  rw [innerTail756_0, scan, if_neg (by decide), innerCandidates756_0_checked]
  change scan edge (search edge 3) 4 [5783] 5783 innerTail756_1
    (if 3 ≤ innerCandidates756_0.length then search edge 3 [5783, 6491] innerCandidates756_0 12274 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan756_1

theorem rootBranch756 : search edge 4 [5783] rootCandidates756 5783 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5783] 5783 innerTail756_0 best = best
  exact innerScan756_0

def rootCandidates755 : List Nat := [6007, 6421, 7591, 7717, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates755_checked : candidates edge 5 0 5779 tail756 best.bound = rootCandidates755 := by
  unfold candidates
  have row : edge 5779 = SearchData.row5779.contains := by funext q; rfl
  rw [row]
  decide

def innerTail755_5 : List Nat := []
theorem innerScan755_5 : scan edge (search edge 3) 4 [5779] 5779 innerTail755_5 best = best := rfl

def innerTail755_4 : List Nat := 8269 :: innerTail755_5
def innerCandidates755_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates755_4_checked : candidates edge 4 5779 8269 innerTail755_5 best.bound = innerCandidates755_4 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan755_4 : scan edge (search edge 3) 4 [5779] 5779 innerTail755_4 best = best := by
  rw [innerTail755_4, scan, if_neg (by decide), innerCandidates755_4_checked]
  change scan edge (search edge 3) 4 [5779] 5779 innerTail755_5
    (if 3 ≤ innerCandidates755_4.length then search edge 3 [5779, 8269] innerCandidates755_4 14048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan755_5

def innerTail755_3 : List Nat := 7717 :: innerTail755_4
def innerCandidates755_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates755_3_checked : candidates edge 4 5779 7717 innerTail755_4 best.bound = innerCandidates755_3 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan755_3 : scan edge (search edge 3) 4 [5779] 5779 innerTail755_3 best = best := by
  rw [innerTail755_3, scan, if_neg (by decide), innerCandidates755_3_checked]
  change scan edge (search edge 3) 4 [5779] 5779 innerTail755_4
    (if 3 ≤ innerCandidates755_3.length then search edge 3 [5779, 7717] innerCandidates755_3 13496 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan755_4

def innerTail755_2 : List Nat := 7591 :: innerTail755_3
def innerCandidates755_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates755_2_checked : candidates edge 4 5779 7591 innerTail755_3 best.bound = innerCandidates755_2 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan755_2 : scan edge (search edge 3) 4 [5779] 5779 innerTail755_2 best = best := by
  rw [innerTail755_2, scan, if_neg (by decide), innerCandidates755_2_checked]
  change scan edge (search edge 3) 4 [5779] 5779 innerTail755_3
    (if 3 ≤ innerCandidates755_2.length then search edge 3 [5779, 7591] innerCandidates755_2 13370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan755_3

def innerTail755_1 : List Nat := 6421 :: innerTail755_2
def innerCandidates755_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates755_1_checked : candidates edge 4 5779 6421 innerTail755_2 best.bound = innerCandidates755_1 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan755_1 : scan edge (search edge 3) 4 [5779] 5779 innerTail755_1 best = best := by
  rw [innerTail755_1, scan, if_neg (by decide), innerCandidates755_1_checked]
  change scan edge (search edge 3) 4 [5779] 5779 innerTail755_2
    (if 3 ≤ innerCandidates755_1.length then search edge 3 [5779, 6421] innerCandidates755_1 12200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan755_2

def innerTail755_0 : List Nat := 6007 :: innerTail755_1
def innerCandidates755_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates755_0_checked : candidates edge 4 5779 6007 innerTail755_1 best.bound = innerCandidates755_0 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan755_0 : scan edge (search edge 3) 4 [5779] 5779 innerTail755_0 best = best := by
  rw [innerTail755_0, scan, if_neg (by decide), innerCandidates755_0_checked]
  change scan edge (search edge 3) 4 [5779] 5779 innerTail755_1
    (if 3 ≤ innerCandidates755_0.length then search edge 3 [5779, 6007] innerCandidates755_0 11786 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan755_1

theorem rootBranch755 : search edge 4 [5779] rootCandidates755 5779 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5779] 5779 innerTail755_0 best = best
  exact innerScan755_0

def rootCandidates754 : List Nat := [5839, 6007, 6199, 6211, 6427, 7537, 7681, 7753, 8101]
set_option maxRecDepth 4096 in
theorem rootCandidates754_checked : candidates edge 5 0 5749 tail755 best.bound = rootCandidates754 := by
  unfold candidates
  have row : edge 5749 = SearchData.row5749.contains := by funext q; rfl
  rw [row]
  decide

def innerTail754_9 : List Nat := []
theorem innerScan754_9 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_9 best = best := rfl

def innerTail754_8 : List Nat := 8101 :: innerTail754_9
def innerCandidates754_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates754_8_checked : candidates edge 4 5749 8101 innerTail754_9 best.bound = innerCandidates754_8 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_8 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_8 best = best := by
  rw [innerTail754_8, scan, if_neg (by decide), innerCandidates754_8_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_9
    (if 3 ≤ innerCandidates754_8.length then search edge 3 [5749, 8101] innerCandidates754_8 13850 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_9

def innerTail754_7 : List Nat := 7753 :: innerTail754_8
def innerCandidates754_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates754_7_checked : candidates edge 4 5749 7753 innerTail754_8 best.bound = innerCandidates754_7 := by
  unfold candidates
  have row : edge 7753 = SearchData.row7753.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_7 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_7 best = best := by
  rw [innerTail754_7, scan, if_neg (by decide), innerCandidates754_7_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_8
    (if 3 ≤ innerCandidates754_7.length then search edge 3 [5749, 7753] innerCandidates754_7 13502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_8

def innerTail754_6 : List Nat := 7681 :: innerTail754_7
def innerCandidates754_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates754_6_checked : candidates edge 4 5749 7681 innerTail754_7 best.bound = innerCandidates754_6 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_6 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_6 best = best := by
  rw [innerTail754_6, scan, if_neg (by decide), innerCandidates754_6_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_7
    (if 3 ≤ innerCandidates754_6.length then search edge 3 [5749, 7681] innerCandidates754_6 13430 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_7

def innerTail754_5 : List Nat := 7537 :: innerTail754_6
def innerCandidates754_5 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates754_5_checked : candidates edge 4 5749 7537 innerTail754_6 best.bound = innerCandidates754_5 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_5 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_5 best = best := by
  rw [innerTail754_5, scan, if_neg (by decide), innerCandidates754_5_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_6
    (if 3 ≤ innerCandidates754_5.length then search edge 3 [5749, 7537] innerCandidates754_5 13286 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_6

def innerTail754_4 : List Nat := 6427 :: innerTail754_5
def innerCandidates754_4 : List Nat := [8101]
set_option maxRecDepth 4096 in
theorem innerCandidates754_4_checked : candidates edge 4 5749 6427 innerTail754_5 best.bound = innerCandidates754_4 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_4 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_4 best = best := by
  rw [innerTail754_4, scan, if_neg (by decide), innerCandidates754_4_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_5
    (if 3 ≤ innerCandidates754_4.length then search edge 3 [5749, 6427] innerCandidates754_4 12176 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_5

def innerTail754_3 : List Nat := 6211 :: innerTail754_4
def innerCandidates754_3 : List Nat := [7753]
set_option maxRecDepth 4096 in
theorem innerCandidates754_3_checked : candidates edge 4 5749 6211 innerTail754_4 best.bound = innerCandidates754_3 := by
  unfold candidates
  have row : edge 6211 = SearchData.row6211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_3 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_3 best = best := by
  rw [innerTail754_3, scan, if_neg (by decide), innerCandidates754_3_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_4
    (if 3 ≤ innerCandidates754_3.length then search edge 3 [5749, 6211] innerCandidates754_3 11960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_4

def innerTail754_2 : List Nat := 6199 :: innerTail754_3
def innerCandidates754_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates754_2_checked : candidates edge 4 5749 6199 innerTail754_3 best.bound = innerCandidates754_2 := by
  unfold candidates
  have row : edge 6199 = SearchData.row6199.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_2 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_2 best = best := by
  rw [innerTail754_2, scan, if_neg (by decide), innerCandidates754_2_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_3
    (if 3 ≤ innerCandidates754_2.length then search edge 3 [5749, 6199] innerCandidates754_2 11948 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_3

def innerTail754_1 : List Nat := 6007 :: innerTail754_2
def innerCandidates754_1 : List Nat := [7681]
set_option maxRecDepth 4096 in
theorem innerCandidates754_1_checked : candidates edge 4 5749 6007 innerTail754_2 best.bound = innerCandidates754_1 := by
  unfold candidates
  have row : edge 6007 = SearchData.row6007.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan754_1 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_1 best = best := by
  rw [innerTail754_1, scan, if_neg (by decide), innerCandidates754_1_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_2
    (if 3 ≤ innerCandidates754_1.length then search edge 3 [5749, 6007] innerCandidates754_1 11756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan754_2

def innerTail754_0 : List Nat := 5839 :: innerTail754_1
def innerCandidates754_0 : List Nat := [6199, 7681, 7753]
set_option maxRecDepth 4096 in
theorem innerCandidates754_0_checked : candidates edge 4 5749 5839 innerTail754_1 best.bound = innerCandidates754_0 := by
  unfold candidates
  have row : edge 5839 = SearchData.row5839.contains := by funext r; rfl
  rw [row]
  decide

set_option maxRecDepth 4096 in
theorem innerChild754_0 : search edge 3 [5749, 5839] innerCandidates754_0 11588 best = best := by decide

theorem innerScan754_0 : scan edge (search edge 3) 4 [5749] 5749 innerTail754_0 best = best := by
  rw [innerTail754_0, scan, if_neg (by decide), innerCandidates754_0_checked]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_1
    (if 3 ≤ innerCandidates754_0.length then search edge 3 [5749, 5839] innerCandidates754_0 11588 best else best) = best
  rw [if_pos (by decide), innerChild754_0]
  exact innerScan754_1

theorem rootBranch754 : search edge 4 [5749] rootCandidates754 5749 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5749] 5749 innerTail754_0 best = best
  exact innerScan754_0

def rootCandidates753 : List Nat := [5857, 6271, 7027, 7507, 7573, 7591, 7717, 7759, 8287, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates753_checked : candidates edge 5 0 5743 tail754 best.bound = rootCandidates753 := by
  unfold candidates
  have row : edge 5743 = SearchData.row5743.contains := by funext q; rfl
  rw [row]
  decide

def innerTail753_10 : List Nat := []
theorem innerScan753_10 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_10 best = best := rfl

def innerTail753_9 : List Nat := 8377 :: innerTail753_10
def innerCandidates753_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_9_checked : candidates edge 4 5743 8377 innerTail753_10 best.bound = innerCandidates753_9 := by
  unfold candidates
  have row : edge 8377 = SearchData.row8377.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_9 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_9 best = best := by
  rw [innerTail753_9, scan, if_neg (by decide), innerCandidates753_9_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_10
    (if 3 ≤ innerCandidates753_9.length then search edge 3 [5743, 8377] innerCandidates753_9 14120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_10

def innerTail753_8 : List Nat := 8287 :: innerTail753_9
def innerCandidates753_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_8_checked : candidates edge 4 5743 8287 innerTail753_9 best.bound = innerCandidates753_8 := by
  unfold candidates
  have row : edge 8287 = SearchData.row8287.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_8 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_8 best = best := by
  rw [innerTail753_8, scan, if_neg (by decide), innerCandidates753_8_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_9
    (if 3 ≤ innerCandidates753_8.length then search edge 3 [5743, 8287] innerCandidates753_8 14030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_9

def innerTail753_7 : List Nat := 7759 :: innerTail753_8
def innerCandidates753_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_7_checked : candidates edge 4 5743 7759 innerTail753_8 best.bound = innerCandidates753_7 := by
  unfold candidates
  have row : edge 7759 = SearchData.row7759.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_7 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_7 best = best := by
  rw [innerTail753_7, scan, if_neg (by decide), innerCandidates753_7_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_8
    (if 3 ≤ innerCandidates753_7.length then search edge 3 [5743, 7759] innerCandidates753_7 13502 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_8

def innerTail753_6 : List Nat := 7717 :: innerTail753_7
def innerCandidates753_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_6_checked : candidates edge 4 5743 7717 innerTail753_7 best.bound = innerCandidates753_6 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_6 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_6 best = best := by
  rw [innerTail753_6, scan, if_neg (by decide), innerCandidates753_6_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_7
    (if 3 ≤ innerCandidates753_6.length then search edge 3 [5743, 7717] innerCandidates753_6 13460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_7

def innerTail753_5 : List Nat := 7591 :: innerTail753_6
def innerCandidates753_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_5_checked : candidates edge 4 5743 7591 innerTail753_6 best.bound = innerCandidates753_5 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_5 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_5 best = best := by
  rw [innerTail753_5, scan, if_neg (by decide), innerCandidates753_5_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_6
    (if 3 ≤ innerCandidates753_5.length then search edge 3 [5743, 7591] innerCandidates753_5 13334 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_6

def innerTail753_4 : List Nat := 7573 :: innerTail753_5
def innerCandidates753_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_4_checked : candidates edge 4 5743 7573 innerTail753_5 best.bound = innerCandidates753_4 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_4 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_4 best = best := by
  rw [innerTail753_4, scan, if_neg (by decide), innerCandidates753_4_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_5
    (if 3 ≤ innerCandidates753_4.length then search edge 3 [5743, 7573] innerCandidates753_4 13316 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_5

def innerTail753_3 : List Nat := 7507 :: innerTail753_4
def innerCandidates753_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_3_checked : candidates edge 4 5743 7507 innerTail753_4 best.bound = innerCandidates753_3 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_3 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_3 best = best := by
  rw [innerTail753_3, scan, if_neg (by decide), innerCandidates753_3_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_4
    (if 3 ≤ innerCandidates753_3.length then search edge 3 [5743, 7507] innerCandidates753_3 13250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_4

def innerTail753_2 : List Nat := 7027 :: innerTail753_3
def innerCandidates753_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_2_checked : candidates edge 4 5743 7027 innerTail753_3 best.bound = innerCandidates753_2 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_2 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_2 best = best := by
  rw [innerTail753_2, scan, if_neg (by decide), innerCandidates753_2_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_3
    (if 3 ≤ innerCandidates753_2.length then search edge 3 [5743, 7027] innerCandidates753_2 12770 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_3

def innerTail753_1 : List Nat := 6271 :: innerTail753_2
def innerCandidates753_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_1_checked : candidates edge 4 5743 6271 innerTail753_2 best.bound = innerCandidates753_1 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_1 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_1 best = best := by
  rw [innerTail753_1, scan, if_neg (by decide), innerCandidates753_1_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_2
    (if 3 ≤ innerCandidates753_1.length then search edge 3 [5743, 6271] innerCandidates753_1 12014 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_2

def innerTail753_0 : List Nat := 5857 :: innerTail753_1
def innerCandidates753_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates753_0_checked : candidates edge 4 5743 5857 innerTail753_1 best.bound = innerCandidates753_0 := by
  unfold candidates
  have row : edge 5857 = SearchData.row5857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan753_0 : scan edge (search edge 3) 4 [5743] 5743 innerTail753_0 best = best := by
  rw [innerTail753_0, scan, if_neg (by decide), innerCandidates753_0_checked]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_1
    (if 3 ≤ innerCandidates753_0.length then search edge 3 [5743, 5857] innerCandidates753_0 11600 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan753_1

theorem rootBranch753 : search edge 4 [5743] rootCandidates753 5743 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5743] 5743 innerTail753_0 best = best
  exact innerScan753_0

def rootCandidates752 : List Nat := [6299, 6833, 6959, 7541, 8093, 8219]
set_option maxRecDepth 4096 in
theorem rootCandidates752_checked : candidates edge 5 0 5741 tail753 best.bound = rootCandidates752 := by
  unfold candidates
  have row : edge 5741 = SearchData.row5741.contains := by funext q; rfl
  rw [row]
  decide

def innerTail752_6 : List Nat := []
theorem innerScan752_6 : scan edge (search edge 3) 4 [5741] 5741 innerTail752_6 best = best := rfl

def innerTail752_5 : List Nat := 8219 :: innerTail752_6
def innerCandidates752_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates752_5_checked : candidates edge 4 5741 8219 innerTail752_6 best.bound = innerCandidates752_5 := by
  unfold candidates
  have row : edge 8219 = SearchData.row8219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan752_5 : scan edge (search edge 3) 4 [5741] 5741 innerTail752_5 best = best := by
  rw [innerTail752_5, scan, if_neg (by decide), innerCandidates752_5_checked]
  change scan edge (search edge 3) 4 [5741] 5741 innerTail752_6
    (if 3 ≤ innerCandidates752_5.length then search edge 3 [5741, 8219] innerCandidates752_5 13960 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan752_6

def innerTail752_4 : List Nat := 8093 :: innerTail752_5
def innerCandidates752_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates752_4_checked : candidates edge 4 5741 8093 innerTail752_5 best.bound = innerCandidates752_4 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan752_4 : scan edge (search edge 3) 4 [5741] 5741 innerTail752_4 best = best := by
  rw [innerTail752_4, scan, if_neg (by decide), innerCandidates752_4_checked]
  change scan edge (search edge 3) 4 [5741] 5741 innerTail752_5
    (if 3 ≤ innerCandidates752_4.length then search edge 3 [5741, 8093] innerCandidates752_4 13834 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan752_5

def innerTail752_3 : List Nat := 7541 :: innerTail752_4
def innerCandidates752_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates752_3_checked : candidates edge 4 5741 7541 innerTail752_4 best.bound = innerCandidates752_3 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan752_3 : scan edge (search edge 3) 4 [5741] 5741 innerTail752_3 best = best := by
  rw [innerTail752_3, scan, if_neg (by decide), innerCandidates752_3_checked]
  change scan edge (search edge 3) 4 [5741] 5741 innerTail752_4
    (if 3 ≤ innerCandidates752_3.length then search edge 3 [5741, 7541] innerCandidates752_3 13282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan752_4

def innerTail752_2 : List Nat := 6959 :: innerTail752_3
def innerCandidates752_2 : List Nat := [8093]
set_option maxRecDepth 4096 in
theorem innerCandidates752_2_checked : candidates edge 4 5741 6959 innerTail752_3 best.bound = innerCandidates752_2 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan752_2 : scan edge (search edge 3) 4 [5741] 5741 innerTail752_2 best = best := by
  rw [innerTail752_2, scan, if_neg (by decide), innerCandidates752_2_checked]
  change scan edge (search edge 3) 4 [5741] 5741 innerTail752_3
    (if 3 ≤ innerCandidates752_2.length then search edge 3 [5741, 6959] innerCandidates752_2 12700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan752_3

def innerTail752_1 : List Nat := 6833 :: innerTail752_2
def innerCandidates752_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates752_1_checked : candidates edge 4 5741 6833 innerTail752_2 best.bound = innerCandidates752_1 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan752_1 : scan edge (search edge 3) 4 [5741] 5741 innerTail752_1 best = best := by
  rw [innerTail752_1, scan, if_neg (by decide), innerCandidates752_1_checked]
  change scan edge (search edge 3) 4 [5741] 5741 innerTail752_2
    (if 3 ≤ innerCandidates752_1.length then search edge 3 [5741, 6833] innerCandidates752_1 12574 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan752_2

def innerTail752_0 : List Nat := 6299 :: innerTail752_1
def innerCandidates752_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates752_0_checked : candidates edge 4 5741 6299 innerTail752_1 best.bound = innerCandidates752_0 := by
  unfold candidates
  have row : edge 6299 = SearchData.row6299.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan752_0 : scan edge (search edge 3) 4 [5741] 5741 innerTail752_0 best = best := by
  rw [innerTail752_0, scan, if_neg (by decide), innerCandidates752_0_checked]
  change scan edge (search edge 3) 4 [5741] 5741 innerTail752_1
    (if 3 ≤ innerCandidates752_0.length then search edge 3 [5741, 6299] innerCandidates752_0 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan752_1

theorem rootBranch752 : search edge 4 [5741] rootCandidates752 5741 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5741] 5741 innerTail752_0 best = best
  exact innerScan752_0

end ProjectEulerSolutions.PrimePairUpper.Search
