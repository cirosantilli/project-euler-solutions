import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates847 : List Nat := [6967, 7129, 7549, 7621, 8269, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates847_checked : candidates edge 5 0 6571 tail848 best.bound = rootCandidates847 := by
  unfold candidates
  have row : edge 6571 = SearchData.row6571.contains := by funext q; rfl
  rw [row]
  decide

def innerTail847_6 : List Nat := []
theorem innerScan847_6 : scan edge (search edge 3) 4 [6571] 6571 innerTail847_6 best = best := rfl

def innerTail847_5 : List Nat := 8293 :: innerTail847_6
def innerCandidates847_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates847_5_checked : candidates edge 4 6571 8293 innerTail847_6 best.bound = innerCandidates847_5 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan847_5 : scan edge (search edge 3) 4 [6571] 6571 innerTail847_5 best = best := by
  rw [innerTail847_5, scan, if_neg (by decide), innerCandidates847_5_checked]
  change scan edge (search edge 3) 4 [6571] 6571 innerTail847_6
    (if 3 ≤ innerCandidates847_5.length then search edge 3 [6571, 8293] innerCandidates847_5 14864 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan847_6

def innerTail847_4 : List Nat := 8269 :: innerTail847_5
def innerCandidates847_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates847_4_checked : candidates edge 4 6571 8269 innerTail847_5 best.bound = innerCandidates847_4 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan847_4 : scan edge (search edge 3) 4 [6571] 6571 innerTail847_4 best = best := by
  rw [innerTail847_4, scan, if_neg (by decide), innerCandidates847_4_checked]
  change scan edge (search edge 3) 4 [6571] 6571 innerTail847_5
    (if 3 ≤ innerCandidates847_4.length then search edge 3 [6571, 8269] innerCandidates847_4 14840 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan847_5

def innerTail847_3 : List Nat := 7621 :: innerTail847_4
def innerCandidates847_3 : List Nat := [8269]
set_option maxRecDepth 4096 in
theorem innerCandidates847_3_checked : candidates edge 4 6571 7621 innerTail847_4 best.bound = innerCandidates847_3 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan847_3 : scan edge (search edge 3) 4 [6571] 6571 innerTail847_3 best = best := by
  rw [innerTail847_3, scan, if_neg (by decide), innerCandidates847_3_checked]
  change scan edge (search edge 3) 4 [6571] 6571 innerTail847_4
    (if 3 ≤ innerCandidates847_3.length then search edge 3 [6571, 7621] innerCandidates847_3 14192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan847_4

def innerTail847_2 : List Nat := 7549 :: innerTail847_3
def innerCandidates847_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates847_2_checked : candidates edge 4 6571 7549 innerTail847_3 best.bound = innerCandidates847_2 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan847_2 : scan edge (search edge 3) 4 [6571] 6571 innerTail847_2 best = best := by
  rw [innerTail847_2, scan, if_neg (by decide), innerCandidates847_2_checked]
  change scan edge (search edge 3) 4 [6571] 6571 innerTail847_3
    (if 3 ≤ innerCandidates847_2.length then search edge 3 [6571, 7549] innerCandidates847_2 14120 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan847_3

def innerTail847_1 : List Nat := 7129 :: innerTail847_2
def innerCandidates847_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates847_1_checked : candidates edge 4 6571 7129 innerTail847_2 best.bound = innerCandidates847_1 := by
  unfold candidates
  have row : edge 7129 = SearchData.row7129.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan847_1 : scan edge (search edge 3) 4 [6571] 6571 innerTail847_1 best = best := by
  rw [innerTail847_1, scan, if_neg (by decide), innerCandidates847_1_checked]
  change scan edge (search edge 3) 4 [6571] 6571 innerTail847_2
    (if 3 ≤ innerCandidates847_1.length then search edge 3 [6571, 7129] innerCandidates847_1 13700 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan847_2

def innerTail847_0 : List Nat := 6967 :: innerTail847_1
def innerCandidates847_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates847_0_checked : candidates edge 4 6571 6967 innerTail847_1 best.bound = innerCandidates847_0 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan847_0 : scan edge (search edge 3) 4 [6571] 6571 innerTail847_0 best = best := by
  rw [innerTail847_0, scan, if_neg (by decide), innerCandidates847_0_checked]
  change scan edge (search edge 3) 4 [6571] 6571 innerTail847_1
    (if 3 ≤ innerCandidates847_0.length then search edge 3 [6571, 6967] innerCandidates847_0 13538 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan847_1

theorem rootBranch847 : search edge 4 [6571] rootCandidates847 6571 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6571] 6571 innerTail847_0 best = best
  exact innerScan847_0

def rootCandidates846 : List Nat := [6689, 6779, 7109]
set_option maxRecDepth 4096 in
theorem rootCandidates846_checked : candidates edge 5 0 6569 tail847 best.bound = rootCandidates846 := by
  unfold candidates
  have row : edge 6569 = SearchData.row6569.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates845 : List Nat := [6569]
set_option maxRecDepth 4096 in
theorem rootCandidates845_checked : candidates edge 5 0 6563 tail846 best.bound = rootCandidates845 := by
  unfold candidates
  have row : edge 6563 = SearchData.row6563.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates844 : List Nat := [8329]
set_option maxRecDepth 4096 in
theorem rootCandidates844_checked : candidates edge 5 0 6553 tail845 best.bound = rootCandidates844 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates843 : List Nat := [6653, 7253, 7481, 8093]
set_option maxRecDepth 4096 in
theorem rootCandidates843_checked : candidates edge 5 0 6551 tail844 best.bound = rootCandidates843 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext q; rfl
  rw [row]
  decide

def innerTail843_4 : List Nat := []
theorem innerScan843_4 : scan edge (search edge 3) 4 [6551] 6551 innerTail843_4 best = best := rfl

def innerTail843_3 : List Nat := 8093 :: innerTail843_4
def innerCandidates843_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates843_3_checked : candidates edge 4 6551 8093 innerTail843_4 best.bound = innerCandidates843_3 := by
  unfold candidates
  have row : edge 8093 = SearchData.row8093.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan843_3 : scan edge (search edge 3) 4 [6551] 6551 innerTail843_3 best = best := by
  rw [innerTail843_3, scan, if_neg (by decide), innerCandidates843_3_checked]
  change scan edge (search edge 3) 4 [6551] 6551 innerTail843_4
    (if 3 ≤ innerCandidates843_3.length then search edge 3 [6551, 8093] innerCandidates843_3 14644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan843_4

def innerTail843_2 : List Nat := 7481 :: innerTail843_3
def innerCandidates843_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates843_2_checked : candidates edge 4 6551 7481 innerTail843_3 best.bound = innerCandidates843_2 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan843_2 : scan edge (search edge 3) 4 [6551] 6551 innerTail843_2 best = best := by
  rw [innerTail843_2, scan, if_neg (by decide), innerCandidates843_2_checked]
  change scan edge (search edge 3) 4 [6551] 6551 innerTail843_3
    (if 3 ≤ innerCandidates843_2.length then search edge 3 [6551, 7481] innerCandidates843_2 14032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan843_3

def innerTail843_1 : List Nat := 7253 :: innerTail843_2
def innerCandidates843_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates843_1_checked : candidates edge 4 6551 7253 innerTail843_2 best.bound = innerCandidates843_1 := by
  unfold candidates
  have row : edge 7253 = SearchData.row7253.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan843_1 : scan edge (search edge 3) 4 [6551] 6551 innerTail843_1 best = best := by
  rw [innerTail843_1, scan, if_neg (by decide), innerCandidates843_1_checked]
  change scan edge (search edge 3) 4 [6551] 6551 innerTail843_2
    (if 3 ≤ innerCandidates843_1.length then search edge 3 [6551, 7253] innerCandidates843_1 13804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan843_2

def innerTail843_0 : List Nat := 6653 :: innerTail843_1
def innerCandidates843_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates843_0_checked : candidates edge 4 6551 6653 innerTail843_1 best.bound = innerCandidates843_0 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan843_0 : scan edge (search edge 3) 4 [6551] 6551 innerTail843_0 best = best := by
  rw [innerTail843_0, scan, if_neg (by decide), innerCandidates843_0_checked]
  change scan edge (search edge 3) 4 [6551] 6551 innerTail843_1
    (if 3 ≤ innerCandidates843_0.length then search edge 3 [6551, 6653] innerCandidates843_0 13204 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan843_1

theorem rootBranch843 : search edge 4 [6551] rootCandidates843 6551 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6551] 6551 innerTail843_0 best = best
  exact innerScan843_0

def rootCandidates842 : List Nat := [6637, 6703, 6841, 6871, 6967, 7561, 7681, 7723]
set_option maxRecDepth 4096 in
theorem rootCandidates842_checked : candidates edge 5 0 6547 tail843 best.bound = rootCandidates842 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext q; rfl
  rw [row]
  decide

def innerTail842_8 : List Nat := []
theorem innerScan842_8 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_8 best = best := rfl

def innerTail842_7 : List Nat := 7723 :: innerTail842_8
def innerCandidates842_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates842_7_checked : candidates edge 4 6547 7723 innerTail842_8 best.bound = innerCandidates842_7 := by
  unfold candidates
  have row : edge 7723 = SearchData.row7723.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_7 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_7 best = best := by
  rw [innerTail842_7, scan, if_neg (by decide), innerCandidates842_7_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_8
    (if 3 ≤ innerCandidates842_7.length then search edge 3 [6547, 7723] innerCandidates842_7 14270 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_8

def innerTail842_6 : List Nat := 7681 :: innerTail842_7
def innerCandidates842_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates842_6_checked : candidates edge 4 6547 7681 innerTail842_7 best.bound = innerCandidates842_6 := by
  unfold candidates
  have row : edge 7681 = SearchData.row7681.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_6 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_6 best = best := by
  rw [innerTail842_6, scan, if_neg (by decide), innerCandidates842_6_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_7
    (if 3 ≤ innerCandidates842_6.length then search edge 3 [6547, 7681] innerCandidates842_6 14228 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_7

def innerTail842_5 : List Nat := 7561 :: innerTail842_6
def innerCandidates842_5 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates842_5_checked : candidates edge 4 6547 7561 innerTail842_6 best.bound = innerCandidates842_5 := by
  unfold candidates
  have row : edge 7561 = SearchData.row7561.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_5 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_5 best = best := by
  rw [innerTail842_5, scan, if_neg (by decide), innerCandidates842_5_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_6
    (if 3 ≤ innerCandidates842_5.length then search edge 3 [6547, 7561] innerCandidates842_5 14108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_6

def innerTail842_4 : List Nat := 6967 :: innerTail842_5
def innerCandidates842_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates842_4_checked : candidates edge 4 6547 6967 innerTail842_5 best.bound = innerCandidates842_4 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_4 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_4 best = best := by
  rw [innerTail842_4, scan, if_neg (by decide), innerCandidates842_4_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_5
    (if 3 ≤ innerCandidates842_4.length then search edge 3 [6547, 6967] innerCandidates842_4 13514 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_5

def innerTail842_3 : List Nat := 6871 :: innerTail842_4
def innerCandidates842_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates842_3_checked : candidates edge 4 6547 6871 innerTail842_4 best.bound = innerCandidates842_3 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_3 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_3 best = best := by
  rw [innerTail842_3, scan, if_neg (by decide), innerCandidates842_3_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_4
    (if 3 ≤ innerCandidates842_3.length then search edge 3 [6547, 6871] innerCandidates842_3 13418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_4

def innerTail842_2 : List Nat := 6841 :: innerTail842_3
def innerCandidates842_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates842_2_checked : candidates edge 4 6547 6841 innerTail842_3 best.bound = innerCandidates842_2 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_2 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_2 best = best := by
  rw [innerTail842_2, scan, if_neg (by decide), innerCandidates842_2_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_3
    (if 3 ≤ innerCandidates842_2.length then search edge 3 [6547, 6841] innerCandidates842_2 13388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_3

def innerTail842_1 : List Nat := 6703 :: innerTail842_2
def innerCandidates842_1 : List Nat := [7723]
set_option maxRecDepth 4096 in
theorem innerCandidates842_1_checked : candidates edge 4 6547 6703 innerTail842_2 best.bound = innerCandidates842_1 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_1 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_1 best = best := by
  rw [innerTail842_1, scan, if_neg (by decide), innerCandidates842_1_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_2
    (if 3 ≤ innerCandidates842_1.length then search edge 3 [6547, 6703] innerCandidates842_1 13250 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_2

def innerTail842_0 : List Nat := 6637 :: innerTail842_1
def innerCandidates842_0 : List Nat := [6841]
set_option maxRecDepth 4096 in
theorem innerCandidates842_0_checked : candidates edge 4 6547 6637 innerTail842_1 best.bound = innerCandidates842_0 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan842_0 : scan edge (search edge 3) 4 [6547] 6547 innerTail842_0 best = best := by
  rw [innerTail842_0, scan, if_neg (by decide), innerCandidates842_0_checked]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_1
    (if 3 ≤ innerCandidates842_0.length then search edge 3 [6547, 6637] innerCandidates842_0 13184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan842_1

theorem rootBranch842 : search edge 4 [6547] rootCandidates842 6547 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6547] 6547 innerTail842_0 best = best
  exact innerScan842_0

def rootCandidates841 : List Nat := [6547, 6961, 7027, 7549, 8221, 8329]
set_option maxRecDepth 4096 in
theorem rootCandidates841_checked : candidates edge 5 0 6529 tail842 best.bound = rootCandidates841 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext q; rfl
  rw [row]
  decide

def innerTail841_6 : List Nat := []
theorem innerScan841_6 : scan edge (search edge 3) 4 [6529] 6529 innerTail841_6 best = best := rfl

def innerTail841_5 : List Nat := 8329 :: innerTail841_6
def innerCandidates841_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates841_5_checked : candidates edge 4 6529 8329 innerTail841_6 best.bound = innerCandidates841_5 := by
  unfold candidates
  have row : edge 8329 = SearchData.row8329.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan841_5 : scan edge (search edge 3) 4 [6529] 6529 innerTail841_5 best = best := by
  rw [innerTail841_5, scan, if_neg (by decide), innerCandidates841_5_checked]
  change scan edge (search edge 3) 4 [6529] 6529 innerTail841_6
    (if 3 ≤ innerCandidates841_5.length then search edge 3 [6529, 8329] innerCandidates841_5 14858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan841_6

def innerTail841_4 : List Nat := 8221 :: innerTail841_5
def innerCandidates841_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates841_4_checked : candidates edge 4 6529 8221 innerTail841_5 best.bound = innerCandidates841_4 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan841_4 : scan edge (search edge 3) 4 [6529] 6529 innerTail841_4 best = best := by
  rw [innerTail841_4, scan, if_neg (by decide), innerCandidates841_4_checked]
  change scan edge (search edge 3) 4 [6529] 6529 innerTail841_5
    (if 3 ≤ innerCandidates841_4.length then search edge 3 [6529, 8221] innerCandidates841_4 14750 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan841_5

def innerTail841_3 : List Nat := 7549 :: innerTail841_4
def innerCandidates841_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates841_3_checked : candidates edge 4 6529 7549 innerTail841_4 best.bound = innerCandidates841_3 := by
  unfold candidates
  have row : edge 7549 = SearchData.row7549.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan841_3 : scan edge (search edge 3) 4 [6529] 6529 innerTail841_3 best = best := by
  rw [innerTail841_3, scan, if_neg (by decide), innerCandidates841_3_checked]
  change scan edge (search edge 3) 4 [6529] 6529 innerTail841_4
    (if 3 ≤ innerCandidates841_3.length then search edge 3 [6529, 7549] innerCandidates841_3 14078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan841_4

def innerTail841_2 : List Nat := 7027 :: innerTail841_3
def innerCandidates841_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates841_2_checked : candidates edge 4 6529 7027 innerTail841_3 best.bound = innerCandidates841_2 := by
  unfold candidates
  have row : edge 7027 = SearchData.row7027.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan841_2 : scan edge (search edge 3) 4 [6529] 6529 innerTail841_2 best = best := by
  rw [innerTail841_2, scan, if_neg (by decide), innerCandidates841_2_checked]
  change scan edge (search edge 3) 4 [6529] 6529 innerTail841_3
    (if 3 ≤ innerCandidates841_2.length then search edge 3 [6529, 7027] innerCandidates841_2 13556 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan841_3

def innerTail841_1 : List Nat := 6961 :: innerTail841_2
def innerCandidates841_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates841_1_checked : candidates edge 4 6529 6961 innerTail841_2 best.bound = innerCandidates841_1 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan841_1 : scan edge (search edge 3) 4 [6529] 6529 innerTail841_1 best = best := by
  rw [innerTail841_1, scan, if_neg (by decide), innerCandidates841_1_checked]
  change scan edge (search edge 3) 4 [6529] 6529 innerTail841_2
    (if 3 ≤ innerCandidates841_1.length then search edge 3 [6529, 6961] innerCandidates841_1 13490 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan841_2

def innerTail841_0 : List Nat := 6547 :: innerTail841_1
def innerCandidates841_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates841_0_checked : candidates edge 4 6529 6547 innerTail841_1 best.bound = innerCandidates841_0 := by
  unfold candidates
  have row : edge 6547 = SearchData.row6547.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan841_0 : scan edge (search edge 3) 4 [6529] 6529 innerTail841_0 best = best := by
  rw [innerTail841_0, scan, if_neg (by decide), innerCandidates841_0_checked]
  change scan edge (search edge 3) 4 [6529] 6529 innerTail841_1
    (if 3 ≤ innerCandidates841_0.length then search edge 3 [6529, 6547] innerCandidates841_0 13076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan841_1

theorem rootBranch841 : search edge 4 [6529] rootCandidates841 6529 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6529] 6529 innerTail841_0 best = best
  exact innerScan841_0

def rootCandidates840 : List Nat := [7043, 7589, 8039, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates840_checked : candidates edge 5 0 6521 tail841 best.bound = rootCandidates840 := by
  unfold candidates
  have row : edge 6521 = SearchData.row6521.contains := by funext q; rfl
  rw [row]
  decide

def innerTail840_4 : List Nat := []
theorem innerScan840_4 : scan edge (search edge 3) 4 [6521] 6521 innerTail840_4 best = best := rfl

def innerTail840_3 : List Nat := 8369 :: innerTail840_4
def innerCandidates840_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates840_3_checked : candidates edge 4 6521 8369 innerTail840_4 best.bound = innerCandidates840_3 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan840_3 : scan edge (search edge 3) 4 [6521] 6521 innerTail840_3 best = best := by
  rw [innerTail840_3, scan, if_neg (by decide), innerCandidates840_3_checked]
  change scan edge (search edge 3) 4 [6521] 6521 innerTail840_4
    (if 3 ≤ innerCandidates840_3.length then search edge 3 [6521, 8369] innerCandidates840_3 14890 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan840_4

def innerTail840_2 : List Nat := 8039 :: innerTail840_3
def innerCandidates840_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates840_2_checked : candidates edge 4 6521 8039 innerTail840_3 best.bound = innerCandidates840_2 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan840_2 : scan edge (search edge 3) 4 [6521] 6521 innerTail840_2 best = best := by
  rw [innerTail840_2, scan, if_neg (by decide), innerCandidates840_2_checked]
  change scan edge (search edge 3) 4 [6521] 6521 innerTail840_3
    (if 3 ≤ innerCandidates840_2.length then search edge 3 [6521, 8039] innerCandidates840_2 14560 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan840_3

def innerTail840_1 : List Nat := 7589 :: innerTail840_2
def innerCandidates840_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates840_1_checked : candidates edge 4 6521 7589 innerTail840_2 best.bound = innerCandidates840_1 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan840_1 : scan edge (search edge 3) 4 [6521] 6521 innerTail840_1 best = best := by
  rw [innerTail840_1, scan, if_neg (by decide), innerCandidates840_1_checked]
  change scan edge (search edge 3) 4 [6521] 6521 innerTail840_2
    (if 3 ≤ innerCandidates840_1.length then search edge 3 [6521, 7589] innerCandidates840_1 14110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan840_2

def innerTail840_0 : List Nat := 7043 :: innerTail840_1
def innerCandidates840_0 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates840_0_checked : candidates edge 4 6521 7043 innerTail840_1 best.bound = innerCandidates840_0 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan840_0 : scan edge (search edge 3) 4 [6521] 6521 innerTail840_0 best = best := by
  rw [innerTail840_0, scan, if_neg (by decide), innerCandidates840_0_checked]
  change scan edge (search edge 3) 4 [6521] 6521 innerTail840_1
    (if 3 ≤ innerCandidates840_0.length then search edge 3 [6521, 7043] innerCandidates840_0 13564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan840_1

theorem rootBranch840 : search edge 4 [6521] rootCandidates840 6521 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6521] 6521 innerTail840_0 best = best
  exact innerScan840_0

end ProjectEulerSolutions.PrimePairUpper.Search
