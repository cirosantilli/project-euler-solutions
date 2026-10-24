import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates652 : List Nat := [4933, 5659, 5683, 6271, 6991, 7789]
set_option maxRecDepth 4096 in
theorem rootCandidates652_checked : candidates edge 5 0 4903 tail653 best.bound = rootCandidates652 := by
  unfold candidates
  have row : edge 4903 = SearchData.row4903.contains := by funext q; rfl
  rw [row]
  decide

def innerTail652_6 : List Nat := []
theorem innerScan652_6 : scan edge (search edge 3) 4 [4903] 4903 innerTail652_6 best = best := rfl

def innerTail652_5 : List Nat := 7789 :: innerTail652_6
def innerCandidates652_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates652_5_checked : candidates edge 4 4903 7789 innerTail652_6 best.bound = innerCandidates652_5 := by
  unfold candidates
  have row : edge 7789 = SearchData.row7789.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan652_5 : scan edge (search edge 3) 4 [4903] 4903 innerTail652_5 best = best := by
  rw [innerTail652_5, scan, if_neg (by decide), innerCandidates652_5_checked]
  change scan edge (search edge 3) 4 [4903] 4903 innerTail652_6
    (if 3 ≤ innerCandidates652_5.length then search edge 3 [4903, 7789] innerCandidates652_5 12692 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan652_6

def innerTail652_4 : List Nat := 6991 :: innerTail652_5
def innerCandidates652_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates652_4_checked : candidates edge 4 4903 6991 innerTail652_5 best.bound = innerCandidates652_4 := by
  unfold candidates
  have row : edge 6991 = SearchData.row6991.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan652_4 : scan edge (search edge 3) 4 [4903] 4903 innerTail652_4 best = best := by
  rw [innerTail652_4, scan, if_neg (by decide), innerCandidates652_4_checked]
  change scan edge (search edge 3) 4 [4903] 4903 innerTail652_5
    (if 3 ≤ innerCandidates652_4.length then search edge 3 [4903, 6991] innerCandidates652_4 11894 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan652_5

def innerTail652_3 : List Nat := 6271 :: innerTail652_4
def innerCandidates652_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates652_3_checked : candidates edge 4 4903 6271 innerTail652_4 best.bound = innerCandidates652_3 := by
  unfold candidates
  have row : edge 6271 = SearchData.row6271.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan652_3 : scan edge (search edge 3) 4 [4903] 4903 innerTail652_3 best = best := by
  rw [innerTail652_3, scan, if_neg (by decide), innerCandidates652_3_checked]
  change scan edge (search edge 3) 4 [4903] 4903 innerTail652_4
    (if 3 ≤ innerCandidates652_3.length then search edge 3 [4903, 6271] innerCandidates652_3 11174 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan652_4

def innerTail652_2 : List Nat := 5683 :: innerTail652_3
def innerCandidates652_2 : List Nat := [6271]
set_option maxRecDepth 4096 in
theorem innerCandidates652_2_checked : candidates edge 4 4903 5683 innerTail652_3 best.bound = innerCandidates652_2 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan652_2 : scan edge (search edge 3) 4 [4903] 4903 innerTail652_2 best = best := by
  rw [innerTail652_2, scan, if_neg (by decide), innerCandidates652_2_checked]
  change scan edge (search edge 3) 4 [4903] 4903 innerTail652_3
    (if 3 ≤ innerCandidates652_2.length then search edge 3 [4903, 5683] innerCandidates652_2 10586 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan652_3

def innerTail652_1 : List Nat := 5659 :: innerTail652_2
def innerCandidates652_1 : List Nat := [6271]
set_option maxRecDepth 4096 in
theorem innerCandidates652_1_checked : candidates edge 4 4903 5659 innerTail652_2 best.bound = innerCandidates652_1 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan652_1 : scan edge (search edge 3) 4 [4903] 4903 innerTail652_1 best = best := by
  rw [innerTail652_1, scan, if_neg (by decide), innerCandidates652_1_checked]
  change scan edge (search edge 3) 4 [4903] 4903 innerTail652_2
    (if 3 ≤ innerCandidates652_1.length then search edge 3 [4903, 5659] innerCandidates652_1 10562 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan652_2

def innerTail652_0 : List Nat := 4933 :: innerTail652_1
def innerCandidates652_0 : List Nat := [6271]
set_option maxRecDepth 4096 in
theorem innerCandidates652_0_checked : candidates edge 4 4903 4933 innerTail652_1 best.bound = innerCandidates652_0 := by
  unfold candidates
  have row : edge 4933 = SearchData.row4933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan652_0 : scan edge (search edge 3) 4 [4903] 4903 innerTail652_0 best = best := by
  rw [innerTail652_0, scan, if_neg (by decide), innerCandidates652_0_checked]
  change scan edge (search edge 3) 4 [4903] 4903 innerTail652_1
    (if 3 ≤ innerCandidates652_0.length then search edge 3 [4903, 4933] innerCandidates652_0 9836 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan652_1

theorem rootBranch652 : search edge 4 [4903] rootCandidates652 4903 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4903] 4903 innerTail652_0 best = best
  exact innerScan652_0

def rootCandidates651 : List Nat := [5333, 5669, 6053, 6311, 7187, 7481, 7643, 8123, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates651_checked : candidates edge 5 0 4889 tail652 best.bound = rootCandidates651 := by
  unfold candidates
  have row : edge 4889 = SearchData.row4889.contains := by funext q; rfl
  rw [row]
  decide

def innerTail651_9 : List Nat := []
theorem innerScan651_9 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_9 best = best := rfl

def innerTail651_8 : List Nat := 8237 :: innerTail651_9
def innerCandidates651_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_8_checked : candidates edge 4 4889 8237 innerTail651_9 best.bound = innerCandidates651_8 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_8 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_8 best = best := by
  rw [innerTail651_8, scan, if_neg (by decide), innerCandidates651_8_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_9
    (if 3 ≤ innerCandidates651_8.length then search edge 3 [4889, 8237] innerCandidates651_8 13126 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_9

def innerTail651_7 : List Nat := 8123 :: innerTail651_8
def innerCandidates651_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_7_checked : candidates edge 4 4889 8123 innerTail651_8 best.bound = innerCandidates651_7 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_7 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_7 best = best := by
  rw [innerTail651_7, scan, if_neg (by decide), innerCandidates651_7_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_8
    (if 3 ≤ innerCandidates651_7.length then search edge 3 [4889, 8123] innerCandidates651_7 13012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_8

def innerTail651_6 : List Nat := 7643 :: innerTail651_7
def innerCandidates651_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_6_checked : candidates edge 4 4889 7643 innerTail651_7 best.bound = innerCandidates651_6 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_6 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_6 best = best := by
  rw [innerTail651_6, scan, if_neg (by decide), innerCandidates651_6_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_7
    (if 3 ≤ innerCandidates651_6.length then search edge 3 [4889, 7643] innerCandidates651_6 12532 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_7

def innerTail651_5 : List Nat := 7481 :: innerTail651_6
def innerCandidates651_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_5_checked : candidates edge 4 4889 7481 innerTail651_6 best.bound = innerCandidates651_5 := by
  unfold candidates
  have row : edge 7481 = SearchData.row7481.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_5 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_5 best = best := by
  rw [innerTail651_5, scan, if_neg (by decide), innerCandidates651_5_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_6
    (if 3 ≤ innerCandidates651_5.length then search edge 3 [4889, 7481] innerCandidates651_5 12370 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_6

def innerTail651_4 : List Nat := 7187 :: innerTail651_5
def innerCandidates651_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_4_checked : candidates edge 4 4889 7187 innerTail651_5 best.bound = innerCandidates651_4 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_4 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_4 best = best := by
  rw [innerTail651_4, scan, if_neg (by decide), innerCandidates651_4_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_5
    (if 3 ≤ innerCandidates651_4.length then search edge 3 [4889, 7187] innerCandidates651_4 12076 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_5

def innerTail651_3 : List Nat := 6311 :: innerTail651_4
def innerCandidates651_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_3_checked : candidates edge 4 4889 6311 innerTail651_4 best.bound = innerCandidates651_3 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_3 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_3 best = best := by
  rw [innerTail651_3, scan, if_neg (by decide), innerCandidates651_3_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_4
    (if 3 ≤ innerCandidates651_3.length then search edge 3 [4889, 6311] innerCandidates651_3 11200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_4

def innerTail651_2 : List Nat := 6053 :: innerTail651_3
def innerCandidates651_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_2_checked : candidates edge 4 4889 6053 innerTail651_3 best.bound = innerCandidates651_2 := by
  unfold candidates
  have row : edge 6053 = SearchData.row6053.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_2 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_2 best = best := by
  rw [innerTail651_2, scan, if_neg (by decide), innerCandidates651_2_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_3
    (if 3 ≤ innerCandidates651_2.length then search edge 3 [4889, 6053] innerCandidates651_2 10942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_3

def innerTail651_1 : List Nat := 5669 :: innerTail651_2
def innerCandidates651_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_1_checked : candidates edge 4 4889 5669 innerTail651_2 best.bound = innerCandidates651_1 := by
  unfold candidates
  have row : edge 5669 = SearchData.row5669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_1 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_1 best = best := by
  rw [innerTail651_1, scan, if_neg (by decide), innerCandidates651_1_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_2
    (if 3 ≤ innerCandidates651_1.length then search edge 3 [4889, 5669] innerCandidates651_1 10558 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_2

def innerTail651_0 : List Nat := 5333 :: innerTail651_1
def innerCandidates651_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates651_0_checked : candidates edge 4 4889 5333 innerTail651_1 best.bound = innerCandidates651_0 := by
  unfold candidates
  have row : edge 5333 = SearchData.row5333.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan651_0 : scan edge (search edge 3) 4 [4889] 4889 innerTail651_0 best = best := by
  rw [innerTail651_0, scan, if_neg (by decide), innerCandidates651_0_checked]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_1
    (if 3 ≤ innerCandidates651_0.length then search edge 3 [4889, 5333] innerCandidates651_0 10222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan651_1

theorem rootBranch651 : search edge 4 [4889] rootCandidates651 4889 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4889] 4889 innerTail651_0 best = best
  exact innerScan651_0

def rootCandidates650 : List Nat := [5171, 5501, 5693, 5711, 5813, 5981, 6257, 6359, 6551, 6719, 6947, 6977, 7307, 7349, 7559, 7643, 8123, 8231, 8291]
set_option maxRecDepth 4096 in
theorem rootCandidates650_checked : candidates edge 5 0 4877 tail651 best.bound = rootCandidates650 := by
  unfold candidates
  have row : edge 4877 = SearchData.row4877.contains := by funext q; rfl
  rw [row]
  decide

def innerTail650_19 : List Nat := []
theorem innerScan650_19 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_19 best = best := rfl

def innerTail650_18 : List Nat := 8291 :: innerTail650_19
def innerCandidates650_18 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_18_checked : candidates edge 4 4877 8291 innerTail650_19 best.bound = innerCandidates650_18 := by
  unfold candidates
  have row : edge 8291 = SearchData.row8291.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_18 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_18 best = best := by
  rw [innerTail650_18, scan, if_neg (by decide), innerCandidates650_18_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_19
    (if 3 ≤ innerCandidates650_18.length then search edge 3 [4877, 8291] innerCandidates650_18 13168 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_19

def innerTail650_17 : List Nat := 8231 :: innerTail650_18
def innerCandidates650_17 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_17_checked : candidates edge 4 4877 8231 innerTail650_18 best.bound = innerCandidates650_17 := by
  unfold candidates
  have row : edge 8231 = SearchData.row8231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_17 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_17 best = best := by
  rw [innerTail650_17, scan, if_neg (by decide), innerCandidates650_17_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_18
    (if 3 ≤ innerCandidates650_17.length then search edge 3 [4877, 8231] innerCandidates650_17 13108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_18

def innerTail650_16 : List Nat := 8123 :: innerTail650_17
def innerCandidates650_16 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_16_checked : candidates edge 4 4877 8123 innerTail650_17 best.bound = innerCandidates650_16 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_16 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_16 best = best := by
  rw [innerTail650_16, scan, if_neg (by decide), innerCandidates650_16_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_17
    (if 3 ≤ innerCandidates650_16.length then search edge 3 [4877, 8123] innerCandidates650_16 13000 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_17

def innerTail650_15 : List Nat := 7643 :: innerTail650_16
def innerCandidates650_15 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_15_checked : candidates edge 4 4877 7643 innerTail650_16 best.bound = innerCandidates650_15 := by
  unfold candidates
  have row : edge 7643 = SearchData.row7643.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_15 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_15 best = best := by
  rw [innerTail650_15, scan, if_neg (by decide), innerCandidates650_15_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_16
    (if 3 ≤ innerCandidates650_15.length then search edge 3 [4877, 7643] innerCandidates650_15 12520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_16

def innerTail650_14 : List Nat := 7559 :: innerTail650_15
def innerCandidates650_14 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_14_checked : candidates edge 4 4877 7559 innerTail650_15 best.bound = innerCandidates650_14 := by
  unfold candidates
  have row : edge 7559 = SearchData.row7559.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_14 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_14 best = best := by
  rw [innerTail650_14, scan, if_neg (by decide), innerCandidates650_14_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_15
    (if 3 ≤ innerCandidates650_14.length then search edge 3 [4877, 7559] innerCandidates650_14 12436 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_15

def innerTail650_13 : List Nat := 7349 :: innerTail650_14
def innerCandidates650_13 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_13_checked : candidates edge 4 4877 7349 innerTail650_14 best.bound = innerCandidates650_13 := by
  unfold candidates
  have row : edge 7349 = SearchData.row7349.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_13 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_13 best = best := by
  rw [innerTail650_13, scan, if_neg (by decide), innerCandidates650_13_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_14
    (if 3 ≤ innerCandidates650_13.length then search edge 3 [4877, 7349] innerCandidates650_13 12226 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_14

def innerTail650_12 : List Nat := 7307 :: innerTail650_13
def innerCandidates650_12 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_12_checked : candidates edge 4 4877 7307 innerTail650_13 best.bound = innerCandidates650_12 := by
  unfold candidates
  have row : edge 7307 = SearchData.row7307.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_12 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_12 best = best := by
  rw [innerTail650_12, scan, if_neg (by decide), innerCandidates650_12_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_13
    (if 3 ≤ innerCandidates650_12.length then search edge 3 [4877, 7307] innerCandidates650_12 12184 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_13

def innerTail650_11 : List Nat := 6977 :: innerTail650_12
def innerCandidates650_11 : List Nat := [7349]
set_option maxRecDepth 4096 in
theorem innerCandidates650_11_checked : candidates edge 4 4877 6977 innerTail650_12 best.bound = innerCandidates650_11 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_11 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_11 best = best := by
  rw [innerTail650_11, scan, if_neg (by decide), innerCandidates650_11_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_12
    (if 3 ≤ innerCandidates650_11.length then search edge 3 [4877, 6977] innerCandidates650_11 11854 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_12

def innerTail650_10 : List Nat := 6947 :: innerTail650_11
def innerCandidates650_10 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_10_checked : candidates edge 4 4877 6947 innerTail650_11 best.bound = innerCandidates650_10 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_10 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_10 best = best := by
  rw [innerTail650_10, scan, if_neg (by decide), innerCandidates650_10_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_11
    (if 3 ≤ innerCandidates650_10.length then search edge 3 [4877, 6947] innerCandidates650_10 11824 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_11

def innerTail650_9 : List Nat := 6719 :: innerTail650_10
def innerCandidates650_9 : List Nat := [7349]
set_option maxRecDepth 4096 in
theorem innerCandidates650_9_checked : candidates edge 4 4877 6719 innerTail650_10 best.bound = innerCandidates650_9 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_9 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_9 best = best := by
  rw [innerTail650_9, scan, if_neg (by decide), innerCandidates650_9_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_10
    (if 3 ≤ innerCandidates650_9.length then search edge 3 [4877, 6719] innerCandidates650_9 11596 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_10

def innerTail650_8 : List Nat := 6551 :: innerTail650_9
def innerCandidates650_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_8_checked : candidates edge 4 4877 6551 innerTail650_9 best.bound = innerCandidates650_8 := by
  unfold candidates
  have row : edge 6551 = SearchData.row6551.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_8 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_8 best = best := by
  rw [innerTail650_8, scan, if_neg (by decide), innerCandidates650_8_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_9
    (if 3 ≤ innerCandidates650_8.length then search edge 3 [4877, 6551] innerCandidates650_8 11428 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_9

def innerTail650_7 : List Nat := 6359 :: innerTail650_8
def innerCandidates650_7 : List Nat := [8123]
set_option maxRecDepth 4096 in
theorem innerCandidates650_7_checked : candidates edge 4 4877 6359 innerTail650_8 best.bound = innerCandidates650_7 := by
  unfold candidates
  have row : edge 6359 = SearchData.row6359.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_7 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_7 best = best := by
  rw [innerTail650_7, scan, if_neg (by decide), innerCandidates650_7_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_8
    (if 3 ≤ innerCandidates650_7.length then search edge 3 [4877, 6359] innerCandidates650_7 11236 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_8

def innerTail650_6 : List Nat := 6257 :: innerTail650_7
def innerCandidates650_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_6_checked : candidates edge 4 4877 6257 innerTail650_7 best.bound = innerCandidates650_6 := by
  unfold candidates
  have row : edge 6257 = SearchData.row6257.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_6 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_6 best = best := by
  rw [innerTail650_6, scan, if_neg (by decide), innerCandidates650_6_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_7
    (if 3 ≤ innerCandidates650_6.length then search edge 3 [4877, 6257] innerCandidates650_6 11134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_7

def innerTail650_5 : List Nat := 5981 :: innerTail650_6
def innerCandidates650_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_5_checked : candidates edge 4 4877 5981 innerTail650_6 best.bound = innerCandidates650_5 := by
  unfold candidates
  have row : edge 5981 = SearchData.row5981.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_5 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_5 best = best := by
  rw [innerTail650_5, scan, if_neg (by decide), innerCandidates650_5_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_6
    (if 3 ≤ innerCandidates650_5.length then search edge 3 [4877, 5981] innerCandidates650_5 10858 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_6

def innerTail650_4 : List Nat := 5813 :: innerTail650_5
def innerCandidates650_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_4_checked : candidates edge 4 4877 5813 innerTail650_5 best.bound = innerCandidates650_4 := by
  unfold candidates
  have row : edge 5813 = SearchData.row5813.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_4 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_4 best = best := by
  rw [innerTail650_4, scan, if_neg (by decide), innerCandidates650_4_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_5
    (if 3 ≤ innerCandidates650_4.length then search edge 3 [4877, 5813] innerCandidates650_4 10690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_5

def innerTail650_3 : List Nat := 5711 :: innerTail650_4
def innerCandidates650_3 : List Nat := [8291]
set_option maxRecDepth 4096 in
theorem innerCandidates650_3_checked : candidates edge 4 4877 5711 innerTail650_4 best.bound = innerCandidates650_3 := by
  unfold candidates
  have row : edge 5711 = SearchData.row5711.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_3 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_3 best = best := by
  rw [innerTail650_3, scan, if_neg (by decide), innerCandidates650_3_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_4
    (if 3 ≤ innerCandidates650_3.length then search edge 3 [4877, 5711] innerCandidates650_3 10588 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_4

def innerTail650_2 : List Nat := 5693 :: innerTail650_3
def innerCandidates650_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_2_checked : candidates edge 4 4877 5693 innerTail650_3 best.bound = innerCandidates650_2 := by
  unfold candidates
  have row : edge 5693 = SearchData.row5693.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_2 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_2 best = best := by
  rw [innerTail650_2, scan, if_neg (by decide), innerCandidates650_2_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_3
    (if 3 ≤ innerCandidates650_2.length then search edge 3 [4877, 5693] innerCandidates650_2 10570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_3

def innerTail650_1 : List Nat := 5501 :: innerTail650_2
def innerCandidates650_1 : List Nat := [6359, 7349]
set_option maxRecDepth 4096 in
theorem innerCandidates650_1_checked : candidates edge 4 4877 5501 innerTail650_2 best.bound = innerCandidates650_1 := by
  unfold candidates
  have row : edge 5501 = SearchData.row5501.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_1 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_1 best = best := by
  rw [innerTail650_1, scan, if_neg (by decide), innerCandidates650_1_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_2
    (if 3 ≤ innerCandidates650_1.length then search edge 3 [4877, 5501] innerCandidates650_1 10378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_2

def innerTail650_0 : List Nat := 5171 :: innerTail650_1
def innerCandidates650_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates650_0_checked : candidates edge 4 4877 5171 innerTail650_1 best.bound = innerCandidates650_0 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan650_0 : scan edge (search edge 3) 4 [4877] 4877 innerTail650_0 best = best := by
  rw [innerTail650_0, scan, if_neg (by decide), innerCandidates650_0_checked]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_1
    (if 3 ≤ innerCandidates650_0.length then search edge 3 [4877, 5171] innerCandidates650_0 10048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan650_1

theorem rootBranch650 : search edge 4 [4877] rootCandidates650 4877 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4877] 4877 innerTail650_0 best = best
  exact innerScan650_0

def rootCandidates649 : List Nat := [5237, 6389, 6719, 6947, 6959, 7649, 7673, 8069, 8123, 8237]
set_option maxRecDepth 4096 in
theorem rootCandidates649_checked : candidates edge 5 0 4871 tail650 best.bound = rootCandidates649 := by
  unfold candidates
  have row : edge 4871 = SearchData.row4871.contains := by funext q; rfl
  rw [row]
  decide

def innerTail649_10 : List Nat := []
theorem innerScan649_10 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_10 best = best := rfl

def innerTail649_9 : List Nat := 8237 :: innerTail649_10
def innerCandidates649_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_9_checked : candidates edge 4 4871 8237 innerTail649_10 best.bound = innerCandidates649_9 := by
  unfold candidates
  have row : edge 8237 = SearchData.row8237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_9 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_9 best = best := by
  rw [innerTail649_9, scan, if_neg (by decide), innerCandidates649_9_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_10
    (if 3 ≤ innerCandidates649_9.length then search edge 3 [4871, 8237] innerCandidates649_9 13108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_10

def innerTail649_8 : List Nat := 8123 :: innerTail649_9
def innerCandidates649_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_8_checked : candidates edge 4 4871 8123 innerTail649_9 best.bound = innerCandidates649_8 := by
  unfold candidates
  have row : edge 8123 = SearchData.row8123.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_8 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_8 best = best := by
  rw [innerTail649_8, scan, if_neg (by decide), innerCandidates649_8_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_9
    (if 3 ≤ innerCandidates649_8.length then search edge 3 [4871, 8123] innerCandidates649_8 12994 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_9

def innerTail649_7 : List Nat := 8069 :: innerTail649_8
def innerCandidates649_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_7_checked : candidates edge 4 4871 8069 innerTail649_8 best.bound = innerCandidates649_7 := by
  unfold candidates
  have row : edge 8069 = SearchData.row8069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_7 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_7 best = best := by
  rw [innerTail649_7, scan, if_neg (by decide), innerCandidates649_7_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_8
    (if 3 ≤ innerCandidates649_7.length then search edge 3 [4871, 8069] innerCandidates649_7 12940 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_8

def innerTail649_6 : List Nat := 7673 :: innerTail649_7
def innerCandidates649_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_6_checked : candidates edge 4 4871 7673 innerTail649_7 best.bound = innerCandidates649_6 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_6 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_6 best = best := by
  rw [innerTail649_6, scan, if_neg (by decide), innerCandidates649_6_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_7
    (if 3 ≤ innerCandidates649_6.length then search edge 3 [4871, 7673] innerCandidates649_6 12544 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_7

def innerTail649_5 : List Nat := 7649 :: innerTail649_6
def innerCandidates649_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_5_checked : candidates edge 4 4871 7649 innerTail649_6 best.bound = innerCandidates649_5 := by
  unfold candidates
  have row : edge 7649 = SearchData.row7649.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_5 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_5 best = best := by
  rw [innerTail649_5, scan, if_neg (by decide), innerCandidates649_5_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_6
    (if 3 ≤ innerCandidates649_5.length then search edge 3 [4871, 7649] innerCandidates649_5 12520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_6

def innerTail649_4 : List Nat := 6959 :: innerTail649_5
def innerCandidates649_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_4_checked : candidates edge 4 4871 6959 innerTail649_5 best.bound = innerCandidates649_4 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_4 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_4 best = best := by
  rw [innerTail649_4, scan, if_neg (by decide), innerCandidates649_4_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_5
    (if 3 ≤ innerCandidates649_4.length then search edge 3 [4871, 6959] innerCandidates649_4 11830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_5

def innerTail649_3 : List Nat := 6947 :: innerTail649_4
def innerCandidates649_3 : List Nat := [8069]
set_option maxRecDepth 4096 in
theorem innerCandidates649_3_checked : candidates edge 4 4871 6947 innerTail649_4 best.bound = innerCandidates649_3 := by
  unfold candidates
  have row : edge 6947 = SearchData.row6947.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_3 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_3 best = best := by
  rw [innerTail649_3, scan, if_neg (by decide), innerCandidates649_3_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_4
    (if 3 ≤ innerCandidates649_3.length then search edge 3 [4871, 6947] innerCandidates649_3 11818 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_4

def innerTail649_2 : List Nat := 6719 :: innerTail649_3
def innerCandidates649_2 : List Nat := [8237]
set_option maxRecDepth 4096 in
theorem innerCandidates649_2_checked : candidates edge 4 4871 6719 innerTail649_3 best.bound = innerCandidates649_2 := by
  unfold candidates
  have row : edge 6719 = SearchData.row6719.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_2 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_2 best = best := by
  rw [innerTail649_2, scan, if_neg (by decide), innerCandidates649_2_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_3
    (if 3 ≤ innerCandidates649_2.length then search edge 3 [4871, 6719] innerCandidates649_2 11590 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_3

def innerTail649_1 : List Nat := 6389 :: innerTail649_2
def innerCandidates649_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_1_checked : candidates edge 4 4871 6389 innerTail649_2 best.bound = innerCandidates649_1 := by
  unfold candidates
  have row : edge 6389 = SearchData.row6389.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_1 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_1 best = best := by
  rw [innerTail649_1, scan, if_neg (by decide), innerCandidates649_1_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_2
    (if 3 ≤ innerCandidates649_1.length then search edge 3 [4871, 6389] innerCandidates649_1 11260 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_2

def innerTail649_0 : List Nat := 5237 :: innerTail649_1
def innerCandidates649_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates649_0_checked : candidates edge 4 4871 5237 innerTail649_1 best.bound = innerCandidates649_0 := by
  unfold candidates
  have row : edge 5237 = SearchData.row5237.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan649_0 : scan edge (search edge 3) 4 [4871] 4871 innerTail649_0 best = best := by
  rw [innerTail649_0, scan, if_neg (by decide), innerCandidates649_0_checked]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_1
    (if 3 ≤ innerCandidates649_0.length then search edge 3 [4871, 5237] innerCandidates649_0 10108 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan649_1

theorem rootBranch649 : search edge 4 [4871] rootCandidates649 4871 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4871] 4871 innerTail649_0 best = best
  exact innerScan649_0

def rootCandidates648 : List Nat := [5659, 5689, 5953, 6163, 6229, 6361, 6829, 7243, 7573, 7879]
set_option maxRecDepth 4096 in
theorem rootCandidates648_checked : candidates edge 5 0 4861 tail649 best.bound = rootCandidates648 := by
  unfold candidates
  have row : edge 4861 = SearchData.row4861.contains := by funext q; rfl
  rw [row]
  decide

def innerTail648_10 : List Nat := []
theorem innerScan648_10 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_10 best = best := rfl

def innerTail648_9 : List Nat := 7879 :: innerTail648_10
def innerCandidates648_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_9_checked : candidates edge 4 4861 7879 innerTail648_10 best.bound = innerCandidates648_9 := by
  unfold candidates
  have row : edge 7879 = SearchData.row7879.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_9 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_9 best = best := by
  rw [innerTail648_9, scan, if_neg (by decide), innerCandidates648_9_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_10
    (if 3 ≤ innerCandidates648_9.length then search edge 3 [4861, 7879] innerCandidates648_9 12740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_10

def innerTail648_8 : List Nat := 7573 :: innerTail648_9
def innerCandidates648_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_8_checked : candidates edge 4 4861 7573 innerTail648_9 best.bound = innerCandidates648_8 := by
  unfold candidates
  have row : edge 7573 = SearchData.row7573.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_8 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_8 best = best := by
  rw [innerTail648_8, scan, if_neg (by decide), innerCandidates648_8_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_9
    (if 3 ≤ innerCandidates648_8.length then search edge 3 [4861, 7573] innerCandidates648_8 12434 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_9

def innerTail648_7 : List Nat := 7243 :: innerTail648_8
def innerCandidates648_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_7_checked : candidates edge 4 4861 7243 innerTail648_8 best.bound = innerCandidates648_7 := by
  unfold candidates
  have row : edge 7243 = SearchData.row7243.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_7 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_7 best = best := by
  rw [innerTail648_7, scan, if_neg (by decide), innerCandidates648_7_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_8
    (if 3 ≤ innerCandidates648_7.length then search edge 3 [4861, 7243] innerCandidates648_7 12104 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_8

def innerTail648_6 : List Nat := 6829 :: innerTail648_7
def innerCandidates648_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_6_checked : candidates edge 4 4861 6829 innerTail648_7 best.bound = innerCandidates648_6 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_6 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_6 best = best := by
  rw [innerTail648_6, scan, if_neg (by decide), innerCandidates648_6_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_7
    (if 3 ≤ innerCandidates648_6.length then search edge 3 [4861, 6829] innerCandidates648_6 11690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_7

def innerTail648_5 : List Nat := 6361 :: innerTail648_6
def innerCandidates648_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_5_checked : candidates edge 4 4861 6361 innerTail648_6 best.bound = innerCandidates648_5 := by
  unfold candidates
  have row : edge 6361 = SearchData.row6361.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_5 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_5 best = best := by
  rw [innerTail648_5, scan, if_neg (by decide), innerCandidates648_5_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_6
    (if 3 ≤ innerCandidates648_5.length then search edge 3 [4861, 6361] innerCandidates648_5 11222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_6

def innerTail648_4 : List Nat := 6229 :: innerTail648_5
def innerCandidates648_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_4_checked : candidates edge 4 4861 6229 innerTail648_5 best.bound = innerCandidates648_4 := by
  unfold candidates
  have row : edge 6229 = SearchData.row6229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_4 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_4 best = best := by
  rw [innerTail648_4, scan, if_neg (by decide), innerCandidates648_4_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_5
    (if 3 ≤ innerCandidates648_4.length then search edge 3 [4861, 6229] innerCandidates648_4 11090 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_5

def innerTail648_3 : List Nat := 6163 :: innerTail648_4
def innerCandidates648_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_3_checked : candidates edge 4 4861 6163 innerTail648_4 best.bound = innerCandidates648_3 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_3 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_3 best = best := by
  rw [innerTail648_3, scan, if_neg (by decide), innerCandidates648_3_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_4
    (if 3 ≤ innerCandidates648_3.length then search edge 3 [4861, 6163] innerCandidates648_3 11024 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_4

def innerTail648_2 : List Nat := 5953 :: innerTail648_3
def innerCandidates648_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates648_2_checked : candidates edge 4 4861 5953 innerTail648_3 best.bound = innerCandidates648_2 := by
  unfold candidates
  have row : edge 5953 = SearchData.row5953.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_2 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_2 best = best := by
  rw [innerTail648_2, scan, if_neg (by decide), innerCandidates648_2_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_3
    (if 3 ≤ innerCandidates648_2.length then search edge 3 [4861, 5953] innerCandidates648_2 10814 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_3

def innerTail648_1 : List Nat := 5689 :: innerTail648_2
def innerCandidates648_1 : List Nat := [6163, 7573]
set_option maxRecDepth 4096 in
theorem innerCandidates648_1_checked : candidates edge 4 4861 5689 innerTail648_2 best.bound = innerCandidates648_1 := by
  unfold candidates
  have row : edge 5689 = SearchData.row5689.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_1 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_1 best = best := by
  rw [innerTail648_1, scan, if_neg (by decide), innerCandidates648_1_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_2
    (if 3 ≤ innerCandidates648_1.length then search edge 3 [4861, 5689] innerCandidates648_1 10550 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_2

def innerTail648_0 : List Nat := 5659 :: innerTail648_1
def innerCandidates648_0 : List Nat := [5953]
set_option maxRecDepth 4096 in
theorem innerCandidates648_0_checked : candidates edge 4 4861 5659 innerTail648_1 best.bound = innerCandidates648_0 := by
  unfold candidates
  have row : edge 5659 = SearchData.row5659.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan648_0 : scan edge (search edge 3) 4 [4861] 4861 innerTail648_0 best = best := by
  rw [innerTail648_0, scan, if_neg (by decide), innerCandidates648_0_checked]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_1
    (if 3 ≤ innerCandidates648_0.length then search edge 3 [4861, 5659] innerCandidates648_0 10520 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan648_1

theorem rootBranch648 : search edge 4 [4861] rootCandidates648 4861 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4861] 4861 innerTail648_0 best = best
  exact innerScan648_0

def rootCandidates647 : List Nat := [4951, 5209, 5623, 6451, 6553, 6691, 6967, 7069, 7699, 8191]
set_option maxRecDepth 4096 in
theorem rootCandidates647_checked : candidates edge 5 0 4831 tail648 best.bound = rootCandidates647 := by
  unfold candidates
  have row : edge 4831 = SearchData.row4831.contains := by funext q; rfl
  rw [row]
  decide

def innerTail647_10 : List Nat := []
theorem innerScan647_10 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_10 best = best := rfl

def innerTail647_9 : List Nat := 8191 :: innerTail647_10
def innerCandidates647_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_9_checked : candidates edge 4 4831 8191 innerTail647_10 best.bound = innerCandidates647_9 := by
  unfold candidates
  have row : edge 8191 = SearchData.row8191.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_9 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_9 best = best := by
  rw [innerTail647_9, scan, if_neg (by decide), innerCandidates647_9_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_10
    (if 3 ≤ innerCandidates647_9.length then search edge 3 [4831, 8191] innerCandidates647_9 13022 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_10

def innerTail647_8 : List Nat := 7699 :: innerTail647_9
def innerCandidates647_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_8_checked : candidates edge 4 4831 7699 innerTail647_9 best.bound = innerCandidates647_8 := by
  unfold candidates
  have row : edge 7699 = SearchData.row7699.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_8 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_8 best = best := by
  rw [innerTail647_8, scan, if_neg (by decide), innerCandidates647_8_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_9
    (if 3 ≤ innerCandidates647_8.length then search edge 3 [4831, 7699] innerCandidates647_8 12530 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_9

def innerTail647_7 : List Nat := 7069 :: innerTail647_8
def innerCandidates647_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_7_checked : candidates edge 4 4831 7069 innerTail647_8 best.bound = innerCandidates647_7 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_7 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_7 best = best := by
  rw [innerTail647_7, scan, if_neg (by decide), innerCandidates647_7_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_8
    (if 3 ≤ innerCandidates647_7.length then search edge 3 [4831, 7069] innerCandidates647_7 11900 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_8

def innerTail647_6 : List Nat := 6967 :: innerTail647_7
def innerCandidates647_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_6_checked : candidates edge 4 4831 6967 innerTail647_7 best.bound = innerCandidates647_6 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_6 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_6 best = best := by
  rw [innerTail647_6, scan, if_neg (by decide), innerCandidates647_6_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_7
    (if 3 ≤ innerCandidates647_6.length then search edge 3 [4831, 6967] innerCandidates647_6 11798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_7

def innerTail647_5 : List Nat := 6691 :: innerTail647_6
def innerCandidates647_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_5_checked : candidates edge 4 4831 6691 innerTail647_6 best.bound = innerCandidates647_5 := by
  unfold candidates
  have row : edge 6691 = SearchData.row6691.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_5 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_5 best = best := by
  rw [innerTail647_5, scan, if_neg (by decide), innerCandidates647_5_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_6
    (if 3 ≤ innerCandidates647_5.length then search edge 3 [4831, 6691] innerCandidates647_5 11522 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_6

def innerTail647_4 : List Nat := 6553 :: innerTail647_5
def innerCandidates647_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_4_checked : candidates edge 4 4831 6553 innerTail647_5 best.bound = innerCandidates647_4 := by
  unfold candidates
  have row : edge 6553 = SearchData.row6553.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_4 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_4 best = best := by
  rw [innerTail647_4, scan, if_neg (by decide), innerCandidates647_4_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_5
    (if 3 ≤ innerCandidates647_4.length then search edge 3 [4831, 6553] innerCandidates647_4 11384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_5

def innerTail647_3 : List Nat := 6451 :: innerTail647_4
def innerCandidates647_3 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates647_3_checked : candidates edge 4 4831 6451 innerTail647_4 best.bound = innerCandidates647_3 := by
  unfold candidates
  have row : edge 6451 = SearchData.row6451.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_3 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_3 best = best := by
  rw [innerTail647_3, scan, if_neg (by decide), innerCandidates647_3_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_4
    (if 3 ≤ innerCandidates647_3.length then search edge 3 [4831, 6451] innerCandidates647_3 11282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_4

def innerTail647_2 : List Nat := 5623 :: innerTail647_3
def innerCandidates647_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_2_checked : candidates edge 4 4831 5623 innerTail647_3 best.bound = innerCandidates647_2 := by
  unfold candidates
  have row : edge 5623 = SearchData.row5623.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_2 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_2 best = best := by
  rw [innerTail647_2, scan, if_neg (by decide), innerCandidates647_2_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_3
    (if 3 ≤ innerCandidates647_2.length then search edge 3 [4831, 5623] innerCandidates647_2 10454 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_3

def innerTail647_1 : List Nat := 5209 :: innerTail647_2
def innerCandidates647_1 : List Nat := [8191]
set_option maxRecDepth 4096 in
theorem innerCandidates647_1_checked : candidates edge 4 4831 5209 innerTail647_2 best.bound = innerCandidates647_1 := by
  unfold candidates
  have row : edge 5209 = SearchData.row5209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_1 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_1 best = best := by
  rw [innerTail647_1, scan, if_neg (by decide), innerCandidates647_1_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_2
    (if 3 ≤ innerCandidates647_1.length then search edge 3 [4831, 5209] innerCandidates647_1 10040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_2

def innerTail647_0 : List Nat := 4951 :: innerTail647_1
def innerCandidates647_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates647_0_checked : candidates edge 4 4831 4951 innerTail647_1 best.bound = innerCandidates647_0 := by
  unfold candidates
  have row : edge 4951 = SearchData.row4951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan647_0 : scan edge (search edge 3) 4 [4831] 4831 innerTail647_0 best = best := by
  rw [innerTail647_0, scan, if_neg (by decide), innerCandidates647_0_checked]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_1
    (if 3 ≤ innerCandidates647_0.length then search edge 3 [4831, 4951] innerCandidates647_0 9782 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan647_1

theorem rootBranch647 : search edge 4 [4831] rootCandidates647 4831 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [4831] 4831 innerTail647_0 best = best
  exact innerScan647_0

end ProjectEulerSolutions.PrimePairUpper.Search
