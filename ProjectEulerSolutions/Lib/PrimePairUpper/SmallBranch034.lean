import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates775 : List Nat := [6421, 6427, 6871, 7177, 7417, 7963]
set_option maxRecDepth 4096 in
theorem rootCandidates775_checked : candidates edge 5 0 5923 tail776 best.bound = rootCandidates775 := by
  unfold candidates
  have row : edge 5923 = SearchData.row5923.contains := by funext q; rfl
  rw [row]
  decide

def innerTail775_6 : List Nat := []
theorem innerScan775_6 : scan edge (search edge 3) 4 [5923] 5923 innerTail775_6 best = best := rfl

def innerTail775_5 : List Nat := 7963 :: innerTail775_6
def innerCandidates775_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates775_5_checked : candidates edge 4 5923 7963 innerTail775_6 best.bound = innerCandidates775_5 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan775_5 : scan edge (search edge 3) 4 [5923] 5923 innerTail775_5 best = best := by
  rw [innerTail775_5, scan, if_neg (by decide), innerCandidates775_5_checked]
  change scan edge (search edge 3) 4 [5923] 5923 innerTail775_6
    (if 3 ≤ innerCandidates775_5.length then search edge 3 [5923, 7963] innerCandidates775_5 13886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan775_6

def innerTail775_4 : List Nat := 7417 :: innerTail775_5
def innerCandidates775_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates775_4_checked : candidates edge 4 5923 7417 innerTail775_5 best.bound = innerCandidates775_4 := by
  unfold candidates
  have row : edge 7417 = SearchData.row7417.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan775_4 : scan edge (search edge 3) 4 [5923] 5923 innerTail775_4 best = best := by
  rw [innerTail775_4, scan, if_neg (by decide), innerCandidates775_4_checked]
  change scan edge (search edge 3) 4 [5923] 5923 innerTail775_5
    (if 3 ≤ innerCandidates775_4.length then search edge 3 [5923, 7417] innerCandidates775_4 13340 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan775_5

def innerTail775_3 : List Nat := 7177 :: innerTail775_4
def innerCandidates775_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates775_3_checked : candidates edge 4 5923 7177 innerTail775_4 best.bound = innerCandidates775_3 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan775_3 : scan edge (search edge 3) 4 [5923] 5923 innerTail775_3 best = best := by
  rw [innerTail775_3, scan, if_neg (by decide), innerCandidates775_3_checked]
  change scan edge (search edge 3) 4 [5923] 5923 innerTail775_4
    (if 3 ≤ innerCandidates775_3.length then search edge 3 [5923, 7177] innerCandidates775_3 13100 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan775_4

def innerTail775_2 : List Nat := 6871 :: innerTail775_3
def innerCandidates775_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates775_2_checked : candidates edge 4 5923 6871 innerTail775_3 best.bound = innerCandidates775_2 := by
  unfold candidates
  have row : edge 6871 = SearchData.row6871.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan775_2 : scan edge (search edge 3) 4 [5923] 5923 innerTail775_2 best = best := by
  rw [innerTail775_2, scan, if_neg (by decide), innerCandidates775_2_checked]
  change scan edge (search edge 3) 4 [5923] 5923 innerTail775_3
    (if 3 ≤ innerCandidates775_2.length then search edge 3 [5923, 6871] innerCandidates775_2 12794 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan775_3

def innerTail775_1 : List Nat := 6427 :: innerTail775_2
def innerCandidates775_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates775_1_checked : candidates edge 4 5923 6427 innerTail775_2 best.bound = innerCandidates775_1 := by
  unfold candidates
  have row : edge 6427 = SearchData.row6427.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan775_1 : scan edge (search edge 3) 4 [5923] 5923 innerTail775_1 best = best := by
  rw [innerTail775_1, scan, if_neg (by decide), innerCandidates775_1_checked]
  change scan edge (search edge 3) 4 [5923] 5923 innerTail775_2
    (if 3 ≤ innerCandidates775_1.length then search edge 3 [5923, 6427] innerCandidates775_1 12350 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan775_2

def innerTail775_0 : List Nat := 6421 :: innerTail775_1
def innerCandidates775_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates775_0_checked : candidates edge 4 5923 6421 innerTail775_1 best.bound = innerCandidates775_0 := by
  unfold candidates
  have row : edge 6421 = SearchData.row6421.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan775_0 : scan edge (search edge 3) 4 [5923] 5923 innerTail775_0 best = best := by
  rw [innerTail775_0, scan, if_neg (by decide), innerCandidates775_0_checked]
  change scan edge (search edge 3) 4 [5923] 5923 innerTail775_1
    (if 3 ≤ innerCandidates775_0.length then search edge 3 [5923, 6421] innerCandidates775_0 12344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan775_1

theorem rootBranch775 : search edge 4 [5923] rootCandidates775 5923 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5923] 5923 innerTail775_0 best = best
  exact innerScan775_0

def rootCandidates774 : List Nat := [6143, 6311, 6827, 7001, 7109, 7211, 7457, 7523, 7877, 8009]
set_option maxRecDepth 4096 in
theorem rootCandidates774_checked : candidates edge 5 0 5903 tail775 best.bound = rootCandidates774 := by
  unfold candidates
  have row : edge 5903 = SearchData.row5903.contains := by funext q; rfl
  rw [row]
  decide

def innerTail774_10 : List Nat := []
theorem innerScan774_10 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_10 best = best := rfl

def innerTail774_9 : List Nat := 8009 :: innerTail774_10
def innerCandidates774_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_9_checked : candidates edge 4 5903 8009 innerTail774_10 best.bound = innerCandidates774_9 := by
  unfold candidates
  have row : edge 8009 = SearchData.row8009.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_9 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_9 best = best := by
  rw [innerTail774_9, scan, if_neg (by decide), innerCandidates774_9_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_10
    (if 3 ≤ innerCandidates774_9.length then search edge 3 [5903, 8009] innerCandidates774_9 13912 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_10

def innerTail774_8 : List Nat := 7877 :: innerTail774_9
def innerCandidates774_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_8_checked : candidates edge 4 5903 7877 innerTail774_9 best.bound = innerCandidates774_8 := by
  unfold candidates
  have row : edge 7877 = SearchData.row7877.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_8 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_8 best = best := by
  rw [innerTail774_8, scan, if_neg (by decide), innerCandidates774_8_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_9
    (if 3 ≤ innerCandidates774_8.length then search edge 3 [5903, 7877] innerCandidates774_8 13780 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_9

def innerTail774_7 : List Nat := 7523 :: innerTail774_8
def innerCandidates774_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_7_checked : candidates edge 4 5903 7523 innerTail774_8 best.bound = innerCandidates774_7 := by
  unfold candidates
  have row : edge 7523 = SearchData.row7523.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_7 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_7 best = best := by
  rw [innerTail774_7, scan, if_neg (by decide), innerCandidates774_7_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_8
    (if 3 ≤ innerCandidates774_7.length then search edge 3 [5903, 7523] innerCandidates774_7 13426 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_8

def innerTail774_6 : List Nat := 7457 :: innerTail774_7
def innerCandidates774_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_6_checked : candidates edge 4 5903 7457 innerTail774_7 best.bound = innerCandidates774_6 := by
  unfold candidates
  have row : edge 7457 = SearchData.row7457.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_6 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_6 best = best := by
  rw [innerTail774_6, scan, if_neg (by decide), innerCandidates774_6_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_7
    (if 3 ≤ innerCandidates774_6.length then search edge 3 [5903, 7457] innerCandidates774_6 13360 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_7

def innerTail774_5 : List Nat := 7211 :: innerTail774_6
def innerCandidates774_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_5_checked : candidates edge 4 5903 7211 innerTail774_6 best.bound = innerCandidates774_5 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_5 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_5 best = best := by
  rw [innerTail774_5, scan, if_neg (by decide), innerCandidates774_5_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_6
    (if 3 ≤ innerCandidates774_5.length then search edge 3 [5903, 7211] innerCandidates774_5 13114 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_6

def innerTail774_4 : List Nat := 7109 :: innerTail774_5
def innerCandidates774_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_4_checked : candidates edge 4 5903 7109 innerTail774_5 best.bound = innerCandidates774_4 := by
  unfold candidates
  have row : edge 7109 = SearchData.row7109.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_4 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_4 best = best := by
  rw [innerTail774_4, scan, if_neg (by decide), innerCandidates774_4_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_5
    (if 3 ≤ innerCandidates774_4.length then search edge 3 [5903, 7109] innerCandidates774_4 13012 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_5

def innerTail774_3 : List Nat := 7001 :: innerTail774_4
def innerCandidates774_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_3_checked : candidates edge 4 5903 7001 innerTail774_4 best.bound = innerCandidates774_3 := by
  unfold candidates
  have row : edge 7001 = SearchData.row7001.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_3 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_3 best = best := by
  rw [innerTail774_3, scan, if_neg (by decide), innerCandidates774_3_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_4
    (if 3 ≤ innerCandidates774_3.length then search edge 3 [5903, 7001] innerCandidates774_3 12904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_4

def innerTail774_2 : List Nat := 6827 :: innerTail774_3
def innerCandidates774_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_2_checked : candidates edge 4 5903 6827 innerTail774_3 best.bound = innerCandidates774_2 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_2 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_2 best = best := by
  rw [innerTail774_2, scan, if_neg (by decide), innerCandidates774_2_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_3
    (if 3 ≤ innerCandidates774_2.length then search edge 3 [5903, 6827] innerCandidates774_2 12730 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_3

def innerTail774_1 : List Nat := 6311 :: innerTail774_2
def innerCandidates774_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates774_1_checked : candidates edge 4 5903 6311 innerTail774_2 best.bound = innerCandidates774_1 := by
  unfold candidates
  have row : edge 6311 = SearchData.row6311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_1 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_1 best = best := by
  rw [innerTail774_1, scan, if_neg (by decide), innerCandidates774_1_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_2
    (if 3 ≤ innerCandidates774_1.length then search edge 3 [5903, 6311] innerCandidates774_1 12214 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_2

def innerTail774_0 : List Nat := 6143 :: innerTail774_1
def innerCandidates774_0 : List Nat := [6311]
set_option maxRecDepth 4096 in
theorem innerCandidates774_0_checked : candidates edge 4 5903 6143 innerTail774_1 best.bound = innerCandidates774_0 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan774_0 : scan edge (search edge 3) 4 [5903] 5903 innerTail774_0 best = best := by
  rw [innerTail774_0, scan, if_neg (by decide), innerCandidates774_0_checked]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_1
    (if 3 ≤ innerCandidates774_0.length then search edge 3 [5903, 6143] innerCandidates774_0 12046 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan774_1

theorem rootBranch774 : search edge 4 [5903] rootCandidates774 5903 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5903] 5903 innerTail774_0 best = best
  exact innerScan774_0

def rootCandidates773 : List Nat := [6131, 6143, 6977, 7019]
set_option maxRecDepth 4096 in
theorem rootCandidates773_checked : candidates edge 5 0 5897 tail774 best.bound = rootCandidates773 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext q; rfl
  rw [row]
  decide

def innerTail773_4 : List Nat := []
theorem innerScan773_4 : scan edge (search edge 3) 4 [5897] 5897 innerTail773_4 best = best := rfl

def innerTail773_3 : List Nat := 7019 :: innerTail773_4
def innerCandidates773_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates773_3_checked : candidates edge 4 5897 7019 innerTail773_4 best.bound = innerCandidates773_3 := by
  unfold candidates
  have row : edge 7019 = SearchData.row7019.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan773_3 : scan edge (search edge 3) 4 [5897] 5897 innerTail773_3 best = best := by
  rw [innerTail773_3, scan, if_neg (by decide), innerCandidates773_3_checked]
  change scan edge (search edge 3) 4 [5897] 5897 innerTail773_4
    (if 3 ≤ innerCandidates773_3.length then search edge 3 [5897, 7019] innerCandidates773_3 12916 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan773_4

def innerTail773_2 : List Nat := 6977 :: innerTail773_3
def innerCandidates773_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates773_2_checked : candidates edge 4 5897 6977 innerTail773_3 best.bound = innerCandidates773_2 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan773_2 : scan edge (search edge 3) 4 [5897] 5897 innerTail773_2 best = best := by
  rw [innerTail773_2, scan, if_neg (by decide), innerCandidates773_2_checked]
  change scan edge (search edge 3) 4 [5897] 5897 innerTail773_3
    (if 3 ≤ innerCandidates773_2.length then search edge 3 [5897, 6977] innerCandidates773_2 12874 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan773_3

def innerTail773_1 : List Nat := 6143 :: innerTail773_2
def innerCandidates773_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates773_1_checked : candidates edge 4 5897 6143 innerTail773_2 best.bound = innerCandidates773_1 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan773_1 : scan edge (search edge 3) 4 [5897] 5897 innerTail773_1 best = best := by
  rw [innerTail773_1, scan, if_neg (by decide), innerCandidates773_1_checked]
  change scan edge (search edge 3) 4 [5897] 5897 innerTail773_2
    (if 3 ≤ innerCandidates773_1.length then search edge 3 [5897, 6143] innerCandidates773_1 12040 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan773_2

def innerTail773_0 : List Nat := 6131 :: innerTail773_1
def innerCandidates773_0 : List Nat := [7019]
set_option maxRecDepth 4096 in
theorem innerCandidates773_0_checked : candidates edge 4 5897 6131 innerTail773_1 best.bound = innerCandidates773_0 := by
  unfold candidates
  have row : edge 6131 = SearchData.row6131.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan773_0 : scan edge (search edge 3) 4 [5897] 5897 innerTail773_0 best = best := by
  rw [innerTail773_0, scan, if_neg (by decide), innerCandidates773_0_checked]
  change scan edge (search edge 3) 4 [5897] 5897 innerTail773_1
    (if 3 ≤ innerCandidates773_0.length then search edge 3 [5897, 6131] innerCandidates773_0 12028 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan773_1

theorem rootBranch773 : search edge 4 [5897] rootCandidates773 5897 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5897] 5897 innerTail773_0 best = best
  exact innerScan773_0

def rootCandidates772 : List Nat := [7723, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates772_checked : candidates edge 5 0 5881 tail773 best.bound = rootCandidates772 := by
  unfold candidates
  have row : edge 5881 = SearchData.row5881.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates771 : List Nat := [6263, 6779, 7127, 8273]
set_option maxRecDepth 4096 in
theorem rootCandidates771_checked : candidates edge 5 0 5879 tail772 best.bound = rootCandidates771 := by
  unfold candidates
  have row : edge 5879 = SearchData.row5879.contains := by funext q; rfl
  rw [row]
  decide

def innerTail771_4 : List Nat := []
theorem innerScan771_4 : scan edge (search edge 3) 4 [5879] 5879 innerTail771_4 best = best := rfl

def innerTail771_3 : List Nat := 8273 :: innerTail771_4
def innerCandidates771_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates771_3_checked : candidates edge 4 5879 8273 innerTail771_4 best.bound = innerCandidates771_3 := by
  unfold candidates
  have row : edge 8273 = SearchData.row8273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan771_3 : scan edge (search edge 3) 4 [5879] 5879 innerTail771_3 best = best := by
  rw [innerTail771_3, scan, if_neg (by decide), innerCandidates771_3_checked]
  change scan edge (search edge 3) 4 [5879] 5879 innerTail771_4
    (if 3 ≤ innerCandidates771_3.length then search edge 3 [5879, 8273] innerCandidates771_3 14152 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan771_4

def innerTail771_2 : List Nat := 7127 :: innerTail771_3
def innerCandidates771_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates771_2_checked : candidates edge 4 5879 7127 innerTail771_3 best.bound = innerCandidates771_2 := by
  unfold candidates
  have row : edge 7127 = SearchData.row7127.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan771_2 : scan edge (search edge 3) 4 [5879] 5879 innerTail771_2 best = best := by
  rw [innerTail771_2, scan, if_neg (by decide), innerCandidates771_2_checked]
  change scan edge (search edge 3) 4 [5879] 5879 innerTail771_3
    (if 3 ≤ innerCandidates771_2.length then search edge 3 [5879, 7127] innerCandidates771_2 13006 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan771_3

def innerTail771_1 : List Nat := 6779 :: innerTail771_2
def innerCandidates771_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates771_1_checked : candidates edge 4 5879 6779 innerTail771_2 best.bound = innerCandidates771_1 := by
  unfold candidates
  have row : edge 6779 = SearchData.row6779.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan771_1 : scan edge (search edge 3) 4 [5879] 5879 innerTail771_1 best = best := by
  rw [innerTail771_1, scan, if_neg (by decide), innerCandidates771_1_checked]
  change scan edge (search edge 3) 4 [5879] 5879 innerTail771_2
    (if 3 ≤ innerCandidates771_1.length then search edge 3 [5879, 6779] innerCandidates771_1 12658 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan771_2

def innerTail771_0 : List Nat := 6263 :: innerTail771_1
def innerCandidates771_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates771_0_checked : candidates edge 4 5879 6263 innerTail771_1 best.bound = innerCandidates771_0 := by
  unfold candidates
  have row : edge 6263 = SearchData.row6263.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan771_0 : scan edge (search edge 3) 4 [5879] 5879 innerTail771_0 best = best := by
  rw [innerTail771_0, scan, if_neg (by decide), innerCandidates771_0_checked]
  change scan edge (search edge 3) 4 [5879] 5879 innerTail771_1
    (if 3 ≤ innerCandidates771_0.length then search edge 3 [5879, 6263] innerCandidates771_0 12142 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan771_1

theorem rootBranch771 : search edge 4 [5879] rootCandidates771 5879 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5879] 5879 innerTail771_0 best = best
  exact innerScan771_0

def rootCandidates770 : List Nat := [6067, 6163, 6709, 6829, 6961, 7069, 8089, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates770_checked : candidates edge 5 0 5869 tail771 best.bound = rootCandidates770 := by
  unfold candidates
  have row : edge 5869 = SearchData.row5869.contains := by funext q; rfl
  rw [row]
  decide

def innerTail770_8 : List Nat := []
theorem innerScan770_8 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_8 best = best := rfl

def innerTail770_7 : List Nat := 8209 :: innerTail770_8
def innerCandidates770_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates770_7_checked : candidates edge 4 5869 8209 innerTail770_8 best.bound = innerCandidates770_7 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_7 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_7 best = best := by
  rw [innerTail770_7, scan, if_neg (by decide), innerCandidates770_7_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_8
    (if 3 ≤ innerCandidates770_7.length then search edge 3 [5869, 8209] innerCandidates770_7 14078 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_8

def innerTail770_6 : List Nat := 8089 :: innerTail770_7
def innerCandidates770_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates770_6_checked : candidates edge 4 5869 8089 innerTail770_7 best.bound = innerCandidates770_6 := by
  unfold candidates
  have row : edge 8089 = SearchData.row8089.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_6 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_6 best = best := by
  rw [innerTail770_6, scan, if_neg (by decide), innerCandidates770_6_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_7
    (if 3 ≤ innerCandidates770_6.length then search edge 3 [5869, 8089] innerCandidates770_6 13958 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_7

def innerTail770_5 : List Nat := 7069 :: innerTail770_6
def innerCandidates770_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates770_5_checked : candidates edge 4 5869 7069 innerTail770_6 best.bound = innerCandidates770_5 := by
  unfold candidates
  have row : edge 7069 = SearchData.row7069.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_5 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_5 best = best := by
  rw [innerTail770_5, scan, if_neg (by decide), innerCandidates770_5_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_6
    (if 3 ≤ innerCandidates770_5.length then search edge 3 [5869, 7069] innerCandidates770_5 12938 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_6

def innerTail770_4 : List Nat := 6961 :: innerTail770_5
def innerCandidates770_4 : List Nat := [7069]
set_option maxRecDepth 4096 in
theorem innerCandidates770_4_checked : candidates edge 4 5869 6961 innerTail770_5 best.bound = innerCandidates770_4 := by
  unfold candidates
  have row : edge 6961 = SearchData.row6961.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_4 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_4 best = best := by
  rw [innerTail770_4, scan, if_neg (by decide), innerCandidates770_4_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_5
    (if 3 ≤ innerCandidates770_4.length then search edge 3 [5869, 6961] innerCandidates770_4 12830 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_5

def innerTail770_3 : List Nat := 6829 :: innerTail770_4
def innerCandidates770_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates770_3_checked : candidates edge 4 5869 6829 innerTail770_4 best.bound = innerCandidates770_3 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_3 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_3 best = best := by
  rw [innerTail770_3, scan, if_neg (by decide), innerCandidates770_3_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_4
    (if 3 ≤ innerCandidates770_3.length then search edge 3 [5869, 6829] innerCandidates770_3 12698 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_4

def innerTail770_2 : List Nat := 6709 :: innerTail770_3
def innerCandidates770_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates770_2_checked : candidates edge 4 5869 6709 innerTail770_3 best.bound = innerCandidates770_2 := by
  unfold candidates
  have row : edge 6709 = SearchData.row6709.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_2 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_2 best = best := by
  rw [innerTail770_2, scan, if_neg (by decide), innerCandidates770_2_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_3
    (if 3 ≤ innerCandidates770_2.length then search edge 3 [5869, 6709] innerCandidates770_2 12578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_3

def innerTail770_1 : List Nat := 6163 :: innerTail770_2
def innerCandidates770_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates770_1_checked : candidates edge 4 5869 6163 innerTail770_2 best.bound = innerCandidates770_1 := by
  unfold candidates
  have row : edge 6163 = SearchData.row6163.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_1 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_1 best = best := by
  rw [innerTail770_1, scan, if_neg (by decide), innerCandidates770_1_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_2
    (if 3 ≤ innerCandidates770_1.length then search edge 3 [5869, 6163] innerCandidates770_1 12032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_2

def innerTail770_0 : List Nat := 6067 :: innerTail770_1
def innerCandidates770_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates770_0_checked : candidates edge 4 5869 6067 innerTail770_1 best.bound = innerCandidates770_0 := by
  unfold candidates
  have row : edge 6067 = SearchData.row6067.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan770_0 : scan edge (search edge 3) 4 [5869] 5869 innerTail770_0 best = best := by
  rw [innerTail770_0, scan, if_neg (by decide), innerCandidates770_0_checked]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_1
    (if 3 ≤ innerCandidates770_0.length then search edge 3 [5869, 6067] innerCandidates770_0 11936 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan770_1

theorem rootBranch770 : search edge 4 [5869] rootCandidates770 5869 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5869] 5869 innerTail770_0 best = best
  exact innerScan770_0

def rootCandidates769 : List Nat := [6329, 7001]
set_option maxRecDepth 4096 in
theorem rootCandidates769_checked : candidates edge 5 0 5867 tail770 best.bound = rootCandidates769 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates768 : List Nat := [5987, 6143, 6827, 7043, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates768_checked : candidates edge 5 0 5861 tail769 best.bound = rootCandidates768 := by
  unfold candidates
  have row : edge 5861 = SearchData.row5861.contains := by funext q; rfl
  rw [row]
  decide

def innerTail768_5 : List Nat := []
theorem innerScan768_5 : scan edge (search edge 3) 4 [5861] 5861 innerTail768_5 best = best := rfl

def innerTail768_4 : List Nat := 8171 :: innerTail768_5
def innerCandidates768_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates768_4_checked : candidates edge 4 5861 8171 innerTail768_5 best.bound = innerCandidates768_4 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan768_4 : scan edge (search edge 3) 4 [5861] 5861 innerTail768_4 best = best := by
  rw [innerTail768_4, scan, if_neg (by decide), innerCandidates768_4_checked]
  change scan edge (search edge 3) 4 [5861] 5861 innerTail768_5
    (if 3 ≤ innerCandidates768_4.length then search edge 3 [5861, 8171] innerCandidates768_4 14032 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan768_5

def innerTail768_3 : List Nat := 7043 :: innerTail768_4
def innerCandidates768_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates768_3_checked : candidates edge 4 5861 7043 innerTail768_4 best.bound = innerCandidates768_3 := by
  unfold candidates
  have row : edge 7043 = SearchData.row7043.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan768_3 : scan edge (search edge 3) 4 [5861] 5861 innerTail768_3 best = best := by
  rw [innerTail768_3, scan, if_neg (by decide), innerCandidates768_3_checked]
  change scan edge (search edge 3) 4 [5861] 5861 innerTail768_4
    (if 3 ≤ innerCandidates768_3.length then search edge 3 [5861, 7043] innerCandidates768_3 12904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan768_4

def innerTail768_2 : List Nat := 6827 :: innerTail768_3
def innerCandidates768_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates768_2_checked : candidates edge 4 5861 6827 innerTail768_3 best.bound = innerCandidates768_2 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan768_2 : scan edge (search edge 3) 4 [5861] 5861 innerTail768_2 best = best := by
  rw [innerTail768_2, scan, if_neg (by decide), innerCandidates768_2_checked]
  change scan edge (search edge 3) 4 [5861] 5861 innerTail768_3
    (if 3 ≤ innerCandidates768_2.length then search edge 3 [5861, 6827] innerCandidates768_2 12688 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan768_3

def innerTail768_1 : List Nat := 6143 :: innerTail768_2
def innerCandidates768_1 : List Nat := [8171]
set_option maxRecDepth 4096 in
theorem innerCandidates768_1_checked : candidates edge 4 5861 6143 innerTail768_2 best.bound = innerCandidates768_1 := by
  unfold candidates
  have row : edge 6143 = SearchData.row6143.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan768_1 : scan edge (search edge 3) 4 [5861] 5861 innerTail768_1 best = best := by
  rw [innerTail768_1, scan, if_neg (by decide), innerCandidates768_1_checked]
  change scan edge (search edge 3) 4 [5861] 5861 innerTail768_2
    (if 3 ≤ innerCandidates768_1.length then search edge 3 [5861, 6143] innerCandidates768_1 12004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan768_2

def innerTail768_0 : List Nat := 5987 :: innerTail768_1
def innerCandidates768_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates768_0_checked : candidates edge 4 5861 5987 innerTail768_1 best.bound = innerCandidates768_0 := by
  unfold candidates
  have row : edge 5987 = SearchData.row5987.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan768_0 : scan edge (search edge 3) 4 [5861] 5861 innerTail768_0 best = best := by
  rw [innerTail768_0, scan, if_neg (by decide), innerCandidates768_0_checked]
  change scan edge (search edge 3) 4 [5861] 5861 innerTail768_1
    (if 3 ≤ innerCandidates768_0.length then search edge 3 [5861, 5987] innerCandidates768_0 11848 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan768_1

theorem rootBranch768 : search edge 4 [5861] rootCandidates768 5861 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5861] 5861 innerTail768_0 best = best
  exact innerScan768_0

end ProjectEulerSolutions.PrimePairUpper.Search
