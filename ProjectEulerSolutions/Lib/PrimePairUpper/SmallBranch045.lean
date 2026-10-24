import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates689 : List Nat := [5701, 6733, 6997, 7591, 7963, 8209, 8221]
set_option maxRecDepth 4096 in
theorem rootCandidates689_checked : candidates edge 5 0 5197 tail690 best.bound = rootCandidates689 := by
  unfold candidates
  have row : edge 5197 = SearchData.row5197.contains := by funext q; rfl
  rw [row]
  decide

def innerTail689_7 : List Nat := []
theorem innerScan689_7 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_7 best = best := rfl

def innerTail689_6 : List Nat := 8221 :: innerTail689_7
def innerCandidates689_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates689_6_checked : candidates edge 4 5197 8221 innerTail689_7 best.bound = innerCandidates689_6 := by
  unfold candidates
  have row : edge 8221 = SearchData.row8221.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan689_6 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_6 best = best := by
  rw [innerTail689_6, scan, if_neg (by decide), innerCandidates689_6_checked]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_7
    (if 3 ≤ innerCandidates689_6.length then search edge 3 [5197, 8221] innerCandidates689_6 13418 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan689_7

def innerTail689_5 : List Nat := 8209 :: innerTail689_6
def innerCandidates689_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates689_5_checked : candidates edge 4 5197 8209 innerTail689_6 best.bound = innerCandidates689_5 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan689_5 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_5 best = best := by
  rw [innerTail689_5, scan, if_neg (by decide), innerCandidates689_5_checked]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_6
    (if 3 ≤ innerCandidates689_5.length then search edge 3 [5197, 8209] innerCandidates689_5 13406 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan689_6

def innerTail689_4 : List Nat := 7963 :: innerTail689_5
def innerCandidates689_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates689_4_checked : candidates edge 4 5197 7963 innerTail689_5 best.bound = innerCandidates689_4 := by
  unfold candidates
  have row : edge 7963 = SearchData.row7963.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan689_4 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_4 best = best := by
  rw [innerTail689_4, scan, if_neg (by decide), innerCandidates689_4_checked]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_5
    (if 3 ≤ innerCandidates689_4.length then search edge 3 [5197, 7963] innerCandidates689_4 13160 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan689_5

def innerTail689_3 : List Nat := 7591 :: innerTail689_4
def innerCandidates689_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates689_3_checked : candidates edge 4 5197 7591 innerTail689_4 best.bound = innerCandidates689_3 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan689_3 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_3 best = best := by
  rw [innerTail689_3, scan, if_neg (by decide), innerCandidates689_3_checked]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_4
    (if 3 ≤ innerCandidates689_3.length then search edge 3 [5197, 7591] innerCandidates689_3 12788 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan689_4

def innerTail689_2 : List Nat := 6997 :: innerTail689_3
def innerCandidates689_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates689_2_checked : candidates edge 4 5197 6997 innerTail689_3 best.bound = innerCandidates689_2 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan689_2 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_2 best = best := by
  rw [innerTail689_2, scan, if_neg (by decide), innerCandidates689_2_checked]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_3
    (if 3 ≤ innerCandidates689_2.length then search edge 3 [5197, 6997] innerCandidates689_2 12194 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan689_3

def innerTail689_1 : List Nat := 6733 :: innerTail689_2
def innerCandidates689_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates689_1_checked : candidates edge 4 5197 6733 innerTail689_2 best.bound = innerCandidates689_1 := by
  unfold candidates
  have row : edge 6733 = SearchData.row6733.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan689_1 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_1 best = best := by
  rw [innerTail689_1, scan, if_neg (by decide), innerCandidates689_1_checked]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_2
    (if 3 ≤ innerCandidates689_1.length then search edge 3 [5197, 6733] innerCandidates689_1 11930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan689_2

def innerTail689_0 : List Nat := 5701 :: innerTail689_1
def innerCandidates689_0 : List Nat := [6733, 8209]
set_option maxRecDepth 4096 in
theorem innerCandidates689_0_checked : candidates edge 4 5197 5701 innerTail689_1 best.bound = innerCandidates689_0 := by
  unfold candidates
  have row : edge 5701 = SearchData.row5701.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan689_0 : scan edge (search edge 3) 4 [5197] 5197 innerTail689_0 best = best := by
  rw [innerTail689_0, scan, if_neg (by decide), innerCandidates689_0_checked]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_1
    (if 3 ≤ innerCandidates689_0.length then search edge 3 [5197, 5701] innerCandidates689_0 10898 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan689_1

theorem rootBranch689 : search edge 4 [5197] rootCandidates689 5197 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5197] 5197 innerTail689_0 best = best
  exact innerScan689_0

def rootCandidates688 : List Nat := [6329, 6491]
set_option maxRecDepth 4096 in
theorem rootCandidates688_checked : candidates edge 5 0 5189 tail689 best.bound = rootCandidates688 := by
  unfold candidates
  have row : edge 5189 = SearchData.row5189.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates687 : List Nat := [5683, 5851, 6301, 6763, 7873, 8209]
set_option maxRecDepth 4096 in
theorem rootCandidates687_checked : candidates edge 5 0 5179 tail688 best.bound = rootCandidates687 := by
  unfold candidates
  have row : edge 5179 = SearchData.row5179.contains := by funext q; rfl
  rw [row]
  decide

def innerTail687_6 : List Nat := []
theorem innerScan687_6 : scan edge (search edge 3) 4 [5179] 5179 innerTail687_6 best = best := rfl

def innerTail687_5 : List Nat := 8209 :: innerTail687_6
def innerCandidates687_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates687_5_checked : candidates edge 4 5179 8209 innerTail687_6 best.bound = innerCandidates687_5 := by
  unfold candidates
  have row : edge 8209 = SearchData.row8209.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan687_5 : scan edge (search edge 3) 4 [5179] 5179 innerTail687_5 best = best := by
  rw [innerTail687_5, scan, if_neg (by decide), innerCandidates687_5_checked]
  change scan edge (search edge 3) 4 [5179] 5179 innerTail687_6
    (if 3 ≤ innerCandidates687_5.length then search edge 3 [5179, 8209] innerCandidates687_5 13388 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan687_6

def innerTail687_4 : List Nat := 7873 :: innerTail687_5
def innerCandidates687_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates687_4_checked : candidates edge 4 5179 7873 innerTail687_5 best.bound = innerCandidates687_4 := by
  unfold candidates
  have row : edge 7873 = SearchData.row7873.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan687_4 : scan edge (search edge 3) 4 [5179] 5179 innerTail687_4 best = best := by
  rw [innerTail687_4, scan, if_neg (by decide), innerCandidates687_4_checked]
  change scan edge (search edge 3) 4 [5179] 5179 innerTail687_5
    (if 3 ≤ innerCandidates687_4.length then search edge 3 [5179, 7873] innerCandidates687_4 13052 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan687_5

def innerTail687_3 : List Nat := 6763 :: innerTail687_4
def innerCandidates687_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates687_3_checked : candidates edge 4 5179 6763 innerTail687_4 best.bound = innerCandidates687_3 := by
  unfold candidates
  have row : edge 6763 = SearchData.row6763.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan687_3 : scan edge (search edge 3) 4 [5179] 5179 innerTail687_3 best = best := by
  rw [innerTail687_3, scan, if_neg (by decide), innerCandidates687_3_checked]
  change scan edge (search edge 3) 4 [5179] 5179 innerTail687_4
    (if 3 ≤ innerCandidates687_3.length then search edge 3 [5179, 6763] innerCandidates687_3 11942 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan687_4

def innerTail687_2 : List Nat := 6301 :: innerTail687_3
def innerCandidates687_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates687_2_checked : candidates edge 4 5179 6301 innerTail687_3 best.bound = innerCandidates687_2 := by
  unfold candidates
  have row : edge 6301 = SearchData.row6301.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan687_2 : scan edge (search edge 3) 4 [5179] 5179 innerTail687_2 best = best := by
  rw [innerTail687_2, scan, if_neg (by decide), innerCandidates687_2_checked]
  change scan edge (search edge 3) 4 [5179] 5179 innerTail687_3
    (if 3 ≤ innerCandidates687_2.length then search edge 3 [5179, 6301] innerCandidates687_2 11480 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan687_3

def innerTail687_1 : List Nat := 5851 :: innerTail687_2
def innerCandidates687_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates687_1_checked : candidates edge 4 5179 5851 innerTail687_2 best.bound = innerCandidates687_1 := by
  unfold candidates
  have row : edge 5851 = SearchData.row5851.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan687_1 : scan edge (search edge 3) 4 [5179] 5179 innerTail687_1 best = best := by
  rw [innerTail687_1, scan, if_neg (by decide), innerCandidates687_1_checked]
  change scan edge (search edge 3) 4 [5179] 5179 innerTail687_2
    (if 3 ≤ innerCandidates687_1.length then search edge 3 [5179, 5851] innerCandidates687_1 11030 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan687_2

def innerTail687_0 : List Nat := 5683 :: innerTail687_1
def innerCandidates687_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates687_0_checked : candidates edge 4 5179 5683 innerTail687_1 best.bound = innerCandidates687_0 := by
  unfold candidates
  have row : edge 5683 = SearchData.row5683.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan687_0 : scan edge (search edge 3) 4 [5179] 5179 innerTail687_0 best = best := by
  rw [innerTail687_0, scan, if_neg (by decide), innerCandidates687_0_checked]
  change scan edge (search edge 3) 4 [5179] 5179 innerTail687_1
    (if 3 ≤ innerCandidates687_0.length then search edge 3 [5179, 5683] innerCandidates687_0 10862 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan687_1

theorem rootBranch687 : search edge 4 [5179] rootCandidates687 5179 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5179] 5179 innerTail687_0 best = best
  exact innerScan687_0

def rootCandidates686 : List Nat := [5297, 5393, 5897, 7187, 8039, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates686_checked : candidates edge 5 0 5171 tail687 best.bound = rootCandidates686 := by
  unfold candidates
  have row : edge 5171 = SearchData.row5171.contains := by funext q; rfl
  rw [row]
  decide

def innerTail686_6 : List Nat := []
theorem innerScan686_6 : scan edge (search edge 3) 4 [5171] 5171 innerTail686_6 best = best := rfl

def innerTail686_5 : List Nat := 8111 :: innerTail686_6
def innerCandidates686_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates686_5_checked : candidates edge 4 5171 8111 innerTail686_6 best.bound = innerCandidates686_5 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan686_5 : scan edge (search edge 3) 4 [5171] 5171 innerTail686_5 best = best := by
  rw [innerTail686_5, scan, if_neg (by decide), innerCandidates686_5_checked]
  change scan edge (search edge 3) 4 [5171] 5171 innerTail686_6
    (if 3 ≤ innerCandidates686_5.length then search edge 3 [5171, 8111] innerCandidates686_5 13282 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan686_6

def innerTail686_4 : List Nat := 8039 :: innerTail686_5
def innerCandidates686_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates686_4_checked : candidates edge 4 5171 8039 innerTail686_5 best.bound = innerCandidates686_4 := by
  unfold candidates
  have row : edge 8039 = SearchData.row8039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan686_4 : scan edge (search edge 3) 4 [5171] 5171 innerTail686_4 best = best := by
  rw [innerTail686_4, scan, if_neg (by decide), innerCandidates686_4_checked]
  change scan edge (search edge 3) 4 [5171] 5171 innerTail686_5
    (if 3 ≤ innerCandidates686_4.length then search edge 3 [5171, 8039] innerCandidates686_4 13210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan686_5

def innerTail686_3 : List Nat := 7187 :: innerTail686_4
def innerCandidates686_3 : List Nat := [8111]
set_option maxRecDepth 4096 in
theorem innerCandidates686_3_checked : candidates edge 4 5171 7187 innerTail686_4 best.bound = innerCandidates686_3 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan686_3 : scan edge (search edge 3) 4 [5171] 5171 innerTail686_3 best = best := by
  rw [innerTail686_3, scan, if_neg (by decide), innerCandidates686_3_checked]
  change scan edge (search edge 3) 4 [5171] 5171 innerTail686_4
    (if 3 ≤ innerCandidates686_3.length then search edge 3 [5171, 7187] innerCandidates686_3 12358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan686_4

def innerTail686_2 : List Nat := 5897 :: innerTail686_3
def innerCandidates686_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates686_2_checked : candidates edge 4 5171 5897 innerTail686_3 best.bound = innerCandidates686_2 := by
  unfold candidates
  have row : edge 5897 = SearchData.row5897.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan686_2 : scan edge (search edge 3) 4 [5171] 5171 innerTail686_2 best = best := by
  rw [innerTail686_2, scan, if_neg (by decide), innerCandidates686_2_checked]
  change scan edge (search edge 3) 4 [5171] 5171 innerTail686_3
    (if 3 ≤ innerCandidates686_2.length then search edge 3 [5171, 5897] innerCandidates686_2 11068 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan686_3

def innerTail686_1 : List Nat := 5393 :: innerTail686_2
def innerCandidates686_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates686_1_checked : candidates edge 4 5171 5393 innerTail686_2 best.bound = innerCandidates686_1 := by
  unfold candidates
  have row : edge 5393 = SearchData.row5393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan686_1 : scan edge (search edge 3) 4 [5171] 5171 innerTail686_1 best = best := by
  rw [innerTail686_1, scan, if_neg (by decide), innerCandidates686_1_checked]
  change scan edge (search edge 3) 4 [5171] 5171 innerTail686_2
    (if 3 ≤ innerCandidates686_1.length then search edge 3 [5171, 5393] innerCandidates686_1 10564 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan686_2

def innerTail686_0 : List Nat := 5297 :: innerTail686_1
def innerCandidates686_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates686_0_checked : candidates edge 4 5171 5297 innerTail686_1 best.bound = innerCandidates686_0 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan686_0 : scan edge (search edge 3) 4 [5171] 5171 innerTail686_0 best = best := by
  rw [innerTail686_0, scan, if_neg (by decide), innerCandidates686_0_checked]
  change scan edge (search edge 3) 4 [5171] 5171 innerTail686_1
    (if 3 ≤ innerCandidates686_0.length then search edge 3 [5171, 5297] innerCandidates686_0 10468 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan686_1

theorem rootBranch686 : search edge 4 [5171] rootCandidates686 5171 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5171] 5171 innerTail686_0 best = best
  exact innerScan686_0

def rootCandidates685 : List Nat := [5737, 6661, 6967, 7177, 7741]
set_option maxRecDepth 4096 in
theorem rootCandidates685_checked : candidates edge 5 0 5167 tail686 best.bound = rootCandidates685 := by
  unfold candidates
  have row : edge 5167 = SearchData.row5167.contains := by funext q; rfl
  rw [row]
  decide

def innerTail685_5 : List Nat := []
theorem innerScan685_5 : scan edge (search edge 3) 4 [5167] 5167 innerTail685_5 best = best := rfl

def innerTail685_4 : List Nat := 7741 :: innerTail685_5
def innerCandidates685_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates685_4_checked : candidates edge 4 5167 7741 innerTail685_5 best.bound = innerCandidates685_4 := by
  unfold candidates
  have row : edge 7741 = SearchData.row7741.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan685_4 : scan edge (search edge 3) 4 [5167] 5167 innerTail685_4 best = best := by
  rw [innerTail685_4, scan, if_neg (by decide), innerCandidates685_4_checked]
  change scan edge (search edge 3) 4 [5167] 5167 innerTail685_5
    (if 3 ≤ innerCandidates685_4.length then search edge 3 [5167, 7741] innerCandidates685_4 12908 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan685_5

def innerTail685_3 : List Nat := 7177 :: innerTail685_4
def innerCandidates685_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates685_3_checked : candidates edge 4 5167 7177 innerTail685_4 best.bound = innerCandidates685_3 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan685_3 : scan edge (search edge 3) 4 [5167] 5167 innerTail685_3 best = best := by
  rw [innerTail685_3, scan, if_neg (by decide), innerCandidates685_3_checked]
  change scan edge (search edge 3) 4 [5167] 5167 innerTail685_4
    (if 3 ≤ innerCandidates685_3.length then search edge 3 [5167, 7177] innerCandidates685_3 12344 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan685_4

def innerTail685_2 : List Nat := 6967 :: innerTail685_3
def innerCandidates685_2 : List Nat := [7177]
set_option maxRecDepth 4096 in
theorem innerCandidates685_2_checked : candidates edge 4 5167 6967 innerTail685_3 best.bound = innerCandidates685_2 := by
  unfold candidates
  have row : edge 6967 = SearchData.row6967.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan685_2 : scan edge (search edge 3) 4 [5167] 5167 innerTail685_2 best = best := by
  rw [innerTail685_2, scan, if_neg (by decide), innerCandidates685_2_checked]
  change scan edge (search edge 3) 4 [5167] 5167 innerTail685_3
    (if 3 ≤ innerCandidates685_2.length then search edge 3 [5167, 6967] innerCandidates685_2 12134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan685_3

def innerTail685_1 : List Nat := 6661 :: innerTail685_2
def innerCandidates685_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates685_1_checked : candidates edge 4 5167 6661 innerTail685_2 best.bound = innerCandidates685_1 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan685_1 : scan edge (search edge 3) 4 [5167] 5167 innerTail685_1 best = best := by
  rw [innerTail685_1, scan, if_neg (by decide), innerCandidates685_1_checked]
  change scan edge (search edge 3) 4 [5167] 5167 innerTail685_2
    (if 3 ≤ innerCandidates685_1.length then search edge 3 [5167, 6661] innerCandidates685_1 11828 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan685_2

def innerTail685_0 : List Nat := 5737 :: innerTail685_1
def innerCandidates685_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates685_0_checked : candidates edge 4 5167 5737 innerTail685_1 best.bound = innerCandidates685_0 := by
  unfold candidates
  have row : edge 5737 = SearchData.row5737.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan685_0 : scan edge (search edge 3) 4 [5167] 5167 innerTail685_0 best = best := by
  rw [innerTail685_0, scan, if_neg (by decide), innerCandidates685_0_checked]
  change scan edge (search edge 3) 4 [5167] 5167 innerTail685_1
    (if 3 ≤ innerCandidates685_0.length then search edge 3 [5167, 5737] innerCandidates685_0 10904 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan685_1

theorem rootBranch685 : search edge 4 [5167] rootCandidates685 5167 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5167] 5167 innerTail685_0 best = best
  exact innerScan685_0

def rootCandidates684 : List Nat := [5849, 5867, 6047, 6101, 6971, 7529, 7589, 8297]
set_option maxRecDepth 4096 in
theorem rootCandidates684_checked : candidates edge 5 0 5153 tail685 best.bound = rootCandidates684 := by
  unfold candidates
  have row : edge 5153 = SearchData.row5153.contains := by funext q; rfl
  rw [row]
  decide

def innerTail684_8 : List Nat := []
theorem innerScan684_8 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_8 best = best := rfl

def innerTail684_7 : List Nat := 8297 :: innerTail684_8
def innerCandidates684_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates684_7_checked : candidates edge 4 5153 8297 innerTail684_8 best.bound = innerCandidates684_7 := by
  unfold candidates
  have row : edge 8297 = SearchData.row8297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_7 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_7 best = best := by
  rw [innerTail684_7, scan, if_neg (by decide), innerCandidates684_7_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_8
    (if 3 ≤ innerCandidates684_7.length then search edge 3 [5153, 8297] innerCandidates684_7 13450 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_8

def innerTail684_6 : List Nat := 7589 :: innerTail684_7
def innerCandidates684_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates684_6_checked : candidates edge 4 5153 7589 innerTail684_7 best.bound = innerCandidates684_6 := by
  unfold candidates
  have row : edge 7589 = SearchData.row7589.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_6 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_6 best = best := by
  rw [innerTail684_6, scan, if_neg (by decide), innerCandidates684_6_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_7
    (if 3 ≤ innerCandidates684_6.length then search edge 3 [5153, 7589] innerCandidates684_6 12742 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_7

def innerTail684_5 : List Nat := 7529 :: innerTail684_6
def innerCandidates684_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates684_5_checked : candidates edge 4 5153 7529 innerTail684_6 best.bound = innerCandidates684_5 := by
  unfold candidates
  have row : edge 7529 = SearchData.row7529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_5 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_5 best = best := by
  rw [innerTail684_5, scan, if_neg (by decide), innerCandidates684_5_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_6
    (if 3 ≤ innerCandidates684_5.length then search edge 3 [5153, 7529] innerCandidates684_5 12682 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_6

def innerTail684_4 : List Nat := 6971 :: innerTail684_5
def innerCandidates684_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates684_4_checked : candidates edge 4 5153 6971 innerTail684_5 best.bound = innerCandidates684_4 := by
  unfold candidates
  have row : edge 6971 = SearchData.row6971.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_4 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_4 best = best := by
  rw [innerTail684_4, scan, if_neg (by decide), innerCandidates684_4_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_5
    (if 3 ≤ innerCandidates684_4.length then search edge 3 [5153, 6971] innerCandidates684_4 12124 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_5

def innerTail684_3 : List Nat := 6101 :: innerTail684_4
def innerCandidates684_3 : List Nat := [7589]
set_option maxRecDepth 4096 in
theorem innerCandidates684_3_checked : candidates edge 4 5153 6101 innerTail684_4 best.bound = innerCandidates684_3 := by
  unfold candidates
  have row : edge 6101 = SearchData.row6101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_3 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_3 best = best := by
  rw [innerTail684_3, scan, if_neg (by decide), innerCandidates684_3_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_4
    (if 3 ≤ innerCandidates684_3.length then search edge 3 [5153, 6101] innerCandidates684_3 11254 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_4

def innerTail684_2 : List Nat := 6047 :: innerTail684_3
def innerCandidates684_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates684_2_checked : candidates edge 4 5153 6047 innerTail684_3 best.bound = innerCandidates684_2 := by
  unfold candidates
  have row : edge 6047 = SearchData.row6047.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_2 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_2 best = best := by
  rw [innerTail684_2, scan, if_neg (by decide), innerCandidates684_2_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_3
    (if 3 ≤ innerCandidates684_2.length then search edge 3 [5153, 6047] innerCandidates684_2 11200 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_3

def innerTail684_1 : List Nat := 5867 :: innerTail684_2
def innerCandidates684_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates684_1_checked : candidates edge 4 5153 5867 innerTail684_2 best.bound = innerCandidates684_1 := by
  unfold candidates
  have row : edge 5867 = SearchData.row5867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_1 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_1 best = best := by
  rw [innerTail684_1, scan, if_neg (by decide), innerCandidates684_1_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_2
    (if 3 ≤ innerCandidates684_1.length then search edge 3 [5153, 5867] innerCandidates684_1 11020 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_2

def innerTail684_0 : List Nat := 5849 :: innerTail684_1
def innerCandidates684_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates684_0_checked : candidates edge 4 5153 5849 innerTail684_1 best.bound = innerCandidates684_0 := by
  unfold candidates
  have row : edge 5849 = SearchData.row5849.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan684_0 : scan edge (search edge 3) 4 [5153] 5153 innerTail684_0 best = best := by
  rw [innerTail684_0, scan, if_neg (by decide), innerCandidates684_0_checked]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_1
    (if 3 ≤ innerCandidates684_0.length then search edge 3 [5153, 5849] innerCandidates684_0 11002 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan684_1

theorem rootBranch684 : search edge 4 [5153] rootCandidates684 5153 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5153] 5153 innerTail684_0 best = best
  exact innerScan684_0

def rootCandidates683 : List Nat := [5231, 5273, 5297, 6857, 6959, 7331, 7487, 7823]
set_option maxRecDepth 4096 in
theorem rootCandidates683_checked : candidates edge 5 0 5147 tail684 best.bound = rootCandidates683 := by
  unfold candidates
  have row : edge 5147 = SearchData.row5147.contains := by funext q; rfl
  rw [row]
  decide

def innerTail683_8 : List Nat := []
theorem innerScan683_8 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_8 best = best := rfl

def innerTail683_7 : List Nat := 7823 :: innerTail683_8
def innerCandidates683_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates683_7_checked : candidates edge 4 5147 7823 innerTail683_8 best.bound = innerCandidates683_7 := by
  unfold candidates
  have row : edge 7823 = SearchData.row7823.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_7 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_7 best = best := by
  rw [innerTail683_7, scan, if_neg (by decide), innerCandidates683_7_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_8
    (if 3 ≤ innerCandidates683_7.length then search edge 3 [5147, 7823] innerCandidates683_7 12970 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_8

def innerTail683_6 : List Nat := 7487 :: innerTail683_7
def innerCandidates683_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates683_6_checked : candidates edge 4 5147 7487 innerTail683_7 best.bound = innerCandidates683_6 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_6 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_6 best = best := by
  rw [innerTail683_6, scan, if_neg (by decide), innerCandidates683_6_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_7
    (if 3 ≤ innerCandidates683_6.length then search edge 3 [5147, 7487] innerCandidates683_6 12634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_7

def innerTail683_5 : List Nat := 7331 :: innerTail683_6
def innerCandidates683_5 : List Nat := [7487]
set_option maxRecDepth 4096 in
theorem innerCandidates683_5_checked : candidates edge 4 5147 7331 innerTail683_6 best.bound = innerCandidates683_5 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_5 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_5 best = best := by
  rw [innerTail683_5, scan, if_neg (by decide), innerCandidates683_5_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_6
    (if 3 ≤ innerCandidates683_5.length then search edge 3 [5147, 7331] innerCandidates683_5 12478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_6

def innerTail683_4 : List Nat := 6959 :: innerTail683_5
def innerCandidates683_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates683_4_checked : candidates edge 4 5147 6959 innerTail683_5 best.bound = innerCandidates683_4 := by
  unfold candidates
  have row : edge 6959 = SearchData.row6959.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_4 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_4 best = best := by
  rw [innerTail683_4, scan, if_neg (by decide), innerCandidates683_4_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_5
    (if 3 ≤ innerCandidates683_4.length then search edge 3 [5147, 6959] innerCandidates683_4 12106 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_5

def innerTail683_3 : List Nat := 6857 :: innerTail683_4
def innerCandidates683_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates683_3_checked : candidates edge 4 5147 6857 innerTail683_4 best.bound = innerCandidates683_3 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_3 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_3 best = best := by
  rw [innerTail683_3, scan, if_neg (by decide), innerCandidates683_3_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_4
    (if 3 ≤ innerCandidates683_3.length then search edge 3 [5147, 6857] innerCandidates683_3 12004 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_4

def innerTail683_2 : List Nat := 5297 :: innerTail683_3
def innerCandidates683_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates683_2_checked : candidates edge 4 5147 5297 innerTail683_3 best.bound = innerCandidates683_2 := by
  unfold candidates
  have row : edge 5297 = SearchData.row5297.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_2 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_2 best = best := by
  rw [innerTail683_2, scan, if_neg (by decide), innerCandidates683_2_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_3
    (if 3 ≤ innerCandidates683_2.length then search edge 3 [5147, 5297] innerCandidates683_2 10444 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_3

def innerTail683_1 : List Nat := 5273 :: innerTail683_2
def innerCandidates683_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates683_1_checked : candidates edge 4 5147 5273 innerTail683_2 best.bound = innerCandidates683_1 := by
  unfold candidates
  have row : edge 5273 = SearchData.row5273.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_1 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_1 best = best := by
  rw [innerTail683_1, scan, if_neg (by decide), innerCandidates683_1_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_2
    (if 3 ≤ innerCandidates683_1.length then search edge 3 [5147, 5273] innerCandidates683_1 10420 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_2

def innerTail683_0 : List Nat := 5231 :: innerTail683_1
def innerCandidates683_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates683_0_checked : candidates edge 4 5147 5231 innerTail683_1 best.bound = innerCandidates683_0 := by
  unfold candidates
  have row : edge 5231 = SearchData.row5231.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan683_0 : scan edge (search edge 3) 4 [5147] 5147 innerTail683_0 best = best := by
  rw [innerTail683_0, scan, if_neg (by decide), innerCandidates683_0_checked]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_1
    (if 3 ≤ innerCandidates683_0.length then search edge 3 [5147, 5231] innerCandidates683_0 10378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan683_1

theorem rootBranch683 : search edge 4 [5147] rootCandidates683 5147 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5147] 5147 innerTail683_0 best = best
  exact innerScan683_0

def rootCandidates682 : List Nat := [5233, 5503, 6073, 6337, 6529, 6637, 7207, 7459, 7537, 7621]
set_option maxRecDepth 4096 in
theorem rootCandidates682_checked : candidates edge 5 0 5119 tail683 best.bound = rootCandidates682 := by
  unfold candidates
  have row : edge 5119 = SearchData.row5119.contains := by funext q; rfl
  rw [row]
  decide

def innerTail682_10 : List Nat := []
theorem innerScan682_10 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_10 best = best := rfl

def innerTail682_9 : List Nat := 7621 :: innerTail682_10
def innerCandidates682_9 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_9_checked : candidates edge 4 5119 7621 innerTail682_10 best.bound = innerCandidates682_9 := by
  unfold candidates
  have row : edge 7621 = SearchData.row7621.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_9 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_9 best = best := by
  rw [innerTail682_9, scan, if_neg (by decide), innerCandidates682_9_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_10
    (if 3 ≤ innerCandidates682_9.length then search edge 3 [5119, 7621] innerCandidates682_9 12740 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_10

def innerTail682_8 : List Nat := 7537 :: innerTail682_9
def innerCandidates682_8 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_8_checked : candidates edge 4 5119 7537 innerTail682_9 best.bound = innerCandidates682_8 := by
  unfold candidates
  have row : edge 7537 = SearchData.row7537.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_8 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_8 best = best := by
  rw [innerTail682_8, scan, if_neg (by decide), innerCandidates682_8_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_9
    (if 3 ≤ innerCandidates682_8.length then search edge 3 [5119, 7537] innerCandidates682_8 12656 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_9

def innerTail682_7 : List Nat := 7459 :: innerTail682_8
def innerCandidates682_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_7_checked : candidates edge 4 5119 7459 innerTail682_8 best.bound = innerCandidates682_7 := by
  unfold candidates
  have row : edge 7459 = SearchData.row7459.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_7 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_7 best = best := by
  rw [innerTail682_7, scan, if_neg (by decide), innerCandidates682_7_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_8
    (if 3 ≤ innerCandidates682_7.length then search edge 3 [5119, 7459] innerCandidates682_7 12578 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_8

def innerTail682_6 : List Nat := 7207 :: innerTail682_7
def innerCandidates682_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_6_checked : candidates edge 4 5119 7207 innerTail682_7 best.bound = innerCandidates682_6 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_6 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_6 best = best := by
  rw [innerTail682_6, scan, if_neg (by decide), innerCandidates682_6_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_7
    (if 3 ≤ innerCandidates682_6.length then search edge 3 [5119, 7207] innerCandidates682_6 12326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_7

def innerTail682_5 : List Nat := 6637 :: innerTail682_6
def innerCandidates682_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_5_checked : candidates edge 4 5119 6637 innerTail682_6 best.bound = innerCandidates682_5 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_5 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_5 best = best := by
  rw [innerTail682_5, scan, if_neg (by decide), innerCandidates682_5_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_6
    (if 3 ≤ innerCandidates682_5.length then search edge 3 [5119, 6637] innerCandidates682_5 11756 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_6

def innerTail682_4 : List Nat := 6529 :: innerTail682_5
def innerCandidates682_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_4_checked : candidates edge 4 5119 6529 innerTail682_5 best.bound = innerCandidates682_4 := by
  unfold candidates
  have row : edge 6529 = SearchData.row6529.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_4 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_4 best = best := by
  rw [innerTail682_4, scan, if_neg (by decide), innerCandidates682_4_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_5
    (if 3 ≤ innerCandidates682_4.length then search edge 3 [5119, 6529] innerCandidates682_4 11648 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_5

def innerTail682_3 : List Nat := 6337 :: innerTail682_4
def innerCandidates682_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_3_checked : candidates edge 4 5119 6337 innerTail682_4 best.bound = innerCandidates682_3 := by
  unfold candidates
  have row : edge 6337 = SearchData.row6337.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_3 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_3 best = best := by
  rw [innerTail682_3, scan, if_neg (by decide), innerCandidates682_3_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_4
    (if 3 ≤ innerCandidates682_3.length then search edge 3 [5119, 6337] innerCandidates682_3 11456 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_4

def innerTail682_2 : List Nat := 6073 :: innerTail682_3
def innerCandidates682_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_2_checked : candidates edge 4 5119 6073 innerTail682_3 best.bound = innerCandidates682_2 := by
  unfold candidates
  have row : edge 6073 = SearchData.row6073.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_2 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_2 best = best := by
  rw [innerTail682_2, scan, if_neg (by decide), innerCandidates682_2_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_3
    (if 3 ≤ innerCandidates682_2.length then search edge 3 [5119, 6073] innerCandidates682_2 11192 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_3

def innerTail682_1 : List Nat := 5503 :: innerTail682_2
def innerCandidates682_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_1_checked : candidates edge 4 5119 5503 innerTail682_2 best.bound = innerCandidates682_1 := by
  unfold candidates
  have row : edge 5503 = SearchData.row5503.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_1 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_1 best = best := by
  rw [innerTail682_1, scan, if_neg (by decide), innerCandidates682_1_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_2
    (if 3 ≤ innerCandidates682_1.length then search edge 3 [5119, 5503] innerCandidates682_1 10622 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_2

def innerTail682_0 : List Nat := 5233 :: innerTail682_1
def innerCandidates682_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates682_0_checked : candidates edge 4 5119 5233 innerTail682_1 best.bound = innerCandidates682_0 := by
  unfold candidates
  have row : edge 5233 = SearchData.row5233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan682_0 : scan edge (search edge 3) 4 [5119] 5119 innerTail682_0 best = best := by
  rw [innerTail682_0, scan, if_neg (by decide), innerCandidates682_0_checked]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_1
    (if 3 ≤ innerCandidates682_0.length then search edge 3 [5119, 5233] innerCandidates682_0 10352 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan682_1

theorem rootBranch682 : search edge 4 [5119] rootCandidates682 5119 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [5119] 5119 innerTail682_0 best = best
  exact innerScan682_0

end ProjectEulerSolutions.PrimePairUpper.Search
