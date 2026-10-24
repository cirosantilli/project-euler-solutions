import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates879 : List Nat := [7121, 7151, 7193, 7829, 7907]
set_option maxRecDepth 4096 in
theorem rootCandidates879_checked : candidates edge 5 0 6857 tail880 best.bound = rootCandidates879 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext q; rfl
  rw [row]
  decide

def innerTail879_5 : List Nat := []
theorem innerScan879_5 : scan edge (search edge 3) 4 [6857] 6857 innerTail879_5 best = best := rfl

def innerTail879_4 : List Nat := 7907 :: innerTail879_5
def innerCandidates879_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates879_4_checked : candidates edge 4 6857 7907 innerTail879_5 best.bound = innerCandidates879_4 := by
  unfold candidates
  have row : edge 7907 = SearchData.row7907.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan879_4 : scan edge (search edge 3) 4 [6857] 6857 innerTail879_4 best = best := by
  rw [innerTail879_4, scan, if_neg (by decide), innerCandidates879_4_checked]
  change scan edge (search edge 3) 4 [6857] 6857 innerTail879_5
    (if 3 ≤ innerCandidates879_4.length then search edge 3 [6857, 7907] innerCandidates879_4 14764 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan879_5

def innerTail879_3 : List Nat := 7829 :: innerTail879_4
def innerCandidates879_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates879_3_checked : candidates edge 4 6857 7829 innerTail879_4 best.bound = innerCandidates879_3 := by
  unfold candidates
  have row : edge 7829 = SearchData.row7829.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan879_3 : scan edge (search edge 3) 4 [6857] 6857 innerTail879_3 best = best := by
  rw [innerTail879_3, scan, if_neg (by decide), innerCandidates879_3_checked]
  change scan edge (search edge 3) 4 [6857] 6857 innerTail879_4
    (if 3 ≤ innerCandidates879_3.length then search edge 3 [6857, 7829] innerCandidates879_3 14686 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan879_4

def innerTail879_2 : List Nat := 7193 :: innerTail879_3
def innerCandidates879_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates879_2_checked : candidates edge 4 6857 7193 innerTail879_3 best.bound = innerCandidates879_2 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan879_2 : scan edge (search edge 3) 4 [6857] 6857 innerTail879_2 best = best := by
  rw [innerTail879_2, scan, if_neg (by decide), innerCandidates879_2_checked]
  change scan edge (search edge 3) 4 [6857] 6857 innerTail879_3
    (if 3 ≤ innerCandidates879_2.length then search edge 3 [6857, 7193] innerCandidates879_2 14050 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan879_3

def innerTail879_1 : List Nat := 7151 :: innerTail879_2
def innerCandidates879_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates879_1_checked : candidates edge 4 6857 7151 innerTail879_2 best.bound = innerCandidates879_1 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan879_1 : scan edge (search edge 3) 4 [6857] 6857 innerTail879_1 best = best := by
  rw [innerTail879_1, scan, if_neg (by decide), innerCandidates879_1_checked]
  change scan edge (search edge 3) 4 [6857] 6857 innerTail879_2
    (if 3 ≤ innerCandidates879_1.length then search edge 3 [6857, 7151] innerCandidates879_1 14008 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan879_2

def innerTail879_0 : List Nat := 7121 :: innerTail879_1
def innerCandidates879_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates879_0_checked : candidates edge 4 6857 7121 innerTail879_1 best.bound = innerCandidates879_0 := by
  unfold candidates
  have row : edge 7121 = SearchData.row7121.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan879_0 : scan edge (search edge 3) 4 [6857] 6857 innerTail879_0 best = best := by
  rw [innerTail879_0, scan, if_neg (by decide), innerCandidates879_0_checked]
  change scan edge (search edge 3) 4 [6857] 6857 innerTail879_1
    (if 3 ≤ innerCandidates879_0.length then search edge 3 [6857, 7121] innerCandidates879_0 13978 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan879_1

theorem rootBranch879 : search edge 4 [6857] rootCandidates879 6857 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6857] 6857 innerTail879_0 best = best
  exact innerScan879_0

def rootCandidates878 : List Nat := [6991, 7039, 8377]
set_option maxRecDepth 4096 in
theorem rootCandidates878_checked : candidates edge 5 0 6841 tail879 best.bound = rootCandidates878 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates877 : List Nat := [6869, 6899, 7487, 7673, 8087]
set_option maxRecDepth 4096 in
theorem rootCandidates877_checked : candidates edge 5 0 6833 tail878 best.bound = rootCandidates877 := by
  unfold candidates
  have row : edge 6833 = SearchData.row6833.contains := by funext q; rfl
  rw [row]
  decide

def innerTail877_5 : List Nat := []
theorem innerScan877_5 : scan edge (search edge 3) 4 [6833] 6833 innerTail877_5 best = best := rfl

def innerTail877_4 : List Nat := 8087 :: innerTail877_5
def innerCandidates877_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates877_4_checked : candidates edge 4 6833 8087 innerTail877_5 best.bound = innerCandidates877_4 := by
  unfold candidates
  have row : edge 8087 = SearchData.row8087.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan877_4 : scan edge (search edge 3) 4 [6833] 6833 innerTail877_4 best = best := by
  rw [innerTail877_4, scan, if_neg (by decide), innerCandidates877_4_checked]
  change scan edge (search edge 3) 4 [6833] 6833 innerTail877_5
    (if 3 ≤ innerCandidates877_4.length then search edge 3 [6833, 8087] innerCandidates877_4 14920 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan877_5

def innerTail877_3 : List Nat := 7673 :: innerTail877_4
def innerCandidates877_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates877_3_checked : candidates edge 4 6833 7673 innerTail877_4 best.bound = innerCandidates877_3 := by
  unfold candidates
  have row : edge 7673 = SearchData.row7673.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan877_3 : scan edge (search edge 3) 4 [6833] 6833 innerTail877_3 best = best := by
  rw [innerTail877_3, scan, if_neg (by decide), innerCandidates877_3_checked]
  change scan edge (search edge 3) 4 [6833] 6833 innerTail877_4
    (if 3 ≤ innerCandidates877_3.length then search edge 3 [6833, 7673] innerCandidates877_3 14506 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan877_4

def innerTail877_2 : List Nat := 7487 :: innerTail877_3
def innerCandidates877_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates877_2_checked : candidates edge 4 6833 7487 innerTail877_3 best.bound = innerCandidates877_2 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan877_2 : scan edge (search edge 3) 4 [6833] 6833 innerTail877_2 best = best := by
  rw [innerTail877_2, scan, if_neg (by decide), innerCandidates877_2_checked]
  change scan edge (search edge 3) 4 [6833] 6833 innerTail877_3
    (if 3 ≤ innerCandidates877_2.length then search edge 3 [6833, 7487] innerCandidates877_2 14320 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan877_3

def innerTail877_1 : List Nat := 6899 :: innerTail877_2
def innerCandidates877_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates877_1_checked : candidates edge 4 6833 6899 innerTail877_2 best.bound = innerCandidates877_1 := by
  unfold candidates
  have row : edge 6899 = SearchData.row6899.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan877_1 : scan edge (search edge 3) 4 [6833] 6833 innerTail877_1 best = best := by
  rw [innerTail877_1, scan, if_neg (by decide), innerCandidates877_1_checked]
  change scan edge (search edge 3) 4 [6833] 6833 innerTail877_2
    (if 3 ≤ innerCandidates877_1.length then search edge 3 [6833, 6899] innerCandidates877_1 13732 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan877_2

def innerTail877_0 : List Nat := 6869 :: innerTail877_1
def innerCandidates877_0 : List Nat := [8087]
set_option maxRecDepth 4096 in
theorem innerCandidates877_0_checked : candidates edge 4 6833 6869 innerTail877_1 best.bound = innerCandidates877_0 := by
  unfold candidates
  have row : edge 6869 = SearchData.row6869.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan877_0 : scan edge (search edge 3) 4 [6833] 6833 innerTail877_0 best = best := by
  rw [innerTail877_0, scan, if_neg (by decide), innerCandidates877_0_checked]
  change scan edge (search edge 3) 4 [6833] 6833 innerTail877_1
    (if 3 ≤ innerCandidates877_0.length then search edge 3 [6833, 6869] innerCandidates877_0 13702 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan877_1

theorem rootBranch877 : search edge 4 [6833] rootCandidates877 6833 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6833] 6833 innerTail877_0 best = best
  exact innerScan877_0

def rootCandidates876 : List Nat := [7057, 7213, 7219, 7507, 8317]
set_option maxRecDepth 4096 in
theorem rootCandidates876_checked : candidates edge 5 0 6829 tail877 best.bound = rootCandidates876 := by
  unfold candidates
  have row : edge 6829 = SearchData.row6829.contains := by funext q; rfl
  rw [row]
  decide

def innerTail876_5 : List Nat := []
theorem innerScan876_5 : scan edge (search edge 3) 4 [6829] 6829 innerTail876_5 best = best := rfl

def innerTail876_4 : List Nat := 8317 :: innerTail876_5
def innerCandidates876_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates876_4_checked : candidates edge 4 6829 8317 innerTail876_5 best.bound = innerCandidates876_4 := by
  unfold candidates
  have row : edge 8317 = SearchData.row8317.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan876_4 : scan edge (search edge 3) 4 [6829] 6829 innerTail876_4 best = best := by
  rw [innerTail876_4, scan, if_neg (by decide), innerCandidates876_4_checked]
  change scan edge (search edge 3) 4 [6829] 6829 innerTail876_5
    (if 3 ≤ innerCandidates876_4.length then search edge 3 [6829, 8317] innerCandidates876_4 15146 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan876_5

def innerTail876_3 : List Nat := 7507 :: innerTail876_4
def innerCandidates876_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates876_3_checked : candidates edge 4 6829 7507 innerTail876_4 best.bound = innerCandidates876_3 := by
  unfold candidates
  have row : edge 7507 = SearchData.row7507.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan876_3 : scan edge (search edge 3) 4 [6829] 6829 innerTail876_3 best = best := by
  rw [innerTail876_3, scan, if_neg (by decide), innerCandidates876_3_checked]
  change scan edge (search edge 3) 4 [6829] 6829 innerTail876_4
    (if 3 ≤ innerCandidates876_3.length then search edge 3 [6829, 7507] innerCandidates876_3 14336 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan876_4

def innerTail876_2 : List Nat := 7219 :: innerTail876_3
def innerCandidates876_2 : List Nat := [8317]
set_option maxRecDepth 4096 in
theorem innerCandidates876_2_checked : candidates edge 4 6829 7219 innerTail876_3 best.bound = innerCandidates876_2 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan876_2 : scan edge (search edge 3) 4 [6829] 6829 innerTail876_2 best = best := by
  rw [innerTail876_2, scan, if_neg (by decide), innerCandidates876_2_checked]
  change scan edge (search edge 3) 4 [6829] 6829 innerTail876_3
    (if 3 ≤ innerCandidates876_2.length then search edge 3 [6829, 7219] innerCandidates876_2 14048 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan876_3

def innerTail876_1 : List Nat := 7213 :: innerTail876_2
def innerCandidates876_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates876_1_checked : candidates edge 4 6829 7213 innerTail876_2 best.bound = innerCandidates876_1 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan876_1 : scan edge (search edge 3) 4 [6829] 6829 innerTail876_1 best = best := by
  rw [innerTail876_1, scan, if_neg (by decide), innerCandidates876_1_checked]
  change scan edge (search edge 3) 4 [6829] 6829 innerTail876_2
    (if 3 ≤ innerCandidates876_1.length then search edge 3 [6829, 7213] innerCandidates876_1 14042 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan876_2

def innerTail876_0 : List Nat := 7057 :: innerTail876_1
def innerCandidates876_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates876_0_checked : candidates edge 4 6829 7057 innerTail876_1 best.bound = innerCandidates876_0 := by
  unfold candidates
  have row : edge 7057 = SearchData.row7057.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan876_0 : scan edge (search edge 3) 4 [6829] 6829 innerTail876_0 best = best := by
  rw [innerTail876_0, scan, if_neg (by decide), innerCandidates876_0_checked]
  change scan edge (search edge 3) 4 [6829] 6829 innerTail876_1
    (if 3 ≤ innerCandidates876_0.length then search edge 3 [6829, 7057] innerCandidates876_0 13886 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan876_1

theorem rootBranch876 : search edge 4 [6829] rootCandidates876 6829 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6829] 6829 innerTail876_0 best = best
  exact innerScan876_0

def rootCandidates875 : List Nat := [6977, 7103, 7487, 7499, 7817, 8369]
set_option maxRecDepth 4096 in
theorem rootCandidates875_checked : candidates edge 5 0 6827 tail876 best.bound = rootCandidates875 := by
  unfold candidates
  have row : edge 6827 = SearchData.row6827.contains := by funext q; rfl
  rw [row]
  decide

def innerTail875_6 : List Nat := []
theorem innerScan875_6 : scan edge (search edge 3) 4 [6827] 6827 innerTail875_6 best = best := rfl

def innerTail875_5 : List Nat := 8369 :: innerTail875_6
def innerCandidates875_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates875_5_checked : candidates edge 4 6827 8369 innerTail875_6 best.bound = innerCandidates875_5 := by
  unfold candidates
  have row : edge 8369 = SearchData.row8369.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan875_5 : scan edge (search edge 3) 4 [6827] 6827 innerTail875_5 best = best := by
  rw [innerTail875_5, scan, if_neg (by decide), innerCandidates875_5_checked]
  change scan edge (search edge 3) 4 [6827] 6827 innerTail875_6
    (if 3 ≤ innerCandidates875_5.length then search edge 3 [6827, 8369] innerCandidates875_5 15196 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan875_6

def innerTail875_4 : List Nat := 7817 :: innerTail875_5
def innerCandidates875_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates875_4_checked : candidates edge 4 6827 7817 innerTail875_5 best.bound = innerCandidates875_4 := by
  unfold candidates
  have row : edge 7817 = SearchData.row7817.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan875_4 : scan edge (search edge 3) 4 [6827] 6827 innerTail875_4 best = best := by
  rw [innerTail875_4, scan, if_neg (by decide), innerCandidates875_4_checked]
  change scan edge (search edge 3) 4 [6827] 6827 innerTail875_5
    (if 3 ≤ innerCandidates875_4.length then search edge 3 [6827, 7817] innerCandidates875_4 14644 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan875_5

def innerTail875_3 : List Nat := 7499 :: innerTail875_4
def innerCandidates875_3 : List Nat := [8369]
set_option maxRecDepth 4096 in
theorem innerCandidates875_3_checked : candidates edge 4 6827 7499 innerTail875_4 best.bound = innerCandidates875_3 := by
  unfold candidates
  have row : edge 7499 = SearchData.row7499.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan875_3 : scan edge (search edge 3) 4 [6827] 6827 innerTail875_3 best = best := by
  rw [innerTail875_3, scan, if_neg (by decide), innerCandidates875_3_checked]
  change scan edge (search edge 3) 4 [6827] 6827 innerTail875_4
    (if 3 ≤ innerCandidates875_3.length then search edge 3 [6827, 7499] innerCandidates875_3 14326 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan875_4

def innerTail875_2 : List Nat := 7487 :: innerTail875_3
def innerCandidates875_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates875_2_checked : candidates edge 4 6827 7487 innerTail875_3 best.bound = innerCandidates875_2 := by
  unfold candidates
  have row : edge 7487 = SearchData.row7487.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan875_2 : scan edge (search edge 3) 4 [6827] 6827 innerTail875_2 best = best := by
  rw [innerTail875_2, scan, if_neg (by decide), innerCandidates875_2_checked]
  change scan edge (search edge 3) 4 [6827] 6827 innerTail875_3
    (if 3 ≤ innerCandidates875_2.length then search edge 3 [6827, 7487] innerCandidates875_2 14314 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan875_3

def innerTail875_1 : List Nat := 7103 :: innerTail875_2
def innerCandidates875_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates875_1_checked : candidates edge 4 6827 7103 innerTail875_2 best.bound = innerCandidates875_1 := by
  unfold candidates
  have row : edge 7103 = SearchData.row7103.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan875_1 : scan edge (search edge 3) 4 [6827] 6827 innerTail875_1 best = best := by
  rw [innerTail875_1, scan, if_neg (by decide), innerCandidates875_1_checked]
  change scan edge (search edge 3) 4 [6827] 6827 innerTail875_2
    (if 3 ≤ innerCandidates875_1.length then search edge 3 [6827, 7103] innerCandidates875_1 13930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan875_2

def innerTail875_0 : List Nat := 6977 :: innerTail875_1
def innerCandidates875_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates875_0_checked : candidates edge 4 6827 6977 innerTail875_1 best.bound = innerCandidates875_0 := by
  unfold candidates
  have row : edge 6977 = SearchData.row6977.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan875_0 : scan edge (search edge 3) 4 [6827] 6827 innerTail875_0 best = best := by
  rw [innerTail875_0, scan, if_neg (by decide), innerCandidates875_0_checked]
  change scan edge (search edge 3) 4 [6827] 6827 innerTail875_1
    (if 3 ≤ innerCandidates875_0.length then search edge 3 [6827, 6977] innerCandidates875_0 13804 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan875_1

theorem rootBranch875 : search edge 4 [6827] rootCandidates875 6827 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6827] 6827 innerTail875_0 best = best
  exact innerScan875_0

def rootCandidates874 : List Nat := [7867, 7951, 7993, 8059, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates874_checked : candidates edge 5 0 6823 tail875 best.bound = rootCandidates874 := by
  unfold candidates
  have row : edge 6823 = SearchData.row6823.contains := by funext q; rfl
  rw [row]
  decide

def innerTail874_5 : List Nat := []
theorem innerScan874_5 : scan edge (search edge 3) 4 [6823] 6823 innerTail874_5 best = best := rfl

def innerTail874_4 : List Nat := 8311 :: innerTail874_5
def innerCandidates874_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates874_4_checked : candidates edge 4 6823 8311 innerTail874_5 best.bound = innerCandidates874_4 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan874_4 : scan edge (search edge 3) 4 [6823] 6823 innerTail874_4 best = best := by
  rw [innerTail874_4, scan, if_neg (by decide), innerCandidates874_4_checked]
  change scan edge (search edge 3) 4 [6823] 6823 innerTail874_5
    (if 3 ≤ innerCandidates874_4.length then search edge 3 [6823, 8311] innerCandidates874_4 15134 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan874_5

def innerTail874_3 : List Nat := 8059 :: innerTail874_4
def innerCandidates874_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates874_3_checked : candidates edge 4 6823 8059 innerTail874_4 best.bound = innerCandidates874_3 := by
  unfold candidates
  have row : edge 8059 = SearchData.row8059.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan874_3 : scan edge (search edge 3) 4 [6823] 6823 innerTail874_3 best = best := by
  rw [innerTail874_3, scan, if_neg (by decide), innerCandidates874_3_checked]
  change scan edge (search edge 3) 4 [6823] 6823 innerTail874_4
    (if 3 ≤ innerCandidates874_3.length then search edge 3 [6823, 8059] innerCandidates874_3 14882 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan874_4

def innerTail874_2 : List Nat := 7993 :: innerTail874_3
def innerCandidates874_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates874_2_checked : candidates edge 4 6823 7993 innerTail874_3 best.bound = innerCandidates874_2 := by
  unfold candidates
  have row : edge 7993 = SearchData.row7993.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan874_2 : scan edge (search edge 3) 4 [6823] 6823 innerTail874_2 best = best := by
  rw [innerTail874_2, scan, if_neg (by decide), innerCandidates874_2_checked]
  change scan edge (search edge 3) 4 [6823] 6823 innerTail874_3
    (if 3 ≤ innerCandidates874_2.length then search edge 3 [6823, 7993] innerCandidates874_2 14816 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan874_3

def innerTail874_1 : List Nat := 7951 :: innerTail874_2
def innerCandidates874_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates874_1_checked : candidates edge 4 6823 7951 innerTail874_2 best.bound = innerCandidates874_1 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan874_1 : scan edge (search edge 3) 4 [6823] 6823 innerTail874_1 best = best := by
  rw [innerTail874_1, scan, if_neg (by decide), innerCandidates874_1_checked]
  change scan edge (search edge 3) 4 [6823] 6823 innerTail874_2
    (if 3 ≤ innerCandidates874_1.length then search edge 3 [6823, 7951] innerCandidates874_1 14774 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan874_2

def innerTail874_0 : List Nat := 7867 :: innerTail874_1
def innerCandidates874_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates874_0_checked : candidates edge 4 6823 7867 innerTail874_1 best.bound = innerCandidates874_0 := by
  unfold candidates
  have row : edge 7867 = SearchData.row7867.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan874_0 : scan edge (search edge 3) 4 [6823] 6823 innerTail874_0 best = best := by
  rw [innerTail874_0, scan, if_neg (by decide), innerCandidates874_0_checked]
  change scan edge (search edge 3) 4 [6823] 6823 innerTail874_1
    (if 3 ≤ innerCandidates874_0.length then search edge 3 [6823, 7867] innerCandidates874_0 14690 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan874_1

theorem rootBranch874 : search edge 4 [6823] rootCandidates874 6823 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6823] 6823 innerTail874_0 best = best
  exact innerScan874_0

def rootCandidates873 : List Nat := [6977, 7649, 8123]
set_option maxRecDepth 4096 in
theorem rootCandidates873_checked : candidates edge 5 0 6803 tail874 best.bound = rootCandidates873 := by
  unfold candidates
  have row : edge 6803 = SearchData.row6803.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates872 : List Nat := [7039, 7159, 7717, 7951, 8293]
set_option maxRecDepth 4096 in
theorem rootCandidates872_checked : candidates edge 5 0 6793 tail873 best.bound = rootCandidates872 := by
  unfold candidates
  have row : edge 6793 = SearchData.row6793.contains := by funext q; rfl
  rw [row]
  decide

def innerTail872_5 : List Nat := []
theorem innerScan872_5 : scan edge (search edge 3) 4 [6793] 6793 innerTail872_5 best = best := rfl

def innerTail872_4 : List Nat := 8293 :: innerTail872_5
def innerCandidates872_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates872_4_checked : candidates edge 4 6793 8293 innerTail872_5 best.bound = innerCandidates872_4 := by
  unfold candidates
  have row : edge 8293 = SearchData.row8293.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan872_4 : scan edge (search edge 3) 4 [6793] 6793 innerTail872_4 best = best := by
  rw [innerTail872_4, scan, if_neg (by decide), innerCandidates872_4_checked]
  change scan edge (search edge 3) 4 [6793] 6793 innerTail872_5
    (if 3 ≤ innerCandidates872_4.length then search edge 3 [6793, 8293] innerCandidates872_4 15086 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan872_5

def innerTail872_3 : List Nat := 7951 :: innerTail872_4
def innerCandidates872_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates872_3_checked : candidates edge 4 6793 7951 innerTail872_4 best.bound = innerCandidates872_3 := by
  unfold candidates
  have row : edge 7951 = SearchData.row7951.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan872_3 : scan edge (search edge 3) 4 [6793] 6793 innerTail872_3 best = best := by
  rw [innerTail872_3, scan, if_neg (by decide), innerCandidates872_3_checked]
  change scan edge (search edge 3) 4 [6793] 6793 innerTail872_4
    (if 3 ≤ innerCandidates872_3.length then search edge 3 [6793, 7951] innerCandidates872_3 14744 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan872_4

def innerTail872_2 : List Nat := 7717 :: innerTail872_3
def innerCandidates872_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates872_2_checked : candidates edge 4 6793 7717 innerTail872_3 best.bound = innerCandidates872_2 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan872_2 : scan edge (search edge 3) 4 [6793] 6793 innerTail872_2 best = best := by
  rw [innerTail872_2, scan, if_neg (by decide), innerCandidates872_2_checked]
  change scan edge (search edge 3) 4 [6793] 6793 innerTail872_3
    (if 3 ≤ innerCandidates872_2.length then search edge 3 [6793, 7717] innerCandidates872_2 14510 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan872_3

def innerTail872_1 : List Nat := 7159 :: innerTail872_2
def innerCandidates872_1 : List Nat := [7717]
set_option maxRecDepth 4096 in
theorem innerCandidates872_1_checked : candidates edge 4 6793 7159 innerTail872_2 best.bound = innerCandidates872_1 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan872_1 : scan edge (search edge 3) 4 [6793] 6793 innerTail872_1 best = best := by
  rw [innerTail872_1, scan, if_neg (by decide), innerCandidates872_1_checked]
  change scan edge (search edge 3) 4 [6793] 6793 innerTail872_2
    (if 3 ≤ innerCandidates872_1.length then search edge 3 [6793, 7159] innerCandidates872_1 13952 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan872_2

def innerTail872_0 : List Nat := 7039 :: innerTail872_1
def innerCandidates872_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates872_0_checked : candidates edge 4 6793 7039 innerTail872_1 best.bound = innerCandidates872_0 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan872_0 : scan edge (search edge 3) 4 [6793] 6793 innerTail872_0 best = best := by
  rw [innerTail872_0, scan, if_neg (by decide), innerCandidates872_0_checked]
  change scan edge (search edge 3) 4 [6793] 6793 innerTail872_1
    (if 3 ≤ innerCandidates872_0.length then search edge 3 [6793, 7039] innerCandidates872_0 13832 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan872_1

theorem rootBranch872 : search edge 4 [6793] rootCandidates872 6793 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6793] 6793 innerTail872_0 best = best
  exact innerScan872_0

end ProjectEulerSolutions.PrimePairUpper.Search
