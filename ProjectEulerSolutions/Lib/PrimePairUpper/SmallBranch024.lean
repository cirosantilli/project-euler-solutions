import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates855 : List Nat := [7043]
set_option maxRecDepth 4096 in
theorem rootCandidates855_checked : candidates edge 5 0 6659 tail856 best.bound = rootCandidates855 := by
  unfold candidates
  have row : edge 6659 = SearchData.row6659.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates854 : List Nat := [7103, 7649, 8363]
set_option maxRecDepth 4096 in
theorem rootCandidates854_checked : candidates edge 5 0 6653 tail855 best.bound = rootCandidates854 := by
  unfold candidates
  have row : edge 6653 = SearchData.row6653.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates853 : List Nat := [6841, 6997, 7039, 7309, 7351, 8233]
set_option maxRecDepth 4096 in
theorem rootCandidates853_checked : candidates edge 5 0 6637 tail854 best.bound = rootCandidates853 := by
  unfold candidates
  have row : edge 6637 = SearchData.row6637.contains := by funext q; rfl
  rw [row]
  decide

def innerTail853_6 : List Nat := []
theorem innerScan853_6 : scan edge (search edge 3) 4 [6637] 6637 innerTail853_6 best = best := rfl

def innerTail853_5 : List Nat := 8233 :: innerTail853_6
def innerCandidates853_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates853_5_checked : candidates edge 4 6637 8233 innerTail853_6 best.bound = innerCandidates853_5 := by
  unfold candidates
  have row : edge 8233 = SearchData.row8233.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan853_5 : scan edge (search edge 3) 4 [6637] 6637 innerTail853_5 best = best := by
  rw [innerTail853_5, scan, if_neg (by decide), innerCandidates853_5_checked]
  change scan edge (search edge 3) 4 [6637] 6637 innerTail853_6
    (if 3 ≤ innerCandidates853_5.length then search edge 3 [6637, 8233] innerCandidates853_5 14870 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan853_6

def innerTail853_4 : List Nat := 7351 :: innerTail853_5
def innerCandidates853_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates853_4_checked : candidates edge 4 6637 7351 innerTail853_5 best.bound = innerCandidates853_4 := by
  unfold candidates
  have row : edge 7351 = SearchData.row7351.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan853_4 : scan edge (search edge 3) 4 [6637] 6637 innerTail853_4 best = best := by
  rw [innerTail853_4, scan, if_neg (by decide), innerCandidates853_4_checked]
  change scan edge (search edge 3) 4 [6637] 6637 innerTail853_5
    (if 3 ≤ innerCandidates853_4.length then search edge 3 [6637, 7351] innerCandidates853_4 13988 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan853_5

def innerTail853_3 : List Nat := 7309 :: innerTail853_4
def innerCandidates853_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates853_3_checked : candidates edge 4 6637 7309 innerTail853_4 best.bound = innerCandidates853_3 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan853_3 : scan edge (search edge 3) 4 [6637] 6637 innerTail853_3 best = best := by
  rw [innerTail853_3, scan, if_neg (by decide), innerCandidates853_3_checked]
  change scan edge (search edge 3) 4 [6637] 6637 innerTail853_4
    (if 3 ≤ innerCandidates853_3.length then search edge 3 [6637, 7309] innerCandidates853_3 13946 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan853_4

def innerTail853_2 : List Nat := 7039 :: innerTail853_3
def innerCandidates853_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates853_2_checked : candidates edge 4 6637 7039 innerTail853_3 best.bound = innerCandidates853_2 := by
  unfold candidates
  have row : edge 7039 = SearchData.row7039.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan853_2 : scan edge (search edge 3) 4 [6637] 6637 innerTail853_2 best = best := by
  rw [innerTail853_2, scan, if_neg (by decide), innerCandidates853_2_checked]
  change scan edge (search edge 3) 4 [6637] 6637 innerTail853_3
    (if 3 ≤ innerCandidates853_2.length then search edge 3 [6637, 7039] innerCandidates853_2 13676 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan853_3

def innerTail853_1 : List Nat := 6997 :: innerTail853_2
def innerCandidates853_1 : List Nat := [7351]
set_option maxRecDepth 4096 in
theorem innerCandidates853_1_checked : candidates edge 4 6637 6997 innerTail853_2 best.bound = innerCandidates853_1 := by
  unfold candidates
  have row : edge 6997 = SearchData.row6997.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan853_1 : scan edge (search edge 3) 4 [6637] 6637 innerTail853_1 best = best := by
  rw [innerTail853_1, scan, if_neg (by decide), innerCandidates853_1_checked]
  change scan edge (search edge 3) 4 [6637] 6637 innerTail853_2
    (if 3 ≤ innerCandidates853_1.length then search edge 3 [6637, 6997] innerCandidates853_1 13634 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan853_2

def innerTail853_0 : List Nat := 6841 :: innerTail853_1
def innerCandidates853_0 : List Nat := [7039]
set_option maxRecDepth 4096 in
theorem innerCandidates853_0_checked : candidates edge 4 6637 6841 innerTail853_1 best.bound = innerCandidates853_0 := by
  unfold candidates
  have row : edge 6841 = SearchData.row6841.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan853_0 : scan edge (search edge 3) 4 [6637] 6637 innerTail853_0 best = best := by
  rw [innerTail853_0, scan, if_neg (by decide), innerCandidates853_0_checked]
  change scan edge (search edge 3) 4 [6637] 6637 innerTail853_1
    (if 3 ≤ innerCandidates853_0.length then search edge 3 [6637, 6841] innerCandidates853_0 13478 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan853_1

theorem rootBranch853 : search edge 4 [6637] rootCandidates853 6637 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6637] 6637 innerTail853_0 best = best
  exact innerScan853_0

def rootCandidates852 : List Nat := [7591, 7603, 8101, 8179, 8311]
set_option maxRecDepth 4096 in
theorem rootCandidates852_checked : candidates edge 5 0 6619 tail853 best.bound = rootCandidates852 := by
  unfold candidates
  have row : edge 6619 = SearchData.row6619.contains := by funext q; rfl
  rw [row]
  decide

def innerTail852_5 : List Nat := []
theorem innerScan852_5 : scan edge (search edge 3) 4 [6619] 6619 innerTail852_5 best = best := rfl

def innerTail852_4 : List Nat := 8311 :: innerTail852_5
def innerCandidates852_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates852_4_checked : candidates edge 4 6619 8311 innerTail852_5 best.bound = innerCandidates852_4 := by
  unfold candidates
  have row : edge 8311 = SearchData.row8311.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan852_4 : scan edge (search edge 3) 4 [6619] 6619 innerTail852_4 best = best := by
  rw [innerTail852_4, scan, if_neg (by decide), innerCandidates852_4_checked]
  change scan edge (search edge 3) 4 [6619] 6619 innerTail852_5
    (if 3 ≤ innerCandidates852_4.length then search edge 3 [6619, 8311] innerCandidates852_4 14930 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan852_5

def innerTail852_3 : List Nat := 8179 :: innerTail852_4
def innerCandidates852_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates852_3_checked : candidates edge 4 6619 8179 innerTail852_4 best.bound = innerCandidates852_3 := by
  unfold candidates
  have row : edge 8179 = SearchData.row8179.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan852_3 : scan edge (search edge 3) 4 [6619] 6619 innerTail852_3 best = best := by
  rw [innerTail852_3, scan, if_neg (by decide), innerCandidates852_3_checked]
  change scan edge (search edge 3) 4 [6619] 6619 innerTail852_4
    (if 3 ≤ innerCandidates852_3.length then search edge 3 [6619, 8179] innerCandidates852_3 14798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan852_4

def innerTail852_2 : List Nat := 8101 :: innerTail852_3
def innerCandidates852_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates852_2_checked : candidates edge 4 6619 8101 innerTail852_3 best.bound = innerCandidates852_2 := by
  unfold candidates
  have row : edge 8101 = SearchData.row8101.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan852_2 : scan edge (search edge 3) 4 [6619] 6619 innerTail852_2 best = best := by
  rw [innerTail852_2, scan, if_neg (by decide), innerCandidates852_2_checked]
  change scan edge (search edge 3) 4 [6619] 6619 innerTail852_3
    (if 3 ≤ innerCandidates852_2.length then search edge 3 [6619, 8101] innerCandidates852_2 14720 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan852_3

def innerTail852_1 : List Nat := 7603 :: innerTail852_2
def innerCandidates852_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates852_1_checked : candidates edge 4 6619 7603 innerTail852_2 best.bound = innerCandidates852_1 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan852_1 : scan edge (search edge 3) 4 [6619] 6619 innerTail852_1 best = best := by
  rw [innerTail852_1, scan, if_neg (by decide), innerCandidates852_1_checked]
  change scan edge (search edge 3) 4 [6619] 6619 innerTail852_2
    (if 3 ≤ innerCandidates852_1.length then search edge 3 [6619, 7603] innerCandidates852_1 14222 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan852_2

def innerTail852_0 : List Nat := 7591 :: innerTail852_1
def innerCandidates852_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates852_0_checked : candidates edge 4 6619 7591 innerTail852_1 best.bound = innerCandidates852_0 := by
  unfold candidates
  have row : edge 7591 = SearchData.row7591.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan852_0 : scan edge (search edge 3) 4 [6619] 6619 innerTail852_0 best = best := by
  rw [innerTail852_0, scan, if_neg (by decide), innerCandidates852_0_checked]
  change scan edge (search edge 3) 4 [6619] 6619 innerTail852_1
    (if 3 ≤ innerCandidates852_0.length then search edge 3 [6619, 7591] innerCandidates852_0 14210 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan852_1

theorem rootBranch852 : search edge 4 [6619] rootCandidates852 6619 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6619] 6619 innerTail852_0 best = best
  exact innerScan852_0

def rootCandidates851 : List Nat := [6763, 7867]
set_option maxRecDepth 4096 in
theorem rootCandidates851_checked : candidates edge 5 0 6607 tail852 best.bound = rootCandidates851 := by
  unfold candidates
  have row : edge 6607 = SearchData.row6607.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates850 : List Nat := [7499, 8081, 8111]
set_option maxRecDepth 4096 in
theorem rootCandidates850_checked : candidates edge 5 0 6599 tail851 best.bound = rootCandidates850 := by
  unfold candidates
  have row : edge 6599 = SearchData.row6599.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates849 : List Nat := [6857, 7541, 7883, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates849_checked : candidates edge 5 0 6581 tail850 best.bound = rootCandidates849 := by
  unfold candidates
  have row : edge 6581 = SearchData.row6581.contains := by funext q; rfl
  rw [row]
  decide

def innerTail849_4 : List Nat := []
theorem innerScan849_4 : scan edge (search edge 3) 4 [6581] 6581 innerTail849_4 best = best := rfl

def innerTail849_3 : List Nat := 8171 :: innerTail849_4
def innerCandidates849_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates849_3_checked : candidates edge 4 6581 8171 innerTail849_4 best.bound = innerCandidates849_3 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan849_3 : scan edge (search edge 3) 4 [6581] 6581 innerTail849_3 best = best := by
  rw [innerTail849_3, scan, if_neg (by decide), innerCandidates849_3_checked]
  change scan edge (search edge 3) 4 [6581] 6581 innerTail849_4
    (if 3 ≤ innerCandidates849_3.length then search edge 3 [6581, 8171] innerCandidates849_3 14752 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan849_4

def innerTail849_2 : List Nat := 7883 :: innerTail849_3
def innerCandidates849_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates849_2_checked : candidates edge 4 6581 7883 innerTail849_3 best.bound = innerCandidates849_2 := by
  unfold candidates
  have row : edge 7883 = SearchData.row7883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan849_2 : scan edge (search edge 3) 4 [6581] 6581 innerTail849_2 best = best := by
  rw [innerTail849_2, scan, if_neg (by decide), innerCandidates849_2_checked]
  change scan edge (search edge 3) 4 [6581] 6581 innerTail849_3
    (if 3 ≤ innerCandidates849_2.length then search edge 3 [6581, 7883] innerCandidates849_2 14464 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan849_3

def innerTail849_1 : List Nat := 7541 :: innerTail849_2
def innerCandidates849_1 : List Nat := [7883]
set_option maxRecDepth 4096 in
theorem innerCandidates849_1_checked : candidates edge 4 6581 7541 innerTail849_2 best.bound = innerCandidates849_1 := by
  unfold candidates
  have row : edge 7541 = SearchData.row7541.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan849_1 : scan edge (search edge 3) 4 [6581] 6581 innerTail849_1 best = best := by
  rw [innerTail849_1, scan, if_neg (by decide), innerCandidates849_1_checked]
  change scan edge (search edge 3) 4 [6581] 6581 innerTail849_2
    (if 3 ≤ innerCandidates849_1.length then search edge 3 [6581, 7541] innerCandidates849_1 14122 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan849_2

def innerTail849_0 : List Nat := 6857 :: innerTail849_1
def innerCandidates849_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates849_0_checked : candidates edge 4 6581 6857 innerTail849_1 best.bound = innerCandidates849_0 := by
  unfold candidates
  have row : edge 6857 = SearchData.row6857.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan849_0 : scan edge (search edge 3) 4 [6581] 6581 innerTail849_0 best = best := by
  rw [innerTail849_0, scan, if_neg (by decide), innerCandidates849_0_checked]
  change scan edge (search edge 3) 4 [6581] 6581 innerTail849_1
    (if 3 ≤ innerCandidates849_0.length then search edge 3 [6581, 6857] innerCandidates849_0 13438 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan849_1

theorem rootBranch849 : search edge 4 [6581] rootCandidates849 6581 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6581] 6581 innerTail849_0 best = best
  exact innerScan849_0

def rootCandidates848 : List Nat := [6661, 6703, 6883, 7213, 7603, 7669, 7687, 8269]
set_option maxRecDepth 4096 in
theorem rootCandidates848_checked : candidates edge 5 0 6577 tail849 best.bound = rootCandidates848 := by
  unfold candidates
  have row : edge 6577 = SearchData.row6577.contains := by funext q; rfl
  rw [row]
  decide

def innerTail848_8 : List Nat := []
theorem innerScan848_8 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_8 best = best := rfl

def innerTail848_7 : List Nat := 8269 :: innerTail848_8
def innerCandidates848_7 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates848_7_checked : candidates edge 4 6577 8269 innerTail848_8 best.bound = innerCandidates848_7 := by
  unfold candidates
  have row : edge 8269 = SearchData.row8269.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_7 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_7 best = best := by
  rw [innerTail848_7, scan, if_neg (by decide), innerCandidates848_7_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_8
    (if 3 ≤ innerCandidates848_7.length then search edge 3 [6577, 8269] innerCandidates848_7 14846 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_8

def innerTail848_6 : List Nat := 7687 :: innerTail848_7
def innerCandidates848_6 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates848_6_checked : candidates edge 4 6577 7687 innerTail848_7 best.bound = innerCandidates848_6 := by
  unfold candidates
  have row : edge 7687 = SearchData.row7687.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_6 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_6 best = best := by
  rw [innerTail848_6, scan, if_neg (by decide), innerCandidates848_6_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_7
    (if 3 ≤ innerCandidates848_6.length then search edge 3 [6577, 7687] innerCandidates848_6 14264 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_7

def innerTail848_5 : List Nat := 7669 :: innerTail848_6
def innerCandidates848_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates848_5_checked : candidates edge 4 6577 7669 innerTail848_6 best.bound = innerCandidates848_5 := by
  unfold candidates
  have row : edge 7669 = SearchData.row7669.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_5 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_5 best = best := by
  rw [innerTail848_5, scan, if_neg (by decide), innerCandidates848_5_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_6
    (if 3 ≤ innerCandidates848_5.length then search edge 3 [6577, 7669] innerCandidates848_5 14246 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_6

def innerTail848_4 : List Nat := 7603 :: innerTail848_5
def innerCandidates848_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates848_4_checked : candidates edge 4 6577 7603 innerTail848_5 best.bound = innerCandidates848_4 := by
  unfold candidates
  have row : edge 7603 = SearchData.row7603.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_4 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_4 best = best := by
  rw [innerTail848_4, scan, if_neg (by decide), innerCandidates848_4_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_5
    (if 3 ≤ innerCandidates848_4.length then search edge 3 [6577, 7603] innerCandidates848_4 14180 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_5

def innerTail848_3 : List Nat := 7213 :: innerTail848_4
def innerCandidates848_3 : List Nat := [7669]
set_option maxRecDepth 4096 in
theorem innerCandidates848_3_checked : candidates edge 4 6577 7213 innerTail848_4 best.bound = innerCandidates848_3 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_3 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_3 best = best := by
  rw [innerTail848_3, scan, if_neg (by decide), innerCandidates848_3_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_4
    (if 3 ≤ innerCandidates848_3.length then search edge 3 [6577, 7213] innerCandidates848_3 13790 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_4

def innerTail848_2 : List Nat := 6883 :: innerTail848_3
def innerCandidates848_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates848_2_checked : candidates edge 4 6577 6883 innerTail848_3 best.bound = innerCandidates848_2 := by
  unfold candidates
  have row : edge 6883 = SearchData.row6883.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_2 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_2 best = best := by
  rw [innerTail848_2, scan, if_neg (by decide), innerCandidates848_2_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_3
    (if 3 ≤ innerCandidates848_2.length then search edge 3 [6577, 6883] innerCandidates848_2 13460 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_3

def innerTail848_1 : List Nat := 6703 :: innerTail848_2
def innerCandidates848_1 : List Nat := [7213, 7603]
set_option maxRecDepth 4096 in
theorem innerCandidates848_1_checked : candidates edge 4 6577 6703 innerTail848_2 best.bound = innerCandidates848_1 := by
  unfold candidates
  have row : edge 6703 = SearchData.row6703.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_1 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_1 best = best := by
  rw [innerTail848_1, scan, if_neg (by decide), innerCandidates848_1_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_2
    (if 3 ≤ innerCandidates848_1.length then search edge 3 [6577, 6703] innerCandidates848_1 13280 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_2

def innerTail848_0 : List Nat := 6661 :: innerTail848_1
def innerCandidates848_0 : List Nat := [7687]
set_option maxRecDepth 4096 in
theorem innerCandidates848_0_checked : candidates edge 4 6577 6661 innerTail848_1 best.bound = innerCandidates848_0 := by
  unfold candidates
  have row : edge 6661 = SearchData.row6661.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan848_0 : scan edge (search edge 3) 4 [6577] 6577 innerTail848_0 best = best := by
  rw [innerTail848_0, scan, if_neg (by decide), innerCandidates848_0_checked]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_1
    (if 3 ≤ innerCandidates848_0.length then search edge 3 [6577, 6661] innerCandidates848_0 13238 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan848_1

theorem rootBranch848 : search edge 4 [6577] rootCandidates848 6577 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [6577] 6577 innerTail848_0 best = best
  exact innerScan848_0

end ProjectEulerSolutions.PrimePairUpper.Search
