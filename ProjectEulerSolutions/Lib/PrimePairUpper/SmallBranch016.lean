import ProjectEulerSolutions.Lib.PrimePairUpper.SmallSearchSetup

/-! Split bounded search: every child is checked separately. -/
namespace ProjectEulerSolutions.PrimePairUpper.Search
open CliqueSearch

def rootCandidates919 : List Nat := [7411, 7669]
set_option maxRecDepth 4096 in
theorem rootCandidates919_checked : candidates edge 5 0 7213 tail920 best.bound = rootCandidates919 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates918 : List Nat := [7853]
set_option maxRecDepth 4096 in
theorem rootCandidates918_checked : candidates edge 5 0 7211 tail919 best.bound = rootCandidates918 := by
  unfold candidates
  have row : edge 7211 = SearchData.row7211.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates917 : List Nat := [7927, 8059, 8179]
set_option maxRecDepth 4096 in
theorem rootCandidates917_checked : candidates edge 5 0 7207 tail918 best.bound = rootCandidates917 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates916 : List Nat := [7451, 7757, 7841]
set_option maxRecDepth 4096 in
theorem rootCandidates916_checked : candidates edge 5 0 7193 tail917 best.bound = rootCandidates916 := by
  unfold candidates
  have row : edge 7193 = SearchData.row7193.contains := by funext q; rfl
  rw [row]
  decide

def rootCandidates915 : List Nat := [7229, 7331, 8111, 8171]
set_option maxRecDepth 4096 in
theorem rootCandidates915_checked : candidates edge 5 0 7187 tail916 best.bound = rootCandidates915 := by
  unfold candidates
  have row : edge 7187 = SearchData.row7187.contains := by funext q; rfl
  rw [row]
  decide

def innerTail915_4 : List Nat := []
theorem innerScan915_4 : scan edge (search edge 3) 4 [7187] 7187 innerTail915_4 best = best := rfl

def innerTail915_3 : List Nat := 8171 :: innerTail915_4
def innerCandidates915_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates915_3_checked : candidates edge 4 7187 8171 innerTail915_4 best.bound = innerCandidates915_3 := by
  unfold candidates
  have row : edge 8171 = SearchData.row8171.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan915_3 : scan edge (search edge 3) 4 [7187] 7187 innerTail915_3 best = best := by
  rw [innerTail915_3, scan, if_neg (by decide), innerCandidates915_3_checked]
  change scan edge (search edge 3) 4 [7187] 7187 innerTail915_4
    (if 3 ≤ innerCandidates915_3.length then search edge 3 [7187, 8171] innerCandidates915_3 15358 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan915_4

def innerTail915_2 : List Nat := 8111 :: innerTail915_3
def innerCandidates915_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates915_2_checked : candidates edge 4 7187 8111 innerTail915_3 best.bound = innerCandidates915_2 := by
  unfold candidates
  have row : edge 8111 = SearchData.row8111.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan915_2 : scan edge (search edge 3) 4 [7187] 7187 innerTail915_2 best = best := by
  rw [innerTail915_2, scan, if_neg (by decide), innerCandidates915_2_checked]
  change scan edge (search edge 3) 4 [7187] 7187 innerTail915_3
    (if 3 ≤ innerCandidates915_2.length then search edge 3 [7187, 8111] innerCandidates915_2 15298 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan915_3

def innerTail915_1 : List Nat := 7331 :: innerTail915_2
def innerCandidates915_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates915_1_checked : candidates edge 4 7187 7331 innerTail915_2 best.bound = innerCandidates915_1 := by
  unfold candidates
  have row : edge 7331 = SearchData.row7331.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan915_1 : scan edge (search edge 3) 4 [7187] 7187 innerTail915_1 best = best := by
  rw [innerTail915_1, scan, if_neg (by decide), innerCandidates915_1_checked]
  change scan edge (search edge 3) 4 [7187] 7187 innerTail915_2
    (if 3 ≤ innerCandidates915_1.length then search edge 3 [7187, 7331] innerCandidates915_1 14518 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan915_2

def innerTail915_0 : List Nat := 7229 :: innerTail915_1
def innerCandidates915_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates915_0_checked : candidates edge 4 7187 7229 innerTail915_1 best.bound = innerCandidates915_0 := by
  unfold candidates
  have row : edge 7229 = SearchData.row7229.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan915_0 : scan edge (search edge 3) 4 [7187] 7187 innerTail915_0 best = best := by
  rw [innerTail915_0, scan, if_neg (by decide), innerCandidates915_0_checked]
  change scan edge (search edge 3) 4 [7187] 7187 innerTail915_1
    (if 3 ≤ innerCandidates915_0.length then search edge 3 [7187, 7229] innerCandidates915_0 14416 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan915_1

theorem rootBranch915 : search edge 4 [7187] rootCandidates915 7187 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7187] 7187 innerTail915_0 best = best
  exact innerScan915_0

def rootCandidates914 : List Nat := [7207, 7309, 7393, 7933]
set_option maxRecDepth 4096 in
theorem rootCandidates914_checked : candidates edge 5 0 7177 tail915 best.bound = rootCandidates914 := by
  unfold candidates
  have row : edge 7177 = SearchData.row7177.contains := by funext q; rfl
  rw [row]
  decide

def innerTail914_4 : List Nat := []
theorem innerScan914_4 : scan edge (search edge 3) 4 [7177] 7177 innerTail914_4 best = best := rfl

def innerTail914_3 : List Nat := 7933 :: innerTail914_4
def innerCandidates914_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates914_3_checked : candidates edge 4 7177 7933 innerTail914_4 best.bound = innerCandidates914_3 := by
  unfold candidates
  have row : edge 7933 = SearchData.row7933.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan914_3 : scan edge (search edge 3) 4 [7177] 7177 innerTail914_3 best = best := by
  rw [innerTail914_3, scan, if_neg (by decide), innerCandidates914_3_checked]
  change scan edge (search edge 3) 4 [7177] 7177 innerTail914_4
    (if 3 ≤ innerCandidates914_3.length then search edge 3 [7177, 7933] innerCandidates914_3 15110 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan914_4

def innerTail914_2 : List Nat := 7393 :: innerTail914_3
def innerCandidates914_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates914_2_checked : candidates edge 4 7177 7393 innerTail914_3 best.bound = innerCandidates914_2 := by
  unfold candidates
  have row : edge 7393 = SearchData.row7393.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan914_2 : scan edge (search edge 3) 4 [7177] 7177 innerTail914_2 best = best := by
  rw [innerTail914_2, scan, if_neg (by decide), innerCandidates914_2_checked]
  change scan edge (search edge 3) 4 [7177] 7177 innerTail914_3
    (if 3 ≤ innerCandidates914_2.length then search edge 3 [7177, 7393] innerCandidates914_2 14570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan914_3

def innerTail914_1 : List Nat := 7309 :: innerTail914_2
def innerCandidates914_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates914_1_checked : candidates edge 4 7177 7309 innerTail914_2 best.bound = innerCandidates914_1 := by
  unfold candidates
  have row : edge 7309 = SearchData.row7309.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan914_1 : scan edge (search edge 3) 4 [7177] 7177 innerTail914_1 best = best := by
  rw [innerTail914_1, scan, if_neg (by decide), innerCandidates914_1_checked]
  change scan edge (search edge 3) 4 [7177] 7177 innerTail914_2
    (if 3 ≤ innerCandidates914_1.length then search edge 3 [7177, 7309] innerCandidates914_1 14486 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan914_2

def innerTail914_0 : List Nat := 7207 :: innerTail914_1
def innerCandidates914_0 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates914_0_checked : candidates edge 4 7177 7207 innerTail914_1 best.bound = innerCandidates914_0 := by
  unfold candidates
  have row : edge 7207 = SearchData.row7207.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan914_0 : scan edge (search edge 3) 4 [7177] 7177 innerTail914_0 best = best := by
  rw [innerTail914_0, scan, if_neg (by decide), innerCandidates914_0_checked]
  change scan edge (search edge 3) 4 [7177] 7177 innerTail914_1
    (if 3 ≤ innerCandidates914_0.length then search edge 3 [7177, 7207] innerCandidates914_0 14384 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan914_1

theorem rootBranch914 : search edge 4 [7177] rootCandidates914 7177 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7177] 7177 innerTail914_0 best = best
  exact innerScan914_0

def rootCandidates913 : List Nat := [7213, 7219, 7411, 7639, 7717, 8353]
set_option maxRecDepth 4096 in
theorem rootCandidates913_checked : candidates edge 5 0 7159 tail914 best.bound = rootCandidates913 := by
  unfold candidates
  have row : edge 7159 = SearchData.row7159.contains := by funext q; rfl
  rw [row]
  decide

def innerTail913_6 : List Nat := []
theorem innerScan913_6 : scan edge (search edge 3) 4 [7159] 7159 innerTail913_6 best = best := rfl

def innerTail913_5 : List Nat := 8353 :: innerTail913_6
def innerCandidates913_5 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates913_5_checked : candidates edge 4 7159 8353 innerTail913_6 best.bound = innerCandidates913_5 := by
  unfold candidates
  have row : edge 8353 = SearchData.row8353.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan913_5 : scan edge (search edge 3) 4 [7159] 7159 innerTail913_5 best = best := by
  rw [innerTail913_5, scan, if_neg (by decide), innerCandidates913_5_checked]
  change scan edge (search edge 3) 4 [7159] 7159 innerTail913_6
    (if 3 ≤ innerCandidates913_5.length then search edge 3 [7159, 8353] innerCandidates913_5 15512 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan913_6

def innerTail913_4 : List Nat := 7717 :: innerTail913_5
def innerCandidates913_4 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates913_4_checked : candidates edge 4 7159 7717 innerTail913_5 best.bound = innerCandidates913_4 := by
  unfold candidates
  have row : edge 7717 = SearchData.row7717.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan913_4 : scan edge (search edge 3) 4 [7159] 7159 innerTail913_4 best = best := by
  rw [innerTail913_4, scan, if_neg (by decide), innerCandidates913_4_checked]
  change scan edge (search edge 3) 4 [7159] 7159 innerTail913_5
    (if 3 ≤ innerCandidates913_4.length then search edge 3 [7159, 7717] innerCandidates913_4 14876 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan913_5

def innerTail913_3 : List Nat := 7639 :: innerTail913_4
def innerCandidates913_3 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates913_3_checked : candidates edge 4 7159 7639 innerTail913_4 best.bound = innerCandidates913_3 := by
  unfold candidates
  have row : edge 7639 = SearchData.row7639.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan913_3 : scan edge (search edge 3) 4 [7159] 7159 innerTail913_3 best = best := by
  rw [innerTail913_3, scan, if_neg (by decide), innerCandidates913_3_checked]
  change scan edge (search edge 3) 4 [7159] 7159 innerTail913_4
    (if 3 ≤ innerCandidates913_3.length then search edge 3 [7159, 7639] innerCandidates913_3 14798 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan913_4

def innerTail913_2 : List Nat := 7411 :: innerTail913_3
def innerCandidates913_2 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates913_2_checked : candidates edge 4 7159 7411 innerTail913_3 best.bound = innerCandidates913_2 := by
  unfold candidates
  have row : edge 7411 = SearchData.row7411.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan913_2 : scan edge (search edge 3) 4 [7159] 7159 innerTail913_2 best = best := by
  rw [innerTail913_2, scan, if_neg (by decide), innerCandidates913_2_checked]
  change scan edge (search edge 3) 4 [7159] 7159 innerTail913_3
    (if 3 ≤ innerCandidates913_2.length then search edge 3 [7159, 7411] innerCandidates913_2 14570 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan913_3

def innerTail913_1 : List Nat := 7219 :: innerTail913_2
def innerCandidates913_1 : List Nat := []
set_option maxRecDepth 4096 in
theorem innerCandidates913_1_checked : candidates edge 4 7159 7219 innerTail913_2 best.bound = innerCandidates913_1 := by
  unfold candidates
  have row : edge 7219 = SearchData.row7219.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan913_1 : scan edge (search edge 3) 4 [7159] 7159 innerTail913_1 best = best := by
  rw [innerTail913_1, scan, if_neg (by decide), innerCandidates913_1_checked]
  change scan edge (search edge 3) 4 [7159] 7159 innerTail913_2
    (if 3 ≤ innerCandidates913_1.length then search edge 3 [7159, 7219] innerCandidates913_1 14378 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan913_2

def innerTail913_0 : List Nat := 7213 :: innerTail913_1
def innerCandidates913_0 : List Nat := [7411]
set_option maxRecDepth 4096 in
theorem innerCandidates913_0_checked : candidates edge 4 7159 7213 innerTail913_1 best.bound = innerCandidates913_0 := by
  unfold candidates
  have row : edge 7213 = SearchData.row7213.contains := by funext r; rfl
  rw [row]
  decide

theorem innerScan913_0 : scan edge (search edge 3) 4 [7159] 7159 innerTail913_0 best = best := by
  rw [innerTail913_0, scan, if_neg (by decide), innerCandidates913_0_checked]
  change scan edge (search edge 3) 4 [7159] 7159 innerTail913_1
    (if 3 ≤ innerCandidates913_0.length then search edge 3 [7159, 7213] innerCandidates913_0 14372 best else best) = best
  rw [if_neg (by decide)]
  exact innerScan913_1

theorem rootBranch913 : search edge 4 [7159] rootCandidates913 7159 best = best := by
  rw [search_succ_eq_scan]
  change scan edge (search edge 3) 4 [7159] 7159 innerTail913_0 best = best
  exact innerScan913_0

def rootCandidates912 : List Nat := [7211, 7529, 8081]
set_option maxRecDepth 4096 in
theorem rootCandidates912_checked : candidates edge 5 0 7151 tail913 best.bound = rootCandidates912 := by
  unfold candidates
  have row : edge 7151 = SearchData.row7151.contains := by funext q; rfl
  rw [row]
  decide

end ProjectEulerSolutions.PrimePairUpper.Search
